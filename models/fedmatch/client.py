import gc
import cv2
import time
import random
import tensorflow as tf 
import numpy as np
from PIL import Image
from keras import backend as K

from misc.utils import *
from modules.federated import ClientModule

class Client(ClientModule):

    def __init__(self, gid, args):
        """ FedMatch Client

        Performs fedmatch cleint algorithms 
        Inter-client consistency, agreement-based labeling, disjoint learning, etc.

        Created by:
            Wonyong Jeong (wyjeong@kaist.ac.kr)
        """
        super(Client, self).__init__(gid, args)
        self.kl_divergence = tf.keras.losses.KLDivergence()
        self.cross_entropy = tf.keras.losses.CategoricalCrossentropy()
        self.init_model()

    def init_model(self):
        self.local_model = self.net.build_resnet9(decomposed=True)
        self.helpers = [self.net.build_resnet9(decomposed=False) for _ in range(self.args.num_helpers)]
        self.sig = self.net.get_sigma()
        self.psi = self.net.get_psi()
        for h in self.helpers:
            h.trainable = False

    def _init_state(self):
        self.train.set_details({
            'loss_fn_s': self.loss_fn_s,
            'loss_fn_u': self.loss_fn_u,
            'model': self.local_model,
            'trainables_s': self.sig,
            'trainables_u': self.psi,
            'batch_size': self.args.batch_size_client,
            'num_epochs': self.args.num_epochs_client,
        })

    def _train_one_round(self, client_id, curr_round, sigma, psi, helpers=None):
        self.train.cal_s2c(self.state['curr_round'], sigma, psi, helpers)
        self.set_weights(sigma, psi)
        if helpers is None:
            self.is_helper_available = False
        else:
            self.is_helper_available = True
            self.restore_helpers(helpers)
        self.train.train_one_round(self.state['curr_round'], self.state['round_cnt'], self.state['curr_task'])

        self.logger.save_current_state(self.state['client_id'], {
            's2c': self.train.get_s2c(),
            'c2s': self.train.get_c2s(),
            'scores': self.train.get_scores()
        })

    def loss_fn_s(self, x, y):
        # loss function for supervised learning
        x = self.loader.scale(x)
        y_pred = self.local_model(x)
        loss_s = self.cross_entropy(y, y_pred) * self.args.lambda_s
        print(f"Loss_s = {loss_s.numpy():.4f}")
        return y_pred, loss_s

    def em_loss(self, pred_s, pseudo_label, k, mask_pred, p_cutoff):
        softmax_pred = tf.nn.softmax(pred_s, axis=-1)

        # 将 k 转换为 (index, column) 对应关系的二维索引
        batch_indices = tf.range(tf.shape(pred_s)[0])  # 形状为 [batch_size]
        k_indices = tf.stack([batch_indices, k[:, 0]], axis=-1)  # 形状为 [batch_size, 2]

        # 创建 mask_k 张量，并在指定索引位置更新为 1
        mask_k = tf.zeros_like(pseudo_label)
        mask_k = tf.tensor_scatter_nd_update(mask_k, k_indices, tf.ones([tf.shape(k)[0]], dtype=tf.float32))

        label = tf.argmax(pseudo_label, axis=-1, output_type=tf.int32)
        label_indices = tf.stack([batch_indices, label], axis=-1)
        mask_k = tf.tensor_scatter_nd_update(mask_k, label_indices, tf.ones([tf.shape(label)[0]], dtype=tf.float32))

        # Debugging: Check intermediate values
        yg = tf.reduce_sum(tf.boolean_mask(softmax_pred, tf.cast(mask_k, dtype=tf.bool)), axis=-1, keepdims=True)
        yg = tf.clip_by_value(yg, 1e-7, 1 - 1e-7)  # 限制 yg 的值在合理范围内
        # print(f"yg = {yg.numpy()}")

        soft_ml = (1 - yg + 1e-7) / (tf.cast(tf.shape(k)[1] - 1, tf.float32))
        # print(f"soft_ml = {soft_ml.numpy()}")

        mask = 1 - mask_k
        mask = mask * tf.expand_dims(mask_pred, axis=-1)
        mask = tf.where((mask == 1) & (softmax_pred > p_cutoff ** 2), tf.zeros_like(mask), mask)

        # print(f"Final Mask sum = {tf.reduce_sum(mask).numpy()}")

        loss_em = -(soft_ml * tf.math.log(softmax_pred + 1e-10) + (1 - soft_ml) * tf.math.log(1 - softmax_pred + 1e-10))
        loss_em = tf.reduce_sum(loss_em * mask) / (tf.reduce_sum(mask) + 1e-10)

        # print(f"Loss_em after computation = {loss_em.numpy()}")

        return loss_em

    def loss_fn_u(self, x):
        # 原有的无监督学习损失计算
        loss_u = 0
        y_pred_weak = self.local_model(self.loader.scale(x))

        # 获取高置信度样本的索引
        conf = tf.where(tf.reduce_max(y_pred_weak, axis=1) >= self.args.confidence)
        conf = tf.squeeze(conf, axis=-1)

        if tf.size(conf) > 0:
            x_conf = self.loader.scale(tf.gather(x, conf))
            y_pred_conf = tf.gather(y_pred_weak, conf)

            if True:  # inter-client consistency
                if self.is_helper_available:
                    y_preds = [rm(x_conf).numpy() for rid, rm in enumerate(self.helpers)]
                    if self.state['curr_round'] > 0:
                        # inter-client consistency loss
                        for hid, pred in enumerate(y_preds):
                            loss_u += (self.kl_divergence(pred, y_pred_conf) / len(y_preds)) * self.args.lambda_i
                else:
                    y_preds = None

                # Agreement-based Pseudo Labeling
                y_hard = self.local_model(self.loader.scale(self.loader.augment(x_conf, soft=False)))
                y_pseu = self.agreement_based_labeling(y_pred_conf, y_preds)
                loss_u += tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(labels=y_pseu, logits=y_hard)) * self.args.lambda_a
            else:
                y_hard = self.local_model(self.loader.scale(self.loader.augment(x_conf, soft=False)))
                loss_u += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_pred_conf, logits=y_hard)) * self.args.lambda_a

            # 添加低置信度样本的KL散度损失 (Lkl)
            pseudo_label = tf.nn.softmax(y_pred_weak, axis=-1)
            max_probs = tf.reduce_max(pseudo_label, axis=-1)
            mask_low_confidence = tf.cast(max_probs < self.args.threshold, tf.float32)

            sharpened_pseudo_label = pseudo_label ** (1 / self.args.T)
            sharpened_pseudo_label /= tf.reduce_sum(sharpened_pseudo_label, axis=1, keepdims=True)

            log_softmax_pred = tf.nn.log_softmax(y_pred_weak, axis=-1)
            # 计算KL散度
            Lkl = tf.reduce_mean(tf.reduce_sum(mask_low_confidence[:, tf.newaxis] * sharpened_pseudo_label * (
                        log_softmax_pred - tf.math.log(sharpened_pseudo_label)), axis=-1
                ))
            loss_u += Lkl
            # print(f"Lkl = {Lkl.numpy():.4f}")

            # 添加高置信度样本的Entropy Meaning Loss (loss_em)
            mask_high_confidence = tf.cast(max_probs >= self.args.threshold, tf.float32)
            k_value = tf.math.top_k(sharpened_pseudo_label, 2)[1]
            loss_em = self.em_loss(y_pred_weak, sharpened_pseudo_label, k_value, mask_high_confidence, self.args.threshold)
            loss_u += loss_em

            # print(f"Loss_em = {loss_em.numpy():.4f}")

        # 原有的正则化项
        for lid, psi in enumerate(self.psi):
            # l1 regularization
            loss_u += tf.reduce_sum(tf.abs(psi)) * self.args.lambda_l1
            # l2 regularization
            loss_u += tf.reduce_sum(tf.square(self.sig[lid] - psi)) * self.args.lambda_l2

        return y_pred_weak, loss_u, tf.size(conf)

    # def loss_fn_u(self, x):
    #     # loss function for unsupervised learning
    #     loss_u = 0
    #     y_pred = self.local_model(self.loader.scale(x))
    #     conf = np.where(np.max(y_pred.numpy(), axis=1)>=self.args.confidence)[0]
    #     if len(conf)>0:
    #         x_conf = self.loader.scale(x[conf])
    #         y_pred = K.gather(y_pred, conf)
    #         if True: # inter-client consistency
    #             if self.is_helper_available:
    #                 y_preds = [rm(x_conf).numpy() for rid, rm in enumerate(self.helpers)]
    #                 if self.state['curr_round']>0:
    #                     #inter-client consistency loss
    #                     for hid, pred in enumerate(y_preds):
    #                         loss_u += (self.kl_divergence(pred, y_pred)/len(y_preds))*self.args.lambda_i
    #             else:
    #                 y_preds = None
    #             # Agreement-based Pseudo Labeling
    #             y_hard = self.local_model(self.loader.scale(self.loader.augment(x[conf], soft=False)))
    #             y_pseu = self.agreement_based_labeling(y_pred, y_preds)
    #             loss_u += self.cross_entropy(y_pseu, y_hard) * self.args.lambda_a
    #         else:
    #             y_hard = self.local_model(self.loader.scale(self.loader.augment(x[conf], soft=False)))
    #             loss_u += self.cross_entropy(y_pred, y_hard) * self.args.lambda_a
    #     # additional regularization
    #     for lid, psi in enumerate(self.psi):
    #         # l1 regularization
    #         loss_u += tf.reduce_sum(tf.abs(psi)) * self.args.lambda_l1
    #         # l2 regularization
    #         loss_u += tf.math.reduce_sum(tf.math.square(self.sig[lid]-psi)) * self.args.lambda_l2
    #
    #     return y_pred, loss_u, len(conf)

    def agreement_based_labeling(self, y_pred, y_preds=None):
        y_pseudo = np.array(y_pred)
        if self.is_helper_available:
            y_vote = tf.keras.utils.to_categorical(np.argmax(y_pseudo, axis=1), self.args.num_classes)
            y_votes = np.sum([tf.keras.utils.to_categorical(np.argmax(y_rm, axis=1), self.args.num_classes) for y_rm in y_preds], axis=0)
            y_vote = np.sum([y_vote, y_votes], axis=0)
            y_pseudo = tf.keras.utils.to_categorical(np.argmax(y_vote, axis=1), self.args.num_classes)
        else:
            y_pseudo = tf.keras.utils.to_categorical(np.argmax(y_pseudo, axis=1), self.args.num_classes)
        return y_pseudo

    def restore_helpers(self, helper_weights):
        for hid, hwgts in enumerate(helper_weights):
            wgts = self.helpers[hid].get_weights()
            for i in range(len(wgts)):
                wgts[i] = self.sig[i].numpy() + hwgts[i] # sigma + psi
            self.helpers[hid].set_weights(wgts)

    def get_weights(self):
        if self.args.scenario == 'labels-at-client':
            sigs = [sig.numpy() for sig in self.sig]
            psis = [psi.numpy() for psi in self.psi] 
            return np.concatenate([sigs,psis], axis=0)
        elif self.args.scenario == 'labels-at-server':
            return [psi.numpy() for psi in self.psi]

    def set_weights(self, sigma, psi):
        for i, sig in enumerate(sigma):
            self.sig[i].assign(sig)
        for i, p in enumerate(psi):
            self.psi[i].assign(p)

    