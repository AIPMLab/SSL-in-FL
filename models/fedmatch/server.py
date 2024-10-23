import threading
import tensorflow as tf 
import keras.metrics as tf_metrics
from scipy import spatial
from scipy.stats import truncnorm

from misc.utils import *
from models.fedmatch.client import Client
from modules.federated import ServerModule
from data.loader import DataLoader

class Server(ServerModule):

    def __init__(self, args):
        super(Server, self).__init__(args, Client)
        self.c2s_sum = []
        self.c2s_sig = []
        self.c2s_psi = []
        self.s2c_sum = []
        self.s2c_sig = []
        self.s2c_psi = []
        self.s2c_hlp = []
        self.restored_clients = {}
        self.rid_to_cid = {}
        self.cid_to_vectors = {}
        self.cid_to_weights = {}
        self.curr_round = -1
        mu,std,lower,upper = 125,125,0,255
        self.rgauss = self.loader.scale(truncnorm((lower-mu)/std,(upper-mu)/std, 
                        loc=mu, scale=std).rvs((1,32,32,3))) # fixed gaussian noise for model embedding
        self.best_val_loss = np.inf
        self.best_val_acc = 0
        self.global_model = self.net.build_resnet9(decomposed=True)  # 初始化全局模型

        # 实例化 DataLoader 类
        data_loader = DataLoader(args)

        # 使用实例调用 get_valid 和 get_test 方法
        x_valid, y_valid = data_loader.get_valid()
        x_test, y_test = data_loader.get_test()

        # 将验证和测试数据存储在 self.task 中
        self.task = {
            'x_valid': x_valid,
            'y_valid': y_valid,
            'x_test': x_test,
            'y_test': y_test
        }
        self.metrics = {
            'valid_lss': tf_metrics.Mean(name='valid_lss'),
            'valid_acc': tf_metrics.CategoricalAccuracy(name='valid_acc'),
            'test_lss': tf_metrics.Mean(name='test_lss'),
            'test_acc': tf_metrics.CategoricalAccuracy(name='test_acc')
        }
        self.state = {
            'scores': {
                'valid_loss': [],
                'valid_acc': [],
                'test_loss': [],
                'test_acc': [],
                'aggr_acc': [],
                'aggr_lss': []
            }
        }

    def build_network(self):
        self.global_model = self.net.build_resnet9(decomposed=True)
        self.sig = self.net.get_sigma()
        self.psi = self.net.get_psi()
        self.trainables = [sig for sig in self.sig] # only sigma will be updated at server (Labels at Serve scenario)
        num_connected = int(round(self.args.num_clients*self.args.frac_clients))
        self.restored_clients = {i:self.net.build_resnet9(decomposed=False) for i in range(num_connected)}
        for rid, rm in self.restored_clients.items():
            rm.trainable = False

    def validate_global_model(self):
        tf.keras.backend.set_learning_phase(0)

        for i in range(0, len(self.task['x_valid']), self.args.batch_size_test):
            x_batch = self.task['x_valid'][i:i + self.args.batch_size_test]
            y_batch = self.task['y_valid'][i:i + self.args.batch_size_test]
            y_pred = self.global_model(x_batch)

            # 计算损失
            loss = tf.keras.losses.categorical_crossentropy(y_batch, y_pred)

            # 记录性能
            self.add_performance('valid_lss', 'valid_acc', loss, y_batch, y_pred)

        # 汇总性能数据
        vlss, vacc = self.measure_performance('valid_lss', 'valid_acc')

        # 记录验证损失和准确率
        self.state['scores']['valid_loss'].append(vlss)
        self.state['scores']['valid_acc'].append(vacc)

        return vlss, vacc

    def add_performance(self, lss_name, acc_name, loss, y_true, y_pred):
        self.metrics[lss_name](loss)
        self.metrics[acc_name](y_true, y_pred)

    def save_global_model(self, path):
        abs_path = os.path.join(self.args.check_pts, path)
        print(f"Saving model to path: {abs_path}")

        # 编译模型（如果尚未编译）
        if not self.global_model.optimizer:
            print("Compiling the model before saving...")
            self.global_model.compile(optimizer='adam',
                                      loss='categorical_crossentropy',
                                      metrics=['accuracy'])

        if not os.path.exists(self.args.check_pts):
            print(f"Checkpoint directory {self.args.check_pts} does not exist.")
            return

        try:
            self.global_model.save(abs_path)
            print(f"Entire model successfully saved at: {abs_path}")
        except Exception as e:
            print(f"Failed to save model: {str(e)}")

        if os.path.exists(abs_path):
            print(f"Model file exists at: {abs_path}")
        else:
            print(f"Model file does NOT exist at: {abs_path}")

    def measure_performance(self, lss_name, acc_name):
        lss = float(self.metrics[lss_name].result())
        acc = float(self.metrics[acc_name].result())
        self.metrics[lss_name].reset_states()
        self.metrics[acc_name].reset_states()
        return lss, acc

    def _train_clients(self):
        sigma = [s.numpy() for s in self.sig]
        psi = [p.numpy() for p in self.psi]
        while len(self.connected_ids)>0:
            for gpu_id, gpu_client in self.clients.items():
                cid = self.connected_ids.pop(0)
                helpers = self.get_similar_models(cid)
                with tf.device('/device:GPU:{}'.format(gpu_id)): 
                    # each client will be trained in parallel 
                    thrd = threading.Thread(target=self.invoke_client, args=(gpu_client, cid, self.curr_round, sigma, psi, helpers))
                    self.threads.append(thrd)
                    thrd.start()
                if len(self.connected_ids) == 0:
                    break
            # wait all threads per gpu
            for thrd in self.threads:
                thrd.join()   
            self.threads = []

        # 聚合客户端更新后的权重
        if self.updates:
            # print(f"Updates received from clients: {len(self.updates)}")
            self.set_weights(self.aggregate(self.updates))
        else:
            print("No updates received from clients; skipping this round.")

            # 验证全局模型
        val_loss, val_acc = self.validate_global_model()  # 获取验证损失和准确率

        print(f"Current val_acc: {val_acc}, Best val_acc: {self.best_val_acc}")
        if val_acc > self.best_val_acc or val_loss < self.best_val_loss:
            print("New best model found, saving...")
            self.best_val_acc = val_acc
            self.best_val_loss = val_loss
            self.save_global_model("best_global_model.h5")

            # 保存当前轮次的聚合模型
            # self.save_global_model("global_model_aggregated.h5")

        self.client_similarity(self.updates)
        if self.updates:
            self.set_weights(self.aggregate(self.updates))
        else:
            print("No updates received from clients; skipping this round.")
        self.train.evaluate_after_aggr()
        self.avg_c2s()
        self.avg_s2c()
        self.logger.save_current_state('server', {
            'c2s': {
                'sum': self.c2s_sum,
                'sig': self.c2s_sig,
                'psi': self.c2s_psi,
            },
            's2c': {
                'sum': self.s2c_sum,
                'sig': self.s2c_sig,
                'psi': self.s2c_psi,
                'hlp': self.s2c_hlp,
            },
            'scores': self.train.get_scores()
        }) 
        self.updates = []

    def invoke_client(self, client, cid, curr_round, sigma, psi, helpers):
        # print(f"Invoking client {cid} for round {curr_round} with sigma and psi.")
        update = client.train_one_round(cid, curr_round, sigma=sigma, psi=psi, helpers=helpers)
        # print(f"Update received from client {cid}: {update}")
        self.updates.append(update)

    def client_similarity(self, updates):
        self.restore_clients(updates)
        for rid, rmodel in self.restored_clients.items():
            if rid in self.rid_to_cid:
                cid = self.rid_to_cid[rid]
                self.cid_to_vectors[cid] = np.squeeze(rmodel(self.rgauss))  # embed models
            else:
                # 处理不存在的情况，记录日志并跳过该rid
                print(f"Warning: Key {rid} not found in rid_to_cid dictionary. Skipping this rid.")
                continue

        self.vid_to_cid = list(self.cid_to_vectors.keys())
        self.vectors = list(self.cid_to_vectors.values())
        self.tree = spatial.KDTree(self.vectors)

    def restore_clients(self, updates):
        rid = 0
        self.rid_to_cid = {}
        for cwgts, csize, cid, _, _ in updates:
            self.cid_to_weights[cid] = cwgts
            rwgts = self.restored_clients[rid].get_weights()
            if self.args.scenario == 'labels-at-client':
                half = len(cwgts)//2
                for lid in range(len(rwgts)):
                    rwgts[lid] = cwgts[lid] + cwgts[lid+half] # sigma + psi
            elif self.args.scenario == 'labels-at-server':
                for lid in range(len(rwgts)):
                    rwgts[lid] = self.sig[lid].numpy() + cwgts[lid] # sigma + psi
            self.restored_clients[rid].set_weights(rwgts)
            self.rid_to_cid[rid] = cid
            rid += 1

    def get_similar_models(self, cid):
        if cid in self.cid_to_vectors and (self.curr_round+1)%self.args.h_interval == 0:
            cout = self.cid_to_vectors[cid]
            sims = self.tree.query(cout, self.args.num_helpers+1)
            hids = []
            weights = []
            for vid in sims[1]:
                selected_cid = self.vid_to_cid[vid]
                if selected_cid == cid:
                    continue
                w = self.cid_to_weights[selected_cid]
                if self.args.scenario == 'labels-at-client':
                    half = len(w)//2
                    w = w[half:]
                weights.append(w)
                hids.append(selected_cid)
            return weights[:self.args.num_helpers]
        else:
            return None 

    def set_weights(self, new_weights):
        if self.args.scenario == 'labels-at-client':
            half = len(new_weights)//2
            for i, nwghts in enumerate(new_weights):
                if i < half:
                    self.sig[i].assign(new_weights[i])
                else:
                    self.psi[i-half].assign(new_weights[i])
        elif self.args.scenario == 'labels-at-server':
            for i, nwghts in enumerate(new_weights):
                self.psi[i].assign(new_weights[i])
    
    def avg_c2s(self): # client-wise average
        ratio_list = []
        sig_list = []
        psi_list = []
        for upd in self.updates:
            c2s = upd[3]
            ratio_list.append(c2s['ratio'][-1])
            sig_list.append(c2s['sig_ratio'][-1])
            psi_list.append(c2s['psi_ratio'][-1])
        try:
            self.c2s_sum.append(np.mean(ratio_list, axis=0))
            self.c2s_sig.append(np.mean(sig_list, axis=0))
            self.c2s_psi.append(np.mean(psi_list, axis=0))
        except:
            pdb.set_trace()

    def avg_s2c(self): # client-wise average
        sum_list = []
        sig_list = []
        psi_list = []
        hlp_list = []
        for upd in self.updates:
            s2c = upd[4]
            sum_list.append(s2c['ratio'][-1])
            sig_list.append(s2c['sig_ratio'][-1])
            psi_list.append(s2c['psi_ratio'][-1])
            hlp_list.append(s2c['hlp_ratio'][-1])
        self.s2c_sum.append(np.mean(sum_list, axis=0))
        self.s2c_sig.append(np.mean(sig_list, axis=0))
        self.s2c_psi.append(np.mean(psi_list, axis=0))
        self.s2c_hlp.append(np.mean(hlp_list, axis=0))
    
    
