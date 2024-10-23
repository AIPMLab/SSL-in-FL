import os
import glob
import numpy as np
from PIL import Image
from scipy.ndimage import shift
from third_party.rand_augment.randaug import RandAugment
import tensorflow as tf
import cv2
from misc.utils import *
from config import *

class DataLoader:

    def __init__(self, args):
        self.args = args
        self.shape = (28, 28, 1)  # Fashion-MNIST 原始图像的尺寸
        self.target_shape = (32, 32, 3)  # 目标图像的尺寸 (32, 32, 3)
        self.rand_augment = RandAugment()
        self.base_dir = os.path.join(self.args.dataset_path, self.args.task)
        self.stats = [{
            'mean': [0.2869, 0.2869, 0.2869],  # Fashion-MNIST 的均值
            'std': [0.3530, 0.3530, 0.3530]    # Fashion-MNIST 的标准差
        }]

    def get_s_by_id(self, client_id):
        task = np_load(self.base_dir, f's_{self.args.dataset_id_to_name[self.args.dataset_id]}_{client_id}.npy')
        task['x'] = self.resize_images(self.convert_to_rgb(task['x']))  # 将灰度图像转换为 RGB 并调整尺寸
        return task['x'], task['y'], task['name']

    def get_u_by_id(self, client_id, task_id):
        path = os.path.join(self.base_dir, f'u_{self.args.dataset_id_to_name[self.args.dataset_id]}_{client_id}*')
        tasks = sorted([os.path.basename(p) for p in glob.glob(path)])
        task = np_load(self.base_dir, tasks[task_id])
        task['x'] = self.resize_images(self.convert_to_rgb(task['x']))  # 将灰度图像转换为 RGB 并调整尺寸
        return task['x'], task['y'], task['name']

    def get_s_server(self):
        task = np_load(self.base_dir, f's_{self.args.dataset_id_to_name[self.args.dataset_id]}.npy')
        task['x'] = self.resize_images(self.convert_to_rgb(task['x']))  # 将灰度图像转换为 RGB 并调整尺寸
        return task['x'], task['y'], task['name']

    def get_test(self):
        task = np_load(self.base_dir, f'test_{self.args.dataset_id_to_name[self.args.dataset_id]}.npy')
        task['x'] = self.resize_images(self.convert_to_rgb(task['x']))  # 将灰度图像转换为 RGB 并调整尺寸
        return task['x'], task['y']

    def get_valid(self):
        task = np_load(self.base_dir, f'valid_{self.args.dataset_id_to_name[self.args.dataset_id]}.npy')
        task['x'] = self.resize_images(self.convert_to_rgb(task['x']))  # 将灰度图像转换为 RGB 并调整尺寸
        return task['x'], task['y']

    def scale(self, x):
        x = tf.cast(x, tf.float32) / 255  # 归一化到 [0,1] 之间
        return x

    def resize_images(self, images):
        resized_images = []
        for img in images:
            img_resized = cv2.resize(img, (32, 32))  # 调整大小为 32x32
            assert img_resized.shape == (32, 32, 3), f"Resized image has unexpected shape: {img_resized.shape}"
            resized_images.append(img_resized)
        return np.array(resized_images)

    def convert_to_rgb(self, images):
        """
        将灰度图像转换为 RGB 三通道图像
        """
        # print(f"Original images shape before conversion: {images.shape}")  # 打印转换前的形状

        if len(images.shape) == 4 and images.shape[1] == 1:  # 检查图像是否是灰度图像，形状应为 (batch_size, 1, height, width)
            images = np.squeeze(images, axis=1)  # 去掉通道维度 (batch_size, 28, 28)
            images = np.stack([images] * 3, axis=-1)  # 转换为 RGB 三通道 (batch_size, 28, 28) -> (batch_size, 28, 28, 3)

        elif len(images.shape) == 4 and images.shape[-1] != 3:
            raise ValueError(f"Unexpected number of channels: {images.shape[-1]}. Expected 1 or 3.")

        # print(f"Converted RGB images shape: {images.shape}")  # 打印转换后的形状
        return images

    def augment(self, images, soft=True):
        if soft:
            indices = np.arange(len(images)).tolist()
            sampled = random.sample(indices, int(round(0.5 * len(indices))))  # flip horizontally 50%
            images[sampled] = np.fliplr(images[sampled])
            sampled = random.sample(sampled, int(round(0.25 * len(sampled))))  # flip vertically 25% from above
            images[sampled] = np.flipud(images[sampled])
            return np.array(
                [shift(img, [random.randint(-2, 2), random.randint(-2, 2), 0]) for img in images])  # random shift
        else:
            images = images.numpy() if hasattr(images, 'numpy') else images
            return np.array([np.array(self.rand_augment(Image.fromarray(np.reshape(img.astype(np.uint8), self.target_shape)),
                                                        M=random.randint(2, 5))) for img in images])
