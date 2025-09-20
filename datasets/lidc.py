
import os
import platform
import random
import pickle
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import h5py
import copy 
import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms.functional as tf
from torch.utils.data.dataset import Subset

FILE_PATH_pkl = "../data/lidc_attributes.pkl"

NUM_CLASSES = 2
RESOLUTION = 128
BACKGROUND_CLASS = None

def pad_im(image, size, value=0):
    shape = image.shape
    if len(shape) == 2:
        h, w = shape
    else:
        h, w, c = shape

    if h == w:
        if h == size:
            padded_im = image
        else:
            padded_im = cv2.resize(image, (size, size), cv2.INTER_CUBIC)
    else:
        if h > w:
            pad_1 = (h - w) // 2
            pad_2 = (h - w) - pad_1
            padded_im = cv2.copyMakeBorder(image, 0, 0, pad_1, pad_2, cv2.BORDER_CONSTANT, value=value)
        else:
            pad_1 = (w - h) // 2
            pad_2 = (w - h) - pad_1
            padded_im = cv2.copyMakeBorder(image, pad_1, pad_2, 0, 0, cv2.BORDER_CONSTANT, value=value)
    if padded_im.shape[0] != size:
        padded_im = cv2.resize(padded_im, (size, size), cv2.INTER_CUBIC)

    return padded_im

# 随机选取

class BaseDataset(Dataset):
    def __init__(self, dataset_location, input_size=128):
        self.images = []
        self.mask_labels = []
        self.series_uid = []

        # read dataset
        max_bytes = 2 ** 31 - 1
        data = {}
        print("Loading file", dataset_location)
        bytes_in = bytearray(0)
        file_size = os.path.getsize(dataset_location)
        with open(dataset_location, 'rb') as f_in:
            for _ in range(0, file_size, max_bytes):
                bytes_in += f_in.read(max_bytes)
        new_data = pickle.loads(bytes_in)
        data.update(new_data)

        # load dataset
        for key, value in data.items():
            # image 0-255, alpha 0-255, mask [0,1]
            self.images.append(pad_im(value['image'], input_size))
            masks = []
            for mask in value['masks']:
                masks.append(pad_im(mask, input_size))
            self.mask_labels.append(masks)
            self.series_uid.append(value['series_uid'])

        # check
        assert (len(self.images) == len(self.mask_labels) == len(self.series_uid))
        for image in self.images:
            assert np.max(image) <= 255 and np.min(image) >= 0
        for mask in self.mask_labels:
            assert np.max(mask) <= 1 and np.min(mask) >= 0

        # free
        del new_data
        del data

    def __getitem__(self, index):
        image = copy.deepcopy(self.images[index])
        mask_labels = copy.deepcopy(self.mask_labels[index])
        series_uid = self.series_uid[index]

        return image, mask_labels, series_uid

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.images)



class LIDC_IDRI(BaseDataset):
    '''
    为了适应第二阶段的微调方式
    '''
    def __init__(self, stage, dataset_location, num_experts=4,input_size=128, transform=None):
        super().__init__(dataset_location, input_size)
        self.stage = stage
        self.transform = transform
        self.num_experts = num_experts  # LIDC-IDRI有4个专家标注

    def __getitem__(self, index):
        # 使用基类的数据加载逻辑
        image, labels, series_uid = super().__getitem__(index)
        image = np.expand_dims(image, axis=0) 
        
        if self.stage == 1:
            # probability consensus strategy
            # 1. 生成随机权重向量 w ∈ [0,1]^M, ||w||_1 ≤ M
            w = self._generate_probabilistic_weights()
            
            # 2. 计算共识标签 z^c = w × [z^1, z^2, ..., z^M]^T
            consensus_label = self._compute_consensus_label(labels, w)
            
            if self.transform is not None:
                image, consensus_label = self.transform(image, consensus_label)
                
            return image, consensus_label
            
        elif self.stage == 2:
            # random sample an expert
            prompt = random.randint(0, len(labels)-1)   
            # target 
            label = labels[prompt].astype(float) 
            if self.transform is not None:
                image, label = self.transform(image, label)
            return image, prompt, label
        else:
            raise ValueError("Stage must be '1' or '2'")

    def _generate_probabilistic_weights(self):
        """
        生成概率权重向量 w ∈ [0,1]^M
        支持三种场景：
        1. Single Vote: ||w||_1 = 1 (只有一个专家参与)
        2. Subgroup Consensus: 1 < ||w||_1 < M (部分专家参与)
        3. Full Vote: ||w||_1 = M (所有专家平等参与)
        """
        w = np.zeros(self.num_experts, dtype=np.float32)
        
        # 随机选择至少一个专家
        num_active = random.randint(1, self.num_experts)
        active_indices = random.sample(range(self.num_experts), num_active)
        
        # 设置选中专家的权重为1
        for idx in active_indices:
            w[idx] = 1.0
            
        return w

    def _compute_consensus_label(self, expert_labels, weights):
        """
        计算加权共识标签
        expert_labels: list of expert annotations [M, H, W] or [M, H, W, L]
        weights: binary weight vector w ∈ {0,1}^M
        """
        # 确保专家标注数量与权重向量长度一致
        assert len(expert_labels) == len(weights), "Number of experts must match weight vector length"
        num_selected = int(weights.sum())
        
        # 初始化共识标签
        consensus_label = np.zeros_like(expert_labels[0], dtype=np.float32)
        
        # 计算被选中专家的平均标注
        for i, (label, weight) in enumerate(zip(expert_labels, weights)):
            if weight > 0:  # 只考虑权重为1的专家
                consensus_label += label.astype(np.float32)
        
        # 求平均
        consensus_label /= num_selected
        
        return consensus_label

    def __len__(self):
        return len(self.images)


def one_hot_encoding(arr: np.ndarray) -> np.ndarray:
    res = np.zeros(arr.shape + (NUM_CLASSES,), dtype=np.float32)
    h, w = np.ix_(np.arange(arr.shape[0]), np.arange(arr.shape[1]))
    res[h, w, arr] = 1.0

    return res


def to_tensor(arr: np.ndarray) -> torch.Tensor:
    if arr.ndim == 2:
        arr = arr[:, :, None]

    return torch.from_numpy(arr.transpose((2, 0, 1))).contiguous()


def transform(image, labels):
    labels = labels.astype(int)
    labels = one_hot_encoding(labels)

    image = tf.to_tensor(image.transpose((1, 2, 0))).float()
    labels = to_tensor(labels)

    if torch.rand(1) < 0.5:
        image = tf.hflip(image)
        labels = tf.hflip(labels)

    if torch.rand(1) < 0.5:
        image = tf.vflip(image)
        labels = tf.vflip(labels)

    rots = np.random.randint(0, 4)
    image = torch.rot90(image, rots, [1, 2])
    labels = torch.rot90(labels, rots, [1, 2])

    image = image * 2
    return image, labels


def batch_transform(image, labels):
    image = tf.to_tensor(image.transpose((1, 2, 0))).float()
    image = image * 2

    for i in range(4):
        labels[str(i)] = labels[str(i)].astype(int)
        labels[str(i)] = one_hot_encoding(labels[str(i)])
        labels[str(i)] = to_tensor(labels[str(i)])

    labels = torch.cat((labels['0'][None], labels['1'][None], labels['2'][None], labels['3'][None]), dim=0)
    return image, labels



class Test_LIDC(BaseDataset):

    def __init__(self, stage, dataset_location, input_size=128, transform=None):
        super().__init__(dataset_location, input_size)
        self.stage = stage
        self.transform = transform

    def __getitem__(self, index):
        #使用基类的数据加载逻辑
        image, mask_labels, series_uid = super().__getitem__(index)

        image = np.expand_dims(image, axis=0) #(self.dataset["images"][index], axis=0)
        #sorted_labels = sorted_label(self.dataset["labels"][index])
        # Select the four labels for this image
        labels = {}

        for i in range(4):
            labels[str(i)] = mask_labels[i] #self.dataset["labels"][index][i]
        
        if self.transform is not None:
            image, labels = self.transform(image, labels)
          
        return image, labels

    def __len__(self):
        return len(self.images) #(self.dataset["images"])


def get_num_classes() -> int:
    return NUM_CLASSES


def get_ignore_class() -> int:
    return BACKGROUND_CLASS


def all_dataset(stage):
    return LIDC_IDRI(stage, dataset_location=FILE_PATH_pkl, input_size=128, transform=transform)


def all_test_dataset(stage):
    return Test_LIDC(stage, dataset_location=FILE_PATH_pkl, input_size=128, transform=batch_transform)
