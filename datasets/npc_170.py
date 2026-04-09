import os
import torch
import random
import numpy as np
from torch.utils.data import Dataset
import h5py
from scipy.ndimage.interpolation import zoom
from scipy import ndimage
from skimage import exposure
import torchvision.transforms.functional as tf

BACKGROUND_CLASS = None
NUM_CLASSES = 2

FILE_PATH="/data/zhanghan/MMIS2024TASK1"


class BaseDataSet(Dataset):
    def __init__(
        self,
        base_dir=None,
        split="train",
        modality="t1c"
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.modality = modality

        if self.split == "train":
            self.sample_list = os.listdir(self._base_dir + "/training_2d")
        elif self.split == "val":
            self.sample_list = os.listdir(self._base_dir + "/validation")
        elif self.split == "test":
            self.sample_list = os.listdir(self._base_dir + "/testing")
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = h5py.File(self._base_dir +"/training_2d/{}".format(case), "r")
        elif self.split == "val":
            h5f = h5py.File(self._base_dir + "/validation/{}".format(case), "r")
        elif self.split == "test":
            h5f = h5py.File(self._base_dir + "/testing/{}".format(case), "r")

        # image = h5f[self.modality][:]
        image_modality_list = ["t1", "t1c", "t2"]
        image = np.array([h5f[modality][:] for modality in image_modality_list])
        
        if self.split == "train":
            label = np.zeros((4, image.shape[1], image.shape[2]))
            label[0] = h5f["label_a1"][:]
            label[1] = h5f["label_a2"][:]
            label[2] = h5f["label_a3"][:]
            label[3] = h5f["label_a4"][:]
        else:
            label = np.zeros((4, image.shape[1], image.shape[2], image.shape[3]))
            label[0] = h5f["label_a1"][:]
            label[1] = h5f["label_a2"][:]
            label[2] = h5f["label_a3"][:]
            label[3] = h5f["label_a4"][:]
        
        sample = {"image": image, "label": label, "idx": case}
        return sample
    
class NPC_170(BaseDataSet):
    def __init__(self, stage, base_dir=None, split="train", modality="t1c", transform=None):
        super().__init__(base_dir, split, modality)
        self.stage = stage
        self.transform = transform 
        self.num_experts = 4
    def __getitem__(self, idx):
        # 首先从父类中获取基本的图像和标签数据
        sample = super().__getitem__(idx)
        image = sample['image']
        labels = sample['label']
    
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
        return len(self.sample_list)

class Test_NPC_2D(BaseDataSet):

    def __init__(self, stage, base_dir=None, split="train", modality="t1c", transform=None):
        super().__init__(base_dir, split, modality)
        self.stage = stage
        self.transform = transform 
        self.num_experts = 4

    def __getitem__(self, index):
        #使用基类的数据加载逻辑
        sample = super().__getitem__(index)
        image = sample['image']
        mask_labels = sample['label']
        #sorted_labels = sorted_label(self.dataset["labels"][index])
        # Select the four labels for this image
        labels = {}

        for i in range(self.num_experts):
            labels[str(i)] = mask_labels[i] #self.dataset["labels"][index][i]
        
        if self.transform is not None:
            image, labels = self.transform(image, labels)
          
        return image, labels

    def __len__(self):
        return len(self.sample_list) #(self.dataset["images"])
    

class Test_NPC_3D(BaseDataSet):
    def __init__(self, stage, base_dir=None, split="val", modality="t1c", transform=None):
        super().__init__(base_dir, split, modality)
        self.num_experts = 4
        self.transform = transform

    def __getitem__(self, idx):
        # 首先从父类中获取基本的图像和标签数据
        sample = super().__getitem__(idx)
        image = sample['image']#.transpose(1,0,2,3)
        label = sample['label']
        #print(f"Checking label: 是否全为0: {np.all(label == 0)}")
        image_ = []
        label_ = []
        for i in range(0, label.shape[1]):
            label_slice = label[:,i]
            image_slice = image[:,i]
            label_slice_dict = {
                str(expert_idx): label_slice[expert_idx]  
                for expert_idx in range(self.num_experts)
            }

            image_slice, label_slice = self.transform(image_slice, label_slice_dict)

            label_.append(label_slice[None,:])
            image_.append(image_slice[None,:])
        image = torch.cat(image_, dim=0)
        label = torch.cat(label_, dim=0)

        return image.transpose(1,0), label.transpose(1,0)

def one_hot_encoding(arr: np.ndarray) -> np.ndarray:
    res = np.zeros(arr.shape + (NUM_CLASSES,), dtype=np.float32)
    h, w = np.ix_(np.arange(arr.shape[0]), np.arange(arr.shape[1]))
    res[h, w, arr] = 1.0
    return res

def to_tensor(arr: np.ndarray) -> torch.Tensor:
    if arr.ndim == 2:
        arr = arr[:, :, None]

    return torch.from_numpy(arr.transpose((2, 0, 1))).contiguous()



def transform(image, labels, target_size=(128, 128)):
    zoom_generator = RandomGenerator_Multi_Rater(output_size=target_size)
    image, labels = zoom_generator(image, labels)

    labels = labels.astype(int)
    labels = one_hot_encoding(labels)
    image = tf.to_tensor(image.transpose((1, 2, 0))).float()
    labels = to_tensor(labels)
    image = image * 2

    return image, labels

def batch_transform(image, labels, target_size=(128, 128)):
    zoom_generator = ZoomGenerator(output_size=target_size)
    image = zoom_generator(image) 

    for i in range(len(labels.keys())):
        label = labels[str(i)]
        label = zoom_generator(label)
        label = label.astype(int)
        label = one_hot_encoding(label)
        labels[str(i)] = to_tensor(label)

    labels = torch.cat((labels['0'][None], labels['1'][None], labels['2'][None], labels['3'][None]), dim=0)
    image = tf.to_tensor(image.transpose((1, 2, 0))).float()
    image = image * 2
    return image, labels

def random_rot_flip(image, label=None):
    k = np.random.randint(0, 4)
    axis = np.random.randint(0, 2)

    if len(image.shape) == 2:
        image = np.rot90(image, k)
        image = np.flip(image, axis=axis).copy()
    elif len(image.shape) == 3:
        for i in range(image.shape[0]):
            image[i] = np.rot90(image[i], k)
            image[i] = np.flip(image[i], axis=axis).copy()
    if len(label.shape) == 2:
        label = np.rot90(label, k)
        label = np.flip(label, axis=axis).copy()
    elif len(label.shape) == 3:
        for i in range(label.shape[0]):
            label[i] = np.rot90(label[i], k)
            label[i] = np.flip(label[i], axis=axis).copy()

    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    if len(image.shape) == 2:
        image = ndimage.rotate(image, angle, order=0, reshape=False)
    elif len(image.shape) == 3:
        for i in range(image.shape[0]):
            image[i] = ndimage.rotate(image[i], angle, order=0, reshape=False)
    if len(label.shape) == 2:
        label = ndimage.rotate(label, angle, order=0, reshape=False)
    elif len(label.shape) == 3:
        for i in range(label.shape[0]):
            label[i] = ndimage.rotate(label[i], angle, order=0, reshape=False)
    return image, label

def random_noise(image, label, mu=0, sigma=0.1):
    if len(image.shape) == 2:
        noise = np.clip(sigma * np.random.randn(image.shape[0], image.shape[1]), -2 * sigma, 2 * sigma)
    elif len(image.shape) == 3:
        noise = np.clip(sigma * np.random.randn(image.shape[0], image.shape[1], image.shape[2]), -2 * sigma, 2 * sigma)
    else:
        pass
    noise = noise + mu
    image = image + noise
    return image, label


def random_rescale_intensity(image, label):
    image = exposure.rescale_intensity(image)
    return image, label

def random_equalize_hist(image, label):
    image = exposure.equalize_hist(image)
    return image, label

class RandomGenerator_Multi_Rater(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, image, label):
        #image, label = sample["image"], sample["label"]
        # image, label = sample["image"], sample["label"]
        _, x, y = image.shape

        image = zoom(image, (1, self.output_size[0] / x, self.output_size[1] / y), order=0)
        if len(label.shape) == 2:
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        elif len(label.shape) == 3:
            label = zoom(label, (1, self.output_size[0] / x, self.output_size[1] / y), order=0)

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        if random.random() > 0.5:
            image, label = random_rotate(image, label)
        if random.random() > 0.5:
            image, label = random_noise(image, label)
        # # if random.random() > 0.5:
        # #     image, label = random_rescale_intensity(image, label)
        # # if random.random() > 0.5:
        # #     image, label = random_equalize_hist(image, label)
        # image = torch.from_numpy(image.astype(np.float32))
        # label = torch.from_numpy(label.astype(np.uint8))
        return image, label

class ZoomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, input):
        # image, label = sample["image"], sample["label"]

        #image = zoom(image, (1, self.output_size[0] / x, self.output_size[1] / y), order=0)
        if len(input.shape) == 2:
            x, y = input.shape
            output = zoom(input, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        elif len(input.shape) == 3:
            _, x, y = input.shape
            output = zoom(input, (1, self.output_size[0] / x, self.output_size[1] / y), order=0)

        return output


def get_ignore_class() -> int:
    return BACKGROUND_CLASS

def all_dataset(stage):
    return NPC_170(stage, base_dir=FILE_PATH, transform=transform)


def all_test_dataset(stage):
    return Test_NPC_2D(stage, base_dir=FILE_PATH, transform=batch_transform)
