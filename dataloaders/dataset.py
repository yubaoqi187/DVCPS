import os
import cv2
import torch
import random
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import h5py
from scipy.ndimage.interpolation import zoom
from torchvision import transforms
import itertools
from scipy import ndimage
from torch.utils.data.sampler import Sampler
import augmentations
from augmentations.ctaugment import OPS
import matplotlib.pyplot as plt
from PIL import Image ,ImageFilter
from copy import deepcopy
from torchvision.transforms import functional as F

class DVCPS_BaseDataSets(Dataset):
    def __init__(
        self,
        base_dir=None,
        split="train",
        size=None,
        num=None
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.size = size

        if self.split == "train":
            with open(self._base_dir + "/train_slices.list", "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        elif self.split == "val":
            with open(self._base_dir + "/val.list", "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = h5py.File(self._base_dir + "/data/slices/{}.h5".format(case), "r")
        else:
            h5f = h5py.File(self._base_dir + "/data/{}.h5".format(case), "r")
        img = h5f["image"]
        mask = h5f["label"]
        img_w1, mask_w1 = img, mask
        img_w2, mask_w2 = img, mask

        if random.random() > 0.5:
            img_w1, mask_w1 = random_rot_flip(img, mask)  # 随机旋转、水平翻转或垂直翻转
        elif random.random() > 0.5:
            img_w1, mask_w1 = random_rotate(img, mask)  # 随机角度的旋转
        (x1, y1) = img_w1.shape
        img_w1 = zoom(img_w1, (self.size[0] / x1, self.size[1] / y1), order=0)
        mask_w1 = zoom(mask_w1, (self.size[0] / x1, self.size[1] / y1), order=0)
        image1 = torch.from_numpy(img_w1.astype(np.float32)).unsqueeze(0)
        label1 = torch.from_numpy(mask_w1.astype(np.uint8))
        
        img_w1 = Image.fromarray((img_w1 * 255).astype(np.uint8))
        img_s1,img_s2= deepcopy(img_w1),deepcopy(img_w1)
        if random.random() < 0.8:
            img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)
        img_s1 = blur(img_s1, p=0.5)  
        cutmix_box1 = obtain_cutmix_box(self.size[0], p=0.5)
        img_s1 = torch.from_numpy(np.array(img_s1)).unsqueeze(0).float() / 255.0
        if random.random() < 0.8:
            img_s2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s2)
        img_s2 = blur(img_s2, p=0.5)
        cutmix_box2 = obtain_cutmix_box(self.size[0], p=0.5)
        img_s2 = torch.from_numpy(np.array(img_s2)).unsqueeze(0).float() / 255.0

        if random.random() > 0.5:
            img_w2, mask_w2 = random_rot_flip(img, mask)  # 随机旋转、水平翻转或垂直翻转
        elif random.random() > 0.5:
            img_w2, mask_w2 = random_rotate(img, mask)  # 随机角度的旋转
        (x2, y2) = img_w2.shape
        img_w2 = zoom(img_w2, (self.size[0] / x2, self.size[1] / y2), order=0)
        mask_w2 = zoom(mask_w2, (self.size[0] / x2, self.size[1] / y2), order=0)
        image2 = torch.from_numpy(img_w2.astype(np.float32)).unsqueeze(0)
        label2 = torch.from_numpy(mask_w2.astype(np.uint8))

        img_w2 = Image.fromarray((img_w2 * 255).astype(np.uint8))
        img_s1_m,img_s2_m = deepcopy(img_w2),deepcopy(img_w2)
        if random.random() < 0.8:
            img_s1_m = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1_m)
        img_s1_m = blur(img_s1_m, p=0.5)   
        img_s1_m = torch.from_numpy(np.array(img_s1_m)).unsqueeze(0).float() / 255.0
        if random.random() < 0.8:
            img_s2_m = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s2_m)
        img_s2_m = blur(img_s2_m, p=0.5)   
        img_s2_m = torch.from_numpy(np.array(img_s2_m)).unsqueeze(0).float() / 255.0

        sample = {"image_w": image1, "label_w": label1, "image_w_m": image2, "label_w_m": label2, 
                   "image_s1": img_s1, "image_s1_m": img_s1_m, "cutmix_box1": cutmix_box1,
                   "image_s2": img_s2, "image_s2_m": img_s2_m, "cutmix_box2": cutmix_box2}
        sample["idx"] = idx
        return sample
  
class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size, with_sdf=False):
        self.output_size = output_size
        self.with_sdf = with_sdf

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if self.with_sdf:
            sdf = sample['sdf']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            if self.with_sdf:
                sdf = np.pad(sdf, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape

        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        if self.with_sdf:
            sdf = sdf[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            return {'image': image, 'label': label, 'sdf': sdf}
        else:
            return {'image': image, 'label': label}


class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image, label = random_rot_flip(image, label)

        return {'image': image, 'label': label}

def blur(img, p=0.5):
    if random.random() < p:
        sigma = np.random.uniform(0.1, 2.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return img

def obtain_cutmix_box(img_size, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3, ratio_2=1/0.3):
    mask = torch.zeros(img_size, img_size)
    if random.random() > p:
        return mask

    size = np.random.uniform(size_min, size_max) * img_size * img_size
    while True:
        ratio = np.random.uniform(ratio_1, ratio_2)
        cutmix_w = int(np.sqrt(size / ratio))
        cutmix_h = int(np.sqrt(size * ratio))
        x = np.random.randint(0, img_size)
        y = np.random.randint(0, img_size)

        if x + cutmix_w <= img_size and y + cutmix_h <= img_size:
            break

    mask[y:y + cutmix_h, x:x + cutmix_w] = 1

    return mask

def obtain_cutmix_box3d(img_size, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3, ratio_2=1/0.3):
    mask = torch.zeros(img_size[0], img_size[1], img_size[2])
    if random.random() > p:
        return mask

    size = np.random.uniform(size_min, size_max) * img_size[0] * img_size[1] * img_size[2]
    while True:
        ratio = np.random.uniform(ratio_1, ratio_2)
        cutmix_w = int(np.sqrt(size / ratio))
        cutmix_h = int(np.sqrt(size * ratio))
        cutmix_d = int(np.sqrt(size * ratio))
        x = np.random.randint(0, img_size[0])
        y = np.random.randint(0, img_size[1])
        z = np.random.randint(0, img_size[2])

        if x + cutmix_w <= img_size[0] and y + cutmix_h <= img_size[1] and z + cutmix_d <= img_size[2]:
            break

    mask[y:y + cutmix_h, x:x + cutmix_w, z:z + cutmix_d] = 1

    return mask

class BaseDataSets(Dataset):
    def __init__(
        self,
        base_dir=None,
        split="train",
        num=None,
        transform=None,
        ops_weak=None,
        ops_strong=None,
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.ops_weak = ops_weak
        self.ops_strong = ops_strong

        assert bool(ops_weak) == bool(
            ops_strong
        ), "For using CTAugment learned policies, provide both weak and strong batch augmentation policy"

        if self.split == "train":
            with open(self._base_dir + "/train_slices.list", "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        elif self.split == "val":
            with open(self._base_dir + "/val.list", "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = h5py.File(self._base_dir + "/data/slices/{}.h5".format(case), "r")
        else:
            h5f = h5py.File(self._base_dir + "/data/{}.h5".format(case), "r")
        image = h5f["image"][:]
        label = h5f["label"][:]
        sample = {"image": image, "label": label}
        if self.split == "train":
            if None not in (self.ops_weak, self.ops_strong):
                sample = self.transform(sample, self.ops_weak, self.ops_strong)
            else:
                sample = self.transform(sample)
        sample["idx"] = idx
        return sample

def random_crop(image,label,output_size,sdf=None):

    if label.shape[0] <= output_size[0] or label.shape[1] <= output_size[1] or label.shape[2] <= output_size[2]:
        pw = max((output_size[0] - label.shape[0]) // 2 + 3, 0)
        ph = max((output_size[1] - label.shape[1]) // 2 + 3, 0)
        pd = max((output_size[2] - label.shape[2]) // 2 + 3, 0)
        image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
        label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
        if sdf is not None:
            sdf = np.pad(sdf, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

    (w, h, d) = image.shape
    # if np.random.uniform() > 0.33:
    #     w1 = np.random.randint((w - self.output_size[0])//4, 3*(w - self.output_size[0])//4)
    #     h1 = np.random.randint((h - self.output_size[1])//4, 3*(h - self.output_size[1])//4)
    # else:
    w1 = np.random.randint(0, w - output_size[0])
    h1 = np.random.randint(0, h - output_size[1])
    d1 = np.random.randint(0, d - output_size[2])

    label = label[w1:w1 + output_size[0], h1:h1 + output_size[1], d1:d1 + output_size[2]]
    image = image[w1:w1 + output_size[0], h1:h1 + output_size[1], d1:d1 + output_size[2]]
    if sdf is not None:
        sdf = sdf[w1:w1 + output_size[0], h1:h1 + output_size[1], d1:d1 + output_size[2]]
        return image, label, sdf
    else:
        return image, label

def random_rot_flip(image, label=None):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    if label is not None:
        label = np.rot90(label, k)
        label = np.flip(label, axis=axis).copy()
        return image, label
    else:
        return image


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label

def color_jitter2(image, p=1.0):
    # if not torch.is_tensor(image):
    #     np_to_tensor = transforms.ToTensor()
    #     image = np_to_tensor(image)
    # s is the strength of color distortion.
    # s = 1.0
    # jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    jitter = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)
    if np.random.random() < p:
        image = jitter(image)
    return image
def color_jitter(image):
    if not torch.is_tensor(image):
        np_to_tensor = transforms.ToTensor()
        image = np_to_tensor(image)

    # s is the strength of color distortion.
    s = 1.0
    jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    return jitter(image)


class CTATransform(object):
    def __init__(self, output_size, cta):
        self.output_size = output_size
        self.cta = cta

    def __call__(self, sample, ops_weak, ops_strong):
        image, label = sample["image"], sample["label"]
        image = self.resize(image)
        label = self.resize(label)
        to_tensor = transforms.ToTensor()

        # fix dimensions
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))

        # apply augmentations
        image_weak = augmentations.cta_apply(transforms.ToPILImage()(image), ops_weak)
        image_strong = augmentations.cta_apply(image_weak, ops_strong)
        label_aug = augmentations.cta_apply(transforms.ToPILImage()(label), ops_weak)
        label_aug = to_tensor(label_aug).squeeze(0)
        label_aug = torch.round(255 * label_aug).int()

        sample = {
            "image_weak": to_tensor(image_weak),
            "image_strong": to_tensor(image_strong),
            "label_aug": label_aug,
        }
        return sample

    def cta_apply(self, pil_img, ops):
        if ops is None:
            return pil_img
        for op, args in ops:
            pil_img = OPS[op].f(pil_img, *args)
        return pil_img

    def resize(self, image):
        x, y = image.shape
        return zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        # ind = random.randrange(0, img.shape[0])
        # image = img[ind, ...]
        # label = lab[ind, ...]
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        sample = {"image": image, "label": label}
        return sample
    
 
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        if 'onehot_label' in sample:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long(),
                    'onehot_label': torch.from_numpy(sample['onehot_label']).long()}
        else:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long()}

class WeakStrongAugment(object):
    """returns weakly and strongly augmented images

    Args:
        object (tuple): output size of network
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        image = self.resize(image)
        label = self.resize(label)
        # weak augmentation is rotation / flip
        image_weak, label = random_rot_flip(image, label)
        # strong augmentation is color jitter
        image_strong = color_jitter(image_weak).type("torch.FloatTensor")
        # fix dimensions
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        image_weak = torch.from_numpy(image_weak.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))

        sample = {
            "image": image,
            "image_weak": image_weak,
            "image_strong": image_strong,
            "label_aug": label,
        }
        return sample

    def resize(self, image):
        x, y = image.shape
        return zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch) in zip(
                grouper(primary_iter, self.primary_batch_size),
                grouper(secondary_iter, self.secondary_batch_size),
            )
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)

# def add_gaussian_noise(tensor, mean, std_dev,clamp=False):
#     noise = torch.randn_like(tensor) * std_dev + mean
#     if clamp is not False:
#          noise = torch.clamp((noise) , -0.2, 0.2)
#     noisy_tensor = tensor + noise
#     return noisy_tensor

# def calculate_batch_stats(tensor):
#     batch_mean = tensor.mean()
#     batch_std_dev = tensor.std()
#     return batch_mean, batch_std_dev

def add_gaussian_noise(tensor, mean, std_dev, clamp=False):
    noise = np.random.randn(*tensor.shape) * std_dev + mean
    if clamp is not False:
        noise = np.clip(noise, -0.2, 0.2)
    noisy_tensor = tensor + noise
    return noisy_tensor

def calculate_batch_stats(tensor):
    batch_mean = np.mean(tensor)
    batch_std_dev = np.std(tensor)
    return batch_mean, batch_std_dev

def func_strong_augs(image, p_color=0.8, p_blur=0.5):
    img = Image.fromarray((image * 255).astype(np.uint8))
    img = color_jitter2(img, p_color)
    img = blur(img, p_blur)

    img = torch.from_numpy(np.array(img)).unsqueeze(0).float() / 255.0

    return img