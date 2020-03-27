import numpy as np
from PIL import Image
Image.LOAD_TRUNCATED_IMAGES = True
import random
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F
from torch.utils import data


class LabelUtil(data.Dataset):
    """
    label tool 
    """
    colors = [  # [  0,   0,   0],
        [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
        [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
        [107, 142, 35], [152, 251, 152], [0, 130, 180], [220, 20, 60],
        [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100],
        [0, 0, 230], [119, 11, 32]
    ]
    void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
    valid_classes = [
        7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32,
        33
    ]
    class_names = [
        'unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
        'traffic_light', 'traffic_sign', 'vegetation', 'terrain', 'sky',
        'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
        'bicycle'
    ]
    trainId2colours = dict(zip(range(19), colors))
    id2tra = dict(zip(valid_classes, range(19)))
    ignored = 255

    def __init__(self):
        super(LabelUtil, self).__init__()

    def label2Tensor(self, image, label):
        """
          toTensor with out normalization 
        """
        if (isinstance(label, np.ndarray)):
            return image, torch.from_numpy(label).type(torch.LongTensor)
        if (isinstance(label, Image.Image)):
            return image, torch.from_numpy(np.array(label)).type(
                torch.LongTensor)

    def id2TrainId(self, image, label):
        for void in LabelUtil.void_classes:
            label[label == void] = LabelUtil.ignored
        for trainId, id in enumerate(LabelUtil.id2tra):
            label[label == id] = trainId
        return image, label

    def trainId2Id(self, image, label):
        label[label == LabelUtil.ignored] = 0
        for trainId, id in enumerate(LabelUtil.id2tra):
            label[label == trainId] = id
        return image, label


def pad_if_smaller(img, size, fill=0):
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img


class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)

        target = torch.as_tensor(np.asarray(target),
                                 dtype=torch.int64)  # torch.LongTensor
        if target.dim() == 3:
            target = target.permute((2, 0, 1)).contiguous()
        else:
            target = target.unsqueeze(0)
        target = target[0]
        return image, target


class RandomResize(object):
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        size = random.randint(self.min_size, self.max_size)
        image = F.resize(image, size)  # size 为整数时，以最短边为标准，进行等比缩放
        target = F.resize(target, size, interpolation=Image.NEAREST)
        return image, target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class IdtoTrainId(object):
    def __call__(self, image, target):
        labelUtil = LabelUtil()
        image, target = labelUtil.id2TrainId(image, target)
        return image, target