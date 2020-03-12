import os
import torch
import numpy as np
import PIL.Image as Image
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

    def label2Tensor(image, label):
        """
          toTensor with out normalization 
        """
        if (isinstance(label, np.ndarray)):
            return image, torch.from_numpy(label).type(torch.LongTensor)
        if (isinstance(label, Image.Image)):
            return image, torch.from_numpy(np.array(label)).type(
                torch.LongTensor)

    def id2TrainId(image, label):
        if (isinstance(label, Image.Image)):
            label = np.array(label)
        for void in LabelUtil.void_classes:
            label[label == void] = LabelUtil.ignored
        for trainId, id in enumerate(LabelUtil.id2tra):
            label[label == id] = trainId
        return image, label
