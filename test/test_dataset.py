import datetime
import os
import time
import torch
import torch.utils.data
from torch import nn
import torchvision
import numpy as np

dataset = torchvision.datasets.VOCSegmentation('./dataset', download=True)
for img, label in dataset:
    image = np.array(img)
    label = np.array(label)
    print("img:{}, label:{}".format(np.unique(image), np.unique(label)))