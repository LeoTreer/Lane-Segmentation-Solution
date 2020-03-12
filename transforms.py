import numpy as np
from PIL import Image
import random
from utils.label_tool import LabelUtil
from torchvision import transforms as T
from torchvision.transforms import functional as F


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, image, target):
        for t in self.transforms
            image, target = t(image, target)
            return image, target

class ToTensor(object):
  def __call__(self, image, target):
      iamge = F.to_tensor(image)
      target = torch.as_tensor(np.array(target), dtype=torch.int64) # torch.LongTensor
      return image, taget

class RandomResize(object):
    def __init__(self,min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size
    
    def __call__(self, image, target):
        size = random.randint(self.min_size, self.max_size)
        image = F.resize(image, size)                                # size 为整数时，以最短边为标准，进行等比缩放
        target = F.resize(target, size, interpolation=Image.NEAREST)
        return image, target

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean 
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=mean, std=std)
        return image,target

class IdtoTrainId(object):
    def __call__(self, image, target):
        labelUtil = LabelUtil()
        image, target = labelUtil.id2TrainId(image, target)
        return image, target