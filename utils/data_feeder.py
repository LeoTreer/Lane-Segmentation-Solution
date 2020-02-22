import os
import torch
import numpy as np
import pandas as pd
from utils import process_labels
# import process_labels
import torchvision
from PIL import Image
from torch.utils.data import Dataset

class LSSDataset(Dataset):
    """
        百度车道线检测数据集
    """
    def __init__(self, root, dataName,img_transforms=None, label_transforms=None):
        self.root = root
        self.dataName = dataName
        self.img_transforms =img_transforms
        self.label_transforms=label_transforms
        
        #  加载图像列表
        path = os.path.join(root, dataName + '.csv')
        print('csv path is {}'.format(path))

        if os.path.exists(path):
            self.data = pd.read_csv(path)
        else:
            os.error('path {} no found'.format(path))

        if self.data.shape[1] < 2:
            self.image = self.data['image']
            self.lable = None
        else:
            self.image = self.data['image']
            self.label = self.data['label']

    def __getitem__(self, idx):
        img = Image.open(self.image[idx])
        # img = torch.as_tensor(np.transpose(np.array(img), (2, 0, 1)),  dtype=torch.float32)
        # img = np.transpose(np.array(img, dtype=np.uint8), (2, 0, 1))
        # img = np.array(img, dtype=np.uint8)
    
        label = Image.open(self.label[idx]) 
        label = process_labels.decode_labels(np.array(label, dtype=np.uint8)) 
        label = Image.fromarray(label)

        if self.img_transforms is not None:
            img = self.img_transforms(img)
        
        if self.label_transforms is not None:
            label = self.label_transforms(label)

        print('idx is {}'.format(idx))
        return img, label

    def __len__(self):
        return self.data.shape[0]
    

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join('D:\\workSpace\\Lane-Segmentation-Solution')))

    trans = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

    dataset = LSSDataset(root = r'D:\workSpace\Lane-Segmentation-Solution\data_list', dataName='test', transforms=trans)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)
    for img, target in loader:
        print(img)
        print('img:',img.shape)
        print('target:',target['label'].shape)
        break

        


