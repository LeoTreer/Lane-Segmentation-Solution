import os
import re
import sys
import pandas as pd 
from sklearn.utils import shuffle

image_dir = ''
label_dir = '' 

env = os.environ.get('ENV', 'development') 

if sys.platform.startswith('win'): 
    image_dir = r'D:\workSpace\CV\1. dataSet\Lane Stagement 初赛数据\ColorImage'
    label_dir = r'D:\workSpace\CV\1. dataSet\Lane Stagement 初赛数据\Gray_Label'
    save_path = r'D:\workSpace\Lane-Segmentation-Solution\data_list'

else:
    image_dir = '/root/data/LaneSeg/Image_Data/'
    label_dir = '/root/data/LaneSeg/Gray_Label/'
    save_path = r'/root/private/Lane-Segmentation-Solution'

image_list = []
label_list = []

def _makeList():
    dir_iter = os.walk(image_dir)
    for parent, dirs, files in dir_iter:
        if len(files):
            # print('parent', parent)
            reObj = re.search( r'(road\d+).*(record\d+).*(camera \d)', parent, re.I)
            road, record, camera = reObj.group(1), reObj.group(2), reObj.group(3)
            label_parent =os.path.join(label_dir, "Label_"+str.lower(road), "Label", record, camera)
            for img in files:
                image = os.path.join(parent, img)
                label = os.path.join(label_parent, img.replace('.jpg','_bin.png'))
                if not os.path.exists(label): # 如果label文件缺失
                    print("数据缺失:", label)
                    continue
                image_list.append(image)
                label_list.append(label)

    assert len(image_list) == len(label_list) # 可用数据 20764 张
    print("The length of image dataset is {}, and label is {}".format(len(image_list), len(label_list)))
    return (image_list, label_list)

def _slicer(image_list, label_list, propotion = (6,2,2)):
    r"""
      数据分割

      Args:
        propotion: (train,test,validation) default: (6,2,2)
        train_num = total*0.6,
        test_num = total*0.2,
        validation_num = total*0.2,
    """

    total_length = len(image_list)
    all_data = pd.DataFrame({'image':image_list, 'label':label_list})
    
    shuffle(all_data)

    train_num = int(total_length*propotion[0]/10)
    test_num = int(total_length*propotion[1]/10)
    val_num = int(total_length*propotion[2]/10)

    train_data = all_data[:train_num]
    test_data = all_data[train_num: train_num + test_num]
    val_data = all_data[train_num + test_num : train_num + test_num + val_num]

    print('The length of (train_date, test_data, val_data, total_num) is ',(len(train_data), len(test_data), len(val_data), len(train_data)+len(test_data)+len(val_data)))
    return (train_data, test_data, val_data)


if __name__ == "__main__":
    train_data, test_data, val_data = _slicer(*_makeList())
    train_data.to_csv('./data_list/train.csv', index=False)
    val_data.to_csv('./data_list/val.csv', index=False)
    test_data.to_csv('./data_list/test.csv', index=False)
