import torchvision

# use pretreained model for quick start

num_classes = 8

pretraind = {}

# deeplabv3
pretraind['deeplabv3_resnet50'] = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False, progress=True, num_classes=num_classes, aux_loss=None)
pretraind['deeplabv3_resnet101'] = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=False, progress=True, num_classes=num_classes, aux_loss=None)

# fcn
pretraind['fcn_resnet50'] = torchvision.models.segmentation.fcn_resnet50(pretrained=False, progress=True, num_classes=num_classes, aux_loss=None)
pretraind['fcn_resnet101'] = torchvision.models.segmentation.fcn_resnet101(pretrained=False, progress=True, num_classes=num_classes, aux_loss=None)


def get_model_instance(model_name):
    """
    预置了
        deeplabv3_resnet50
        deeplabv3_resnet101
        fcn_resnet50
        fcn_resnet101
    """
    return pretraind[model_name]
  
if __name__ == "__main__":
    import sys
    import os
    import torch
    import numpy as np 
    sys.path.append(os.path.abspath(os.path.join('D:\\workSpace\\Lane-Segmentation-Solution')))
    print(sys.path)
    from utils.data_feeder import LSSDataset

    model = get_model_instance('fcn_resnet50')

    trans = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

    data = LSSDataset(root = r'D:\workSpace\Lane-Segmentation-Solution\data_list', dataName='train', transforms=trans)
    loader = torch.utils.data.DataLoader(data, 1, shuffle=False)

    img, target = next(iter(loader))

    out = model(img)

    print(out.shape)

    with open(r'D:\workSpace\Lane-Segmentation-Solution\debug.log', 'w') as f:
        print(model,file=f,)
