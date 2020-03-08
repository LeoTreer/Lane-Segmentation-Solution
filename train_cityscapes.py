import sys
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import PIL.Image as Image

if sys.platform.startswith('win'):
    import utils.img as img

import numpy as np
from utils.label_tool import LabelUtil

labTool = LabelUtil()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device(
    'cpu')

# ----------hyper param----------
use_dataParallel = True
batch_size = 1
num_workers = 4
num_classes = 10
epoch = 2
#-------------------------------

transform = transforms.Compose([
    transforms.Resize((512, 1024)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

target_transform = transforms.Compose([
    transforms.Resize((512, 1024), interpolation=Image.NEAREST),
    torchvision.transforms.Lambda(labTool.id2TrainId),
    torchvision.transforms.Lambda(
        # lambda a: torch.from_numpy(np.array(a)).type(torch.LongTensor))
        labTool.label2Tensor)
])

train = torchvision.datasets.Cityscapes('./dataset/cityscapes',
                                        split='train',
                                        mode='fine',
                                        target_type='semantic',
                                        transform=transform,
                                        target_transform=target_transform)
trainloader = torch.utils.data.DataLoader(train,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=num_workers)

test = torchvision.datasets.Cityscapes('./dataset/cityscapes',
                                       split='test',
                                       mode='fine',
                                       target_type='semantic',
                                       transform=transform,
                                       target_transform=target_transform)
testloader = torch.utils.data.DataLoader(test,
                                         batch_size=batch_size,
                                         shuffle=True,
                                         num_workers=num_workers)

train_iter = iter(trainloader)
images, labels = train_iter.next()

# --------------查看信息-------------------------
print("trainSet size is %5d" % train.__len__())
print('images shape is ', images.shape)
print('labels shape is ', labels.shape)
print('labels type is ', labels.type())

# img.imgshow(torchvision.utils.make_grid(images))
# img.imgshow(torchvision.utils.make_grid(labels[0]))

print(labels[0].unique())
# ----------------------------------------------

net = torchvision.models.segmentation.fcn_resnet50(pretrained=False,
                                                   progress=True,
                                                   num_classes=num_classes,
                                                   aux_loss=None)

if use_dataParallel:
    net = nn.DataParallel(net)
else:
    net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)

#--------------------------------------------------
# get data
# feed data
# zero_grad
# 计算loss
# loss.backward
# step
#---------------------------------------------------

for epoch in range(epoch):
    running_loss = 0.0
    count = 0
    for images, labels in train_iter:
        count += 1
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images)['out']
        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if count % 200 == 0:
            print('%d, loss:%.3f' % (epoch + 1, running_loss / 200))
        running_loss = 0.0