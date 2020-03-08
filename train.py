import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import utils.img as img
import models.twolayer as model
import torch.optim as optim

device = torch.device('cuda') if torch.cuda.is_available() else torch.device(
    'cpu')

batch_size = 4
num_workers = 0

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# train = torchvision.datasets.Cityscapes('./dataset/cityscapes',
#                                         split='train',
#                                         mode='fine',
#                                         target_type='semantic',
#                                         transform=transform)
# train_iter = torch.utils.data.Dataloader(train,
#                                          batch_size=batch_size,
#                                          shuffle=True,
#                                          num_workers=num_workers)

# test = torchvision.datasets.Cityscapes('./dataset/cityscapes',
#                                        split='test',
#                                        mode='fine',
#                                        target_type='semantic',
#                                        transform=transform)
# test_iter = torch.utils.data.Dataloader(test,
#                                         batch_size=batch_size,
#                                         shuffle=True,
#                                         num_workers=num_workers)

trainset = torchvision.datasets.CIFAR10(root='./data',
                                        train=True,
                                        download=True,
                                        transform=transform)
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=num_workers)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
           'ship', 'truck')

# 查看图片
dataiter = iter(trainloader)
images, labels = dataiter.next()

print("trainset size is ", trainset.__len__())

# img.imgshow(torchvision.utils.make_grid(images))

# print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

net = model.Net()
net.to(device)
# 创建optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:
            print('[%d, %5d] loss:%.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
print('Finished Training')

# save the train model
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)
