import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import models.twolayer as model

PATH = './cifar_net.pth'

batch_size = 10
num_workers = 0

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

testset = torchvision.datasets.CIFAR10(root='./data',
                                       train=False,
                                       download=True,
                                       transform=transform)
testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=batch_size,
                                         shuffle=False,
                                         num_workers=num_workers)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
           'ship', 'truck')

net = model.Net()
net.load_state_dict(torch.load(PATH))

testiter = iter(testloader)
images, labels = testiter.next()

outputs = net(images)

_, predicted = torch.max(outputs, 1)


def evaluate_accuracy(iter, net):
    correct, total = 0, 0
    with torch.no_grad():
        net.eval()
        for images, labels in iter:
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.shape[0]
    net.train()
    return (correct / total, total)


def class_accuracy(iter, net):
    class_correct, class_total = list(0. for i in range(10)), list(
        0. for i in range(10))

    with torch.no_grad():
        net.eval()
        for images, labels in iter:
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(labels.shape[0]):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
        net.train()
    for i in range(10):
        print('Accuracy of %5s : %2d %%' %
              (classes[i], 100 * class_correct[i] / class_total[i]))


def class_accuracy_np(iter, net):
    class_correct, class_total = np.zeros(10), np.zeros(10)

    with torch.no_grad():
        net.eval()
        for images, labels in iter:
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            confuse = np.bincount(labels.numpy().flatten() * 10 +
                                  predicted.numpy().flatten(),
                                  minlength=10 * 10).reshape((10, 10))
            class_correct += np.diag(confuse)
            class_total += confuse.sum(axis=1)
        acc = (class_correct / class_total).tolist()
        net.train()
    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * acc[i]))


print('GroundTruth: ',
      ' '.join('%5s' % classes[labels[j]] for j in range(labels.shape[0])))

print('Predicted: ',
      ' '.join('%5s' % classes[predicted[j]] for j in range(labels.shape[0])))

# print('Predict Accuracy is : {0:.5f}, total size is {1:d}'.format(
#     *evaluate_accuracy(testiter, net)))

class_accuracy_np(testiter, net)