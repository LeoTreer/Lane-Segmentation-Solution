import torch 
import torchvision
import sys
import time
import numpy as np
from PIL import Image
from progress.bar import Bar
from models import pretraind 
from utils import data_feeder

def main():
    # create progress Bar 
    print('prepare settings')
    bar = Bar('Processing', max=6)

    # set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 
    bar.next()
    
    # 通过系统描述设置线程
    if sys.platform.startswith('win'):
        num_workers = 0 
        root = r'D:\workSpace\Lane-Segmentation-Solution\data_list'
    else:
        device = torch.device('cuda:3')
        num_workers = 4
        root = r"./data_list"
    bar.next()

    # set num_classes,batch_size
    num_classes =4 
    batch_size = 1

    # learning rate and num_epochs
    lr,num_epochs =0.001, 1

    # perpare model 
    model = pretraind.get_model_instance('fcn_resnet50')

    # print model as a file for debug
    # with open('.\debug.log','w') as f:
    #     print(model,file=f)
    bar.next()

    trans=[]
    trans.append(torchvision.transforms.Resize((768,256)))
    trans.append(torchvision.transforms.ToTensor())
    img_transform = torchvision.transforms.Compose(trans)

    trans = []
    trans.append(torchvision.transforms.Resize((768,256),interpolation=Image.NEAREST))
    # trans.append(torchvision.transforms.ToTensor())
    label_transform = torchvision.transforms.Compose(trans)

    print('root is {}'.format(root))
    bar.next()

    # prepare dataset
    train_data = data_feeder.LSSDataset(root = root, dataName='train', img_transforms=img_transform, label_transforms=label_transform, cut_size=1)
    test_data = data_feeder.LSSDataset(root = root, dataName='test', img_transforms=img_transform, label_transforms=label_transform, cut_size=1)
    train_iter = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print('size:',len(train_data))
    bar.next()

    # constrct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr)

    # learning rate adjustor 
    # 每 step_size 个 epochs ir = ir*gamma
    # last_epoches参数是做什么用的？
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)
    
    loss = torch.nn.CrossEntropyLoss()
    bar.next()

    train(model, train_iter, test_iter, loss, batch_size, optimizer, device, num_epochs)

# def train_one_epoch(model, loss, optimizer, data_loader, device, epoch, print_freq):
#     model.train()
#     for x, y in train_iter:
        
#         # 计算y_hat
#         y_hat = net(x)



# loss 函数
# def criterion(inputs,target):
#     pass

#  评估,用测试集计算Metric
def evaluate(model, data_loader, device, num_classes):
    model.eval()

    #  生成混淆矩阵

def _get_confuseMetric(label_true, label_pred, num_class):

    # 提取有效值
    mask = (label_true >= 0) & (label_true < num_class)

    # np.bincount, 生成索引=数值，value=数值出现次数的Vector
    hist = np.bincount(
        # 进行位置映射
        num_class * label_true[mask].astype(int) +
        label_pred[mask] , minlength=num_class ** 2).reshape(num_class, num_class)
    return hist

def label_accuracy_score(label_trues, label_preds, num_class):
    """计算如下Metric
    - overall accuracy
    - mean accuracy
    - mean IU
    - fwavacc
    """
    hist = np.zeros((num_class, num_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _get_confuseMetric(lt.flatten(), lp.flatten(), num_class)
    acc = np.diag(hist).sum() / hist.sum() # Pixel Accuracy
    
    print('diag',np.diag(hist))
    
    with np.errstate(divide='ignore', invalid='ignore'): # 错误管理
        acc_cls = np.diag(hist) / hist.sum(axis=1) # TP / TP + TN
    acc_cls = np.nanmean(acc_cls) # Mean Pixel Accuracy MPA
    with np.errstate(divide='ignore', invalid='ignore'):
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum() # FWIoU
    return acc, acc_cls, mean_iu, fwavacc

def train(net , train_iter, test_iter, loss, batch_size, optimizer, device, num_epochs):

    # 将网络放到GPU
    net = net.to(device)

    loss = loss 

    for epoch in range(num_epochs):
      
      # 初始化相关数据
      mean_loss, train_l_sum, mean_iu,n, batch_count, start = 0.0 , 0.0, 0.0,0, 0, time.time()

      for x, y in train_iter:
          
          print('epoch is {}'.format(epoch))
          # 后续优化
          # y = y['label'] 

          # 将数据放到GPU
          x = x.to(device)
          y = y.to(device)

          # 计算y_hat 
          y_hat = net(x)['out']
        
          # get_numpy

          y_hat_flat = torch.flatten(y_hat[0],1).transpose(1,0)
          target = torch.flatten(y[0],0)

          # 计算loss
          l = loss(y_hat_flat,target.long())

          # 更新梯度
          optimizer.zero_grad()
          l.backward()              # loss值向后传播
          optimizer.step()

          mean_loss += l.cpu().item()
          mean_iu += label_accuracy_score(y.cpu().detach().numpy()[0].astype(np.uint64), y_hat.cpu().detach().numpy().argmax(axis=1)[0], num_class=8)[2]
        #   mean_acc_sum += (y_hat.argmax(axis=1)==y).sum().cpu().item()
          train_l_sum += l.cpu().item() # 把值放到CPU里并取出来
          n += y.shape[0]
          batch_count += 1

      # 每个epoch输出一次test结果
      test_acc = evaluate_accuracy(test_iter, net)
      print('epoch %d, loss %.4f, train mean_iu %.3f, test acc %.3f, time %.1f sec'
            % (epoch + 1, train_l_sum / batch_count, mean_iu / n, test_acc, time.time() - start))

def evaluate_accuracy(data_iter, net, device=None):
      if device is None and isinstance(net, torch.nn.Module):
          device = list(net.parameters())[0].device
      acc_sum, n =0.0 ,0
      with torch.no_grad():
          for x,y in data_iter:
              net.eval()   # 固定BN和DropOutnn
              acc_sum += (net(x.to(device))['out'].argmax(dim=1)[0] == y.to(device)[0]).float().sum().cpu().item()
              net.train()  # 切换回trian模式
              n += y.shape[-2]*y.shape[-1]
      return acc_sum/n

def one_hot_encode(label, num_class=8):

    if isinstance(label,torch.Tensor):
        label = label.numpy()
    
    # 生成mask
    mask = np.tile(np.arange(0,num_class,1).reshape(8,1,1),(1,label.shape[-2],label.shape[-1]))
    one_hot = np.zeros_like(mask)

    # 生成one_hot
    one_hot[mask==label] =1
    one_hot = one_hot.reshape(one_hot.shape[0],-1) 
    one_hot = one_hot.transpose((1,0))
    return torch.Tensor(one_hot)

if __name__ == "__main__":
    main()
