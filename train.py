import torch 
import torchvision
import sys
import time
import numpy as np
from PIL import Image
from models import pretraind 
from utils import data_feeder

def main():

    # set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 
    # device = torch.device('cpu')
    
    # 通过系统描述设置线程
    if sys.platform.startswith('win'):
        num_workers = 0 
        root = r'D:\workSpace\Lane-Segmentation-Solution\data_list'
    else:
        device = torch.device('cuda:3')
        num_workers = 4
        root = r"./data_list"

    # set num_classes,batch_size
    num_classes =2 
    batch_size = 1 

    # learning rate and num_epochs
    lr,num_epochs =0.001, 5

    # perpare model 
    model = pretraind.get_model_instance('fcn_resnet50')

    # print model as a file for debug
    with open('.\debug.log','w') as f:
        print(model,file=f)

    trans=[]
    trans.append(torchvision.transforms.Resize((768,256)))
    trans.append(torchvision.transforms.ToTensor())
    img_transform = torchvision.transforms.Compose(trans)

    trans = []
    trans.append(torchvision.transforms.Resize((768,256),interpolation=Image.NEAREST))
    trans.append(torchvision.transforms.ToTensor())
    label_transform = torchvision.transforms.Compose(trans)

    print('root is {}'.format(root))

    # prepare dataset
    train_data = data_feeder.LSSDataset(root = root, dataName='train', img_transforms=img_transform, label_transforms=label_transform)
    test_data = data_feeder.LSSDataset(root = root, dataName='test', img_transforms=img_transform, label_transforms=label_transform)
    train_iter = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

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
    # confmat = 

def _get_confuseMetric(label_true, label_pred, n_class):

    # 提取有效值
    mask = (label_true >= 0) & (label_true < n_class)

    # np.bincount, 生成索引=数值，value=数值出现次数的Vector
    hist = np.bincount(
        # 进行位置映射
        n_class * label_true[mask].astype(int) +
        label_pred[mask] , minlength=n_class ** 2).reshape(n_class, n_class)
    return hist

def label_accuracy_score(label_trues, label_preds, n_class):
    """计算如下Metric
    - overall accuracy
    - mean accuracy
    - mean IU
    - fwavacc
    """
    hist = np.zero((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _get_confuseMetric(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum() # Pixel Accuracy
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
          print(x.shape)
          y_hat = net(x)['out']
                 
        #   one_ hot = 
          # 计算loss
          l = loss(one_hot_encode(y_hat.argmax(axis=1)), one_hot_encode(y))

          # 更新梯度
          optimizer.zero_grad()
          l.backward()              # loss值向后传播
          optimizer.step()

          mean_loss += l.cpu().item()
          mean_iu += label_accuracy_score(y, y_hat,  num_class=8)[2].cpu().item()
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
              acc_sum += (net(x.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
              net.train()  # 切换回trian模式
              n += y.shape[0]
      return acc_sum/n

def one_hot_encode(label, num_class=8):

    if isinstance(label,torch.Tensor):
        label = label.numpy()
    # 生成mask
    mask = np.tile(np.arange(0,num_class,1).reshape(8,1,1),(1,10,10))
    one_hot = np.zeros_like(mask)

    # 生成one_hot
    one_hot = one_hot[mask==label] =1
    return one_hot

if __name__ == "__main__":
    main()
