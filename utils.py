from collections import defaultdict, deque
import datetime
import math
import time
import torch
import torch.distributed as dist
import pandas as pd
import numpy as np
from prettytable import PrettyTable

import errno
import os


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """
    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total],
                         dtype=torch.float64,
                         device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(median=self.median,
                               avg=self.avg,
                               global_avg=self.global_avg,
                               max=self.max,
                               value=self.value)


class ConfusionMatrix(object):
    def __init__(self, num_classes, classes_name=None):
        self.num_classes = num_classes
        self.mat = None
        self.classes_name = classes_name

    # 原理基本上和numpy版本一样
    def update(self, a, b):
        n = self.num_classes
        if self.mat is None:  # 懒工厂模式
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=a.device)
        with torch.no_grad():
            k = (a >= 0) & (a < n)  # 生成有效值mask
            inds = n * a[k].to(torch.int64) + b[k]
            self.mat += torch.bincount(inds, minlength=n**2).reshape(n, n)

    def reset(self):
        self.mat.zero_()

    # 计算acc， iu
    def compute(self):
        h = self.mat.float()
        acc_global = torch.diag(h).sum() / h.sum()
        acc = torch.diag(h) / h.sum(1)
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return acc_global, acc, iu

    # 分布式计算
    def reduce_from_all_processes(self):
        if not torch.distributed.is_available():
            return
        if not torch.distributed.is_initialized():
            return
        torch.distributed.barrier()
        torch.distributed.all_reduce(self.mat)

    def __str__(self):
        acc_global, acc, iu = self.compute()
        if self.classes_name is not None:
            return ('global correct: {:.1f}\n'
                    'average row correct: {}\n'
                    'IoU: {}\n'
                    'mean IoU: {:.1f}').format(acc_global.item() * 100, [
                        '{}:{:.1f}'.format(self.classes_name[i], i)
                        for i in (acc * 100).tolist()
                    ], [
                        '{}:{:.1f}'.format(self.classes_name[i], i)
                        for i in (iu * 100).tolist()
                    ],
                                               iu.mean().item() * 100)
        else:
            return ('global correct: {:.1f}\n'
                    'average row correct: {}\n'
                    'IoU: {}\n'
                    'mean IoU: {:.1f}').format(
                        acc_global.item() * 100,
                        ['{:.1f}'.format(i) for i in (acc * 100).tolist()],
                        ['{:.1f}'.format(i) for i in (iu * 100).tolist()],
                        iu.mean().item() * 100)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header, '[{0' + space_fmt + '}/{1}]', 'eta: {eta}', '{meters}',
                'time: {time}', 'data: {data}', 'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header, '[{0' + space_fmt + '}/{1}]', 'eta: {eta}', '{meters}',
                'time: {time}', 'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(
                        log_msg.format(i,
                                       len(iterable),
                                       eta=eta_string,
                                       meters=str(self),
                                       time=str(iter_time),
                                       data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {}'.format(header, total_time_str))


# torch.save
def save(*args, **kwargs):
    torch.save(*args, **kwargs)


def printArgs(args):
    title = []
    value = []
    for key in args.__dict__:
        title.append(key)
        value.append(args.__dict__[key])
    table = PrettyTable(title)
    table.add_row(value)
    print(table)


def checkDevice(args):
    if "cuda" in args.device:
        args.device = args.device if torch.cuda.is_available(
        ) else torch.device('cpu')


# 根据输入形状填充右下padding成同一形状
def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images), ) + max_size  # 列表连接
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs


def collate_fn(batch):
    images, targets = list(zip(*batch))
    batched_imgs = cat_list(images, fill_value=0)
    batched_targets = cat_list(targets, fill_value=255)
    return batched_imgs, batched_targets


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


class CSVUtil(object):
    def __init__(self, root, file_name, title, model="train"):
        self.root = root
        self.file_name = file_name
        self.title = title
        self.path = os.path.join(root, "report", model, self.file_name)
        dirpath = os.path.join(root, "report", model)
        if not os.path.exists(dirpath):
            mkdir(dirpath)
        self.createCSV(self.path, self.title)

    def createCSV(self, path, title=None, **kwargs):
        if title:
            assert isinstance(title, tuple)
            df_empty = pd.DataFrame(columns=title)
            df_empty.to_csv(path, index=False)
        else:
            obj = {}
            for key, value in kwargs.items():
                obj[key] = value
            df = pd.DataFrame(obj)
            df.to_csv(path, index=False)

    def append(self, obj):
        df = pd.DataFrame(obj)
        df.to_csv(self.path, mode="a", header=False)


def one_hot_encode(target, classes_num, ignore_index=None):
    """one_hot encode
    Args:
        labels: A tensor of shape (N,H,W)
        classes_num: num of classes
    Return:
        output: A tensor of shape (N, classes_num , H, W)
    """
    assert len(target.shape) == 3
    n, h, w = target.shape
    return torch.zeros(n, classes_num, h,
                       w).to(target.device).scatter_(1, target.unsqueeze(1), 1)


# if __name__ == "__main__":
# CSV 测试
# util = CSVUtil(r"./", "test.csv", title=("epoch", "acc", "accg", "iu"))
# util.append(np.array([[1, 2, 3, 4], [5, 6, 7, 8]]))

# one_hot
# target = torch.tensor(
#     np.random.randint(0, 8, 100, dtype=np.int64).reshape(1, 10, 10))
# one_hot = one_hot_encode(target, 8)
# print(one_hot.shape)
