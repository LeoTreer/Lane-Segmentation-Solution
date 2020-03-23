import sys
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import PIL.Image as Image
import transforms as T
import utils
import time
import datetime
import os
import losses as l


def get_dataset(name, set, transform):
    assert set == "train" or set == "val" or set == "test"
    city_classes = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
                    'traffic_light', 'traffic_sign', 'vegetation', 'terrain',
                    'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
                    'motorcycle', 'bicycle')
    paths = {
        "cityscapes": ('./dataset/cityscapes', torchvision.datasets.Cityscapes,
                       20, city_classes),
        "small": ('./dataset/smallerdata/cityscapes',
                  torchvision.datasets.Cityscapes, 20, city_classes)
    }
    p, ds_fn, num_classes, classes_name = paths[name]
    ds = ds_fn(p,
               split=set,
               mode="fine",
               target_type="semantic",
               transforms=get_transform(train=True))
    return ds, num_classes, classes_name


def evaluate(model, loader, device, num_classes, classes_name=None):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes, classes_name)
    metric_logger = utils.MetricLogger(delimiter=" ")
    header = "Test"
    with torch.no_grad():
        for image, target in metric_logger.log_every(loader, 100, header):
            image, target = image.to(device), target.to(device)
            output = model(image)
            output = output['out']
            confmat.update(target.flatten(), output.argmax(1).flatten())
        # confmat.reduce_from_all_processes()  # 分布式
        return confmat


def get_transform(train):
    base_size = 1024  # 等比缩放，只需要设置最短边
    # crop_size = 480

    min_size = int((0.25 if train else 1.0) * base_size)
    max_size = int((0.5 if train else 1.0) * base_size)
    transforms = []
    transforms.append(T.RandomResize(min_size, max_size))
    if train:
        pass  # 预留
    transforms.append(T.ToTensor())
    transforms.append(T.IdtoTrainId())
    transforms.append(T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    return T.Compose(transforms)


def criterion_ce(inputs, target, classes_num=None, focal=False, dice=False):
    losses = {}
    for name, x in inputs.items():
        losses[name] = nn.functional.cross_entropy(x, target)

    if len(losses) == 1:
        return losses['out']

    return losses['out'] + 0.5 * losses['aux']


def criterion_focal(inputs, target):
    losses = {}
    alpha = [75 / 19] * 20
    alpha[0] = 25
    losses['out'] = nn.functional.cross_entropy(inputs['out'], target)
    losses['focal'] = l.focal_loss(inputs['out'], target)
    return losses['out'] * 0.5 + losses["focal"]


def criterion_dice(inputs, target):
    losses = {}
    losses['out'] = nn.functional.cross_entropy(inputs['out'], target)
    losses['dice'] = l.dice_loss(inputs['out'], target)
    return losses['out'] * 0.5 + losses["dice"]


def get_criterion(name):
    if name == "focal":
        return criterion_focal
    elif name == "dice":
        return criterion_dice
    else:
        return criterion_ce


def train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler,
                    device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr',
                            utils.SmoothedValue(window_size=1, fmt='{value}'))
    header = 'Epoch:[{}]'.format(epoch)
    for image, target in metric_logger.log_every(data_loader, print_freq):
        image, target = image.to(device), target.to(device)
        output = model(image)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lr_scheduler.step()
        metric_logger.update(loss=loss.item(),
                             lr=optimizer.param_groups[0]["lr"])


def get_CSV(model="train"):
    title = ("epoch", 'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
             'traffic_light', 'traffic_sign', 'vegetation', 'terrain', 'sky',
             'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
             'bicycle')
    acc_report = utils.CSVUtil(r"./", "acc.csv", title=title, model=model)
    accg_report = utils.CSVUtil(r"./",
                                "accg.csv",
                                title=("epoch", "accg"),
                                model=model)
    iu_report = utils.CSVUtil(r"./", "iu.csv", title=title, model=model)
    return acc_report, accg_report, iu_report


def main(args):

    # criterion
    criterion = get_criterion(args.criterion)

    # set[device]
    device = args.device
    if args.device == "cuda":
        device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device(device)

    # set[detaset]
    if args.smalldata:
        train_set, num_classes, classes_name = get_dataset(
            "small", "train", get_transform(True))
        test_set, _, _ = get_dataset("small", "test", get_transform(False))
    else:
        train_set, num_classes, classes_name = get_dataset(
            "cityscapes", "train", get_transform(True))
        test_set, _, _ = get_dataset("cityscapes", "test",
                                     get_transform(False))

    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=args.batch_size,
                                               num_workers=args.workers,
                                               collate_fn=utils.collate_fn,
                                               drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=1,
                                              num_workers=args.workers,
                                              collate_fn=utils.collate_fn)

    # set[model]
    model = torchvision.models.segmentation.__dict__[args.model](
        pretrained=args.pretrained,
        num_classes=num_classes,
        progress=True,
        aux_loss=args.aux_loss)

    model.to(device)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    if args.test_only:
        # set[report]
        acc_report, accg_report, iu_report = get_CSV(model="val")
        confmat = evaluate(model,
                           test_loader,
                           device=device,
                           num_classes=num_classes)

        acc_globle, acc, iu = confmat.compute()

        # 保存数据
        accg_report.append(acc_globle.cpu().numpy()[None, ])
        acc_report.append(acc.cpu().numpy()[None, ])
        iu_report.append(iu.cpu().numpy()[None, ])

        print(confmat)

        return

    # 只传递需要梯度的参数
    # TODO 为什么需要分开列出来？
    params_to_optimize = [{
        "params": [p for p in model.backbone.parameters() if p.requires_grad]
    }, {
        "params":
        [p for p in model.classifier.parameters() if p.requires_grad]
    }]

    # TODO aux-loss
    if args.aux_loss:
        print("TODO aux_loss is now unavalible")

    optimizer = torch.optim.SGD(params_to_optimize,
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # x : epoch 当前第几个epoch
    # lr = initial_lr*function(epoch)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: (1 - epoch /
                                  (len(train_loader) * args.epochs))**0.9)

    start_time = time.time()

    # set[report]
    acc_report, accg_report, iu_report = get_CSV()

    for epoch in range(args.epochs):
        train_one_epoch(model, criterion, optimizer, train_loader,
                        lr_scheduler, device, epoch, args.print_freq)
        confmat = evaluate(model,
                           test_loader,
                           device=device,
                           num_classes=num_classes)
        acc_globle, acc, iu = confmat.compute()

        # 保存数据
        accg_report.append(acc_globle.cpu().numpy()[None, ])
        acc_report.append(acc.cpu().numpy()[None, ])
        iu_report.append(iu.cpu().numpy()[None, ])

        print(confmat)

        utils.save(
            {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'args': args
            },
            os.path.join(
                args.output_dir,
                'model{model}_{epoch}.pth'.format(model=args.model,
                                                  epoch=epoch)))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Trian Model Using CityScape')
    parser.add_argument('-m',
                        "--model",
                        default='fcn_resnet50',
                        choices=('deeplabv3_resnet50', 'deeplabv3_resnet101',
                                 'fcn_resnet50', 'fcn_resnet10'),
                        help='Crate from Torchvision')

    # device 默认：None
    parser.add_argument('-d',
                        '--device',
                        default="cuda",
                        help='witch device script to use',
                        type=str)
    # bach_size 默认：8
    parser.add_argument('-b',
                        '--batch_size',
                        default=4,
                        help='batch_size',
                        type=int)
    # epochs 默认：30
    parser.add_argument('-e', '--epochs', default=30, help='epochs', type=int)
    # worker 默认：4
    parser.add_argument('-w',
                        '--workers',
                        default=4,
                        type=int,
                        help='number of data loading workers')
    # lr 默认 le-2
    parser.add_argument('--lr',
                        default=0.01,
                        type=float,
                        help='initial learnign rate')
    # momentum 默认：0.9
    parser.add_argument('--momentum',
                        default=0.9,
                        type=float,
                        help='for optim')
    # weight-decay 默认：le-4
    parser.add_argument('--wd',
                        '--weight-decay',
                        default=1e-4,
                        type=float,
                        metavar='W',
                        help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    # aux_loss
    parser.add_argument('--aux-loss',
                        action='store_true',
                        help='auxiliar loss')
    # freq 默认：10
    parser.add_argument('--print_freq',
                        default=10,
                        type=int,
                        help='print frequency')
    # output_dir
    parser.add_argument('--output_dir',
                        default='./checkpoint',
                        help='path where to save')
    # resume 读档文件名
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    # test model
    parser.add_argument('--test_only',
                        action='store_true',
                        help='Only test the model')
    # pretrained
    parser.add_argument('--pretrained',
                        help='Use Pre-trained models from the modelzoo',
                        action='store_true')

    parser.add_argument('--smalldata',
                        help='Use smalldata',
                        action='store_true')

    parser.add_argument('-c',
                        "--criterion",
                        default='ce',
                        choices=("ce", "focal", "dice"),
                        help='criterion')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    utils.checkDevice(args)
    utils.printArgs(args)
    main(args)