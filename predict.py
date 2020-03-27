import torch
import torchvision
import torch.nn as nn
import numpy as np
import os
import utils
import transforms as T
import torchvision.transforms as trans
from torchvision.transforms import functional as F
import PIL.Image as Image


def get_dataset():
    transforms = []
    transforms.append(T.ToTensor())
    # transforms.append(T.IdtoTrainId())
    # transforms.append(T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))

    ds = torchvision.datasets.Cityscapes('./dataset/cityscapes',
                                         split="val",
                                         target_type="semantic",
                                         transforms=T.Compose(transforms))
    return ds


def predict(model, loader, device):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter=" ")
    header = "Predict"
    labelUtil = T.LabelUtil()
    with torch.no_grad():
        for image, target in metric_logger.log_every(loader, 100, header):
            image, target = image.to(device), target.to(device)
            output = model(image)
            output = output['out']

            _, output = labelUtil.trainId2Id(None, output)

            image, predict = trans2PIL(image, output)
            utils.parrallelShow(image, target[0], predict, save=True)


def trans2PIL(image, output):
    output = output.argmax(dim=1)
    image, output = ToPILImage(image, output)
    return image, output


def ToPILImage(image, target):
    image = trans.ToPILImage()(image[0])
    label = np.array(target)
    label = label.astype(np.uint8)
    label = Image.fromarray(label[0], mode="L")
    return image, label


def main(args):
    device = args.device
    ds = get_dataset()
    loader = torch.utils.data.DataLoader(ds,
                                         batch_size=1,
                                         num_workers=args.workers,
                                         collate_fn=utils.collate_fn)

    model = torchvision.models.segmentation.fcn_resnet50(pretrained=False,
                                                         num_classes=19,
                                                         progress=True,
                                                         aux_loss=False)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])

    model.to(device)

    predict(model, loader, device)


def parse_arg():
    import argparse
    parser = argparse.ArgumentParser(description="Predict and Output")
    parser.add_argument('-r',
                        "--resume",
                        default='./checkpoint',
                        help='load checkpoint')
    parser.add_argument('--out', default='./validate', help='output dir')
    parser.add_argument('-d',
                        '--device',
                        default="cuda",
                        help='witch device script to use',
                        type=str)
    parser.add_argument('-w',
                        '--workers',
                        default=0,
                        type=int,
                        help='number of data loading workers')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arg()
    utils.checkDevice(args)
    utils.printArgs(args)
    main(args)