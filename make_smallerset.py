import os
import pandas as pd
import utils
import sys
import re
from sklearn.utils import shuffle
import shutil


def main(args):
    root = args.root
    size = args.size
    dir_test = os.path.join(root, "leftImg8bit", "test")
    dir_val = os.path.join(root, "leftImg8bit", "val")
    dir_train = os.path.join(root, "leftImg8bit", "train")

    train_img, train_label, train_city = makeList(dir_train)
    val_img, val_label, val_city = makeList(dir_val)
    test_img, test_label, test_city = makeList(dir_test)

    test_csv = pd.DataFrame({
        'image': test_img,
        'label': test_label,
        'city': test_city
    })
    train_csv = pd.DataFrame({
        'image': train_img,
        'label': train_label,
        'city': train_city
    })
    val_csv = pd.DataFrame({
        'image': val_img,
        'label': val_label,
        'city': val_city
    })

    if not args.no_list:
        utils.mkdir(r'./smallerdata/list')
        test_csv.to_csv(open(r'./smallerdata/list/test.csv', 'w'), index=False)
        train_csv.to_csv(open(r'./smallerdata/list/train.csv', 'w'),
                         index=False)
        val_csv.to_csv(open(r'./smallerdata/list/val.csv', 'w'), index=False)

    if args.copy_data:
        root = r'./smallerdata/cityscapes'
        if os.path.exists(root):
            try:
                shutil.rmtree(root)
                print("删除目标文件夹{}".format(root))
            except OSError as error:
                print(error)
                print("File path can not be removed")
                return
        citys = {
            "train": pd.unique(train_city),
            "test": pd.unique(test_city),
            "val": pd.unique(val_city)
        }
        dataset = {"train": train_csv, "val": val_csv, "test": test_csv}
        for index, subset in enumerate(citys):
            for city in citys[subset]:
                datas = dataset[subset]
                # 准备文件夹
                label_dst = os.path.join(root, 'gtFine', subset, city)
                image_dst = os.path.join(root, 'leftImg8bit', subset, city)
                utils.mkdir(label_dst)
                utils.mkdir(image_dst)
                sub_data = datas[datas["city"] == city]
                sub_data = shuffle(sub_data)[:args.size]
                for i, row in sub_data.iterrows():
                    shutil.copy(row["image"], image_dst)
                    shutil.copy(row["label"], label_dst)


def makeList(dir):
    image_list = []
    label_list = []
    city_list = []
    gen = os.walk(dir)
    for root, dirs, files in gen:
        for file in files:
            abs_img = os.path.join(root, file)
            root_lab = root.replace("leftImg8bit", "gtFine")
            abs_lab = os.path.join(root_lab,
                                   file).replace("leftImg8bit.png",
                                                 "gtFine_labelIds.png")
            city = re.search(r'\w*$', root).group()
            if not os.path.exists(abs_lab):
                print("label file {} is not found".format(abs_lab))
                continue
            image_list.append(abs_img)
            label_list.append(abs_lab)
            city_list.append(city)
    return image_list, label_list, city_list


def parse_args():
    import argparse
    parser = argparse.ArgumentParser("Generate small dataset for test")
    parser.add_argument("-r",
                        "--root",
                        default="./dataset/cityscapes",
                        help="root")
    parser.add_argument("-s",
                        "--size",
                        default="10",
                        type=int,
                        help="each classes")
    parser.add_argument("-n",
                        "--no_list",
                        action="store_true",
                        help="don't make data list")
    parser.add_argument("-c",
                        "--copy_data",
                        action="store_true",
                        help="genarator small size dataset")
    args = parser.parse_args()
    utils.printArgs(args)
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)