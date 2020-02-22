import PIL
import numpy as np
import os
from PIL import Image
from imgaug import augmenters as iaa
from process_labels import decode_labels, decode_labels, decoder_color_labels

def crop_resize_data(image, label=None, new_size=(1024, 384), offset=690):
    """
    crop and resize data
    label interpolated in PIL.Image.NEAREST
    image interpolated in PIL.Image.BILINEAR
    """

    roi_img = image[offset: , : ]

    proc_image = image.resize(new_size, resample = PIL.Image.BILINEAR)
    res = (proc_image, )

    if label is not None:
        roi_label = label[offset: , : ]

        proc_label = image.resize(new_size, resample=PIL.Image.NEAREST)
        res = (proc_image, proc_label)
    return res

def expand_resize_data(prediction=None, submission_size=[3384,17170], offset=690):

    # 数组转换成Image.Image对象
    if not isinstance(prediction, Image.Image):
        prediction = Image.fromarray(prediction)

    pred_mask = decode_labels(prediction) # 生成灰度图
    expand_mask = prediction.resize((submission_size[0], submission_size[1] -offset), resample= Image.NEAREST)
    submission_mask = np.zeros((submission_size[0], submission_size[1]), dtype='uint8')
    submission_mask[offset:, :] = expand_mask
    return submission_mask

def expand_resize_color_data(prediction=None, submission_size=[3384,1701], offset=690):
    if not isinstance(prediction, Image.Image):
        prediction = Image.fromarray(prediction)
    color_pred_mask = decoder_color_labels(prediction)
    color_pred_mask = np.transpose(color_pred_mask, (1,2,0))
    color_expand_mask = color_pred_mask.resize((submission_size[0], submission_size[1] -offset), resample=PIL.Image.NEAREST)
    color_submission_mask = np.zeros((submission_size[1], submission_size[0], 3), dtype='uint8')
    color_submission_mask[offset:, :, :] = color_expand_mask
    return color_submission_mask

def random_crop(image, label):
    random_seed = np.random.randint(0, 10)
    if random_seed < 5:
        return image, label
    else:
        width, height = image.shape[1], image.shape[0]
        new_width = int(float(np.random.randint(88, 99))/100.0 * width)  # TODO 为什么取这个比例？
        new_height = int(float(np.random.randint(88,99))/100.0*height)
        offset_w = np.random.randint(0, width - new_width - 1)           # 随机位置
        offset_h = np.random.randint(0, height - new_height - 1)
        new_image = image[offset_h : offset_h + new_height, offset_w : offset_w + new_width]
        new_label = label[offset_h: offset_h + new_height, offset_w: offset_w + new_width]
        return new_image, new_label

# TODO 这里需要重新实现
def image_augmentation(ori_img):
    random_seed = np.random.randint(0, 10)
    if random_seed > 5:
        seq = iaa.Sequential([iaa.OneOf([
                iaa.Sharpen(alpha=(0.1, 0.3), lightness=(0.7, 1.3)),
                iaa.GaussianBlur(sigma=(0, 1.0))])])
        ori_img = seq.augment_image(ori_img)
    return ori_img

# if __name__ == "__main__":
#     img = Image.fromarray(np.random.randint(0,255,100).reshape((10,10)))
#     print(isinstance(img, Image.Image))

