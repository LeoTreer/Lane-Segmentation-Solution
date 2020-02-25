"""
Descript: 根据赛事的label说明
      - encode_labels() 将 id 转为 trainId
      - decode_labels() 将 trainId 转为 id
      - decoder_color_labels 将Gray_label转为Color_labels
      - 原trainId == 4我们舍弃，原0，1，2，3，4，5，6，7，8 -》 0，1，2，3，-，4，5，6，7
      - id to trainId Rule
      {
        0:(249, 255, 213, 206, 207, 211, 208, 216, 215, 202, 231, 230, 228, 229, 233, 212, 223),
        1:(200, 204, 209),
        2:(201, 203)，
        3:(217),
        4:(210, 232),
        5:(214),
        6:(220, 221, 222, 224, 225, 226),
        7:(205, 227, 250)
      }
      - trainId to id Rule
      {
        0: 0,
        1: 204,
        2: 203,
        3: 217,
        4: 210,
        5: 214,
        6: 224,
        7: 227
      }
      - trainId to color
      ((0, 0, 0), (70, 130, 180), (0, 0, 142), (153, 153, 153),
       (128, 64, 128), (190, 153, 153), (0, 0, 230), (255, 128, 0))
"""

import numpy as np

id2TrainId = ((249, 255, 213, 206, 207, 211, 208, 216, 215, 202, 231, 230, 228, 229, 233, 212, 223),
              (200, 204, 209), (201, 203), (217,), (220, 221, 222, 224, 225, 226), (205, 227, 250))

traindId2Id = (0, 204, 203, 217, 210, 214, 224, 227)

trainId2Color = ((0, 0, 0), (70, 130, 180), (0, 0, 142), (153, 153, 153),
                 (128, 64, 128), (190, 153, 153), (0, 0, 230), (255, 128, 0))


def encode_labels(color_mask):
    """
    灰度转标签
    id to trainId

    Args:
        color_mask: Gray Label
    Return:
        encode_mask
    """

    encode_mask = np.zeros((color_mask.shape[0], color_mask.shape[1]), dtype="uint8")
    for trainId, ids in enumerate(id2TrainId):
        for id in ids: 
            encode_mask[color_mask == id] = trainId
    return encode_mask


def decode_labels(labels):
    """
    标签转灰度
    trainId to id

    Args:
        labels: encoded label
    Return:
        decode_mask
    """

    decode_mask = np.zeros((labels.shape[0], labels.shape[1]), dtype="uint8")
    for trainId, id in enumerate(traindId2Id):
        decode_mask[labels == trainId] = id
    return decode_mask


def decoder_color_labels(labels):
    """
    灰度图转彩图

    """
    decode_mask = np.zeros(
        (3, labels.shape[0], labels.shape[1]), dtype="uint8")
    for trainId, colors in enumerate(trainId2Color):
        decode_mask[0][labels == trainId] = colors[0]
        decode_mask[1][labels == trainId] = colors[1]
        decode_mask[2][labels == trainId] = colors[2]
    return decode_mask


def verify_labels(labels):
    """
    列出像素值
    """
    pixels = np.unique(labels)
    if 0 not in pixels:
        pixels.append(0)
    return pixels


# if __name__ == "__main__":
#     for trainId, id in enumerate(id2TrainId):
#         print((trainId, id))
