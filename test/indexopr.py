"""
Decription: 测试numpy矩阵下表运算机制
"""
import numpy as np


def encode(matrix):
    encode_mask = np.zeros((matrix.shape[0], matrix.shape[1]))
    encode_mask[matrix == 201] = 301
    return encode_mask


if __name__ == "__main__":

    # 输入矩阵
    # matrix = np.random.randint(200, 210, (10, 10))
    # print(matrix)
    # print(matrix == 201)
    # print(encode(matrix))

    # 直接使用bool Matrix
    # matrix2 = np.random.randint(0, 2, (10, 10)) == 1
    # print(matrix2)
    # encode_mask = np.zeros((10, 10))
    # encode_mask[matrix2] = 101
    # print(encode_mask)

    # print(encode_mask[0])

    # in 逻辑运算
    # matrix = np.random.randint(0,20, (10,10))
    # print(matrix)
    # print(matrix==(11,12)) # 不行

    # resize
    matrix = np.random.randint(0,20, (3, 10,10))
    print(matrix)
    print(matrix.reshape((10,10,3)))