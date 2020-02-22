import sys
import os
sys.path.append(os.getcwd())
from PIL import Image
import numpy as np
from utils.process_labels import encode_labels, decode_labels, decoder_color_labels

img_path = r"d:\workSpace\Lane-Segmentation-Solution\test\resource"
img_name = (
    "170927_063811892_Camera_5_bin.png",
    "170927_063812246_Camera_5_bin.png",
    "170927_063812587_Camera_5_bin.png"
)

save_path = r"d:\workSpace\Lane-Segmentation-Solution\test\resource"

gray_img1 = Image.open(os.path.join(img_path, img_name[0]))

# gray_img1.show()

# img对象转npArray
# np.array(img_obj) 将img对象转为np.array

img_array = np.array(gray_img1)
print(img_array[1660,1567]) # 查看值

# test encode_labels()
id_label = encode_labels(img_array)
print(np.unique(img_array))
print(np.unique(id_label))

# print full matrix
# np.set_printoptions(threshold=sys.maxsize)
# print(id_label)

# test decode_color_labels
color_label = decoder_color_labels(id_label)
color_img = Image.fromarray(np.transpose(color_label, [1, 2, 0]))
color_img.show()

# test decode_labels()
trainId_label = decode_labels(id_label)
decode_img = Image.fromarray(trainId_label)
# decode_img.show()

# if __name__ == "__main__":
#     print(sys.path)