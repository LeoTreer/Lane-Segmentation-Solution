import numpy as np

# 测试
from PIL import Image

def resize(src,new_size,padding=True):
    """
    双线性插值算法resize
    使用边缘padding = 1 来避免边缘空值问题
    Args:
      src: 原图像numpy数组，采用h*w*c维度设计.
      new_size: 新尺寸

    Return:
      dst: 目标结果
    """
    
    # 参数声明
    max_batch = 500 

    # 获取原图大小
    src_h, src_w = src.shape[:2]
    new_h, new_w = new_size
    channel = src.shape[0]

    if (src_w, src_h) == (new_w, new_h) :
        return src.copy()

    if padding :
         src = np.pad(src, ((0,1), (0, 1),(0,0)), 'edge')

    # 计算scale
    scale_w , scale_h = float(src_w) / new_w , float(src_h) / new_h

    dst = np.zeros((new_w, new_h, 3),dtype="uint8")

    # 计算batch
    tmp_batch = new_w // 8
    batch = min(tmp_batch, max_batch)
    current = 0
    next = 0
    dst = None 
    while True:

        current = next
        next = min(current + batch, new_h)
        size = next-current

        # 结束条件
        if current >= new_h:
            break

        mask = None

        # 没有必要生成完整蒙版，按照批量大小生成部分蒙版即可
        # 假如pointer在第n行
        # 蒙版大小应该是(src_w,min(current + batch, src_h))
        # 蒙版start_h = current, stop = next ,step = 1 

        mask_w = np.arange(0, new_w, 1, dtype="uint16").reshape((-1, new_w)).repeat(size,axis=0).reshape(size,new_h) # 按行扫描，宽不变
        mask_h = np.arange(current, next , 1, dtype="uint16").reshape((size, -1)).repeat(new_w,axis=1).reshape(size,new_h)
        
        # section = src[mask_h, mask_w].copy()

        # 开始计算

        # 目标在源上的中心点
        src_y = (mask_w + 0.5) * scale_h - 0.5
        src_x = (mask_h + 0.5) * scale_w - 0.5

        # 处理负值
        src_y[src_y <0] =0
        src_x[src_x <0] =0 

        # 计算四个近邻点
        src_y1 = np.floor(src_y).astype('uint16')
        src_x1 = np.floor(src_x).astype('uint16')
        src_y2 = (src_y + 1).astype("uint16")
        src_x2 = (src_x + 1).astype("uint16")

        src_y2[src_y2>=src_h] = src_h 
        src_x2[src_x2>=src_w] = src_w 


        # 双线性插值
        value0 = np.repeat((src_y2 - src_y)[:,:,np.newaxis],3,axis=2) * src[src_x1, src_y1] + np.repeat((src_y - src_y1)[:,:,np.newaxis],3,axis=2) * src[src_x1, src_y2]
        value1 = np.repeat((src_y2 - src_y)[:,:,np.newaxis],3,axis=2) * src[src_x2, src_y1] + np.repeat((src_y - src_y1)[:,:,np.newaxis],3,axis=2) * src[src_x2, src_y2]

        if dst is None:
            dst = (np.repeat((src_x2 - src_x)[:,:,np.newaxis],3,axis=2) * value0 + np.repeat((src_x -src_x1)[:,:,np.newaxis],3,axis=2)*value1).astype("uint8")
        else:
            tmp = (np.repeat((src_x2 - src_x)[:,:,np.newaxis],3,axis=2) * value0 + np.repeat((src_x -src_x1)[:,:,np.newaxis],3,axis=2)*value1).astype("uint8")
            dst = np.vstack((dst,tmp))
    return dst

def _bilinear_interpolation():
    """
    双线性generator实现
   
    Args:
      pass
    Return:
      pass
    """
    pass
    

if __name__ == "__main__":
    img = Image.open(r'D:\workSpace\Lane-Segmentation-Solution\test\resource\EBTl2e1UYAEZCK8.jpg')
    # img.show()
    img_matrix = np.array(img) # 增加维度
    img_proc=  resize(img_matrix,(1500,1500))

    img_proc = Image.fromarray(img_proc)
    # img_proc.show()
    img_proc.save(r'D:\workSpace\Lane-Segmentation-Solution\test\resource\processed.jpg',quality=100)

    # img_proc0 = Image.fromarray(img_proc[:,:,0])
    # img_proc0.show()
    # img_proc1 = Image.fromarray(img_proc[:,:,1])
    # img_proc1.show()
    # img_proc2 = Image.fromarray(img_proc[:,:,2])
    # img_proc2.show()