import numpy as np

# 总结
# 1. 使用arange(start,stop,step,dtype)生成等差序列维度为1的ndarray
# 2. reshape 成相应的行，列向量
# 3. 使用repeat复制，reshape
# 4. C[mask_h,mask_w]
# 5. axis 0 -> x，axis 1 -> y

a = np.arange(0,20*30,1,dtype="uint32")
w_index = np.arange(0,30,dtype="uint8").reshape((-1,30))
h_index = np.arange(0,20,dtype="uint8").reshape((20,-1))
mask_w = w_index.repeat(20,axis=0).reshape((20,30))
mask_h = h_index.repeat(30,axis=1).reshape((20,30))
# print(mask_w)
# print(mask_h)
c = a.reshape((20, 30), order="C")
# print(c)
print(c[mask_h[1:3,14:18],mask_w[4:5,3:7]])

# repeat用法
# matrix1 = np.array([[1,2],[3,4]])
# matrix2 = np.repeat(matrix1,3,axis=1)
# print(matrix2)

# print (np.array(mask_x))
# print (mask_y)
# print(c[mask_x, mask_y])

def make_mask(src):
    src_h, src_w = src.shape[0], src.shape[1]
    current = 0
    next = 0
    batch = 50
    scale_x = 2
    scale_y =3
    while True:

        current = next
        next = min(current + batch,src_h)

        if current >= src_h:
          break

        print('location:',(current,next))
        mask = None
        
        # 没有必要生成完整蒙版，按照批量大小生成部分蒙版即可
        # 假如current在第n行
        # 蒙版大小应该是(src_w, min(current - batch,batch))
        # 蒙版start_h = current, stop = current + batch ,step = 1 

        mask_w = np.arange(0, src_w, 1, dtype="uint16").reshape((-1, src_w)).repeat(next-current, axis=0).reshape(next-current, src_w) # 按行扫描，宽不变
        mask_h = np.arange(current, next , 1, dtype="uint16").reshape((next-current,-1)).repeat(src_w,axis=1).reshape(next-current, src_w)
        # print(src[mask_h, mask_w])

        src_y = (mask_w + 0.5) * scale_x - 0.5
        src_x = (mask_h + 0.5) * scale_y - 0.5

        src_y1 = np.floor(src_y).astype('uint16')
        src_x1 = np.floor(src_x).astype('uint16')
        src_y2 = (src_y + 1).astype('uint16')
        src_x2 = (src_x + 1).astype('uint16')
        src_y2[src_y2>src_w] = src_w
        src_x2[src_x2>src_h] = src_h

        # print(src_y2>src_w)
        # print(src_x2>src_h)
    
        print(src_x1)
        print(src_y1)

        print(src_x2)
        print(src_y2)
        break
if __name__ == "__main__":
    make_mask(np.arange(0,200*300,1).reshape(200,300))
    # a = np.arange(0,200*300,1).reshape(200,300)
    # section =  a[2:5,6:8].copy()
    # print(section)
    # print(a[2][6])
    # section[0][0] = 100
    # print(section)
    # print(a[2][6])