# Pytorch Computer Visualize BaseLine

## Introduction

一个能够根据需要快速运行，调整的基线代码



## 更新记录

| 日期       | 内容                                                         |
| ---------- | ------------------------------------------------------------ |
| 2020/03/02 | -  雄赳赳气昂昂地创建项目                                    |
| 2020/03/26 | -  写了一堆东西后开始稳定输出了<br />-  增加了optimizer读档功能，再跑一次试试 |
|            |                                                              |

## 训练记录

### 第一次

使用先跑0.25-0.5倍随机缩放，CELoss学习基本特征。然后使用CE+Focal优化。再增大图形放大比例。

| No   | Model        | Epoch | Enhance             | Loss     | Metric  |
| ---- | ------------ | ----- | ------------------- | -------- | ------- |
| 1    | fcn_resnet50 | 3     | RandomSize:0.25-0.5 | CE       | MIoU:49 |
| 2    | fcn_resnet50 | 5     | RandomSize:0.25-0.5 | CE+Focal | MIoU:56 |
|      |              |       |                     |          |         |

**总结**

​	每次加载模型时不加载optimizer，导致Miou在第二轮开始时又会下降。3~5个Epoch后才重新恢复到上一轮的值。增大图形放大比例后Miou略有下降。存在 Hard Sample -> [train] 这个类别。focal怎加Miou的主要原因了提升了这个类别的Miou。从0%提升到20%左右 