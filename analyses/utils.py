import PIL.Image as Image
import torch
import torchvision
import torch.nn.functional as F
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号


class PicStatistics(object):
    def __init__(self, dataloader):
        self.loader = dataloader

    def hist_batch_pic(self):
        plt.ion()  # 打开交互模式
        cul = None
        bins = None
        for _, labels in self.loader:
            data = self._transfomer(labels)
            if cul is None:
                cul, bins = np.histogram(data, np.arange(34), density=False)
            else:
                cul += np.histogram(data, np.arange(34), density=False)[0]
            plt.cla()
            x = np.arange(33)
            y = self._nomelize1d(cul)
            map = dict(zip(x, y))
            maxlist = np.argsort(y)[-3:]
            for a in maxlist:
                b = map[a]
                plt.annotate('(%d,%0.3f)' % (a, b),
                             xy=(a, b),
                             xytext=(-20, 10),
                             textcoords='offset points')
            plt.plot(x, y, "ro-", linewidth=2.0)
            plt.grid(True)
            plt.title("更新中")
            plt.xticks(np.arange(0, 34, 1))
            plt.pause(0.1)
        plt.title("更新完成")
        plt.ioff()  # 关闭交互模式
        plt.show()
        return

    def _nomelize1d(self, ndarry):
        max, min = ndarry.max(), ndarry.min()
        return ndarry / ndarry.sum()

    def _transfomer(self, pil_obj):
        ndarry = (np.array(pil_obj)).ravel()
        # a = np.hstack(ndarry, bins=np.arange(34), density=False)
        return ndarry


if __name__ == "__main__":
    dataloader = torchvision.datasets.Cityscapes('./dataset/cityscapes',
                                                 split="train",
                                                 mode="fine",
                                                 target_type="semantic")
    paint = PicStatistics(dataloader)
    paint.hist_batch_pic()

    # a = np.arange(5)
    # hist, bin_edges = np.histogram(a, np.arange(5), density=False)
    # print(a)
    # print(hist)
    # print(bin_edges)

    # rng = np.random.RandomState(10)  # deterministic random data
    # a = np.hstack((rng.normal(size=1000), rng.normal(loc=5, scale=2,
    #                                                  size=1000)))
    # print(a)
    # _ = plt.hist(a, bins='auto')  # arguments are passed to np.histogram
    # plt.title("Histogram with 'auto' bins")
    # plt.show()
