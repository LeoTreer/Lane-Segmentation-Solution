import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv(r'report\3ea74\3ea74_iu.csv')

print(
    df.plot(
        x="epoch",
        y=[
            'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
            'traffic_light', 'traffic_sign', 'vegetation', 'terrain', 'sky',
            'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
            'bicycle'
        ],
        # color="b",
        # marker="o",
        legend=True))

plt.title('train report')
plt.xlabel('epoch')
plt.ylabel('%')
plt.show()

# 根据数据绘制图形
fig = plt.figure(dpi=128, figsize=(10, 6))
#处的实参alpha 指定颜色的透明度。 Alpha 值为0表示完全透明，
#1（ 默认设置） 表示完全不透明
plt.plot(dates, highs, c='red', alpha=0.5)
plt.plot(dates, lows, c='blue', alpha=0.5)

# 设置图形的格式
plt.title("Daily high and low Stock, 2017-(01,05)", fontsize=24)
plt.xlabel('', fontsize=16)
fig.autofmt_xdate()
plt.ylabel("Number(F)", fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.show()