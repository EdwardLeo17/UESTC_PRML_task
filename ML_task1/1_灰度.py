from scipy import io
from scipy.stats import norm
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import math

"""
任务1：对训练数据用极大似然，估计出两类区域 灰度值 的概率密度函数，并用最小错误贝叶斯对fish.bmpROI灰度图像进行分割。(x维度为1)
任务2：对训练数据用极大似然，估计出两类区域 RGB值 的概率密度函数，并用最小错误贝叶斯对fish.bmpROI彩色图像进行分割。(x维度为3)
"""

# 数据的加载
image = Image.open('./data/309.bmp')
mask = io.loadmat('./data/Mask.mat')['Mask']
sample = io.loadmat('./data/array_sample.mat')['array_sample']

# 图片处理，获取对应的ROI区域
gray_img = np.array(image.convert('L'))     # 灰度ndarray (240, 320)
rgb_img = np.array(image)                   # 原图ndarray (240, 320, 3)
# 【重要】这里/255是为了将[0,255]的像素值映射到[0,1]，因为训练数据给的是[0,1]
gray_roi = gray_img * mask / 255
rgb_roi = rgb_img * mask[:, :, np.newaxis] / 255
# (240, 320) (240, 320, 3)

# 根据label拆分样本
gray1 = []
gray2 = []
rgb1 = []
rgb2 = []
for i in range(len(sample)):
    if sample[i][4] == 1.:
        gray1.append(sample[i][0])
        rgb1.append(sample[i][1:4])
    else:
        gray2.append(sample[i][0])
        rgb2.append(sample[i][1:4])
rgb1 = np.array(rgb1)
rgb2 = np.array(rgb2)

# 统计先验概率
P_pre1 = len(gray1) / len(sample)
P_pre2 = 1 - P_pre1

""" 一维：此时用到的是灰度图片，根据公式计算两类分布的均值和方差 """
gray1_m = np.mean(gray1)    # 第一类的均值
gray1_s = np.std(gray1)     # 第一类的方差
gray2_m = np.mean(gray2)    # 第二类的均值
gray2_s = np.std(gray2)     # 第二类的方差

# 分别绘制 两个分布对应的 类条件概率密度曲线
x = np.arange(0, 1, 1/1000)
gray1_pdf = norm.pdf(x, gray1_m, gray1_s)
gray2_pdf = norm.pdf(x, gray2_m, gray2_s)
plt.plot(x, gray1_pdf, color='red', label='class_1')
plt.plot(x, gray2_pdf, color='blue', label='class_2')
plt.xticks(np.arange(0, 1, 1/20))  # x轴刻度
plt.xlabel('x')
plt.ylabel('p(x|w)')
plt.legend()
plt.show()

# 有了 先验概率 和 类条件概率密度函数，采用MAP进行决策
gray_out = np.zeros_like(gray_img)
for i in range(len(gray_roi)):
    for j in range(len(gray_roi[0])):
        if gray_roi[i][j] == 0:
            continue
        elif P_pre1 * norm.pdf(gray_roi[i][j], gray1_m, gray1_s) > P_pre2 * norm.pdf(gray_roi[i][j], gray2_m, gray2_s):
            gray_out[i][j] = 125
        else:
            gray_out[i][j] = 255

print(gray_out.shape)  # (240, 320)

# 展示分类的图片
gray_out_show = Image.fromarray(gray_out)  # 0*255=0还是黑色，1*255=255表示白色
gray_out_show.show()