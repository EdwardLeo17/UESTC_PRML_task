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
gray_img = np.array(image.convert('L'))  # 灰度ndarray (240, 320)
rgb_img = np.array(image)  # 原图ndarray (240, 320, 3)
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
P_pre1 = len(rgb1) / len(sample)
P_pre2 = 1 - P_pre1

""" 多维：此时用到的是rgb图片，用最大似然估计两个类别条件概率pdf的参数——协方差与均值"""
# 均值
rgb1_m = np.mean(rgb1, axis=0)
rgb2_m = np.mean(rgb2, axis=0)

# # 协方差矩阵1
# cov_sum1 = 0
# cov_sum2 = 0
# for i in range(len(rgb1)):
#     cov_sum1 += np.dot((rgb1[i] - rgb1_m).reshape(3, 1), (rgb1[i] - rgb1_m).reshape(1, 3))
# for i in range(len(rgb2)):
#     cov_sum2 += np.dot((rgb2[i] - rgb2_m).reshape(3, 1), (rgb2[i] - rgb2_m).reshape(1, 3))
# rgb1_cov = cov_sum1/(len(rgb1)-1)
# rgb2_cov = cov_sum2/(len(rgb2)-1)
# # print(rgb1_cov, rgb2_cov)

# 协方差矩阵2
rgb1 -= rgb1_m.reshape(1, 3)
rgb2 -= rgb2_m.reshape(1, 3)
rgb1_cov = np.cov(rgb1, rowvar=False)  # rowvar=False是每一行是一个样本
rgb2_cov = np.cov(rgb2, rowvar=False)
# print(rgb1_cov, rgb2_cov)

# 有了 先验概率 和 类条件概率密度函数，采用MAP进行决策
rgb_out = np.zeros_like(rgb_roi)
for i in range(len(rgb_roi)):
    for j in range(len(rgb_roi[0])):
        if np.sum(rgb_roi[i][j]) == 0:
            continue
        elif P_pre1 * multivariate_normal.pdf(rgb_roi[i][j], rgb1_m, rgb1_cov) > P_pre2 * multivariate_normal.pdf(
                rgb_roi[i][j], rgb2_m, rgb2_cov):  # 贝叶斯公式分子比较
            rgb_out[i][j] = [255, 0, 0]
        else:
            rgb_out[i][j] = [0, 255, 0]

print(rgb_out.shape)  # (240, 320, 3)

# 显示RGB ROI，与彩色分割结果
plt.figure(3)
cx = plt.subplot(1, 1, 1)
cx.set_title('RGB ROI')
cx.imshow(rgb_roi)
plt.figure(4)
cx1 = plt.subplot(1, 1, 1)
cx1.set_title('RGB segment result')
cx1.imshow(rgb_out)
plt.show()
