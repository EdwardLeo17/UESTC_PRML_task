import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt
import math


#正态分布计算函数
def pdf(x, mu, sigma):
    sqrt_2pi = math.sqrt(2 * math.pi)
    index = -0.5 * ((x - mu) / sigma) ** 2
    y = (1 / (sqrt_2pi * sigma)) * np.exp(index)
    return y


mu = 0  # 均值
sigma = 1  # ⽅差

# 绘制正态分布的概率密度函数图形
x_normal = np.linspace(-4, 4, 100)  # x轴范围
y_normal = pdf(x_normal, mu, sigma)  # 计算概率密度函数值
plt.plot(x_normal, y_normal, color='red', label='N(0,1)')
plt.xticks(np.arange(-4, 5, 1))  # x轴刻度
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.title('Normal Distribution')
plt.legend()
plt.show()
