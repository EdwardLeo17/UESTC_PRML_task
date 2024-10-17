import scipy.io as sio
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt
import math

# 读入309图像
image_path = "../data/311.bmp"
img = Image.open(image_path)

# 灰度化图像
# 将每个像素⽤8个bit表⽰，0表⽰⿊，255表⽰⽩，其他数字表⽰不同的灰度。
# 转换公式：L = R * 299/1000 + G * 587/1000+ B * 114/1000。
img_gray = img.convert('L')

# 将待处理的图片转化为ndarray
img_gray = np.array(img_gray)  # (240, 320) [0-255]
img_rgb = np.array(img)  # (240, 320, 3) [0-255]

# 加载309_Mask，格式为ndarray
mask_path = "../data/311.mat"
mat_data = sio.loadmat(mask_path)
mask = mat_data['mask']  # <class 'numpy.ndarray'> (240, 320)

# 【重点】对图像进行掩码操作
masked_gray = img_gray * mask
masked_rgb = img_rgb * mask[:, :, np.newaxis]  # 将mask重塑为(240, 320, 1) 借助广播机制做乘法

# 将mask后的图像显示出来
masked_gray_show = Image.fromarray(masked_gray)  # 0*255=0还是黑色，1*255=255表示白色
masked_gray_show.show()
masked_rgb_show = Image.fromarray(masked_rgb)
masked_rgb_show.show()


'''
# 两种方式做RGB图像的mask结果
# 【方法1】
masked_rgb = img_rgb * mask[:, :, np.newaxis]  # 将mask重塑为(240, 320, 1) 借助广播机制做乘法
masked_rgb_show = Image.fromarray(masked_rgb)
masked_rgb_show.show()

# 【方法2】
masked_rgb = img_rgb * np.array([mask, mask, mask]).transpose(1, 2, 0)
masked_rgb_show = Image.fromarray(masked_rgb)
masked_rgb_show.show()
'''