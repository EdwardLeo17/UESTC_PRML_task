import scipy.io as sio
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt
import math

# 加载309_Mask，格式为ndarray
mask_path = "./data/Mask.mat"
mat_data = sio.loadmat(mask_path)
mask = mat_data['Mask']  # <class 'numpy.ndarray'> (240, 320)

# 读入309图像
image_path = "./data/309.bmp"  # (320, 240)
img = Image.open(image_path)

# # 显示图像
# img.show()

# # 打印图像大小和属性
# print(img.size, img.mode, img.format, type(img))

# 灰度化图像
# 将每个像素⽤8个bit表⽰，0表⽰⿊，255表⽰⽩，其他数字表⽰不同的灰度。
# 转换公式：L = R * 299/1000 + G * 587/1000+ B * 114/1000。
img_gray = img.convert('L')
# img_gray.show()

# 将mask转化为图像
mask_show = Image.fromarray(mask * 255)  # 0*255=0还是黑色，1*255=255表示白色
# mask_show.show()

# 将待处理的图片转化为ndarray
img_rgb = np.array(img)  # (240, 320, 3)
img_gray = np.array(img_gray)  # (240, 320)

# 对图像进行掩码操作
masked_gray = img_gray * mask/255
mask_rgb = np.array([mask, mask, mask]).transpose(1, 2, 0)  # 将mask也转化为(240, 320, 3)
masked_rgb = img * img_rgb/255

# 将mask后的图像显示出来
masked_gray_show = Image.fromarray(masked_gray * 255)  # 0*255=0还是黑色，1*255=255表示白色
