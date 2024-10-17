import cv2
import numpy as np


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


def cv_show(name, image, width=None, height=None, inter=cv2):
    img = resize(image, width, height, inter)
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


"""采用Canny边缘检测"""
image = cv2.imread('./data/311.bmp')

# 转化为灰度图片
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv_show('image_gray', image_gray)
# 【高斯滤波】
image_gaussian = cv2.GaussianBlur(image_gray, (3, 3), 0)
# cv_show('image_gaussian', image_gaussian)
# 【边缘检测】
edge = cv2.Canny(image_gaussian, 40, 80)
cv_show('edge', edge)

# 【形态学操作：膨胀和闭操作】：
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # 更大的内核可以连接更大距离的边缘
edge_dilated = cv2.dilate(edge, kernel, iterations=1)  # 膨胀边缘
closed = cv2.morphologyEx(edge_dilated, cv2.MORPH_CLOSE, kernel)
cv_show('Dilated Edges', edge_dilated)
cv_show('Closed Edges', closed)

contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
mask = np.zeros_like(image_gray)
if contours:
    # 选择最大的轮廓并绘制
    largest_contour = max(contours, key=cv2.contourArea)
    cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    cv_show('Largest Contour', mask)

# 将mask应用到原图像
result = cv2.bitwise_and(image, image, mask=mask)
cv_show('Extracted Object with Smooth Contour', result)
