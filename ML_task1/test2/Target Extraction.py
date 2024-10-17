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


image = cv2.imread('./data/317.bmp')

# 转换为HSV颜色空间，初步定位金鱼
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# cv_show('hsv', hsv)

# 金鱼彩色部分
lower_red = np.array([10, 150, 50])
upper_red = np.array([20, 240, 255])
# 金鱼白色部分
lower_white = np.array([70, 10, 150])
upper_white = np.array([130, 50, 255])

mask_red = cv2.inRange(hsv, lower_red, upper_red)
mask_white = cv2.inRange(hsv, lower_white, upper_white)

# 膨胀操作扩大红色和白色区域的掩膜
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # 膨胀核大小可以调整
mask_red_dilated = cv2.dilate(mask_red, kernel, iterations=1)
mask_white_dilated = cv2.dilate(mask_white, kernel, iterations=3)
# 合并红色和白色掩膜
combined_mask = cv2.bitwise_or(mask_red_dilated, mask_white_dilated)
# cv_show('Combined Color Mask', combined_mask)
# 形态学操作去噪
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
combined_mask_clean = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

# 找到掩膜中的轮廓
contours, _ = cv2.findContours(combined_mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 如果找到了轮廓
if contours:
    # 找到面积最大的轮廓
    largest_contour = max(contours, key=cv2.contourArea)
    # 计算边界矩形
    x, y, w, h = cv2.boundingRect(largest_contour)
    # 扩大矩形的宽度和高度
    expansion = 10  # 扩大20个像素
    x = max(0, x - expansion)  # 确保不越界
    y = max(0, y - expansion)
    w += 2 * expansion
    h += 2 * expansion
    # 在原图上绘制边界矩形
    image_with_rect = image.copy()
    cv2.rectangle(image_with_rect, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv_show('Image with Bounding Box', image_with_rect)
    # 提取ROI
    roi = image[y:y + h, x:x + w]
    cv_show('Extracted ROI', roi)