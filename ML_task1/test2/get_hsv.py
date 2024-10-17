import cv2
import numpy as np
from matplotlib import pyplot as plt

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
def getpos(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # 定义一个鼠标左键按下去的事件
        print(HSV[y, x])


image = cv2.imread('data/311.bmp')
image = resize(image, width=800)

HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

cv2.imshow("imageHSV", HSV)
cv2.imshow('image', image)
cv2.setMouseCallback("imageHSV", getpos)
cv2.waitKey(0)