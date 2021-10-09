import cv2
import numpy as np
import time
from matplotlib import pyplot as plt
# cv2.namedWindow("show", cv2.WINDOW_AUTOSIZE)

img = cv2.imread('1.jpg')
img_blur = cv2.GaussianBlur(img, (9, 9),0)
cv2.namedWindow('test', cv2.WINDOW_NORMAL)
cv2.resizeWindow('test', 1000, 1000)
gradX = cv2.Sobel(img, ddepth=cv2.CV_32F, dx=1, dy=0)
gradY = cv2.Sobel(img, ddepth=cv2.CV_32F, dx=0, dy=1)
gradient = cv2.subtract(gradX, gradY)
gradient = cv2.convertScaleAbs(gradient)
cv2.imshow('test', gradient)
# (cnts, _) = cv2.contourArea(gradient.copy(),cv2.CHAIN_APPROX_NONE)
# c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
# rect = cv2.minAreaRect(c)
# box = np.int0(cv2.boxPoints(rect))
# draw_img = cv2.drawContours(img.copy(), [box], -1, (0, 0, 255), 3)
cv2.imshow("draw_img", gradient)
cv2.waitKey(0)
