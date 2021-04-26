import numpy as np
import cv2

#img0 = cv2.imread('lena.jpg', -1)
img0 = np.zeros([512, 512, 3], np.uint8)
img1 = cv2.imread('lena.jpg', 0)
img2 = cv2.imread('lena.jpg', 1)

img0 = cv2.line(img0, (0,0), (255,0), (0, 255, 0), 5)
img0 = cv2.line(img0, (0,0), (0,255), (255, 255, 0), 15)
img0 = cv2.arrowedLine(img0, (0,0), (255,255), (0, 0, 255), 15)

img1 = cv2.rectangle(img1, (384, 0), (399, 128), (0, 0, 255), 5)
img1 = cv2.rectangle(img1, (57, 44), (67, 100), (0, 255, 255), -1)

img2 = cv2.circle(img2, (50, 50), 12, (0, 255, 255), -1)

font = cv2.FONT_HERSHEY_SIMPLEX
img2 = cv2.putText(img2, 'openCV', (10, 500), font, 4, (255, 255, 0), 10, cv2.LINE_AA)

cv2.imshow('img0', img0)
cv2.imshow('img1', img1)
cv2.imshow('img2', img2)

cv2.waitKey(0)
cv2.destroyAllWindows()