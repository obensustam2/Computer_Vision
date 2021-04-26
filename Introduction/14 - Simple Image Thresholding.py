import cv2
import numpy as np

img = cv2.imread('gradient.png', 0)

#first parameter is the thresholding value. Which is 100 below

_, th1 = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY) # the pixel value which is lower than 100 will be 0. Higher than 100 will be 1
_, th2 = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY_INV) # vise versa of BINARY
_, th3 = cv2.threshold(img, 186, 255, cv2.THRESH_TRUNC) # until first parameter(186), colors remain original, then they(186-255) get same pixel value(186) as first parameter
_, th4 = cv2.threshold(img, 100, 255, cv2.THRESH_TOZERO) # the pixel value which is lower than 100 will be 0. Higher than 100 will remain the same
_, th5 = cv2.threshold(img, 100, 255, cv2.THRESH_TOZERO_INV) # the pixel value which is higher than 100 will be 0. Lower than 100 will remain the same

cv2.imshow("Image", img)
cv2.imshow("th1", th1)
cv2.imshow("th2", th2)
cv2.imshow("th3", th3)
cv2.imshow("th4", th4)
cv2.imshow("th5", th5)

cv2.waitKey(0)
cv2.destroyAllWindows()