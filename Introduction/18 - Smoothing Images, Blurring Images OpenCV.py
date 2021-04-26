import cv2
import numpy as np
from matplotlib import pyplot as pt

img = cv2.imread('lena.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

kernel = np.ones((5,5), np.float32)/25
dst = cv2.filter2D(img, -1, kernel) # blurred edges
blur = cv2.blur(img, (5,5))
gblur = cv2.GaussianBlur(img, (5,5), 0) # it removes the high frequency noises
median = cv2.medianBlur(img, 5)
bilateralFilter = cv2.bilateralFilter(img, 9, 75, 75) # high frequency noise removal with keeping image edges

titles = ['Image', '2D Convolution', 'Blur', 'Gaussian Blur', 'Median', 'bilateralFilter']
images = [img, dst, blur, gblur, median, bilateralFilter]

for i in range(6):
    pt.subplot(2, 3, i+1), pt.imshow(images[i])
    pt.title(titles[i])
    pt.xticks([]), pt.yticks([])

pt.show()