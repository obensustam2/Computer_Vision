import numpy as np
import cv2

img = cv2.imread('messi5.jpg')
img2 = cv2.imread('opencv-logo.png')

print(img.shape) # returns a tuple of number of the rows, columns and channels
print(img.size) # returns the total number of pixels is accessed
print(img.dtype) # returns image data type which is obtained

b, g, r = cv2.split(img)
img = cv2.merge((b, g, r))

ball = img[280:340, 330:390] # region of interest(ROI)
img[273:333, 100:160] = ball

img = cv2.resize(img, (512, 512))
img2 = cv2.resize(img2, (512, 512))


#dst = cv2.add(img, img2) # img2 is added on img
dst = cv2.addWeighted(img, .6, img2, .6, 0) # img is dominant compared to img2
cv2.imshow('image', dst)

cv2.waitKey(0)
cv2.destroyAllWindows()