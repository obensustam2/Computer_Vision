import cv2
import numpy as np

img1 = np.zeros((512, 512, 3), np.uint8)
img1 = cv2.rectangle(img1,(200, 0), (300, 100), (255, 255, 255), -1)
img2 = cv2.imread("image_1.png")
img2 = cv2.resize(img2, (512, 512))

bitOr = cv2.bitwise_or(img2, img1)

cv2.imshow("img1", img1)
cv2.imshow("img2", img2)
cv2.imshow("bitOr", bitOr)

cv2.waitKey(0)
#cv2.destroyAllWindows()