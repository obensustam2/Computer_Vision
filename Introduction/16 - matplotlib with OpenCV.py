import cv2
from matplotlib import pyplot as plt

img = cv2.imread('lena.jpg', -1)
cv2.imshow('image', img)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # bgr 2 rgb conversion is necessary to have pics with real color in plot format
img = cv2.rectangle(img, (384, 0), (399, 128), (0, 0, 255), 5)

plt.imshow(img)
plt.xticks([]), plt.yticks([]) # hide x and y lines
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()