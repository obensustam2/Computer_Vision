import numpy
import cv2

img = cv2.imread('pic1.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

corners = cv2.goodFeaturesToTrack(gray, 10, 0.3, 10)
corners = numpy.int0(corners)

for i in corners:
    x, y = i.ravel()
    cv2.circle(img, (x, y), 3, [255, 255, 0], -1)

cv2.imshow('Shi-Tomasi Corner Detector', img)

if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()