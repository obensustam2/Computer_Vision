import cv2
import numpy as np

def nothing(x):
    pass

while True:
    frame = cv2.imread('smarties.png')

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # color format conversion from bgr(blue, greeen, red) to hsv(hue, saturation, value)

    l_b = np.array([110, 50, 50]) # lower hsv value for blue color
    u_b = np.array([130, 255, 255]) # upper hsv value for blue color

    mask = cv2.inRange(hsv, l_b, u_b)

    result = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow("frame", frame)
    cv2.imshow("mask", mask)
    cv2.imshow("result", result)

    key = cv2.waitKey(1)
    if key == 27:
        break

cv2.destroyAllWindows()