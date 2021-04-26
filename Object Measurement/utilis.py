import cv2
import numpy as np

def getContours(img, cThr=[100, 100], showCanny=False, minArea=1000, filter=0, draw=False):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, cThr[0], cThr[1])

    kernel = np.ones((5, 5))
    imgDila = cv2.dilate(imgCanny, kernel, iterations=3) # dilation
    imgEros = cv2.erode(imgDila, kernel, iterations=2) # erosion

    if showCanny: cv2.imshow('Canny', imgEros)

    contours, hiearchy = cv2.findContours(imgEros, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    finalContours = []

    for i in contours:
        area = cv2.contourArea(i)

        if area > minArea:
            peri = cv2.arcLength(i, True) # contour perimeter
            approx = cv2.approxPolyDP(i, 0.02 * peri, True) # contour approximation
            bbox = cv2.boundingRect(approx)

            if filter > 0:
                if len(approx) == filter:
                    finalContours.append([len(approx), area, approx, bbox, i])
            else:
                finalContours.append([len(approx), area, approx, bbox, i])

    finalContours = sorted(finalContours, key=lambda x: x[1], reverse=True)

    if draw:
        for con in finalContours:
            cv2.drawContours(img, con[4], -1, (0, 0, 255), 3)

    return img, finalContours


def reOrder(myPoints):
    print(myPoints.shape)
    myPointsNew = np.zeros_like(myPoints)
    myPoints = myPoints.reshape((4,2)) # 4 points, x&y(2) for each points

    add = myPoints.sum(1) # on x axis
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]

    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]

    return myPointsNew

def warpImg(img, points, w, h):
    print(points)
    print(reOrder(points))








