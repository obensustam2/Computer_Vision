from djitellopy import Tello
import cv2
import numpy as np

def initializeTello():
    myDrone = Tello()
    myDrone.connect()
    myDrone.for_back_velocity = 0
    myDrone.left_right_velocity = 0
    myDrone.up_down_velocity = 0
    myDrone.yaw_velocity = 0
    myDrone.speed = 0
    print(myDrone.get_battery())
    myDrone.streamoff()
    myDrone.stream_on()
    return myDrone

def TelloGetFrame(myDrone, w, h):
    myFrame  = myDrone.get_frame_read()
    myFrame = myFrame.frame
    img = cv2.resize(myFrame, (w, h))
    return img

def findFace(img):
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(imgGray, 1.2, 4)
    myFaceListCenter = []
    myFaceListArea = []

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0 ,0), 3)
        cx = x + w//2
        cy = y + h//2
        area = w*h
        myFaceListArea.append(area)
        myFaceListCenter.append([cx, cy])

    if len(myFaceListArea) !=0: # There is a face
        i = myFaceListArea.index(max(myFaceListArea))
        return img, [myFaceListCenter[i], myFaceListArea[i]]
    else:
        return img,[[0, 0], 0]

def trackFace(myDrone, info, w, pid, pError):
    error = info[0][0] - w//2  # Actual value - where it should be
    speed = pid[0]*error + pid[2]*(error-pError)
    speed = int(np.clip(speed, -100, 100))

    if info[0][0] != 0:
        myDrone.yaw_velocity = speed
    else:
        myDrone.for_back_velocity = 0
        myDrone.left_right_velocity = 0
        myDrone.up_down_velocity = 0
        myDrone.yaw_velocity = 0
        myDrone.speed = 0
        error = 0

    if myDrone.send_rc_control:
        myDrone.send_rc_control(myDrone.for_back_velocity,
                                myDrone.left_right_velocity,
                                myDrone.up_down_velocity,
                                myDrone.yaw_velocity)

    return error






