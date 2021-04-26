from Tello_Main_48 import *
import cv2

w, h = 360, 240
pid = [0.5, 0, 0.5]
pError = 0
startCounter = 0 # For no Flight 1 - For Flight 0

myDrone = initializeTello()

while True:

    ## Flight
    if startCounter == 0:
        myDrone.takeoff()
        startCounter = 1
    ## Step 1
    img = TelloGetFrame(myDrone, w, h)

    ## Step 2
    img, info = findFace(img)

    ## Step 3
    pError = trackFace(myDrone, info, w, pid, pError)
    print(info[0][0])
    cv2.imshow('Image', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        myDrone.land()
        break

