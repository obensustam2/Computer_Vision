import cv2

cap = cv2.VideoCapture(0)

if (cap.isOpened() == False):
    print("Unable to read camera feed")

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter('aa.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

while (True):
    ret, frame = cap.read()

    if ret == True:
        out.write(frame)
        cv2.imshow('frame', frame)
        cv2.waitKey(1)

cap.release()
out.release()
cv2.destroyAllWindows()