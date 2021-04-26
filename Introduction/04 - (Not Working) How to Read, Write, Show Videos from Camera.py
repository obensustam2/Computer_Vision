import cv2

cap = cv2.VideoCapture(0)

if (cap.isOpened() == False):
    print("Unable to read camera feed")

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter('output8.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if(ret == True):
        cv2.imshow('gray frame', gray)
        out.write(gray)
        cv2.waitKey(1)

cap.release()
out.release()
cv2.destroyAllWindows()