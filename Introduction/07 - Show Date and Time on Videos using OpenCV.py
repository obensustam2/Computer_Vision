import cv2
import datetime

cap = cv2.VideoCapture(0)
print(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) #3
print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) #4

while(cap.isOpened()):
    ret, frame1 = cap.read()  # ret is a boolean variable checks whether frame is available

    if (ret == True):

        fontType = cv2.FONT_HERSHEY_SIMPLEX
        text = 'Width: ' + str(cap.get(3)) + ' Height: ' + str(cap.get(4))
        dateT = str(datetime.datetime.now())

        frame1 = cv2.putText(frame1, text, (150, 100), fontType, 1, (255, 255, 0), 2, cv2.LINE_AA)
        frame1 = cv2.putText(frame1, dateT, (70, 50), fontType, 1, (0, 255, 0), 4, cv2.LINE_AA)


        cv2.imshow('frame1', frame1)

        if cv2.waitKey(1) & 0xFF == ord('a'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
