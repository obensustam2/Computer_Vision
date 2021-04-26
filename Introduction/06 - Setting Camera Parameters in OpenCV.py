import cv2

cap = cv2.VideoCapture(0)
print(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) #3
print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) #4

cap.set(3, 3000)
cap.set(4, 3000)

print(cap.get(3))
print(cap.get(4))

while(cap.isOpened()):
    ret, frame1 = cap.read() #ret is a boolean variable checks whether frame is available
    if(ret == True):

        gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)  # colourconversion
        cv2.imshow('frame1', gray)

        if cv2.waitKey(1) & 0xFF == ord('a'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()