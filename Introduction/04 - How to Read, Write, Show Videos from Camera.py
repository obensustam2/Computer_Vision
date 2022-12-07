import cv2

cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output8.mp4', fourcc, 20.0, (640, 480))

while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        out.write(frame)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray frame', gray)
    cv2.waitKey(1)

cap.release()
out.release()
cv2.destroyAllWindows()
