import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')

cap = cv2.VideoCapture(0)

while cap.isOpened():
    _, img = cap.read()

    img_gray =cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img_gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0 ,0), 3)

        eye_gray = img_gray[y: y+h, x: x+w]
        eye_color = img[y: y+h, x: x+w]
        eyes = eye_cascade.detectMultiScale(eye_gray)

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(eye_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 3)

    cv2.imshow("img", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()