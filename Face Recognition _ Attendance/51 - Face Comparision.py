import cv2
import face_recognition

# Original Image
imgOrg = face_recognition.load_image_file('Images Attendance/Oben.JPEG')
imgOrg = cv2.resize(imgOrg, (580, 780))
imgOrg = cv2.cvtColor(imgOrg, cv2.COLOR_BGR2RGB)
faceLocation = face_recognition.face_locations(imgOrg)[0]
encodeOrg = face_recognition.face_encodings(imgOrg)[0]
cv2.rectangle(imgOrg, (faceLocation[3], faceLocation[0]), (faceLocation[1], faceLocation[2]), (255, 0, 255), 2)

# Test Image
imgTest = face_recognition.load_image_file('Images Attendance/Tenzile.jpeg')
imgTest = cv2.resize(imgTest, (580, 780))
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)
faceLocationTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (faceLocationTest[3], faceLocationTest[0]),
             (faceLocationTest[1], faceLocationTest[2]), (255, 0, 255), 2)

results = face_recognition.compare_faces([encodeOrg], encodeTest)
faceDistance = face_recognition.face_distance([encodeOrg], encodeTest)
print(results, faceDistance)
cv2.putText(imgTest, f'{results} {round(faceDistance[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)

# Show Images
cv2.imshow('Original Image', imgOrg)
cv2.imshow('Test Image', imgTest)
cv2.waitKey(0)