import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe'

img = cv2.imread('plate3.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imshow('Original Image', img)

# Detecting Words & Numbers
hImg, wImg, _ = img.shape
boxes = pytesseract.image_to_data(img)

for x, b in enumerate(boxes.splitlines()):
    if x != 0:
        b = b.split()
        if len(b) == 12:
            print(b)

            x, y, w, h = int(b[6]), int(b[7]), int(b[8]), int(b[9])
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(img, b[11], (x, y-20), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 2)

cv2.imshow('Result', img)
cv2.waitKey(0)
