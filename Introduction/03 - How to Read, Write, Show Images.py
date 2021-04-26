import cv2

img = cv2.imread('lena.jpg', -1) #number represents the colour
cv2.imshow('image', img)
k = cv2.waitKey(10000)

if k == 27: #Esc key
    cv2.destroyAllWindows()
elif k == ord('s'): #s key
    cv2.imwrite('lena_copy.jpg', img) #new jpg file is created
    cv2.destroyAllWindows()


