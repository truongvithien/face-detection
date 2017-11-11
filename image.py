import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

img = cv2.imread('img/IMG_9039.jpg',1)
img_gs = cv2.cvtColor (img, cv2.COLOR_BGR2GRAY)
img_sm_gs = cv2.resize(img_gs, (0,0), fx=0.2, fy=0.2)
img_sm = cv2.resize(img, (0,0), fx=0.2, fy=0.2)

faces = face_cascade.detectMultiScale(img_sm_gs, 1.3, 5)
for (x,y,w,h) in faces:
    cv2.rectangle(img_sm, (x,y), (x+w, y+h), (255,0,0), 2)

cv2.imshow('image',img_sm)
cv2.waitKey(0)
cv2.destroyAllWindows()
