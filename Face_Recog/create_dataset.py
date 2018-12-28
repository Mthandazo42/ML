"""
NAME: DATASET CREATER
DESRCIPTION: USING PYTHON TO CREATE A PYTHON
AUTHOR: MTHANDAZO NDHLOVU
DATE: 27/12/2018
"""

import os
import cv2
#import argparse

#ap = argparse.ArgumentParse()
#ap.add_argument("")

cam = cv2.VideoCapture(0)
cam.set(3, 680)
cam.set(4, 480)

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_id = input("\n Enter user id: ")
print("\n Initializing the camera for photoshot look at the camera")

count = 0
while True:
    ret, img = cam.read()
   # img = cv2.flip(img, -1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255,0,0), 2)
        count += 1
        cv2.imwrite("datasets/User." + str(face_id) + "." + str(count) + ".jpg", gray[y:y+h, x:x+w])
        cv2.imshow("image", img)
        if count >= 30:
            break
        #k = cv2.waitKey(100) & 0xff
        """if k == 27:
            break
        elif count >= 30:
            break"""

print("\n Exiting the program and cleaning up the staff")
cam.release()
cv.destroyAllWindows()
