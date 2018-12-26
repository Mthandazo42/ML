"""
NAME: AGE AND GENDER PREDICTION USING OPENCV
AUTHOR: MTHANDAZO NDHLOVU
DATE: 26/12/2018
DESCRIPTION: USAGE OF OPENCV TO DO GENDER AND AGE PREDICTIONS ON VIDEO

"""

#imports

import numpy as np
import cv2
import pafy
import argparse

#Link to youtube video which will be analysed
url = 'https://youtu.be/Rsj9O-ZutIQ'
#vpafy = pafy.new(url)
#grap the .mp4 format of the video
#play = vpafy.getbest(preftype="mp4")

#ap = argparse.ArgumentParser()
#ap.add_argument("-v", "--video", required=True, help="path to the video file")
#args = vars(ap.parse_args())

cap = cv2.VideoCapture(0)
#set width and the height of the frame
cap.set(3, 480)
cap.set(4, 640)

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)','(60, 100)']
gender_list = ['Male', 'Female']

#load the models and their respective prototxt
def load_caffe_models():
    age_net = cv2.dnn.readNetFromCaffe('deploy_age.prototxt', 'age_net.caffemodel')
    gender_net = cv2.dnn.readNetFromCaffe('deploy_gender.prototxt', 'gender_net.caffemodel')
    return (age_net, gender_net)

#video detector
def video_detector(age_net, gender_net):
    font = cv2.FONT_HERSHEY_SIMPLEX
    while True:
        ret, image = cap.read()
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        if (len(faces) > 0):
            print(f"Found {str(len(faces))} faces")

            for (x, y, w, h) in faces:
                cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 0), 2)
                #obtain face from faces
                face_img = image[y:y+h, h:h+w].copy()
                blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

                #predict gender
                gender_net.setInput(blob)
                gender_preds = gender_net.forward()
                gender = gender_list[gender_preds[0].argmax()]
                print("Gender " + gender)

                #predict age
                age_net.setInput(blob)
                age_preds = age_net.forward()
                age = age_list[age_preds[0].argmax()]
                print("Age " + age)

                overlay_text = "%s %s" % (gender, age)
                cv2.putText(image, overlay_text, (x, y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow('frame',image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

if __name__ == "__main__":
    age_net, gender_net = load_caffe_models()
    video_detector(age_net, gender_net)
