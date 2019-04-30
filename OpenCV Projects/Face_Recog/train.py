import cv2
import numpy as np
from PIL import Image
import os

path = "datasets"
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
#function to get the image and the label data
def getImagesAndLabel(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    face_samples = []
    ids = []
    for imagePath in imagePaths:
        if imagePath == 'datasets/.DS_Store':
            print("found something bad")
        else:
            PIL_img = Image.open(imagePath).convert('L')
            img_numpy = np.array(PIL_img, 'uint8')
            id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces = detector.detectMultiScale(img_numpy)
            for (x, y, w, h) in faces:
                face_samples.append(img_numpy[y:y+h, x:x+w])
                ids.append(id)
    return face_samples, ids
print("\n Training the model using the the faces provided")
faces, ids = getImagesAndLabel(path)
recognizer.train(faces, np.array(ids))
#save the model into the trainer/trainer.yml
recognizer.save("trainer/trainer.yml")
print("\n model trained on {0} faces".format(len(np.unique(ids))))
