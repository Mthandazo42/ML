"""
NAME: OBJECT DETECTION USING PYTHON AND OPEN CV
AUTHOR: MTHANDAZO NDHLOVU
DESCRIPTION: SIMPLE SCRIPT TO SHOW HOW OPEN CV CAN BE USED TO MODEL AN OBJECT DETECTION ON A VIDEO
DATE: 26/12/2018
EMAIL: mthandazondhlovu34@gmail.com
"""

#imports
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2

#constructing an argument parser to accept arguments straight from the command line
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True, help="path to Caffe deploy prototxt file")
ap.add_argument("-m", "--model", required=True, help="path to Caffe pre-trained file")
ap.add_argument("-c", "--confidence", type=float, default=0.2, help="minimum probability to filter weak predictions")
ap.add_argument("-v", "--video", required=True, help="path to the video file")
args = vars(ap.parse_args())

#classes present in our training data
CLASSES =  ["background", "aeroplane","bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person","pottedplant", "sheep",
        "sofa", "train", "tvmonitor"]
#RANDOM COLORS CORRESPONDING TO THE ABOVE CLASSES
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

#load the serial model from the disk
print("INFO[] LOADING THE MODEL")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

#initialize the video stream, allow the camera sensor to warmup
#and initialize the FPS counter
print("INFO[]: STARTING THE VIDEO STREAM")
vs = VideoStream(src=args["video"]).start()
time.sleep(2.0)
fps = FPS().start()

#loop over the frames from the video stream
while True:
    #grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    #grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    #pass the blob through the network and obtain the detections and predictions
    net.setInput(blob)
    detections = net.forward()
    #loop over all the detections
    for i in np.arange(0, detections.shape[2]):
        #extract the confidence associated with each prediction
        confidence = detections[0, 0, i, 2]
        #filter out weak detections by ensuring that the confidence is above the minimum parse as an argument
        if confidence > args["confidence"]:
            #extract the index of class label from the detections, then compute the (x, y)-coordinates of the
            #bounding box for the object
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            #draw the prediction on the frame
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
            #show the output frame
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            #check if the pressed key is q
            if key == ord("q"):
                break
            #update the fps counter
            fps.update()
#stop the timer and display FPS information
fps.stop()
print("INFO[]: elasped time: {:.2f}".format(fps.elapsed()))
print("INFO[] approx FPS: {:.2f}".format(fps.fps()))

#the clean up
cv2.destroyAllWindows()
vs.stop()


