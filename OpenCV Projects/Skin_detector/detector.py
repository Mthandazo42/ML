"""
NAME: SKIN DETECTOR
DESCRIPTION: SIMPLE PYTHON SCRIPT THAT DETECTS SKIN, IT CAN BE USED TO DETECT NUDITY IN IMAGES
DATE: 27/12/2018
AUTHOR: MTHANDAZO NDHLOVU

"""
#inports
import cv2
import imutils
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
args = vars(ap.parse_args())

#define the upper and lower boundaries of the HSV pixel
#intensities to be considered as skin
lower = np.array([0, 48, 80], dtype = "uint8")
upper = np.array([20, 255, 255], dtype = "uint8")

if not args.get("video", False):
    camera = cv2.VideoCapture(0)
else:
    camera = cv2.VideoCapture(args["video"])

while True:
    (grabbed, frame) = camera.read()
    if args.get("video") and not grabbed:
        break
    #resize the frame and convert it to the HSV color space and determine
    #the HSV pixel intensities that fall into and the specified upper and lower boundaries
    frame = imutils.resize(frame, width =  400)
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    skinMask = cv2.inRange(converted, lower, upper)
    #apply a series of erosions and dilations to the mask using eliptical kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skinMask = cv2.erode(skinMask, kernel, iterations = 2)
    skinMask = cv2.dilate(skinMask, kernel, iterations = 2)

    #blur the mask to help remove noise and the apply the mask to the frame
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
    skin = cv2.bitwise_and(frame, frame, mask = skinMask)

    #show the skin in the image along with the mask
    cv2.imshow("images", np.hstack([frame, skin]))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#doing some cleaning
camera.release()
cv2.destroyAllWindows()
