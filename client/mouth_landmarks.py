# import the necessary packages
from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2

shape_predictor_src = "shape_predictor_68_face_landmarks.dat"

def get_mouth_landmarks(width_size, image_src):
    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shape_predictor_src)

    # load the input image, resize it, and convert it to grayscale
    image = cv2.imread(image_src)
    image = imutils.resize(image, width=width_size)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale image
    rects = detector(gray, 1)
    rect = rects[0]
    cutted_image = gray[rect.top():rect.bottom(), rect.left():rect.right()]
    cutted_image = imutils.resize(cutted_image, height=100)
    rect2 = dlib.rectangle(0, 0, len(cutted_image[0]), len(cutted_image))
    # REMEMBER TO CREATE A RECTANGLE OBJECT FOR THIS TO WORK
    shape = predictor(cutted_image, rect2)
    shape = face_utils.shape_to_np(shape)[48:]
    
    return shape.tolist()
    # loop over the face detections
    # for (i, rect) in enumerate(rects):
    # 	# determine the facial landmarks for the face region, then
    # 	# convert the facial landmark (x, y)-coordinates to a NumPy
    # 	# array

# construct the argument parser and parse the arguments
# shape_predictor_src = "shape_predictor_68_face_landmarks.dat"
# image_src = "images/example_03.jpg"
# get_mouth_landmarks(shape_predictor_src,image_src)
