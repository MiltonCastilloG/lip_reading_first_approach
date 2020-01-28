#Code that aims to reseize the images and return only the mouth keypoints 
from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
from pprint import pprint


def get_mouth_landmarks(shape_predictor_src, width_size, name_img, image_src):
    # initialize dlib's face detector 
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
    #IMPORTANT THE NEW RECTANGLE IS THE ONE WITH THE "STANDARIZED" DATA
    #SO IS IMPORTANT TO MAKE THE CALCULOUS OVER IT AND NO OVER THE 
    #SIMPLE "RECT" OBJECT RETURNED BY THE LIBRARY
    rect2 = dlib.rectangle(0, 0, len(cutted_image[0]), len(cutted_image))    
    shape = predictor(cutted_image, rect2)
    #Finlly, lets remove all the "noise" data from the rest of the face 
    #keypoints
    shape = face_utils.shape_to_np(shape)[48:]
    
    return shape.tolist()
