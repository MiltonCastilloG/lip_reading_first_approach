#Cargado del modelo propio
from keras.models import model_from_json

json_file = open("network_clase.json", "r")
vowels_model_json = json_file.read()
json_file.close()
vowels_model_json = model_from_json(vowels_model_json)

# Cargar los pesos (weights) en un nuevo modelo
vowels_model_json.load_weights("network_weights_clase.h5")
print("Modelo propio cargado desde el disco")

# import the necessary packages
from imutils.video import VideoStream
from imutils import face_utils
import datetime
import argparse
import imutils
import time
import dlib
import cv2
import numpy as np

from landmarks_distance_calculation import calculate_distance

letters=["a","e","i","o","u"]

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] camera sensor warming up...")
vs = VideoStream(False).start()
time.sleep(2.0)


while True:	
	frame = vs.read()
	frame = imutils.resize(frame, width=400)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale frame
	rects = detector(gray, 0)
	rect=rects[0]

	#drwaing mouth points 
	shape2 = predictor(gray, rect)
	shape2 = face_utils.shape_to_np(shape2)[48:]
	for (x, y) in shape2:
		cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)		
	
	cutted_image = gray[rect.top():rect.bottom(), rect.left():rect.right()]
	cutted_image = imutils.resize(cutted_image, height=100)
	rect2 = dlib.rectangle(0, 0, len(cutted_image[0]), len(cutted_image))
    # REMEMBER TO CREATE A RECTANGLE OBJECT FOR THIS TO WORK
	shape = predictor(cutted_image, rect2)
	shape = face_utils.shape_to_np(shape)[48:]	

	#convert to list the shape 
	mouth_landmarks = shape.tolist()	
	
	#calculate point distances
	mouth_landmarks_distances=[calculate_distance(mouth_landmarks)]
	#convert to np array for input in neuronal network
	mouth_landmarks_distances=np.array(mouth_landmarks_distances)
	# # predict over image	
	prediction = vowels_model_json.predict_classes(mouth_landmarks_distances)
	conclution = prediction[0]		
	# cv2.putText(frame, 'hello', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
	print("The prediction is letter: ",letters[conclution])

	# # show the frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()