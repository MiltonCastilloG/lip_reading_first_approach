from mouth_landmarks import get_mouth_landmarks
from landmarks_distance_calculation import calculate_distance
import numpy as np 

letters=["a","e","i","o","u"]

#Cargado del modelo
from keras.models import model_from_json

json_file = open("network_clase.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# Cargar los pesos (weights) en un nuevo modelo
loaded_model.load_weights("network_weights_clase.h5")
print("Modelo cargado desde el disco")

#Preprocesamiento de los datos
image_src="./a.jpg"
width_size=500

mouth_landmarks=get_mouth_landmarks(width_size,image_src)
# print(mouth_landmarks[0])
mouth_landmarks_distances=[calculate_distance(mouth_landmarks)]
mouth_landmarks_distances=np.array(mouth_landmarks_distances)
# # Predecir sobre la imagen
prediction = loaded_model.predict_classes(mouth_landmarks_distances)
conclution = prediction[0]
print("The prediction is letter: ",letters[conclution])