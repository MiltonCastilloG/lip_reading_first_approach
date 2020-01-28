#File that gets all the tuples (x,y) of the mouth keypoints
#using shape_predictor_68_face_landmarks.dat using as 
#auxiliar function get_mouth_landmarks
import os
from get_mouth_landmarks import get_mouth_landmarks
import json

data = {
  "a": [],
  "e": [],
  "i": [],
  "o": [],
  "u": [],
  "si": [],
  "no": []
}
mapped_data = {
  "a": [],
  "e": [],
  "i": [],
  "o": [],
  "u": [],
  "si": [],
  "no": []
}
shape_predictor_src = "shape_predictor_68_face_landmarks.dat"
width_size = 500

for folder_name in os.listdir('/Users/josepaniagua/Desktop/NewDataSet'):
    image_folder_src = '/Users/josepaniagua/Desktop/NewDataSet/{}'.format(folder_name)
    for img in os.listdir(image_folder_src):        
        mouth_landmarks = get_mouth_landmarks(shape_predictor_src, width_size, img, image_folder_src + "/{}".format(img) )
        data[folder_name].append(mouth_landmarks)
        mapped_pos = []
        for pos in mouth_landmarks:            
            mapped_pos.append(pos[1]*100 + pos[0])
        mapped_data[folder_name].append(mapped_pos)

with open('array_data.json', 'w') as outfile:
    json.dump(data, outfile, sort_keys=True, indent=2)

with open('mapped_data.json', 'w') as outfile:
    json.dump(mapped_data, outfile)
print("done")
