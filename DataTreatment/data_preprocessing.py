#Code that aims to return the distance between the two central (internal and external)
#points of the mouth, this result is written to disk in json format with name 
# "train_set.json"

import json
import math  

with open('array_data.json') as json_file:
    data = json.load(json_file)

y_data=[]
x_data=[]

for letter_index, letter in enumerate(data):           
    for mouse in data[letter]:
        output=[0,0,0,0,0]                        
        landmarks_distances=[]
        for i in range(12):
            x_diference=mouse[3][0]-mouse[i][0]
            y_diference=mouse[3][1]-mouse[i][1]
            distance=math.sqrt(pow(x_diference,2)+pow(y_diference,2))
            landmarks_distances.append(distance)

        for i in range(12,20):
            x_diference=mouse[14][0]-mouse[i][0]
            y_diference=mouse[14][1]-mouse[i][1]
            distance=math.sqrt(pow(x_diference,2)+pow(y_diference,2))
            landmarks_distances.append(distance)
        x_data.append(landmarks_distances)
        output[letter_index]=1
        y_data.append(output)

data_set=[
    x_data,
    y_data
]

with open('train_set.json', 'w') as outfile:
    json.dump(data_set, outfile, sort_keys=True, indent=2)       