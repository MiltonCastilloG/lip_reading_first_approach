import math  

def calculate_distance(mouse_landmarks):
    landmarks_distances=[]
    for i in range(12):
        x_diference=mouse_landmarks[3][0]-mouse_landmarks[i][0]
        y_diference=mouse_landmarks[3][1]-mouse_landmarks[i][1]
        distance=math.sqrt(pow(x_diference,2)+pow(y_diference,2))
        landmarks_distances.append(distance)

    for i in range(12,20):
        x_diference=mouse_landmarks[14][0]-mouse_landmarks[i][0]
        y_diference=mouse_landmarks[14][1]-mouse_landmarks[i][1]
        distance=math.sqrt(pow(x_diference,2)+pow(y_diference,2))
        landmarks_distances.append(distance)

    for i in range(12):            
            x_diference_2=mouse_landmarks[0][0]-mouse_landmarks[i][0]
            y_diference_2=mouse_landmarks[0][1]-mouse_landmarks[i][1]
            distance=math.sqrt(pow(x_diference_2,2)+pow(y_diference_2,2))
            landmarks_distances.append(distance)
    
    for i in range(12,20):
            x_diference_2=mouse_landmarks[19][0]-mouse_landmarks[i][0]
            y_diference_2=mouse_landmarks[19][1]-mouse_landmarks[i][1]
            distance=math.sqrt(pow(x_diference_2,2)+pow(y_diference_2,2))
            landmarks_distances.append(distance)
            
    return landmarks_distances