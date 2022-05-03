import os

import cv2
import numpy as np

data_path = f'data'

people_list = os.listdir(data_path)
print('Lista de personas: ', people_list)
labels = []
faces_data = []
label = 0
for name_dir in people_list:
    person_path = data_path + '/' + name_dir
    print('Leyendo las imágenes')
    for file_name in os.listdir(person_path):
        print('Rostros: ', name_dir + '/' + file_name)
        labels.append(label)
        faces_data.append(cv2.imread(person_path + '/' + file_name, 0))
        # image = cv2.imread(personPath+'/'+fileName,0)
        # cv2.imshow('image',image)
        # cv2.waitKey(10)
    label = label + 1
print('labels= ', labels)
print('Número de etiquetas 0: ', np.count_nonzero(np.array(labels) == 0))
print('Número de etiquetas 1: ', np.count_nonzero(np.array(labels) == 1))
# Entrenar el reconocedor
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Entrenando el reconocedor de rostros
print("Entrenando...")
face_recognizer.train(faces_data, np.array(labels))
face_recognizer.write('modeloLBPHFace.xml')
print("Modelo almacenado...")
