import os

import cv2
import imutils

person_name = 'jmrivas'

if not os.path.exists(f'data/{person_name}'):
    print(f'Carpeta creada: {person_name}')
    os.makedirs(f'data/{person_name}')

# To capture video from webcam.
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
count = 0
while True:
    ret, frame = cap.read()
    if not ret: break
    frame = imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aux_frame = frame.copy()

    faces = faceClassif.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        rostro = aux_frame[y:y + h, x:x + w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(f'data/{person_name}/rostro_{count}.jpg', rostro)
        count = count + 1

    cv2.imshow('frame', frame)

    # Stop if escape key is pressed
    k = cv2.waitKey(1)
    if k == 27 or count >= 300:
        break

cap.release()
cv2.destroyAllWindows()
