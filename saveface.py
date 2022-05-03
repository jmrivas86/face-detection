import os

import cv2

person_name = 'jmrivas'

if not os.path.exists(f'data_{person_name}'):
    print(f'Carpeta creada: {person_name}')
    os.makedirs(f'data_{person_name}')

# To capture video from webcam.
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
count = 0
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aux_frame = frame.copy()
    faces = faceClassif.detectMultiScale(gray, 1.3, 5)
    k = cv2.waitKey(1)
    if k == 27:
        break
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (128, 0, 255), 2)
        rostro = aux_frame[y:y + h, x:x + w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
        if k == ord('s'):
            cv2.imwrite(f'data_{person_name}/rostro_{count}.jpg', rostro)
            cv2.imshow('rostro', rostro)
            count = count + 1
    cv2.rectangle(frame, (10, 5), (450, 25), (255, 255, 255), -1)
    cv2.putText(
        frame,
        'Presione s, para almacenar los rostros encontrados',
        (10, 20), 2, 0.5, (128, 0, 255), 1,
        cv2.LINE_AA
    )
    cv2.imshow('frame', frame)

    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
