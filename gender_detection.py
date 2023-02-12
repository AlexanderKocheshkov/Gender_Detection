import numpy as np
import argparse
import cv2
import os
import cvlib as cv
from tensorflow.keras.utils import get_file
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import face_recognition

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = ap.parse_args()

#Загружаем изображение
image = face_recognition.load_image_file(args.image)

#Подключаем модель
model = load_model('gender.model')

classes = ["man", "woman"]

#Ищем лица на изображении и переходим в другой цветовой диапазон
face_locations = face_recognition.face_locations(image)
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

for face in face_locations:
    # Определяем границы лиц
    top, right, bottom, left = face
    cv2.rectangle(image, (left,bottom), (right,top), (0,255,0), 2)
    for idx, f in enumerate(face):
        face_image = image[top:bottom, left:right]
        face_crop = np.copy(face_image)
        #Подготавливаем изображение для работы с моделью
        face_crop = cv2.resize(face_crop, (96,96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)

        conf = model.predict(face_crop)[0]
        idx = np.argmax(conf)
        #Накладываем на изображение рамку, предсказание и вероятность
        label = classes[idx]
        label = "{}: {:.2f}%".format(label, conf[idx] * 100)
        Y = top - 10 if top - 10 > 10 else top + 20

        cv2.putText(image, label, (left, Y),  cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)


cv2.imshow("gender detection", image)          
cv2.waitKey()
cv2.destroyAllWindows()
