from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
from model.smallervggnet import SmallerVGGNet
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os
import glob
import sys
import sklearn.metrics as metrics
from tensorflow.keras.metrics import *


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True)
ap.add_argument("-m", "--model", type = str, default = "gender.model")
args = ap.parse_args()

#Параметры
epochs = 100
lr = 1e-3
batch_size = 64
img_dims = (96,96,3)

data = []
labels = []

#Загрузка изображений
image_files = [f for f in glob.glob(args.dataset + "/**/*", recursive=True) if not os.path.isdir(f)] 
random.seed(42)
random.shuffle(image_files)

#Создание групп
for img in image_files:
	image = cv2.imread(img)
	image = cv2.resize(image, (img_dims[0],img_dims[1]))
	image = img_to_array(image)
	data.append(image)
	
	label = img.split(os.path.sep)[-2]
	if label == "woman":
		label = 1
	else:
		label = 0
	labels.append([label])

data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
#Разделение данных на тренировочные и валидационные (8 к 2)
trainX, testX, trainY, testY = train_test_split(data, labels, test_size = 0.2,
                                                  random_state = 42)
trainY = to_categorical(trainY, num_classes = 2)
testY = to_categorical(testY, num_classes = 2)

aug = ImageDataGenerator(rotation_range = 25, width_shift_range = 0.1,
                         height_shift_range = 0.1, shear_range = 0.2, zoom_range = 0.2,
                         horizontal_flip = True, fill_mode = "nearest")

#Построение модели
model = SmallerVGGNet.build(width = img_dims[0], height = img_dims[1], depth = img_dims[2],
                            classes = 2)

#Компиляция модели с использованием оптимизатора Adam
opt = Adam(learning_rate = lr, decay = lr / epochs)
model.compile(loss = "binary_crossentropy", optimizer = opt, metrics = ["accuracy"])

#Тренировка модели
H = model.fit(aug.flow(trainX, trainY, batch_size = batch_size),
                        validation_data = (testX, testY),
                        steps_per_epoc = len(trainX) // batch_size,
                        epochs = epochs, verbose = 1)

#Сохранение модели
model.save(args.model)
