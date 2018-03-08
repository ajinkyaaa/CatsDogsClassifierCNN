#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 16:03:55 2017

@author: ajinkya.parkar@ibm.com
"""

"""One-time script for extracting all the cat and dog images from CIFAR-10,
and creating training and validation sets.

Before running, download the CIFAR-10 data using these commands:

$ curl -O http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
$ tar xvf cifar-10-python.tar.gz

"""

import numpy as np
from PIL import Image
from keras.models import model_from_json

TRAIN_PATH = "train.npy"
VAL_PATH = "validation.npy"

CAT_OUTPUT_LABEL = 1
DOG_OUTPUT_LABEL = 0

def load(npy_file):
  data = np.load(npy_file).item()
  return data['images'], data['labels']



np.random.seed(0)

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))

model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              #optimizer='rmsprop',
              optimizer='adam',
              metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling

train_generator = train_datagen.flow(train_images,train_labels,batch_size=50)

# fits the model on batches with real-time data augmentation:
model.fit_generator(train_generator,
                    steps_per_epoch=len(train_images) / 50, epochs=50)
model.save_weights('TrainedModel.h5')
model_json = model.to_json()
with open("TrainedModel.json", "w") as json_file:
    json_file.write(model_json)