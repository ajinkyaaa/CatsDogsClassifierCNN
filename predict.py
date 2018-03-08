#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 20:04:09 2017

@author: ajinkya.parkar@ibm.com
"""

""" This code demonstrates reading the test data and writing 
predictions to an output file.
It should be run from the command line, with one argument:
$ python predict_starter.py [test_file]
where test_file is a .npy file with an identical format to those 
produced by extract_cats_dogs.py for training and validation.
(To test this script, you can use one of those).
This script will create an output file in the same directory 
where it's run, called "predictions.txt".
"""

import sys
import numpy as np
import random
import os
import numpy as np
from PIL import Image
from keras.models import model_from_json

CAT_OUTPUT_LABEL = 1
DOG_OUTPUT_LABEL = 0

TEST_FILE = sys.argv[1] 

data = np.load(TEST_FILE).item()

# these are images in exactly the same format
# as your train and validation set
images = data["images"]

# the testing data also contains a unique id
# for each testing image
ids = []

# This file will be created if it does not exist
# and overwritten if it does
OUT_FILE = "predictions.txt"

# make a prediction on each image
# and write output to disk
out = open(OUT_FILE, "w")

if "ids" in data:
    ids = data["ids"]
else:
    #if it's not contained, a sequence is used
    ids = list(range(0,len(images)))

json_file = open('TrainedModel.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("TrainedModel.h5")

loaded_model.compile(loss='binary_crossentropy',
              #optimizer='rmsprop',
              optimizer='adam',
              metrics=['accuracy'])


pred = loaded_model.predict(images, batch_size = 20, verbose=1)
counter = 0
for i, image in enumerate(images):

  image_id = ids[i]
  if data["labels"][i] == pred[i]:
        counter+=1
      #print(counter)
  # here, we'll create a "random" prediction
  # to demonsate the format of the output file
  # this should be "1" for Cat and "0" for dog.
 
  prediction = round(float(pred[i]))
  line = str(image_id) + " " + str(prediction) + "\n"
  out.write(line)


out.close()