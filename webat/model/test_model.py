from __future__ import print_function
import matplotlib.pyplot as plt 
import cv2
import os
import numpy as np 
import csv
import importlib
from itertools import zip_longest

import tkinter as tk
from tkinter import filedialog
from tkinter import *

import tensorflow as tf
from tensorflow import keras

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers import AveragePooling2D, Conv2DTranspose, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization

from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from sklearn.metrics import classification_report

from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from keras.utils.np_utils import to_categorical
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.utils import plot_model


test_folder = os.getcwd() + '/unified_images/test'
pretrained_model_name = 'VGG16'

# 'preprocess_input' function varies based on the backbone model type
if pretrained_model_name.startswith('VGG16'):
    backbone = getattr(importlib.import_module("tensorflow.keras.applications.vgg16"), pretrained_model_name)
    preprocess_input = getattr(importlib.import_module("tensorflow.keras.applications.vgg16"), 'preprocess_input')

elif pretrained_model_name.startswith('ResNet50'):
    backbone = getattr(importlib.import_module("tensorflow.keras.applications.resnet50"), pretrained_model_name)
    preprocess_input = getattr(importlib.import_module("tensorflow.keras.applications.resnet50"), 'preprocess_input')

elif pretrained_model_name.startswith('InceptionV3'):
    backbone = getattr(importlib.import_module("tensorflow.keras.applications.inception_v3"), pretrained_model_name)
    preprocess_input = getattr(importlib.import_module("tensorflow.keras.applications.inception_v3"), 'preprocess_input')

elif pretrained_model_name.startswith('Xception'):
    backbone = getattr(importlib.import_module("tensorflow.keras.applications.xception"), pretrained_model_name)
    preprocess_input = getattr(importlib.import_module("tensorflow.keras.applications.xception"), 'preprocess_input')

else:
    print("Please check the pretrained model name...")
    raise

# Create dataset from image directories
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# Automatically one hot encoding for categorical labels
test_image_gen = test_datagen.flow_from_directory(test_folder,
                                                target_size = (50, 50),
                                                batch_size = 64,
                                                class_mode = 'categorical',
                                                shuffle = False,
                                                seed = 42) # not to be shuffled to compare with various metrics

class_labels = list(test_image_gen.class_indices.keys())

# Load ML model to test
model_name = 'vgg16_params_v1_best'
model_extension = '.hdf5'
## Loading the machine learning model
model = keras.models.load_model(model_name + model_extension)
print("Loaded model..")
plot_model(model, to_file="best_model_final.png", show_shapes=True)

# Get the ground truth labels of test data
true_classes = test_image_gen.classes

# Predict the class with the highest probability of data
prediction_classes = np.argmax(model.predict(test_image_gen), axis=-1)

# Confusion matrix
cm = confusion_matrix(true_classes, prediction_classes, labels = [0, 1, 2, 3])
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = class_labels)
disp.plot()
plt.savefig(model_name + '_cm', format='png')

# Classification report
print(classification_report(true_classes, prediction_classes, target_names=class_labels))

score = model.evaluate(test_image_gen)
print("score: ", score)