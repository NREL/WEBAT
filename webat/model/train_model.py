import datetime
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np
import csv
import os

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import layers, models
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import SGD, Adam

from tensorflow.keras import preprocessing
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras_preprocessing.image import img_to_array, load_img

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, TensorBoard
import keras_tuner as kt
from tensorflow.keras.utils import plot_model
from utils_ml import *

from livelossplot import PlotLossesKeras
from livelossplot.outputs import MatplotlibPlot

def train():
    # For reproducibility
    np.random.seed(42)

    # General settings for params
    params = {'input_shape': (50, 50, 3),
              'n_classes': 4,
              'batch_size': 64,
              'epochs': 30,
              'pretrained_model': 'VGG16'}      # pretrained model: VGG16, ResNet50, InceptionV3, Xception, ...

    pretrained_model = 'vgg16'
    model_name = pretrained_model + '_best_test'

    start = datetime.datetime.now()

    # Create a dataset from image directories
    train_folder = os.getcwd() + '/unified_images/train'
    val_folder = os.getcwd() + '/unified_images/val'
    train_generator, val_generator, backbone = build_dataset(train_folder, val_folder, params)

    steps_per_epoch = train_generator.n // train_generator.batch_size      # STEP_SIZE_TRAIN
    val_steps_per_epoch = val_generator.n // val_generator.batch_size       # STEP_SIZE_VAL

    compile_params = {'step_size_train': steps_per_epoch,
                      'step_size_val': val_steps_per_epoch}
    
    # Train the model with the best hyperparameters found - take one option from the below.
    # best_hps = tune_hyperparameters()     # Find the best hyperparameters now
    best_hps = {'finetune': 5, 'kernel_initializer': 'normal', 'learning_rate': 0.0001, 'decay_rate': 0.63, 'optimizer': 'Adam', 'dropout': 0, 'units_1': 50, 'units_2': 30}
    model = create_model_opt(best_hps, backbone, params, compile_params)

    # Learning schedule callback
    # lr_rate = LearningRateScheduler(lr_schedule)

    # Model Checkpoint callback - save best weights
    checkpoint = ModelCheckpoint(filepath = model_name + '_best.hdf5',
                                 save_best_only = True,
                                 verbose = 1)
    
    # Early Stopping
    es = EarlyStopping(monitor='val_loss', mode='min', patience=5, restore_best_weights=True)

    # Live plot
    live_plot = PlotLossesKeras(outputs=[MatplotlibPlot(figpath=model_name+'_plot.png')])    

    # Training session
    results = model.fit(train_generator,
                        epochs = params['epochs'],
                        steps_per_epoch = compile_params['step_size_train'],
                        validation_data = val_generator,
                        validation_steps = compile_params['step_size_val'],
                        batch_size = params['batch_size'],
                        callbacks = [checkpoint, es, live_plot],
                        verbose = 2)        # verbose 2: one line per epoch

    # Save history as CSV file
    acc = results.history['accuracy']
    val_acc = results.history['val_accuracy']

    loss = results.history['loss']
    val_loss = results.history['val_loss']

    d = [acc, val_acc, loss, val_loss]
    export_data = zip_longest(*d, fillvalue='')
    with open(model_name + '_results.csv', 'w', encoding="ISO-8859-1", newline='') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(('accuracy', 'val_accuracy', 'loss', 'val_loss'))
        wr.writerows(export_data)
    myfile.close()

    end = datetime.datetime.now()
    elapsed = end - start
    print('Time: ', elapsed)




if __name__ == "__main__":
    train()