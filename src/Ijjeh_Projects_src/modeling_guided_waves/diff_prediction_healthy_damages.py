#!/usr/bin/env python
# coding: utf-8


# ============================================
__title__ = 'Modelling guided waves'
__author__ = "Abdalraheem A. Ijjeh"
__maintainer__ = "Abdalraheem A. Ijjeh"
__email__ = "aijjeh@imp.gda.pl"

# ============================================

import os
from abc import ABC

import keras.losses
import numpy as np
import json
import csv
import tensorflow as tf
import neptune.new as neptune
import matplotlib.pyplot as plt
import PIL
import re
import scipy.fftpack as spfft

from IPython.display import Image, display
from tensorflow.keras.preprocessing.image import load_img
from PIL import ImageOps
from tensorflow.keras.models import Model
from keras import backend as K
from keras.callbacks import Callback
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error
from decouple import config
from dotenv import load_dotenv

load_dotenv()

# # Link to Neptune
access_token = os.getenv('NEPTUNE_API_TOKEN')
run = neptune.init_run(project='abdalraheem.ijjeh/Guided-waves-modelling',
                       api_token=access_token,
                       tags=['diff prediction healthy vs damaged']
                       )

# # Hyperparameters

run["LR_input_based"] = "diff prediction healthy vs damaged"
params = {'samples': 15200,
          'n_size': 194560,
          'normalised': True,
          'batches': 16,
          'num_filters': 16,
          'kernel_size': 3,
          'shape': (512, 32, 32, 1),
          'epochs': 1000,
          'dropout': 0.2,
          'levels': 3,
          'learning_rate': 0.00014329,
          'patience_epochs': 100,
          'val_split': 0.08,
          'hidden_layer': 3,
          'decay:steps': 100000,
          'decay_rate': 0.96,
          'time_stamps': 16
          }

run["model/parameters"] = params

# # Run on GPU


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

# # Save the hyperparameters


env_path = '/home/aijjeh/Desktop/Phd_Projects/Modeling_guided_waves/'
os.chdir(env_path)
json = json.dumps(params)
f = open('hyper_par_VAE_diff_prediction.json', 'w')
f.write(json)
f.close()
checkpoint_filepath = env_path + 'temp/checkpoint/encoder_ann_%s_%d_lr_filters_%d_levels_%d_batches_%d_epochs_%d_dropout_%s_val_split_%s_hidden_%d.h5' % (
    params['samples'],
    params['learning_rate'],
    params['num_filters'],
    params['levels'],
    params['batches'],
    params['epochs'],
    params['dropout'],
    params['val_split'],
    params['hidden_layer'])

callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                              patience=params.get('patience_epochs'),
                                              mode='min'),
             tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                monitor='val_loss',
                                                save_best_only=True)]


class MonitoringCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        for metric_name, metric_value in logs.items():
            run[metric_name].log(metric_value)


def load_dataset():

    os.chdir(env_path + 'num/')
    pred_diff_arr = np.load('diff_healthy_damage_prediction_num.npy')
    pred_diff_arr = pred_diff_arr.reshape((475, 512, 32, 32, 1))

    os.chdir('/home/aijjeh/Desktop/Phd_Projects/Modeling_guided_waves/dataset')
    # delam_coords = np.load('LR_GT_del.npy')
    # delam_coords = delam_coords.reshape((475, 1, 5, 1))
    # delam_coords = np.repeat(delam_coords, 512, axis=1)
    # delam_coords = delam_coords.reshape((475 * 512, 5, 1))
    # delam_coords = np.repeat(delam_coords, 32, axis=1)
    # delam_coords = np.repeat(delam_coords, 32, axis=2)

    # delam_GT = np.load('LR_GT_labels_img.npy')
    # delam_GT_img = delam_GT.reshape((475, 1, 32, 32))
    # delam_GT_img = np.repeat(delam_GT_img, 512, axis=1)
    # delam_GT_img = delam_GT_img.reshape((475, 512, 32, 32, 1))
    # # delam_GT_img = np.transpose(delam_GT_img, (0, -1, 1))
    # # delam_GT_img = np.invert(delam_GT_img)
    # print(delam_GT_img.shape)

    x_lr_frame = np.load('LR_ref_frames.npy')
    x_lr_frame = np.reshape(x_lr_frame, (475, 512, 32, 32, 1))
    print(x_lr_frame.shape)

    Y_train = np.load('LR_labels.npy')
    Y_train = np.reshape(Y_train, (475, 512, 32, 32, 1))

    X_train = np.add(x_lr_frame, pred_diff_arr)
    # Y_train = np.subtract(x_lr_frame, Y_train_)  # difference between healthy and damaged
    # fig, ax = plt.subplots(2)
    # ax[0].imshow(delam_GT_img[0, 80])
    # ax[1].imshow(Y_train[0, 80])
    # plt.show()
    # exit()
    Train_x, Val_x_samples, Train_label, Val_y_samples = train_test_split(X_train, Y_train,
                                                                          train_size=0.80,
                                                                          shuffle=False,
                                                                          random_state=1988)

    return Train_x, Train_label


train_x, train_label = load_dataset()
print(train_x.shape)
print(train_label.shape)


def PSNR(y_true, y_pred):
    # # cast the target images to integer
    # y_true = y_true * 255.0
    # y_true = tf.cast(y_true, tf.uint8)
    # y_true = tf.clip_by_value(y_true, 0, 1)
    # # cast the predicted images to integer
    # y_pred = y_pred * 255.0
    # y_pred = tf.cast(y_pred, tf.uint8)
    # y_pred = tf.clip_by_value(y_pred, 0, 255)
    # # return the psnr
    return tf.image.psnr(y_true, y_pred, max_val=1)


def build_model():
    inputs = tf.keras.Input(shape=params.get('shape'))
    x = tf.keras.layers.ConvLSTM2D(4, 3, 1, padding='same', return_sequences=True, activation='relu')(inputs)
    # x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ConvLSTM2D(4, 3, 1, padding='same', return_sequences=True, activation='relu')(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ConvLSTM2D(4, 3, 1, padding='same', return_sequences=True, activation='relu')(x)
    x = tf.keras.layers.ConvLSTM2D(4, 3, 1, padding='same', return_sequences=True, activation='relu')(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.Flatten()(inputs)
    #
    # x = tf.keras.layers.Dense(x.shape[-1], activation='relu')(x)
    # x = tf.keras.layers.Dense(x.shape[-1] ** 2, activation='relu')(x)
    # x = tf.keras.layers.Dense(x.shape[-1] ** 2, activation='relu')(x)
    #
    # x = tf.keras.layers.Reshape((25, 25, 1))(x)
    # x = tf.keras.layers.UpSampling2D(4)(x)
    # x = tf.keras.layers.Conv2D(1, 5, strides=1, padding='valid')(x)
    # x = tf.keras.layers.AveragePooling2D((3, 3))(x)
    output = tf.keras.layers.ConvLSTM2D(1, 1, padding='same', activation='sigmoid', return_sequences=True)(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1.5e-4),
                  loss='mse',
                  metrics=[PSNR])
    return model


model_ = build_model()
model_.summary()

model_.fit(train_x, train_label,
           validation_split=params.get('val_split'),
           batch_size=params.get('batches'),
           epochs=params.get('epochs'),
           callbacks=[callbacks, MonitoringCallback()])  #
