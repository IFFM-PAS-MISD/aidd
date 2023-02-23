# ============================================
__title__ = 'Autoencoder-ConvLSTM model for Modelling guided waves in CFRP'
__author__ = "Abdalraheem A. Ijjeh"
__maintainer__ = "Abdalraheem A. Ijjeh"
__email__ = "aijjeh@imp.gda.pl"

# ============================================


import os
from IPython.display import Image, display
from tensorflow.keras.preprocessing.image import load_img
import PIL
from PIL import ImageOps
import re
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from keras.layers import MaxPooling3D, \
    Input, ConvLSTM2D, UpSampling2D, \
    MaxPooling2D, Conv2D, Concatenate, Conv3D, \
    Dropout, BatchNormalization, Add, MaxPool2D, Conv2DTranspose, MaxPool3D, Conv3DTranspose, UpSampling3D, \
    concatenate, MaxPooling3D, Bidirectional, TimeDistributed
from tensorflow.python.client import device_lib
import random
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from keras import backend as K
from keras.callbacks import Callback

import neptune.new as neptune

from tensorflow.keras.models import Model
from keras import backend as K
from sklearn.model_selection import train_test_split, KFold
from decouple import config
from sklearn.metrics import r2_score

access_token = config('NEPTUNE_API_TOKEN')
########################################################################################################################
run = neptune.init_run(project='abdalraheem.ijjeh/Guided-waves-modelling',
                       api_token=access_token,
                       tags=['Model 3 the latent space and skip connections to full wavefield']
                       )

params = {'batches': 1,
          'kernel_size': 3,
          'height': 512,
          'width': 512,
          'time_stamps': 12,
          'num_filters': 8,
          'filter_size': 3,
          'epochs': 500,
          'dropout': 0.2,
          'levels': 9,
          'learning_rate': 0.00014329,
          'patience_epochs': 100,
          'val_split': 0.08,
          'hidden_layer': 3,
          'decay:steps': 100000,
          'decay_rate': 0.96,
          }

h = params.get('height')
w = params.get('width')
img_size = (h, w)

run["model/parameters"] = params

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
config_ = tf.compat.v1.ConfigProto()
config_.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config_)

########################################################################################################################
#  Load dataset
########################################################################################################################
os.chdir('/home/aijjeh/Desktop/Phd_Projects/Modeling_guided_waves/Model_2_mapping_GT_to_LS/Predictions_model_2')
latent_space = np.load('latent_space.npy')
print(latent_space.shape)
skip_0 = np.load('skip_connection_0.npy', mmap_mode='r+')
skip_1 = np.load('skip_connection_1.npy', mmap_mode='r+')
skip_2 = np.load('skip_connection_2.npy', mmap_mode='r+')
skip_3 = np.load('skip_connection_3.npy', mmap_mode='r+')
skip_4 = np.load('skip_connection_4.npy', mmap_mode='r+')
skip_5 = np.load('skip_connection_5.npy', mmap_mode='r+')
skip_6 = np.load('skip_connection_6.npy', mmap_mode='r+')
skip_7 = np.load('skip_connection_7.npy', mmap_mode='r+')
skip_8 = np.load('skip_connection_8.npy', mmap_mode='r+')

print(skip_0.shape)
print(skip_1.shape)
print(skip_2.shape)
print(skip_3.shape)
print(skip_4.shape)
print(skip_5.shape)
print(skip_6.shape)
print(skip_7.shape)
print(skip_8.shape)

os.chdir('/home/aijjeh/Desktop/Phd_Projects/Modeling_guided_waves/Model_2_mapping_GT_to_LS')
gt_img_delamination = np.load('GT_model_3.npy')
print(gt_img_delamination.shape)


def normalize_batch(bn_input):
    return keras.layers.BatchNormalization()(bn_input)


def get_time_distributed(time_input, n_filters, f_size):
    x_ = TimeDistributed(Conv2D(filters=n_filters,
                                kernel_size=(f_size, f_size),
                                strides=1, padding='same',
                                activation='relu'))(time_input)
    return x_


def PSNR(y_true, y_pred):
    # cast the target images to integer
    y_true = y_true * 255.0
    y_true = tf.cast(y_true, tf.uint8)
    y_true = tf.clip_by_value(y_true, 0, 255)
    # cast the predicted images to integer
    y_pred = y_pred * 255.0
    y_pred = tf.cast(y_pred, tf.uint8)
    y_pred = tf.clip_by_value(y_pred, 0, 255)
    # return the psnr
    return tf.image.psnr(y_true, y_pred, max_val=255)


def get_model():
    input1 = Input(shape=(960,))
    input2 = Input(shape=(12, 1, 1, 72))
    input3 = Input(shape=(12, 2, 2, 64))
    input4 = Input(shape=(12, 4, 4, 56))
    input5 = Input(shape=(12, 8, 8, 48))
    input6 = Input(shape=(12, 16, 16, 40))
    input7 = Input(shape=(12, 32, 32, 32))
    input8 = Input(shape=(12, 64, 64, 24))
    input9 = Input(shape=(12, 128, 128, 16))
    input10 = Input(shape=(12, 256, 256, 8))

    # skip_connection = [input10, input9, input8, input7, input6, input5, input4, input3, input2]
    # print(skip_connection)
    x_layer = tf.keras.layers.Reshape((12, 1, 1, 80))(input1)
    factor = 8
    ####################################################################################################################
    # Decoder
    ####################################################################################################################
    x_layer = concatenate([x_layer, input2], axis=-1)
    x_layer = TimeDistributed(UpSampling2D((2, 2)))(x_layer)
    x_layer = keras.layers.Dropout(params.get('dropout'))(x_layer)  # adding dropout layer
    x_layer = get_time_distributed(x_layer, params.get('num_filters') * 9, 3)
    x_layer = get_time_distributed(x_layer, params.get('num_filters') * 9, 3)

    x_layer = concatenate([x_layer, input3], axis=-1)
    x_layer = TimeDistributed(UpSampling2D((2, 2)))(x_layer)
    x_layer = keras.layers.Dropout(params.get('dropout'))(x_layer)  # adding dropout layer
    x_layer = get_time_distributed(x_layer, params.get('num_filters') * 8, 3)
    x_layer = get_time_distributed(x_layer, params.get('num_filters') * 8, 3)

    x_layer = concatenate([x_layer, input4], axis=-1)
    x_layer = TimeDistributed(UpSampling2D((2, 2)))(x_layer)
    x_layer = keras.layers.Dropout(params.get('dropout'))(x_layer)  # adding dropout layer
    x_layer = get_time_distributed(x_layer, params.get('num_filters') * 7, 3)
    x_layer = get_time_distributed(x_layer, params.get('num_filters') * 7, 3)

    x_layer = concatenate([x_layer, input5], axis=-1)
    x_layer = TimeDistributed(UpSampling2D((2, 2)))(x_layer)
    x_layer = keras.layers.Dropout(params.get('dropout'))(x_layer)  # adding dropout layer
    x_layer = get_time_distributed(x_layer, params.get('num_filters') * 6, 3)
    x_layer = get_time_distributed(x_layer, params.get('num_filters') * 6, 3)

    x_layer = concatenate([x_layer, input6], axis=-1)
    x_layer = TimeDistributed(UpSampling2D((2, 2)))(x_layer)
    x_layer = keras.layers.Dropout(params.get('dropout'))(x_layer)  # adding dropout layer
    x_layer = get_time_distributed(x_layer, params.get('num_filters') * 5, 3)
    x_layer = get_time_distributed(x_layer, params.get('num_filters') * 5, 3)

    x_layer = concatenate([x_layer, input7], axis=-1)
    x_layer = TimeDistributed(UpSampling2D((2, 2)))(x_layer)
    x_layer = keras.layers.Dropout(params.get('dropout'))(x_layer)  # adding dropout layer
    x_layer = get_time_distributed(x_layer, params.get('num_filters') * 4, 3)
    x_layer = get_time_distributed(x_layer, params.get('num_filters') * 4, 3)

    x_layer = concatenate([x_layer, input8], axis=-1)
    x_layer = TimeDistributed(UpSampling2D((2, 2)))(x_layer)
    x_layer = keras.layers.Dropout(params.get('dropout'))(x_layer)  # adding dropout layer
    x_layer = get_time_distributed(x_layer, params.get('num_filters') * 3, 3)
    x_layer = get_time_distributed(x_layer, params.get('num_filters') * 3, 3)

    x_layer = concatenate([x_layer, input9], axis=-1)
    x_layer = TimeDistributed(UpSampling2D((2, 2)))(x_layer)
    x_layer = keras.layers.Dropout(params.get('dropout'))(x_layer)  # adding dropout layer
    x_layer = get_time_distributed(x_layer, params.get('num_filters') * 2, 3)
    x_layer = get_time_distributed(x_layer, params.get('num_filters') * 2, 3)

    x_layer = concatenate([x_layer, input10], axis=-1)
    x_layer = TimeDistributed(UpSampling2D((2, 2)))(x_layer)
    x_layer = keras.layers.Dropout(params.get('dropout'))(x_layer)  # adding dropout layer
    x_layer = get_time_distributed(x_layer, params.get('num_filters') * 1, 3)
    x_layer = get_time_distributed(x_layer, params.get('num_filters') * 1, 3)

    ####################################################################################################################
    x_layer = ConvLSTM2D(params.get('time_stamps'), 1, padding='same', return_sequences=True)(x_layer)
    ####################################################################################################################
    # Output layer
    output = get_time_distributed(x_layer, 1, 1)
    ####################################################################################################################
    model_ = Model(inputs=[input1,
                           input2,
                           input3,
                           input4,
                           input5,
                           input6,
                           input7,
                           input8,
                           input9,
                           input10], outputs=output)
    ####################################################################################################################
    model_.compile(optimizer='adam',
                   loss='mse',
                   metrics=[PSNR, tf.keras.metrics.RootMeanSquaredError()])
    return model_


keras.backend.clear_session()

model = get_model()

model.summary()

env_path = '/home/aijjeh/Desktop/Phd_Projects/Modeling_guided_waves/'

checkpoint_filepath = '/home/aijjeh/Desktop/Phd_Projects/Modeling_guided_waves/temp/checkpoint/model_3_reconstructing_latent_space.h5'

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


model.fit([latent_space,
           skip_8,
           skip_7,
           skip_6,
           skip_5,
           skip_4,
           skip_3,
           skip_2,
           skip_1,
           skip_0], gt_img_delamination,
          batch_size=1,
          epochs=params.get('epochs'),
          validation_split=0.2,
          callbacks=[MonitoringCallback(), callbacks])  #
