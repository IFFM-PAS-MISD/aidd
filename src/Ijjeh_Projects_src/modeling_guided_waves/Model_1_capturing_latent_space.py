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
                       tags=['capturing latent space with autoencoder']
                       )

params = {'batches': 1,
          'kernel_size': 3,
          'height': 256,
          'width': 256,
          'time_stamps': 32,
          'num_filters': 16,
          'filter_size': 3,
          'epochs': 500,
          'dropout': 0.2,
          'levels': 6,
          'learning_rate': 0.00014329,
          'patience_epochs': 100,
          'val_split': 0.08,
          'hidden_layer': 3,
          'decay_steps': 100000,
          'decay_rate': 0.96,
          }
h = params.get('height')
w = params.get('width')
img_size = (h, w, 1)
latent_space = (12, 4, 4, 56)

run["model/parameters"] = params

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
config_ = tf.compat.v1.ConfigProto()
config_.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config_)

os.chdir(
    '/home/aijjeh/Desktop/Phd_Projects/Sequence_prediction/Full_wavefield_frames_time_series_project/Datasets/label_set')

file_frame = np.load('frames_initial.npy')
file_frame = file_frame[:]


class Full_wavefield_frames(keras.utils.Sequence):
    def __init__(self, batch_size_, img_size_, input_img_paths_total_, target_img_paths_, time_stamps_):
        self.batch_size = batch_size_
        self.img_size = img_size_
        self.input_img_paths_total = input_img_paths_total_
        self.target_img_paths = target_img_paths_
        self.time_stamps = time_stamps_

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        index_ = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths_total[index_:index_ + self.batch_size]
        batch_target_img_paths = self.target_img_paths[index_:index_ + self.batch_size]

        x = np.zeros((self.batch_size,) + (self.time_stamps,) + self.img_size + (1,), dtype="float16")  #

        for batch_num in range(self.batch_size):
            batch_input_img_paths = batch_input_img_paths[batch_num][
                                    file_frame[index_] - self.time_stamps // 2: file_frame[
                                                                                    index_] + self.time_stamps // 2]
            for j, path in enumerate(batch_input_img_paths):
                img_sample = load_img(path, target_size=self.img_size, color_mode="grayscale")
                img_sample = np.expand_dims(img_sample, 2)
                img_sample = img_sample / 255.0
                x[batch_num][j] = img_sample
        # y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="float16")
        # for j, path in enumerate(batch_target_img_paths):
        #     img_sample = load_img(path, target_size=self.img_size, color_mode="grayscale")
        #     img_sample = np.expand_dims(img_sample, 2)
        #     img_sample = img_sample / 255.0
        #     y[j] = img_sample
        #     # Ground truth labels are 1, 2, 3. Subtract one to make them 0, 1, 2:
        #     # y[j] -= 1
        return x, x


os.chdir('/home/aijjeh/Desktop/Phd_Projects/Modeling_guided_waves/Model_1_2_3_datasets/Model_1_dataset')

gt = np.load('GT_full_wavefields.npy')
x = gt
y = gt


#
# val_samples = int(0.20 * len(my_list))
#
# train_input_img_paths = input_img_paths_total[:-val_samples]
# train_target_img_paths = target_img_paths[:-val_samples]
# val_input_img_paths = input_img_paths_total[-val_samples:]
# val_target_img_paths = target_img_paths[-val_samples:]
#
#
# train_gen = Full_wavefield_frames(params['batches'], img_size,
#                                   train_input_img_paths, train_target_img_paths,
#                                   params['time_stamps'])
#
# val_gen = Full_wavefield_frames(params['batches'], img_size,
#                                 val_input_img_paths, val_target_img_paths,
#                                 params['time_stamps'])


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


def VAE():
    inputs_encoder = Input(shape=((params.get('time_stamps'),) + (params.get('height'), params.get('width'), 1)))
    x_layer = inputs_encoder
    skip_connection = []
    ####################################################################################################################
    # Encoder
    ####################################################################################################################
    factor = 1
    for level in range(params.get('levels')):
        x_layer = get_time_distributed(x_layer, params.get('num_filters') * factor, 5)
        x_layer = normalize_batch(x_layer)
        x_layer = TimeDistributed(tf.keras.layers.AvgPool2D((2, 2), strides=(2, 2)))(x_layer)
        x_layer = keras.layers.Dropout(params.get('dropout'), name='skip_connection_%d' % level)(x_layer)
        skip_connection.append(x_layer)
        factor += 1

    ####################################################################################################################
    # bottleneck layer
    ####################################################################################################################

    x_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(params.get('num_filters') * (factor + 1), 1,
                                                                     padding='same',
                                                                     activation='relu'), name='latent_space')(x_layer)
    ####################################################################################################################
    # Decoder
    ####################################################################################################################

    for level_ in reversed(range(params.get('levels'))):
        x_layer = concatenate([x_layer, skip_connection[level_]], axis=-1)
        x_layer = TimeDistributed(UpSampling2D((2, 2)))(x_layer)
        x_layer = keras.layers.Dropout(params.get('dropout'))(x_layer)  # adding dropout layer
        x_layer = get_time_distributed(x_layer, params.get('num_filters') * factor, 3)
        factor -= 1
    ####################################################################################################################
    x_layer = ConvLSTM2D(params.get('time_stamps'), 1, padding='same', return_sequences=True)(x_layer)
    ####################################################################################################################
    # Output layer
    output = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(1, 1,
                                                                    padding='same',
                                                                    activation='sigmoid'), name='output')(x_layer)
    ####################################################################################################################
    VAE_model = tf.keras.models.Model(inputs_encoder, output, name="VAE")
    VAE_model.summary()
    VAE_model.compile(optimizer=tf.keras.optimizers.Adam(params.get('learning_rate')),
                      loss='mse',
                      metrics=PSNR)

    return VAE_model
    ####################################################################################################################


####################################################################################################################
# VAE model
####################################################################################################################

VAE_Model = VAE()
keras.backend.clear_session()

env_path = '/home/aijjeh/Desktop/Phd_Projects/Modeling_guided_waves/'

checkpoint_filepath = '/home/aijjeh/Desktop/Phd_Projects/Modeling_guided_waves/temp/checkpoint/model_1_capturing_latent_space.h5'

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


VAE_Model.fit(x, y,
              batch_size=params['batches'],
              epochs=params.get('epochs'),
              validation_split=0.2,
              callbacks=[MonitoringCallback(), callbacks])  #
