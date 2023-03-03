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
                       tags=['Model 2 mapping GT of delaminations to the latent space and skip connections']
                       )

params = {'batches': 1,
          'kernel_size': 3,
          'height': 256,
          'width': 256,
          'time_stamps': 32,
          'num_filters': 16,
          'filter_size': 3,
          'epochs': 1500,
          'dropout': 0.2,
          'levels': 2,
          'learning_rate': 0.0002,
          'patience_epochs': 1200,
          'val_split': 0.2,
          'hidden_layer': 3,
          'decay_steps': 100000,
          'decay_rate': 0.96,
          }
h = params.get('height')
w = params.get('width')
img_size = (h, w)

run["model/parameters"] = params

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
config_ = tf.compat.v1.ConfigProto()
config_.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config_)

########################################################################################################################
#  Load dataset
########################################################################################################################
os.chdir('/home/aijjeh/Desktop/Phd_Projects/Modeling_guided_waves/Model_1_2_3_datasets/Model_2_dataset')
samples_gt_delamination = np.load('delamination_ground_truths.npy', mmap_mode='r+')
# samples_gt_delamination = 1 - samples_gt_delamination

healthy_ref = np.load('health_full_wave_fields.npy', mmap_mode='r+')
# healthy_ref = healthy_ref * samples_gt_delamination
train_set = np.concatenate([samples_gt_delamination, healthy_ref], axis=-1)

########################################################################################################################
#  GT
########################################################################################################################
latent_space = np.load('predicted_Latent_space.npy')
# skip_0 = np.load('predicted_skip_connection_0.npy', mmap_mode='r+')
# skip_1 = np.load('predicted_skip_connection_1.npy', mmap_mode='r+')
# skip_2 = np.load('predicted_skip_connection_2.npy', mmap_mode='r+')
# skip_3 = np.load('predicted_skip_connection_3.npy', mmap_mode='r+')
# skip_4 = np.load('predicted_skip_connection_4.npy', mmap_mode='r+')
# skip_5 = np.load('predicted_skip_connection_5.npy', mmap_mode='r+')
print(latent_space.shape)

train_x, test_x, train_y, test_y = train_test_split(train_set,
                                                    latent_space,
                                                    train_size=0.95,
                                                    shuffle=True,
                                                    random_state=1988)


def PSNR(y_true, y_pred):
    # cast the target images to integer
    # y_true = y_true * 255.0
    # y_true = tf.cast(y_true, tf.uint8)
    y_true = tf.clip_by_value(y_true, 0, 1)
    # cast the predicted images to integer
    # y_pred = y_pred * 255.0
    # y_pred = tf.cast(y_pred, tf.uint8)
    y_pred = tf.clip_by_value(y_pred, 0, 1)
    # return the psnr
    return tf.image.psnr(y_true, y_pred, max_val=1)


def normalize_batch(bn_input):
    return keras.layers.BatchNormalization()(bn_input)


def get_time_distributed(time_input, n_filters, f_size):
    x_ = TimeDistributed(Conv2D(filters=n_filters,
                                kernel_size=(f_size, f_size),
                                strides=1, padding='same',
                                activation='relu'))(time_input)
    return x_


def get_model(img_size_):
    input_1 = Input(shape=((params.get('time_stamps'),) + img_size_ + (1,)))
    # input_2 = Input(shape=((params.get('time_stamps'),) + img_size_ + (1,)))
    # x_layer = concatenate([input_1, input_2], axis=-1)
    x_layer = input_1
    # skip_connection = list()
    # skip_connection = []
    ####################################################################################################################
    # Encoder
    ####################################################################################################################
    for level in range(params.get('levels')):
        x_layer = ConvLSTM2D(params.get('time_stamps'), 3, padding='same', return_sequences=True)(x_layer)
        x_layer = tf.keras.layers.AveragePooling3D((1, 2, 2), strides=(1, 2, 2))(x_layer)
        x_layer = keras.layers.Dropout(params.get('dropout'))(x_layer)
        x_layer = tf.keras.layers.BatchNormalization(name='skip_connection_%d' % level)(x_layer)
        # skip_connection.append(x_layer)
    ####################################################################################################################
    # bottleneck layer
    ####################################################################################################################
    x_layer = ConvLSTM2D(params.get('time_stamps'), 3, padding='same', return_sequences=True)(x_layer)
    x_layer = Conv2D(32, 1, 1, padding='same', activation='sigmoid')(x_layer)
    x_layer = tf.keras.layers.BatchNormalization(name='encoded_latent_space')(x_layer)
    encoder_ = Model(input_1, x_layer, name='encoder')
    ####################################################################################################################
    encoder_.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params['learning_rate']),
                     loss=tf.keras.losses.mse,
                     metrics=[tf.keras.metrics.RootMeanSquaredError(), PSNR])
    return encoder_


keras.backend.clear_session()

Encoder = get_model(img_size)

Encoder.summary()

env_path = '/home/aijjeh/Desktop/Phd_Projects/Modeling_guided_waves/temp/checkpoint/'

checkpoint_filepath = 'Model_2_encoder_mapping_GT_Ref_latent_space_skip_connections.h5'

callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                              patience=params.get('patience_epochs'),
                                              mode='min'),
             tf.keras.callbacks.ModelCheckpoint(filepath=env_path + checkpoint_filepath,
                                                monitor='val_loss',
                                                save_best_only=True)]


class MonitoringCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        for metric_name, metric_value in logs.items():
            run[metric_name].log(metric_value)


Encoder.fit(x=train_x,
            y=train_y,
            batch_size=1,
            epochs=params.get('epochs'),
            validation_split=0.1,
            callbacks=[MonitoringCallback(), callbacks])  #
