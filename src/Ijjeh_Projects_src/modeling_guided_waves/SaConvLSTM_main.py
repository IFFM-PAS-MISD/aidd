# ============================================
__title__ = 'Modelling guided waves'
__author__ = "Abdalraheem A. Ijjeh"
__maintainer__ = "Abdalraheem A. Ijjeh"
__email__ = "aijjeh@imp.gda.pl"

# ============================================

import os
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
from SaConvLSTM import *

load_dotenv()

# # Link to Neptune
access_token = os.getenv('NEPTUNE_API_TOKEN')
run = neptune.init_run(project='abdalraheem.ijjeh/Guided-waves-modelling',
                       api_token=access_token,
                       tags=['Self attention ConvLSTM']
                       )

run["SA ConvLSTM"] = "Cascaded SA_ConvLSTM Layers"

params = {'samples': 15200,
          'n_size': 24320,
          'normalised': True,
          'batches': 16,
          'num_filters': 16,
          'kernel_size': 3,
          'shape': (32, 32, 1),
          'epochs': 1000,
          'dropout': 0.2,
          'levels': 5,
          'learning_rate': 0.00014329,
          'patience_epochs': 100,
          'val_split': 0.08,
          'hidden_layer': 3,
          'decay:steps': 100000,
          'decay_rate': 0.96,
          'time_stamps': 16
          }

run["model/parameters"] = params

env_path = '/home/aijjeh/Desktop/Phd_Projects/Modeling_guided_waves/'
os.chdir(env_path)
json = json.dumps(params)
f = open('hyper_par_GWM_mse_SA_ConvLSMT.json', 'w')
f.write(json)
f.close()

# # Run on GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

os.chdir('/home/aijjeh/Desktop/Phd_Projects/Modeling_guided_waves/dataset')
############################################################################
delam_coords = np.load('LR_GT_del.npy') * 100
delam_coords = delam_coords.reshape((475, 1, 1, 1, 5))
delam_coords = np.repeat(delam_coords, 512, axis=1)
delam_coords = delam_coords.reshape((475 * 64, 8, 1, 1, 5))
delam_coords = delam_coords[:params['n_size']]

############################################################################

delam_GT = np.load('LR_GT_labels_img.npy')
delam_GT_img = delam_GT.reshape((475, 1, 32, 32)).astype('uint8')
delam_GT_img = np.repeat(delam_GT_img, 512, axis=1)
delam_GT_img = delam_GT_img.reshape((475 * 64, 8, 32, 32, 1))
delam_GT_img = np.invert(delam_GT_img[:params['n_size']]) - 254


#############################################################################

def load_dataset():
    os.chdir('/home/aijjeh/Desktop/Phd_Projects/Modeling_guided_waves/dataset')

    x_lr_frame = np.load('LR_ref_frames.npy')
    print('LR input frames ', x_lr_frame.shape)
    x_lr_frame = np.reshape(x_lr_frame, (475 * 64, 8, 32, 32, 1))

    y_lr_frame = np.load('LR_labels.npy')
    print('LR GT frames ', y_lr_frame.shape)
    y_lr_frame = np.reshape(y_lr_frame, (475 * 64, 8, 32, 32, 1))

    lr_inputs = x_lr_frame[:params['n_size']]
    lr_labels = y_lr_frame[:params['n_size']]
    return lr_inputs, lr_labels


X1, Y = load_dataset()

print(X1.shape, delam_GT_img.shape, Y.shape)
"""
Dataset visualisation, to make sure all training samples are consistent
"""


def get_time_distributed(time_input, n_filters):
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(filters=n_filters,
                                                               kernel_size=params['kernel_size'],
                                                               strides=1,
                                                               padding='same',
                                                               activation='relu'))(time_input)
    return x


def Sa_build_model():
    input_1 = tf.keras.layers.Input(shape=(8, 32, 32, 1))
    input_2 = tf.keras.layers.Input(shape=(8, 32, 32, 1))
    input_3 = tf.keras.layers.Input(shape=(8, 1, 1, 5))
    ####################################################################################################################
    encoder = tf.keras.layers.multiply((input_1, input_2))

    encoder = SaConvLSTM2D(filters=params['num_filters'], kernel_size=(3, 3), padding="same",
                           return_sequences=True)(encoder)

    encoder = tf.keras.layers.BatchNormalization()(encoder)

    skip_tensor = []
    for cnt in range(5):
        encoder = SaConvLSTM2D(filters=params['num_filters'], kernel_size=(3, 3), padding="same",
                               return_sequences=True)(encoder)
        encoder = tf.keras.layers.BatchNormalization()(encoder)
        encoder = tf.keras.layers.TimeDistributed(tf.keras.layers.AveragePooling2D((2, 2), strides=(2, 2)))(encoder)
        skip_tensor.append(encoder)
        print(encoder.shape)

    bottleneck = tf.keras.layers.concatenate([input_3, encoder], axis=-1)
    bottleneck = SaConvLSTM2D(filters=params['num_filters'], kernel_size=(3, 3), padding="same",
                              return_sequences=True)(bottleneck)
    bottleneck = tf.keras.layers.BatchNormalization()(bottleneck)

    decoder = bottleneck

    for j in (range(1, params['levels'] + 1)):
        decoder = tf.keras.layers.concatenate((decoder, skip_tensor[-j]))
        decoder = tf.keras.layers.TimeDistributed(tf.keras.layers.UpSampling2D((2, 2)))(decoder)
        decoder = SaConvLSTM2D(filters=params['num_filters'], kernel_size=(3, 3), padding="same",
                               return_sequences=True)(decoder)
        decoder = tf.keras.layers.BatchNormalization()(decoder)

    output = tf.keras.layers.Conv3D(filters=1,
                                    kernel_size=(3, 3, 3),
                                    activation="sigmoid",
                                    padding="same")(decoder)

    model_ = Model(inputs=[input_1, input_2, input_3], outputs=output)
    ####################################################################################################################
    model_.compile(tf.keras.optimizers.Adam(params['learning_rate']),
                   loss='mse',
                   metrics=[tf.keras.metrics.RootMeanSquaredError()],
                   run_eagerly=True)
    return model_


tf.keras.backend.clear_session()
model = Sa_build_model()
model.summary()

checkpoint_filepath = env_path + 'temp/checkpoint/SA_ConvLSTM_%s_time_stamps_%s_%s_%d_lr_filters_%d_levels_%d_batches_%d_epochs_%d_dropout_%s_val_split_%s_hidden_%d.h5' % (
    params['shape'],
    params['time_stamps'],
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
                                              patience=params['patience_epochs'],
                                              mode='min'),
             tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                monitor='val_loss',
                                                save_best_only=True)]


class MonitoringCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        for metric_name, metric_value in logs.items():
            run[metric_name].log(metric_value)


model.fit([X1, delam_GT_img, delam_coords], Y,
          batch_size=params['batches'],
          validation_split=params['val_split'],
          epochs=params['epochs'],
          callbacks=[callbacks, MonitoringCallback()])  # ,
