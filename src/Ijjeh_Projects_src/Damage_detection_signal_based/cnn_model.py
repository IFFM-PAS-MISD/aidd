import scipy.io
import math
import os
from IPython.display import Image, display
from tensorflow.keras.preprocessing.image import load_img
import PIL
from PIL import ImageOps
import re
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.python.client import device_lib
import random
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
import natsort
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from scipy import signal
from sewar.full_ref import mse, rmse, psnr, uqi, ssim, ergas, scc, rase, sam, msssim, vifp
import neptune
from decouple import config
from keras.callbacks import Callback
from keras.layers import Conv2D, Add

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
print(device_lib.list_local_devices())

run = neptune.init(project_qualified_name='abdalraheem.ijjeh/Signal-prediction',
                   api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxOWE1Njk4NC03MWQxLTQwY2EtODJmMS1kZTczM2M1Y2VkMjkifQ==')
neptune.create_experiment('CNN_model_signals_damage_detection')
neptune.append_tag('CNN')

os.chdir('/home/aijjeh/Desktop/Phd_Projects/compressive_sensing_project')
mat = scipy.io.loadmat('CNNpreparedData_normalizedInputs.mat')

x_train_1 = mat['inputs_training']
x_train_2 = mat['inputs_training_HS1']
y_train = mat['S1_training']
x_train_1 = np.transpose(x_train_1, [0, -1, 1])
x_train_2 = np.transpose(x_train_2, [0, -1, 1])

print(x_train_1.shape)
print(x_train_2.shape)
print(y_train.shape)

x_train = x_train_1 / (np.abs(x_train_2) + .001)
print(x_train.shape)

x_test1 = mat['inputs_testing']
x_test1 = np.transpose(x_test1, [0, -1, 1])
x_test2 = mat['inputs_testing_HS1']
x_test2 = np.transpose(x_test2, [0, -1, 1])

x_test = x_test1 / (np.abs(x_test2) + .001)
print(x_test.shape)
y_test = mat['S1_testing']
print(y_test.shape)

batches = 8
# train_set = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batches)
test_set = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batches)

########################################################################################################################
# First CNN model
########################################################################################################################
# Encoder
########################################################################################################################
inputs_1 = tf.keras.Input(shape=(400, 5))

layer11 = tf.keras.layers.Conv1D(64, 1,
                                 padding='same',
                                 activation='relu')(inputs_1)
layer12 = tf.keras.layers.Conv1D(64, 3,
                                 padding='same',
                                 activation='relu')(inputs_1)
layer13 = tf.keras.layers.Conv1D(64, 7,
                                 padding='same',
                                 activation='relu')(inputs_1)

layer14 = tf.keras.layers.concatenate([layer11, layer12, layer13], axis=-1)
DS1 = tf.keras.layers.MaxPool1D(2, 2)(layer14)

layer21 = tf.keras.layers.Conv1D(128, 3,
                                 padding='same',
                                 activation='relu', )(DS1)
layer22 = tf.keras.layers.Conv1D(128, 3,
                                 padding='same',
                                 activation='relu', )(layer21)
DS2 = tf.keras.layers.MaxPool1D(2, 2)(layer22)

layer31 = tf.keras.layers.Conv1D(256, 3,
                                 padding='same',
                                 activation='relu', )(DS2)
layer32 = tf.keras.layers.Conv1D(256, 3,
                                 padding='same',
                                 activation='relu', )(layer31)
# BN3 = tf.keras.layers.BatchNormalization()(layer32)
DS3 = tf.keras.layers.MaxPool1D(2, 2)(layer32)

layer41 = tf.keras.layers.Conv1D(512, 3,
                                 padding='same',
                                 activation='relu', )(DS3)
layer42 = tf.keras.layers.Conv1D(512, 3,
                                 padding='same',
                                 activation='relu', )(layer41)
DS4 = tf.keras.layers.MaxPool1D(2, 2)(layer42)
########################################################################################################################
# BottleNeck
########################################################################################################################
layer51 = tf.keras.layers.Conv1D(512, 3, padding='same', activation='relu')(DS4)
layer52 = tf.keras.layers.Conv1D(512, 3, padding='same', activation='relu')(layer51)
########################################################################################################################
# Decoder
########################################################################################################################
UP1 = tf.keras.layers.Conv1DTranspose(512, 3, 2, padding='same')(layer52)  # tf.keras.layers.UpSampling1D(2)(layer52)
concat1 = tf.keras.layers.concatenate([UP1, layer42], axis=-1)
layer61 = tf.keras.layers.Conv1D(512, 3, padding='same', activation='relu')(concat1)
layer62 = tf.keras.layers.Conv1D(512, 3, padding='same', activation='relu')(layer61)

UP2 = tf.keras.layers.Conv1DTranspose(256, 3, 2, padding='same')(layer62)  # tf.keras.layers.UpSampling1D(2)(layer62)
concat2 = tf.keras.layers.concatenate([UP2, layer32], axis=-1)
layer71 = tf.keras.layers.Conv1D(256, 3, padding='same', activation='relu')(concat2)
layer72 = tf.keras.layers.Conv1D(256, 3, padding='same', activation='relu')(layer71)

UP3 = tf.keras.layers.Conv1DTranspose(128, 3, 2, padding='same')(layer72)  # tf.keras.layers.UpSampling1D(2)(layer72)
concat3 = tf.keras.layers.concatenate([UP3, layer22], axis=-1)
layer81 = tf.keras.layers.Conv1D(128, 3, padding='same', activation='relu')(concat3)
layer82 = tf.keras.layers.Conv1D(128, 3, padding='same', activation='relu')(layer81)

UP4 = tf.keras.layers.Conv1DTranspose(64, 3, 2, padding='same')(layer82)  # tf.keras.layers.UpSampling1D(2)(layer82)
concat4 = tf.keras.layers.concatenate([UP4, layer14], axis=-1)
layer91 = tf.keras.layers.Conv1D(64, 3, padding='same', activation='relu')(concat4)
layer92 = tf.keras.layers.Conv1D(64, 3, padding='same', activation='relu')(layer91)
########################################################################################################################
# Output layer
########################################################################################################################
Flatten = tf.keras.layers.Flatten()(layer92)
cnn_output = tf.keras.layers.Dense(400)(Flatten)
cnn_output = tf.keras.layers.Dense(600)(cnn_output)
cnn_output = tf.keras.layers.Dense(400)(cnn_output)

cnn_model = tf.keras.models.Model(inputs_1, cnn_output, name="CNN_model")
cnn_model.summary()

########################################################################################################################
# Second CNN model
########################################################################################################################
inputs_2 = tf.keras.Input(shape=(400, 1))

layer11 = tf.keras.layers.Conv1D(64, 1,
                                 padding='same',
                                 activation='relu')(inputs_2)
layer12 = tf.keras.layers.Conv1D(64, 3,
                                 padding='same',
                                 activation='relu')(inputs_2)
layer13 = tf.keras.layers.Conv1D(64, 7, padding='same', activation='relu')(inputs_2)

layer14 = tf.keras.layers.concatenate([layer11, layer12, layer13], axis=-1)

DS1 = tf.keras.layers.MaxPool1D(2, 2)(layer14)

layer21 = tf.keras.layers.Conv1D(128, 3,
                                 padding='same',
                                 activation='relu')(DS1)
layer22 = tf.keras.layers.Conv1D(128, 3,
                                 padding='same',
                                 activation='relu')(layer21)
DS2 = tf.keras.layers.MaxPool1D(2, 2)(layer22)

layer31 = tf.keras.layers.Conv1D(256, 3,
                                 padding='same',
                                 activation='relu')(DS2)
layer32 = tf.keras.layers.Conv1D(256, 3,
                                 padding='same',
                                 activation='relu')(layer31)
DS3 = tf.keras.layers.MaxPool1D(2, 2)(layer32)

layer41 = tf.keras.layers.Conv1D(512, 3,
                                 padding='same',
                                 activation='relu')(DS3)
layer42 = tf.keras.layers.Conv1D(512, 3,
                                 padding='same',
                                 activation='relu')(layer41)
DS4 = tf.keras.layers.MaxPool1D(2, 2)(layer42)
###################################################################################
layer51 = tf.keras.layers.Conv1D(512, 3, padding='same', activation='relu')(DS4)
layer52 = tf.keras.layers.Conv1D(512, 3, padding='same', activation='relu')(layer51)
###################################################################################
UP1 = tf.keras.layers.Conv1DTranspose(512, 3, 2, padding='same')(layer52)  # tf.keras.layers.UpSampling1D(2)(layer52)
concat1 = tf.keras.layers.concatenate([UP1, layer42], axis=-1)
layer61 = tf.keras.layers.Conv1D(512, 3, padding='same', activation='relu')(concat1)
layer62 = tf.keras.layers.Conv1D(512, 3, padding='same', activation='relu')(layer61)

UP2 = tf.keras.layers.Conv1DTranspose(256, 3, 2, padding='same')(layer62)  # tf.keras.layers.UpSampling1D(2)(layer62)
concat2 = tf.keras.layers.concatenate([UP2, layer32], axis=-1)
layer71 = tf.keras.layers.Conv1D(256, 3, padding='same', activation='relu')(concat2)
layer72 = tf.keras.layers.Conv1D(256, 3, padding='same', activation='relu')(layer71)

UP3 = tf.keras.layers.Conv1DTranspose(128, 3, 2, padding='same')(layer72)  # tf.keras.layers.UpSampling1D(2)(layer72)
concat3 = tf.keras.layers.concatenate([UP3, layer22], axis=-1)
layer81 = tf.keras.layers.Conv1D(128, 3, padding='same', activation='relu')(concat3)
layer82 = tf.keras.layers.Conv1D(128, 3, padding='same', activation='relu')(layer81)

UP4 = tf.keras.layers.Conv1DTranspose(64, 3, 2, padding='same')(layer82)  # tf.keras.layers.UpSampling1D(2)(layer82)
concat4 = tf.keras.layers.concatenate([UP4, layer14], axis=-1)
layer91 = tf.keras.layers.Conv1D(64, 3, padding='same', activation='relu')(concat4)
layer92 = tf.keras.layers.Conv1D(64, 3, padding='same', activation='relu')(layer91)
########################################################################################################################
# Output layer
########################################################################################################################
Flatten = tf.keras.layers.Flatten()(layer92)
cnn_output = tf.keras.layers.Dense(400)(Flatten)
cnn_output = tf.keras.layers.Dense(600)(cnn_output)
cnn_output = tf.keras.layers.Dense(400)(cnn_output)

cnn_model_2 = tf.keras.models.Model(inputs_2, cnn_output, name="CNN_model2")
cnn_model_2.summary()

########################################################################################################################
# Variational AutoEncoder (VAE) model
########################################################################################################################

vae_input = tf.keras.layers.Input(shape=(400, 5), name="VAE_input")
vae_decoder_output = cnn_model_2(cnn_model(vae_input))
vae = Model(vae_input, vae_decoder_output)

########################################################################################################################
# Model compile and training
########################################################################################################################
vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
            loss=tf.keras.losses.MSE,
            metrics=[tf.keras.metrics.CosineSimilarity(axis=-1)])


class MonitoringCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        for metric_name, metric_value in logs.items():
            neptune.log_metric(metric_name, metric_value)


callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=1000, min_delta=0)
vae.fit(x_train, y_train,
        batch_size=32,
        validation_split=0.1,
        epochs=50000,
        callbacks=[callback, MonitoringCallback()])

vae.evaluate(test_set)
os.chdir('/home/aijjeh/Desktop/Phd_Projects/compressive_sensing_project/signal_prediction/h5_models/')
vae.save('ann_model_signal_prediction_VAE_6.h5')
