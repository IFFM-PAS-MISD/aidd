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
                       tags=['Encoder to ANN']
                       )

# # Hyperparameters

run["LR_input_based"] = "AE"
params = {'samples': 15200,
          'n_size': 194560,
          'normalised': True,
          'batches': 16,
          'num_filters': 16,
          'kernel_size': 3,
          'shape': (32, 32, 6),
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
f = open('hyper_par_VAE.json', 'w')
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
    os.chdir('/home/aijjeh/Desktop/Phd_Projects/Modeling_guided_waves/dataset')
    # delam_coords = np.load('LR_GT_del.npy')
    # delam_coords = delam_coords.reshape((475, 1, 1, 1, 5))
    # delam_coords = np.repeat(delam_coords, 512, axis=1)
    # delam_coords = delam_coords.reshape((475 * 512, 1, 1, 5))
    # delam_coords = np.repeat(delam_coords, 32, axis=1)
    # delam_coords = np.repeat(delam_coords, 32, axis=2)

    delam_GT = np.load('LR_GT_labels_img.npy')
    delam_GT_img = delam_GT.reshape((475, 1, 32, 32))  # .astype('uint8')
    delam_GT_img = np.repeat(delam_GT_img, 512, axis=1)
    delam_GT_img = delam_GT_img.reshape((475 * 512, 32 * 32)) * 10
    # delam_GT_img = np.transpose(delam_GT_img, (0, -1, 1))
    # delam_GT_img = np.invert(delam_GT_img) - 254
    print(delam_GT_img.shape)

    x_lr_frame = np.load('LR_ref_frames.npy')
    x_lr_frame = np.reshape(x_lr_frame, (475 * 512, 32 * 32))
    # x_lr_frame = np.transpose(x_lr_frame, (0, -1, 1))
    print(x_lr_frame.shape)

    Y_train = np.load('LR_labels.npy')
    Y_train = np.reshape(Y_train, (475 * 512, 32, 32, 1))

    X_train = np.concatenate([x_lr_frame, delam_GT_img], axis=-1)
    Train_x, Val_x_samples, Train_label, Val_y_samples = train_test_split(X_train, Y_train,
                                                                          train_size=0.8,
                                                                          shuffle=False,
                                                                          random_state=1988)

    return Train_x, Train_label


train_x, train_label = load_dataset()

train_x_1 = train_x[:, 0:1024]
train_x_2 = train_x[:, 1024:]

print(train_x_1.shape)
print(train_x_2.shape)
print(train_label.shape)


def PSNR(y_true, y_pred):
    # cast the target images to integer
    # y_true = y_true * 255.0
    # y_true = tf.cast(y_true, tf.uint8)
    # y_true = tf.clip_by_value(y_true, 0, 1)
    # cast the predicted images to integer
    # y_pred = y_pred * 255.0
    # y_pred = tf.cast(y_pred, tf.uint8)
    # y_pred = tf.clip_by_value(y_pred, 0, 255)
    # return the psnr
    return tf.image.psnr(y_true, y_pred, max_val=1)


lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    params['learning_rate'],
    decay_steps=10000,
    decay_rate=0.96,
    staircase=False)


# def custom_loss_FFT2D(y_true, y_pred):
#     """
#     :param y_true:
#     :param y_pred:
#     :return: mse of the fourier domain + mse of the spatial domain
#     """
#
#     """
#     Transpose y_true and y_pred from shape [batches, time stamps, x-dim, y-dim, features] to
#     [batches, time stamps, features, x-dim, y-dim]
#     """
#     # y_true = tf.transpose(y_true, [0, 3, 1, 2])
#     # y_pred = tf.transpose(y_pred, [0, 3, 1, 2])
#
#     fft2d_true = tf.signal.fft3d(tf.cast(y_true, dtype=tf.complex64))
#     fft2d_true /= 1024
#     fft2d_pred = tf.signal.fft3d(tf.cast(y_pred, dtype=tf.complex64))
#     fft2d_pred /= 1024
#     # temp_fft2d_true = fft2d_true
#     # temp_fft2d_pred = fft2d_pred
#
#     # fft2d_true = tf.where(abs(temp_fft2d_true) < tf.reduce_max(abs(temp_fft2d_true)),
#     #                       abs(temp_fft2d_true),
#     #                       tf.zeros_like(abs(temp_fft2d_true)))
#     #
#     # fft2d_pred = tf.where(abs(temp_fft2d_pred) < tf.reduce_max(abs(temp_fft2d_pred)),
#     #                       abs(temp_fft2d_pred),
#     #                       tf.zeros_like(abs(temp_fft2d_pred)))
#
#     fft2d_shift_true = tf.signal.fftshift(fft2d_true, axes=(-2, -1))
#     fft2d_shift_pred = tf.signal.fftshift(fft2d_pred, axes=(-2, -1))
#
#     fft2d_shift_true = tf.cast(fft2d_shift_true, dtype=tf.complex64)
#     fft2d_shift_pred = tf.cast(fft2d_shift_pred, dtype=tf.complex64)
#
#     MSE_Fourier_domain = tf.losses.MSE(abs(fft2d_shift_true), abs(fft2d_shift_pred))
#
#     ifft2d_shift_true = tf.signal.ifftshift(fft2d_shift_true, axes=(-2, -1))
#     ifft2d_shift_pred = tf.signal.ifftshift(fft2d_shift_pred, axes=(-2, -1))
#
#     # ifft2d_shift_true = tf.signal.ifft3d(ifft2d_shift_true)
#     # ifft2d_shift_pred = tf.signal.ifft3d(ifft2d_shift_pred)
#
#     # ifft2d_shift_true = tf.transpose(ifft2d_shift_true, [0, 2, 3, 1])
#     # ifft2d_shift_pred = tf.transpose(ifft2d_shift_pred, [0, 2, 3, 1])
#     MSE_Spatial = tf.losses.MSE(abs(ifft2d_shift_true), abs(ifft2d_shift_pred))
#
#     # MSE_Spatial = tf.losses.MSE(y_true, y_pred)
#     # """
#     # Cast into complex 64 then calculate the FFT2D for the y_true and y_pred
#     # """
#     # fft2d_true = tf.signal.fft2d(tf.cast(y_true, dtype=tf.complex64))
#     # fft2d_pred = tf.signal.fft2d(tf.cast(y_pred, dtype=tf.complex64))
#     #
#     # fft2d_true = tf.signal.fftshift(fft2d_true, axes=(-2, -1))
#     # fft2d_pred = tf.signal.fftshift(fft2d_pred, axes=(-2, -1))
#     #
#     # """
#     # Normalise the FFT2D by dividing the calculated tensors by N (size of the fft2d(input))
#     # """
#     # N = tf.size(fft2d_true)
#     # N = tf.cast(N, dtype=tf.complex64)
#     # fft2d_true = tf.divide(fft2d_true, N)
#     # fft2d_pred = tf.divide(fft2d_pred, N)
#     #
#     # """
#     # Remove the higher modes (high frequencies)
#     # """
#     # for _ in modes:
#     #     fft2d_true = tf.where(abs(fft2d_true) < tf.reduce_max(abs(fft2d_true)),
#     #                           abs(fft2d_true),
#     #                           tf.zeros_like(abs(fft2d_true)))
#     #     fft2d_pred = tf.where(abs(fft2d_pred) < tf.reduce_max(abs(fft2d_pred)),
#     #                           abs(fft2d_pred),
#     #                           tf.zeros_like(abs(fft2d_pred)))
#     #
#     # """
#     # Transpose the to calculate the MSE in the Fourier domain
#     # """
#     # fft2d_true_ = tf.transpose(fft2d_true, [0, 1, 3, 4, 2])
#     # fft2d_pred_ = tf.transpose(fft2d_pred, [0, 1, 3, 4, 2])
#     #
#     # MSE_Fourier_domain = tf.losses.MSE(abs(fft2d_true_), abs(fft2d_pred_))
#     #
#     # """
#     # Here, I performed the ifft2d to calculate the mse for y_true and y_pred in spatial domain
#     # """
#     # fft2d_true = tf.cast(fft2d_true, dtype=tf.complex64)
#     # fft2d_pred = tf.cast(fft2d_pred, dtype=tf.complex64)
#     #
#     # ifft2d_true = tf.signal.ifft2d(fft2d_true) * N
#     # ifft2d_pred = tf.signal.ifft2d(fft2d_pred) * N
#     #
#     # ifft2d_true = tf.signal.fftshift(ifft2d_true, axes=(-2, -1))
#     # ifft2d_pred = tf.signal.fftshift(ifft2d_pred, axes=(-2, -1))
#     #
#     # ifft2d_true = tf.transpose(ifft2d_true, [0, 1, 3, 4, 2])
#     # ifft2d_pred = tf.transpose(ifft2d_pred, [0, 1, 3, 4, 2])
#     # MSE_Spatial = tf.losses.MSE(abs(ifft2d_true), abs(ifft2d_pred))
#
#     # for cont in range(16):
#     #     fig1, [ax10, ax20] = plt.subplots(1, 2, figsize=(10, 5))
#     #
#     #     a = abs(ifft2d_shift_true[0][cont])
#     #     a = a.numpy()
#     #
#     #     b = abs(ifft2d_shift_pred[0][cont])
#     #     b = b.numpy()
#     #
#     #     ax10.imshow(np.transpose(a, [1, 2, 0]), cmap='jet')
#     #     ax20.imshow(np.transpose(b, [1, 2, 0]), cmap='jet')
#     #     plt.show()
#     #
#     # """
#     # calculating the total loss in Fourier and spatial domains
#     # """
#     total_loss = tf.add(MSE_Fourier_domain, MSE_Spatial)
#
#     return total_loss


def get_model():
    inputs1 = tf.keras.Input(shape=(1024, 1))
    inputs2 = tf.keras.Input(shape=(1024, 1))
    encoder_1, encoder_2 = inputs1, inputs2

    encoder_1 = tf.keras.layers.concatenate([encoder_1, encoder_2], axis=-1)
    encoder_1 = tf.reduce_sum(encoder_1, axis=-1, keepdims=True)
    # skip_tensor = []
    for cnt in range(params.get('levels')):
        encoder_1 = tf.keras.layers.Convolution1D(params.get('num_filters') * 2 * (cnt + 1),
                                                  params.get('kernel_size'),
                                                  strides=1,
                                                  padding='same',
                                                  activation='relu')(encoder_1)
        # encoder_1 = tf.keras.layers.MaxPool2D((2, 2))(encoder_1)
        # encoder_2 = tf.keras.layers.MaxPool2D((2, 2))(encoder_2)
        # skip_tensor.append(encoder_1)

    # bottleneck = tf.keras.layers.concatenate([encoder_1, encoder_2], axis=-1)
    # bottleneck = tf.reduce_prod(bottleneck, axis=-1, keepdims=True)
    bottleneck = tf.keras.layers.Flatten()(encoder_1)

    bottleneck = tf.keras.layers.Dense(1024, activation='relu')(bottleneck)
    bottleneck = tf.keras.layers.Dense(1024 * 2, activation='relu')(bottleneck)
    bottleneck = tf.keras.layers.Dropout(params.get('dropout'))(bottleneck)
    bottleneck = tf.keras.layers.Dense(1024 * 2, activation='relu')(bottleneck)
    bottleneck = tf.keras.layers.Dropout(params.get('dropout'))(bottleneck)

    # bottleneck = tf.keras.layers.Dense(bottleneck.shape[-1] * 2, activation='relu')(bottleneck)
    # bottleneck = tf.keras.layers.Dropout(params.get('dropout'))(bottleneck)

    # bottleneck = tf.keras.layers.Dense(bottleneck.shape[-1] * 2, activation='relu')(bottleneck)
    # bottleneck = tf.keras.layers.Dropout(params.get('dropout'))(bottleneck)

    bottleneck = tf.keras.layers.Dense(1024, activation='sigmoid')(bottleneck)
    bottleneck = tf.keras.layers.Reshape((32, 32, 1))(bottleneck)

    ###################################################################################################################
    # Output layer
    ####################################################################################################################
    model_ = Model(inputs=[inputs1, inputs2], outputs=bottleneck)
    ####################################################################################################################
    model_.compile(tf.keras.optimizers.Adam(lr_schedule),
                   loss='mse',
                   metrics=[PSNR],
                   run_eagerly=True)
    return model_


model = get_model()
model.summary()
model.fit([train_x_1, train_x_2], train_label,
          validation_split=params.get('val_split'),
          batch_size=params.get('batches'),
          epochs=params.get('epochs'),
          callbacks=[callbacks, MonitoringCallback()])  #
