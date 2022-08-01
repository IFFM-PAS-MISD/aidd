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
from tensorflow.python.client import device_lib
import random
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from keras import backend as K
import neptune.new as neptune
from neptunecontrib.monitoring.keras import NeptuneMonitor
from keras.callbacks import Callback
from sklearn.model_selection import train_test_split
from neptune.new.integrations.tensorflow_keras import NeptuneCallback
from datetime import datetime
from tensorflow.keras import regularizers

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

# Hyperparameters

samples = 7000
n_size = int(samples * 0.92)
normalised = True
batches = 32
filter_size = 8
img_shape = (256, 256, 1)
epochs = 30000
dropout = 0.2
levels = 8
learning_rate = 3e-4
patience_epochs = 10000
val_split = 0.08

# Learning schedule

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    learning_rate,
    decay_steps=10000,
    decay_rate=0.9,
    staircase=False)

# Path to working env

env_path = '/home/aijjeh/Desktop/Phd_Projects/PC_uint_cell_dispersion_curves/'


#  Loading dataset

def load_dataset():
    os.chdir(env_path + 'dataset/')

    X_train = np.load('train_xy_img_samples_%d.npy' % samples)
    print(X_train.shape)

    Y_train = np.load('train_y_samples_%d.npy' % samples)

    print(Y_train.shape)

    if normalised:
        Y_train = Y_train[:, :, 0] / np.max(Y_train)

        normal_ = 'normalised'
    else:
        Y_train = Y_train[:, :, 0]
        normal_ = 'not_normalised'

    x_train = X_train[:n_size]
    y_train = Y_train[:n_size]

    return x_train, y_train, normal_


def conv_block(in_blk, num, k_size):
    layer11 = tf.keras.layers.Conv2D(num,
                                     (k_size, k_size),
                                     padding='same',
                                     activation='relu')(in_blk)
    layer11 = tf.keras.layers.Conv2D(num,
                                     (k_size, k_size),
                                     padding='same',
                                     activation='relu')(layer11)
    return layer11


########################################################################################################################
# Model
########################################################################################################################
def build_model():
    inputs_1 = tf.keras.Input(shape=img_shape)
    x = inputs_1
    skip_tensor = []
    for i in range(levels):
        x = conv_block(x, filter_size * 2 ** i, 3)
        x = tf.keras.layers.MaxPool2D((2, 2))(x)
        skip_tensor.append(x)

    # for j in reversed(range(levels)):
    #     x = tf.keras.layers.concatenate((x, skip_tensor[j]), axis=-1)
    #     x = tf.keras.layers.UpSampling2D((2, 2))(x)
    #     x = conv_block(x, filter_size * 2 ** j, 3)

    output1 = tf.keras.layers.Flatten()(x)
    ####################################
    output1 = tf.keras.layers.Dense(1024,
                                    activation='relu')(output1)
    #
    output1 = tf.keras.layers.Dense(1024 * 2,
                                    activation='relu')(output1)
    output1 = tf.keras.layers.Dropout(dropout)(output1)
    #
    # output1 = tf.keras.layers.Dense(1024 * 2,
    #                                 activation='relu')(output1)
    # output1 = tf.keras.layers.Dropout(dropout)(output1)

    output1 = tf.keras.layers.Dense(1464, activation='sigmoid')(output1)
    ####################################################################################################################
    # Model
    ####################################################################################################################
    model = tf.keras.models.Model(inputs=inputs_1, outputs=output1, name='AE_Model')
    model.summary()
    return model


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def custom_loss(y_true, y_pred):
    return tf.keras.losses.MSE(y_true, y_pred)


ANN_model = build_model()

ANN_model.compile(tf.keras.optimizers.Adam(lr_schedule),  # Adam(lr_schedule),
                  loss=custom_loss,
                  metrics=[tf.keras.metrics.MeanSquaredError()])

# calling dataset func
X, Y, s = load_dataset()

checkpoint_filepath = env_path + 'temp/checkpoint/best_model_%s_%d.h5' % (s, samples)

callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                              patience=patience_epochs,
                                              mode='min'),
             tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                monitor='val_loss',
                                                save_best_only=True)]

ANN_model.fit(x=X,
              y=Y,
              batch_size=batches,
              validation_split=val_split,
              epochs=epochs,
              callbacks=[callbacks])

os.chdir(env_path + 'h5_models/')
ANN_model.save("ANN_Quarter_samples_%d_encoder_latent__dense.h5" % samples)
