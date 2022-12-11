import os
import numpy as np
import matplotlib.pyplot as plt
import json
import keras_tuner as kt
import csv
import tensorflow as tf
import keras
import re
import random
import neptune

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from tensorflow.python.client import device_lib
from tensorflow.keras.models import Model
from keras import layers
from keras import backend as K
from keras.callbacks import Callback
from PIL import ImageOps
from IPython.display import Image, display
from sklearn.model_selection import train_test_split
from datetime import datetime
from decouple import config
from keras.models import load_model

access_token = config('NEPTUNE_API_TOKEN')
########################################################################################################################
run = neptune.init(project_qualified_name='abdalraheem.ijjeh/PhCs-dispersion-curves',
                   api_token=access_token)
neptune.create_experiment('regression model_custom_loss')
neptune.append_tag('regression_custom_loss')
########################################################################################################################

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
########################################################################################################################

# working path env
env_path = '/home/aijjeh/Desktop/Phd_Projects/PC_uint_cell_dispersion_curves/'

# Hyperparameters
hyperparameters = {'samples': 7000,
                   'n_size': 6650,
                   'normalised': True,
                   'batches': 32,
                   'num_filters': 8,
                   'kernel_size': 3,
                   'img_shape': (256, 256, 1),
                   'epochs': 10000,
                   'dropout': 0.2,
                   'levels': 8,
                   'learning_rate': 0.00014329,
                   'patience_epochs': 5000,
                   'val_split': 0.075,
                   'hidden_layer': 3,
                   'decay:steps': 100000,
                   'decay_rate': 0.96}

os.chdir(env_path)
json = json.dumps(hyperparameters)
f = open('hyper_par_test_custom_loss_weighted_mse.json', 'w')
f.write(json)
f.close()

csv_eval_file = 'models_evaluation_n_size_custom_loss_test.csv'

W = list(range(1, 1465))
W /= np.sum(W)
W = W.astype('float32')
W = np.asarray(W)


def load_dataset():
    os.chdir(env_path + 'dataset/')
    X_train = np.load('train_xy_img_samples_%d.npy' % hyperparameters['samples'])
    Y_train = np.load('train_y_samples_%d.npy' % hyperparameters['samples'])
    if hyperparameters['normalised']:
        Y_train = Y_train[:, :, 0] / np.max(Y_train[:, :, 0])
        normal_ = 'normalised'
    else:
        Y_train = Y_train[:, :, 0]
        normal_ = 'not_normalised'

    Train_x, test_x_samples, Train_label, test_y_samples = train_test_split(X_train, Y_train,
                                                                            train_size=0.95,
                                                                            shuffle=False,
                                                                            random_state=1988)

    return Train_x, Train_label, normal_, test_x_samples, test_y_samples


########################################################################################################################
# Model
########################################################################################################################
def conv_block(in_blk, num, k_size):
    layer11 = tf.keras.layers.Conv2D(num,
                                     (k_size, k_size),
                                     padding='same',
                                     activation='relu',
                                     )(in_blk)
    return layer11


def custom_mse(class_weights):
    def weighted_MSE(y_true, y_pred):
        # Get sorted indices along last axis
        inds = tf.argsort(y_pred, -1)

        # Since gather preserves order, gather from the second axis.
        sorted_features = tf.gather(class_weights, inds)

        # Formula:
        # w_1*(y_1-y'_1)^2 + ... + w_100*(y_100-y'_100)^2 / sum(weights)
        # y_true = tf.sort(y_true, direction='ASCENDING')
        # y_pred = tf.sort(y_pred, direction='ASCENDING')
        return K.sum(sorted_features * K.square(y_true - y_pred)) / K.sum(class_weights)

    return weighted_MSE


# def weighted_mse(y_true, y_pred):
#     ones = K.ones_like(y_true[0, :])  # a simple vector with ones shaped as (60,)
#     idx = K.cumsum(ones)  # similar to a 'range(1,61)'
#
#     return K.mean((1 / idx) * K.square(y_true - y_pred))


def build_model():
    inputs_1 = tf.keras.Input(shape=hyperparameters['img_shape'])
    x_in = inputs_1
    for i in range(7):  # Int specifies the dtype of the values
        k_size = 5
        x_in = conv_block(x_in, hyperparameters['num_filters'] * 2 ** i, k_size)
        if i in [0, 1, 3, 4, 5, 6]:
            x_in = tf.keras.layers.AvgPool2D()(x_in)
        else:
            x_in = tf.keras.layers.MaxPool2D()(x_in)

    output1 = tf.keras.layers.Flatten()(x_in)
    output1 = tf.keras.layers.Dense(output1.shape[-1], activation='relu')(output1)

    h_levels = 1
    dropouts_l = 0.2

    for h in range(h_levels):
        dense_units = 6656
        output1 = tf.keras.layers.Dense(dense_units, activation='relu')(output1)
        output1 = tf.keras.layers.Dropout(dropouts_l)(output1)

    dense_units_1 = 1944
    output1 = tf.keras.layers.Dense(dense_units_1, activation='relu')(output1)
    output1 = tf.keras.layers.Dropout(dropouts_l)(output1)
    output1 = tf.keras.layers.Dropout(dropouts_l)(output1)
    output1 = tf.keras.layers.Dense(1464, activation='sigmoid')(output1)
    ####################################################################################################################
    # Model
    ####################################################################################################################
    temp_model = tf.keras.models.Model(inputs=inputs_1, outputs=output1, name='AE_Model')
    temp_model.compile(optimizer=keras.optimizers.Adam(learning_rate=hyperparameters['learning_rate']),
                       loss=custom_mse(W),
                       metrics=[tf.keras.metrics.RootMeanSquaredError()],
                       run_eagerly=True)
    temp_model.summary()

    return temp_model


# calling dataset func
X, Y, s, x_test, y_test = load_dataset()
print(X.shape)
print(Y.shape)
checkpoint_filepath = env_path + 'temp/checkpoint/custom_loss_PC_model_%s_%d_lr_%s_filters_%d_levels_%d_batches_%d_epochs_%d_dropout_%s_val_split_%s_hidden_%d.h5' % (
    s,
    hyperparameters['samples'],
    hyperparameters['learning_rate'],
    hyperparameters['num_filters'],
    hyperparameters['levels'],
    hyperparameters['batches'],
    hyperparameters['epochs'],
    hyperparameters['dropout'],
    hyperparameters['val_split'],
    hyperparameters['hidden_layer'])

callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_root_mean_squared_error',
                                              patience=hyperparameters['patience_epochs'],
                                              mode='min',
                                              restore_best_weights=True),
             tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                monitor='val_loss',
                                                save_best_only=True)]

dispersion_model = build_model()
dispersion_model.fit(x=X,
                     y=Y,
                     batch_size=hyperparameters['batches'],
                     validation_split=hyperparameters['val_split'],
                     epochs=hyperparameters['epochs'],
                     callbacks=[callbacks])

model = load_model(checkpoint_filepath, compile=True)
model.summary()

score = model.evaluate(x_test, y_test, verbose=0, batch_size=32)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
