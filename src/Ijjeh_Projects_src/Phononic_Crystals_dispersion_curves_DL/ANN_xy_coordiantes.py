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
from keras.callbacks import Callback
from sklearn.model_selection import train_test_split
from datetime import datetime
from tensorflow.keras import regularizers
from pathlib import Path
import neptune
from decouple import config
import json

tf.compat.v1.disable_eager_execution()

# from neptune.new.integrations.tensorflow_keras import NeptuneCallback

access_token = config('NEPTUNE_API_TOKEN')
########################################################################################################################
run = neptune.init(project_qualified_name='abdalraheem.ijjeh/PhCs-dispersion-curves',
                   api_token=access_token)
neptune.create_experiment('regression model xy coordinates')
neptune.append_tag('regression with xy coordinates')
########################################################################################################################
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

# working path env
env_path = '/home/aijjeh/Desktop/Phd_Projects/PC_uint_cell_dispersion_curves/'

# Hyperparameters
hyperparameters = {'samples': 6000,
                   'n_size': 5700,
                   'normalised': True,
                   'batches': 64,
                   'num_filters': 64,
                   'kernel_size': 3,
                   'features_shape': (121, 2),
                   'epochs': 50000,
                   'dropout': 0.2,
                   'levels': 1,
                   'learning_rate': 1e-4,
                   'patience_epochs': 5000,
                   'val_split': 0.08,
                   'hidden_layer': 2,
                   'decay:steps': 100000,
                   'decay_rate': 0.95}

os.chdir(env_path)
json = json.dumps(hyperparameters)
f = open('hyper_par_vectors.json', 'w')
f.write(json)
f.close()


def get_dataset():
    envPath = '/home/aijjeh/Desktop/Phd_Projects/PC_uint_cell_dispersion_curves/'
    os.chdir(envPath + 'dataset/')

    X_train = np.load('train_xy_coordinates_samples_%d.npy' % (hyperparameters['samples'] + 1000))
    # x_coordinates = X_train[:, :, 0]
    # y_coordinates = X_train[:, :, 1]
    # X_train = x_coordinates * 10 + y_coordinates
    print(X_train.shape)

    Y_train = np.load('train_y_samples_%d.npy' % (hyperparameters['samples'] + 1000))
    Y_train = Y_train[1000:]
    print(Y_train.shape)

    if hyperparameters['normalised']:
        Y_train = Y_train[:, :, 0] / np.max(Y_train)
        normal = 'normalised'
    else:
        Y_train = Y_train[:, :, 0]
        normal = 'not_normalised'
    print(normal)

    # x_train = X_train[:hyperparameters['n_size']]
    # y_train = Y_train[:hyperparameters['n_size']]

    Train_x, test_x_samples, Train_label, test_y_samples = train_test_split(X_train, Y_train,
                                                                            test_size=(1 - hyperparameters['n_size'] /
                                                                                       hyperparameters['samples']),
                                                                            shuffle=False,
                                                                            random_state=1988)

    return Train_x, Train_label, normal


X_set, Y_set, normalise_flag = get_dataset()

# X_set = np.reshape(X_set, (X_set.shape[0]*X_set.shape[1], X_set.shape[-1]))
########################################################################################################################
#  ANN model with XY coordinates as input
########################################################################################################################

# Learning schedule
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    hyperparameters['learning_rate'],
    decay_steps=hyperparameters['decay:steps'],
    decay_rate=hyperparameters['decay_rate'],
    staircase=False)

Optimizer = keras.optimizers.Adam(lr_schedule)


# Optimizer = tf.compat.v1.train.ProximalAdagradOptimizer(
#     0.001,
#     initial_accumulator_value=0.1,
#     l1_regularization_strength=0.2,
#     l2_regularization_strength=0.1,
#     use_locking=False,
#     name='ProximalAdagrad'
# )
#

def custom_loss(y_true, y_pred):
    Z = tf.nn.l2_loss((y_true - y_pred) ** 2, name="loss")
    loss = tf.reduce_mean(input_tensor=tf.square(Z))
    return loss


def get_model():
    inputs_1 = tf.keras.Input(shape=hyperparameters['features_shape'])
    x = tf.keras.layers.Conv1D(hyperparameters['num_filters'],
                               hyperparameters['kernel_size'],
                               strides=1,
                               activation='relu',
                               )(inputs_1)
    # x = tf.keras.layers.MaxPool1D(2, 2, padding='same')(x)
    # x = tf.keras.layers.Conv1D(hyperparameters['num_filters'],
    #                            hyperparameters['kernel_size'],
    #                            strides=1,
    #                            activation='relu',
    #                            )(x)
    # x = tf.keras.layers.MaxPool1D(2, 2, padding='same')(x)
    # x = tf.keras.layers.Conv1D(hyperparameters['num_filters'],
    #                            hyperparameters['kernel_size'],
    #                            strides=1,
    #                            activation='relu',
    #                            )(x)
    # x = tf.keras.layers.MaxPool1D(2, 2, padding='same')(x)
    # x = tf.keras.layers.Conv1D(hyperparameters['num_filters'],
    #                            hyperparameters['kernel_size'],
    #                            strides=1,
    #                            activation='relu',
    #                            )(x)
    # x = tf.keras.layers.MaxPool1D(2, 2, padding='same')(x)
    # x = tf.keras.layers.Conv1D(32,
    #                            3,
    #                            strides=1,
    #                            activation='relu',
    #                            )(x)
    # x = tf.keras.layers.Conv1D(64,
    #                            3,
    #                            strides=1,
    #                            activation='relu',
    #                            )(x)
    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(x.shape[-1], input_dim=1, activation='relu')(x)
    # x = tf.keras.layers.Flatten()(x)
    # for i in range(hyperparameters['hidden_layer']):
    #     x = tf.keras.layers.Dense(int(x.shape[-1]), activation='relu')(x)
    #     x = tf.keras.layers.Dropout(hyperparameters['dropout'])(x)
    x = tf.keras.layers.Dense(int(x.shape[-1] * 2 / 3), activation='relu')(x)
    x = tf.keras.layers.Dropout(hyperparameters['dropout'])(x)
    x = tf.keras.layers.Dense(int(x.shape[-1] * 2 / 3), activation='relu')(x)
    x = tf.keras.layers.Dropout(hyperparameters['dropout'])(x)
    x = tf.keras.layers.Dense(1464, activation='sigmoid')(x)
    ########################################################################################################################
    # Output layer
    ########################################################################################################################
    model = tf.keras.models.Model(inputs=inputs_1, outputs=x, name='AE_Model')

    model.summary()

    model.compile(Optimizer,
                  loss=tf.keras.losses.mean_squared_logarithmic_error,
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model


PhC_model = get_model()

checkpoint_filepath = env_path + 'temp/checkpoint/'
file_name = 'PC_model_vectors_%s_%d_lr_%s_filters_%d_levels_%d_batches_%d_epochs_%d_dropout_%s_val_split_%s_hidden_%d.h5' % (
    normalise_flag,
    hyperparameters['samples'],
    hyperparameters['learning_rate'],
    hyperparameters['num_filters'],
    hyperparameters['levels'],
    hyperparameters['batches'],
    hyperparameters['epochs'],
    hyperparameters['dropout'],
    hyperparameters['val_split'],
    hyperparameters['hidden_layer'])

callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                              patience=hyperparameters['patience_epochs'],
                                              mode='min'),
             tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath + file_name,
                                                monitor='val_loss',
                                                save_best_only=True)]


# callback = tf.keras.callbacks.EarlyStopping(monitor='val_mean_squared_error',
#                                             patience=100,
#                                             min_delta=0,
#                                             mode="min",
#                                             restore_best_weights=True
#                                             )
#
#
# model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
#     filepath=checkpoint_filepath,
#     save_weights_only=True,
#     monitor='val_mean_squared_error',
#     mode='min',
#     save_best_only=True)
#

class MonitoringCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        for metric_name, metric_value in logs.items():
            neptune.log_metric(metric_name, metric_value)


PhC_model.fit(x=X_set,
              y=Y_set,
              batch_size=hyperparameters['batches'],
              validation_split=hyperparameters['val_split'],
              epochs=hyperparameters['epochs'],
              callbacks=[callbacks])

# os.chdir('/home/aijjeh/Desktop/Phd_Projects/PC_uint_cell_dispersion_curves/h5_models/')
# AE_model.save("VAE_Quarter_%s_encoder_latent_decoder_dense_ver_3.h5" % s)
