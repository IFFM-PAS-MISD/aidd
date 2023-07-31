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
import neptune
from decouple import config
import json
import keras_tuner as kt
import csv

# from neptune.new.integrations.tensorflow_keras import NeptuneCallback

access_token = config('NEPTUNE_API_TOKEN')
########################################################################################################################
run = neptune.init(project_qualified_name='abdalraheem.ijjeh/PhCs-dispersion-curves',
                   api_token=access_token)
neptune.create_experiment('regression model xy coordinates_HPO')
neptune.append_tag('regression with xy coordinates_HPO')
########################################################################################################################
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

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
f = open('hyper_par_vectors_HPO.json', 'w')
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


X, Y, s = get_dataset()

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


def get_model(HP):
    inputs_1 = tf.keras.Input(shape=hyperparameters['features_shape'])
    x = inputs_1

    for i in range(HP.Int('conv_blocks', min_value=1, max_value=5, step=1)):  # Int specifies the dtype of the values
        filters = HP.Int('filters_' + str(i), min_value=8, max_value=128, step=8)
        k_size = HP.Int('k_size', min_value=3, max_value=6, step=2)
        x = tf.keras.layers.Conv1D(filters, k_size, padding='same', activation='relu')(x)
        if HP.Choice('pooling_' + str(i), ['avg', 'max']) == 'max':  # hp.Choice chooses from a list of values
            x = tf.keras.layers.MaxPool1D()(x)
        else:
            x = tf.keras.layers.AvgPool1D()(x)

    output1 = tf.keras.layers.Flatten()(x)
    output1 = tf.keras.layers.Dense(output1.shape[-1],
                                    activation='relu')(output1)

    h_levels = HP.Int('h_levels', min_value=1, max_value=3, step=1)
    dropouts_l = HP.Float('dropout', min_value=.15, max_value=.3, step=0.05, default=None)

    for h in range(h_levels):
        dense_units = HP.Int('units', min_value=2048, max_value=int(2048 * 2), step=64)
        output1 = tf.keras.layers.Dense(dense_units,
                                        activation='relu')(output1)
        output1 = tf.keras.layers.Dropout(dropouts_l)(output1)

    dense_units_1 = HP.Int('units1', min_value=1464, max_value=4096, step=32)

    output1 = tf.keras.layers.Dense(dense_units_1,
                                    activation='relu')(output1)
    output1 = tf.keras.layers.Dropout(dropouts_l)(output1)
    output1 = tf.keras.layers.Dense(1464, activation='sigmoid')(output1)

    ########################################################################################################################
    # Output layer
    ########################################################################################################################
    model = tf.keras.models.Model(inputs=inputs_1, outputs=output1, name='AE_Model')

    model.summary()

    model.compile(optimizer=keras.optimizers.Adam(HP.Float('learning_rate',
                                                           min_value=5e-5,
                                                           max_value=5e-4,
                                                           sampling='log')),
                  loss=tf.keras.losses.mean_squared_logarithmic_error,
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model


tuner = kt.Hyperband(
    hypermodel=get_model,
    objective='val_loss',
    max_epochs=200,
    factor=3,
    hyperband_iterations=1,
    directory="Keras_tuner_dir_ANN_coords_Hyperband",
    project_name='PhC_dispersion_diagrams_ANN_coords',
    overwrite=True)

checkpoint_filepath = env_path + 'temp/checkpoint/'
file_name = 'PC_model_vectors_HPO_%s_%d_lr_%s_filters_%d_levels_%d_batches_%d_epochs_%d_dropout_%s_val_split_%s_hidden_%d.h5' % (
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

callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                              patience=hyperparameters['patience_epochs'],
                                              mode='min'),
             tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath + file_name,
                                                monitor='val_loss',
                                                save_best_only=True)]

tuner.search(X, Y,
             validation_split=hyperparameters['val_split'],
             epochs=1000,
             callbacks=[callbacks],
             batch_size=hyperparameters['batches'])

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(1)[0]
# get the best model
best_model = tuner.get_best_models(1)[0]

n_blocks = best_hps.get('conv_blocks')
# n_levels = best_hps.get('h_levels')

print(f'Number of conv blocks: {n_blocks}')
# f'filters_{i}' for i in range(n_blocks)] +[f'pooling_{i}' for i in range(n_blocks)] + 
for hp in [f'filters_{i}' for i in range(n_blocks)] + [f'pooling_{i}' for i in range(n_blocks)] + \
          ['units1'] + ['dropout'] + ['learning_rate'] + ['k_size'] + ['h_levels'] + ['units']:
    print(f'{hp}: {best_hps.get(hp)}')

best_model.summary()

PhC_model = tuner.hypermodel.build(best_hps)


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


history = PhC_model.fit(x=X,
                        y=Y,
                        batch_size=hyperparameters['batches'],
                        validation_split=hyperparameters['val_split'],
                        epochs=hyperparameters['epochs'],
                        callbacks=[callbacks])

val_acc_per_epoch = history.history['val_loss']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))

hypermodel = tuner.hypermodel.build(best_hps)
hypermodel.fit(X, Y, epochs=best_epoch, validation_split=0.2)

# os.chdir('/home/aijjeh/Desktop/Phd_Projects/PC_uint_cell_dispersion_curves/h5_models/')
# AE_model.save("VAE_Quarter_%s_encoder_latent_decoder_dense_ver_3.h5" % s)
