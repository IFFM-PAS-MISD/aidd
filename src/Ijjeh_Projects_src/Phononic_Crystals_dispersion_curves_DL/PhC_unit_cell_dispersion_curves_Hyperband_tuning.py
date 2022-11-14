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
# import neptune.new as neptune
import neptune
from decouple import config
import json
import keras_tuner as kt

access_token = config('NEPTUNE_API_TOKEN')
########################################################################################################################
run = neptune.init(project_qualified_name='abdalraheem.ijjeh/PhCs-dispersion-curves',
                   api_token=access_token)
neptune.create_experiment('regression model')
neptune.append_tag('regression')
########################################################################################################################
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

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
                   'learning_rate': 3e-4,
                   'patience_epochs': 3000,
                   'val_split': 0.075,
                   'hidden_layer': 3,
                   'decay:steps': 100000,
                   'decay_rate': 0.96}

os.chdir(env_path)
json = json.dumps(hyperparameters)
f = open('hyper_par.json', 'w')
f.write(json)
f.close()


# Learning schedule
# lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#     hyperparameters['learning_rate'],
#     decay_steps=hyperparameters['decay:steps'],
#     decay_rate=hyperparameters['decay_rate'],
#     staircase=False)
#
# Optimizer = keras.optimizers.Adam(hyperparameters['learning_rate'])


#  Loading dataset
def load_dataset():
    os.chdir(env_path + 'dataset/')

    X_train = np.load('train_xy_img_samples_%d.npy' % hyperparameters['samples'])
    print(X_train.shape)

    Y_train = np.load('train_y_samples_%d.npy' % hyperparameters['samples'])

    print(Y_train.shape)

    if hyperparameters['normalised']:
        Y_train = Y_train[:, :, 0] / np.max(Y_train[:, :, 0])
        normal_ = 'normalised'
    else:
        Y_train = Y_train[:, :, 0]
        normal_ = 'not_normalised'

    x_train = X_train[:hyperparameters['n_size']]
    y_train = Y_train[:hyperparameters['n_size']]

    return x_train, y_train, normal_


########################################################################################################################
# Model
########################################################################################################################
def conv_block(in_blk, num, k_size):
    layer11 = tf.keras.layers.Conv2D(num,
                                     (k_size, k_size),
                                     padding='same',
                                     activation='relu',
                                     )(in_blk)
    # layer12 = tf.keras.layers.Conv2D(num,
    #                                  (k_size, k_size),
    #                                  padding='same',
    #                                  activation='relu')(layer11)
    # concat = tf.keras.layers.concatenate([layer11, layer12], axis=-1)  # , in_blk
    # BN_layer = tf.keras.layers.BatchNormalization()(layer11)
    return layer11


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def custom_loss(y_true, y_pred):
    return tf.keras.losses.MSE(y_true, y_pred)


def coeff_determination(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())


def build_model(hp):
    inputs_1 = tf.keras.Input(shape=hyperparameters['img_shape'])
    # x = tf.image.resize(inputs_1, (128, 128))
    # x = tf.keras.layers.Flatten()(inputs_1)

    # x1 = inputs_1
    # x_fft = tf.transpose(x1, [0, 3, 1, 2])
    # x_fft = tf.signal.fft2d(tf.cast(x_fft, dtype=tf.complex64))
    # x_fft = tf.signal.fftshift(x_fft, axes=(-2, -1))
    # x_fft = tf.transpose(x_fft, [0, 2, 3, 1])
    # x_fft = abs(x_fft)
    # print(x_fft)

    # x = tf.keras.layers.concatenate([inputs_1, x_fft], axis=-1)
    # print(x)
    x = inputs_1
    # skip_tensor = []
    for i in range(hp.Int('conv_blocks', min_value=3, max_value=8, default=3)):  # Int specifies the dtype of the values
        filters = hp.Int('filters_' + str(i), min_value=8, max_value=32, step=4)
        k_size = hp.Int('k_size', min_value=3, max_value=7, step=2)
        x = conv_block(x, filters, k_size)
        if hp.Choice('pooling_' + str(i), ['avg', 'max']) == 'max':  # hp.Choice chooses from a list of values
            x = tf.keras.layers.MaxPool2D()(x)
        else:
            x = tf.keras.layers.AvgPool2D()(x)
        # x = tf.keras.layers.MaxPool2D((2, 2))(x)
        # skip_tensor.append(x)

    # for j in reversed(range(int(hyperparameters['levels'] / 2) - 1, hyperparameters['levels'])):
    #     x = tf.keras.layers.concatenate((x, skip_tensor[j]), axis=-1)
    #     x = tf.keras.layers.UpSampling2D((2, 2))(x)
    #     x = conv_block(x, hyperparameters['num_filters'] * 2 ** j, hyperparameters['kernel_size'])

    # x = tf.keras.layers.Conv2D(1, 1, padding='same', activation='sigmoid')(x)
    # x = tf.keras.layers.Flatten()(x)
    # output1 = tf.keras.layers.GlobalAvgPool2D()(x)

    output1 = tf.keras.layers.Flatten()(x)
    output1 = tf.keras.layers.Dense(output1.shape[-1],
                                    activation='relu')(output1)

    h_levels = hp.Int('h_levels', min_value=1, max_value=3, step=1)

    for h in range(h_levels):
        dropouts_l = hp.Float('dropout', min_value=.1, max_value=.35, step=0.05, default=None)
        dense_units = hp.Int('units', min_value=1024, max_value=8192, step=512)
        output1 = tf.keras.layers.Dense(dense_units,
                                        activation='relu')(output1)
        output1 = tf.keras.layers.Dropout(dropouts_l)(output1)

    # dense_units_1 = hp.Int('units1', min_value=1464, max_value=2042, step=32)
    #
    # output1 = tf.keras.layers.Dense(dense_units_1,
    #                                 activation='relu')(output1)
    # output1 = tf.keras.layers.Dropout(dropouts_l)(output1)
    # output1 = tf.keras.layers.Dense(dense_units_1,
    #                                 activation='relu')(output1)
    # output1 = tf.keras.layers.Dropout(hp.Float('dropout',
    #                                            min_value=.15,
    #                                            max_value=.35,
    #                                            step=0.05,
    #                                            default=None))(output1)
    output1 = tf.keras.layers.Dense(1464)(output1)

    ####################################################################################################################
    # Model
    ####################################################################################################################
    model = tf.keras.models.Model(inputs=inputs_1, outputs=output1, name='AE_Model')
    model.compile(optimizer=keras.optimizers.Adam(hp.Float('learning_rate',
                                                           min_value=9e-5,
                                                           max_value=9e-4,
                                                           sampling='log')),
                  loss=custom_loss,
                  metrics=[tf.keras.metrics.RootMeanSquaredError()],
                  run_eagerly=True)

    model.summary()

    return model


# initialize tuner to run the model.
# using the Hyper band search algorithm
tuner = kt.Hyperband(
    hypermodel=build_model,
    objective='val_loss',
    max_epochs=200,
    factor=3,
    hyperband_iterations=1,
    directory="Keras_tuner_dir",
    project_name="Keras_tuner_Demo")

# calling dataset func
X, Y, s = load_dataset()

checkpoint_filepath = env_path + 'temp/checkpoint/PC_model_%s_%d_lr_%s_filters_%d_levels_%d_batches_%d_epochs_%d_dropout_%s_val_split_%s_hidden_%d.h5' % (
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
                                              mode='min'),
             tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                monitor='val_loss',
                                                save_best_only=True)]

# Run the search
tuner.search(X, Y,
             validation_split=hyperparameters['val_split'],
             epochs=hyperparameters['epochs'],
             callbacks=[callbacks])

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(1)[0]

# get the best model
best_model = tuner.get_best_models(1)[0]

nblocks = best_hps.get('conv_blocks')
nlevels = best_hps.get('h_levels')

print(f'Number of conv blocks: {nblocks}')
for hyparam in [f'filters_{i}' for i in range(nblocks)] + [f'pooling_{i}' for i in range(nblocks)] + ['h_levels'] \
               + ['units'] + ['dropout'] + ['learning_rate'] + ['k_size']:
    print(f'{hyparam}: {best_hps.get(hyparam)}')

# display model structure
# plot_model(best_model, 'best_model.png', show_shapes=True)

# show model summary
best_model.summary()
ANN_model = tuner.hypermodel.build(best_hps)

history = ANN_model.fit(x=X,
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

os.chdir(env_path + 'h5_models/')
hypermodel.save("VAE_Quarter_samples_%d_encoder_latent_decoder_dense_ver_3.h5" % hyperparameters['samples'])

