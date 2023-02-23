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

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

# hyperparameters
samples = 475
n_size = 194560
learning_rate = 3e-4
features_shape = (512, 1)
hidden_layers = 2
dropout = 0.2
batch_size = 64
val_split = 0.2
epochs = 500
levels = 5
filter_size = 8
env_path = '/home/aijjeh/Desktop/Phd_Projects/Modeling_guided_waves/'

# Learning schedule
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    learning_rate,
    decay_steps=10000,
    decay_rate=0.96,
    staircase=False)

Optimizer = keras.optimizers.Adam(lr_schedule)


def load_dataset():
    os.chdir('/home/aijjeh/Desktop/Phd_Projects/Modeling_guided_waves/dataset')

    X_train_1 = np.load('LR_GT_del_images.npy')
    X_train_1 = np.reshape(X_train_1, (475 * 512, 4))

    X_train_2 = np.load('LR_ref_frames_images.npy')
    X_train_2 = np.reshape(X_train_2, (475 * 512, 32, 32))

    Y_train = np.load('LR_labels_images.npy')
    Y_train = np.reshape(Y_train, (475 * 512, 32, 32))

    X1_train = X_train_1[:n_size]
    X2_train = X_train_2[:n_size]
    y_in_train = Y_train[:n_size]
    return X2_train, X1_train, y_in_train


X1, X2, Y = load_dataset()

print(X1.shape, X2.shape, Y.shape)


def conv_block(in_blk, num, k_size):
    layer11 = tf.keras.layers.Conv2D(num,
                                     k_size,
                                     strides=1,
                                     padding='same',
                                     activation='relu', )(in_blk)
    return layer11


#  Model ############################

inputA = tf.keras.layers.Input(shape=(32, 32, 1), name='LR_img_input')
inputB = tf.keras.layers.Input(shape=(4,), name='Coordinates')

x1 = conv_block(inputA, filter_size, 3)
skip_tensor = []

for i in range(levels):
    x1 = conv_block(x1, filter_size * 2 ** i, 3)
    x1 = tf.keras.layers.MaxPool2D(2, 2)(x1)
    skip_tensor.append(x1)

print(x1)

x1 = tf.keras.layers.Flatten()(x1)
x1 = tf.keras.layers.concatenate([inputB, x1], axis=-1)
x2 = tf.keras.layers.Dense(1024, activation='relu')(x1)
x2 = tf.keras.layers.Dense(1024, activation='relu')(x2)

####################################################################
####################################################################
x3 = tf.keras.layers.Reshape((1, 1, 1024))(x2)
print(x3)
for j in reversed(range(levels)):
    x3 = tf.keras.layers.concatenate((x3, skip_tensor[j]), axis=-1)
    x3 = tf.keras.layers.UpSampling2D(2)(x3)
    x3 = conv_block(x3, filter_size * 2 ** j, 3)
output = tf.keras.layers.Conv2D(1, 1, padding='same', activation='sigmoid')(x3)

#
########################################################################################################################
# Output layer
########################################################################################################################

# layer_A = tf.signal.fft(tf.cast(inputA, dtype=tf.complex64))
# layer_A = tf.signal.fftshift(layer_A)
# layer_A = abs(layer_A)
# layer_A = tf.math.l2_normalize(layer_A.numpy())

# layer_A = tf.keras.layers.Dense(512, activation='relu')(inputA)
# for i in range(levels):
#     layer_A = tf.keras.layers.Dense(512, activation='relu')(layer_A)
#     layer_A = tf.keras.layers.Dropout(dropout)(layer_A)
# # flat_layer = tf.keras.layers.Flatten()(layer_A)
#
# concat_layer = tf.keras.layers.concatenate([inputB, layer_A], axis=-1)
# output = tf.keras.layers.Dense(516, activation='relu')(concat_layer)
# output = tf.keras.layers.Dropout(dropout)(output)
# output = tf.keras.layers.Dense(768, activation='relu')(output)
# output = tf.keras.layers.Dropout(dropout)(output)
# output = tf.keras.layers.Dense(512, activation='sigmoid')(output)

# define a model with a list of two inputs
# print(output)
# model3 = tf.keras.Model(inputs=[inputs_2], outputs=output, name='model_3')
# model3.summary()
####################################################################

# vae_input1 = tf.keras.layers.Input(shape=(512, 1), name="VAE_input1")
# vae_input2 = tf.keras.layers.Input(shape=(4,), name="VAE_input2")

# vae_decoder_output = model3(model2([model1(inputA), inputB]))
AE_model = tf.keras.models.Model([inputA, inputB], output, name='VAE')


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


def custom_loss(y_true, y_pred):
    y_true_k = y_true
    y_pred_k = y_pred
    # y_true_k = tf.transpose(y_true_k, perm=[0, 3, 1, 2])  # ---> [batch, channels, rows, cols]
    fft2d_true = tf.signal.fft2d(tf.cast(y_true_k, dtype=tf.complex64))
    fft2d_true = tf.signal.fftshift(fft2d_true, axes=(1, 2))
    # fft2d_true = tf.transpose(fft2d_true, perm=[0, 2, 3, 1])  # ---> [batch, rows, cols, channels]
    fft2d_true = abs(fft2d_true)
    fft2d_true_norm = tf.math.l2_normalize(fft2d_true.numpy())

    # y_pred_k = tf.transpose(y_pred_k, perm=[0, 3, 1, 2])  # ---> [batch, channels, rows, cols]
    fft2d_pred = tf.signal.fft2d(tf.cast(y_pred_k, dtype=tf.complex64))
    fft2d_pred = tf.signal.fftshift(fft2d_pred, axes=(1, 2))
    # fft2d_pred = tf.transpose(fft2d_pred, perm=[0, 2, 3, 1])  # ---> [batch, rows, cols, channels]
    fft2d_pred = abs(fft2d_pred)
    fft2d_pred_norm = tf.math.l2_normalize(fft2d_pred.numpy())

    MSE_Fourier_domain = tf.losses.MSE(fft2d_true_norm, fft2d_pred_norm)
    MSE_Spatial = tf.losses.MSE(y_true, y_pred)
    # a = (fft2d_true_norm[0])
    # b = (fft2d_pred_norm[0])
    # 
    # ax1 = plt.subplot(1, 2, 1)
    # ax1.imshow(a, cmap='flag')
    # ax1.set_title('GT')
    # 
    # ax2 = plt.subplot(1, 2, 2)
    # ax2.imshow(b, cmap='flag')
    # ax2.set_title('Pred')
    # plt.show()

    return MSE_Fourier_domain + MSE_Spatial


AE_model.summary()
AE_model.compile(Optimizer,
                 loss=custom_loss,
                 metrics=PSNR,
                 run_eagerly=True)

checkpoint_filepath = env_path + 'temp/checkpoints/'
file_name = 'model_images.h5'

callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                              patience=3000,
                                              mode='min'),
             tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath + file_name,
                                                monitor='val_loss',
                                                save_best_only=True)]

AE_model.fit(x=[X1, X2],
             y=Y,
             batch_size=batch_size,
             validation_split=val_split,
             epochs=epochs,
             callbacks=[callbacks])
