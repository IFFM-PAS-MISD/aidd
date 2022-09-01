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

########################################################################################################################
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
print(device_lib.list_local_devices())

strategy = tf.distribute.MirroredStrategy(devices=["GPU:2"])
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

########################################################################################################################
# Link to neptune ai for monitoring
########################################################################################################################
run = neptune.init(project_qualified_name='abdalraheem.ijjeh/aidd',
                   api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxOWE1Njk4NC03MWQxLTQwY2EtODJmMS1kZTczM2M1Y2VkMjkifQ==')
neptune.create_experiment('Super Resolution with 20% CR of full wavefield')
neptune.append_tag('SR pixel-shuffle-super-resolution')

########################################################################################################################
new_dim = 32
CR = (new_dim ** 2) / (69 ** 2)  # Compression Ratio
print(CR)


########################################################################################################################


def load_dataset():
    path_cs_dataset = '/home/aijjeh/Desktop/Phd_Projects/compressive_sensing_project/cs_datasets'
    os.chdir(path_cs_dataset)

    Full_W_dataset = np.load('CS_dataset_labels_Full_wavefield_475_128_512_512.npy', mmap_mode='r+')
    print(Full_W_dataset.shape)
    # CS_arr = np.load('CS_dataset_labels_Full_wavefield_475_128_512_512.npy', mmap_mode='r+')
    # CS_arr = np.load('CS_dataset_interpolated_CR_0.215_percent_nyquist_rate_applied_totally_random_points.npy',
    #                  mmap_mode='r+')
    # CS_arr = np.load('CS_dataset_interpolated_CR_0.215_percent_nyquist_rate_applied_UNIFROM_GRID_32_32.npy',
    #                  mmap_mode='r+')
    # CS_arr = CS_arr.reshape((475 * 128, 32, 32, 1))
    CS_arr = np.load('CS_dataset_CR_0.215_percent_nyquist_rate_applied_Uniform_grid.npy')  # used with DLSS paper

    CS_arr = CS_arr.reshape((475 * 128, 32, 32, 1))
    print(CS_arr.shape)
    x_train = CS_arr[:304 * 128]
    y_train = Full_W_dataset[:304 * 128]
    x_val = CS_arr[304 * 128:380 * 128]
    y_val = Full_W_dataset[304 * 128:380 * 128]
    x_test = CS_arr[380 * 128:]
    y_test = Full_W_dataset[380 * 128:]

    batches = 1

    train_set = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batches)
    val_set = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batches)
    test_set = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batches)
    return train_set, val_set, test_set


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

    y_true_k = tf.transpose(y_true_k, perm=[0, 3, 1, 2])  # ---> [batch, channels, rows, cols]
    fft2d_true = tf.signal.fft2d(tf.cast(y_true_k, dtype=tf.complex64))
    fft2d_true = tf.signal.fftshift(fft2d_true, axes=(2, 3))
    fft2d_true = tf.transpose(fft2d_true, perm=[0, 2, 3, 1])  # ---> [batch, rows, cols, channels]
    fft2d_true = abs(fft2d_true)
    fft2d_true_norm = tf.math.l2_normalize(fft2d_true.numpy())

    y_pred_k = tf.transpose(y_pred_k, perm=[0, 3, 1, 2])  # ---> [batch, channels, rows, cols]
    fft2d_pred = tf.signal.fft2d(tf.cast(y_pred_k, dtype=tf.complex64))
    fft2d_pred = tf.signal.fftshift(fft2d_pred, axes=(2, 3))
    fft2d_pred = tf.transpose(fft2d_pred, perm=[0, 2, 3, 1])  # ---> [batch, rows, cols, channels]
    fft2d_pred = abs(fft2d_pred)
    fft2d_pred_norm = tf.math.l2_normalize(fft2d_pred.numpy())

    MSE_Fourier_domain = tf.losses.MSE(fft2d_true_norm, fft2d_pred_norm)
    MSE_Spatial = tf.losses.MSE(y_true, y_pred)
    # a = (fft2d_true_norm[0])
    # b = (fft2d_pred_norm[0])
    #
    # print(a.numpy().shape)
    # print(np.max(a.numpy()))
    # plt.imshow(a, cmap='flag')
    # plt.show()
    #
    # print(b.numpy())
    # plt.imshow(b, cmap='flag')
    # plt.show()

    return MSE_Fourier_domain + MSE_Spatial


def rdb_block(inputs, numLayers):
    # determine the number of channels present in the current input
    # and initialize a list with the current inputs for concatenation
    channels = inputs.get_shape()[-1]
    storedOutputs = [inputs]
    # iterate through the number of residual dense layers
    for _ in range(numLayers):
        # concatenate the previous outputs and pass it through a
        # CONV layer, and append the output to the ongoing concatenation
        localConcat = tf.concat(storedOutputs, axis=-1)
        out = Conv2D(filters=2 ** (channels * _), kernel_size=3, padding="same",
                     activation="relu",
                     kernel_initializer="Orthogonal")(localConcat)

        storedOutputs.append(out)
        # concatenate all the outputs, pass it through a pointwise
        # convolutional layer, and add the outputs to initial inputs
        finalConcat = tf.concat(storedOutputs, axis=-1)
        finalOut = Conv2D(filters=inputs.get_shape()[-1], kernel_size=1,  # inputs.get_shape()[-1]
                          padding="same", activation="relu",
                          kernel_initializer="Orthogonal")(finalConcat)
        finalOut = Add()([finalOut, inputs])
        # return the final output
        return finalOut


with strategy.scope():
    def SISR_model():
        rdb_Layers = 2  # 4
        inputs = tf.keras.Input(shape=(32, 32, 1))
        # x = tf.keras.layers.experimental.preprocessing.Resizing(32, 32, interpolation='bicubic')(inputs)
        x1 = Conv2D(64, 5, padding="same", activation="relu",
                    kernel_initializer="Orthogonal")(inputs)
        x2 = Conv2D(64, 3, padding="same", activation="relu",
                    kernel_initializer="Orthogonal")(x1)

        # x2 = Conv2D(64, 3, padding="same", activation="relu",
        #             kernel_initializer="Orthogonal")(inputs)
        # x2 = Conv2D(64, 5, padding="same", activation="relu",
        #             kernel_initializer="Orthogonal")(x2)

        x_to_RDB = tf.concat([x1, x2], axis=-1)

        x_RDB1 = rdb_block(x1, numLayers=rdb_Layers)

        x = Conv2D(32, 3, padding="same", activation="relu",
                   kernel_initializer="Orthogonal")(x_RDB1)
        x_RDB2 = rdb_block(x, numLayers=rdb_Layers)

        x_RDB_output = tf.keras.layers.concatenate([x_RDB1, x_RDB2])  # Concatenating outputs of RDBs

        x = tf.keras.layers.concatenate([x_RDB_output, x_to_RDB])  # Adding global residual skip connection GRL

        x = Conv2D(x_RDB_output.get_shape()[-1] * (rescale_factor ** 2), 3, padding="same",
                   activation="relu", kernel_initializer="Orthogonal")(x)
        outputs = tf.nn.depth_to_space(x, rescale_factor)
        outputs = Conv2D(1, 1, activation='sigmoid', padding='same')(outputs)
        # construct the final model and return it
        model_SR = Model(inputs, outputs)

        model_SR.summary()

        model_SR.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
                         loss=custom_loss,
                         metrics=PSNR,
                         run_eagerly=True)
        return model_SR


def ssim_loss(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))


if __name__ == '__main__':
    train_dataset, val_dataset, test_dataset = load_dataset()
    rescale_factor = int(512 / new_dim)
    print(rescale_factor)
    ####################################################################################################################
    # Parameters
    ####################################################################################################################
    epochs = 300
    ####################################################################################################################
    # calling the model
    ####################################################################################################################
    model = SISR_model()


    class MonitoringCallback(Callback):
        def on_epoch_end(self, epoch, logs={}):
            for metric_name, metric_value in logs.items():
                neptune.log_metric(metric_name, metric_value)


    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=40, min_delta=1e-8)

    checkpoint_filepath = '/home/aijjeh/Desktop/Phd_Projects/compressive_sensing_project/h5_models/checkpoint/checkpoint.h5'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True)

    model.fit(train_dataset,
              validation_data=val_dataset,
              epochs=epochs,
              callbacks=[callback, model_checkpoint_callback, MonitoringCallback()])

    model.evaluate(test_dataset)
    os.chdir('/home/aijjeh/Desktop/Phd_Projects/compressive_sensing_project/h5_models/')
    model.save(
        'super_resolution_AE_rescale_factor_%d_Uniform_grid_32_32_adding_GRL_custom_loss_FFT2D.h5' % rescale_factor)
