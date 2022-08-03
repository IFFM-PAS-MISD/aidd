import math
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
import natsort
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from scipy import signal
from tensorflow.python.framework.ops import disable_eager_execution
from sewar.full_ref import mse, rmse, psnr, uqi, ssim, ergas, scc, rase, sam, msssim, vifp

# disable_eager_execution()

########################################################################################################################
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
print(device_lib.list_local_devices())
########################################################################################################################
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
########################################################################################################################
dataset_path = '/home/aijjeh/Desktop/Phd_Project/GT_RMS_waves/Dataset/'
os.chdir(dataset_path)
########################################################################################################################
dataset_x = np.load('GT2RMS2waves_Training_x_thresholded_augmented_h_v_d.npy', mmap_mode='r')
# dataset_x = np.load('GT2RMS2waves_Training_x_thresholded_diff_array.npy', mmap_mode='r')
# dataset_y = np.load('GT2RMS2waves_Labels_y.npy', mmap_mode='r')
dataset_y = np.load('GT2RMS2waves_Labels_y_augmented_h_v_d.npy', mmap_mode='r')

print(dataset_x.shape)
print(dataset_y.shape)
########################################################################################################################
filters = 64
f_size = 3
channels = dataset_x.shape[-1]
new_dim = 512

x_train = dataset_x[:1216]
y_train = dataset_y[:1216]

x_val = dataset_x[1216:1520]
y_val = dataset_y[1216:1520]

x_test = dataset_x[1520:]
y_test = dataset_y[1520:]

print("The model starts here ")

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

BATCH_SIZE = 4
train_dataset = train_dataset.batch(BATCH_SIZE)
val_dataset = val_dataset.batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)


####################################################################################################################
# Custom functions
####################################################################################################################
def get_timedistributed_conv_encoder(input_, num_filters, k_size, strides_, l_name):
    layer_x = keras.layers.Conv2D(num_filters,
                                  (k_size, k_size),
                                  padding='same',
                                  strides=strides_,
                                  name=l_name,
                                  )(input_)
    return layer_x


def iou_metric(y_true, y_pred, smooth=1):
    # y_true_f = K.flatten(y_true)
    # y_pred_f = K.flatten(y_pred)
    intersection = K.sum(K.abs(y_true * y_pred))
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou


def get_cosine_Sim(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))
    # y_true_f = K.flatten(y_true)
    # y_pred_f = K.flatten(y_pred)
    # normalize_a = tf.nn.l2_normalize(y_true, axis=-1)
    # normalize_b = tf.nn.l2_normalize(y_pred, axis=-1)
    # cos_similarity = tf.reduce_sum(tf.multiply(normalize_a, normalize_b))
    # return cos_similarity


def get_cosine_Sim_loss(y_true, y_pred):
    return 1 - get_cosine_Sim(y_true, y_pred)


def correlation_coefficient_loss(y_true, y_pred):
    X = y_true
    Y = y_pred
    mx = K.mean(X)
    my = K.mean(Y)
    xm, ym = X - mx, Y - my
    r_num = K.sum(tf.multiply(xm, ym))
    r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
    r = r_num / r_den

    r = K.maximum(K.minimum(r, 1.0), -1.0)
    return K.square(r)


def get_mse(y_true, y_pred):
    return tf.keras.losses.MSE(y_true, y_pred)


def get_correlation(y_true, y_pred):
    return tf.contrib.metrics.streaming_pearson_correlation(y_pred, y_true)


def to_Fourier(inputs_model):
    # Fourier fast transform
    input_FFT_x = tf.signal.fft(tf.cast(inputs_model, dtype=tf.complex64))
    # input_FFT_x = tf.signal.fftshift(input_FFT_x)
    input_to_AE = tf.cast(input_FFT_x, dtype=tf.float32)
    return input_to_AE


def Fourier_to_time(inputs_):
    inputs_ = tf.cast(inputs_, dtype=tf.complex64)
    # input_FFT_x = tf.signal.ifftshift(inputs_)
    input_FFT_x = tf.signal.ifft(inputs_)
    input_to_AE = tf.cast(input_FFT_x, dtype=tf.float32)
    return input_to_AE


########################################################################################################################
# Encoder
########################################################################################################################
inputs = tf.keras.layers.Input(shape=(new_dim, new_dim, channels))

x = get_timedistributed_conv_encoder(inputs, 64, 4, 2, 'encoder_depth0_conv1')  # 512
x_1 = tf.keras.layers.Activation('relu')(x)

x = get_timedistributed_conv_encoder(x_1, 128, 4, 2, 'encoder_depth1_conv1')  # 256
x = tf.keras.layers.BatchNormalization(name='depth_1')(x)
x_2 = tf.keras.layers.Activation('relu')(x)

x = get_timedistributed_conv_encoder(x_2, 256, 4, 2, 'encoder_depth2_conv1')  # 128
x = tf.keras.layers.BatchNormalization(name='depth_2')(x)
x_3 = tf.keras.layers.Activation('relu')(x)

x = get_timedistributed_conv_encoder(x_3, 512, 4, 2, 'encoder_depth3_conv1')  # 64
x = tf.keras.layers.BatchNormalization(name='depth_3')(x)
x_4 = tf.keras.layers.Activation('relu')(x)

x = get_timedistributed_conv_encoder(x_4, 512, 4, 2, 'encoder_depth4_conv1')  # 32
x = tf.keras.layers.BatchNormalization(name='depth_4')(x)
x_5 = tf.keras.layers.Activation('relu')(x)

x = get_timedistributed_conv_encoder(x_5, 512, 4, 2, 'encoder_depth5_conv1')  # 16
x = tf.keras.layers.BatchNormalization(name='depth_5')(x)
x_6 = tf.keras.layers.Activation('relu')(x)

x = get_timedistributed_conv_encoder(x_6, 512, 4, 2, 'encoder_depth6_conv1')  # 8
x = tf.keras.layers.BatchNormalization(name='depth_6')(x)
x = tf.keras.layers.Dropout(0.5)(x)
x_7 = tf.keras.layers.Activation('relu')(x)

x = get_timedistributed_conv_encoder(x_7, 512, 4, 2, 'encoder_depth7_conv1')  # 4
x = tf.keras.layers.BatchNormalization(name='depth_7')(x)
x_8 = tf.keras.layers.Activation('relu')(x)

x = get_timedistributed_conv_encoder(x_8, 512, 4, 2, 'encoder_depth8_conv1')  # 2
x = tf.keras.layers.BatchNormalization(name='depth_8')(x)
x_9 = tf.keras.layers.Activation('relu')(x)

x = get_timedistributed_conv_encoder(x_9, 512, 4, 2, 'encoder_depth9_conv1')  # 1
x = tf.keras.layers.BatchNormalization(name='depth_9')(x)
x_10 = tf.keras.layers.Activation('relu')(x)

########################################################################################################################
# Bottleneck
########################################################################################################################
encoded_output = x_10  # tf.keras.layers.BatchNormalization(name='depth_6')(x)
########################################################################################################################
# Decoder
########################################################################################################################
decoder_inputs = encoded_output

x = tf.keras.layers.UpSampling2D(2)(decoder_inputs)
x = tf.keras.layers.concatenate([x, x_8], axis=-1)
x = get_timedistributed_conv_encoder(x, 512, 3, 1, 'decoder_depth9_conv1')  # 2
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Activation('relu')(x)

x = tf.keras.layers.concatenate([x, x_8], axis=-1)
x = tf.keras.layers.UpSampling2D(2)(x)
x = get_timedistributed_conv_encoder(x, 512, 3, 1, 'decoder_depth8_conv1')  # 4
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Activation('relu')(x)

x = tf.keras.layers.concatenate([x, x_7], axis=-1)
x = tf.keras.layers.UpSampling2D(2)(x)
x = get_timedistributed_conv_encoder(x, 512, 3, 1, 'decoder_depth7_conv1')  # 8
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Activation('relu')(x)

x = tf.keras.layers.concatenate([x, x_6], axis=-1)
x = tf.keras.layers.UpSampling2D(2)(x)
x = get_timedistributed_conv_encoder(x, 512, 3, 1, 'decoder_depth6_conv1')  # 16
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Activation('relu')(x)

x = tf.keras.layers.concatenate([x, x_5], axis=-1)
x = tf.keras.layers.UpSampling2D(2)(x)
x = get_timedistributed_conv_encoder(x, 512, 3, 1, 'decoder_depth5_conv1')  # 32
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Activation('relu')(x)

x = tf.keras.layers.concatenate([x, x_4], axis=-1)
x = tf.keras.layers.UpSampling2D(2)(x)
x = get_timedistributed_conv_encoder(x, 256, 3, 1, 'decoder_depth4_conv1')  # 64
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Activation('relu')(x)

x = tf.keras.layers.concatenate([x, x_3], axis=-1)
x = tf.keras.layers.UpSampling2D(2)(x)
x = get_timedistributed_conv_encoder(x, 128, 3, 1, 'decoder_depth3_conv1')  # 128
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Activation('relu')(x)

x = tf.keras.layers.concatenate([x, x_2], axis=-1)
x = tf.keras.layers.UpSampling2D(2)(x)
x = get_timedistributed_conv_encoder(x, 64, 3, 1, 'decoder_depth2_conv1')  # 256
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Activation('relu')(x)

x = tf.keras.layers.concatenate([x, x_1], axis=-1)
x = tf.keras.layers.UpSampling2D(2)(x)
x = get_timedistributed_conv_encoder(x, 1, 1, 1, 'decoder_depth1_conv1')  # 512
x = tf.keras.layers.BatchNormalization()(x)
decoded = tf.keras.layers.Activation('sigmoid')(x)

generator = tf.keras.Model(inputs=inputs, outputs=decoded, name='Encoder_decoder')
generator.summary()
########################################################################################################################
# Building Discriminator model
########################################################################################################################

initial_learning_rate = 3e-4
epochs = 100
decay = initial_learning_rate / epochs


def lr_time_based_decay(epoch, lr):
    return lr * 1 / (1 + decay * epoch)


def step_decay(epoch):
    initial_l_rate = 3e-4
    drop = 0.5
    epochs_drop = 10.0
    l_rate = initial_l_rate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return l_rate


# Define configuration parameters
start_lr = 2e-4
ram_pup_epochs = 10
exp_decay = 0.01


# Define the scheduling function
def schedule(epoch):
    def lr(epoch_, start_lr_, rampup__epochs_, exp_decay_):
        if epoch_ < rampup__epochs_:
            return start_lr_
        else:
            return start_lr_ * math.exp(-exp_decay_ * epoch_)

    return lr(epoch, start_lr, ram_pup_epochs, exp_decay)


# callback = tf.keras.callbacks.LearningRateScheduler(lr_time_based_decay, verbose=1)
# lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate=initial_learning_rate,
#     decay_steps=10000,
#     decay_rate=0.9
# )

########################################################################################################################

VAE_model = Model(inputs=inputs, outputs=generator(inputs), name='Variational_AE')

VAE_model.summary()

VAE_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4),
                  loss=tf.keras.losses.MSE,
                  metrics=[get_cosine_Sim],  # [correlation_coefficient_loss],
                  # experimental_run_tf_function=True
                  )

VAE_model.fit(train_dataset,
              validation_data=val_dataset,
              epochs=epochs,
              callbacks=[tf.keras.callbacks.LearningRateScheduler(schedule, verbose=True)],
              )

VAE_model.evaluate(test_dataset)

os.chdir('/home/aijjeh/Desktop/Phd_Project/GT_RMS_waves/h5_models')
VAE_model.save('seq_2_seq_prediction_all_frames_window_32_signals_augmented_%d.h5' % epochs)
