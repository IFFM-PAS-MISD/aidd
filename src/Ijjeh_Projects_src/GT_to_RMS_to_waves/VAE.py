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
from tensorflow.keras import backend as K
import natsort
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from scipy import signal
from sewar.full_ref import mse, rmse, psnr, uqi, ssim, ergas, scc, rase, sam, msssim, vifp

# from tensorflow.python.framework.ops import disable_eager_execution
# tf.compat.v1.disable_eager_execution()

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
dataset_x = np.load('GT2RMS2waves_Training_x_thresholded_augmented_h_v_d_RGB.npy', mmap_mode='r')
# dataset_x = np.load('GT2RMS2waves_Training_x_thresholded_diff_array.npy', mmap_mode='r')
# dataset_y = np.load('GT2RMS2waves_Labels_y.npy', mmap_mode='r')
dataset_y = np.load('GT2RMS2waves_Labels_y_augmented_h_v_d_RGB.npy', mmap_mode='r')

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


def sampling(mu_log_variance):
    mu, log_variance = mu_log_variance
    epsilon = tf.keras.backend.random_normal(shape=tf.keras.backend.shape(mu), mean=0.0, stddev=1.0)
    random_sample = mu + tf.keras.backend.exp(0.5 * log_variance) * epsilon
    return random_sample


inputs = tf.keras.layers.Input(shape=(new_dim, new_dim, channels), name="encoder_input")
########################################################################################################################
encoder_conv_layer1 = tf.keras.layers.Conv2D(filters=1,
                                             kernel_size=(3, 3),
                                             padding="same",
                                             strides=1,
                                             name="encoder_conv_1")(inputs)
encoder_norm_layer1 = tf.keras.layers.BatchNormalization(name="encoder_norm_1")(encoder_conv_layer1)
encoder_activ_layer1 = tf.keras.layers.Activation('relu')(encoder_norm_layer1)  #
########################################################################################################################
encoder_conv_layer2 = tf.keras.layers.Conv2D(filters=32,
                                             kernel_size=(3, 3),
                                             padding="same",
                                             strides=1,
                                             name="encoder_conv_2")(encoder_activ_layer1)
encoder_norm_layer2 = tf.keras.layers.BatchNormalization(name="encoder_norm_2")(encoder_conv_layer2)
encoder_activ_layer2 = tf.keras.layers.Activation('relu')(encoder_norm_layer2)
########################################################################################################################
encoder_conv_layer3 = tf.keras.layers.Conv2D(filters=64,
                                             kernel_size=(3, 3),
                                             padding="same",
                                             strides=2,
                                             name="encoder_conv_3")(encoder_activ_layer2)
encoder_norm_layer3 = tf.keras.layers.BatchNormalization(name="encoder_norm_3")(encoder_conv_layer3)
encoder_activ_layer3 = tf.keras.layers.Activation('relu')(encoder_norm_layer3)
########################################################################################################################
encoder_conv_layer4 = tf.keras.layers.Conv2D(filters=64,
                                             kernel_size=(3, 3),
                                             padding="same",
                                             strides=2,
                                             name="encoder_conv_4")(encoder_activ_layer3)
encoder_norm_layer4 = tf.keras.layers.BatchNormalization(name="encoder_norm_4")(encoder_conv_layer4)
encoder_activ_layer4 = tf.keras.layers.Activation('relu')(encoder_norm_layer4)
########################################################################################################################
encoder_conv_layer5 = tf.keras.layers.Conv2D(filters=64,
                                             kernel_size=(3, 3),
                                             padding="same",
                                             strides=2,
                                             name="encoder_conv_5")(encoder_activ_layer4)
encoder_norm_layer5 = tf.keras.layers.BatchNormalization(name="encoder_norm_5")(encoder_conv_layer5)
encoder_activ_layer5 = tf.keras.layers.Activation('relu')(encoder_norm_layer5)
########################################################################################################################
encoder_conv_layer6 = tf.keras.layers.Conv2D(filters=64,
                                             kernel_size=(3, 3),
                                             padding="same",
                                             strides=2,
                                             name="encoder_conv_6")(encoder_activ_layer5)
encoder_norm_layer6 = tf.keras.layers.BatchNormalization(name="encoder_norm_6")(encoder_conv_layer6)
encoder_activ_layer6 = tf.keras.layers.Activation('relu')(encoder_norm_layer6)
########################################################################################################################
encoder_conv_layer7 = tf.keras.layers.Conv2D(filters=64,
                                             kernel_size=(3, 3),
                                             padding="same",
                                             strides=2,
                                             name="encoder_conv_7")(encoder_activ_layer6)
encoder_norm_layer7 = tf.keras.layers.BatchNormalization(name="encoder_norm_7")(encoder_conv_layer7)
encoder_activ_layer7 = tf.keras.layers.Activation('relu')(encoder_norm_layer7)
########################################################################################################################
encoder_conv_layer8 = tf.keras.layers.Conv2D(filters=64,
                                             kernel_size=(3, 3),
                                             padding="same",
                                             strides=2,
                                             name="encoder_conv_8")(encoder_activ_layer7)
encoder_norm_layer8 = tf.keras.layers.BatchNormalization(name="encoder_norm_8")(encoder_conv_layer8)
encoder_activ_layer8 = tf.keras.layers.Activation('relu')(encoder_norm_layer8)
########################################################################################################################
encoder_conv_layer9 = tf.keras.layers.Conv2D(filters=64,
                                             kernel_size=(3, 3),
                                             padding="same",
                                             strides=1,
                                             name="encoder_conv_9")(encoder_activ_layer8)
encoder_norm_layer9 = tf.keras.layers.BatchNormalization(name="encoder_norm_9")(encoder_conv_layer9)
encoder_activ_layer9 = tf.keras.layers.Activation('relu')(encoder_norm_layer9)
########################################################################################################################
shape_before_flatten = tf.keras.backend.int_shape(encoder_activ_layer9)[1:]
encoder_flatten = tf.keras.layers.Flatten()(encoder_activ_layer9)

print(shape_before_flatten)
########################################################################################################################
latent_space_dim = 2
encoder_mu = tf.keras.layers.Dense(units=latent_space_dim, name="encoder_mu")(encoder_flatten)
encoder_log_variance = tf.keras.layers.Dense(units=latent_space_dim, name="encoder_log_variance")(encoder_flatten)
########################################################################################################################
encoder_output = tf.keras.layers.Lambda(sampling,
                                        name="encoder_output",
                                        # output_shape=(latent_space_dim,)
                                        )([encoder_mu, encoder_log_variance])
encoder = tf.keras.models.Model(inputs, [encoder_mu, encoder_log_variance, encoder_output], name="encoder_model")
encoder.summary()
########################################################################################################################
#  Decoder
########################################################################################################################
decoder_input = tf.keras.layers.Input(shape=(latent_space_dim,), name="decoder_input")

decoder_dense_layer1 = tf.keras.layers.Dense(units=np.prod(shape_before_flatten),
                                             name="decoder_dense_1")(decoder_input)

decoder_reshape = tf.keras.layers.Reshape(target_shape=shape_before_flatten)(decoder_dense_layer1)
########################################################################################################################
decoder_conv_tran_layer1 = tf.keras.layers.Conv2DTranspose(filters=64,
                                                           kernel_size=(3, 3),
                                                           padding="same",
                                                           strides=1,
                                                           name="decoder_conv_tran_1")(decoder_reshape)
decoder_norm_layer1 = tf.keras.layers.BatchNormalization(name="decoder_norm_1")(decoder_conv_tran_layer1)
decoder_activ_layer1 = tf.keras.layers.Activation('relu')(decoder_norm_layer1)
########################################################################################################################
decoder_conv_tran_layer2 = tf.keras.layers.Conv2DTranspose(filters=64,
                                                           kernel_size=(3, 3),
                                                           padding="same",
                                                           strides=2,
                                                           name="decoder_conv_tran_2")(decoder_activ_layer1)
decoder_norm_layer2 = tf.keras.layers.BatchNormalization(name="decoder_norm_2")(decoder_conv_tran_layer2)
decoder_activ_layer2 = tf.keras.layers.Activation('relu')(decoder_norm_layer2)
########################################################################################################################
decoder_conv_tran_layer3 = tf.keras.layers.Conv2DTranspose(filters=64,
                                                           kernel_size=(3, 3),
                                                           padding="same",
                                                           strides=2,
                                                           name="decoder_conv_tran_3")(decoder_activ_layer2)
decoder_norm_layer3 = tf.keras.layers.BatchNormalization(name="decoder_norm_3")(decoder_conv_tran_layer3)
decoder_activ_layer3 = tf.keras.layers.Activation('relu')(decoder_norm_layer3)
########################################################################################################################
decoder_conv_tran_layer4 = tf.keras.layers.Conv2DTranspose(filters=64,
                                                           kernel_size=(3, 3),
                                                           padding="same",
                                                           strides=2,
                                                           name="decoder_conv_tran_4")(decoder_activ_layer3)
decoder_norm_layer4 = tf.keras.layers.BatchNormalization(name="decoder_norm_4")(decoder_conv_tran_layer4)
decoder_activ_layer4 = tf.keras.layers.Activation('relu')(decoder_norm_layer4)
########################################################################################################################
decoder_conv_tran_layer5 = tf.keras.layers.Conv2DTranspose(filters=64,
                                                           kernel_size=(3, 3),
                                                           padding="same",
                                                           strides=2,
                                                           name="decoder_conv_tran_5")(decoder_activ_layer4)
decoder_norm_layer5 = tf.keras.layers.BatchNormalization(name="decoder_norm_5")(decoder_conv_tran_layer5)
decoder_activ_layer5 = tf.keras.layers.Activation('relu')(decoder_norm_layer5)
########################################################################################################################
decoder_conv_tran_layer6 = tf.keras.layers.Conv2DTranspose(filters=64,
                                                           kernel_size=(3, 3),
                                                           padding="same",
                                                           strides=2,
                                                           name="decoder_conv_tran_6")(decoder_activ_layer5)
decoder_norm_layer6 = tf.keras.layers.BatchNormalization(name="decoder_norm_6")(decoder_conv_tran_layer6)
decoder_activ_layer6 = tf.keras.layers.Activation('relu')(decoder_norm_layer6)
########################################################################################################################
decoder_conv_tran_layer7 = tf.keras.layers.Conv2DTranspose(filters=64,
                                                           kernel_size=(3, 3),
                                                           padding="same",
                                                           strides=2,
                                                           name="decoder_conv_tran_7")(decoder_activ_layer6)
decoder_norm_layer7 = tf.keras.layers.BatchNormalization(name="decoder_norm_7")(decoder_conv_tran_layer7)
decoder_activ_layer7 = tf.keras.layers.Activation('relu')(decoder_norm_layer7)
########################################################################################################################
decoder_conv_tran_layer8 = tf.keras.layers.Conv2DTranspose(filters=64,
                                                           kernel_size=(3, 3),
                                                           padding="same",
                                                           strides=1,
                                                           name="decoder_conv_tran_8")(decoder_activ_layer7)
decoder_norm_layer8 = tf.keras.layers.BatchNormalization(name="decoder_norm_8")(decoder_conv_tran_layer8)
decoder_activ_layer8 = tf.keras.layers.Activation('relu')(decoder_norm_layer8)
########################################################################################################################
decoder_conv_tran_layer9 = tf.keras.layers.Conv2DTranspose(filters=3,
                                                           kernel_size=(3, 3),
                                                           padding="same",
                                                           strides=1,
                                                           name="decoder_conv_tran_9")(decoder_activ_layer8)
decoder_output = tf.keras.layers.Activation('sigmoid')(decoder_conv_tran_layer9)
shape_before_output = tf.keras.backend.int_shape(decoder_output)[1:]

decoder = tf.keras.models.Model(decoder_input, decoder_output, name="decoder_model")
decoder.summary()

########################################################################################################################
# VAE model
########################################################################################################################

vae_input = tf.keras.layers.Input(shape=(new_dim, new_dim, channels), name="VAE_input")
vae_decoder_output = decoder(encoder(vae_input)[2])

vae = tf.keras.models.Model(vae_input, vae_decoder_output, name="VAE")
vae.summary()


########################################################################################################################


def loss_func(encoder_mu, encoder_log_variance):

    def vae_reconstruction_loss(y_true, y_predict):
        reconstruction_loss_factor = 512*512
        reconstruction_loss = K.mean(K.square(y_true-y_predict), axis=-1)
        return reconstruction_loss_factor * reconstruction_loss

    def vae_kl_loss(encoder_mu, encoder_log_variance):
        kl_loss = -0.5 * K.sum(1.0 + encoder_log_variance - K.square(encoder_mu) - K.exp(encoder_log_variance), axis=-1)
        return kl_loss

    def vae_kl_loss_metric(y_true, y_predict):
        kl_loss = -0.5 * K.sum(1.0 + encoder_log_variance - K.square(encoder_mu) - K.exp(encoder_log_variance), axis=-1)
        return kl_loss

    def vae_loss(y_true, y_predict):
        reconstruction_loss = vae_reconstruction_loss(y_true, y_predict)
        kl_loss = vae_kl_loss(y_true, y_predict)
        loss = reconstruction_loss + kl_loss
        return loss
    return vae_loss


epochs = 50

# Define configuration parameters
start_lr = 3e-4
ram_pup_epochs = 20
exp_decay = 0.005


# Define the scheduling function
def schedule(epoch):
    def lr(epoch_, start_lr_, rampup__epochs_, exp_decay_):
        if epoch_ < rampup__epochs_:
            return start_lr_
        else:
            return start_lr_ * math.exp(-exp_decay_ * epoch_)

    return lr(epoch, start_lr, ram_pup_epochs, exp_decay)


def get_cosine_Sim(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))


vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=start_lr),
            loss=loss_func(encoder_mu, encoder_log_variance),
            metrics=[get_cosine_Sim])

vae.fit(train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=[tf.keras.callbacks.LearningRateScheduler(schedule, verbose=True)],
        )

vae.evaluate(test_dataset)

os.chdir('/home/aijjeh/Desktop/Phd_Project/GT_RMS_waves/h5_models')
encoder.save("VAE_encoder.h5")
decoder.save("VAE_decoder.h5")
vae.save('VAE_prediction_all_frames_window_32_signals_augmented_%d_new.h5' % epochs)
