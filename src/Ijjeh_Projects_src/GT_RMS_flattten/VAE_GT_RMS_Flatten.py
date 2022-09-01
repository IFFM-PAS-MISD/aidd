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
import neptune
from decouple import config
from keras.callbacks import Callback

########################################################################################################################
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
print(device_lib.list_local_devices())
########################################################################################################################
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

########################################################################################################################
# Link to neptune ai for monitoring
########################################################################################################################
run = neptune.init(project_qualified_name='abdalraheem.ijjeh/aidd',
                   api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJhZGJjNTcyNC0yN2ViLTQ5YzctOGFkZC1jODNlZmU1Y2Q4ZDcifQ==')
neptune.create_experiment('Variational_AutoEncoder_with_FFT')
neptune.append_tag('VAE_FFT')
########################################################################################################################
# Scheduler
########################################################################################################################
# Define configuration parameters
start_lr = 3e-4
ram_pup_epochs = 20
exp_decay = 0.01


# class FConv2D(tf.keras.layers.Layer):
#     def __init__(self, no_of_kernels, kernel_shape, **kwargs):
#         self.no_of_kernels = no_of_kernels
#         self.kernel_shape = kernel_shape
#         super(FConv2D, self).__init__(**kwargs)
#
#     def build(self, input_shape):
#         self.kernel = self.add_weight(shape=self.kernel_shape + (input_shape[-1], self.no_of_kernels),
#                                       initializer='random_normal',
#                                       trainable=True)
#         self.bias = self.add_weight(shape=(self.no_of_kernels,),
#                                     initializer='random_normal',
#                                     trainable=True)
#         super(FConv2D, self).build(input_shape)
#
#     def call(self, x):
#         crop_size = self.kernel.get_shape().as_list()[0] // 2
#         shape = x.get_shape().as_list()[1] + self.kernel.get_shape().as_list()[0] - 1
#         X = tf.transpose(x, perm=[0, 3, 1, 2])
#         W = tf.transpose(self.kernel, perm=[3, 2, 0, 1])
#         X = tf.signal.rfft2d(X, [shape, shape])
#         W = tf.signal.rfft2d(W, [shape, shape])
#         X = tf.einsum('imkl,jmkl->ijkl', X, W)
#         output = tf.signal.irfft2d(X, [shape, shape])
#         output = tf.transpose(output, perm=[0, 2, 3, 1])
#         output = tf.nn.bias_add(output, self.bias)[:, crop_size:-1 * crop_size, crop_size:-1 * crop_size, :]
#         return output


# Define the scheduling function
def schedule(epoch):
    def lr(epoch_, start_lr_, rampup__epochs_, exp_decay_):
        if epoch_ < rampup__epochs_:
            return start_lr_
        else:
            return start_lr_ * math.exp(-exp_decay_ * epoch_)

    return lr(epoch, start_lr, ram_pup_epochs, exp_decay)


def get_cosine_Sim(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0), axis=-1)


def to_FFT(x):
    x = tf.cast(x, tf.float32)
    x = tf.signal.rfft(x)
    x = tf.cast(x, tf.float32)
    return x


def to_inverse_FFT(x):
    x = tf.cast(x, tf.complex64)
    x = tf.signal.irfft(x)
    x = tf.cast(x, tf.float32)
    return x


def to_short_time_FT(x):
    x = tf.dtypes.cast(x, tf.float32)
    stft_var = tf.signal.stft(x,
                              frame_length=2,
                              frame_step=1,
                              pad_end=False
                              )
    return tf.dtypes.cast(stft_var, tf.float32)


def to_inverse_short_time_FT(x):
    x = tf.cast(x, tf.complex64)
    istft = tf.signal.inverse_stft(
        x,
        frame_length=2,
        frame_step=1,
    )
    return istft


########################################################################################################################
# Sampling function
########################################################################################################################
def sampling(mu_log_variance):
    mu, log_variance = mu_log_variance
    epsilon = tf.keras.backend.random_normal(shape=tf.keras.backend.shape(mu), mean=0.0, stddev=1.0)
    random_sample = mu + tf.keras.backend.exp(0.5 * log_variance) * epsilon
    return random_sample


########################################################################################################################
# VAE loss function
########################################################################################################################
def loss_func(encoder_mu, encoder_log_variance):
    def vae_reconstruction_loss(y_true, y_predict):
        alpha = 512 * 512
        reconstruction_loss = K.sqrt(K.mean(K.square(y_true - y_predict), axis=-1))
        return reconstruction_loss * alpha

    def vae_kl_loss(encoder_mu, encoder_log_variance):  #
        kl_loss = -0.5 * K.sum(1.0 + encoder_log_variance - K.square(encoder_mu) - K.exp(encoder_log_variance),
                               axis=-1)

        return kl_loss

    def vae_kl_loss_metric(y_true, y_predict):
        kl_loss = -0.5 * K.sum(1.0 + encoder_log_variance - K.square(encoder_mu) - K.exp(encoder_log_variance), axis=-1)
        return kl_loss

    def vae_loss(y_true, y_predict):
        reconstruction_loss = vae_reconstruction_loss(y_true, y_predict)
        kl_loss = vae_kl_loss(y_true, y_predict)  #
        loss = reconstruction_loss + kl_loss
        return loss

    return vae_loss


def get_iou_metric(y_true, y_pred, smooth=100):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(K.abs(y_true_f * y_pred_f))
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou


def get_conv_block(x, l_filters, l_f_size, l_strides, l_name):
    layer_x = tf.keras.layers.Conv2D(filters=l_filters,
                                     kernel_size=l_f_size,
                                     padding="same",
                                     strides=l_strides,
                                     name=l_name)(x)
    layer_x = tf.keras.layers.BatchNormalization()(layer_x)
    layer_x = tf.keras.layers.Activation('relu')(layer_x)
    return layer_x


def get_TransConv_block(x, l_filters, l_f_size, l_strides, l_name):
    layer_x = tf.keras.layers.Conv2DTranspose(filters=l_filters,
                                              kernel_size=l_f_size,
                                              padding="same",
                                              strides=l_strides,
                                              name=l_name)(x)
    layer_x = tf.keras.layers.BatchNormalization()(layer_x)
    layer_x = tf.keras.layers.Activation('relu')(layer_x)
    return layer_x


def correlation_coefficient_loss(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x - mx, y - my
    r_num = K.sum(tf.multiply(xm, ym))
    r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
    r = r_num / r_den
    r = K.maximum(K.minimum(r, 1.0), -1.0)
    return 1 - K.square(r)


def correlation_coefficient_metric(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x - mx, y - my
    r_num = K.sum(tf.multiply(xm, ym))
    r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
    r = r_num / r_den
    r = K.maximum(K.minimum(r, 1.0), -1.0)
    return K.square(r)


def root_mean_square_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def build_VAE():
    ####################################################################################################################
    #  Encoder
    ####################################################################################################################
    inputs = tf.keras.layers.Input(shape=(height, width, channels), name="encoder_input")
    # l_fft = tf.keras.layers.Lambda(
    #     lambda v: tf.cast(tf.compat.v1.spectral.fft(tf.cast(v, dtype=tf.complex64)), tf.float32))(inputs)
    encoder_l = get_TransConv_block(inputs, 64, 3, 1, "encoder_conv_inputs_1")
    for i in range(depth):
        encoder_l = get_conv_block(encoder_l, 64, 3, 2, 'encoder_conv_%d' % (i + 1))
        encoder_l = get_conv_block(encoder_l, 64, 3, 1, 'encoder_conv_%d_%d' % (i + 1, i + 1))

    encoder_l = get_conv_block(encoder_l, 64, 3, 1, 'encoder_conv_output')
    shape_before_flatten = tf.keras.backend.int_shape(encoder_l)[1:]
    encoder_flatten = tf.keras.layers.Flatten()(encoder_l)
    # encoder = tf.keras.models.Model(inputs, encoder_l, name="encoder_model")
    # encoder.summary()
    ####################################################################################################################
    #  latent space
    ####################################################################################################################
    ########################################################################################################################
    # latent_space_dim = tf.keras.layers.Input(shape=shape_before_flatten)
    # latent_flatten = tf.keras.layers.Flatten()(latent_space_dim)
    latent_space_dim = 2
    encoder_mu = tf.keras.layers.Dense(units=latent_space_dim, name="encoder_mu")(encoder_flatten)
    encoder_log_variance = tf.keras.layers.Dense(units=latent_space_dim, name="encoder_log_variance")(encoder_flatten)
    ######################################################################################################################
    encoder_output = tf.keras.layers.Lambda(sampling,
                                            name="encoder_output", )([encoder_mu, encoder_log_variance])
    encoder = tf.keras.models.Model(inputs, [encoder_mu, encoder_log_variance, encoder_output], name="encoder_model")
    encoder.summary()
    # ########################################################################################################################
    # latent_inputs = tf.keras.layers.Input(shape=shape_before_flatten)
    # latent_flatten = tf.keras.layers.Flatten()(latent_inputs)
    # latent_dense1 = tf.keras.layers.Dense(units=np.prod(shape_before_flatten))(latent_flatten)
    # latent_dense2 = tf.keras.layers.Dropout(0.5)(latent_dense1)
    # latent_dense3 = tf.keras.layers.Dense(units=np.prod(shape_before_flatten))(latent_dense2)
    # latent_reshape = tf.keras.layers.Reshape(target_shape=shape_before_flatten)(latent_dense3)
    # latent_model = tf.keras.Model(latent_inputs, latent_reshape, name='latent_space_model')
    # latent_model.summary()

    ####################################################################################################################
    #  Decoder
    ####################################################################################################################
    decoder_input = tf.keras.layers.Input(shape=latent_space_dim, name="decoder_input")

    decoder_dense_layer1 = tf.keras.layers.Dense(units=np.prod(shape_before_flatten),
                                                 name="decoder_dense_1")(decoder_input)

    decoder_reshape = tf.keras.layers.Reshape(target_shape=shape_before_flatten)(decoder_dense_layer1)

    # decoder_input = tf.keras.layers.Input(shape=shape_before_flatten, name="decoder_input")
    decoder_l = get_TransConv_block(decoder_reshape, 64, 3, 1, 'decoder_conv_tran_inputs_0')
    for i in range(depth, 0, -1):
        decoder_l = get_TransConv_block(decoder_l, 64, 3, 2, 'decoder_conv_tran_%d' % (i + 1))
        decoder_l = get_TransConv_block(decoder_l, 64, 3, 1, 'decoder_conv_tran_%d_%d' % (i + 1, i + 1))

    decoder_l = tf.keras.layers.Conv2DTranspose(filters=64,
                                                kernel_size=1,
                                                padding="same",
                                                strides=1,
                                                name="decoder_conv_tran_output")(decoder_l)
    decoder_output = tf.keras.layers.Activation('sigmoid')(decoder_l)

    decoder = tf.keras.models.Model(decoder_input, decoder_output, name="decoder_model")
    decoder.summary()
    ####################################################################################################################
    # VAE model
    ####################################################################################################################
    vae_input = tf.keras.layers.Input(shape=(height, width, channels), name="VAE_input")
    vae_output = decoder(encoder(vae_input)[2])
    vae_output = tf.transpose(vae_output, perm=[0, 3, 1, 2])
    # vae_output = tf.keras.layers.Lambda(
    #     lambda v: tf.cast(tf.compat.v1.spectral.ifft(tf.cast(v, dtype=tf.complex64)), tf.float32))(vae_output)
    VAE_model = tf.keras.models.Model(vae_input, vae_output, name="VAE")
    VAE_model.summary()
    VAE_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=start_lr),
                      loss=loss_func(encoder_mu, encoder_log_variance),
                      metrics=[get_cosine_Sim]
                      )

    class MonitoringCallback(Callback):
        def on_epoch_end(self, epoch, logs={}):
            for metric_name, metric_value in logs.items():
                neptune.log_metric(metric_name, metric_value)

    VAE_model.fit(train_dataset,
                  validation_data=val_dataset,
                  epochs=epochs,
                  callbacks=[MonitoringCallback(), tf.keras.callbacks.LearningRateScheduler(schedule, verbose=True)],
                  )
    VAE_model.evaluate(test_dataset)
    os.chdir('/home/aijjeh/Desktop/Phd_Projects/Sequence_prediction/Sequence_to_sequence/h5_models')

    encoder.save("VAE_encoder_seq2seq_%d_new.h5" % epochs)
    decoder.save("VAE_decoder__seq2seq_%d_new.h5" % epochs)
    VAE_model.save('VAE_seq2seq_%d_new.h5' % epochs)
    return VAE_model


########################################################################################################################
# Main
########################################################################################################################
if __name__ == '__main__':
    ####################################################################################################################
    # Load dataset
    ####################################################################################################################
    # dataset_path = '/home/aijjeh/Desktop/Phd_Project/GT_RMS_waves/Dataset/'
    # os.chdir(dataset_path)
    # dataset_x_GT = np.load('GT2RMS2waves_Training_x_thresholded_augmented_h_v_d_RGB.npy', mmap_mode='r')
    # dataset_y_RMS = np.load('GT2RMS2waves_Labels_y_augmented_h_v_d_RGB.npy', mmap_mode='r')
    # dataset_x_GT = np.reshape(dataset_x_GT,
    #                           (dataset_x_GT.shape[0],
    #                            dataset_x_GT.shape[1] * dataset_x_GT.shape[2],
    #                            dataset_x_GT.shape[3]))
    # dataset_y_RMS = np.reshape(dataset_y_RMS,
    #                            (dataset_y_RMS.shape[0],
    #                             dataset_y_RMS.shape[1] * dataset_y_RMS.shape[2],
    #                             dataset_y_RMS.shape[3]))
    ####################################################################################################################
    ####################################################################################################################
    # Load dataset
    ####################################################################################################################
    dataset_path = '/home/aijjeh/Desktop/Phd_Projects/Sequence_prediction/Sequence_to_sequence/Dataset_Full_wavefield_outputs_bottom/'
    os.chdir(dataset_path)
    dataset_x_GT = np.load('Refrence_waves_GT_64_frame_475_512_512_65.npy', mmap_mode='r+')
    dataset_y_RMS = np.load('Full_wavefield_labels_64_frame_475_64_512_512.npy', mmap_mode='r+')

    x_train = dataset_x_GT[:304]
    y_train = dataset_y_RMS[:304]
    x_val = dataset_x_GT[304:380]
    y_val = dataset_y_RMS[304:380]
    x_test = dataset_x_GT[380:]
    y_test = dataset_y_RMS[380:]

    print(dataset_x_GT.shape)
    print(dataset_y_RMS.shape)

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    BATCH_SIZE = 1
    train_dataset = train_dataset.batch(BATCH_SIZE)
    val_dataset = val_dataset.batch(BATCH_SIZE)
    test_dataset = test_dataset.batch(BATCH_SIZE)
    ####################################################################################################################
    # Parameters
    ####################################################################################################################
    epochs = 600
    filters = 64
    f_size = 3
    channels = dataset_x_GT.shape[-1]
    height = dataset_x_GT.shape[1]
    width = dataset_x_GT.shape[2]
    depth = 5
    ####################################################################################################################
    # calling the VAE
    ####################################################################################################################
    create_var = build_VAE()