#!/usr/bin/env python
# coding: utf-8


# ============================================
__title__ = 'Modelling guided waves'
__author__ = "Abdalraheem A. Ijjeh"
__maintainer__ = "Abdalraheem A. Ijjeh"
__email__ = "aijjeh@imp.gda.pl"

# ============================================

import os
import numpy as np
import json
import csv
import tensorflow as tf
import neptune.new as neptune
import matplotlib.pyplot as plt
import PIL
import re
import scipy.fftpack as spfft

from IPython.display import Image, display
from tensorflow.keras.preprocessing.image import load_img
from PIL import ImageOps
from tensorflow.keras.models import Model
from keras import backend as K
from keras.callbacks import Callback
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error
from decouple import config
from dotenv import load_dotenv

load_dotenv()

# # Link to Neptune
access_token = os.getenv('NEPTUNE_API_TOKEN')
run = neptune.init_run(project='abdalraheem.ijjeh/Guided-waves-modelling',
                       api_token=access_token,
                       tags=['DL HR_FFT2D_guided_waves_modelling', 'Fourier_domain']
                       )

# # Hyperparameters

run["Signal_based"] = "AE"
params = {'samples': 15200,
          'n_size': 24320,
          'normalised': True,
          'batches': 16,
          'num_filters': 16,
          'kernel_size': 3,
          'shape': (32, 32, 1),
          'epochs': 1000,
          'dropout': 0.2,
          'levels': 3,
          'learning_rate': 0.00014329,
          'patience_epochs': 100,
          'val_split': 0.08,
          'hidden_layer': 3,
          'decay:steps': 100000,
          'decay_rate': 0.96,
          'time_stamps': 16
          }

run["model/parameters"] = params

# # Run on GPU


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

# # Save the hyperparameters


env_path = '/home/aijjeh/Desktop/Phd_Projects/Modeling_guided_waves/'
os.chdir(env_path)
json = json.dumps(params)
f = open('hyper_par_GWM_mse.json', 'w')
f.write(json)
f.close()

# def atoi(text):
#     return int(text) if text.isdigit() else text
#
#
# def natural_keys(text):
#     """
#     alist.sort(key=natural_keys) sorts in human order
#     http://nedbatchelder.com/blog/200712/human_sorting.html
#     (See Tooth's implementation in the comments)
#     """
#     return [atoi(c) for c in re.split(r'(\d+)', text)]


# input_dir = "/aijjeh_odroid_sensors/aidd/data/raw/num/wavefield_dataset_undelam_bottom_out/1_output"
# target_dir_ = "/aijjeh_odroid_sensors/aidd/data/raw/num/wavefield_dataset2_bottom_out"
# my_list = os.listdir(input_dir)
# my_list.sort(key=natural_keys)
#
# input_img_paths_total = []
# target_img_paths_total = []
#
# for k in range(475):
#     input_img_paths = sorted(
#         [
#             os.path.join(input_dir, f_name)
#             for f_name in os.listdir(input_dir)
#             if f_name.endswith(".png")
#         ]
#     )
#     input_img_paths.sort(key=natural_keys)
#     input_img_paths_total.append(input_img_paths)
#
# for i in range(475):
#     target_dir = target_dir_ + '/%d_output' % (i + 1)  # str(my_list[i])
#     target_img_paths = sorted(
#         [
#             os.path.join(target_dir, f_name)
#             for f_name in os.listdir(target_dir)
#             if f_name.endswith(".png")
#         ]
#     )
#     target_img_paths.sort(key=natural_keys)
#     target_img_paths_total.append(target_img_paths)
#
# os.chdir(
#     '/home/aijjeh/Desktop/Phd_Projects/Sequence_prediction/Full_wavefield_frames_time_series_project/Datasets/label_set')
# file_frame = np.load('frames_initial.npy')
# file_frame = file_frame[:380]
# print(file_frame[0])

os.chdir('/home/aijjeh/Desktop/Phd_Projects/Modeling_guided_waves/dataset')
x_coords = np.load('LR_GT_labels_img.npy')
delam_coords = x_coords.reshape((475, 1, 32, 32)).astype('uint8')
delam_coords = np.repeat(delam_coords, 512, axis=1)
# # print(delam_coords.shape)
# delam_coords = delam_coords.reshape((475, 32, 1, 5))
# delam_coords = np.repeat(delam_coords, 32, axis=2)
# # print(delam_coords.shape)
# delam_coords = delam_coords.reshape((475, 1, 32, 32, 5))
# delam_coords = np.repeat(delam_coords, 512, axis=1)
delam_coords = delam_coords.reshape((475 * 64, 8, 32, 32, 1))
delam_coords = np.invert(delam_coords[:params['n_size']]) - 254

print(np.max(delam_coords))


# print(delam_coords.shape)
# print(np.max(delam_coords))


def load_dataset():
    os.chdir('/home/aijjeh/Desktop/Phd_Projects/Modeling_guided_waves/dataset')

    x_lr_frame = np.load('LR_ref_frames.npy')
    print('LR input frames ', x_lr_frame.shape)
    x_lr_frame = np.reshape(x_lr_frame, (475 * 64, 8, 32, 32, 1))

    y_lr_frame = np.load('LR_labels.npy')
    print('LR GT frames ', y_lr_frame.shape)
    y_lr_frame = np.reshape(y_lr_frame, (475 * 64, 8, 32, 32, 1))

    lr_inputs = x_lr_frame[:params['n_size']]
    lr_labels = y_lr_frame[:params['n_size']]
    return lr_inputs, lr_labels


X1, Y = load_dataset()

print(X1.shape, delam_coords.shape, Y.shape)
"""
Dataset visualisation, to make sure all training samples are consistent
"""


# for i in range(0, 15200):
#     for j in range(16):
#         fig, [ax1, ax7] = plt.subplots(1, 2)
#         plt.title('output_%d' % (i//32 + 1))
#         ax1.imshow(X1[i][j])
#         ax1.axis('off')
#         # ax2.imshow(delam_coords[i, j, :, :, 0])
#         # ax2.axis('off')
#         # ax3.imshow(delam_coords[i, j, :, :, 1])
#         # ax3.axis('off')
#         # ax4.imshow(delam_coords[i, j, :, :, 2])
#         # ax4.axis('off')
#         # ax5.imshow(delam_coords[i, j, :, :, 3])
#         # ax5.axis('off')
#         # ax6.imshow(delam_coords[i, j, :, :, 4])
#         # ax6.axis('off')
#         ax7.imshow(Y[i][j])
#         ax7.axis('off')
#         plt.show()
#
# exit()


# class Full_wavefield_frames(tf.keras.utils.Sequence):
#     def __init__(self, batch_size, img_size, input_imgs_paths_total, target_imgs_paths_total, input_coords,
#                  time_stamps):
#         self.batch_size = batch_size
#         self.img_size = img_size
#         self.input_img_paths_total = input_imgs_paths_total
#         self.input_coord = input_coords
#         self.target_img_paths = target_imgs_paths_total
#         self.time_stamps = time_stamps
#
#     def __len__(self):
#         return len(self.input_img_paths_total) // self.batch_size
#
#     def __getitem__(self, idx):
#         """Returns tuple (input, target) correspond to batch #idx."""
#         z = idx * self.batch_size
#         batch_input_img_paths = self.input_img_paths_total[z:z + self.batch_size]
#         batch_target_img_paths = self.target_img_paths[z:z + self.batch_size]
#         batch_input_coords = self.input_coord[z:z + self.batch_size]
#
#         x1 = np.zeros((self.batch_size,) + (self.time_stamps,) + self.img_size + (1,), dtype="float16")  #
#         x2 = np.zeros((self.batch_size,) + (self.time_stamps,) + self.img_size + (5,), dtype="float16")  #
#
#         x_in = np.zeros((self.batch_size,) + (self.time_stamps,) + self.img_size + (6,), dtype="float16")  #
#
#         for batch_num in range(self.batch_size):
#             batch_input_img_paths = batch_input_img_paths[batch_num][file_frame[z] - 8: file_frame[z] + 8]
#             batch_input_coords = batch_input_coords[batch_num][:]  # file_frame[z] - 8: file_frame[z] + 8
#             count = 0
#             for j, path in enumerate(batch_input_img_paths):
#                 img = load_img(path, target_size=self.img_size, color_mode="grayscale")
#                 img = np.expand_dims(img, 2)
#                 img = img / 255.0
#
#                 x1[batch_num][j] = img
#
#                 coords = delam_coords[batch_num][count]
#                 x2[batch_num][j] = coords
#                 count += 1
#             x_in = np.concatenate([x1, x2], axis=-1)
#             # x[batch_num][j] = x_in
#
#         y = np.zeros((self.batch_size,) + (self.time_stamps,) + self.img_size + (1,), dtype="float16")  #
#
#         for batch_num in range(self.batch_size):
#             batch_target_img_paths = batch_target_img_paths[batch_num][file_frame[z] - 8:file_frame[z] + 8]
#             for j, path in enumerate(batch_target_img_paths):
#                 img = load_img(path, target_size=self.img_size, color_mode="grayscale")
#                 img = np.expand_dims(img, 2)
#                 img = img / 255.0
#                 y[batch_num][j] = img
#
#         return x_in, y


# # Split our img paths into a training and a validation set


# train_input_img_paths = input_img_paths_total[:304]
# print(len(train_input_img_paths))
# train_target_img_paths = target_img_paths_total[:304]
# print(train_target_img_paths[0][0])
# val_input_img_paths = input_img_paths_total[304:380]
# print(len(val_input_img_paths))
# val_target_img_paths = target_img_paths_total[304:380]
# print(len(val_target_img_paths))


# # Instantiate data Sequences for each split

# train_gen = Full_wavefield_frames(params['batches'],
#                                   params['shape'],
#                                   train_input_img_paths,
#                                   train_target_img_paths,
#                                   arr[:304],
#                                   params['time_stamps'])
#
# val_gen = Full_wavefield_frames(params['batches'],
#                                 params['shape'],
#                                 val_input_img_paths,
#                                 val_target_img_paths,
#                                 arr[304:380],
#                                 params['time_stamps'])
#
# print(type(train_gen))
#
# print(len(train_gen))
#
# print('1st axis', len(train_gen[0]))


def get_time_distributed(time_input, n_filters):
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(filters=n_filters,
                                                               kernel_size=params['kernel_size'],
                                                               strides=1,
                                                               padding='same',
                                                               activation='relu'))(time_input)
    return x


def normalize_batch(bn_input):
    return tf.keras.layers.BatchNormalization()(bn_input)


modes = [1]


def custom_loss_FFT2D(y_true, y_pred):
    """
    :param y_true:
    :param y_pred:
    :return: mse of the fourier domain + mse of the spatial domain
    """

    """
    Transpose y_true and y_pred from shape [batches, time stamps, x-dim, y-dim, features] to 
    [batches, time stamps, features, x-dim, y-dim]
    """
    y_true = tf.transpose(y_true, [0, 1, 4, 2, 3])
    y_pred = tf.transpose(y_pred, [0, 1, 4, 2, 3])

    fft2d_true = tf.signal.fft2d(tf.cast(y_true, dtype=tf.complex64))
    fft2d_pred = tf.signal.fft2d(tf.cast(y_pred, dtype=tf.complex64))

    temp_fft2d_true = fft2d_true
    temp_fft2d_pred = fft2d_pred

    # fft2d_true = tf.where(abs(temp_fft2d_true) < tf.reduce_max(abs(temp_fft2d_true)),
    #                       abs(temp_fft2d_true),
    #                       tf.zeros_like(abs(temp_fft2d_true)))
    # 
    # fft2d_pred = tf.where(abs(temp_fft2d_pred) < tf.reduce_max(abs(temp_fft2d_pred)),
    #                       abs(temp_fft2d_pred),
    #                       tf.zeros_like(abs(temp_fft2d_pred)))

    fft2d_shift_true = tf.signal.fftshift(fft2d_true, axes=(-2, -1))
    fft2d_shift_pred = tf.signal.fftshift(fft2d_pred, axes=(-2, -1))

    fft2d_shift_true = tf.cast(fft2d_shift_true, dtype=tf.complex64)
    fft2d_shift_pred = tf.cast(fft2d_shift_pred, dtype=tf.complex64)

    MSE_Fourier_domain = tf.losses.MSE(abs(fft2d_shift_true), abs(fft2d_shift_pred))

    ifft2d_shift_true = tf.signal.ifftshift(fft2d_shift_true, axes=(-2, -1))
    ifft2d_shift_pred = tf.signal.ifftshift(fft2d_shift_pred, axes=(-2, -1))

    ifft2d_shift_true = tf.signal.ifft2d(ifft2d_shift_true)
    ifft2d_shift_pred = tf.signal.ifft2d(ifft2d_shift_pred)

    MSE_Spatial = tf.losses.MSE(y_true, y_pred)
    # """
    # Cast into complex 64 then calculate the FFT2D for the y_true and y_pred
    # """
    # fft2d_true = tf.signal.fft2d(tf.cast(y_true, dtype=tf.complex64))
    # fft2d_pred = tf.signal.fft2d(tf.cast(y_pred, dtype=tf.complex64))
    #
    # fft2d_true = tf.signal.fftshift(fft2d_true, axes=(-2, -1))
    # fft2d_pred = tf.signal.fftshift(fft2d_pred, axes=(-2, -1))
    #
    # """
    # Normalise the FFT2D by dividing the calculated tensors by N (size of the fft2d(input))
    # """
    # N = tf.size(fft2d_true)
    # N = tf.cast(N, dtype=tf.complex64)
    # fft2d_true = tf.divide(fft2d_true, N)
    # fft2d_pred = tf.divide(fft2d_pred, N)
    #
    # """
    # Remove the higher modes (high frequencies)
    # """
    # for _ in modes:
    #     fft2d_true = tf.where(abs(fft2d_true) < tf.reduce_max(abs(fft2d_true)),
    #                           abs(fft2d_true),
    #                           tf.zeros_like(abs(fft2d_true)))
    #     fft2d_pred = tf.where(abs(fft2d_pred) < tf.reduce_max(abs(fft2d_pred)),
    #                           abs(fft2d_pred),
    #                           tf.zeros_like(abs(fft2d_pred)))
    #
    # """
    # Transpose the to calculate the MSE in the Fourier domain
    # """
    # fft2d_true_ = tf.transpose(fft2d_true, [0, 1, 3, 4, 2])
    # fft2d_pred_ = tf.transpose(fft2d_pred, [0, 1, 3, 4, 2])
    #
    # MSE_Fourier_domain = tf.losses.MSE(abs(fft2d_true_), abs(fft2d_pred_))
    #
    # """
    # Here, I performed the ifft2d to calculate the mse for y_true and y_pred in spatial domain
    # """
    # fft2d_true = tf.cast(fft2d_true, dtype=tf.complex64)
    # fft2d_pred = tf.cast(fft2d_pred, dtype=tf.complex64)
    #
    # ifft2d_true = tf.signal.ifft2d(fft2d_true) * N
    # ifft2d_pred = tf.signal.ifft2d(fft2d_pred) * N
    #
    # ifft2d_true = tf.signal.fftshift(ifft2d_true, axes=(-2, -1))
    # ifft2d_pred = tf.signal.fftshift(ifft2d_pred, axes=(-2, -1))
    #
    # ifft2d_true = tf.transpose(ifft2d_true, [0, 1, 3, 4, 2])
    # ifft2d_pred = tf.transpose(ifft2d_pred, [0, 1, 3, 4, 2])
    # MSE_Spatial = tf.losses.MSE(abs(ifft2d_true), abs(ifft2d_pred))

    # for cont in range(16):
    #     fig1, [ax10, ax20] = plt.subplots(1, 2, figsize=(10, 5))
    #
    #     a = abs(ifft2d_shift_true[0][cont])
    #     a = a.numpy()
    #
    #     b = abs(ifft2d_shift_pred[0][cont])
    #     b = b.numpy()
    #
    #     ax10.imshow(np.transpose(a, [1, 2, 0]), cmap='jet')
    #     ax20.imshow(np.transpose(b, [1, 2, 0]), cmap='jet')
    #     plt.show()
    #
    # """
    # calculating the total loss in Fourier and spatial domains
    # """
    total_loss = tf.add(MSE_Fourier_domain, MSE_Spatial)

    return total_loss


def custom_metric(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, 1)


lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    params['learning_rate'],
    decay_steps=10000,
    decay_rate=0.96,
    staircase=False)


def get_model():
    input_1 = tf.keras.layers.Input(shape=(8, 32, 32, 1))
    input_2 = tf.keras.layers.Input(shape=(8, 32, 32, 1))
    ####################################################################################################################
    # ref_frames_encoder = get_time_distributed(input_1, params['num_filters'])
    # coords_layer = input_2
    concat_layer = tf.keras.layers.multiply((input_1, input_2))
    skip_tensor = []
    for cnt in range(params['levels']):
        concat_layer = get_time_distributed(concat_layer, params['num_filters'] * 2 ** (cnt + 1))
        concat_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.AveragePooling2D((2, 2),
                                                                                        strides=(2, 2)))(
            concat_layer)
        skip_tensor.append(concat_layer)
        # concat_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2)))(concat_layer)

    # print(ref_frames_encoder)
    ####################################################################################################################
    # LS_input = tf.keras.layers.concatenate((ref_frames_encoder, coords_layer), axis=-1)
    # LS_input = tf.math.reduce_prod(LS_input, axis=-1, keepdims=True)
    bottleneck = get_time_distributed(concat_layer, params['num_filters'] * 2 ** 5)
    bottleneck = tf.keras.layers.ConvLSTM2D(1, (1, 1), padding='same', return_sequences=True)(bottleneck)
    bottleneck = get_time_distributed(bottleneck, params['num_filters'] * 2 ** 5)
    decoder = bottleneck
    # ####################################################################################################################

    for j in (range(1, params['levels'] + 1)):
        decoder = tf.keras.layers.concatenate((decoder, skip_tensor[-j]))
        decoder = tf.keras.layers.TimeDistributed(tf.keras.layers.UpSampling2D((2, 2)))(decoder)
        decoder = get_time_distributed(decoder, params['num_filters'] * 2 ** (j + 1))

    ####################################################################################################################
    lstm_layer = tf.keras.layers.ConvLSTM2D(1, (1, 1), padding='same', return_sequences=True)(decoder)
    ####################################################################################################################
    # Output layer
    #################################################################################################################### 
    model_ = Model(inputs=[input_1, input_2], outputs=lstm_layer)
    ####################################################################################################################
    model_.compile(tf.keras.optimizers.Adam(lr_schedule),
                   loss='mse',
                   metrics=[tf.keras.metrics.RootMeanSquaredError()],
                   run_eagerly=True)
    return model_


# In[30]:


tf.keras.backend.clear_session()
# Build model
model = get_model()
model.summary()

# In[31]:


checkpoint_filepath = env_path + 'temp/checkpoint/Frame_%s_time_stamps_%s_%s_%d_lr_filters_%d_levels_%d_batches_%d_epochs_%d_dropout_%s_val_split_%s_hidden_%d.h5' % (
    params['shape'],
    params['time_stamps'],
    params['samples'],
    params['learning_rate'],
    params['num_filters'],
    params['levels'],
    params['batches'],
    params['epochs'],
    params['dropout'],
    params['val_split'],
    params['hidden_layer'])

callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                              patience=params['patience_epochs'],
                                              mode='min'),
             tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                monitor='val_loss',
                                                save_best_only=True)]


class MonitoringCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        for metric_name, metric_value in logs.items():
            run[metric_name].log(metric_value)


# In[32]:


model.fit([X1, delam_coords], Y,
          batch_size=params['batches'],
          validation_split=params['val_split'],
          epochs=params['epochs'],
          callbacks=[callbacks, MonitoringCallback()])  #
