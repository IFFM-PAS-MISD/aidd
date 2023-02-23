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
from keras.models import load_model

load_dotenv()


def get_dif_pred():
    env_path = '/home/aijjeh/Desktop/Phd_Projects/Modeling_guided_waves/'

    # os.chdir(env_path + 'num/')
    #
    # arr = np.load('diff_healthy_damage_prediction_num.npy')
    # arr = arr.reshape((475, 512, 32, 32))
    #
    # for i in range(arr.shape[0]):
    #     for j in range(0, arr.shape[1], 32):
    #         plt.imshow(arr[i][j])
    #         plt.show()
    #
    # exit()

    os.chdir(env_path)

    # with open('hyper_par_GWM_mse_SA_ConvLSMT.json', 'r') as f:
    #     params = json.load(f)

    # # Link to Neptune
    # access_token = os.getenv('NEPTUNE_API_TOKEN')
    # run = neptune.init_run(project='abdalraheem.ijjeh/Guided-waves-modelling',
    #                        api_token=access_token,
    #                        tags=['DL HR_FFT2D_guided_waves_modelling', 'Fourier_domain']
    #                        )

    # run["Prediction file"] = "for predicting the LR GWP"
    # run["model/parameters"] = params
    # Hyperparameters

    params = {'samples': 15200,
              'n_size': 194560,
              'normalised': True,
              'batches': 16,
              'num_filters': 16,
              'kernel_size': 3,
              'shape': (512, 32, 32, 1),
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

    def load_dataset():
        os.chdir('/home/aijjeh/Desktop/Phd_Projects/Modeling_guided_waves/dataset')
        # delam_coords = np.load('LR_GT_del.npy')
        # delam_coords = delam_coords.reshape((475, 1, 5, 1))
        # delam_coords = np.repeat(delam_coords, 512, axis=1)
        # delam_coords = delam_coords.reshape((475 * 512, 5, 1))
        # delam_coords = np.repeat(delam_coords, 32, axis=1)
        # delam_coords = np.repeat(delam_coords, 32, axis=2)

        delam_GT = np.load('LR_GT_labels_img.npy')
        delam_GT_img = delam_GT.reshape((475, 1, 32, 32))
        delam_GT_img = np.repeat(delam_GT_img, 512, axis=1)
        delam_GT_img = delam_GT_img.reshape((475, 512, 32, 32, 1))
        # delam_GT_img = np.transpose(delam_GT_img, (0, -1, 1))
        # delam_GT_img = np.invert(delam_GT_img)
        print(delam_GT_img.shape)

        x_lr_frame = np.load('LR_ref_frames.npy')
        x_lr_frame = np.reshape(x_lr_frame, (475, 512, 32, 32, 1))
        print(x_lr_frame.shape)

        Y_train_ = np.load('LR_labels.npy')
        Y_train_ = np.reshape(Y_train_, (475, 512, 32, 32, 1))

        X_train = delam_GT_img
        Y_train = np.subtract(x_lr_frame, Y_train_)  # difference between healthy and damaged
        # fig, ax = plt.subplots(2)
        # ax[0].imshow(delam_GT_img[0, 80])
        # ax[1].imshow(Y_train[0, 80])
        # plt.show()
        # Train_x, Val_x_samples, Train_label, Val_y_samples = train_test_split(X_train, Y_train,
        #                                                                       train_size=0.99,
        #                                                                       shuffle=False,
        #                                                                       random_state=1988)

        return X_train, Y_train

    x_test, y_test = load_dataset()
    print(x_test.shape)
    print(y_test.shape)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

    # physical_devices = tf.config.list_physical_devices('GPU')
    # for device in physical_devices:
    #     tf.config.experimental.set_memory_growth(device, True)

    # def atoi(text):
    #     return int(text) if text.isdigit() else text
    #
    #
    # def natural_keys(text):
    #     return [atoi(c) for c in re.split(r'(\d+)', text)]

    # input_dir = "/aijjeh_odroid_sensors/aidd/data/raw/num/wavefield_dataset_undelam_bottom_out/1_output"
    # target_dir_ = "/aijjeh_odroid_sensors/aidd/data/raw/num/wavefield_dataset2_bottom_out"
    #
    # my_list = os.listdir(input_dir)
    # my_list.sort(key=natural_keys)
    #
    # test_img_paths_total = []
    # test_label_img_paths_total = []
    #
    # for k in range(475):
    #     test_img_paths = sorted(
    #         [
    #             os.path.join(input_dir, f_name)
    #             for f_name in os.listdir(input_dir)
    #             if f_name.endswith(".png")
    #         ]
    #     )
    #     test_img_paths.sort(key=natural_keys)
    #     test_img_paths_total.append(test_img_paths)
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
    #     test_label_img_paths_total.append(target_img_paths)
    #
    # os.chdir(
    #     '/home/aijjeh/Desktop/Phd_Projects/Sequence_prediction/Full_wavefield_frames_time_series_project/Datasets/label_set')
    # file_frame = np.load('frames_initial.npy')
    # file_frame = file_frame[:380]
    # print(file_frame[0])

    # coords_array = coords_array[380:0]
    # print(coords_array.shape)

    # class Full_wavefield_frames(tf.keras.utils.Sequence):
    #     def __init__(self,
    #                  batch_size,
    #                  img_size,
    #                  input_imgs_paths_total,
    #                  target_imgs_paths_total,
    #                  input_coords,
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
    #                 # plt.imshow(img)
    #                 # plt.show()
    #
    #                 x1[batch_num][j] = img
    #
    #                 coords = coords_array[batch_num][count]
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

    # test_input_img_paths = test_img_paths_total[380:]
    # print(len(test_input_img_paths))
    # test_target_img_paths = test_label_img_paths_total[380:]
    # print(test_target_img_paths[0][0])

    # # Instantiate data Sequences for each split

    # test_gen = Full_wavefield_frames(params['batches'],
    #                                  params['shape'],
    #                                  test_input_img_paths,
    #                                  test_target_img_paths,
    #                                  coords_array[380:],
    #                                  params['time_stamps'])
    #
    # print(type(test_gen))
    # print('number of test samples ', len(test_gen))
    # print('1st axis', len(test_gen[0]))

    os.chdir(env_path + 'temp/checkpoint/')
    model_name = 'diff_prediction_encoder_ann_%s_%d_lr_filters_%d_levels_%d_batches_%d_epochs_%d_dropout_%s_val_split_%s_hidden_%d.h5' % (
        params.get('samples'),
        params.get('learning_rate'),
        params.get('num_filters'),
        params.get('levels'),
        params.get('batches'),
        params.get('epochs'),
        params.get('dropout'),
        params.get('val_split'),
        params.get('hidden_layer'))

    model = load_model(model_name, compile=False)
    model.summary()

    prediction = model.predict(x_test, batch_size=params['batches'])
    print(prediction.shape)
    os.chdir(env_path + 'num/')
    np.save('diff_healthy_damage_prediction_num', prediction)


def get_LR_pred():
    env_path = '/home/aijjeh/Desktop/Phd_Projects/Modeling_guided_waves/'

    # os.chdir(env_path + 'num/')
    #
    # arr = np.load('diff_healthy_damage_prediction_num.npy')
    # arr = arr.reshape((475, 512, 32, 32))
    #
    # for i in range(arr.shape[0]):
    #     for j in range(0, arr.shape[1], 32):
    #         plt.imshow(arr[i][j])
    #         plt.show()
    #
    # exit()

    os.chdir(env_path)

    # with open('hyper_par_GWM_mse_SA_ConvLSMT.json', 'r') as f:
    #     params = json.load(f)

    # # Link to Neptune
    # access_token = os.getenv('NEPTUNE_API_TOKEN')
    # run = neptune.init_run(project='abdalraheem.ijjeh/Guided-waves-modelling',
    #                        api_token=access_token,
    #                        tags=['DL HR_FFT2D_guided_waves_modelling', 'Fourier_domain']
    #                        )

    # run["Prediction file"] = "for predicting the LR GWP"
    # run["model/parameters"] = params
    # Hyperparameters

    params = {'samples': 15200,
              'n_size': 194560,
              'normalised': True,
              'batches': 16,
              'num_filters': 16,
              'kernel_size': 3,
              'shape': (512, 32, 32, 1),
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

    def load_dataset():
        os.chdir(env_path + 'num/')
        pred_diff_arr = np.load('diff_healthy_damage_prediction_num.npy')
        pred_diff_arr = pred_diff_arr.reshape((475, 512, 32, 32, 1))

        os.chdir('/home/aijjeh/Desktop/Phd_Projects/Modeling_guided_waves/dataset')
        # delam_coords = np.load('LR_GT_del.npy')
        # delam_coords = delam_coords.reshape((475, 1, 5, 1))
        # delam_coords = np.repeat(delam_coords, 512, axis=1)
        # delam_coords = delam_coords.reshape((475 * 512, 5, 1))
        # delam_coords = np.repeat(delam_coords, 32, axis=1)
        # delam_coords = np.repeat(delam_coords, 32, axis=2)

        # delam_GT = np.load('LR_GT_labels_img.npy')
        # delam_GT_img = delam_GT.reshape((475, 1, 32, 32))
        # delam_GT_img = np.repeat(delam_GT_img, 512, axis=1)
        # delam_GT_img = delam_GT_img.reshape((475, 512, 32, 32, 1))
        # # delam_GT_img = np.transpose(delam_GT_img, (0, -1, 1))
        # # delam_GT_img = np.invert(delam_GT_img)
        # print(delam_GT_img.shape)

        x_lr_frame = np.load('LR_ref_frames.npy')
        x_lr_frame = np.reshape(x_lr_frame, (475, 512, 32, 32, 1))
        print(x_lr_frame.shape)

        Y_train = np.load('LR_labels.npy')
        Y_train = np.reshape(Y_train, (475, 512, 32, 32, 1))

        X_train = np.add(x_lr_frame, pred_diff_arr)
        # Y_train = np.subtract(x_lr_frame, Y_train_)  # difference between healthy and damaged
        # fig, ax = plt.subplots(2)
        # ax[0].imshow(delam_GT_img[0, 80])
        # ax[1].imshow(Y_train[0, 80])
        # plt.show()
        # exit()
        Train_x, Val_x_samples, Train_label, Val_y_samples = train_test_split(X_train, Y_train,
                                                                              train_size=0.80,
                                                                              shuffle=False,
                                                                              random_state=1988)

        return Val_x_samples, Val_y_samples

    x_test, y_test = load_dataset()
    print(x_test.shape)
    print(y_test.shape)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

    # physical_devices = tf.config.list_physical_devices('GPU')
    # for device in physical_devices:
    #     tf.config.experimental.set_memory_growth(device, True)

    # def atoi(text):
    #     return int(text) if text.isdigit() else text
    #
    #
    # def natural_keys(text):
    #     return [atoi(c) for c in re.split(r'(\d+)', text)]

    # input_dir = "/aijjeh_odroid_sensors/aidd/data/raw/num/wavefield_dataset_undelam_bottom_out/1_output"
    # target_dir_ = "/aijjeh_odroid_sensors/aidd/data/raw/num/wavefield_dataset2_bottom_out"
    #
    # my_list = os.listdir(input_dir)
    # my_list.sort(key=natural_keys)
    #
    # test_img_paths_total = []
    # test_label_img_paths_total = []
    #
    # for k in range(475):
    #     test_img_paths = sorted(
    #         [
    #             os.path.join(input_dir, f_name)
    #             for f_name in os.listdir(input_dir)
    #             if f_name.endswith(".png")
    #         ]
    #     )
    #     test_img_paths.sort(key=natural_keys)
    #     test_img_paths_total.append(test_img_paths)
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
    #     test_label_img_paths_total.append(target_img_paths)
    #
    # os.chdir(
    #     '/home/aijjeh/Desktop/Phd_Projects/Sequence_prediction/Full_wavefield_frames_time_series_project/Datasets/label_set')
    # file_frame = np.load('frames_initial.npy')
    # file_frame = file_frame[:380]
    # print(file_frame[0])

    # coords_array = coords_array[380:0]
    # print(coords_array.shape)

    # class Full_wavefield_frames(tf.keras.utils.Sequence):
    #     def __init__(self,
    #                  batch_size,
    #                  img_size,
    #                  input_imgs_paths_total,
    #                  target_imgs_paths_total,
    #                  input_coords,
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
    #                 # plt.imshow(img)
    #                 # plt.show()
    #
    #                 x1[batch_num][j] = img
    #
    #                 coords = coords_array[batch_num][count]
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

    # test_input_img_paths = test_img_paths_total[380:]
    # print(len(test_input_img_paths))
    # test_target_img_paths = test_label_img_paths_total[380:]
    # print(test_target_img_paths[0][0])

    # # Instantiate data Sequences for each split

    # test_gen = Full_wavefield_frames(params['batches'],
    #                                  params['shape'],
    #                                  test_input_img_paths,
    #                                  test_target_img_paths,
    #                                  coords_array[380:],
    #                                  params['time_stamps'])
    #
    # print(type(test_gen))
    # print('number of test samples ', len(test_gen))
    # print('1st axis', len(test_gen[0]))

    os.chdir(env_path + 'temp/checkpoint/')
    model_name = 'encoder_ann_%s_%d_lr_filters_%d_levels_%d_batches_%d_epochs_%d_dropout_%s_val_split_%s_hidden_%d.h5' % (
        params.get('samples'),
        params.get('learning_rate'),
        params.get('num_filters'),
        params.get('levels'),
        params.get('batches'),
        params.get('epochs'),
        params.get('dropout'),
        params.get('val_split'),
        params.get('hidden_layer'))

    model = load_model(model_name, compile=False)
    model.summary()

    prediction = model.predict(x_test, batch_size=params['batches'])
    print(prediction.shape)
    os.chdir(env_path + 'num/')
    np.save('LR_damage_prediction_num', prediction)


get_dif_pred()
get_LR_pred()
