import os
from IPython.display import Image, display
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import natsort
import cv2
from skimage.transform import radon
from tensorflow.python.client import device_lib
import gc
import scipy
from scipy.interpolate import griddata

from clearml import Task
task = Task.init(project_name="CS_main_dataset_generator", task_name="create_training_dataset")

os.chdir(
    '/home/aijjeh/Desktop/Phd_Projects/Sequence_prediction/Full_wavefield_frames_time_series_project/Datasets/label_set')
f_start = np.load('frames_initial.npy')


def gen_rand(x):
    numbers = np.random.choice(np.arange(0, 49 ** 2), 32 ** 2, replace=False)
    gc.collect()
    # numbers = np.concatenate(([0], numbers, [47]))
    return numbers


def get_CS_dataset(CS_arr, Original_arr, H):
    for sample in range(Original_arr.shape[0]):
        for frame in range(Original_arr.shape[1]):
            CS_arr[sample, frame] = Original_arr[sample, frame] * H
        # for k in range(len(W)):
        # CS_arr[i, :, H[j], W[k]] = Original_arr[i, :, H[j], W[k]]
        print(sample)
    return CS_arr


def func(x, y):
    return CS_arr[0][140][y][x]


def load_samples():
    path_ = '/home/aijjeh/Desktop/Phd_Projects/compressive_sensing_project/cs_datasets/'
    os.chdir(path_)

    samples_arr = np.load('CS_dataset_labels_Full_wavefield_475_128_512_512.npy', mmap_mode='r+')
    samples_arr = samples_arr.reshape((475, 128, 512, 512))
    print(samples_arr.shape)
    return samples_arr


def uniform_resize(arr):
    temp_arr = np.zeros((475, 128, 32, 32), dtype=np.float16)
    for sample in range(475):
        print(sample)
        for frame in range(128):
            temp_arr[sample][frame] = cv2.resize(np.float32(arr[sample][frame]), (32, 32),
                                                 interpolation=cv2.INTER_CUBIC)
    return temp_arr


def visualize_set(arr):
    print(arr.shape)
    for sample in range(475):
        for frame in range(128):
            plt.imshow(np.float32(arr[sample][frame]))
            plt.show()
            plt.close('all')


path = '/home/aijjeh/Desktop/Phd_Projects/compressive_sensing_project/cs_datasets/'
os.chdir(path)
if __name__ == '__main__':
    #####################################
    Full_W_dataset = load_samples()
    #####################################
    new_dim = 32
    CR = (new_dim ** 2) / (69 ** 2)  # Compression Ratio
    CR = np.round(CR, 4)
    print(CR)
    # ##################################################################################################
    # # preprocessed_dataset = uniform_resize(Full_W_dataset)
    # # preprocessed_dataset = preprocessed_dataset.reshape((475, 128, 32, 32, 1))
    # # np.save('CS_dataset_CR_%s_percent_nyquist_rate_applied_Uniform_grid' % CR, preprocessed_dataset)
    # ##################################################################################################
    # preprocessed_data = np.load('CS_dataset_CR_0.215_percent_nyquist_rate_applied_Uniform_grid.npy')
    # visualize_set(preprocessed_data)
    # exit()
    # ##################################################################################################
    CS_length = int(np.sqrt((69 ** 2) * CR))
    print(CS_length)
    exit()
    #####################################
    CS_temp_arr = np.zeros((Full_W_dataset.shape[0],
                            Full_W_dataset.shape[1],
                            Full_W_dataset.shape[2],
                            Full_W_dataset.shape[3]),
                           dtype='float16')  # Compressed dataset
    # num_list_H = gen_rand(CS_length)  # generate random numbers height
    # num_list_W = gen_rand(CS_length)  # generate random numbers width
    mask = np.zeros((512 * 512))
    values = np.random.choice(np.arange(0, 512 ** 2), 32 ** 2, replace=False)
    values.sort()
    for i in range(len(values)):
        mask[values[i]] = 1
    mask = mask.reshape((512, 512))
    # num_list_H.sort()
    # num_list_W.sort()
    # print(num_list_W)
    # print(len(num_list_W))
    path_cs_dataset = '/home/aijjeh/Desktop/Phd_Projects/compressive_sensing_project/cs_datasets'
    os.chdir(path_cs_dataset)

    # np.save('random_selected_rows__CR_%s' % CR, num_list_H)
    # np.save('random_selected_cols__CR_%s' % CR, num_list_W)

    CS_temp_arr = get_CS_dataset(CS_temp_arr, Full_W_dataset, mask)  # num_list_W
    np.save('CS_dataset_CR_%s_percent_nyquist_rate_applied_totally_random_points' % CR, CS_temp_arr)
    gc.collect()
