import itertools
import os
import time
import psutil
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
import multiprocessing as mp


def get_interpolated(parm):
    array = CS_arr[parm]

    x = range(512)
    y = range(512)

    array = np.ma.masked_values(array, 0)

    xx, yy = np.meshgrid(x, y)
    x1 = xx[~array.mask]
    y1 = yy[~array.mask]

    new_arr_ = array[~array.mask]
    x_new, y_new = np.linspace(0, array.shape[1], CS_length), np.linspace(0, array.shape[1], CS_length)
    xq, yq = np.meshgrid(x_new, y_new)

    GD1 = scipy.interpolate.griddata((x1, y1), new_arr_.ravel(), (xq, yq), method='cubic', fill_value=np.mean(new_arr_))
    finish = time.perf_counter()
    print(f'Finished in {round(finish - start, 2)} seconds')
    return GD1


def load_samples():
    path_ = '/home/aijjeh/Desktop/Phd_Projects/compressive_sensing_project/cs_datasets/'
    os.chdir(path_)

    samples_arr = np.load('CS_dataset_labels_Full_wavefield_475_128_512_512.npy', mmap_mode='r+')
    samples_arr = samples_arr.reshape((475, 128, 512, 512))
    print(samples_arr.shape)
    return samples_arr


def get_CS_dataset(CS_temp_arr_, Original_arr):
    for sample in range(Original_arr.shape[0]):
        for frame in range(Original_arr.shape[1]):
            CS_temp_arr_[sample, frame, 0::16, 0::16] = Original_arr[sample, frame, 0::16, 0::16]
        print(sample)
    return CS_temp_arr_


if __name__ == '__main__':
    # #####################################
    # Full_W_dataset = load_samples()
    # #####################################
    new_dim = 32
    CR = (new_dim ** 2) / (69 ** 2)  # Compression Ratio
    CR = np.round(CR, 3)
    #####################################

    CS_length = new_dim
    print(CS_length)
    path_cs_dataset = '/home/aijjeh/Desktop/Phd_Projects/compressive_sensing_project/cs_datasets/'
    os.chdir(path_cs_dataset)
    #
    # CS_temp_arr = np.zeros((Full_W_dataset.shape[0],
    #                         Full_W_dataset.shape[1],
    #                         Full_W_dataset.shape[2],
    #                         Full_W_dataset.shape[3]),
    #                        dtype='float16')  # Compressed dataset
    # print(CS_temp_arr.shape)
    #
    # CS_temp_arr = get_CS_dataset(CS_temp_arr, Full_W_dataset)
    #
    # print(CS_temp_arr.shape)
    # np.save('Uniform_mesh_applied_512_512_before_interpolation_no_mask', CS_temp_arr)
    # exit()

    CS_arr = np.load('Uniform_mesh_applied_512_512_before_interpolation_no_mask.npy', mmap_mode='r+')

    plt.imshow(CS_arr[0][0].astype(np.float32))
    plt.show()
    exit()

    print(CS_arr.shape)

    # plt.imshow(CS_arr[0][0].astype(np.float32))
    # plt.show()
    # exit()

    start = time.perf_counter()

    pool = mp.Pool(12)

    a = range(475)
    b = range(128)

    param_list = list(itertools.product(a, b))

    results = pool.map(get_interpolated, param_list)

    np.save('CS_dataset_interpolated_CR_%s_percent_nyquist_rate_applied_UNIFROM_GRID_32_32_no_mask.npy' % np.round(CR, 3),
            results)
    pool.close()
