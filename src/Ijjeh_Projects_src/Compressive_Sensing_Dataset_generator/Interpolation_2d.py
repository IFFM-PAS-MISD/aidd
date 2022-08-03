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


if __name__ == '__main__':
    #####################################
    new_dim = 32
    CR = (new_dim ** 2) / (69 ** 2)  # Compression Ratio
    CR = np.round(CR, 3)
    #####################################

    CS_length = new_dim  # int(np.sqrt((49 ** 2) * CR))  # dimension of the interpolated points
    print(CS_length)
    path_cs_dataset = '/home/aijjeh/Desktop/Phd_Projects/compressive_sensing_project/cs_datasets/'
    os.chdir(path_cs_dataset)

    CS_arr = np.load('CS_dataset_CR_%s_percent_nyquist_rate_applied_totally_random_points.npy' % np.round(CR, 3),
                     mmap_mode='r+')  # Loading the compressed dataset

    start = time.perf_counter()

    pool = mp.Pool(12)

    a = range(475)
    b = range(128)

    param_list = list(itertools.product(a, b))

    results = pool.map(get_interpolated, param_list)

    np.save('CS_dataset_interpolated_CR_%s_percent_nyquist_rate_applied_totally_random_points.npy' % np.round(CR, 3),
            results)
    pool.close()
