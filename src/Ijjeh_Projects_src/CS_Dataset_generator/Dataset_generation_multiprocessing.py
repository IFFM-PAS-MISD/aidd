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

# param = []
# for i in range(475):
#     for j in range(512):
#         param.append(
#             '/aijjeh_odroid_sensors/aidd/data/raw/num/wavefield_dataset2_bottom_out/%d_output/%d_flat_shell_Vz_%d_500x500bottom.png' % (
#                 (i + 1), (j + 1), (i + 1)))

path_cs_dataset = '/home/aijjeh/Desktop/Phd_Projects/compressive_sensing_project/cs_datasets'
os.chdir(path_cs_dataset)
# Full_W_dataset = np.load('CS_dataset_labels_Full_wavefield_475_512_49_49.npy', mmap_mode='r+')
# print(Full_W_dataset.shape)
CS_arr = np.load('CS_dataset_interpolated_CR_0.426_percent_nyquist_rate_applied_totally_random_points.npy', mmap_mode='r+')
CS_arr = CS_arr.reshape((475, 512, 32, 32, 1))
print(CS_arr.shape)

os.chdir(
    '/home/aijjeh/Desktop/Phd_Projects/Sequence_prediction/Full_wavefield_frames_time_series_project/Datasets/label_set')
f_start = np.load('frames_initial.npy')
z = (range(0, 475))
print(z)

start = time.perf_counter()


def get_dataset():
    # Full_W_dataset_temp = np.zeros((475, 128, 49, 49, 1), dtype='float16')
    CS_arr_temp = np.zeros((475, 128, 32, 32, 1), dtype='float16')
    for k in range(475):
        CS_arr_temp[k] = CS_arr[k, f_start[k]:f_start[k] + 128]
        # Full_W_dataset_temp[k] = Full_W_dataset[k, f_start[k]:f_start[k] + 128]
        print(k)

    CS_arr_temp = CS_arr_temp.reshape((475 * 128, 32, 32, 1))
    # Full_W_dataset_temp = Full_W_dataset_temp.reshape((475 * 128, 512, 512, 1))

    return CS_arr_temp  # , Full_W_dataset_temp


def get_wavefield(param_list):
    tem_arr = []
    data = cv2.imread(param_list, 0)
    img = cv2.resize(data, (49, 49))
    tem_arr.append(img.astype('float16') / 255.0)
    finish = time.perf_counter()
    print(f'Finished in {round(finish - start, 2)} seconds')
    return tem_arr


def main():
    # pool = mp.Pool(8)
    # LR = CS_arr[0][0]  # .astype('float16')
    # print(LR.shape)
    # print(len(np.argwhere(np.isnan(LR))))

    # results1 = pool.map(get_wavefield, param)
    # pool.close()
    # results1 = np.asarray(results1)
    # results1 = results1.reshape((475, 512, 49, 49, 1))
    # pool.close()
    x = get_dataset()
    path = '/home/aijjeh/Desktop/Phd_Projects/compressive_sensing_project/cs_datasets/'
    os.chdir(path)
    np.save('CS_dataset_interpolated_CR_.426_percent_128_frames_totally_random_points', x)
    # np.save('CS_dataset_labels_Full_wavefield_475_128_49_49', y)


if __name__ == '__main__':
    main()
