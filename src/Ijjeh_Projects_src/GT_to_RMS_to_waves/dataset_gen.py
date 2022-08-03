# ============================================
__author__ = "Abdalraheem A. Ijjeh"
__maintainer__ = "Abdalraheem A. Ijjeh"

# ============================================
import glob
import os
import time

import numpy as np
import scipy
from PIL import Image
from sklearn.cluster import KMeans
import sys
import cv2
import matplotlib.pyplot as plt
import os
import scipy.fftpack
import numpy.fft as fft
from datetime import datetime

########################################################################################################################
path_save = '/home/aijjeh/Desktop/Phd_Project/GT_RMS_waves/Dataset/'
labels_GT = '/aijjeh_odroid_sensors/aidd/data/raw/num/dataset2_labels_out/'


def create_training_x():
    os.chdir(labels_GT)
    array_temp = []
    for i in range(475):
        print(i)
        data_img = cv2.imread('m1_rand_single_delam_%d.png' % (i + 1), 1)
        height, width, channels = data_img.shape
        img = cv2.resize(data_img, (512, 512))
        (T, thresh) = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
        thresh_1 = cv2.flip(thresh, 0)
        thresh_2 = cv2.flip(thresh, 1)
        thresh_3 = cv2.flip(thresh, -1)
        array_temp.append(thresh)
        array_temp.append(thresh_1)
        array_temp.append(thresh_2)
        array_temp.append(thresh_3)

    array_temp = np.asarray(array_temp)
    array_temp = np.reshape(array_temp, (475*4, 512, 512, channels))
    array_temp = array_temp.astype('float32')
    array_temp = array_temp / 255.0
    os.chdir(path_save)
    np.save('GT2RMS2waves_Training_x_thresholded_augmented_h_v_d_RGB', array_temp)


create_training_x()
# exit()
path_RMS = '/aijjeh_odroid_sensors/aidd/data/raw/num/RMS_wavefield_dataset2_out/'


def create_labels_RMS():
    os.chdir(path_RMS)
    array_temp = []
    for i in range(1, 476):
        print(i)
        path = path_RMS + '%d_output/' % i
        print(path)
        os.chdir(path)
        data_img = cv2.imread('RMS_flat_shell_Vz_%d_500x500bottom.png' % i, 1)
        height, width, channels = data_img.shape
        img = cv2.resize(data_img, (512, 512))
        img_1 = cv2.flip(img, 0)
        img_2 = cv2.flip(img, 1)
        img_3 = cv2.flip(img, -1)
        array_temp.append(img)
        array_temp.append(img_1)
        array_temp.append(img_2)
        array_temp.append(img_3)
    array_temp = np.asarray(array_temp)
    array_temp = np.reshape(array_temp, (475*4, 512, 512, channels))
    array_temp = array_temp.astype('float32')
    array_temp = array_temp / 255.0
    os.chdir(path_save)
    np.save('GT2RMS2waves_Labels_y_augmented_h_v_d_RGB', array_temp)


create_labels_RMS()
