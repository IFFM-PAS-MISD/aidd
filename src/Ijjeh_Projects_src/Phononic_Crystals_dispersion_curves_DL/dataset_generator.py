import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse
import skimage.measure
import scipy.io
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy import polyval, polyfit
import mat73

samples = 7000
path_dataset = '/aijjeh_odroid_laser/BOHEME/cavities/'
path_env = '/home/aijjeh/Desktop/Phd_Projects/PC_uint_cell_dispersion_curves/'


def get_x_img(n):
    os.chdir(path_dataset + 'PC_labels')
    img_arr = np.zeros((n, 256, 256), dtype=np.float32)
    for i in range(n):
        img = plt.imread('PC_label_%d.png' % (i + 1), 0)
        img = img[:256, :256]
        img_arr[i] = img
    os.chdir(path_env + 'dataset')
    np.save('train_xy_img_samples_%d' % n, img_arr)
    return img_arr


def get_xy():
    os.chdir(path_dataset + 'PC')
    xy = np.zeros((samples, 121, 2), dtype=np.float32)
    for i in range(samples):
        mat = scipy.io.loadmat('PC_cavity_polygon_%d.mat' % (i + 1001))
        xy[i] = mat['cavity_polygon']
        print(i, " :", mat['cavity_polygon'].shape)
    os.chdir(path_env + 'dataset')
    np.save('train_xy_coordinates_samples_%d' % samples, xy)


def gen_GT(n):
    os.chdir(path_dataset + 'PC_comsol/')
    train_y = np.zeros((n, 1464, 2), dtype=np.float128)
    for i in range(n):
        mat = mat73.loadmat('out_lines_%d_a8_h3.mat' % (i + 1))
        train_y[i, :, 0] = np.reshape(mat['F'], (1464,))
        train_y[i, :, 1] = np.reshape(mat['K'], (1464,))

    os.chdir(path_env + 'dataset')
    np.save('train_y_samples_%d' % n, train_y)


get_x_img(samples)  # Samples (images)
get_xy()  # Samples (XY coordinates)
gen_GT(samples)  # Labels (ground truths)
