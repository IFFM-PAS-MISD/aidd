import os
import matplotlib.pyplot as plt
import numpy as np
import argparse
import scipy.io
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy import polyval, polyfit
import mat73
import cv2
from scipy.io import loadmat

samples = 7000
path_dataset = '/aijjeh_odroid_laser/BOHEME/cavities/'
path_env = '/home/aijjeh/Desktop/Phd_Projects/PC_uint_cell_dispersion_curves/'

'''
Returns quarter of the sample images of Phononic crystal Cell (256, 256) 
'''


def get_x_img(n):
    os.chdir(path_dataset + 'PC_labels')
    img_arr = np.zeros((n, 512, 512), dtype=np.float16)
    for i in range(n):
        img_arr[i] = plt.imread('PC_label_%d.png' % (i + 1), 0)
    os.chdir(path_env + 'dataset')
    np.save('train_xy_img_samples_%d_full_unit_cell' % n, img_arr)
    return img_arr


'''
Returns XY coordinates of the PC cavity polygon 
'''


def get_xy():
    os.chdir(path_dataset + 'PC')
    xy = np.zeros((6000, 121, 2), dtype=np.float32)
    for i in range(6000):
        mat = loadmat('PC_cavity_polygon_%d.mat' % (i + 1001))
        xy[i] = mat['cavity_polygon']
    #     print(i, " :", mat['cavity_polygon'].shape)

    # fig1, ax1 = plt.subplots(1, figsize=(5, 5))
    # ax1.plot(xy[i, :, 0], xy[i, :, 1])
    # ax2.scatter(xy[0, :, 0], xy[0, :, 1])
    # flat_plot = np.ravel(xy[0])
    # ax3.plot(flat_plot)

    # fig2, [ax1, ax2, ax3] = plt.subplots(3, figsize=(5, 15))
    # ax1.plot(xy[1, :, 0], xy[1, :, 1])
    # ax2.scatter(xy[1, :, 0], xy[1, :, 1])
    # flat_plot = np.ravel(xy[1])
    # ax3.plot(flat_plot)
    #
    # fig3, [ax1, ax2, ax3] = plt.subplots(3, figsize=(5, 15))
    # ax1.plot(xy[2, :, 0], xy[2, :, 1])
    # ax2.scatter(xy[2, :, 0], xy[2, :, 1])
    # flat_plot = np.ravel(xy[2])
    # ax3.plot(flat_plot)

    # plt.show()
    os.chdir(path_env + 'dataset')
    np.save('train_xy_coordinates_samples_%d' % samples, xy)


'''
Returns the labels in frequency and wavenumbers 
'''


def gen_GT(n):
    os.chdir(path_dataset + 'PC_comsol/')
    train_y = np.zeros((n, 1464, 2), dtype=np.float128)
    for i in range(n):
        mat = mat73.loadmat('out_lines_%d_a8_h3.mat' % (i + 1))
        train_y[i, :, 0] = np.reshape(mat['F'])
        train_y[i, :, 1] = np.reshape(mat['K'])

    # fig, ax = plt.subplots(1, figsize=(3, 10))
    # ax.scatter(train_y[i, :, 1], train_y[i, :, 0], linewidths=0.5)
    # plt.show()

    os.chdir(path_env + 'dataset')
    np.save('train_y_samples_%d' % n, train_y)


# get_x_img(samples)  # Samples (images)
# get_xy()  # Samples (XY coordinates)
gen_GT(samples)  # Labels (ground truths)

# os.chdir('/home/aijjeh/Desktop/Phd_Projects/PC_uint_cell_dispersion_curves/dataset')
# array = np.load('train_xy_img_samples_%d_full_unit_cell.npy' % samples)
#
# for i in range(array.shape[0]):
#     plt.imshow(array[i])
#     plt.show()
