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


def gen_GT(n):
    os.chdir('/aijjeh_odroid_laser/BOHEME/cavities/PC_comsol/')
    train_y = np.zeros((n, 1464, 2), dtype=np.float128)
    for i in range(n):
        mat = mat73.loadmat('out_lines_%d_a8_h3.mat' % (i + 1))
        train_y[i, :, 0] = np.reshape(mat['F'], (1464,))
        train_y[i, :, 1] = np.reshape(mat['K'], (1464,))

    os.chdir('/home/aijjeh/Desktop/Phd_Projects/PC_uint_cell_dispersion_curves/dataset')
    np.save('train_y_samples_%d' % n, train_y)


gen_GT(samples)

# os.chdir('/home/aijjeh/Desktop/Phd_Projects/PC_uint_cell_dispersion_curves/dataset')
# mat = np.load('train_y_samples_%d.npy' % samples)
# print(np.max(mat))
