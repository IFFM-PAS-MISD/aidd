import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse
import skimage.measure
import scipy.io
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy import polyval, polyfit
from pathlib import Path

samples = 5151


def get_xy():
    path_x = '/aijjeh_odroid_laser/BOHEME/cavities/PC'
    os.chdir(path_x)
    xy = np.zeros((samples, 121, 2), dtype=np.float32)
    for i in range(samples):
        mat = scipy.io.loadmat('PC_cavity_polygon_%d.mat' % (i + 1001))
        xy[i] = mat['cavity_polygon']
        print(i, " :", mat['cavity_polygon'].shape)
    os.chdir('/home/aijjeh/Desktop/Phd_Projects/PC_uint_cell_dispersion_curves/dataset')
    np.save('train_xy_coordinates_samples_%d' % samples, xy)


get_xy()

os.chdir('/home/aijjeh/Desktop/Phd_Projects/PC_uint_cell_dispersion_curves/dataset')
arr = np.load('train_xy_coordinates_samples_%d.npy' % samples)
plt.plot((np.divide((arr[0, :, 0]), (arr[0, :, 1]+1))+1)/2)
plt.show()
