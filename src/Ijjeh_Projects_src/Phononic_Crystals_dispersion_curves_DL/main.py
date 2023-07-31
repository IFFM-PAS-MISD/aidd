import cv2
import os
import matplotlib.pyplot as plt
import numpy as np


path_x = '/aijjeh_odroid_laser/BOHEME/cavities/PC_labels'

train_x = np.zeros((2000, 256, 256), dtype=np.float64)

for i in range(0, 2000):
    os.chdir(path_x)
    img_data = cv2.imread('PC_label_%d.png' % (i+1), 0)
    img_data = img_data[:256, :256]
    train_x[i] = img_data / 255.0

os.chdir('/home/aijjeh/Desktop/Phd_Projects/PC_uint_cell_dispersion_curves/dataset')
np.save('train_x', train_x)
