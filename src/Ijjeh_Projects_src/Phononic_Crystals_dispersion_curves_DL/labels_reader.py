import mat73
import math
import os
from IPython.display import Image, display
import numpy as np
import random
import matplotlib.pyplot as plt
import natsort
from scipy import signal
from sewar.full_ref import mse, rmse, psnr, uqi, ssim, ergas, scc, rase, sam, msssim, vifp
import neptune
import cv2

os.chdir('/aijjeh_odroid_laser/BOHEME/cavities/PC_binary_reducedK')

train_y = np.zeros((479, 512, 64), dtype=np.float32)
for i in range(479):
    mat = cv2.imread('PC_out_%d.png' % (i + 1), 0)
    mat = mat / 255
    train_y[i] = mat

os.chdir('/home/aijjeh/Desktop/Phd_Projects/PC_uint_cell_dispersion_curves/dataset')
np.save('train_y_GT_images', train_y)
