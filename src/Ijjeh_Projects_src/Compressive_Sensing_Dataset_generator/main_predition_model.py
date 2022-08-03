import csv
import gc
import cv2
from tensorflow import keras
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import neptune
from tensorflow.python.client import device_lib
from PIL import Image
import tensorflow as tf
import os
from keras.layers import LeakyReLU
from pathlib import Path
from decouple import config
import matplotlib
from matplotlib import cm
import itertools

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

print(device_lib.list_local_devices())


########################################################################################################################
# Load dataset
########################################################################################################################
def load_dataset():
    path_cs_dataset = '/home/aijjeh/Desktop/Phd_Projects/compressive_sensing_project/cs_datasets'
    os.chdir(path_cs_dataset)
    Full_W_dataset = np.load('CS_dataset_labels_Full_wavefield_475_128_512_512.npy', mmap_mode='r+')
    print(Full_W_dataset.shape)
    # CS_arr = np.load('CS_dataset_interpolated_CR_0.215_percent_nyquist_rate_applied_totally_random_points.npy',
    #                  mmap_mode='r+')
    # CS_arr = np.load('CS_dataset_labels_Full_wavefield_475_128_512_512.npy', mmap_mode='r+')
    # CS_arr = CS_arr.reshape((475 * 128, 512, 512, 1))
    gt_input = np.load('CS_dataset_CR_0.215_percent_nyquist_rate_applied_Uniform_grid.npy')
    gt_input = gt_input.reshape((475 * 128, 32, 32, 1))
    # x_test = CS_arr[380 * 128:]
    y_test = Full_W_dataset[380 * 128:]
    gt_x_input = gt_input[380 * 128:]
    return gt_x_input, y_test, gt_x_input  # x_test


########################################################################################################################
# Ijjeh Model
########################################################################################################################
os.chdir('/home/aijjeh/Desktop/Phd_Projects/compressive_sensing_project/h5_models/')
new_dim = 32
CR = (new_dim ** 2) / (69 ** 2)  # Compression Ratio
rescale_factor = int(512 / new_dim)

# model_name = 'super_resolution_AE_rescale_factor_%d_Uniform_grid_32_32_input.h5' % rescale_factor
model_name = 'super_resolution_AE_rescale_factor_%d_Uniform_grid_32_32_adding_GRL.h5' % rescale_factor  # Used in DLSS paper
model = load_model(model_name, compile=False)
model.summary()

########################################################################################################################
# Saeed Model
########################################################################################################################
# os.chdir('/home/aijjeh/Desktop/Phd_Projects/compressive_sensing_project/Saeed_model/h5_model')
# new_dim = 32
# CR = (new_dim ** 2) / (69 ** 2)  # Compression Ratio
# rescale_factor = int(512 / new_dim)
# model_name = 'Sub_pix_latest.h5'
# model = load_model(model_name, compile=False)
# model.summary()


########################################################################################################################

def testing():
    path = '/home/aijjeh/Desktop/Phd_Projects/compressive_sensing_project/Numerical_results/DLSS_num/Compression_ration_%s' % np.round(
        CR, 3)
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)

    # for output_case in range(1, 96):
    output_case = 95
    x_test, y_test, LR_input = load_dataset()
    path_ = path + '/Output_%d' % (output_case + 380)
    try:
        os.mkdir(path_)
    except OSError:
        print("Creation of the directory %s failed" % path_)
    else:
        print("Successfully created the directory %s " % path_)
    os.chdir(path_)

    x_test = x_test[(output_case - 1) * 128:(output_case - 1) * 128 + 128]
    y_test = y_test[(output_case - 1) * 128:(output_case - 1) * 128 + 128]
    LR_input = LR_input[(output_case - 1) * 128:(output_case - 1) * 128 + 128]
    print(x_test.shape)
    prediction = model.predict(x_test, batch_size=4)
    prediction = np.asarray(prediction)
    print(prediction.shape)
    frames = 128
    for i in range(frames):
        SR_pred = prediction[i].astype('float32')
        lr_input = LR_input[i].astype('float32')
        original = x_test[i].astype('float32')
        GT_label_input = y_test[i].astype('float32')
        ############################################################################################################
        plt.figure(figsize=(5 / 2.54, 5 / 2.54), dpi=600)
        plt.gca().set_axis_off()
        plt.axis('off')
        plt.subplots_adjust(left=0.0, bottom=0.0, right=1, top=1.0, wspace=0.0, hspace=0.0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        ############################################################################################################
        # ax1 = plt.subplot(1, 4, 1)
        # ax1.set_title('Original input')
        # ax1.imshow(original)
        # plt.axis('off')

        # ax1 = plt.subplot(1, 3, 1)
        # ax1.set_title('LR input')
        # ax1.imshow(lr_input)
        # plt.axis('off')
        #
        # ax2 = plt.subplot(1, 3, 2)
        # ax2.set_title('GT')
        # ax2.imshow(GT_label_input)
        # plt.axis('off')

        ax3 = plt.subplot(1, 1, 1)
        # ax3.set_title('SR output')
        ax3.imshow(SR_pred)
        plt.axis('off')

        plt.savefig('Saeed_SR_output_%d_frame_%d' % ((output_case + 380), (i + 1)))
        print((output_case + 380), i)
        plt.close('all')


if __name__ == '__main__':
    testing()
