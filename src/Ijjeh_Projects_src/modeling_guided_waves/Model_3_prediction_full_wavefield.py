import csv
import gc
import os
from pathlib import Path
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import tensorflow as tf
from keras.models import load_model
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from tensorflow.python.client import device_lib
from keras.models import Model
from matplotlib import gridspec
import os
from IPython.display import Image, display
from tensorflow.keras.preprocessing.image import load_img
import PIL
from PIL import ImageOps
import re
import keras
from keras import layers
import math

########################################################################################################################
# Visualizing kernels weights and intermediate outputs
########################################################################################################################
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
print(device_lib.list_local_devices())

params = {'batches': 1,
          'kernel_size': 3,
          'height': 512,
          'width': 512,
          'time_stamps': 12,
          'num_filters': 16,
          'filter_size': 3,
          'epochs': 500,
          'dropout': 0.2,
          'levels': 9,
          'learning_rate': 0.00014329,
          'patience_epochs': 100,
          'val_split': 0.08,
          'hidden_layer': 3,
          'decay:steps': 100000,
          'decay_rate': 0.96,
          }

h = params.get('height')
w = params.get('width')
img_size = (h, w)

########################################################################################################################
#  Load dataset
########################################################################################################################
os.chdir('/home/aijjeh/Desktop/Phd_Projects/Modeling_guided_waves/Model_2_mapping_GT_to_LS/Predictions_model_2')
latent_space = np.load('latent_space.npy')
print(latent_space.shape)
skip_0 = np.load('skip_connection_0.npy', mmap_mode='r+')
skip_1 = np.load('skip_connection_1.npy', mmap_mode='r+')
skip_2 = np.load('skip_connection_2.npy', mmap_mode='r+')
skip_3 = np.load('skip_connection_3.npy', mmap_mode='r+')
skip_4 = np.load('skip_connection_4.npy', mmap_mode='r+')
skip_5 = np.load('skip_connection_5.npy', mmap_mode='r+')
skip_6 = np.load('skip_connection_6.npy', mmap_mode='r+')
skip_7 = np.load('skip_connection_7.npy', mmap_mode='r+')
skip_8 = np.load('skip_connection_8.npy', mmap_mode='r+')

print(skip_0.shape)
print(skip_1.shape)
print(skip_2.shape)
print(skip_3.shape)
print(skip_4.shape)
print(skip_5.shape)
print(skip_6.shape)
print(skip_7.shape)
print(skip_8.shape)

os.chdir('/home/aijjeh/Desktop/Phd_Projects/Modeling_guided_waves/Model_2_mapping_GT_to_LS')
gt_img_delamination = np.load('GT_model_3.npy')
print(gt_img_delamination.shape)
x1 = latent_space[380:381]
x2 = skip_8[380:381]
x3 = skip_7[380:381]
x4 = skip_6[380:381]
x5 = skip_5[380:381]
x6 = skip_4[380:381]
x7 = skip_3[380:381]
x8 = skip_2[380:381]
x9 = skip_1[380:381]
x10 = skip_0[380:381]
y = gt_img_delamination[380:381]

########################################################################################################################
model_name = '/home/aijjeh/Desktop/Phd_Projects/Modeling_guided_waves/temp/checkpoint/model_3_reconstructing_latent_space.h5'
model = load_model(model_name, compile=False)
model.summary()

os.chdir('/home/aijjeh/Desktop/Phd_Projects/Modeling_guided_waves/Model_2_mapping_GT_to_LS/predictions')


def plot_filters(layer, name_layer):
    if "conv2d_" in layer_name:
        filters = layer.get_weights()[0]
        filters = np.asarray(filters)
        filters = np.reshape(filters, (filters.shape[3], filters.shape[0], filters.shape[1], filters.shape[2]))
        print('convolution filter size', filters.shape)

        length = filters.shape[0]
        print(length)
        image = np.zeros((filters.shape[1], filters.shape[2]))
        print(image.shape)
        for j in range(0, length):
            fig = plt.figure(figsize=(length, length))
            plt.gca().set_axis_off()
            plt.axis('off')
            plt.subplots_adjust(left=0.0, bottom=0.0, right=1, top=1.0, wspace=0.0, hspace=0.0)
            plt.margins(0, 0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            gs1 = gridspec.GridSpec(math.ceil(np.sqrt(length)), math.ceil(np.sqrt(length)))
            gs1.update(wspace=0.01, hspace=0.01)
            for depth in range(0, filters.shape[3]):
                image = image + filters[j, :, :, depth]
                ax = fig.add_subplot(gs1[j])
                ax.imshow(image, cmap='gray')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_aspect('equal')
                plt.xticks(np.array([]))
                plt.yticks(np.array([]))
                plt.savefig(name_layer + str('_kernel_weights_length_%d_depth_%d_' % (j, depth)))
                plt.close('all')
    else:
        print("NO filters for this layer")


def intermediate_outputs():
    intermediate_layer_model = Model(inputs=model.get_layer('latent_space').input,
                                     outputs=model.get_layer('time_distributed_56').output)
    intermediate_output_ = intermediate_layer_model.predict([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10])
    for test_case in range(475):
        print('layer shape', intermediate_output_[test_case].shape)
        pred_mat_ = {'Recovered_Full_wavefeild': intermediate_output_[test_case]}
        scipy.io.savemat('Recovered_Full_wavefeild_%d.mat' % test_case, pred_mat_)
    # length = intermediate_output.shape[3]
    # intermediate_output = np.asarray(intermediate_output)
    ################################################################################################################
    # plt.figure(figsize=(5 / 2.54, 5 / 2.54), dpi=600)
    # plt.gca().set_axis_off()
    # plt.axis('off')
    # plt.subplots_adjust(left=0.0, bottom=0.0, right=1, top=1.0, wspace=0.0, hspace=0.0)
    # plt.margins(0, 0)
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())
    ################################################################################################################
    # print(intermediate_output[0, :, :, i])
    ###########################################################################################################
    # fig = plt.figure(figsize=(1, 1), dpi=512)
    # plt.gca().set_axis_off()
    # plt.axis('off')
    # plt.subplots_adjust(left=0.0, bottom=0.0, right=1, top=1.0, wspace=0.0, hspace=0.0)
    # plt.margins(0, 0)
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())
    # #############################################################################################################
    # for j in range(1, (length + 1)):
    #     ax = fig.add_subplot(math.ceil(np.sqrt(length)), math.ceil(np.sqrt(length)), j)
    #     ax.imshow(intermediate_output[0, :, :, (j - 1)], cmap='gray')
    #     ax.set_xticklabels([])
    #     ax.set_yticklabels([])
    #     ax.set_aspect('equal')
    #     plt.axis('off')
    # # plt.tight_layout()
    # # plt.show()
    # plt.savefig(layer_name, bbox_inches='tight', transparent="True", pad_inches=0)
    # plt.close('all')


# for input_image in range(475):
os.chdir('/home/aijjeh/Desktop/Phd_Projects/Modeling_guided_waves/latent_space_pred')
intermediate_output = model.predict([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10])
print('layer shape', intermediate_output.shape)
for i in range(12):
    plt.imshow(intermediate_output[0][i])
    plt.show()
pred_mat = {'Recovered_Full_wavefeild': intermediate_output}
scipy.io.savemat('Recovered_Full_wavefeild_%d.mat' % 381, pred_mat)
# intermediate_outputs()
