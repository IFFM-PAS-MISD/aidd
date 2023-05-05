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
os.chdir('/home/aijjeh/Desktop/Phd_Projects/Modeling_guided_waves/Model_1_2_3_datasets/Model_2_dataset')
samples_gt_delamination = np.load('delamination_ground_truths.npy', mmap_mode='r+')
print(samples_gt_delamination.shape)
healthy_ref = np.load('health_full_wave_fields.npy', mmap_mode='r+')
print(healthy_ref.shape)
########################################################################################################################
#  GT
########################################################################################################################
latent_space = np.load('predicted_Latent_space.npy')
skip_0 = np.load('predicted_skip_connection_0.npy', mmap_mode='r+')
skip_1 = np.load('predicted_skip_connection_1.npy', mmap_mode='r+')
skip_2 = np.load('predicted_skip_connection_2.npy', mmap_mode='r+')
skip_3 = np.load('predicted_skip_connection_3.npy', mmap_mode='r+')
skip_4 = np.load('predicted_skip_connection_4.npy', mmap_mode='r+')
skip_5 = np.load('predicted_skip_connection_5.npy', mmap_mode='r+')
print(latent_space.shape)
print(skip_0.shape)
print(skip_1.shape)
print(skip_2.shape)
print(skip_3.shape)
print(skip_4.shape)
print(skip_5.shape)


########################################################################################################################
model_name = '/home/aijjeh/Desktop/Phd_Projects/Modeling_guided_waves/temp/checkpoint/model_2_mapping_latent_space.h5'
model = load_model(model_name, compile=False)
model.summary()

os.chdir('/home/aijjeh/Desktop/Phd_Projects/Modeling_guided_waves/Models_1_2_3_predictions/Model_2_predictions')


def intermediate_outputs(name):
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer(name).output)
    intermediate_output = intermediate_layer_model.predict([samples_gt_delamination, healthy_ref])
    for test_case in range(475):
        pred_mat_ = {name: intermediate_output[test_case]}
        scipy.io.savemat(name + '_%d.mat' % test_case, pred_mat_)


intermediate_outputs('output')
for j in range(6):
    intermediate_outputs('skip_connection_%d' % j)

