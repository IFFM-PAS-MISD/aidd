import os
from IPython.display import Image, display
from tensorflow.keras.preprocessing.image import load_img
import PIL
from PIL import ImageOps
import re
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from keras.layers import MaxPooling3D, \
    Input, ConvLSTM2D, UpSampling2D, \
    MaxPooling2D, Conv2D, Concatenate, Conv3D, \
    Dropout, BatchNormalization, Add, MaxPool2D, Conv2DTranspose, MaxPool3D, Conv3DTranspose, UpSampling3D, \
    concatenate, MaxPooling3D, Bidirectional, TimeDistributed, Reshape, Flatten
from tensorflow.python.client import device_lib
import random
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from keras import backend as K
import natsort
import cv2

########################################################################################################################
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
print(device_lib.list_local_devices())


def to_FFT(x):
    x = tf.cast(x, tf.float32)
    x = tf.signal.rfft(x)
    x = tf.compat.v1.real(x)
    return x


def to_inverse_FFT(x):
    x = tf.cast(x, tf.complex64)
    x = tf.signal.irfft(x)
    x = tf.compat.v1.real(x)
    return x


########################################################################################################################
# Load dataset
########################################################################################################################
dataset_path = '/home/aijjeh/Desktop/Phd_Projects/Sequence_prediction/Sequence_to_sequence/Dataset_Full_wavefield_outputs_bottom/'
os.chdir(dataset_path)
dataset_x_GT = np.load('Refrence_waves_GT_64_frame_475_512_512_65.npy', mmap_mode='r+')
dataset_y_RMS = np.load('Full_wavefield_labels_64_frame_475_64_512_512.npy', mmap_mode='r+')

# dataset_y_RMS = np.transpose(dataset_y_RMS, [0, 3, 1, 2])

print(dataset_x_GT.shape)
print(dataset_y_RMS.shape)
####################################################################################################################
x_train = dataset_x_GT[:304]
y_train = dataset_y_RMS[:304]
x_val = dataset_x_GT[304:380]
y_val = dataset_y_RMS[304:380]
x_test = dataset_x_GT[380:]
y_test = dataset_y_RMS[380:]

x_test = np.transpose(x_test, [0, 3, 1, 2])
sample = x_test[0][-1]
sample = sample.astype('float32')
plt.imshow(sample)
plt.show()

ref_img = y_test[0][0]
ref_img = ref_img.astype('float32')
plt.imshow(ref_img)
plt.show()
