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

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

print(device_lib.list_local_devices())

# load dataset


dataset_path = '/home/aijjeh/Desktop/Phd_Project/GT_RMS_waves/Dataset/'
os.chdir(dataset_path)
########################################################################################################################
dataset_x = np.load('GT2RMS2waves_Training_x_thresholded.npy', mmap_mode='r')
dataset_y = np.load('GT2RMS2waves_Labels_y_augmented_h_v_d.npy', mmap_mode='r')
x_test = dataset_x[380:]
y_test = dataset_y[380:]

# dataset_x = np.load('GT2RMS2waves_Training_x_thresholded_augmented_h_v_d.npy', mmap_mode='r')
# dataset_y = np.load('GT2RMS2waves_Labels_y_augmented_h_v_d.npy', mmap_mode='r')
# x_test = dataset_x[1520:]
# y_test = dataset_y[1520:]

test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = test_dataset.batch(1)

########################################################################################################################
os.chdir('/home/aijjeh/Desktop/Phd_Project/GT_RMS_waves/h5_models')
encoder_model = "VAE_encoder.h5"
decoder_model = "VAE_decoder.h5"

vae_model = 'VAE_prediction_all_frames_window_32_signals_augmented_2000_new.h5'
en_model = tf.keras.models.load_model(encoder_model,
                                      compile=False)

de_model = tf.keras.models.load_model(decoder_model,
                                      compile=False)
en_model.summary()
de_model.summary()

VAE_model = tf.keras.models.load_model(vae_model, compile=False)
VAE_model.summary()

os.chdir('/home/aijjeh/Desktop/Phd_Project/GT_RMS_waves/num_results')

path_to_csv = "E:/aidd_new/aidd/src/data_processing/PhD/cmap_flipped_jet256.csv"
cmap = matplotlib.colors.ListedColormap(["blue", "green", "red"], name=path_to_csv, N=None)


def testing():
    # prediction1 = en_model.predict(test_dataset, batch_size=1)
    prediction = VAE_model.predict(test_dataset, batch_size=1)
    prediction = np.asarray(prediction)
    for i in range(len(x_test)):
        RMS_pred = prediction[i]

        print(RMS_pred.shape)
        original = x_test[i]  # np.squeeze(x_test[i], axis=2)
        GT_label_input = y_test[i]  # np.squeeze(y_test[i], axis=2)
        print(GT_label_input.shape)
        # exit()
        ############################################################################################################
        plt.figure(figsize=(16 / 2.54, 4 / 2.54), dpi=600)
        plt.gca().set_axis_off()
        plt.axis('off')
        plt.subplots_adjust(left=0.0, bottom=0.0, right=1, top=1.0, wspace=0.0, hspace=0.0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        ############################################################################################################
        ax1 = plt.subplot(1, 4, 1)
        ax1.set_title('GT Input sample')
        ax1.imshow(original)
        plt.axis('off')

        ax2 = plt.subplot(1, 4, 2)
        ax2.set_title('RMS Prediction')
        ax2.imshow(RMS_pred)
        plt.axis('off')
        # 
        ax3 = plt.subplot(1, 4, 3)
        ax3.set_title('input GT')
        ax3.imshow(GT_label_input)
        plt.axis('off')

        ax4 = plt.subplot(1, 4, 4)
        ax4.set_title('input GT')
        sub_img = np.subtract(GT_label_input, RMS_pred)
        print(sub_img)
        ax4.imshow(sub_img > 0.01)
        plt.axis('off')

        # plt.imshow(GT_label_input, cmap='Greys')
        plt.savefig('GT_RMS_%d' % (i + 381))
        ############################################################################################################
        # plt.imshow(original, cmap='Greys')
        # # plt.savefig(path_fcn_figuers_folder_no_threshold+'/FCN_DenseNet_original_454_sigmoid' + str(i+1))
        # ############################################################################################################
        # plt.imshow(GT_label_input, cmap='gist_gray')
        # # plt.savefig(path_fcn_figuers_folder_no_threshold+'/FCN_DenseNet_GT_454_sigmoid' + str(i+1))
        # ############################################################################################################
        # # plt.show()
        print(i)
        plt.close('all')


testing()
