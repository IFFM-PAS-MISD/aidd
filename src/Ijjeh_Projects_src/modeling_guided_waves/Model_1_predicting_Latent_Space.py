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
from keras import backend as K
import matplotlib.animation as animation

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
# Load dataset
#####################################################################################################################
os.chdir('/home/aijjeh/Desktop/Phd_Projects/Modeling_guided_waves/Model_1_2_3_datasets/Model_1_dataset')
full_wavefields_with_damage = np.load('GT_full_wavefields.npy')
full_wavefields_with_damage = np.expand_dims(full_wavefields_with_damage, axis=-1)
print('Full wavefield frames with damage ', full_wavefields_with_damage.shape)
########################################################################################################################
#  Load dataset for Model_II encoder mapping
########################################################################################################################
os.chdir('/home/aijjeh/Desktop/Phd_Projects/Modeling_guided_waves/Model_1_2_3_datasets/Model_2_dataset')
samples_gt_delamination = np.load('delamination_ground_truths.npy', mmap_mode='r+')
# samples_gt_delamination = 1 - samples_gt_delamination

healthy_ref = np.load('health_full_wave_fields.npy', mmap_mode='r+')
# healthy_ref = healthy_ref * samples_gt_delamination

samples_gt_delamination = np.expand_dims(samples_gt_delamination, axis=-1)
print('labels of delaminations as image frames ', samples_gt_delamination.shape)
healthy_ref = np.expand_dims(healthy_ref, axis=-1)
print('healthy full wavefield frames ', healthy_ref.shape)


########################################################################################################################


########################################################################################################################
def intermediate_outputs(name, encoder):
    # intermediate_layer_model = Model(inputs=encoder.input,
    #                                  outputs=encoder.get_layer(name).output)  # get_layer(name)

    intermediate_output_ = encoder.predict(full_wavefields_with_damage)
    os.chdir('/home/aijjeh/Desktop/Phd_Projects/Modeling_guided_waves/Model_1_2_3_datasets/Model_2_dataset')
    np.save('predicted_Latent_space', intermediate_output_)
    # for test_case in range(475):
    #     print('layer shape', intermediate_output_[test_case].shape)
    #     pred_mat = {name: intermediate_output_[test_case]}
    #     scipy.io.savemat(name + '_%d.mat' % test_case, pred_mat)


#####################################################################################################################
# Encoder predictions
#####################################################################################################################
def pred_encoder():
    os.chdir('/home/aijjeh/Desktop/Phd_Projects/Modeling_guided_waves/temp/checkpoint/')
    encoder_name = 'model_1_capturing_latent_space-encoder_fully_ConvLSTM.h5'
    encoder_model = load_model(encoder_name, compile=False)
    encoder_model.summary()
    os.chdir('/home/aijjeh/Desktop/Phd_Projects/Modeling_guided_waves/Models_1_2_3_predictions/Model_1_predictions')
    intermediate_outputs('encoded_latent_space', encoder_model)
    # for i in range(6):
    #     intermediate_outputs('skip_connection_%d' % i, encoder_model)


########################################################################################################################
# Decoder predictions
########################################################################################################################
def pred_decoder():
    os.chdir('/home/aijjeh/Desktop/Phd_Projects/Modeling_guided_waves/temp/checkpoint/')
    test_case = 396
    #  Encoder of Model_I
    encoder_name_model_I = 'model_1_capturing_latent_space-encoder_fully_ConvLSTM.h5'
    encoder_model_model_I = load_model(encoder_name_model_I, compile=False)
    encoder_outputs_model_I = encoder_model_model_I.predict(full_wavefields_with_damage[test_case:test_case + 1])

    #  Encoder of Model_II
    # encoder_name_model_II = 'Model_2_encoder_mapping_GT_Ref_latent_space_skip_connections.h5'
    # encoder_model_model_II = load_model(encoder_name_model_II, compile=False)
    # encoder_outputs_model_II = encoder_model_model_II.predict([samples_gt_delamination[test_case:test_case + 1],
    #                                                            healthy_ref[test_case:test_case + 1]]) #

    #  Decoder of Model_I
    decoder_name = 'model_1_capturing_latent_space-decoder_fully_ConvLSTM.h5'
    decoder_model = load_model(decoder_name, compile=False)
    decoder_model.summary()

    decoder_prediction = decoder_model.predict(encoder_outputs_model_I)

    for i in range(32):
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
        ax[0].imshow(decoder_prediction[0][i])
        ax[0].axis('off')
        ax[1].imshow(full_wavefields_with_damage[test_case][i])
        ax[1].axis('off')
        plt.show()


pred_decoder()
# pred_encoder()
