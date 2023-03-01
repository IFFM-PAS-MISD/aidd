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
# Loading dataset into x, y arrays and reshape them
#####################################################################################################################
# class Full_wavefield_frames(tf.keras.utils.Sequence):
#     def __init__(self, batch_size_, img_size_, input_img_paths_total_, target_img_paths_, time_stamps_):
#         self.batch_size = batch_size_
#         self.img_size = img_size_
#         self.input_img_paths_total = input_img_paths_total_
#         self.target_img_paths = target_img_paths_
#         self.time_stamps = time_stamps_
#
#     def __len__(self):
#         return len(self.target_img_paths) // self.batch_size
#
#     def __getitem__(self, idx):
#         """Returns tuple (input, target) correspond to batch #idx."""
#         index_ = idx * self.batch_size
#         batch_input_img_paths = self.input_img_paths_total[index_:index_ + self.batch_size]
#         batch_target_img_paths = self.target_img_paths[index_:index_ + self.batch_size]
#
#         x = np.zeros((self.batch_size,) + (self.time_stamps,) + self.img_size + (1,), dtype="float16")  #
#
#         for batch_num in range(self.batch_size):
#             batch_input_img_paths = batch_input_img_paths[batch_num][
#                                     file_frame[index_] - self.time_stamps // 2: file_frame[
#                                                                                     index_] + self.time_stamps // 2]
#             for j, path in enumerate(batch_input_img_paths):
#                 img_sample = load_img(path, target_size=self.img_size, color_mode="grayscale")
#                 img_sample = np.expand_dims(img_sample, 2)
#                 img_sample = img_sample / 255.0
#                 x[batch_num][j] = img_sample
#         # y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="float16")
#         # for j, path in enumerate(batch_target_img_paths):
#         #     img_sample = load_img(path, target_size=self.img_size, color_mode="grayscale")
#         #     img_sample = np.expand_dims(img_sample, 2)
#         #     img_sample = img_sample / 255.0
#         #     y[j] = img_sample
#         #     # Ground truth labels are 1, 2, 3. Subtract one to make them 0, 1, 2:
#         #     # y[j] -= 1
#         return x


os.chdir('/home/aijjeh/Desktop/Phd_Projects/Modeling_guided_waves/Model_1_2_3_datasets/Model_1_dataset')
gt = np.load('GT_full_wavefields.npy')

x = gt
y = gt

x = np.expand_dims(x, axis=-1)
print(x.shape)
# print(len(test_gen))
########################################################################################################################
model_name = '/home/aijjeh/Desktop/Phd_Projects/Modeling_guided_waves/temp/checkpoint/model_1_capturing_latent_space.h5'
model = load_model(model_name, compile=False)
model.summary()


########################################################################################################################

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


def intermediate_outputs(name):
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer(name).output)
    intermediate_output_ = intermediate_layer_model.predict(x)
    for test_case in range(475):
        print('layer shape', intermediate_output_[test_case].shape)
        pred_mat = {name: intermediate_output_[test_case]}
        scipy.io.savemat(name + '_%d.mat' % test_case, pred_mat)
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


os.chdir('/home/aijjeh/Desktop/Phd_Projects/Modeling_guided_waves/Models_1_2_3_predictions/Model_1_predictions')
# # Path(str(input_image)).mkdir(parents=True, exist_ok=True)
# # os.chdir('/home/aijjeh/Desktop/Phd_Projects/Modeling_guided_waves/latent_space_pred/%d' % input_image)
# # for layers_count in range(len(model.layers)):
# #     print(layers_count, '/', len(model.layers))
# # layer_name = model.layers[58].get_config()
# # layer_name = layer_name['name']
# # print(layer_name)
# # if layer_name == 'latent_space':
# # plot_filters(model.layers[58], 'latent_space')
# os.chdir('/home/aijjeh/Desktop/Phd_Projects/Modeling_guided_waves/latent_space_pred/%d' % input_image)
intermediate_outputs('latent_space')
for i in range(6):
    intermediate_outputs('skip_connection_%d' % i)

# intermediate_output = model.predict(x)
# print('layer shape', intermediate_output.shape)
# for i in range(32):
#     fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), dpi=100)
#     ax[0].imshow(intermediate_output[0][i])
#     ax[0].axis('off')
#     ax[1].imshow(y[0][i])
#     ax[1].axis('off')
#     plt.show()

#####################################################################################################################
