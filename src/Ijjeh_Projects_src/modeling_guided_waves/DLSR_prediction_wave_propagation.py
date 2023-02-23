import csv
import gc
import cv2
from tensorflow import keras
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
from tensorflow.python.client import device_lib
from PIL import Image
import tensorflow as tf
import os
from keras.layers import LeakyReLU
from pathlib import Path
import matplotlib
from matplotlib import cm
import itertools

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

print(device_lib.list_local_devices())

env_path = '/home/aijjeh/Desktop/Phd_Projects/Modeling_guided_waves/'


########################################################################################################################
# Load dataset
########################################################################################################################
def load_dataset():
    os.chdir('/home/aijjeh/Desktop/Phd_Projects/Modeling_guided_waves/num/')
    arr_pred = np.load('Encoder_ANN_coords_prediction_num.npy')
    print(arr_pred.shape)
    # arr_pred = arr_pred.reshape((95, 1024, 512))
    # arr_pred = arr_pred.transpose((0, -1, 1))
    # arr_pred = arr_pred.reshape((95, 512, 32, 32))
    arr_pred = arr_pred.reshape((95 * 512, 32, 32, 1))
    # for i in range(95):
    #     print(arr_pred)
    #     plt.imshow(arr_pred[i])
    #     plt.show()
    return arr_pred


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


def testing():
    path_GT = '/aijjeh_odroid_sensors/aidd/data/raw/num/wavefield_dataset2_bottom_out/'
    path = '/home/aijjeh/Desktop/Phd_Projects/Modeling_guided_waves/num/'
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)

    for output_case in range(1, 96):
        ##################################################################
        path_GT_ = path_GT + '%d_output/' % (output_case + 380)
        ##################################################################
        x_test = load_dataset()
        path_ = path + '/Output_%d' % (output_case + 380)
        try:
            os.mkdir(path_)
        except OSError:
            print("Creation of the directory %s failed" % path_)
        else:
            print("Successfully created the directory %s " % path_)

        x_test = x_test[(output_case - 1) * 512:(output_case - 1) * 512 + 512]
        print(x_test.shape)
        prediction = model.predict(x_test, batch_size=4)
        prediction = np.asarray(prediction)
        print(prediction.shape)
        frames = 512
        for i in range(frames):
            ############################################################################################################
            os.chdir(path_GT_)
            frame = cv2.imread('%d_flat_shell_Vz_%d_500x500bottom.png' % ((i + 1), (output_case + 380)), 0)
            frame = cv2.resize(frame, (512, 512))
            ############################################################################################################
            os.chdir(path_)
            SR_pred = prediction[i].astype('float32')
            ############################################################################################################
            plt.figure(figsize=(3, 1.45), dpi=512)
            plt.gca().set_axis_off()
            plt.axis('off')
            plt.subplots_adjust(left=0.0, bottom=0.0, right=1, top=1.0, wspace=0.0, hspace=0.0)
            plt.margins(0, 0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())

            ax1 = plt.subplot(1, 2, 1)
            ax1.set_title('prediction')
            ax1.imshow(SR_pred)
            plt.axis('off')

            ax2 = plt.subplot(1, 2, 2)
            ax2.set_title('GT')
            ax2.imshow(frame.astype('float32'))
            plt.axis('off')

            plt.savefig('Ijjeh_SR_output_%d_frame_%d' % ((output_case + 380), (i + 1)))
            print((output_case + 380), i)
            plt.close('all')


if __name__ == '__main__':
    testing()
