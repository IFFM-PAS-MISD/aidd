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


def to_inverse_short_time_FT(x):
    x = tf.cast(x, tf.complex64)
    istft = tf.signal.inverse_stft(
        x,
        frame_length=256,
        frame_step=256,
    )
    return tf.dtypes.cast(istft, tf.float32)


########################################################################################################################
# Load dataset
########################################################################################################################
dataset_path = '/home/aijjeh/Desktop/Phd_Projects/Sequence_prediction/Sequence_to_sequence/Dataset_Full_wavefield_outputs_bottom/'
os.chdir(dataset_path)
dataset_x_GT = np.load('Refrence_waves_GT_64_frame_475_512_512_65.npy', mmap_mode='r+')
dataset_y_RMS = np.load('Full_wavefield_labels_64_frame_475_64_512_512.npy', mmap_mode='r+')

print(dataset_x_GT.shape)
print(dataset_y_RMS.shape)

x_test = dataset_x_GT[380:]
y_test = dataset_y_RMS[380:]

test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
BATCH_SIZE = 1
test_dataset = test_dataset.batch(BATCH_SIZE)

x_test = np.transpose(x_test, [0, 3, 1, 2])

print(x_test.shape)
print(y_test.shape)
####################################################################################################################
# Loading model
########################################################################################################################
os.chdir('/home/aijjeh/Desktop/Phd_Projects/Sequence_prediction/Sequence_to_sequence/h5_models/')
model = 'VAE_seq2seq_600_new.h5'
Conv3D_model = tf.keras.models.load_model(model, compile=False)
Conv3D_model.summary()
os.chdir('/home/aijjeh/Desktop/Phd_Projects/Sequence_prediction/GT_RMS_waves/num_results')

path_to_csv = "E:/aidd_new/aidd/src/data_processing/PhD/cmap_flipped_jet256.csv"
cmap = matplotlib.colors.ListedColormap(["blue", "green", "red"], name=path_to_csv, N=None)

output_dir = 'FT_Nonlinear_Operator_%d' % 1


########################################################################################################################
# Testing function
########################################################################################################################

def Test_sample():
    prediction = Conv3D_model.predict(test_dataset, batch_size=1)
    for i in range(95):
        sequence_frames = prediction[i]
        print(sequence_frames.shape)
        label_frames = y_test[i]
        print(label_frames.shape)
        sample = x_test[i]
        for f in range(64):
            predicted_frame = sequence_frames[f]
            predicted_frame = predicted_frame.astype('float32')
            label_img = label_frames[f]
            label_img = label_img.astype('float32')
            sample_frame = sample[f]
            sample_frame = sample_frame.astype('float32')
            print(sample_frame.shape, predicted_frame.shape, label_img.shape)

            plt.figure(figsize=(15, 5))

            ax1 = plt.subplot(1, 3, 1)
            ax1.set_title('Input sample')
            ax1.imshow(sample_frame)
            plt.axis('off')

            ax2 = plt.subplot(1, 3, 2)
            ax2.set_title('Prediction')
            ax2.imshow(predicted_frame)
            plt.axis('off')

            ax3 = plt.subplot(1, 3, 3)
            ax3.set_title('Full wavefield frames')
            ax3.imshow(label_img)
            plt.axis('off')

            os.chdir(
                '/home/aijjeh/Desktop/Phd_Projects/Sequence_prediction/Sequence_to_sequence/prediction_frames_with_GT/' + output_dir + '/frames/')
            plt.savefig('frame_testsample_%d_frame_%d.png' % (i + 381, f))
            plt.close('all')
        convert_to(i)


def convert_to(q):
    frame_array = []
    Img_size = None

    for z in range(64):
        os.chdir(
            '/home/aijjeh/Desktop/Phd_Projects/Sequence_prediction/Sequence_to_sequence/prediction_frames_with_GT/' + output_dir + '/frames/')
        img = cv2.imread('frame_testsample_%d_frame_%d.png' % (q + 381, z))
        height, width, channels = img.shape
        print(img.shape)
        Img_size = (width, height)
        frame_array.append(img)
    os.chdir(
        '/home/aijjeh/Desktop/Phd_Projects/Sequence_prediction/Sequence_to_sequence/prediction_frames_with_GT/' + output_dir + '/videos/')
    out = cv2.VideoWriter('output_%d.avi' % (q + 381), cv2.VideoWriter_fourcc(*'DIVX'), 4, Img_size)
    for i in range(len(frame_array)):
        out.write(frame_array[i])
    out.release()


########################################################################################################################
# main function
########################################################################################################################

if __name__ == '__main__':
    Test_sample()
