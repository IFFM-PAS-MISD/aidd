import os
import numpy as np
import json
import csv
import tensorflow as tf
import neptune.new as neptune
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from keras import backend as K
from keras.callbacks import Callback
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score
from decouple import config
from sklearn.metrics import mean_squared_error
import cv2

access_token = config('NEPTUNE_API_TOKEN')

########################################################################################################################
run = neptune.init_run(project='abdalraheem.ijjeh/Guided-waves-modelling',
                       api_token=access_token,
                       tags=['Dataset preparation']
                       )

n_size = 194560


def gen_LR_ref():
    path = '/aijjeh_odroid_sensors/aidd/data/raw/num/wavefield_dataset_undelam_bottom_out/1_output/'
    os.chdir(path)
    arr_ref = np.zeros((512, 32, 32), dtype=np.float32)
    for frame in range(512):
        img = plt.imread('%d_flat_shell_Vz_1_500x500bottom.png' % (frame + 1), 0)
        img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_CUBIC)
        arr_ref[frame] = img
    os.chdir('/home/aijjeh/Desktop/Phd_Projects/Modeling_guided_waves/dataset')
    return arr_ref


def gen_LR_GT():
    os.chdir('/pkudela_odroid_sensors/aidd/data/raw/num/dataset2_labels_out')
    # Coordinates = np.genfromtxt('dataset2_labels.csv', delimiter=',')

    temp = np.zeros((475, 32, 32), dtype=np.float32)
    for i in range(475):
        img = plt.imread('m1_rand_single_delam_%d.png' % (i + 1), 0)
        img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_CUBIC)
        temp[i] = img
    #     temp[i, 0] = Coordinates[i + 1][1]
    #     temp[i, 1] = Coordinates[i + 1][2]
    #     temp[i, 2] = Coordinates[i + 1][3]
    #     temp[i, 3] = Coordinates[i + 1][4]
    #     temp[i, 4] = Coordinates[i + 1][5]
    #
    # print(temp.shape[0])
    # print(temp)
    os.chdir('/home/aijjeh/Desktop/Phd_Projects/Modeling_guided_waves/dataset')

    return temp


def gen_labels():
    arr_GT = np.zeros((475, 512, 32, 32), dtype=np.float32)
    for output_case in range(475):
        print(output_case)
        os.chdir(
            '/aijjeh_odroid_sensors/aidd/data/raw/num/wavefield_dataset2_bottom_out/%d_output/' % (output_case + 1))
        for frame in range(512):
            img = plt.imread('%d_flat_shell_Vz_%d_500x500bottom.png' % (frame + 1, output_case + 1), 0)
            img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_CUBIC)
            arr_GT[output_case][frame] = img

            # plt.imshow(arr_GT[output_case][frame])
            # plt.show()

    os.chdir('/home/aijjeh/Desktop/Phd_Projects/Modeling_guided_waves/dataset')
    return arr_GT


def load_dataset():
    os.chdir('/home/aijjeh/Desktop/Phd_Projects/Modeling_guided_waves/dataset')
    X_train_1 = np.load('LR_GT_del.npy')
    X_train_1 = np.reshape(X_train_1, (475, 5, 1))
    X_train_1 = np.transpose(X_train_1, [0, -1, 1])
    X_train_1 = np.repeat(X_train_1, 1024, axis=1)
    X_train_1 = np.reshape(X_train_1, (1024 * 475, 5))
    print(X_train_1.shape)

    X_train_2 = np.load('LR_ref_frames.npy')
    print(X_train_2.shape)
    Y_train = np.load('LR_labels.npy')
    print(Y_train.shape)

    x1_train = X_train_1[:64]
    x2_ref_train = X_train_2[:64]
    y_in_train = Y_train[:64]

    for count in range(x1_train.shape[0]):
        print(count)

        x = x1_train[count][0]
        y = x1_train[count][1]
        a = x1_train[count][2]
        b = x1_train[count][3]
        theta = x1_train[count][-1]

        theta = (theta * 2 * np.pi) / 360  # to rad

        f0 = theta

        t = np.linspace(0, 1, 512)

        x_cords_Amp = (a / b) * np.sqrt((x - 0.25) ** 2 + (y - 0.25) ** 2)

        x_cords_modulated = x_cords_Amp * (np.cos(2 * np.pi * f0 * t))

        x2_ref_train[count] = x2_ref_train[count] * x_cords_modulated

    return x2_ref_train, y_in_train


os.chdir('/home/aijjeh/Desktop/Phd_Projects/Modeling_guided_waves/dataset')

# np.save('LR_ref_frames', gen_LR_ref())
# np.save('LR_GT_labels_img', gen_LR_GT())
# np.save('LR_labels', gen_labels())

# exit()

arr = np.load('LR_GT_labels_img.npy')
delam_coords = arr.reshape((475, 1, 32, 32)).astype('uint8')
delam_coords = np.repeat(delam_coords, 512, axis=1)
delam_coords = (np.invert(delam_coords.reshape((475 * 64, 8, 32, 32, 1))) - 254)


for i in range(0, 475 * 64, 64):
    for j in range(0, 8, 8):
        plt.imshow(delam_coords[i][j])
        plt.show()
exit()

x2, y = load_dataset()

for i in range(0, 512, 8):
    signal_input = x2[i, :]
    signal_gt = y[i, :]

    signal_input = np.fft.fft(signal_input)
    signal_gt = np.fft.fft(signal_gt)

    N = signal_input.size
    print(N)
    signal_input = signal_input / (N // 2)
    signal_input = np.fft.fftshift(signal_input)

    signal_gt = signal_gt / (N // 2)
    signal_gt = np.fft.fftshift(signal_gt)

    freq = np.fft.fftfreq(N)
    fig, [ax1, ax2, ax3, ax4, ax5] = plt.subplots(5, 1, figsize=(15, 9))
    ax1.plot(freq[:N // 2], abs(signal_input[:N // 2]))
    ax1.set_title('FFT(input signal)')
    ax2.plot(freq[:N // 2], abs(signal_gt[:N // 2]))
    ax2.set_title('FFT(Label signal)')
    ax3.plot(freq[:N // 2], abs((signal_input[:N // 2]) - (signal_gt[:N // 2])))
    ax3.set_title('Difference')
    ax4.plot(x2[i, :])
    ax4.set_title('INPUT')
    ax5.plot(y[i, :])
    ax5.set_title('LABEL')
    plt.show()
