import os
import numpy as np
import tensorflow as tf
import keras
import json
import matplotlib.pyplot as plt
import mat73

from tensorflow.python.client import device_lib
from tensorflow.keras.models import Model
from keras import backend as K
from keras.callbacks import Callback
from keras.models import load_model
from sklearn.model_selection import train_test_split


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

env_path = '/home/aijjeh/Desktop/Phd_Projects/Modeling_guided_waves/'
# Hyperparameters
os.chdir(env_path)

with open('hyper_par_GWM_mse.json', 'r') as f:
    params = json.load(f)


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

    x_train_1 = X_train_1[params['n_size']:]
    x2_ref_train = X_train_2[params['n_size']:]
    y_in_train = Y_train[params['n_size']:]

    # for count in range(x1_train.shape[0]):
    #     print(count)
    #
    #     x = x1_train[count][0]
    #     y = x1_train[count][1]
    #     a = x1_train[count][2]
    #     b = x1_train[count][3]
    #     theta = x1_train[count][-1]
    #
    #     theta = (theta * 2 * np.pi) / 360  # to rad
    #
    #     f0 = theta
    #
    #     t = np.linspace(0, 1, 512)
    #
    #     x_cords_Amp = (a / b) * np.sqrt((x - 0.25) ** 2 + (y - 0.25) ** 2)
    #
    #     x_cords_modulated = x_cords_Amp * (np.cos(2 * np.pi * f0 * t))
    #
    #     x2_ref_train[count] = x2_ref_train[count] * x_cords_modulated
    #
    return x_train_1, x2_ref_train, y_in_train


Input_B, Input_A, Y = load_dataset()

os.chdir(env_path + 'temp/checkpoint/')
NAME = 'Signal_based_modelling_%s_%d_lr_filters_%d_levels_%d_batches_%d_epochs_%d_dropout_%s_val_split_%s_hidden_%d.h5' % (
    params['samples'],
    params['learning_rate'],
    params['num_filters'],
    params['levels'],
    params['batches'],
    params['epochs'],
    params['dropout'],
    params['val_split'],
    params['hidden_layer'])

model = load_model(NAME, compile=False)
model.summary()

prediction = model.predict([Input_A, Input_B], batch_size=64)
print(prediction.shape)
os.chdir(env_path + 'num/')
np.save('prediction_num', prediction)

# arr_pred = np.load('prediction_num.npy')
# arr_pred = arr_pred.reshape((95, 1024, 512))
# arr_pred = arr_pred.transpose((0, -1, 1))
# arr_pred = arr_pred.reshape((95, 512, 32, 32))

# for i in range(0, 512, 16):
#     plt.imshow(arr_pred[0, i, :, :])
#     plt.show()
#
