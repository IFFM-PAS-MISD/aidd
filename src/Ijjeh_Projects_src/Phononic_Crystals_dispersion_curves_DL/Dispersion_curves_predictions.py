import os
import numpy as np

import tensorflow as tf
import keras
from tensorflow.python.client import device_lib
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from keras import backend as K
from keras.callbacks import Callback
import mat73
from keras.models import load_model
from sklearn.model_selection import train_test_split
from scipy.io import savemat
import json

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

# Hyperparameters
env_path = '/home/aijjeh/Desktop/Phd_Projects/PC_uint_cell_dispersion_curves/'
os.chdir(env_path)

# with open('hyper_par.json', 'r') as f:
#    hyperparameters = json.load(f)

with open('hyper_par_custom_loss.json', 'r') as f:
    hyperparameters = json.load(f)


def get_dataset():
    os.chdir(env_path + 'dataset/')

    X_train = np.load('train_xy_img_samples_%d.npy' % hyperparameters['samples'])
    # X_train = np.load('val_xy_img_samples_9.npy')

    Y_train = np.load('train_y_samples_%d.npy' % hyperparameters['samples'])
    max_value = np.max(Y_train)
    if hyperparameters['normalised']:
        normal = 'normalised'
    else:
        normal = 'not_normalised'

    test_x_samples = X_train[hyperparameters['n_size']:]
    test_y_samples = Y_train[hyperparameters['n_size']:]

    # Train_x, test_x_samples, Train_label, test_y_samples = train_test_split(X_train, Y_train,
    #                                                                         test_size=(1 - hyperparameters['n_size'] /
    #                                                                                    hyperparameters['samples']),
    #                                                                         shuffle=False,
    #                                                                         random_state=1988)

    return test_x_samples, test_y_samples, normal, max_value


# os.chdir(env_path + 'dataset/')
#
# X_train = np.load('train_xy_img_samples_%d.npy' % hyperparameters['samples'])
# Y_train = np.load('train_y_samples_%d.npy' % hyperparameters['samples'])
#
# x_test = X_train[hyperparameters['n_size']:]
# y_test = Y_train[hyperparameters['n_size']:]

x_test, y_test, s, maxvalue = get_dataset()

# for i in range(x_test.shape[0]):
#     plt.imshow(x_test[i])
#     plt.show()
# exit()
print(x_test.shape)
print(y_test.shape)

os.chdir(env_path + 'temp/checkpoint/')
NAME = 'iter_14_PC_model_%s_%d_lr_%s_filters_%d_levels_%d_batches_%d_epochs_%d_dropout_%s_val_split_%s_hidden_%d.h5' % (
    s,
    hyperparameters['samples'],
    hyperparameters['learning_rate'],
    hyperparameters['num_filters'],
    hyperparameters['levels'],
    hyperparameters['batches'],
    hyperparameters['epochs'],
    hyperparameters['dropout'],
    hyperparameters['val_split'],
    hyperparameters['hidden_layer'])

model = load_model(NAME, compile=False)
model.summary()

os.chdir(env_path + 'num_results/imgs_num_results/')

prediction = model.predict(x_test, batch_size=1)

if 'not' in NAME:
    prediction = prediction
    s = 'not'
else:
    prediction = prediction * maxvalue
    s = ''

for i in range(x_test.shape[0]):
    ####################################################################################################################
    os.chdir(env_path + 'num_results/imgs_num_results/mat_files')
    pred_mat = {'K': y_test[i, :, 1], 'F': prediction[i]}
    savemat('val_test_prediction_img_mat_%d.mat' % (i + 6651), pred_mat)
    ####################################################################################################################
    plt.figure(figsize=(4, 12))
    ax = plt.subplot(1, 1, 1)
    ax.scatter(y_test[i, :, 1], y_test[i, :, 0], marker='.', color=['b'], label='Reference')
    ax.scatter(y_test[i, :, 1], prediction[i], marker='.', color=['r'], label='Prediction')
    ax.legend(['Pred'])
    os.chdir(env_path + 'num_results/imgs_num_results/images')
    plt.savefig('val_test_input_img_predicted_dispersion_curve_case_%d_%s_normalized.png' % (i + 6651, s),
                bbox_inches='tight',
                transparent="True",
                pad_inches=0)
    plt.close()
    ####################################################################################################################
