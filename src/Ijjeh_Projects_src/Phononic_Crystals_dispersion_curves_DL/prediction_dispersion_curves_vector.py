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
import json
from scipy.io import savemat

# Hyperparameters
env_path = '/home/aijjeh/Desktop/Phd_Projects/PC_uint_cell_dispersion_curves/'
os.chdir(env_path)
########################################################################################################################
with open('hyper_par_vectors.json', 'r') as f:
    hyperparameters = json.load(f)
########################################################################################################################
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'


########################################################################################################################


def get_dataset():
    envPath = '/home/aijjeh/Desktop/Phd_Projects/PC_uint_cell_dispersion_curves/'
    os.chdir(envPath + 'dataset/')

    # X_train = np.load('train_xy_coordinates_samples_%d.npy' % (hyperparameters['samples'] + 1000))
    X_train = np.load('val_xy_coordinates_samples_%d.npy' % 9)
    # x_coordinates = X_train[:, :, 0]
    # y_coordinates = X_train[:, :, 1]
    # X_train = x_coordinates * 10 + y_coordinates
    print(X_train.shape)

    y_train = np.load('train_y_samples_%d.npy' % (hyperparameters['samples'] + 1000))
    y_train = y_train[1000:]
    print(y_train.shape)
    if hyperparameters['normalised']:
        normal = 'normalised'
    else:
        normal = 'not_normalised'
    print(normal)

    test_x_samples = X_train  # [:]
    test_y_samples = y_train  # [:]

    # Train_x, test_x_samples, Train_label, test_y_samples = train_test_split(X_train, y_train,
    #                                                                         test_size=(1 - (hyperparameters['n_size'] /
    #                                                                                         hyperparameters[
    #                                                                                             'samples'])),
    #                                                                         shuffle=True,
    #                                                                         random_state=1988)

    return test_x_samples, test_y_samples, normal, np.max(y_train)


########################################################################################################################

x_test, y_test, s, Y_train = get_dataset()
print(x_test.shape)
print(y_test.shape)

os.chdir('/home/aijjeh/Desktop/Phd_Projects/PC_uint_cell_dispersion_curves/temp/checkpoint/')

model_name = 'PC_model_vectors_%s_%d_lr_%s_filters_%d_levels_%d_batches_%d_epochs_%d_dropout_%s_val_split_%s_hidden_%d.h5' % (
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

model = load_model(model_name, compile=False)
model.summary()

prediction = model.predict(x_test, batch_size=1)

if 'not' in model_name:
    prediction = prediction
    s = 'not'
else:
    prediction = prediction * np.max(Y_train)
    s = '_'

for i in range(x_test.shape[0]):
    ####################################################################################################################
    os.chdir('/home/aijjeh/Desktop/Phd_Projects/PC_uint_cell_dispersion_curves/num_results/vectors_inputs/mat_files')
    pred_mat = {'K': y_test[i, :, 1], 'F': prediction[i]}
    savemat('Val_test_prediction_polygon_mat_%d.mat' % (i + 1), pred_mat)  # hyperparameters['n_size'] +
    ####################################################################################################################
    os.chdir('/home/aijjeh/Desktop/Phd_Projects/PC_uint_cell_dispersion_curves/num_results/vectors_inputs/polygon_pred')
    plt.figure(figsize=(4, 12))
    ax = plt.subplot(1, 1, 1)
    # ax.scatter(y_test[i, :, 1], y_test[i, :, 0], marker='.', color=['b'], label='Reference')
    ax.scatter(y_test[i, :, 1], prediction[i], marker='.', color=['b'], label='Prediction')
    ax.legend(['GT', 'Pred'])
    plt.savefig('Val_test_input_polygon_predicted_dispersion_curve_case_%d_%s_normalized.png' % (
        i + 1, s),  # hyperparameters['n_size'] +
                bbox_inches='tight',
                transparent="True",
                pad_inches=0)
    plt.close()
    ####################################################################################################################
