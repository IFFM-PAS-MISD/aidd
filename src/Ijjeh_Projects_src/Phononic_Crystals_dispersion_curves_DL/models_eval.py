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

with open('hyper_par_custom_loss.json', 'r') as f:
    hyperparameters = json.load(f)

csv_eval_file = 'models_evaluation_n_size_custom_loss.csv'


def load_dataset():
    os.chdir(env_path + 'dataset/')
    X_train = np.load('train_xy_img_samples_%d.npy' % hyperparameters['samples'])
    Y_train = np.load('train_y_samples_%d.npy' % hyperparameters['samples'])
    if hyperparameters['normalised']:
        Y_train = Y_train[:, :, 0] / np.max(Y_train[:, :, 0])
        normal_ = 'normalised'
    else:
        Y_train = Y_train[:, :, 0]
        normal_ = 'not_normalised'

    Train_x = X_train[hyperparameters['n_size']:]
    Train_label = Y_train[hyperparameters['n_size']:]

    return Train_x, Train_label, normal_


def get_eval():
    x_ = []
    y_ = []
    for iteration in range(1, 15):
        x_test, y_test, s = load_dataset()

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = '2'

        os.chdir(env_path + 'temp/checkpoint/')
        NAME = 'iter_%d_PC_model_%s_%d_lr_%s_filters_%d_levels_%d_batches_%d_epochs_%d_dropout_%s_val_split_%s_hidden_%d.h5' % (
            iteration,
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

        print(NAME)
        model = load_model(NAME, compile=False)
        model.summary()
        score = model.evaluate(x_test, y_test, verbose=0, batch_size=32)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        x_.append(iteration * 475)
        y_.append(score[1])
        append_list_as_row(csv_eval_file, iteration, score[1])
    return x_, y_


if __name__ == '__main__':
    x, y = get_eval()
    fig = plt.figure(figsize=(10, 5))
    plt.plot(x, y, color='blue', marker='o', linestyle='dashed',
             linewidth=2, markersize=12)
    plt.xlabel('Dataset_size')
    plt.ylabel('loss (MSE)')
    plt.title('Eval with various dataset sizes')
    plt.show()

    os.chdir(env_path + 'num_results/')
    plt.savefig('Eval_with_various_dataset_sizes.png', bbox_inches='tight', transparent="True", pad_inches=0)
    plt.close()
