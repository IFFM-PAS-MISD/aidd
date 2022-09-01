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

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

# Hyperparameters

samples = 7000
split_num = int(samples * 0.95)
normalised = True
batches = 32
filter_size = 8
img_shape = (256, 256, 1)
epochs = 30000
dropout = 0.2
levels = 8
learning_rate = 4e-4
patience_epochs = 10000
val_split = 0.08
s = 'normalised'

env_path = '/home/aijjeh/Desktop/Phd_Projects/PC_uint_cell_dispersion_curves/'

os.chdir(env_path + 'dataset/')

X_train = np.load('train_xy_img_samples_%d.npy' % samples)
Y_train = np.load('train_y_samples_%d.npy' % samples)

x_test = X_train[split_num:]
y_test = Y_train[split_num:]

os.chdir(env_path + 'temp/checkpoint/')
# NAME = 'best_model_%s_%d.h5' % (s, samples)
NAME = 'PC_model_%s_%d_lr_%s_filters_%d_levels_%d_batches_%d_epochs_%d_dropout_%s_val_split_%s.h5' % (s,
                                                                                                      samples,
                                                                                                      learning_rate,
                                                                                                      filter_size,
                                                                                                      levels,
                                                                                                      batches,
                                                                                                      epochs,
                                                                                                      dropout,
                                                                                                      val_split)

model = load_model(NAME, compile=False)
model.summary()

os.chdir(env_path + 'num_results/num_results/')

prediction = model.predict(x_test, batch_size=1)
if 'not' in NAME:
    prediction = prediction
    s = 'not'
else:
    prediction = prediction * np.max(Y_train)
    s = ''

for i in range(x_test.shape[0]):
    plt.figure(figsize=(4, 12))
    ax = plt.subplot(1, 1, 1)
    ax.scatter(y_test[i, :, 1], y_test[i, :, 0], marker='.', color=['b'], label='Reference')
    ax.scatter(y_test[i, :, 1], prediction[i], marker='.', color=['r'], label='Prediction')
    ax.legend(['GT', 'Pred'])
    plt.savefig('predicted_dispersion_cure_case_%d_%s_normalized.png' % (i + split_num, s), bbox_inches='tight',
                transparent="True",
                pad_inches=0)
    plt.close()
