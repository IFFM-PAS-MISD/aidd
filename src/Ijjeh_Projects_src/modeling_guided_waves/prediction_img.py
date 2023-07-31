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

# hyperparameters
samples = 475
n_size = 194560
learning_rate = 2e-3
features_shape = (512,)
hidden_layers = 3
dropout = 0.2
batch_size = 32
val_split = 0.1
epochs = 200
env_path = '/home/aijjeh/Desktop/Phd_Projects/Modeling_guided_waves/'


def load_dataset():
    os.chdir('/home/aijjeh/Desktop/Phd_Projects/Modeling_guided_waves/dataset')

    X_train_1 = np.load('LR_GT_del_images.npy')
    X_train_1 = np.reshape(X_train_1, (475 * 512, 4))

    X_train_2 = np.load('LR_ref_frames_images.npy')
    X_train_2 = np.reshape(X_train_2, (475 * 512, 32, 32))

    Y_train = np.load('LR_labels_images.npy')
    Y_train = np.reshape(Y_train, (475 * 512, 32, 32))

    X1_train = X_train_1[:n_size]
    X2_train = X_train_2[:n_size]
    y_in_train = Y_train[:n_size]
    return X2_train, X1_train, y_in_train


x1, x2, y1 = load_dataset()

os.chdir(env_path + 'temp/checkpoints/')
NAME = 'model_images.h5'

model = load_model(NAME, compile=False)
model.summary()

prediction = model.predict([x1, x2], batch_size=1)
print(prediction.shape)
os.chdir(env_path + 'num/')
np.save('prediction_num_img', prediction)
