import tensorflow as tf
import os
from IPython.display import Image, display
from tensorflow.keras.preprocessing.image import load_img
import PIL
from PIL import ImageOps
import re
import numpy as np
import keras
from keras import layers
from tensorflow.python.client import device_lib
import random
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from keras import backend as K
import neptune.new as neptune
from neptunecontrib.monitoring.keras import NeptuneMonitor
from keras.callbacks import Callback
from sklearn.model_selection import train_test_split
from neptune.new.integrations.tensorflow_keras import NeptuneCallback
from datetime import datetime
from tensorflow.keras import regularizers
import tensorflow_probability as tfp

now = datetime.now()

current_time = now.strftime("%H:%M:%S")

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

########################################################################################################################
# Link to neptune ai for monitoring
# ########################################################################################################################
# run = neptune.init(
#     project="abdalraheem.ijjeh/PhCs-dispersion-curves",
#     api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxOWE1Njk4NC03MWQxLTQwY2EtODJmMS1kZTczM2M1Y2VkMjkifQ==",
#     tags=['Vae model (AE+ANN) models']
# )
# ########################################################################################################################

os.chdir('/home/aijjeh/Desktop/Phd_Projects/PC_uint_cell_dispersion_curves/dataset/')
X_train = np.load('edge_detection_train_x.npy')
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))

print(X_train.shape)
Y_train = np.load('train_y_mat.npy')
print(Y_train.shape)

normalised = True
if normalised:
    Y_train = Y_train[:, :, 0] / 5.2667e5
    s = 'normalised'
else:
    Y_train = Y_train[:, :, 0]
    s = 'not_normalised'
print(s)

model = tf.keras.Sequential([
    tf.keras.layers.Reshape([32, 32, 1]),
    tfp.layers.Convolution2DReparameterization(
        64, kernel_size=5, padding='SAME', activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D(pool_size=[2, 2],
                                 strides=[2, 2],
                                 padding='SAME'),
    tf.keras.layers.Flatten(),
    tfp.layers.DenseReparameterization(1464),
])

n_size = 1900

features = X_train[:n_size]
labels = Y_train[:n_size]

logits = model(features)
neg_log_likelihood = tf.nn.softmax_cross_entropy_with_logits(
    labels=labels, logits=logits)
kl = sum(model.losses)
loss = neg_log_likelihood + kl
train_op = tf.train.AdamOptimizer().minimize(loss)
