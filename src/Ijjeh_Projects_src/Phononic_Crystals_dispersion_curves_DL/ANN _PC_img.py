import os
from IPython.display import Image, display
from tensorflow.keras.preprocessing.image import load_img
import PIL
from PIL import ImageOps
import re
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from keras.layers import MaxPooling3D, Input, ConvLSTM2D, UpSampling2D, MaxPooling2D, Conv2D, Concatenate, Conv3D, \
    Dropout, BatchNormalization, Add, MaxPool2D, Conv2DTranspose, MaxPool3D, Conv3DTranspose, UpSampling3D, concatenate, \
    MaxPooling3D, Bidirectional, TimeDistributed
from tensorflow.python.client import device_lib
import random
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from keras import backend as K
import neptune
from keras.callbacks import Callback
from keras.utils import to_categorical

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

os.chdir('/home/aijjeh/Desktop/Phd_Projects/PC_uint_cell_dispersion_curves/dataset/')
x_train = np.load('train_x.npy')
y_train = np.load('train_y_GT_images.npy')

x_train = x_train[:450]
y_train = y_train[0:450]

x_test = x_train[450:]
y_test = y_train[450:]


# test_set = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batches)


# In[3]:


def conv_block(in_blk, num):
    layer11 = tf.keras.layers.Conv1D(num, 3,
                                     padding='same',
                                     activation='relu')(in_blk)
    layer12 = tf.keras.layers.Conv1D(num, 3,
                                     padding='same',
                                     activation='relu')(layer11)
    return tf.keras.layers.concatenate([layer11, layer12], axis=-1)


filter_size = 32
img_shape = (256, 256, 1)
epochs = 3000

#######################################################################################################################
# First CNN model
########################################################################################################################
# Encoder
########################################################################################################################
inputs_1 = tf.keras.Input(shape=img_shape)
blk1 = conv_block(inputs_1, filter_size)

DS1 = tf.keras.layers.MaxPool2D((2, 2))(blk1)

blk2 = conv_block(DS1, filter_size * 2)

DS2 = tf.keras.layers.MaxPool2D((2, 2))(blk2)

blk3 = conv_block(DS2, filter_size * 4)

DS3 = tf.keras.layers.MaxPool2D((2, 2))(blk3)

blk4 = conv_block(DS3, filter_size * 8)

DS4 = tf.keras.layers.MaxPool2D((2, 2))(blk4)
DS4 = tf.keras.layers.Conv2D(1, (1, 1), padding='same', activation='relu')(DS4)
print(tf.shape(DS4))
########################################################################################################################
# BottleNeck
########################################################################################################################
bottleneck = tf.keras.layers.Flatten()(DS4)
print(tf.shape(bottleneck))
bottleneck = tf.keras.layers.Dense(256, activation='relu')(bottleneck)
bottleneck = tf.keras.layers.Dense(256 * 2, activation='relu')(bottleneck)
bottleneck = tf.keras.layers.Dense(256 * 3, activation='relu')(bottleneck)
bottleneck = tf.keras.layers.Dense(256 * 2, activation='relu')(bottleneck)
bottleneck = tf.keras.layers.Dense(256, activation='relu')(bottleneck)
bottleneck = tf.keras.layers.Reshape((16, 16, 1))(bottleneck)
print(tf.shape(bottleneck))

########################################################################################################################
# Decoder
########################################################################################################################
UP1 = tf.keras.layers.UpSampling2D((2, 2))(bottleneck)
# blk5 = conv_block(UP1, filter_size * 8)
UP2 = tf.keras.layers.UpSampling2D((2, 1))(UP1)
# blk6 = conv_block(UP2, filter_size * 4)
UP3 = tf.keras.layers.UpSampling2D((2, 1))(UP2)
# blk7 = conv_block(UP3, filter_size * 2)
UP4 = tf.keras.layers.UpSampling2D((2, 1))(UP3)
# blk8 = conv_block(UP4, filter_size)
UP5 = tf.keras.layers.UpSampling2D((2, 2))(UP4)
# blk9 = conv_block(UP5, filter_size)

########################################################################################################################
# Output layer
########################################################################################################################
cnn_output = tf.keras.layers.Conv2D(1, (1, 1), padding='same', activation='sigmoid')(UP5)

cnn_model = tf.keras.models.Model(inputs=inputs_1, outputs=cnn_output)
cnn_model.summary()


# In[9]:


# ########################################################################################################################
# # Second CNN model
# ########################################################################################################################
# S_inputs_2 = tf.keras.Input(shape=(32, 32, 1))
# S_blk1 = conv_block(S_inputs_2, filter_size)
# # S_DS1 = tf.keras.layers.MaxPool2D(2, 2)(S_blk1)
# # S_blk2 = conv_block(S_DS1, filter_size * 2)
# # S_DS2 = tf.keras.layers.MaxPool2D(2, 2)(S_blk2)
# # S_blk3 = conv_block(S_DS2, filter_size * 4)
# # S_DS3 = tf.keras.layers.MaxPool2D(2, 2)(S_blk3)
# # S_blk4 = conv_block(S_DS3, filter_size * 8)
# # S_DS4 = tf.keras.layers.MaxPool2D(2, 2)(S_blk4)


# In[10]:


# ########################################################################################################################
# # BottleNeck
# ########################################################################################################################
# S_bottleneck = conv_block(S_blk1, filter_size * 16)
# S_bottleneck = tf.keras.layers.Flatten()(S_bottleneck)
# S_bottleneck = tf.keras.layers.Dense(1024, activation='relu')(S_bottleneck)
# S_bottleneck = tf.keras.layers.Dense(2*1024, activation='relu')(S_bottleneck)
# S_bottleneck = tf.keras.layers.Dense(1024, activation='relu')(S_bottleneck)
# S_bottleneck = tf.keras.layers.Reshape((32, 32, -1))(S_bottleneck)


# In[11]:


# ########################################################################################################################
# # Decoder
# ########################################################################################################################
# S_UP1 = tf.keras.layers.UpSampling2D((2, 2))(S_bottleneck)
# S_blk5 = conv_block(S_UP1, filter_size * 8)
# S_UP2 = tf.keras.layers.UpSampling2D((2, 1))(S_blk5)
# S_blk6 = conv_block(S_UP2, filter_size * 4)
# S_UP3 = tf.keras.layers.UpSampling2D((2, 1))(S_blk6)
# S_blk7 = conv_block(S_UP3, filter_size * 2)
# S_UP4 = tf.keras.layers.UpSampling2D((2, 1))(S_blk7)
# S_blk8 = conv_block(S_UP4, filter_size)


# In[12]:


# ########################################################################################################################
# # Output layer
# ########################################################################################################################
# s_cnn_output = Conv2D(1, (1, 1), padding='same')(S_blk8)
# cnn_model_2 = tf.keras.models.Model(S_inputs_2, s_cnn_output, name="CNN_model2")
# cnn_model_2.summary()


# In[13]:


# ########################################################################################################################
# # Variational AutoEncoder (VAE) model
# ########################################################################################################################
#
# vae_input = tf.keras.layers.Input(shape=img_shape, name="VAE_input")
# vae_decoder_output = cnn_model_2(cnn_model(vae_input))
# vae = Model(vae_input, vae_decoder_output)
# vae.summary()


# In[14]:


########################################################################################################################
# Model compile and training
########################################################################################################################

def custom_loss(y_true, y_pred):
    f_pred = y_pred[:, :, 0]
    k_pred = y_pred[:, :, 1]

    f_true = y_true[:, :, 0]
    k_true = y_true[:, :, 1]
    # loss = 0
    # for j in range(1464):
    #     loss += K.square(tf.math.pow((f_pred[0][j] - f_true[0][j]), 2) + tf.math.pow((k_pred[0][j] - k_true[0][j]), 2))
    return tf.losses.mean_squared_error(f_true, f_pred) + tf.losses.mean_squared_error(k_true, k_pred)


cnn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
                  loss='mse',
                  metrics=[tf.keras.metrics.MeanSquaredError()],
                  run_eagerly=True)

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, min_delta=1e-5)
cnn_model.fit(x_train, y_train,
              batch_size=32,
              validation_split=0.1,
              epochs=epochs,
              callbacks=[callback])

os.chdir('/home/aijjeh/Desktop/Phd_Projects/PC_uint_cell_dispersion_curves/h5_models/')
cnn_model.save("VAE_ANN_PC_uint_cell_img_to_img.h5")
