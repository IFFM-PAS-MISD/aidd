import os
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from tensorflow.python.client import device_lib
from tensorflow.keras.models import Model
from keras import backend as K
from keras.callbacks import Callback
from tensorflow.keras import regularizers
from keras.layers import Layer
from rbflayer import RBFLayer, InitCentersRandom
import matplotlib.pyplot as plt

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

filter_size = 8
epochs = 20000
dropout = 0.2
levels = 8

env_path = '/home/aijjeh/Desktop/Phd_Projects/PC_uint_cell_dispersion_curves/'
os.chdir(env_path + 'dataset/')

samples = 6151

X_train = np.load('train_xy_coordinates_samples_%d.npy' % (samples - 1000))
print(X_train.shape)

Y_train = np.load('train_y_samples_%d.npy' % samples)
Y_train = Y_train[1000:]
print(Y_train.shape)

normalised = True
if normalised:
    Y_train = Y_train[:, :, 0] / 528925.8888046806678

    s = 'normalised'
else:
    Y_train = Y_train[:, :, 0]
    s = 'not_normalised'

n_size = 4636

x_train = X_train[:n_size]
y_train = Y_train[:n_size]

x_test = X_train[n_size:]
y_test = Y_train[n_size:]

# batches = 32
# train_set = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batches)
# # val_set = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batches)
# test_set = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batches)


# In[3]:


########################################################################################################################
# RBNN
########################################################################################################################
inputs_1 = tf.keras.Input(shape=(121, 2))
x = tf.keras.layers.Flatten()(inputs_1)
x = tf.keras.layers.Dense(121, activation='relu')(x)
x = tf.keras.layers.Dense(1464, activation='relu')(x)
x = tf.keras.layers.Dropout(dropout)(x)
x = tf.keras.layers.Dense(1464, activation='relu')(x)
x = tf.keras.layers.Dropout(dropout)(x)

output = tf.keras.layers.Dense(1464)(x)
model = tf.keras.models.Model(inputs=inputs_1, outputs=output, name='AE_Model')
model.summary()


def custom_loss(y_true, y_pred):
    return tf.keras.losses.MSE(y_true, y_pred)


lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
    0.0007,
    decay_steps=50 * 1000,
    decay_rate=1,
    staircase=False)

model.compile(tf.keras.optimizers.Adam(lr_schedule),
              loss=custom_loss,
              metrics=[tf.keras.metrics.MeanSquaredError()])

checkpoint_filepath = env_path + 'temp/checkpoint/RBNN_best_model_%s_%d.h5' % (s, samples)

callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                              patience=2000,
                                              mode='min'),
             tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                monitor='val_loss',
                                                save_best_only=True)]

model.fit(x=x_train,
          y=y_train,
          batch_size=32,
          validation_split=0.1,
          epochs=epochs,
          callbacks=[callbacks])

os.chdir(env_path + 'h5_models/')
model.save("RBNN_%d.h5" % samples)
