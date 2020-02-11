import numpy as np
import gc
from keras.models import Sequential
import talos
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, UpSampling2D, Input, merge
from keras.models import Model
import keras
import tensorflow as tf
from sklearn.utils import shuffle
from keras.activations import relu, elu, sigmoid, softmax
from keras.optimizers import adam, RMSprop, sgd
from keras.losses import categorical_crossentropy, binary_crossentropy, mean_squared_error, \
    sparse_categorical_crossentropy

import talos as ta

p = {
    'lr': (2, 10, 30),
    'filters': [8, 16],
    'activation': [relu, elu, sigmoid],
    'batch_size': [8, 16, 32],
    'losses': [categorical_crossentropy, binary_crossentropy, mean_squared_error],
    'dropout': [0, 0.5],
    'last_activation': [softmax],
    'epochs': [20],
    'optimizer': [adam, RMSprop],
}

# Force the Garbage Collector to release unreferenced memory



gc.collect()

#####################################
x = np.load('E:/src/datasets/Segmentation datasets/augemented/Augmented_data_segmentation.npy')
x = x.reshape(1900, 512, 512, 1)
y = np.load('E:/src/datasets/Segmentation datasets/augemented/Augmented_target_segmentation.npy')
y = y.reshape(1900, 512, 512, 1)
#####################################
# Randomly shuffle the dataset
#####################################
x, y = shuffle(x, y)
#####################################
x_train = x[:1520]
y_train = y[:1520]
#####################################
test_x_samples = x[1520:]
tests_y_samples = y[1520:]


#####################################


def input_model(x_train, y_train):
    # model =
    inputs = Input(shape=(512, 512, 1))
    #####################################
    # Backbone Down sampling convolution followed by max-pooling
    #####################################
    c11 = Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(inputs)
    c13 = Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(c11)
    #####################################
    d1 = MaxPool2D((2, 2), (2, 2))(c13)
    #####################################
    c21 = Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(d1)
    c23 = Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(c21)
    #####################################
    d2 = MaxPool2D((2, 2), (2, 2))(c23)
    #####################################
    c31 = Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(d2)
    c33 = Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(c31)
    #####################################
    d3 = MaxPool2D((2, 2), (2, 2))(c33)
    #####################################
    c41 = Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(d3)
    c43 = Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(c41)
    #####################################
    d4 = MaxPool2D((2, 2), (2, 2))(c43)
    #####################################
    c51 = Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(d4)
    c53 = Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(c51)
    #####################################
    # Up sampling convolution followed by up-sampling
    #####################################
    u1 = UpSampling2D((2, 2))(c53)
    #####################################
    skip4 = keras.layers.Concatenate()([c43, u1])
    c61 = Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(skip4)
    c63 = Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(c61)
    #####################################
    u2 = UpSampling2D((2, 2))(c63)
    #####################################
    skip3 = keras.layers.Concatenate()([c33, u2])
    c71 = Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(skip3)
    c73 = Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(c71)
    #####################################
    u3 = UpSampling2D((2, 2))(c73)
    #####################################
    skip2 = keras.layers.Concatenate()([c23, u3])
    c81 = Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(skip2)
    c83 = Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(c81)
    #####################################
    u4 = UpSampling2D((2, 2))(c83)
    #####################################
    skip1 = keras.layers.Concatenate()([c13, u4])
    c91 = Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(skip1)
    c93 = Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(c91)
    #####################################
    # Output layer
    #####################################
    output = (Conv2D(1, (1, 1), padding='same', activation='sigmoid'))(c93)
    #####################################
    model = Model(inputs=inputs, outputs=output)
    #####################################

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['acc'])

    out = model.fit(x_train, y_train,
                    batch_size=16,
                    epochs=10,
                    validation_split=0.1)


    return model, out


model, out = input_model(x_train, y_train)
model.summary()
model.save('Unet_with_concatenation')
#####################################
