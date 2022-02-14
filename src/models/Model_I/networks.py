import numpy as np

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (ConvLSTM2D, BatchNormalization, Convolution3D, Convolution2D, Conv2D,
                          TimeDistributed, MaxPooling2D, UpSampling2D, Input,  AveragePooling3D)

def binary_net(input_shape):
    print(np.shape(input_shape))
    net = Sequential()
    net.add(ConvLSTM2D(filters=12, kernel_size=3, input_shape=input_shape,
                       padding='same', return_sequences=True))
    net.add(BatchNormalization())
    net.add(ConvLSTM2D(filters=6, kernel_size=3,
                       padding='same', return_sequences=True))
    net.add(BatchNormalization())
    net.add(ConvLSTM2D(filters=12, kernel_size=3,
                       padding='same', return_sequences=False))
    #net.add(BatchNormalization())
    # net.add(AveragePooling3D(pool_size=(5, 1, 1), padding='same'))
    net.add(Conv2D(filters=1, kernel_size=(3, 3), activation='sigmoid',
                     padding='same', data_format='channels_last'))
    return net
