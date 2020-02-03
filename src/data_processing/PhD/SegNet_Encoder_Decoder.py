import numpy as np
import cv2
from glob import glob
import gc
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, UpSampling2D, Input, merge
from keras.models import Model
from keras.optimizers import adam, rmsprop, sgd

# Force the Garbage Collector to release unreferenced memory
from sklearn.utils import shuffle

gc.collect()

#####################################
x = np.load('E:/src/datasets/Segmentation datasets/augemented/Augmented_data_segmentation_New_updates.npy')
x = x.reshape(1900, 512, 512, 1)
y = np.load('E:/src/datasets/Segmentation datasets/augemented/Augmented_target_segmentation.npy')
y = y.reshape(1900, 512, 512, 1)
#####################################
# Randomly shuffle the dataset
#####################################
x, y = shuffle(x,y)
#####################################
x_train = x[:1520]
y_train = y[:1520]
#####################################
test_x_samples = x[1520:]
tests_y_samples = y[1520:]
#####################################
inputs = Input(shape=(512, 512, 1))
#####################################
# Backbone Down sampling convolution followed by max-pooling
#####################################
c11 = Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(inputs)
c13 = Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(c11)
BN1 = keras.layers.BatchNormalization()(c13)
#output1 = keras.layers.concatenate([c11, c13], axis=1)

#####################################
d1 = MaxPool2D((2, 2), (2, 2))(BN1)
#####################################
c21 = Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(d1)
c23 = Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(c21)
BN2 = keras.layers.BatchNormalization()(c23)
#output2= keras.layers.concatenate([c21,c23], axis=1)
#####################################
d2 = MaxPool2D((2, 2), (2, 2))(BN2)
#####################################
c31 = Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(d2)
c32 = Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(c31)
c33 = Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(c32)
BN3 = keras.layers.BatchNormalization()(c33)
#####################################
d3 = MaxPool2D((2, 2), (2, 2))(BN3)
#####################################
c41 = Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(d3)
c42 = Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(c41)
c43 = Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(c42)
BN4 = keras.layers.BatchNormalization()(c43)

#####################################
d4 = MaxPool2D((2, 2), (2, 2))(BN4)
#####################################
c51 = Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(d4)
c52 = Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(c51)
c53 = Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(c52)
BN5 = keras.layers.BatchNormalization()(c53)

#####################################
d5 = MaxPool2D((2, 2), (2, 2))(BN5)
#####################################
c61 = Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(d5)
c62 = Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(c61)
c63 = Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(c62)
BN6 = keras.layers.BatchNormalization()(c63)

#####################################
# Up sampling convolution followed by up-sampling
#####################################
u1 = UpSampling2D((2, 2))(BN6)
#####################################
skip5 = keras.layers.Concatenate()([c53,u1])
c71 = Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(skip5)
c72 = Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(c71)
c73 = Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(c72)
BN7 = keras.layers.BatchNormalization()(c73)

#####################################
u2 = UpSampling2D((2, 2))(BN7)
#####################################
skip4 = keras.layers.Concatenate()([c43,u2])
c81 = Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(skip4)
c82 = Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(c81 )
c83 = Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(c82)
BN8 = keras.layers.BatchNormalization()(c83)

#####################################
u3 = UpSampling2D((2, 2))(BN8)
#####################################
skip3 = keras.layers.Concatenate()([c33,u3])
c91 = Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(skip3)
c92 = Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(c91)
c93 = Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(c92)
BN9 = keras.layers.BatchNormalization()(c93)

#####################################
u4 = UpSampling2D((2, 2))(BN9)
#####################################
skip2 = keras.layers.Concatenate()([c23,u4])
c101 = Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='elu')(skip2)
c103 = Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='elu')(c101)
BN10 = keras.layers.BatchNormalization()(c103)

#####################################
u5 = UpSampling2D((2, 2))(BN10)
#####################################
skip1 = keras.layers.Concatenate()([c13,u5])
c111 = Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(skip1)
c113 = Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(c111)
BN11 = keras.layers.BatchNormalization()(c113)

#####################################
# Output layer
#####################################
output = (Conv2D(1, (1, 1), padding='same', activation='sigmoid'))(BN11)
#####################################
model = Model(inputs=inputs, outputs=output)
#####################################
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.fit(np.array(x_train), np.array(y_train), batch_size=16, epochs=5, validation_split=0.1)
model.summary()
model.save('SegNet_encoder_decoder_new_data.h5')
#####################################