import numpy as np
import gc
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, UpSampling2D, Input, merge
from keras.models import Model
import keras
import tensorflow as tf
from sklearn.utils import shuffle


config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 64} )
sess = tf.Session(config=config)
keras.backend.set_session(sess)

# Force the Garbage Collector to release unreferenced memory

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
c12 = Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(c11)
c13 = Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(c12)
#####################################
d1 = MaxPool2D((2, 2), (2, 2))(c13)
#####################################
c21 = Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(d1)
c22 = Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(c21)
c23 = Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(c22)
#####################################
d2 = MaxPool2D((2, 2), (2, 2))(c23)
#####################################
c31 = Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(d2)
c32 = Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(c31)
c33 = Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(c32)
#####################################
d3 = MaxPool2D((2, 2), (2, 2))(c33)
#####################################
c41 = Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(d3)
c42 = Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(c41)
c43 = Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(c42)
#####################################
d4 = MaxPool2D((2, 2), (2, 2))(c43)
#####################################
c51 = Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(d4)
c52 = Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(c51)
c53 = Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(c52)
#####################################
# Up sampling convolution followed by up-sampling
#####################################
u1 = UpSampling2D((2, 2))(c53)
#####################################
skip4 = keras.layers.Add()([c43,u1])
c61 = Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(skip4)
c62 = Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(c61)
c63 = Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(c62)
#####################################
u2 = UpSampling2D((2, 2))(c63)
#####################################
skip3 = keras.layers.Add()([c33,u2])
c71 = Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(skip3)
c72 = Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(c71 )
c73 = Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(c72)
#####################################
u3 = UpSampling2D((2, 2))(c73)
#####################################
skip2 = keras.layers.Add()([c23,u3])
c81 = Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(skip2)
c82 = Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(c81)
c83 = Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(c82)
#####################################
u4 = UpSampling2D((2, 2))(c83)
#####################################
skip1 = keras.layers.Add()([c13,u4])
c91 = Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(skip1)
c92 = Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(c91)
c93 = Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(c92)
#####################################
# Output layer
#####################################
output = (Conv2D(1, (1, 1), padding='same', activation='sigmoid'))(c93)
#####################################
model = Model(inputs=inputs, outputs=output)
#####################################
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.fit(np.array(x_train), np.array(y_train), batch_size=16, epochs=5, validation_split=0.1)
model.summary()
model.save('Nested_UNet_augmneted_data_skips_updated_kernal_sizes_updated_New_data.h5')
#####################################