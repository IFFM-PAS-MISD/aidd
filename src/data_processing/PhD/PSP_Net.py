import numpy as np
from keras.layers import Dense, Conv2D, AveragePooling2D, Flatten, UpSampling2D, Input, merge, Activation, BatchNormalization, \
    GlobalAveragePooling2D, Conv1D, Conv2DTranspose
from keras.models import Model
from keras.activations import relu, sigmoid, softmax
import keras
import gc
from sklearn.utils import shuffle
import tensorflow as tf
from keras import backend as K
from keras.layers import Reshape
# Hyper parameters
#####################################

lr = .0001
rho = 0.995
filters = 16
filterSize = (3, 3)
activation = relu, sigmoid
batch_size = 3
dropout = 0.1
epochs = 8
validation_split = 0.2
#####################################
# Loading the dataset
#####################################
x = np.load('June_training_dataset.npy')
x = x / 255.
x = x.reshape(1900, 512, 512, 1)
y = np.load('June_labels.npy')
y = y / 255.
y = y.reshape(1900, 512, 512, 1)

#####################################
# Randomly shuffle the dataset
#####################################
#####################################
x_train = x[:1520]
y_train = y[:1520]
x_test = x[1520:]
y_test = y[1520:]

######################################

input_x = Input(shape = (512, 512, 1))

Conv  = Conv2D(filters,filterSize, dilation_rate=4,padding='same',activation='relu')(input_x)
Conv  = Conv2D(filters,filterSize, dilation_rate=4,padding='same',activation='relu')(Conv)
Conv  = Conv2D(filters,filterSize, dilation_rate=4,padding='same',activation='relu')(Conv)

#print(Conv.shape)

red = GlobalAveragePooling2D()(Conv) # Use GlobalAveragePooling2D here
print(red.shape)
red = Reshape((1,1,16))(red)
print(red.shape)
red = Conv2D(64,(1,1), dilation_rate=4, padding='same')(red)
red = UpSampling2D((512,512), interpolation='bilinear')(red)

orange = AveragePooling2D((2,2))(Conv) #Use AveragePooling2D here, with pool size of 2,2
orange = Conv2D(32,(1,1), dilation_rate=4, padding='same')(orange)
orange = UpSampling2D((2,2),interpolation='bilinear')(orange)

blue = AveragePooling2D((4,4), padding='same')(Conv) #Use AveragePooling2D, with pool size of 4,4
blue = Conv2D(32,(1,1), dilation_rate=4, padding='same')(blue)
blue = UpSampling2D((4,4),interpolation='bilinear')(blue)

green = AveragePooling2D((8,8), padding='same')(Conv) #Use AveragePooling2D, with pool size of 8,8
green = Conv2D(32,(1,1), dilation_rate=4, padding='same')(green)
green = UpSampling2D((8,8), interpolation='bilinear')(green)

Concat = keras.layers.Concatenate()([Conv,red,blue,orange,green])

output = Conv2D(16,(3,3), padding='same', activation='relu')(Concat)
output = Conv2D(16,(3,3), padding='same', activation='relu')(output)
output = Conv2D(1,(3,3),  padding='same', activation='sigmoid')(output)
#output = keras.layers.Softmax()(output)

model = Model(inputs=input_x, outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=validation_split, shuffle=True)
model.summary()

model.save('PsPnet.h5')