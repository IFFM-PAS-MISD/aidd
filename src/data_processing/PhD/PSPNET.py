import numpy as np
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, UpSampling2D, Input, merge, Activation, BatchNormalization, \
    GlobalAveragePooling2D, Conv1D, Conv2DTranspose
from keras.models import Model
from keras.activations import relu, sigmoid, softmax
import keras
import gc
from sklearn.utils import shuffle
import tensorflow as tf
from keras import backend as K

# Hyper parameters
#####################################

lr = .0001
rho = 0.995
filters = 16
filterSize = (3, 3)
activation = relu, sigmoid
batch_size = 4
dropout = 0.2
epochs = 3
validation_split = 0.1
#####################################
# Loading the dataset
#####################################
x = np.load('E:/backup/datasets/Segmentation datasets/augemented/Augmneted_train_new_data_RMS_flat_shell_bottom.npy')
x = x / 255.
x = x.reshape(1900, 512, 512, 1)
y = np.load('E:/backup/datasets/Segmentation datasets/augemented/Augmented_target_segmentation.npy')
y = y / 255.
y = y.reshape(1900, 512, 512, 1)

#####################################
# Randomly shuffle the dataset
#####################################
x, y = shuffle(x, y)
#####################################
x_train = x[:1520]
y_train = y[:1520]


######################################

input_x = Input(shape = (512, 512, 1))
Conv  = Conv2D(filters,filterSize,padding='same',activation='relu')(input_x)
Conv  = Conv2D(2*filters,filterSize,padding='same',activation='relu')(input_x)
Conv  = Conv2D(4*filters,filterSize,padding='same',activation='relu')(input_x)

red = MaxPool2D((1,1),strides=1,padding='same')(Conv)
red = Conv2D(64,(1,1),padding='same')(red)
red = UpSampling2D((1,1), interpolation='bilinear')(red)

orange = MaxPool2D((2,2))(Conv)
orange = Conv2D(64,(1,1),padding='same')(orange)
orange = UpSampling2D((2,2),interpolation='bilinear')(orange)

blue = MaxPool2D((4,4), padding='same')(Conv)
blue = Conv2D(64,(1,1),padding='same')(blue)
blue =  UpSampling2D((4,4),interpolation='bilinear')(blue)

green = MaxPool2D((8,8), padding='same')(Conv)
green = Conv2D(64,(1,1),padding='same')(green)
green = UpSampling2D((8,8), interpolation='bilinear')(green)

Concat = keras.layers.Concatenate()([Conv,red,blue,orange,green])

output = Conv2D(32,(3,3),padding='same', activation='relu')(Concat)
output = Conv2D(64,(3,3),padding='same', activation='relu')(Concat)
output = Conv2D(1,(3,3),padding='same', activation='sigmoid')(Concat)
#output = keras.layers.Softmax()(output)

model = Model(inputs=input_x, outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=validation_split)
model.summary()

model.save('PsPnet.h5')
