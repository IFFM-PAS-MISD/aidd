import numpy as np
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, UpSampling2D, Input, merge
from keras.models import Model
import keras
import tensorflow as tf
from sklearn.utils import shuffle
import tensorflow.keras.backend as K

import gc

gc.collect()

# layers
def layer(Layer_input):
    BN = keras.layers.BatchNormalization()(Layer_input)  # Batch Normalization
    CN = keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu')(BN)  # Convolution
    out = keras.layers.Dropout(0.2)(CN)  # Dropout
    return out

# Dense Block
def dense_block(DB_input, layers):
    global Concat
    for i in range(layers):
        temp = layer(DB_input)
        Concat = keras.layers.Concatenate()([temp, DB_input])
        DB_input = temp
    out = Concat
    return out

# Transition Down (Maxpooling)
def Transition_Down(TD_input):
    BN = keras.layers.BatchNormalization()(TD_input)
    CN = keras.layers.Conv2D(filters=16, kernel_size=(1, 1), activation='relu')(BN)
    Drop = keras.layers.Dropout(.2)(CN)
    down = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(Drop)
    return down

# Transition Up (Unsampling)
def Transition_Up(TU_input):
    #Up = keras.layers.Conv2DTranspose(16, (3, 3), strides=(2, 2))(TU_input)
    Up = keras.layers.UpSampling2D(size=(2,2))(TU_input)
    return Up

# Keras Model (FCN Dense Net for semantic Segmentation
def DenseNet_Model(xtrain, ytrain, DB_Num):
    inputs = Input(shape=(512, 512, 1))
    DB1 = dense_block(inputs, DB_Num[0])
    TD1 = Transition_Down(DB1)
    DB2 = dense_block(TD1, DB_Num[1])
    TD2 = Transition_Down(DB2)
    DB3 = dense_block(TD2, DB_Num[2])
    TU1 = Transition_Up(DB3)
    DB4 = dense_block(TU1, DB_Num[3])
    TU2 = Transition_Up(DB4)
    DB5 = dense_block(TU2, DB_Num[4])
    output = keras.layers.Conv2D(1, (1, 1), padding='same', activation='sigmoid')(DB5)
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    model.fit(xtrain, ytrain, batch_size=4, epochs=5, validation_split=0.1)
    model.summary()
    return model

#####################################
# Loading the dataset
#####################################
x = np.load('E:/src/datasets/Segmentation datasets/augemented/Augmented_data_segmentation.npy')
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
DB_Number = [3, 3, 3, 3, 3]
print(len(DB_Number))
x_train = x_train#K.variable(x_train)
y_train = y_train#K.variable(y_train)
L = 10
model = DenseNet_Model(x_train, y_train, DB_Number)
model.save('FCN_DsensNets_Semantic_Segmentation.h5')
