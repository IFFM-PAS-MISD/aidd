import numpy as np
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, UpSampling2D, Input, merge
from keras.models import Model
from keras.activations import relu, sigmoid, softmax
from keras.optimizers import adam, RMSprop
from keras.losses import categorical_crossentropy, binary_crossentropy, mean_squared_error
import keras
import gc
from sklearn.utils import shuffle
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf


# Hyper parameters
#####################################
lr = .0001
rho = 0.995
filters = 32
filterSize = (3, 3)
activation = relu, sigmoid
batch_size = 2
dropout = 0.2
epochs = 5
validation_split = 0.3
#####################################
# Loading the dataset
#####################################
x = np.load('E:/backup/datasets/Segmentation datasets/augemented/Augmneted_train_new_data_RMS_flat_shell_top.npy')
x = x.reshape(1900, 512, 512, 1)
y = np.load('E:/backup/datasets/Segmentation datasets/augemented/Augmented_target_segmentation.npy')
y = y.reshape(1900, 512, 512, 1)
#####################################
# Randomly shuffle the dataset
#####################################
x, y = shuffle(x, y)
#####################################
x_train = x[:1520]
y_train = y[:1520]
######################################
# layers
def layer(Layer_input):
    BN = keras.layers.BatchNormalization()(Layer_input)  # Batch Normalization
    CN = keras.layers.Conv2D(filters=filters, kernel_size=filterSize, padding='same', activation='relu')(
        BN)  # Convolution
    out = keras.layers.Dropout(dropout)(CN)  # Dropout
    return out


# Dense Block
def dense_block(DB_input, layers):
    global Concat
    activate = Dense(1,activation='relu')(DB_input)
    for i in range(layers):
        temp = layer(activate)
        Concat = keras.layers.Concatenate(axis=-1)([temp, activate])
        activate = temp
    out = Concat
    return out


# Transition Down (Max-pooling)
def Transition_Down(TD_input):
    BN = keras.layers.BatchNormalization()(TD_input)
    active = Dense(1, activation="relu")(BN)
    CN = keras.layers.Conv2D(filters=filters, kernel_size=(1,1), padding='same')(active)
    Drop = keras.layers.Dropout(dropout)(CN)
    down = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(Drop)
    return down


# Transition Up (Unsampling)
def Transition_Up(TU_input):
    Up = keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2))(TU_input)
    # Up = keras.layers.Convolution2DTranspose(filters=16, kernel_size=(3, 3), strides=(2, 2))(TU_input)
    return Up


# Keras Model (FCN Dense Net for semantic Segmentation
def DenseNet_Model(x_train, y_train, DB_Num):
    inputs = Input(shape=(512, 512, 1))
    DB1 = dense_block(inputs, DB_Num[0])
    Concat1 = keras.layers.Concatenate(axis=-1)([inputs, DB1])
    TD1 = Transition_Down(Concat1)
    DB2 = dense_block(TD1, DB_Num[1])
    Concat2 = keras.layers.Concatenate(axis=-1)([TD1, DB2])
    TD2 = Transition_Down(DB2)
    DB3 = dense_block(TD2, DB_Num[2])
    TU1 = Transition_Up(DB3)
    Concat3 = keras.layers.Concatenate(axis=-1)([TU1, Concat2])
    DB4 = dense_block(Concat3, DB_Number[3])
    TU2 = Transition_Up(DB4)
    Concat4 = keras.layers.Concatenate(axis=-1)([TU2, Concat1])
    DB5 = dense_block(Concat4, DB_Number[4])
    output = keras.layers.Conv2D(1, (1, 1), padding='same', activation='sigmoid')(DB5)

    segment_model = Model(inputs=inputs, outputs=output)
    segment_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    segment_model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=validation_split)
    segment_model.summary()
    return segment_model
DB_Number = [3, 3, 4, 3, 3]
print(len(DB_Number))
model = DenseNet_Model(x_train, y_train, DB_Number)
model.save('FCN_DsensNets_Semantic_Segmentation' + '_filter' + str(filters) + '_epoch' + str(epochs) + '_kernal' + str(
    filterSize) + '_drpout' + str(dropout) +'batch_size'+str(batch_size) + '.h5')

gc.collect()
