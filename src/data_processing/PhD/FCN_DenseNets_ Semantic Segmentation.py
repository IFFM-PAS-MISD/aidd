import numpy as np
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, UpSampling2D, Input, merge, Activation, BatchNormalization
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
epochs = 10
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
# layers
def layer(Layer_input):
    BN = keras.layers.BatchNormalization()(Layer_input)  # Batch Normalization
    x = Activation('relu')(BN)  # adding activation layer Relu then directs it to the Conv2D
    CN = keras.layers.Conv2D(filters=filters, kernel_size=filterSize, kernel_initializer='he_normal', padding='same')(
        x)  # adding kernal_intializer ,activation='relu'
    out = keras.layers.Dropout(dropout)(CN)  # Dropout
    return out


# Dense Block
def dense_block(DB_input, layers):
    global Concat
    #x = BatchNormalization()(DB_input)
    #x = Activation('relu')(x)
    # activate = Dense(1,activation='relu')(DB_input)
    for i in range(layers):
        temp = layer(DB_input)
        Concat = keras.layers.Concatenate(axis=-1)([temp, DB_input])
        DB_input = temp
    out = Concat
    return out


# Transition Down (Max-pooling)
def Transition_Down(TD_input):
    BN = keras.layers.BatchNormalization()(TD_input)
    active = Activation('relu')(BN)
    CN = keras.layers.Conv2D(filters=filters, kernel_size=(1, 1), padding='same')(active)  # , activation='relu'
    Drop = keras.layers.Dropout(dropout)(CN)
    down = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(Drop)
    return down


# Transition Up (Up sampling)
def Transition_Up(TU_input):
    Up = keras.layers.Convolution2DTranspose(filters=filters, kernel_size=(3, 3), strides=(2, 2), padding='same')( ### maybe we need to make it vaild
        TU_input)
    return Up


###################
# Custom loss function
def custom_loss(y_true, y_pred, smooth=1):  # Dice score function
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return -(2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


###################
# Custom metric
def iou_metric(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(K.abs(y_true_f * y_pred_f))
    union = K.sum(K.abs(y_true_f)) + K.sum(K.abs(y_pred_f)) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou


# Keras Model (FCN Dense Net for semantic Segmentation
def DenseNet_Model(x_train, y_train, DB_Num):
    inputs = Input(shape=(512, 512, 1))

    Conv = Conv2D(filters, filterSize, padding='same', activation='relu')(inputs)
    Conv = Conv2D(filters, filterSize, padding='same', activation='relu')(inputs)
    Conv = Conv2D(filters, filterSize, padding='same', activation='relu')(inputs)
    Conv = keras.layers.Concatenate()([Conv,inputs])

    DB1 = dense_block(Conv, DB_Num[0])
    Concat1 = keras.layers.Concatenate(axis=-1)([Conv, DB1])
    TD1 = Transition_Down(Concat1)
    DB2 = dense_block(TD1, DB_Num[1])
    Concat2 = keras.layers.Concatenate(axis=-1)([TD1, DB2])
    TD2 = Transition_Down(Concat2)  # here was DB2
    DB3 = dense_block(TD2, DB_Num[2])
    ############## new addition
    Concat3  = keras.layers.Concatenate(axis=-1)([TD2, DB3])
    TD3 = Transition_Down(Concat3)
    DB4 = dense_block(TD3, DB_Num[3])

    ##############
    TU1 = Transition_Up(DB4)
    Concat4 = keras.layers.Concatenate(axis=-1)([TU1, Concat3])
    DB5 = dense_block(Concat4, DB_Number[4])
    TU2 = Transition_Up(DB5)
    Concat5 = keras.layers.Concatenate(axis=-1)([TU2, Concat2])
    DB6 = dense_block(Concat5, DB_Number[5])
    TU3 = Transition_Up(DB6)
    Concat6 = keras.layers.Concatenate(axis=-1)([TU3, Concat1])
    DB7 = dense_block(Concat6,DB_Number[6])

    output = keras.layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same')(DB7)  #activation='sigmoid',

    segment_model = Model(inputs=inputs, outputs=output)
    segment_model.compile(optimizer='adam', loss=custom_loss, metrics=[iou_metric])
    segment_model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=validation_split)
    segment_model.summary()
    return segment_model


DB_Number = [4, 4, 4, 4, 4, 4, 4]  # adding extra two DBs [0,1,2,3,4,5,6]
print(len(DB_Number))
model = DenseNet_Model(x_train, y_train, DB_Number)
model.save(
    'E:/backup/models/FCN_DenseNet_models/FCN_DsensNets_Semantic_Segmentation_filter_Using_Conv2DTranspose' + str(
        filters) + '_epoch_' + str(epochs) + '_kernal_' + str(
        filterSize) + '_drpout_' + str(dropout) + '_batch_size_' + str(batch_size) + '_loss_updated_changed_DB _layer.h5')

gc.collect()
