import numpy as np
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, UpSampling2D, Input, merge, Activation, BatchNormalization
from keras.models import Model
from keras.activations import relu, sigmoid, softmax
from keras.optimizers import adam,sgd,RMSprop
import keras
import gc
from sklearn.utils import shuffle
import tensorflow as tf
from keras import backend as K

# Hyper parameters
#####################################

lr = .0001
rho = 0.995
filters = 8
filterSize = (3, 3)
activation = relu, sigmoid
batch_size = 8
dropout = 0.2
epochs = 17
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
test_x_samples = x[1520:1900]
tests_y_samples = y[1520:1900]
test_x_samples, tests_y_samples = shuffle(test_x_samples, tests_y_samples)


######################################
# layers
def layer(Layer_input):
    BN = keras.layers.BatchNormalization()(Layer_input)  # Batch Normalization
    x = Activation('relu')(BN)  # adding activation layer Relu then directs it to the Conv2D
    CN = keras.layers.Conv2D(filters=filters, kernel_size=filterSize, padding='same')(x)
    # kernel_initializer='he_normal', adding kernal_intializer ,activation='relu'
    out = keras.layers.Dropout(dropout)(CN)  # Dropout
    return out


# Dense Block
def dense_block(DB_input, layers):
    global Concat
    # x = BatchNormalization()(DB_input)
    # x = Activation('relu')(x)
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
    Up = keras.layers.Convolution2DTranspose(filters=filters, kernel_size=(3, 3), padding='same', strides=(2, 2))(
        TU_input)  ###    maybe we need to make it vaild
    return Up


###################
# Custom loss functions
def custom_loss(y_true, y_pred, smooth=0):  # Dice score function
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(K.abs(y_true_f * y_pred_f))
    return -(2. * intersection + smooth) / (K.sum(K.abs(y_true_f)) + K.sum(K.abs(y_pred_f)) - intersection + smooth)

#Dice loss / F1
def dice_loss(y_true, y_pred):
  numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=-1)
  denominator = tf.reduce_sum(y_true + y_pred, axis=-1)
  return 1 - (numerator + 1) / (denominator + 1)
###################
# Custom metric
def iou_metric(y_true, y_pred, smooth=0):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(K.abs(y_true_f * y_pred_f))
    union = K.sum(K.abs(y_true_f)) + K.sum(K.abs(y_pred_f)) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou


# Keras Model (FCN Dense Net for semantic Segmentation
def DenseNet_Model(x_train, y_train, DB_Num):
    inputs = Input(shape=(512, 512, 1))

    Conv = Conv2D(filters, filterSize, padding='same')(inputs)
    # Conv = keras.layers.Concatenate()([Conv, inputs])
    DB1 = dense_block(inputs, DB_Num[0])
    Concat1 = keras.layers.Concatenate(axis=-1)([Conv, DB1])
    TD1 = Transition_Down(Concat1)
    DB2 = dense_block(TD1, DB_Num[1])
    Concat2 = keras.layers.Concatenate(axis=-1)([TD1, DB2])
    TD2 = Transition_Down(Concat2)  # here was DB2
    DB3 = dense_block(TD2, DB_Num[2])
    ############## new addition
    Concat3 = keras.layers.Concatenate(axis=-1)([TD2, DB3])
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
    DB7 = dense_block(Concat6, DB_Number[6])

    output = keras.layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same')(DB7)  # activation='sigmoid',

    segment_model = Model(inputs=inputs, outputs=output)
    segment_model.compile(optimizer=adam(lr=lr), loss=dice_loss, metrics=[iou_metric])
    segment_model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=validation_split)
    score = segment_model.evaluate(test_x_samples, tests_y_samples, batch_size=8, verbose=1)
    print(score[0], score[1])
    segment_model.summary()
    return segment_model


DB_Number = [4, 4, 4, 4, 4, 4, 4]  # adding extra two DBs [0,1,2,3,4,5,6]
print(len(DB_Number))
model = DenseNet_Model(x_train, y_train, DB_Number)
model.save(
    'E:/backup/models/FCN_DenseNet_models/FCN_DsensNets_Semantic_Segmentation_filter_Using_Conv2DTranspose' + str(
        filters) + '_epoch_' + str(epochs) + '_kernal_' + str(
        filterSize) + '_drpout_' + str(dropout) + '_batch_size_' + str(
        batch_size) + '_loss_updated_changed_DB _layer_Iou_and_loss_changed.h5')

gc.collect()
