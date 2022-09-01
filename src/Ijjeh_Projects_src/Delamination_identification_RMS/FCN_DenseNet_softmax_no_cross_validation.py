import gc
import keras
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.activations import relu, sigmoid
from keras.layers import Conv2D, Input, Activation
from keras.models import Model
from keras.utils import to_categorical


###########################################   memory growing ###########################################################
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
########################################################################################################################

########################################################################################################################
################################################## Hyper parameters ####################################################
########################################################################################################################
lr = .0001
rho = 0.995
filters = 16
filterSize = (3, 3)
activation = relu, sigmoid
batch_size = 4
dropout = 0.2
epochs = 100
validation_split = 0.2
Convfilter = 16

########################################################################################################################
############################## Loading dataset into x, y arrays and reshape them #######################################
########################################################################################################################
x = np.load('E:/backup/datasets/Segmentation datasets/augemented/June_training_dataset.npy')
x = x.reshape(1900, 512, 512, 1)
x = x / 255.0  # normalizing x,y to (0-1) range
y = np.load('E:/backup/datasets/Segmentation datasets/augemented/June_labels.npy')
y = y.reshape(1900, 512, 512, 1)
y = y / 255.0

########################################################################################################################
###################################### Shuffle the data set at random ##################################################
########################################################################################################################
# x, y = shuffle(x, y)
########################################################################################################################
################# Split dataset into training and testing sets and again re-shuffle them ###############################
########################################################################################################################
x_train = x[:1520]
y_train = y[:1520]

test_x_samples = x[1520:1900]
test_y_samples = y[1520:1900]

y_train = to_categorical(y_train)
test_y_samples = to_categorical(test_y_samples)


########################################################################################################################
# Custom loss functions
def custom_loss(y_true, y_pred, smooth=0):  # Dice score function
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(K.abs(y_true_f * y_pred_f))
    return -(2. * intersection + smooth) / (K.sum(K.abs(y_true_f)) + K.sum(K.abs(y_pred_f)) - intersection + smooth)


########################################################################################################################
# Dice loss / F1
def dice_loss(y_true, y_pred):
    numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=-1)
    denominator = tf.reduce_sum(y_true + y_pred, axis=-1)
    return 1 - 2 * (numerator + 1) / (denominator + 1)


########################################################################################################################
# Custom metric
def iou_metric_abs(y_true, y_pred, smooth=0):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(K.abs(y_true_f * y_pred_f))
    union = K.sum(K.abs(y_true_f)) + K.sum(K.abs(y_pred_f)) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou


########################################################################################################################
def recall(y_true, y_pred):
    y_true = K.ones_like(y_true)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    all_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall_acc = true_positives / (all_positives + K.epsilon())
    return recall_acc


########################################################################################################################
def precision(y_true, y_pred):
    y_true = K.ones_like(y_true)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision_acc = true_positives / (predicted_positives + K.epsilon())
    return precision_acc


########################################################################################################################
def f1_score(y_true, y_pred):
    precision_m = precision(y_true, y_pred)
    recall_m = recall(y_true, y_pred)
    return 2 * ((precision_m * recall_m) / (precision_m + recall_m + K.epsilon()))


########################################################################################################################
def custom_acc(y_true, y_pred):
    return 1 - dice_loss(y_true, y_pred)


########################################################################################################################
def iou_metric(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(K.abs(y_true_f * y_pred_f))
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou


########################################################################################################################
####################################################### layers #########################################################
########################################################################################################################
def layer(Layer_input, downfilter, i):
    BN = keras.layers.BatchNormalization()(Layer_input)  # Batch Normalization
    activation_func = Activation('relu')(BN)  # adding activation layer Relu then directs it to the Conv2D
    CN = keras.layers.Conv2D(filters=downfilter * (i + 1), kernel_size=filterSize, padding='same')(activation_func)
    out = keras.layers.Dropout(dropout)(CN)  # Dropout
    return out


########################################################################################################################
################################################## Dense Block #########################################################
########################################################################################################################
def dense_block(DB_input, layers):
    global Concat
    for i in range(layers):
        temp = layer(DB_input, Convfilter, i)
        Concat = keras.layers.Concatenate(axis=-1)([temp, DB_input])
        DB_input = temp
    out = Concat
    return out


########################################################################################################################
######################################### Transition Down (Max-pooling) ################################################
########################################################################################################################
def Transition_Down(TD_input, downfilter):
    BN = keras.layers.BatchNormalization()(TD_input)
    active = Activation('relu')(BN)
    CN = keras.layers.Conv2D(filters=downfilter, kernel_size=(1, 1), padding='same')(active)  # , activation='relu'
    Drop = keras.layers.Dropout(dropout)(CN)
    down = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(Drop)
    return down


########################################################################################################################
########################################### Transition Up (Up sampling) ################################################
########################################################################################################################
def Transition_Up(TU_input, filters_tu):
    Up = keras.layers.Conv2DTranspose(filters=filters_tu, kernel_size=(3, 3), strides=(2, 2), padding='same')(
        TU_input)
    return Up


########################################################################################################################
############################# Keras Model (FCN Dense Net for semantic Segmentation #####################################
########################################################################################################################
def DenseNet_Model(x_train_set, y_train_set, DB_Num):
    inputs = Input(shape=(512, 512, 1))
    ####################################################################################################################
    Conv = Conv2D(filters, filterSize, padding='same')(inputs)
    # Conv = keras.layers.Concatenate()([Conv, inputs])
    ####################################################################################################################
    DB1 = dense_block(inputs, DB_Num[0])
    Concat1 = keras.layers.Concatenate(axis=-1)([Conv, DB1])
    ####################################################################################################################
    TD1 = Transition_Down(Concat1, Convfilter)
    ####################################################################################################################
    DB2 = dense_block(TD1, DB_Num[1])
    Concat2 = keras.layers.Concatenate(axis=-1)([TD1, DB2])
    ####################################################################################################################
    TD2 = Transition_Down(Concat2, Convfilter)
    ####################################################################################################################
    DB3 = dense_block(TD2, DB_Num[2])
    Concat3 = keras.layers.Concatenate(axis=-1)([TD2, DB3])
    ####################################################################################################################
    TD3 = Transition_Down(Concat3, Convfilter)
    ####################################################################################################################
    DB4 = dense_block(TD3, DB_Num[3])
    ####################################################################################################################
    TU1 = Transition_Up(DB4, Convfilter)
    ####################################################################################################################
    Concat4 = keras.layers.Concatenate(axis=-1)([TU1, Concat3])
    DB5 = dense_block(Concat4, DB_Number[4])
    ####################################################################################################################
    TU2 = Transition_Up(DB5, Convfilter)
    ####################################################################################################################
    Concat5 = keras.layers.Concatenate(axis=-1)([TU2, Concat2])
    DB6 = dense_block(Concat5, DB_Number[5])
    ####################################################################################################################
    TU3 = Transition_Up(DB6, Convfilter)
    ####################################################################################################################
    Concat6 = keras.layers.Concatenate(axis=-1)([TU3, Concat1])
    DB7 = dense_block(Concat6, DB_Number[6])
    ####################################################################################################################
    output = keras.layers.Conv2D(2, (1, 1), activation='softmax')(DB7)
    ####################################################################################################################
    segment_model = Model(inputs=inputs, outputs=output)
    ####################################################################################################################
    segment_model.compile(optimizer='adam',
                          loss=keras.losses.categorical_crossentropy,
                          metrics=[iou_metric])
    ####################################################################################################################
    segment_model.fit(x_train_set, y_train_set,
                      shuffle=True,
                      batch_size=batch_size,
                      epochs=epochs,
                      validation_split=validation_split)
    ####################################################################################################################
    score = segment_model.evaluate(test_x_samples,
                                   test_y_samples,
                                   batch_size=2,
                                   verbose=1)
    ####################################################################################################################
    print(score[0], score[1])
    segment_model.summary()
    return segment_model


########################################################################################################################
DB_Number = [2, 2, 2, 4, 2, 2, 2]  # adding extra two DBs [0,1,2,3,4,5,6]
print(len(DB_Number))
model = DenseNet_Model(x_train, y_train, DB_Number)
model.save('E:/backup/models/FCN_DenseNet_models/Softmax/FCN_softmax_100_epoches.h5')

gc.collect()
