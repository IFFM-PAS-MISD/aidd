import gc

import keras
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Conv2D, MaxPool2D, Input, Dropout, \
    Activation
from keras.models import Model
from sklearn.utils import shuffle

# Force the Garbage Collector to release unreferenced memory
gc.collect()
# Hyper parameters
#####################################
lr = .0001  # RMSprop, adam
rho = 0.995  # RMSprop
filters = 32
filter_size = (3, 3)
batch_size = 4
dropout = 0.2
epochs = 10
validation_split = 0.2
#####################################
# Loading the dataset
#####################################
x_train = np.load('E:/backup/datasets/Segmentation datasets/augemented/June_training_dataset.npy')
x_train = x_train / 255.
x_train = x_train.reshape(1900, 512, 512, 1)
y_train = np.load('E:/backup/datasets/Segmentation datasets/augemented/June_labels.npy')
y_train = y_train / 255.
y_train = y_train.reshape(1900, 512, 512, 1)
#####################################
# Randomly shuffle the dataset
#####################################
x_train, y_train = shuffle(x_train, y_train)
#####################################
x_sample = x_train[:1520]
y_lable = y_train[:1520]
# x_sample, y_lable = shuffle(x_sample, y_lable)
test_x_samples = x_train[1520:1900]
tests_y_samples = y_train[1520:1900]
test_x_samples, tests_y_samples = shuffle(test_x_samples, tests_y_samples)


def layer(input, growth):
    l = Conv2D(filters=filters * growth, kernel_size=filter_size, padding='same')(input)
    b = keras.layers.BatchNormalization()(l)
    x = keras.layers.ReLU()(b)
    return x


# Segnet Model
#####################################
inputs = Input(shape=(512, 512, 1))
#####################################
# Backbone Down sampling convolution followed by max-pooling
#####################################
c11 = layer(inputs, 1)
c12 = layer(c11, 1)

#####################################
d1 = MaxPool2D((2, 2), (2, 2))(c12)
d1 = Dropout(dropout)(d1)
#####################################
c21 = layer(d1, 1)
c22 = layer(c21, 1)

#####################################
d2 = MaxPool2D((2, 2), (2, 2))(c22)
d2 = Dropout(dropout)(d2)
#####################################
c31 = layer(d2, 1)
c32 = layer(c31, 1)
c33 = layer(c32, 1)

#####################################
d3 = MaxPool2D((2, 2), (2, 2))(c33)
d3 = Dropout(dropout)(d3)
#####################################
c41 = layer(d3, 2)
c42 = layer(c41, 2)
c43 = layer(c42, 2)

#####################################
d4 = MaxPool2D((2, 2), (2, 2))(c43)
d4 = Dropout(dropout)(d4)
#####################################
c51 = layer(d4, 4)
c52 = layer(c51, 4)
c53 = layer(c52, 4)

#####################################
d5 = MaxPool2D((2, 2), (2, 2))(c53)
d5 = Dropout(dropout)(d5)
#####################################

# Up sampling convolution followed by up-sampling

#####################################

u1 = keras.layers.UpSampling2D((2, 2))(d5)  # ,interpolation='bilinear'
#####################################

skip5 = keras.layers.Concatenate()([c53, u1])
c71 = layer(skip5, 4)
c72 = layer(c71, 4)
c73 = layer(c72, 4)

#####################################
# u2 = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', activation='relu')(BN7)  # UpSampling2D((2, 2))(BN7)
u2 = keras.layers.UpSampling2D((2, 2))(c73)
#####################################

skip4 = keras.layers.Concatenate()([c43, u2])
c81 = layer(skip4, 2)
c82 = layer(c81, 2)
c83 = layer(c82, 2)

#####################################
# u3 = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', activation='relu')(BN8)  # UpSampling2D((2, 2))(BN8)
u3 = keras.layers.UpSampling2D((2, 2))(c83)
#####################################

skip3 = keras.layers.Concatenate()([c33, u3])
c91 = layer(skip3, 1)
c92 = layer(c91, 1)
c93 = layer(c92, 1)

#####################################
u4 = keras.layers.UpSampling2D((2, 2))(c93)
#####################################

skip2 = keras.layers.Concatenate()([c22, u4])

c101 = layer(skip2, 1)
c102 = layer(c101, 1)
c103 = layer(c102, 1)

#####################################
u5 = keras.layers.UpSampling2D((2, 2))(c103)
#####################################

skip1 = keras.layers.Concatenate()([c12, u5])
c111 = layer(skip1, 1)
c112 = layer(c111, 1)
c113 = layer(c112, 1)

#####################################
# Output layer
#####################################
output = (Conv2D(1, (1, 1), padding='same', activation='sigmoid'))(c113)
#####################################
model = Model(inputs=inputs, outputs=output)


#####################################


# Custom loss function
def custom_loss(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(K.abs(y_true_f * y_pred_f))
    return - ((intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth))


# Custom metric
def iou_metric(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(K.abs(y_true_f * y_pred_f))
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou


# Dice loss / F1
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


########################
def recall(y_true, y_pred):
    y_true = K.ones_like(y_true)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    all_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (all_positives + K.epsilon())
    return recall


def precision(y_true, y_pred):
    y_true = K.ones_like(y_true)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_score(y_true, y_pred):
    precision_m = precision(y_true, y_pred)
    recall_m = recall(y_true, y_pred)
    return 2 * ((precision_m * recall_m) / (precision_m + recall_m + K.epsilon()))


########################
def custom_acc(y_true, y_pred):
    return 1 - dice_loss(y_true, y_pred)


###################
model.compile(optimizer='adam', loss=keras.losses.binary_crossentropy, metrics=[keras.metrics.binary_accuracy])
model.fit(x_sample, y_lable, batch_size=batch_size, epochs=epochs, validation_split=validation_split)
score = model.evaluate(test_x_samples, tests_y_samples, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1] * 100)
model.summary()
############################################
# Evaluating the model using test set
#####################################
score = model.evaluate(test_x_samples, tests_y_samples, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
######################################
model.save('E:/backup/models/SegNet_models/SegNet_Upsampling_added_skip_layer_function.h5')
gc.collect()
