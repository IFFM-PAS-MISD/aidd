import gc

import keras
import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.layers import Conv2D, MaxPool2D, Input, Conv2DTranspose
from keras.losses import binary_crossentropy
from keras.models import Model
from keras.utils import to_categorical
from sklearn.utils import shuffle

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 64} )
# sess = tf.Session(config=config)
# keras.backend.set_session(sess)

# Force the Garbage Collector to release unreferenced memory
gc.collect()

# Hyperparameter
filters = 32
epsilon = 0.1
dropout_rate = 0.2
epochs = 10

x = np.load('E:/backup/datasets/Segmentation datasets/augemented/June_training_dataset.npy')
x = x.reshape(1900, 512, 512, 1)

y = np.load('E:/backup/datasets/Segmentation datasets/augemented/June_labels.npy')
y = y.reshape(1900, 512, 512, 1)

# Randomly shuffle the dataset

x = x / 255.0  # normalizing x,y to (0-1)
y = y / 255.0
x, y = shuffle(x, y)

x_train = x[:1520]
y_train = y[:1520]

test_x_samples = x[1520:]
tests_y_samples = y[1520:]

y_train = to_categorical(y_train)
tests_y_samples = to_categorical(tests_y_samples)

####################
def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true), -1) + K.sum(K.square(y_pred), -1) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


# Dice loss / F1
def dice_loss(y_true, y_pred):
    numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=-1)
    denominator = tf.reduce_sum(y_true + y_pred, axis=-1)

    return 1 - (numerator + 1) / (denominator + 1)


def dice_loss_1(y_true, y_pred):  # did not give good results
    numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3))
    denominator = tf.reduce_sum(y_true + y_pred, axis=(1, 2, 3))
    return 1 - numerator / denominator


# comined loss
def loss(y_true, y_pred):
    def dice_loss(y_true, y_pred):
        numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3))
        denominator = tf.reduce_sum(y_true + y_pred, axis=(1, 2, 3))
        return tf.reshape(1 - numerator / denominator, (-1, 1, 1))

    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)


#######
# Custom metric IoU
def iou_loss_core(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(K.abs(y_true_f * y_pred_f))  # , axis=-1
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection / 2
    iou = (intersection + smooth) / (union + smooth)
    return iou


#######
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


def iou_metric(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(K.abs(y_true_f * y_pred_f))
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou


####################
def batch_normalizer(inputs):
    return keras.layers.BatchNormalization()(inputs)


# The Model
inputs = Input(shape=(512, 512, 1))
# Backbone Down sampling convolution followed by max-pooling

c11 = Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(inputs)
BN = batch_normalizer(c11)
c12 = Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(BN)
BN1 = batch_normalizer(c12)
d1 = MaxPool2D((2, 2), (2, 2))(BN1)
d1 = keras.layers.Dropout(dropout_rate)(d1)

c21 = Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(d1)
BN = batch_normalizer(c21)
c22 = Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(BN)
BN2 = batch_normalizer(c22)
d2 = MaxPool2D((2, 2), (2, 2))(BN2)
d2 = keras.layers.Dropout(dropout_rate)(d2)

c31 = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(d2)
BN = batch_normalizer(c31)
c32 = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(BN)
BN3 = batch_normalizer(c32)
d3 = MaxPool2D((2, 2), (2, 2))(BN3)
d3 = keras.layers.Dropout(dropout_rate)(d3)

c41 = Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(d3)
BN = batch_normalizer(c41)
c42 = Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(BN)
BN4 = batch_normalizer(c42)
d4 = MaxPool2D((2, 2), (2, 2))(BN4)
d4 = keras.layers.Dropout(dropout_rate)(d4)

# bottleneck layer
c51 = Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(d4)
BN = batch_normalizer(c51)
c52 = Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(BN)
BN = batch_normalizer(c52)

# Up sampling convolution followed by ConvTranspose
u1 = Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding='same')(BN)
skip4 = keras.layers.Concatenate()([BN4, u1])
skip4 = keras.layers.Dropout(dropout_rate)(skip4)  # adding dropout layer
c61 = Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(skip4)
BN = batch_normalizer(c61)
c62 = Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(BN)
BN = batch_normalizer(c62)

u2 = Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding='same', activation='relu')(BN)
skip3 = keras.layers.Concatenate()([BN3, u2])
skip3 = keras.layers.Dropout(dropout_rate)(skip3)
c71 = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(skip3)
BN = batch_normalizer(c71)
c72 = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(BN)
BN = batch_normalizer(c72)

u3 = Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding='same')(BN)
skip2 = keras.layers.Concatenate()([BN2, u3])
skip2 = keras.layers.Dropout(dropout_rate)(skip2)
c81 = Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(skip2)
BN = batch_normalizer(c81)
c82 = Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(BN)
BN = batch_normalizer(c82)

u4 = Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding='same', activation='relu')(BN)
skip1 = keras.layers.Concatenate()([BN1, u4])
skip1 = keras.layers.Dropout(dropout_rate)(skip1)
c91 = Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(skip1)
BN = batch_normalizer(c91)
c92 = Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(BN)
BN = batch_normalizer(c92)

# Output layer
output = Conv2D(2, (1, 1), activation='softmax')(BN)
model = Model(inputs=inputs, outputs=output)
model.compile(optimizer="adam", loss=keras.losses.categorical_crossentropy, metrics=[iou_metric])
model.fit(np.array(x_train), np.array(y_train), shuffle=True, batch_size=4, epochs=epochs, validation_split=0.2)
model.save('Unet_softmax.h5')
print("model saved")
score = model.evaluate(test_x_samples, tests_y_samples, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
model.summary()

