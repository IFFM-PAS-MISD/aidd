import numpy as np
import cv2
from glob import glob
import gc
import keras
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, UpSampling2D, Input, merge, Conv2DTranspose
from keras.models import Model
from keras.optimizers import adam, rmsprop, sgd
from keras import backend as K
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

# Force the Garbage Collector to release unreferenced memory

gc.collect()
# Hyper parameters
#####################################
lr = .0001  # RMSprop, adam
rho = 0.995  # RMSprop
filters = 16
filter_size = (3, 3)
batch_size = 16
dropout = 0.5
epochs = 3
validation_split = 0.1
#####################################
x = np.load('E:/backup/datasets/Segmentation datasets/augemented/Augmneted_train_new_data_RMS_flat_shell_top.npy')
x = x.reshape(1900, 512, 512, 1)
y = np.load('E:/backup/datasets/Segmentation datasets/augemented/Augmented_target_segmentation.npy')
y = y.reshape(1900, 512, 512, 1)

experimental = np.load('Experimental_test_images.npy')
print(experimental.shape)
experimental = experimental.reshape(276, 512, 512, 1)

#####################################
# Randomly shuffle the dataset
#####################################
x, y = shuffle(x, y)
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
BatchNorm = keras.layers.BatchNormalization()(inputs)
c0 = Dense(1, activation='relu')(BatchNorm)
c11 = Conv2D(filters=filters, kernel_size=filter_size, strides=1, padding='same', activation='relu')(c0)
c13 = Conv2D(filters=filters, kernel_size=filter_size, strides=1, padding='same', activation='relu')(c11)
BN1 = keras.layers.BatchNormalization()(c13)

#####################################
d1 = MaxPool2D((2, 2), (2, 2))(BN1)
#####################################
c21 = Conv2D(filters=filters, kernel_size=filter_size, strides=1, padding='same', activation='relu')(d1)
c23 = Conv2D(filters=filters, kernel_size=filter_size, strides=1, padding='same', activation='relu')(c21)
BN2 = keras.layers.BatchNormalization()(c23)
# output2= keras.layers.concatenate([c21,c23], axis=1)
#####################################
d2 = MaxPool2D((2, 2), (2, 2))(BN2)
#####################################
c31 = Conv2D(filters=filters, kernel_size=filter_size, strides=1, padding='same', activation='relu')(d2)
c32 = Conv2D(filters=filters, kernel_size=filter_size, strides=1, padding='same', activation='relu')(c31)
c33 = Conv2D(filters=filters, kernel_size=filter_size, strides=1, padding='same', activation='relu')(c32)
BN3 = keras.layers.BatchNormalization()(c33)
#####################################
d3 = MaxPool2D((2, 2), (2, 2))(BN3)
#####################################
c41 = Conv2D(filters=filters, kernel_size=filter_size, strides=1, padding='same', activation='relu')(d3)
c42 = Conv2D(filters=filters, kernel_size=filter_size, strides=1, padding='same', activation='relu')(c41)
c43 = Conv2D(filters=filters, kernel_size=filter_size, strides=1, padding='same', activation='relu')(c42)
BN4 = keras.layers.BatchNormalization()(c43)

#####################################
d4 = MaxPool2D((2, 2), (2, 2))(BN4)
#####################################
c51 = Conv2D(filters=filters, kernel_size=filter_size, strides=1, padding='same', activation='relu')(d4)
c52 = Conv2D(filters=filters, kernel_size=filter_size, strides=1, padding='same', activation='relu')(c51)
c53 = Conv2D(filters=filters, kernel_size=filter_size, strides=1, padding='same', activation='relu')(c52)
BN5 = keras.layers.BatchNormalization()(c53)

#####################################
d5 = MaxPool2D((2, 2), (2, 2))(BN5)
#####################################
# Boltneck1 = Conv2D(filters=filters, kernel_size=filter_size, strides=1, padding='same', activation='relu')(d5)
# Boltneck2 = Conv2D(filters=filters, kernel_size=filter_size, strides=1, padding='same', activation='relu')(Boltneck1)
# Boltneck3 = Conv2D(filters=filters, kernel_size=filter_size, strides=1, padding='same', activation='relu')(Boltneck2)
# BN6 = keras.layers.BatchNormalization()(Boltneck3)

#####################################
# Up sampling convolution followed by up-sampling
#####################################
u1 = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', activation='relu')(d5)  # UpSampling2D((2, 2))(d5)
#####################################
skip5 = keras.layers.Concatenate()([c53, u1])
c71 = Conv2D(filters=filters, kernel_size=filter_size, strides=1, padding='same', activation='relu')(skip5)
c72 = Conv2D(filters=filters, kernel_size=filter_size, strides=1, padding='same', activation='relu')(c71)
c73 = Conv2D(filters=filters, kernel_size=filter_size, strides=1, padding='same', activation='relu')(c72)
BN7 = keras.layers.BatchNormalization()(c73)

#####################################
u2 = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', activation='relu')(BN7)  # UpSampling2D((2, 2))(BN7)
#####################################
skip4 = keras.layers.Concatenate()([c43, u2])
c81 = Conv2D(filters=filters, kernel_size=filter_size, strides=1, padding='same', activation='relu')(skip4)
c82 = Conv2D(filters=filters, kernel_size=filter_size, strides=1, padding='same', activation='relu')(c81)
c83 = Conv2D(filters=filters, kernel_size=filter_size, strides=1, padding='same', activation='relu')(c82)
BN8 = keras.layers.BatchNormalization()(c83)

#####################################
u3 = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', activation='relu')(BN8)  # UpSampling2D((2, 2))(BN8)
#####################################
skip3 = keras.layers.Concatenate()([c33, u3])
c91 = Conv2D(filters=filters, kernel_size=filter_size, strides=1, padding='same', activation='relu')(skip3)
c92 = Conv2D(filters=filters, kernel_size=filter_size, strides=1, padding='same', activation='relu')(c91)
c93 = Conv2D(filters=filters, kernel_size=filter_size, strides=1, padding='same', activation='relu')(c92)
BN9 = keras.layers.BatchNormalization()(c93)

#####################################
u4 = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', activation='relu')(BN9)  # UpSampling2D((2, 2))(BN9)

#####################################
skip2 = keras.layers.Concatenate()([c23, u4])
c101 = Conv2D(filters=filters, kernel_size=filter_size, strides=1, padding='same', activation='elu')(skip2)
c103 = Conv2D(filters=filters, kernel_size=filter_size, strides=1, padding='same', activation='elu')(c101)
BN10 = keras.layers.BatchNormalization()(c103)

#####################################
u5 = Conv2DTranspose(1, (3, 3), strides=(2, 2), padding='same', activation='relu')(BN10)  # UpSampling2D((2, 2))(BN10)
#####################################
skip1 = keras.layers.Concatenate()([c13, u5])
c111 = Conv2D(filters=filters, kernel_size=filter_size, strides=1, padding='same', activation='relu')(skip1)
c113 = Conv2D(filters=filters, kernel_size=filter_size, strides=1, padding='same', activation='relu')(c111)
BN11 = keras.layers.BatchNormalization()(c113)

#####################################
# Output layer
#####################################
output = (Conv2D(1, (1, 1), padding='same', activation='sigmoid'))(BN11)
Out = Dense(1, activation='softmax')(BN11)
#####################################
model = Model(inputs=inputs, outputs=output)


#####################################


# Custom loss function
def custom_loss(y_true, y_pred,smooth =1):  #Dice score function
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return -(2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


# Custom metric
def iou_loss_core(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    union = K.sum(y_true,-1) + K.sum(y_pred,-1) - intersection
    iou = (intersection + smooth) / ( union + smooth)
    return iou
###################
model.compile(optimizer='adam', loss=custom_loss, metrics=[iou_loss_core])
model.fit(np.array(x_train), np.array(y_train), batch_size=batch_size, epochs=epochs, validation_split=validation_split)
model.summary()

############################################
score = model.evaluate(test_x_samples, tests_y_samples, verbose=0)
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])
######################################
# Predicting the output of an image
#####################################
m_IoU = 0
count = 0


#####################################
def Training():
    prediction = model.predict(tests_y_samples, batch_size=1)
    prediction = np.asarray(prediction)
    #####################################
    for i in range(380):
        damage = np.squeeze(prediction[i], axis=2)
        original = np.squeeze(test_x_samples[i], axis=2)
        mask = np.squeeze(tests_y_samples[i], axis=2)
        fig = plt.figure(figsize=(16, 9))
        ax1 = fig.add_subplot(1, 3, 1)
        plt.imshow(damage, cmap='tab20c')
        ax2 = fig.add_subplot(1, 3, 2)
        plt.imshow(original, cmap='gist_yarg')
        # plt.imshow(damage,alpha=0.35,cmap='tab20c')
        ax3 = fig.add_subplot(1, 3, 3)
        plt.imshow(mask, cmap='gist_gray')
        ax1.title.set_text('Detected Damage')
        ax2.title.set_text('Original input Image')
        ax3.title.set_text('Ground Truth / Label')
        plt.show()
        #####################################


def exp():
    prediction = model.predict(experimental, batch_size=1)
    prediction = np.asarray(prediction)
    #####################################
    for i in range(380):
        damage = np.squeeze(prediction[i], axis=2)
        original = np.squeeze(experimental[i], axis=2)
        fig = plt.figure(figsize=(16, 9))
        ax1 = fig.add_subplot(1, 3, 1)
        plt.imshow(damage, cmap='cool')
        ax2 = fig.add_subplot(1, 3, 2)
        plt.imshow(original, cmap='gist_yarg')
        ax3 = fig.add_subplot(1, 3, 3)
        plt.imshow(original, cmap='gist_yarg')
        plt.imshow(damage, alpha=.65, cmap='gist_yarg')

        ax3.title.set_text('Original Image with mask')
        ax1.title.set_text('Detected Damage')
        ax2.title.set_text('Original input Image')
        plt.show()
        #####################################


Training()
#exp()
gc.collect()
############################################


model.save(
    'E:/backup/models/SegNet_models/SegNet_encoder_decoder_new_data_without_botleneck_using_Conv2dTranspose_with_custom_loss.h5')
#####################################
