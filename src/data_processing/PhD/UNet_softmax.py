import gc
import os

import keras
import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.layers import Conv2D, MaxPool2D, Input, Conv2DTranspose, Activation
from keras.models import Model
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split, KFold

import matplotlib.pyplot as plt

#   memory growing

gc.collect()

config = tf.compat.v1.ConfigProto(device_count={'GPU': 2, 'CPU': 32})
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

# Hyper parameters

epsilon = 0.1
dropout_rate = 0.2
epochs = 20


########################################################################################################################

def iou_metric(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(K.abs(y_true_f * y_pred_f))
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou


########################################################################################################################
def batch_normalizer(bn_input):
    return keras.layers.BatchNormalization()(bn_input)


########################################################################################################################
# The Model UNet
########################################################################################################################
def Unet_model():
    inputs = Input(shape=(512, 512, 1))

    # Backbone Down sampling convolution followed by max-pooling
    ####################################################################################################################

    c11 = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same')(inputs)
    BN = keras.layers.BatchNormalization()(c11)
    active_l = Activation('relu')(BN)

    c12 = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same')(active_l)
    BN = keras.layers.BatchNormalization()(c12)
    active_l = Activation('relu')(BN)

    c13 = Conv2D(64, (1, 1), padding='same')(inputs)

    concat = keras.layers.Concatenate()([active_l, c13])
    active_l_1 = Activation('relu')(concat)

    d1 = MaxPool2D((2, 2), (2, 2))(active_l_1)
    d1 = keras.layers.Dropout(dropout_rate)(d1)
    ####################################################################################################################

    c21 = Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same')(d1)
    BN = keras.layers.BatchNormalization()(c21)
    active_l = Activation('relu')(BN)

    c22 = Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same')(active_l)
    BN = keras.layers.BatchNormalization()(c22)
    active_l = Activation('relu')(BN)

    c23 = Conv2D(128, (1, 1), padding='same')(d1)

    concat = keras.layers.Concatenate()([active_l, c23])
    active_l_2 = Activation('relu')(concat)

    d2 = MaxPool2D((2, 2), (2, 2))(active_l_2)
    d2 = keras.layers.Dropout(dropout_rate)(d2)
    ####################################################################################################################
    c31 = Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same')(d2)
    BN = keras.layers.BatchNormalization()(c31)
    active_l = Activation('relu')(BN)

    c32 = Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same')(active_l)
    BN = keras.layers.BatchNormalization()(c32)
    active_l = Activation('relu')(BN)

    c33 = Conv2D(256, (1, 1), padding='same')(d2)

    concat = keras.layers.Concatenate()([active_l, c33])
    active_l_3 = Activation('relu')(concat)

    d3 = MaxPool2D((2, 2), (2, 2))(active_l_3)
    d3 = keras.layers.Dropout(dropout_rate)(d3)
    ####################################################################################################################
    c41 = Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same')(d3)
    BN = keras.layers.BatchNormalization()(c41)
    active_l = Activation('relu')(BN)

    c42 = Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same')(active_l)
    BN = keras.layers.BatchNormalization()(c42)
    active_l = Activation('relu')(BN)

    c43 = Conv2D(512, (1, 1), padding='same')(d3)

    concat = keras.layers.Concatenate()([active_l, c43])
    active_l_4 = Activation('relu')(concat)

    d4 = MaxPool2D((2, 2), (2, 2))(active_l_4)
    d4 = keras.layers.Dropout(dropout_rate)(d4)
    ####################################################################################################################
    # bottleneck layer
    ####################################################################################################################
    c51 = Conv2D(filters=1024, kernel_size=(3, 3), strides=1, padding='same')(d4)
    BN = keras.layers.BatchNormalization()(c51)
    active_l = Activation('relu')(BN)

    c52 = Conv2D(filters=1024, kernel_size=(3, 3), strides=1, padding='same')(active_l)
    BN = keras.layers.BatchNormalization()(c52)
    active_l = Activation('relu')(BN)

    c53 = Conv2D(1024, (1, 1), padding='same')(d4)

    concat = keras.layers.Concatenate()([active_l, c53])
    active_l_5 = Activation('relu')(concat)
    ####################################################################################################################
    # Up sampling convolution followed by ConvTranspose
    ####################################################################################################################
    u1 = Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same')(active_l_5)

    skip4 = keras.layers.Concatenate()([active_l_4, u1])
    skip4 = keras.layers.Dropout(dropout_rate)(skip4)  # adding dropout layer

    c61 = Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same')(skip4)
    BN = keras.layers.BatchNormalization()(c61)
    active_l = Activation('relu')(BN)

    c62 = Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same')(active_l)
    BN = keras.layers.BatchNormalization()(c62)
    active_l = Activation('relu')(BN)

    c63 = Conv2D(512, (1, 1), padding='same')(u1)

    concat = keras.layers.Concatenate()([active_l, c63])
    active_l = Activation('relu')(concat)
    ####################################################################################################################
    u2 = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same')(active_l)

    skip3 = keras.layers.Concatenate()([active_l_3, u2])
    skip3 = keras.layers.Dropout(dropout_rate)(skip3)

    c71 = Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same')(skip3)
    BN = keras.layers.BatchNormalization()(c71)
    active_l = Activation('relu')(BN)

    c72 = Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same')(active_l)
    BN = keras.layers.BatchNormalization()(c72)
    active_l = Activation('relu')(BN)

    c73 = Conv2D(256, (1, 1), padding='same')(u2)

    concat = keras.layers.Concatenate()([active_l, c73])
    active_l = Activation('relu')(concat)
    ####################################################################################################################
    u3 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(active_l)

    skip2 = keras.layers.Concatenate()([active_l_2, u3])
    skip2 = keras.layers.Dropout(dropout_rate)(skip2)

    c81 = Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same')(skip2)
    BN = keras.layers.BatchNormalization()(c81)
    active_l = Activation('relu')(BN)

    c82 = Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same')(active_l)
    BN = keras.layers.BatchNormalization()(c82)
    active_l = Activation('relu')(BN)

    c83 = Conv2D(128, (1, 1), padding='same')(u3)

    concat = keras.layers.Concatenate()([active_l, c83])
    active_l = Activation('relu')(concat)
    ####################################################################################################################
    u4 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(active_l)

    skip1 = keras.layers.Concatenate()([active_l_1, u4])
    skip1 = keras.layers.Dropout(dropout_rate)(skip1)

    c91 = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same')(skip1)
    BN = keras.layers.BatchNormalization()(c91)
    active_l = Activation('relu')(BN)

    c92 = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same')(active_l)
    BN = keras.layers.BatchNormalization()(c92)
    active_l = Activation('relu')(BN)

    c93 = Conv2D(64, (1, 1), padding='same')(u4)

    concat = keras.layers.Concatenate()([active_l, c93])
    active_l = Activation('relu')(concat)

    ####################################################################################################################
    # Output layer
    ####################################################################################################################
    output = Conv2D(2, (1, 1), activation='softmax')(active_l)
    ####################################################################################################################
    model = Model(inputs=inputs, outputs=output)
    ####################################################################################################################
    model.compile(optimizer="adam",
                  loss=keras.losses.categorical_crossentropy,
                  metrics=[iou_metric])
    return model


########################################################################################################################
# Loading dataset into x, y arrays and reshape them
########################################################################################################################

dataset = np.load('delamination_dataset.npy')

print(dataset.dtype)

samples = dataset[:, :, :, 0]
labels = dataset[:, :, :, 1]

Train_x, Test_x, Train_label, Test_label = train_test_split(samples, labels, test_size=0.2, shuffle=False)

Test_x = np.expand_dims(Test_x, axis=3)

Test_label = np.expand_dims(Test_label, axis=3)

Test_label = to_categorical(Test_label, 2)

########################################################################################################################
########################################################################################################################
n_split = 5  # Number of Folds
counter = 1

average_training_loss = []
average_val_loss = []

average_training_accuracy = []
average_val_accuracy = []

########################################################################################################################
#  KFold Cross validation
########################################################################################################################

for train_index, test_index in KFold(n_split, shuffle=True, random_state=52).split(Train_x):
    x_train, x_val = Train_x[train_index], Train_x[test_index]
    y_train, y_val = Train_label[train_index], Train_label[test_index]

    x_train = np.expand_dims(x_train, axis=3)

    y_train = np.expand_dims(y_train, axis=3)

    x_val = np.expand_dims(x_val, axis=3)

    y_val = np.expand_dims(y_val, axis=3)

    y_train = to_categorical(y_train, 2)
    y_val = to_categorical(y_val, 2)

    model_kfold = Unet_model()

    earlystop = EarlyStopping(monitor='val_iou_metric',
                              # min_delta=1,
                              patience=10,
                              verbose=1,
                              mode="max",
                              restore_best_weights=True)
    ####################################################################################################################
    history = model_kfold.fit(x_train,
                              y_train,
                              epochs=epochs,
                              batch_size=8,
                              validation_data=(x_val, y_val))
    # callbacks=[earlystop])

    ####################################################################################################################
    score = model_kfold.evaluate(Test_x,
                                 Test_label,
                                 batch_size=8,
                                 verbose=1)
    ####################################################################################################################

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    model_kfold.summary()

    average_training_loss = + np.asarray(history.history['loss'])
    average_val_loss = + np.asarray(history.history['val_loss'])

    print('Average test loss:', average_training_loss)
    print('Average val loss:', average_val_loss)

    average_training_accuracy = + np.asarray(history.history['iou_metric'])
    average_val_accuracy = + np.asarray(history.history['val_iou_metric'])

    print('Average test iou:', average_training_accuracy)
    print('Average val iou:', average_val_accuracy)

    os.chdir('/home/aijjeh/aijjeh_rexio_share/reports/figures/comparative_study/h5_models/residual_unet_models/')

    # model_kfold.save('updated_Unet_kfold_softmax_trial_%d.h5' % counter)

    counter = counter + 1
    gc.collect()

####################################################################################################################
# plt.figure(figsize=(10 / 2.54, 5 / 2.54), dpi=600)
font = {'family': 'times new roman',
        'weight': 'light',
        'size': 6}
plt.rc('font', **font)
####################################################################################################################
os.chdir('/home/aijjeh/aijjeh_rexio_share/reports/figures/comparative_study/losses_metrics_figures/')

average_training_loss.reshape(-1, epochs)
average_val_loss.reshape(-1, epochs)
average_training_accuracy.reshape(-1, epochs)
average_val_accuracy.reshape(-1, epochs)

plt.plot(average_training_loss, label='training loss')
plt.plot(average_val_loss, label='validation loss')

plt.title('Residual UNet model', font)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
plt.savefig('Unet_kfold_loss_per_epochs_softmax')
plt.close('all')
gc.collect()

####################################################################################################################
# plt.figure(figsize=(10 / 2.54, 5 / 2.54), dpi=600)
####################################################################################################################
plt.plot(average_training_accuracy, label='training iou')
plt.plot(average_val_accuracy, label='validation iou')

plt.title('Residual UNet model', font)
plt.ylabel('accuracy score')
plt.xlabel('epoch')
plt.legend()
plt.savefig('Unet_kfold_iou_per_epochs_softmax')
plt.close('all')
gc.collect()

# model.save('E:/backup/models/UNet_models/Softmax/Unet_100_epoches_softmax.h5')
