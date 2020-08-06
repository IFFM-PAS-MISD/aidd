import gc
import keras
import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.layers import Conv2D, MaxPool2D, Input, Conv2DTranspose
from keras.models import Model
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import matplotlib.pyplot as plt

###########################################   memory growing ###########################################################
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
########################################################################################################################
# Force the Garbage Collector to release unreferenced memory
gc.collect()
########################################################################################################################
############################################ Hyper parameters ##########################################################
########################################################################################################################
filters = 16
epsilon = 0.1
dropout_rate = 0.5
epochs = 100


########################################################################################################################
############################## Loading dataset into x, y arrays and reshape them #######################################
########################################################################################################################
# x = np.load('E:/backup/datasets/Segmentation datasets/augemented/June_training_dataset.npy')
# x = x.reshape(1900, 512, 512, 1)
# x = x / 255.0  # normalizing x,y to (0-1) range
# y = np.load('E:/backup/datasets/Segmentation datasets/augemented/June_labels.npy')
# y = y.reshape(1900, 512, 512, 1)
# y = y / 255.0

########################################################################################################################
###################################### Shuffle the data set at random ##################################################
########################################################################################################################
# x, y = shuffle(x, y)
########################################################################################################################
################# Split dataset into training and testing sets and again re-shuffle them ###############################
########################################################################################################################
# x_train = x[:1520]
# y_train = y[:1520]
#
# test_x_samples = x[1520:1900]
# test_y_samples = y[1520:1900]
#
# y_train = to_categorical(y_train)
# tests_y_samples = to_categorical(test_y_samples)


########################################################################################################################
# def dice_coef(y_true, y_pred, smooth=1):
#     intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
#     return (2. * intersection + smooth) / (K.sum(K.square(y_true), -1) + K.sum(K.square(y_pred), -1) + smooth)
#
#
# ########################################################################################################################
# def dice_coef_loss(y_true, y_pred):
#     return 1 - dice_coef(y_true, y_pred)
#
#
# ########################################################################################################################
# # Dice loss / F1
# def dice_loss(y_true, y_pred):
#     numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=-1)
#     denominator = tf.reduce_sum(y_true + y_pred, axis=-1)
#
#     return 1 - (numerator + 1) / (denominator + 1)
#
#
# ########################################################################################################################
# def dice_loss_1(y_true, y_pred):  # did not give good results
#     numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3))
#     denominator = tf.reduce_sum(y_true + y_pred, axis=(1, 2, 3))
#     return 1 - numerator / denominator
#
#
#
# ########################################################################################################################
# # Custom metric IoU
# def iou_loss_core(y_true, y_pred, smooth=1):
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)
#     intersection = K.sum(K.abs(y_true_f * y_pred_f))  # , axis=-1
#     union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection / 2
#     iou = (intersection + smooth) / (union + smooth)
#     return iou
#
#
# ########################################################################################################################
# def recall(y_true, y_pred):
#     y_true = K.ones_like(y_true)
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     all_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
#     recall_acc = true_positives / (all_positives + K.epsilon())
#     return recall_acc
#
#
# ########################################################################################################################
# def precision(y_true, y_pred):
#     y_true = K.ones_like(y_true)
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#
#     predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
#     precision_acc = true_positives / (predicted_positives + K.epsilon())
#     return precision_acc
#
#
# ########################################################################################################################
# def f1_score(y_true, y_pred):
#     precision_m = precision(y_true, y_pred)
#     recall_m = recall(y_true, y_pred)
#     return 2 * ((precision_m * recall_m) / (precision_m + recall_m + K.epsilon()))
#

########################################################################################################################
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
############################################## The Model UNet ##########################################################
########################################################################################################################
def Unet_softmax():
    inputs = Input(shape=(512, 512, 1))
    # Backbone Down sampling convolution followed by max-pooling
    ####################################################################################################################
    c11 = Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(inputs)
    BN = batch_normalizer(c11)
    c12 = Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(BN)
    BN1 = batch_normalizer(c12)
    d1 = MaxPool2D((2, 2), (2, 2))(BN1)
    d1 = keras.layers.Dropout(dropout_rate)(d1)
    ####################################################################################################################
    c21 = Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(d1)
    BN = batch_normalizer(c21)
    c22 = Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(BN)
    BN2 = batch_normalizer(c22)
    d2 = MaxPool2D((2, 2), (2, 2))(BN2)
    d2 = keras.layers.Dropout(dropout_rate)(d2)
    ####################################################################################################################
    c31 = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(d2)
    BN = batch_normalizer(c31)
    c32 = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(BN)
    BN3 = batch_normalizer(c32)
    d3 = MaxPool2D((2, 2), (2, 2))(BN3)
    d3 = keras.layers.Dropout(dropout_rate)(d3)
    ####################################################################################################################
    c41 = Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(d3)
    BN = batch_normalizer(c41)
    c42 = Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(BN)
    BN4 = batch_normalizer(c42)
    d4 = MaxPool2D((2, 2), (2, 2))(BN4)
    d4 = keras.layers.Dropout(dropout_rate)(d4)
    ####################################################################################################################
    # bottleneck layer
    ####################################################################################################################
    c51 = Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(d4)
    BN = batch_normalizer(c51)
    c52 = Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(BN)
    BN = batch_normalizer(c52)
    ####################################################################################################################
    # Up sampling convolution followed by ConvTranspose
    ####################################################################################################################
    u1 = Conv2DTranspose(filters * 8, (3, 3), strides=(2, 2), padding='same')(BN)
    skip4 = keras.layers.Concatenate()([BN4, u1])
    skip4 = keras.layers.Dropout(dropout_rate)(skip4)  # adding dropout layer
    c61 = Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(skip4)
    BN = batch_normalizer(c61)
    c62 = Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(BN)
    BN = batch_normalizer(c62)
    ####################################################################################################################
    u2 = Conv2DTranspose(filters * 4, (3, 3), strides=(2, 2), padding='same', activation='relu')(BN)
    skip3 = keras.layers.Concatenate()([BN3, u2])
    skip3 = keras.layers.Dropout(dropout_rate)(skip3)
    c71 = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(skip3)
    BN = batch_normalizer(c71)
    c72 = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(BN)
    BN = batch_normalizer(c72)
    ####################################################################################################################
    u3 = Conv2DTranspose(filters * 2, (3, 3), strides=(2, 2), padding='same')(BN)
    skip2 = keras.layers.Concatenate()([BN2, u3])
    skip2 = keras.layers.Dropout(dropout_rate)(skip2)
    c81 = Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(skip2)
    BN = batch_normalizer(c81)
    c82 = Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(BN)
    BN = batch_normalizer(c82)
    ####################################################################################################################
    u4 = Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding='same', activation='relu')(BN)
    skip1 = keras.layers.Concatenate()([BN1, u4])
    skip1 = keras.layers.Dropout(dropout_rate)(skip1)
    c91 = Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(skip1)
    BN = batch_normalizer(c91)
    c92 = Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(BN)
    BN = batch_normalizer(c92)
    ####################################################################################################################
    # Output layer
    ####################################################################################################################
    output = Conv2D(2, (1, 1), activation='softmax')(BN)
    ####################################################################################################################
    model = Model(inputs=inputs, outputs=output)
    ####################################################################################################################
    model.compile(optimizer="adam",
                  loss=keras.losses.categorical_crossentropy,
                  metrics=[iou_metric])
    return model


########################################################################################################################
############################## Loading dataset into x, y arrays and reshape them #######################################
########################################################################################################################

dataset = np.load('delamination_dataset.npy')

print(dataset.dtype)

samples = dataset[:, :, :, 0]
labels = dataset[:, :, :, 1]

Train_x, Test_x, Train_label, Test_label = train_test_split(samples, labels, test_size=0.2, shuffle=False)

Test_x = np.expand_dims(Test_x, axis=3)
Test_label = np.expand_dims(Test_label, axis=3)

Test_label = to_categorical(Test_label)

########################################################################################################################
########################################################################################################################
n_split = 4  # Number of Folds

########################################################################################################################
#########################################  KFold Cross validation   ####################################################
########################################################################################################################

for train_index, test_index in KFold(n_split, shuffle=True, random_state=52).split(Train_x):
    x_train, x_val = Train_x[train_index], Train_x[test_index]
    y_train, y_val = Train_label[train_index], Train_label[test_index]

    x_train = np.expand_dims(x_train, axis=3)
    y_train = np.expand_dims(y_train, axis=3)
    x_val = np.expand_dims(x_val, axis=3)
    y_val = np.expand_dims(y_val, axis=3)

    y_train = to_categorical(y_train)
    y_val = to_categorical(y_val)

    model_kfold = None
    model_kfold = Unet_softmax()

    earlystop = EarlyStopping(monitor='val_iou_metric',
                              # min_delta=1,
                              patience=10,
                              verbose=1,
                              mode="max",
                              restore_best_weights=True)
    ####################################################################################################################
    history = model_kfold.fit(x_train,
                              y_train,
                              epochs=100,
                              batch_size=4,
                              validation_data=(x_val, y_val),
                              callbacks=[earlystop])
    ####################################################################################################################
    score = model_kfold.evaluate(Test_x,
                                 Test_label,
                                 batch_size=4,
                                 verbose=1)
    ####################################################################################################################
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    model_kfold.summary()
    ####################################################################################################################
    print('Average test loss:', np.average(history.history['loss']))
    print('Average val loss:', np.average(history.history['val_loss']))
    ####################################################################################################################
    plt.figure(figsize=(10 / 2.54, 5 / 2.54), dpi=600)
    font = {'family': 'times new roman',
            'weight': 'light',
            'size': 8}
    plt.rc('font', **font)
    # plt.gca().set_axis_off()
    # plt.axis('off')
    # plt.subplots_adjust(left=0.0, bottom=0.0, right=1, top=1.0, wspace=0.0, hspace=0.0)
    # plt.margins(0, 0)
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())
    ################################################################################################################

    plt.plot(history.history['loss'], label='training loss')
    plt.plot(history.history['val_loss'], label='validation loss')
    plt.legend()
    plt.savefig(
        'E:/aidd_new/aidd/reports/figures/comparative_study/losses_metrics_figures/Unet_kfold_loss_per_epochs_softmax')
    plt.close('all')
    gc.collect()

    ####################################################################################################################
    plt.figure(figsize=(10 / 2.54, 5 / 2.54), dpi=600)
    font = {'family': 'times new roman',
            'weight': 'light',
            'size': 11}
    plt.rc('font', **font)
    ####################################################################################################################
    plt.plot(history.history['iou_metric'], label='training iou')
    plt.plot(history.history['val_iou_metric'], label='validation iou')
    plt.legend()
    plt.savefig(
        'E:/aidd_new/aidd/reports/figures/comparative_study/losses_metrics_figures/Unet_kfold_iou_per_epochs_softmax')
    gc.collect()

    plt.close('all')
    gc.collect()
    model_kfold.save('E:/aidd_new/aidd/reports/figures/comparative_study/h5_models/Unet_kfold_softmax.h5')
    gc.collect()

# model.save('E:/backup/models/UNet_models/Softmax/Unet_100_epoches_softmax.h5')
