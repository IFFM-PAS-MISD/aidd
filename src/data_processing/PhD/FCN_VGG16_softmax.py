import gc
import keras
import numpy as np
import tensorflow as tf
from keras import backend as K, regularizers
from keras.callbacks import EarlyStopping
from keras.layers import Conv2D, MaxPool2D, Input, Dropout, BatchNormalization, Activation
from keras.models import Model
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split, KFold
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
filter_size = (3, 3)
dropout = 0.5


########################################################################################################################
############################## Loading dataset into x, y arrays and reshape them #######################################
########################################################################################################################
# x = np.load('E:/backup/datasets/Segmentation datasets/augemented/June_training_dataset.npy')
# x = x.reshape(1900, 512, 512, 1)
# x = x / 255.0  # normalizing x,y to (0-1) range
# y = np.load('E:/backup/datasets/Segmentation datasets/augemented/June_labels.npy')
# y = y.reshape(1900, 512, 512, 1)
# y = y / 255.0
#
# ########################################################################################################################
# ###################################### Shuffle the data set at random ##################################################
# ########################################################################################################################
# # x, y = shuffle(x, y)
# ########################################################################################################################
# ################# Split dataset into training and testing sets and again re-shuffle them ###############################
# ########################################################################################################################
# x_train = x[:1520]
# y_train = y[:1520]
#
# test_x_samples = x[1520:1900]
# test_y_samples = y[1520:1900]
#
# y_train = to_categorical(y_train)
# tests_y_samples = to_categorical(test_y_samples)


########################################################################################################################
# Custom metric
def iou_metric(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(K.abs(y_true_f * y_pred_f))
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou


########################################################################################################################

########################################################################################################################
######################################   VGG16 encoder decoder Model   #################################################
########################################################################################################################
def VGG16_encoder_decoder():
    inputs = Input(shape=(512, 512, 1))
    ####################################################################################################################
    ############################## Backbone Down sampling convolution followed by max-pooling ##########################
    ####################################################################################################################

    conv_1 = Conv2D(filters, filter_size, padding="same")(inputs)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Activation("relu")(conv_1)

    conv_2 = Conv2D(filters, filter_size, padding="same")(conv_1)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Activation("relu")(conv_2)
    ####################################################################################################################
    d1 = MaxPool2D((2, 2), (2, 2))(conv_2)
    d1 = Dropout(dropout)(d1)
    ####################################################################################################################
    conv_3 = Conv2D(filters * 2, filter_size, padding="same")(d1)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = Activation("relu")(conv_3)

    conv_4 = Conv2D(filters * 2, filter_size, padding="same")(conv_3)
    conv_4 = BatchNormalization()(conv_4)
    conv_4 = Activation("relu")(conv_4)
    ####################################################################################################################
    d2 = MaxPool2D((2, 2), (2, 2))(conv_4)
    d2 = Dropout(dropout)(d2)
    ####################################################################################################################
    conv_5 = Conv2D(filters * 4, filter_size, padding="same")(d2)
    conv_5 = BatchNormalization()(conv_5)
    conv_5 = Activation("relu")(conv_5)

    conv_6 = Conv2D(filters * 4, filter_size, padding="same")(conv_5)
    conv_6 = BatchNormalization()(conv_6)
    conv_6 = Activation("relu")(conv_6)

    conv_7 = Conv2D(filters * 4, filter_size, padding="same")(conv_6)
    conv_7 = BatchNormalization()(conv_7)
    conv_7 = Activation("relu")(conv_7)
    ####################################################################################################################
    d3 = MaxPool2D((2, 2), (2, 2))(conv_7)
    d3 = Dropout(dropout)(d3)
    ####################################################################################################################
    conv_8 = Conv2D(filters * 8, filter_size, padding="same")(d3)
    conv_8 = BatchNormalization()(conv_8)
    conv_8 = Activation("relu")(conv_8)

    conv_9 = Conv2D(filters * 8, filter_size, padding="same")(conv_8)
    conv_9 = BatchNormalization()(conv_9)
    conv_9 = Activation("relu")(conv_9)

    conv_10 = Conv2D(filters * 8, filter_size, padding="same")(conv_9)
    conv_10 = BatchNormalization()(conv_10)
    conv_10 = Activation("relu")(conv_10)
    ####################################################################################################################
    d4 = MaxPool2D((2, 2), (2, 2))(conv_10)
    d4 = Dropout(dropout)(d4)
    ####################################################################################################################
    conv_11 = Conv2D(filters * 16, filter_size, padding="same")(d4)
    conv_11 = BatchNormalization()(conv_11)
    conv_11 = Activation("relu")(conv_11)

    conv_12 = Conv2D(filters * 16, filter_size, padding="same")(conv_11)
    conv_12 = BatchNormalization()(conv_12)
    conv_12 = Activation("relu")(conv_12)

    conv_13 = Conv2D(filters * 16, filter_size, padding="same")(conv_12)
    conv_13 = BatchNormalization()(conv_13)
    conv_13 = Activation("relu")(conv_13)
    ####################################################################################################################
    d5 = MaxPool2D((2, 2))(conv_13)
    d5 = Dropout(dropout)(d5)
    ####################################################################################################################
    ################################## Up sampling convolution followed by up-sampling #################################
    ####################################################################################################################
    u1 = keras.layers.UpSampling2D((2, 2), interpolation='nearest')(d5)  # ,interpolation='bilinear'
    ####################################################################################################################
    skip5 = keras.layers.Concatenate()([conv_13, u1])

    conv_14 = Conv2D(filters * 16, filter_size, padding="same")(skip5)
    conv_14 = BatchNormalization()(conv_14)
    conv_14 = Activation("relu")(conv_14)

    conv_15 = Conv2D(filters * 16, filter_size, padding="same")(conv_14)
    conv_15 = BatchNormalization()(conv_15)
    conv_15 = Activation("relu")(conv_15)

    conv_16 = Conv2D(filters * 16, filter_size, padding="same")(conv_15)
    conv_16 = BatchNormalization()(conv_16)
    conv_16 = Activation("relu")(conv_16)
    ####################################################################################################################
    u2 = keras.layers.UpSampling2D((2, 2), interpolation='nearest')(conv_16)
    ####################################################################################################################
    skip4 = keras.layers.Concatenate()([conv_10, u2])

    conv_17 = Conv2D(filters * 8, filter_size, padding="same")(skip4)
    conv_17 = BatchNormalization()(conv_17)
    conv_17 = Activation("relu")(conv_17)

    conv_18 = Conv2D(filters * 8, filter_size, padding="same")(conv_17)
    conv_18 = BatchNormalization()(conv_18)
    conv_18 = Activation("relu")(conv_18)

    conv_19 = Conv2D(filters * 8, filter_size, padding="same")(conv_18)
    conv_19 = BatchNormalization()(conv_19)
    conv_19 = Activation("relu")(conv_19)
    ####################################################################################################################
    u3 = keras.layers.UpSampling2D((2, 2), interpolation='nearest')(conv_19)
    ####################################################################################################################
    skip3 = keras.layers.Concatenate()([conv_7, u3])
    conv_20 = Conv2D(filters * 4, filter_size, padding="same")(skip3)
    conv_20 = BatchNormalization()(conv_20)
    conv_20 = Activation("relu")(conv_20)

    conv_21 = Conv2D(filters * 4, filter_size, padding="same")(conv_20)
    conv_21 = BatchNormalization()(conv_21)
    conv_21 = Activation("relu")(conv_21)

    conv_22 = Conv2D(filters * 4, filter_size, padding="same")(conv_21)
    conv_22 = BatchNormalization()(conv_22)
    conv_22 = Activation("relu")(conv_22)
    ####################################################################################################################
    u4 = keras.layers.UpSampling2D((2, 2), interpolation='nearest')(conv_22)
    ####################################################################################################################
    skip2 = keras.layers.Concatenate()([conv_4, u4])
    conv_23 = Conv2D(filters * 2, filter_size, padding="same")(skip2)
    conv_23 = BatchNormalization()(conv_23)
    conv_23 = Activation("relu")(conv_23)

    conv_24 = Conv2D(filters * 2, filter_size, padding="same")(conv_23)
    conv_24 = BatchNormalization()(conv_24)
    conv_24 = Activation("relu")(conv_24)
    ####################################################################################################################
    u5 = keras.layers.UpSampling2D((2, 2), interpolation='nearest')(conv_24)
    ####################################################################################################################
    skip1 = keras.layers.Concatenate()([conv_2, u5])
    conv_25 = Conv2D(filters, filter_size, padding="same")(skip1)
    conv_25 = BatchNormalization()(conv_25)
    conv_25 = Activation("relu")(conv_25)

    conv_26 = Conv2D(filters, filter_size, padding="same")(conv_25)
    conv_26 = BatchNormalization()(conv_26)
    conv_26 = Activation("relu")(conv_26)
    ####################################################################################################################
    ################################################## Output layer ####################################################
    ####################################################################################################################
    output = (Conv2D(2, (1, 1), padding='same', activation='softmax'))(conv_26)
    ####################################################################################################################
    model = Model(inputs=inputs, outputs=output)
    ####################################################################################################################
    model.compile(optimizer='adam',
                  loss=keras.losses.categorical_crossentropy,
                  metrics=[iou_metric])
    ####################################################################################################################
    return model


########################################################################################################################
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

n_split = 5  # Number of Folds

########################################################################################################################
#########################################  KFold Cross validation   ####################################################
########################################################################################################################

for train_index, test_index in KFold(n_split, shuffle=True, random_state=45).split(Train_x):
    x_train, x_val = Train_x[train_index], Train_x[test_index]
    y_train, y_val = Train_label[train_index], Train_label[test_index]

    x_train = np.expand_dims(x_train, axis=3)
    y_train = np.expand_dims(y_train, axis=3)
    x_val = np.expand_dims(x_val, axis=3)
    y_val = np.expand_dims(y_val, axis=3)

    y_train = to_categorical(y_train)
    y_val = to_categorical(y_val)

    model_kfold = None
    model_kfold = VGG16_encoder_decoder()
    ####################################################################################################################
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
                              batch_size=8,
                              validation_data=(x_val, y_val))
    # callbacks=[earlystop])
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
        'E:/aidd_new/aidd/reports/figures/comparative_study/losses_metrics_figures/VGG16_encoder_decoder_kfold_loss_per_epochs_softmax')
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
        'E:/aidd_new/aidd/reports/figures/comparative_study/losses_metrics_figures/VGG16_encoder_decoder_kfold_iou_per_epochs_softmax')
    gc.collect()

    plt.close('all')
    gc.collect()
    model_kfold.save(
        'E:/aidd_new/aidd/reports/figures/comparative_study/h5_models/VGG16_encoder_decoder_kfold_softmax.h5')
    gc.collect()

# model.save('E:/backup/models/SegNet_models/Softmax/VGG16_100_epochs_softmax.h5')
