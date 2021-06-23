import gc
import os

import keras
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.activations import relu, sigmoid
from keras.layers import Conv2D, Input
from keras.models import Model
from sklearn.model_selection import train_test_split, KFold
from tensorflow.python.keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt

# force to run with Tesla V100 GPU
with tf.device('/gpu:0'):
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

    ####################################################################################################################
    # Hyper parameters

    lr = .0001
    rho = 0.995
    filters = 16
    filterSize = (3, 3)
    activation = relu, sigmoid
    batch_size = 8
    dropout = 0.2
    epochs = 20
    validation_split = 0.2
    ConvFilter = 16

    gcn_block_filter = 64

    ####################################################################################################################
    # K value (greater than 3)

    k_gcn = 7

    ####################################################################################################################
    # Loading dataset into x, y arrays and reshape them

    dataset = np.load('delamination_dataset.npy')

    print(dataset.dtype)

    samples = dataset[:, :, :, 0]
    labels = dataset[:, :, :, 1]

    # defining training and testing sets with its labels

    Train_x, Test_x, Train_label, Test_y = train_test_split(samples, labels, test_size=0.2, shuffle=False)

    # shape become (n,x,y,d)

    Test_x = np.expand_dims(Test_x, axis=3)

    Test_y = np.expand_dims(Test_y, axis=3)

    # change to categorical of 2 classes

    Test_y = to_categorical(Test_y, 2)

    ####################################################################################################################
    # defining the Global Convolutional block

    def GCN_block(gcn_input, n_filters, K_GCN):
        print(gcn_input.shape)
        l11 = Conv2D(n_filters, (K_GCN, 1), padding='same')(gcn_input)
        l12 = Conv2D(n_filters, (1, K_GCN), padding='same')(l11)

        l21 = Conv2D(n_filters, (1, K_GCN), padding='same')(gcn_input)
        l22 = Conv2D(n_filters, (K_GCN, 1), padding='same')(l21)

        return keras.layers.Concatenate(axis=-1)([l12, l22])


    ####################################################################################################################
    # defining the Boundary Refinement block

    def BR_block(br_input):
        BN = keras.layers.BatchNormalization()(br_input)
        l1 = Conv2D(21, (3, 3), activation='relu', padding='same')(BN)
        l2 = Conv2D(21, (3, 3), padding='same')(l1)
        output_layer = keras.layers.Concatenate(axis=-1)([l2, br_input])
        return output_layer


    ####################################################################################################################

    # defining the residual block

    def res(input_res, n):
        BN = keras.layers.BatchNormalization()(input_res)
        conv = Conv2D(n, (5, 5), padding='same', activation='relu')(BN)
        res_ = keras.layers.Concatenate(axis=-1)([input_res, conv])
        down = keras.layers.MaxPooling2D((2, 2), (2, 2))(res_)
        return down


    # defining the deconvolution block

    def Deconv(TU_input, filters_tu):
        Up = keras.layers.Conv2DTranspose(filters=filters_tu,
                                          kernel_size=(3, 3),
                                          strides=(2, 2),
                                          padding='same')(TU_input)
        return Up


    ####################################################################################################################

    # defining the intersection over union metric

    def iou_metric(y_true, y_pred, smooth=1):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(K.abs(y_true_f * y_pred_f))
        union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
        iou = (intersection + smooth) / (union + smooth)
        return iou


    ####################################################################################################################

    # The global convolutional network model

    def GCN_model(K_GCN):
        inputs = Input(shape=(512, 512, 1))

        res1 = res(inputs, 64)
        #########################################################
        res2 = res(res1, 128)
        gcn1 = GCN_block(res2, gcn_block_filter, K_GCN)
        br1 = BR_block(gcn1)
        #########################################################
        res3 = res(res2, 256)
        gcn2 = GCN_block(res3, gcn_block_filter, K_GCN)
        br2 = BR_block(gcn2)
        #########################################################
        res4 = res(res3, 512)
        gcn3 = GCN_block(res4, gcn_block_filter, K_GCN)
        br3 = BR_block(gcn3)
        #########################################################
        res5 = res(res4, 1024)
        gcn4 = GCN_block(res5, gcn_block_filter, K_GCN)
        br4 = BR_block(gcn4)
        #########################################################
        # De-convolutional phase
        #########################################################
        deconv1 = Deconv(br4, gcn_block_filter)
        concat1 = keras.layers.Concatenate(axis=-1)([deconv1, br3])
        br5 = BR_block(concat1)
        deconv2 = Deconv(br5, gcn_block_filter)
        concat2 = keras.layers.Concatenate(axis=-1)([deconv2, br2])
        br6 = BR_block(concat2)
        deconv3 = Deconv(br6, gcn_block_filter)
        concat3 = keras.layers.Concatenate(axis=-1)([deconv3, br1])
        br7 = BR_block(concat3)
        deconv4 = Deconv(br7, gcn_block_filter)
        br8 = BR_block(deconv4)
        deconv5 = Deconv(br8, gcn_block_filter)
        output = keras.layers.Conv2D(2, (1, 1), activation='softmax')(deconv5)

        gcn_model = Model(inputs=inputs, outputs=output)
        ################################################################################################################
        gcn_model.compile(optimizer='adam',
                          loss=keras.losses.categorical_crossentropy,
                          metrics=[iou_metric])

        return gcn_model


    ####################################################################################################################

    # defining the number of folds

    n_split = 5  # Number of Folds
    counter = 1

    average_training_loss = []
    average_val_loss = []

    average_training_accuracy = []
    average_val_accuracy = []

    ####################################################################################################################

    for train_index, test_index in KFold(n_split, shuffle=True).split(Train_x):
        x_train, x_val = Train_x[train_index], Train_x[test_index]
        y_train, y_val = Train_label[train_index], Train_label[test_index]

        x_train = np.expand_dims(x_train, axis=3)

        y_train = np.expand_dims(y_train, axis=3)

        x_val = np.expand_dims(x_val, axis=3)

        y_val = np.expand_dims(y_val, axis=3)

        y_train = to_categorical(y_train, 2)
        y_val = to_categorical(y_val, 2)

        model_kfold = GCN_model(K_GCN=k_gcn)

        ################################################################################################################
        history = model_kfold.fit(x_train, y_train,
                                  shuffle=False,
                                  batch_size=batch_size,
                                  epochs=epochs,
                                  validation_data=(x_val, y_val))
        ################################################################################################################
        score = model_kfold.evaluate(Test_x,
                                     Test_y,
                                     batch_size=2,
                                     verbose=1)
        ################################################################################################################

        print(score[0], score[1])
        model_kfold.summary()

        average_training_loss = + np.asarray(history.history['loss'])
        average_val_loss = + np.asarray(history.history['val_loss'])

        print('Average test loss:', average_training_loss)
        print('Average val loss:', average_val_loss)

        average_training_accuracy = + np.asarray(history.history['iou_metric'])
        average_val_accuracy = + np.asarray(history.history['val_iou_metric'])

        print('Average test iou:', average_training_accuracy)
        print('Average val iou:', average_val_accuracy)

        os.chdir('/home/aijjeh/aijjeh_rexio_share/reports/figures/comparative_study/h5_models/GCN_models/')

        # model_kfold.save('GCN_model_K_7_fold_%d.h5' % counter)
        counter = counter + 1
    ####################################################################################################################
    # plt.figure(figsize=(10 / 2.54, 5 / 2.54), dpi=600)
    font = {'family': 'times new roman',
            'weight': 'light',
            'size': 6}
    plt.rc('font', **font)
    ################################################################################################################

    os.chdir('/home/aijjeh/aijjeh_rexio_share/reports/figures/comparative_study/losses_metrics_figures/')

    average_training_loss.reshape(-1, epochs)
    average_val_loss.reshape(-1, epochs)
    average_training_accuracy.reshape(-1, epochs)
    average_val_accuracy.reshape(-1, epochs)

    plt.plot(average_training_loss, label='training loss')
    plt.plot(average_val_loss, label='validation loss')

    plt.title('GCN model', font)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig('GCN_kfold_loss_per_epochs_softmax')
    plt.close('all')
    gc.collect()

    ####################################################################################################################
    # plt.figure(figsize=(10 / 2.54, 5 / 2.54), dpi=600)
    ####################################################################################################################
    plt.plot(average_training_accuracy, label='training iou')
    plt.plot(average_val_accuracy, label='validation iou')

    plt.title('GCN model', font)
    plt.ylabel('accuracy score')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig('GCN_kfold_iou_per_epochs_softmax')
    plt.close('all')
    gc.collect()
