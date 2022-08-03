# ============================================
__title__ = 'Upsample_downsample for semantic segmentation'
__author__ = "Abdalraheem A. Ijjeh"
__maintainer__ = "Abdalraheem A. Ijjeh"
__email__ = "aijjeh@imp.gda.pl"

# ============================================
import tensorflow as tf
import keras
import tensorflow.python.framework.random_seed
import tensorflow_addons as tfa
import os
import numpy as np
from keras.models import Sequential, Model
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
from tensorflow.python.client import device_lib
import shutil
import random
import glob
import gc
import time
from tensorflow.python.client import device_lib
from keras.callbacks import Callback
from tensorflow.keras.models import Model
from keras import backend as K
from keras.utils import to_categorical
import neptune
from decouple import config
from keras.layers import TimeDistributed, \
    Input, ConvLSTM2D, UpSampling2D, Conv2D, \
    BatchNormalization, Add, MaxPool2D, concatenate, Conv3D, MaxPool3D
from numpy.random import seed
import warnings

# import tensorflow_addons as tfa

warnings.simplefilter(action='ignore', category=FutureWarning)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
print(device_lib.list_local_devices())
# strategy = tf.distribute.MirroredStrategy(devices=["GPU:0", "GPU:1"])
# print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
# dataset = tf.data.Dataset.range(42)
# options.experimental_distribute.auto_shard_policy = AutoShardPolicy.DATA
# dataset = dataset.with_options(options)


# seed(1)
# tensorflow.random.set_seed(2)
########################################################################################################################
# Link to neptune ai for monitoring
########################################################################################################################
# neptune.init(project_qualified_name='abdalraheem.ijjeh/aidd', api_token=config('NEPTUNE_API_TOKEN'))
# neptune.create_experiment('RNN_UNet-model-training')
# neptune.append_tag('RNN_UNet_model')
########################################################################################################################
""" 
- Define the loss and accuracy metrics jaccard distance, jaccard index, 
  focal loss function which is a weighted BCE, and the default iou
"""


def get_jaccard_index(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac


def get_iou_metric(y_true, y_pred, smooth=100):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(K.abs(y_true_f * y_pred_f))
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou


def get_f1_score(y_true, y_pred):
    def precision_m(y_true__, y_pred__):
        TP = K.sum(K.round(K.clip(y_true__ * y_pred__, 0, 1)), axis=-1)
        Pred_Positives = K.sum(K.round(K.clip(y_pred__, 0, 1)), axis=-1)

        precision_ = TP / (Pred_Positives + K.epsilon())
        return precision_

    def recall_m(y_true_, y_pred_):
        TP = K.sum(K.round(K.clip(y_true_ * y_pred_, 0, 1)), axis=-1)
        Positives = K.sum(K.round(K.clip(y_true_, 0, 1)), axis=-1)

        recall_ = TP / (Positives + K.epsilon())
        return recall_

    precision, recall = precision_m(y_true, y_pred), recall_m(y_true, y_pred)

    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def jaccard_loss(y_true, y_pred, smooth=1):
    """
    Arguments:
        y_true : Matrix containing one-hot encoded class labels
                 with the last axis being the number of classes.
        y_pred : Matrix with same dimensions as y_true.
        smooth : smoothing factor for loss function.
    """

    intersection = tf.reduce_sum(y_true * y_pred, axis=-1)
    union = tf.reduce_sum(y_true + y_pred, axis=-1) - intersection
    jac = (intersection + smooth) / (union + smooth)

    return (1 - jac) * smooth


def f1_loss(y_true, y_pred, smooth=1):
    intersection = tf.reduce_sum(y_true * y_pred, axis=-1)
    denominator = tf.reduce_sum(y_true + y_pred, axis=-1)
    f1 = (2 * intersection + smooth) / (denominator + smooth)

    return (1 - f1) * smooth


########################################################################################################################
filters = 16
filter_size = 3
epsilon = 0.1
dropout_rate = 0.2
epochs = 10
depth = 3
########################################################################################################################
# load dataset
########################################################################################################################
def load_dataset():
    os.chdir('/home/aijjeh/Desktop/Phd_Project/Full_wavefield_frames_time_series_project/Datasets/')
    training_set = np.load('training_set/training_consecutive_448_30_consecutive_frames_not_normalised.npy',
                           mmap_mode='r')
    training_set = training_set.reshape((475, 30, 448, 448, 1))
    training_set = training_set.astype('float32')
    training_set = training_set / 255.0
    labels = np.load('label_set/GT_labels_thresholded_448_only_475_labels.npy')
    labels = labels.reshape((475, 448, 448, 1))
    labels = labels.astype('float32')
    train_x = training_set[0:380]
    train_label = labels[0:380]
    # train_label = to_categorical(train_label, 2)
    return train_x, train_label


########################################################################################################################
def group_convolution(inputs_, power):
    # layer = Conv3D(filters,  # * 2 ** power,
    #                kernel_size=(1, filter_size, filter_size),
    #                padding='same',
    #                activation='relu')(inputs_)
    # layer = BatchNormalization()(layer)
    # layer = Conv3D(filters,  # * 2 ** power,
    #                kernel_size=(filter_size, 1, 1),
    #                padding='same',
    #                activation='relu')(layer)
    layer = ConvLSTM2D(10, 3, padding='same', return_sequences=True)(inputs_)
    # layer = BatchNormalization()(layer)
    return layer


########################################################################################################################
def encoder(input_en, depth_):
    layer_encoder = input_en
    list_ = []
    for i in range(depth_):
        layer_encoder = group_convolution(layer_encoder, i)
        # layer_encoder = group_convolution(layer_encoder, i)
        list_.append(layer_encoder)
        layer_encoder = MaxPool3D((2, 2, 2), strides=1, padding='same')(layer_encoder)  # (2, 2)
        layer_encoder = BatchNormalization()(layer_encoder)
        layer_encoder = keras.layers.Dropout(dropout_rate)(layer_encoder)
    return layer_encoder, list_


def decoder(input_de, depth_, list_):
    layer_decoder = input_de
    for i in reversed(range(depth_)):
        layer_decoder = keras.layers.UpSampling3D((1, 1, 1))(layer_decoder)
        layer_decoder = concatenate([layer_decoder, list_[i]])
        layer_decoder = keras.layers.Dropout(dropout_rate)(layer_decoder)
        layer_decoder = group_convolution(layer_decoder, i)
        layer_decoder = group_convolution(layer_decoder, i)
    return layer_decoder


########################################################################################################################

def AE_conv3d_lstm_model():
    inputs = Input(shape=(None, None, None, 1))
    down_layer_1, list__ = encoder(inputs, depth_=depth)
    # bottleneck = group_convolution(down_layer_1, depth)
    # up_layer_1 = decoder(bottleneck, depth_=depth, list_=list__)
    output = ConvLSTM2D(10,
                        (filter_size, filter_size),
                        padding='same',
                        return_sequences=False,
                        )(down_layer_1)
    output = Conv2D(1,
                    (1, 1),
                    padding='same',
                    activation='sigmoid')(output)
    model = Model(inputs=inputs, outputs=output)
    model.summary()

    return model


########################################################################################################################

if __name__ == '__main__':
    Train_x, Train_label = load_dataset()
    # with strategy.scope():
    model_ = AE_conv3d_lstm_model()
    model_.compile(optimizer=Adam(learning_rate=0.0004),
                   loss='binary_crossentropy',  # 'categorical_crossentropy',
                   metrics=[get_jaccard_index])  # get_jaccard_index
    # class MonitoringCallback(Callback):
    #     def on_epoch_end(self, epoch, logs={}):
    #         for metric_name, metric_value in logs.items():
    #             neptune.log_metric(metric_name, metric_value)
    callbacks = tf.keras.callbacks.EarlyStopping(
        monitor='val_get_jaccard_index',
        min_delta=0.001,
        patience=5,
        verbose=0,
        mode='auto',
        restore_best_weights=True)
    history = model_.fit(Train_x, Train_label,
                         batch_size=1,
                         epochs=epochs,
                         validation_split=0.15)
    # callbacks=[callbacks])  # MonitoringCallback(),
    # Finally, save the model
    os.chdir('/home/aijjeh/Desktop/Phd_Project/Upscaling_downscaling_denoising/h5_models')
    model_.save('AE_time_distributed_filters_%d_depth_%d_kernel_5_50kHz_softmax.h5' % (filters, depth))
