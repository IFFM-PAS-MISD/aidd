import numpy as np
import keras
import cv2
import tensorflow as tf
from tensorflow.python.distribute.mirrored_strategy import MirroredStrategy

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import load_data
import networks
import os

import gc

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from tensorflow.python.client import device_lib
devices = device_lib.list_local_devices()
GPUS = ["GPU:0", "GPU:1"]
strategy: MirroredStrategy = tf.distribute.MirroredStrategy(GPUS)

#   memory growing

gc.collect()

config = tf.compat.v1.ConfigProto(device_count={'GPU': 0, 'GPU': 1, 'CPU': 64})
config.gpu_options.allow_growth = True
config.log_device_placement = True
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

q_replay_memory_size=int(1e4)

import keras.backend as K

# def iou_metric(y_pred, y_true, smooth=1):
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)
#     intersection = K.sum(K.abs(y_true_f * y_pred_f))
#     union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
#     iou = (intersection + smooth) / (union + smooth)
#     return iou

def iou_metric(y_pred, y_true):
    I = tf.reduce_sum(y_pred * y_true, axis=(1, 2))
    U = tf.reduce_sum(y_pred + y_true, axis=(1, 2)) - I
    return tf.reduce_mean(I / U)

# def iou_metric(y_pred, y_true):
#     y_pred = K.cast(K.greater(y_pred, .5), dtype='float32') # .5 is the threshold
#     inter = K.sum(K.sum(K.squeeze(y_true * y_pred, axis=3), axis=2), axis=1)
#     union = K.sum(K.sum(K.squeeze(y_true + y_pred, axis=3), axis=2), axis=1) - inter
#     return K.mean((inter + K.epsilon()) / (union + K.epsilon()))

def start_training():
    # X_train, y_train, X_test, y_test = load_data.get_data()

    X_train = np.load('train_data.npy', allow_pickle=True, mmap_mode="r")
    y_train = np.load('train_labels.npy', allow_pickle=True, mmap_mode="r")

    print(X_train.shape)

    X_train = X_train.reshape(X_train.shape[0], 64, 512, 512, 1).astype('float16')
    y_train = y_train.reshape(y_train.shape[0], 512, 512, 1).astype('float16')

    print(X_train.shape)
    print(y_train.shape)

    X_train = X_train / 255.
    y_train = y_train / 255.

    # For one-hot encoding, uncomment below line of code and then use softmax and filter = 2 in the last layer at networks.py

    # y_train = to_categorical(y_train)

    # Get model
    with strategy.scope():
        model = networks.binary_net((X_train.shape[1], X_train.shape[2], X_train.shape[3], X_train.shape[4]))
        #model = tf.keras.models.load_model('full_WF.h5', custom_objects={'iou_metric': iou_metric})
        #if the model terminates with power off or any other option it can be resumed by uncommeting this
        #line and commenting the above line. i.e. networks.binary_net()....
        print(model.summary())

        opt = tf.optimizers.Adadelta(learning_rate=1.0)  # then 0.05, 0.1, 0.5
        # opt = tf.optimizers.Adam(learning_rate=0.1)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=[iou_metric])
        #also check loss = 'binary_crossentropy' for normal and 'categorical_crossentropy' for one hot encoding.

        es = EarlyStopping(monitor='val_iou_metric', mode='max', patience=20)
        # will stop if validation IoU is not improving till 20 epoches.
        ms = ModelCheckpoint('full_WF.h5', monitor='val_iou_metric', mode='max', save_best_only=True)

    training_history = model.fit(x=X_train, y=y_train, validation_split=0.1, batch_size=2, epochs=1000, verbose=1,
                                     callbacks=[es, ms], use_multiprocessing=True, workers=6)

    # Plot train vs test accuracy per epoch
    plt.figure()

    # Use the history metrics
    plt.plot(training_history.history['iou_metric'])
    plt.plot(training_history.history['val_iou_metric'])

    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(5))

    # Make it pretty
    plt.title('IOU of Train and Validation Data')
    plt.ylabel('IOU values')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'])
    plt.show()

if __name__ == "__main__":
    start_training()
