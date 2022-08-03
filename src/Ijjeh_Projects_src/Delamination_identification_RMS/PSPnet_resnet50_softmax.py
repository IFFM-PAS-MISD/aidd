import gc
import os
import keras
import numpy as np
import tensorflow as tf
from keras.activations import relu, sigmoid
from keras.callbacks import EarlyStopping
from keras.layers import Conv2D, UpSampling2D, Input, GlobalAveragePooling2D, MaxPooling2D, \
    BatchNormalization, Activation
from keras.layers import Reshape
from keras.models import Model
from keras.regularizers import l2
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split, KFold
from keras import backend as K
import matplotlib.pyplot as plt

gc.collect()

config = tf.compat.v1.ConfigProto(device_count={'GPU': 2, 'CPU': 32})
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)


# memory growing
# with tf.device('/gpu:0'):
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))


####################################################################################################################
# Custom metric function IoU
####################################################################################################################

def iou_metric(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(K.abs(y_true_f * y_pred_f))
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou


####################################################################################################################
# Hyper parameters
####################################################################################################################
lr = 0.0007
rho = 0.995
filters = 32
filterSize = (3, 3)
activation = relu, sigmoid
batch_size = 8
dropout = 0.2
epochs = 20
validation_split = 0.2
dilation_rate = (2, 2)


####################################################################################################################
#   ResNet50 as a Backbone
####################################################################################################################

def layer_bb(input_bb_layer, i):
    layer1 = Conv2D(i * 32, (1, 1), padding='same')(input_bb_layer)
    layer1 = Activation('relu')(layer1)
    # layer1 = BatchNormalization()(layer1)
    layer2 = Conv2D(i * 32, (3, 3), padding='same')(layer1)
    layer2 = Activation('relu')(layer2)
    # layer2 = BatchNormalization()(layer2)
    layer3 = Conv2D(i * 128, (1, 1), padding='same')(layer2)
    layer3 = Activation('relu')(layer3)
    # layer3 = BatchNormalization()(layer3)
    return keras.layers.Concatenate()([input_bb_layer, layer3])


# i: number of layers per block, n : factor multiplied with filter size, s: whether last block or not
def block(input_bb_block, i, n, s):
    layerx = input_bb_block
    if s == 0:
        for j in range(i):
            layerx = layer_bb(input_bb_block, n)
            input_bb_block = layerx
    else:
        layerx = Conv2D(64, (3, 3), dilation_rate=(2, 2), padding='same')(input_bb_block)
        layerx = Activation('relu')(layerx)
        # layerx = BatchNormalization()(layerx)
        layerx = Conv2D(64, (3, 3), dilation_rate=(4, 4), padding='same')(layerx)
        layerx = Activation('relu')(layerx)
        layerx = BatchNormalization()(layerx)
    return layerx


def PSPNet():
    status = 0
    input_x = Input(shape=(512, 512, 1))
    ################################################################################################################
    Conv1 = Conv2D(64, (7, 7), padding='same')(input_x)
    Conv1 = MaxPooling2D((2, 2), padding='same')(Conv1)
    ################################################################################################################
    B1 = block(Conv1, 3, 1, status)
    B1 = MaxPooling2D((2, 2), padding='same')(B1)
    ################################################################################################################
    B2 = block(B1, 4, 2, status)
    B2 = MaxPooling2D((2, 2), padding='same')(B2)
    ################################################################################################################
    B3 = block(B2, 6, 4, status)
    B3 = MaxPooling2D((2, 2), padding='same')(B3)
    ################################################################################################################
    status = 1
    B4 = block(B3, 3, 5, status)
    B4 = MaxPooling2D((2, 2), padding='same')(B4)

    ################################################################################################################
    # Global average Pooling
    ################################################################################################################
    global_average = GlobalAveragePooling2D()(B4)
    global_average = Reshape((1, 1, 64))(global_average)
    global_average = Conv2D(filters, (1, 1), padding='same')(global_average)  # ,  activation='relu'
    global_average = Activation('relu')(global_average)
    global_average = UpSampling2D((512, 512), interpolation='bilinear')(global_average)

    ################################################################################################################
    # function for Layer creation
    ################################################################################################################
    def layer(input_pspnet_layer, i):
        PSPNet_layer = MaxPooling2D(pool_size=(i, i), padding='same')(input_pspnet_layer)
        PSPNet_layer = Activation('relu')(PSPNet_layer)
        PSPNet_layer = Conv2D(filters, (1, 1), padding='same', kernel_regularizer=l2(0.0005))(PSPNet_layer)
        PSPNet_layer = UpSampling2D((i, i), interpolation='bilinear')(PSPNet_layer)
        return PSPNet_layer

    ################################################################################################################
    blue = layer(B4, 2)
    green = layer(B4, 4)
    orange = layer(B4, 8)

    Conv3 = UpSampling2D((32, 32), interpolation='bilinear')(B4)
    blue = UpSampling2D((32, 32), interpolation='bilinear')(blue)
    green = UpSampling2D((32, 32), interpolation='bilinear')(green)
    orange = UpSampling2D((32, 32), interpolation='bilinear')(orange)

    new_layer = keras.layers.Concatenate()([Conv3, global_average, blue, green, orange])

    output = Conv2D(filters, (3, 3), padding='same', kernel_regularizer=l2(0.0005))(new_layer)
    output = BatchNormalization()(output)
    output = Activation('relu')(output)

    output = Conv2D(filters, (3, 3), padding='same', kernel_regularizer=l2(0.0005))(output)
    output = BatchNormalization()(output)
    output = Activation('relu')(output)
    ################################################################################################################
    output = keras.layers.Conv2D(2, (1, 1), activation='softmax')(output)
    ################################################################################################################
    model = Model(inputs=input_x, outputs=output)
    ################################################################################################################
    model.compile(optimizer='adam',
                  loss=keras.losses.categorical_crossentropy,
                  metrics=[iou_metric])
    return model


####################################################################################################################
# Loading dataset into x, y arrays and reshape them
####################################################################################################################

dataset = np.load('delamination_dataset.npy')

print(dataset.dtype)

samples = dataset[:, :, :, 0]
labels = dataset[:, :, :, 1]

Train_x, Test_x, Train_label, Test_label = train_test_split(samples, labels, test_size=0.2, shuffle=False)

Test_x = np.expand_dims(Test_x, axis=3)

Test_label = np.expand_dims(Test_label, axis=3)

Test_label = to_categorical(Test_label, 2)

####################################################################################################################

n_split = 5  # Number of Folds
counter = 1

average_training_loss = []
average_val_loss = []

average_training_accuracy = []
average_val_accuracy = []

####################################################################################################################
#  KFold Cross validation
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

    # model_kfold = None
    model_kfold = PSPNet()
    ################################################################################################################

    earlystop = EarlyStopping(monitor='val_iou_metric',
                              # min_delta=1,
                              patience=10,
                              verbose=1,
                              mode="max",
                              restore_best_weights=True)
    my_callbacks = [earlystop]

    ################################################################################################################
    history = model_kfold.fit(x_train,
                              y_train,
                              epochs=epochs,
                              batch_size=batch_size,
                              validation_data=(x_val, y_val))
    # callbacks=my_callbacks)
    ################################################################################################################
    score = model_kfold.evaluate(Test_x,
                                 Test_label,
                                 batch_size=4,
                                 verbose=1)
    ################################################################################################################
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

    os.chdir('/home/aijjeh/aijjeh_rexio_share/reports/figures/comparative_study/h5_models/PSPNet_models/')

    # model_kfold.save('updated_PSPNet_kfold_softmax_BN_trial_%d.h5' % counter)
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

plt.title('PSPNet model', font)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
plt.savefig('PSPNet_kfold_loss_per_epochs_softmax')
plt.close('all')
gc.collect()

####################################################################################################################
# plt.figure(figsize=(10 / 2.54, 5 / 2.54), dpi=600)
####################################################################################################################
plt.plot(average_training_accuracy, label='training iou')
plt.plot(average_val_accuracy, label='validation iou')

plt.title('PSPNet model', font)
plt.ylabel('accuracy score')
plt.xlabel('epoch')
plt.legend()
plt.savefig('PSPNet_kfold_iou_per_epochs_softmax')
plt.close('all')
gc.collect()

# model.save('PSPNET_resenet50_1_1_ConvD_softmax.h5')
