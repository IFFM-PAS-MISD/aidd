import gc
import keras
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Conv2D, MaxPool2D, Input, Dropout, BatchNormalization, Activation
from keras.models import Model

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
filters = 32
filter_size = (3, 3)
dropout = 0.2
########################################################################################################################
############################## Loading dataset into x, y arrays and reshape them #######################################
########################################################################################################################
x = np.load('E:/backup/datasets/Segmentation datasets/augemented/June_training_dataset.npy')
x = x.reshape(1900, 512, 512, 1)
x = x / 255.0  # normalizing x,y to (0-1) range
y = np.load('E:/backup/datasets/Segmentation datasets/augemented/June_labels.npy')
y = y.reshape(1900, 512, 512, 1)
y = y / 255.0

########################################################################################################################
###################################### Shuffle the data set at random ##################################################
########################################################################################################################
# x, y = shuffle(x, y)
########################################################################################################################
################# Split dataset into training and testing sets and again re-shuffle them ###############################
########################################################################################################################
x_train = x[:1520]
y_train = y[:1520]

test_x_samples = x[1520:1900]
test_y_samples = y[1520:1900]

########################################################################################################################
######################################   VGG16 encoder decoder Model   #################################################
########################################################################################################################
inputs = Input(shape=(512, 512, 1))
########################################################################################################################
############################## Backbone Down sampling convolution followed by max-pooling ##############################
########################################################################################################################
conv_1 = Conv2D(filters, filter_size, padding="same")(inputs)
conv_1 = BatchNormalization()(conv_1)
conv_1 = Activation("relu")(conv_1)

conv_2 = Conv2D(filters, filter_size, padding="same")(conv_1)
conv_2 = BatchNormalization()(conv_2)
conv_2 = Activation("relu")(conv_2)
########################################################################################################################
d1 = MaxPool2D((2, 2), (2, 2))(conv_2)
d1 = Dropout(dropout)(d1)
########################################################################################################################
conv_3 = Conv2D(filters, filter_size, padding="same")(d1)
conv_3 = BatchNormalization()(conv_3)
conv_3 = Activation("relu")(conv_3)

conv_4 = Conv2D(filters, filter_size, padding="same")(conv_3)
conv_4 = BatchNormalization()(conv_4)
conv_4 = Activation("relu")(conv_4)
########################################################################################################################
d2 = MaxPool2D((2, 2), (2, 2))(conv_4)
d2 = Dropout(dropout)(d2)
########################################################################################################################
conv_5 = Conv2D(filters, filter_size, padding="same")(d2)
conv_5 = BatchNormalization()(conv_5)
conv_5 = Activation("relu")(conv_5)

conv_6 = Conv2D(filters, filter_size, padding="same")(conv_5)
conv_6 = BatchNormalization()(conv_6)
conv_6 = Activation("relu")(conv_6)

conv_7 = Conv2D(filters, filter_size, padding="same")(conv_6)
conv_7 = BatchNormalization()(conv_7)
conv_7 = Activation("relu")(conv_7)
########################################################################################################################
d3 = MaxPool2D((2, 2), (2, 2))(conv_7)
d3 = Dropout(dropout)(d3)
########################################################################################################################
conv_8 = Conv2D(filters, filter_size, padding="same")(d3)
conv_8 = BatchNormalization()(conv_8)
conv_8 = Activation("relu")(conv_8)

conv_9 = Conv2D(filters, filter_size, padding="same")(conv_8)
conv_9 = BatchNormalization()(conv_9)
conv_9 = Activation("relu")(conv_9)

conv_10 = Conv2D(filters, filter_size, padding="same")(conv_9)
conv_10 = BatchNormalization()(conv_10)
conv_10 = Activation("relu")(conv_10)
########################################################################################################################
d4 = MaxPool2D((2, 2), (2, 2))(conv_10)
d4 = Dropout(dropout)(d4)
########################################################################################################################
conv_11 = Conv2D(filters, filter_size, padding="same")(d4)
conv_11 = BatchNormalization()(conv_11)
conv_11 = Activation("relu")(conv_11)

conv_12 = Conv2D(filters, filter_size, padding="same")(conv_11)
conv_12 = BatchNormalization()(conv_12)
conv_12 = Activation("relu")(conv_12)

conv_13 = Conv2D(filters, filter_size, padding="same")(conv_12)
conv_13 = BatchNormalization()(conv_13)
conv_13 = Activation("relu")(conv_13)
########################################################################################################################
d5 = MaxPool2D((2, 2), (2, 2))(conv_13)
d5 = Dropout(dropout)(d5)
########################################################################################################################
################################## Up sampling convolution followed by up-sampling #####################################
########################################################################################################################
u1 = keras.layers.UpSampling2D((2, 2))(d5)  # ,interpolation='bilinear'
########################################################################################################################
skip5 = keras.layers.Concatenate()([conv_13, u1])

conv_14 = Conv2D(filters, filter_size, padding="same")(skip5)
conv_14 = BatchNormalization()(conv_14)
conv_14 = Activation("relu")(conv_14)

conv_15 = Conv2D(filters, filter_size, padding="same")(conv_14)
conv_15 = BatchNormalization()(conv_15)
conv_15 = Activation("relu")(conv_15)

conv_16 = Conv2D(filters, filter_size, padding="same")(conv_15)
conv_16 = BatchNormalization()(conv_16)
conv_16 = Activation("relu")(conv_16)
########################################################################################################################
u2 = keras.layers.UpSampling2D((2, 2))(conv_16)
########################################################################################################################
skip4 = keras.layers.Concatenate()([conv_10, u2])

conv_17 = Conv2D(filters, filter_size, padding="same")(skip4)
conv_17 = BatchNormalization()(conv_17)
conv_17 = Activation("relu")(conv_17)

conv_18 = Conv2D(filters, filter_size, padding="same")(conv_17)
conv_18 = BatchNormalization()(conv_18)
conv_18 = Activation("relu")(conv_18)

conv_19 = Conv2D(filters, filter_size, padding="same")(conv_18)
conv_19 = BatchNormalization()(conv_19)
conv_19 = Activation("relu")(conv_19)
########################################################################################################################
u3 = keras.layers.UpSampling2D((2, 2))(conv_19)
########################################################################################################################
skip3 = keras.layers.Concatenate()([conv_7, u3])
conv_20 = Conv2D(filters, filter_size, padding="same")(skip3)
conv_20 = BatchNormalization()(conv_20)
conv_20 = Activation("relu")(conv_20)

conv_21 = Conv2D(filters, filter_size, padding="same")(conv_20)
conv_21 = BatchNormalization()(conv_21)
conv_21 = Activation("relu")(conv_21)

conv_22 = Conv2D(filters, filter_size, padding="same")(conv_21)
conv_22 = BatchNormalization()(conv_22)
conv_22 = Activation("relu")(conv_22)
########################################################################################################################
u4 = keras.layers.UpSampling2D((2, 2))(conv_22)
########################################################################################################################
skip2 = keras.layers.Concatenate()([conv_4, u4])
conv_23 = Conv2D(filters, filter_size, padding="same")(skip2)
conv_23 = BatchNormalization()(conv_23)
conv_23 = Activation("relu")(conv_23)

conv_24 = Conv2D(filters, filter_size, padding="same")(conv_23)
conv_24 = BatchNormalization()(conv_24)
conv_24 = Activation("relu")(conv_24)
########################################################################################################################
u5 = keras.layers.UpSampling2D((2, 2))(conv_24)
########################################################################################################################
skip1 = keras.layers.Concatenate()([conv_2, u5])
conv_25 = Conv2D(filters, filter_size, padding="same")(skip1)
conv_25 = BatchNormalization()(conv_25)
conv_25 = Activation("relu")(conv_25)

conv_26 = Conv2D(filters, filter_size, padding="same")(conv_25)
conv_26 = BatchNormalization()(conv_26)
conv_26 = Activation("relu")(conv_26)
########################################################################################################################
################################################## Output layer ########################################################
########################################################################################################################
output = (Conv2D(1, (1, 1), padding='same', activation='sigmoid'))(conv_26)
########################################################################################################################
model = Model(inputs=inputs, outputs=output)


########################################################################################################################
# Custom loss function
def custom_loss(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(K.abs(y_true_f * y_pred_f))
    return - ((intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth))


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
# Dice loss / F1
def dice_loss(y_true, y_pred):
    numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=-1)
    denominator = tf.reduce_sum(y_true + y_pred, axis=-1)
    return 1 - (numerator + 1) / (denominator + 1)


########################################################################################################################
# Custom metric
def iou_metric_abs(y_true, y_pred, smooth=0):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(K.abs(y_true_f * y_pred_f))
    union = K.sum(K.abs(y_true_f)) + K.sum(K.abs(y_pred_f)) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou


########################################################################################################################
def recall(y_true, y_pred):
    y_true = K.ones_like(y_true)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    all_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall_acc = true_positives / (all_positives + K.epsilon())
    return recall_acc


def precision(y_true, y_pred):
    y_true = K.ones_like(y_true)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision_acc = true_positives / (predicted_positives + K.epsilon())
    return precision_acc


def f1_score(y_true, y_pred):
    precision_m = precision(y_true, y_pred)
    recall_m = recall(y_true, y_pred)
    return 2 * ((precision_m * recall_m) / (precision_m + recall_m + K.epsilon()))


########################################################################################################################
def custom_acc(y_true, y_pred):
    return 1 - dice_loss(y_true, y_pred)


########################################################################################################################
model.compile(optimizer='adam',
              loss=keras.losses.binary_crossentropy,
              metrics=[iou_metric, recall, precision, f1_score])

model.fit(np.asarray(x_train),
          np.asarray(y_train),
          batch_size=4,
          epochs=100,
          shuffle=True,
          validation_split=0.2)
########################################################################################################################
# Evaluating the model using test set
########################################################################################################################
score = model.evaluate(test_x_samples, test_y_samples, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
model.summary()
########################################################################################################################
model.save(
    'E:/backup/models/SegNet_models/SegNet_Encoder_decoder_added_skip_100_epoches_3_3_conv_vgg_16_archi_layers.h5')
gc.collect()
