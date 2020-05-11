import numpy as np
import gc
import keras
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, UpSampling2D, Input, merge, Activation
from keras.losses import binary_crossentropy
from keras.models import Model
from keras import metrics
import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow_core.python.ops.metrics_impl import mean_iou
import keras.backend as K

#config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 64} )
#sess = tf.Session(config=config)
#keras.backend.set_session(sess)

# Force the Garbage Collector to release unreferenced memory

gc.collect()

#####################################
x = np.load('E:/backup/datasets/Segmentation datasets/augemented/Augmneted_train_new_data_RMS_flat_shell_bottom.npy')
x = x.reshape(1900, 512, 512, 1)
y = np.load('E:/backup/datasets/Segmentation datasets/augemented/Augmented_target_segmentation.npy')
y = y.reshape(1900, 512, 512, 1)
#####################################
# Randomly shuffle the dataset
#####################################
x, y = shuffle(x,y)
#####################################
x_train = x[:1520]
y_train = y[:1520]
#####################################
test_x_samples = x[1520:]
tests_y_samples = y[1520:]
#####################################
inputs = Input(shape=(512, 512, 1))
#####################################
filters = 16

# Backbone Down sampling convolution followed by max-pooling
#####################################
c11 = Conv2D(filters=filters, kernel_size=(3, 3), strides=1, padding='same')(inputs)#, activation='relu')(inputs)
act1 = Activation('relu')(c11)
BatchNorm = keras.layers.BatchNormalization()(act1)
c12 = Conv2D(filters=filters, kernel_size=(3, 3), strides=1, padding='same')(BatchNorm) #, padding='same', activation='relu')
act1 = Activation('relu')(c12)
BatchNorm1 = keras.layers.BatchNormalization()(act1)
c13 = Conv2D(filters=filters, kernel_size=(3, 3), strides=1, padding='same')(BatchNorm)#, padding='same', activation='relu')(BatchNorm)
#####################################
d1 = MaxPool2D((2, 2), (2, 2))(BatchNorm1)

#####################################
c21 = Conv2D(filters=filters, kernel_size=(3, 3), strides=1, padding='same')(d1)#, padding='same', activation='relu')(d1)
act1 = Activation('relu')(c21)
BatchNorm = keras.layers.BatchNormalization()(act1)
c22 = Conv2D(filters=filters, kernel_size=(3, 3), strides=1, padding='same')(BatchNorm)#, padding='same', activation='relu')(c21)
act1 = Activation('relu')(c22)
BatchNorm2 = keras.layers.BatchNormalization()(act1)
c23 = Conv2D(filters=filters, kernel_size=(3, 3), strides=1, padding='same')(BatchNorm)#, padding='same', activation='relu')(c22)
#####################################
d2 = MaxPool2D((2, 2), (2, 2))(BatchNorm2)
#####################################
c31 = Conv2D(filters=filters, kernel_size=(3, 3), strides=1, padding='same')(d2)#, padding='same', activation='relu')(d2)
act1 = Activation('relu')(c31)
BatchNorm = keras.layers.BatchNormalization()(act1)
c32 = Conv2D(filters=filters, kernel_size=(3, 3), strides=1, padding='same')(BatchNorm)#, padding='same', activation='relu')(c31)
act1 = Activation('relu')(c32)
BatchNorm3 = keras.layers.BatchNormalization()(act1)
c33 = Conv2D(filters=filters, kernel_size=(3, 3), strides=1, padding='same')(BatchNorm)#, padding='same', activation='relu')(c32)

#####################################
d3 = MaxPool2D((2, 2), (2, 2))(BatchNorm3)
#####################################
c41 = Conv2D(filters=filters, kernel_size=(3, 3), strides=1, padding='same')(d3)#, strides=1, padding='same', activation='relu')(d3)
act1 = Activation('relu')(c41)
BatchNorm = keras.layers.BatchNormalization()(act1)
c42 = Conv2D(filters=filters, kernel_size=(3, 3), strides=1, padding='same')(BatchNorm)#, strides=1, padding='same', activation='relu')(c41)
act1 = Activation('relu')(c42)
BatchNorm4 = keras.layers.BatchNormalization()(act1)
c43 = Conv2D(filters=filters, kernel_size=(3, 3), strides=1, padding='same')(BatchNorm)#, strides=1, padding='same', activation='relu')(c42)

#####################################
d4 = MaxPool2D((2, 2), (2, 2))(BatchNorm4)
#####################################
c51 = Conv2D(filters=filters, kernel_size=(3, 3), strides=1, padding='same')(d4)#, strides=1, padding='same', activation='relu')(d4)
act1 = Activation('relu')(c51)
BatchNorm = keras.layers.BatchNormalization()(act1)
c52 = Conv2D(filters=filters, kernel_size=(3, 3), strides=1, padding='same')(BatchNorm)#, strides=1, padding='same', activation='relu')(c51)
act1 = Activation('relu')(c52)
BatchNorm5 = keras.layers.BatchNormalization()(act1)
c53 = Conv2D(filters=filters, kernel_size=(3, 3), strides=1, padding='same')(BatchNorm)#, strides=1, padding='same', activation='relu')(c52)

#####################################
# Up sampling convolution followed by up-sampling
#####################################
u1 = UpSampling2D((2, 2))(BatchNorm5)
#####################################
print(u1,BatchNorm4)
skip4 = keras.layers.Concatenate()([BatchNorm4,u1])
c61 = Conv2D(filters=filters, kernel_size=(3, 3), strides=1, padding='same')(skip4)#, padding='same', activation='relu')(skip4)
act1 = Activation('relu')(c61)
BatchNorm = keras.layers.BatchNormalization()(act1)
c62 = Conv2D(filters=filters, kernel_size=(3, 3), strides=1, padding='same')(BatchNorm)#, padding='same', activation='relu')(c61)
act1 = Activation('relu')(c62)
BatchNorm = keras.layers.BatchNormalization()(act1)
c63 = Conv2D(filters=filters, kernel_size=(3, 3), strides=1, padding='same')(BatchNorm)#, padding='same', activation='relu')(c62)
#####################################
u2 = UpSampling2D((2, 2))(BatchNorm)
#####################################
skip3 = keras.layers.Concatenate()([BatchNorm3,u2])
c71 = Conv2D(filters=filters, kernel_size=(3, 3), strides=1, padding='same')(skip3)#, padding='same', activation='relu')(skip3)
act1 = Activation('relu')(c71)
BatchNorm = keras.layers.BatchNormalization()(act1)
c72 = Conv2D(filters=filters, kernel_size=(3, 3), strides=1, padding='same')(BatchNorm)#, padding='same', activation='relu')(c71 )
act1 = Activation('relu')(c72)
BatchNorm = keras.layers.BatchNormalization()(act1)
c73 = Conv2D(filters=filters, kernel_size=(3, 3), strides=1, padding='same')(BatchNorm)#, padding='same', activation='relu')(c72)
#####################################
u3 = UpSampling2D((2, 2))(BatchNorm)
#####################################
skip2 = keras.layers.Concatenate()([BatchNorm2,u3])
c81 = Conv2D(filters=filters, kernel_size=(3, 3), strides=1, padding='same')(skip2)#, padding='same', activation='relu')(skip2)
act1 = Activation('relu')(c81)
BatchNorm = keras.layers.BatchNormalization()(act1)
c82 = Conv2D(filters=filters, kernel_size=(3, 3), strides=1, padding='same')(BatchNorm)#, padding='same', activation='relu')(c81)
act1 = Activation('relu')(c82)
BatchNorm = keras.layers.BatchNormalization()(act1)
c83 = Conv2D(filters=filters, kernel_size=(3, 3), strides=1, padding='same')(BatchNorm)#, padding='same', activation='relu')(c82)
#####################################
u4 = UpSampling2D((2, 2))(BatchNorm)
#####################################
skip1 = keras.layers.Concatenate()([BatchNorm1,u4])
c91 = Conv2D(filters=filters, kernel_size=(3, 3), strides=1, padding='same')(skip1)#, padding='same', activation='relu')(skip1)
act1 = Activation('relu')(c91)
BatchNorm = keras.layers.BatchNormalization()(act1)
c92 = Conv2D(filters=filters, kernel_size=(3, 3), strides=1, padding='same')(BatchNorm)#, padding='same', activation='relu')(c91)
act1 = Activation('relu')(c92)
BatchNorm = keras.layers.BatchNormalization()(act1)
c93 = Conv2D(filters=filters, kernel_size=(3, 3), strides=1, padding='same')(BatchNorm)#, padding='same', activation='relu')(c92)
#####################################
# Output layer
#####################################
output = (Conv2D(1, (1, 1), padding='same', activation='sigmoid'))(BatchNorm)
#####################################
model = Model(inputs=inputs, outputs=output)
#####################################
# Custom loss function (Dice score function)
#def custom_loss(y_true, y_pred,smooth =1):
#    y_true_f = K.flatten(y_true)
#    y_pred_f = K.flatten(y_pred)
#    intersection = K.sum(y_true_f * y_pred_f)
#    return -(2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


#########################################################

#Dice loss / F1
def dice_loss(y_true, y_pred):
  numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=-1)
  denominator = tf.reduce_sum(y_true + y_pred, axis=-1)

  return 1 - (numerator + 1) / (denominator + 1)

def dice_loss_1(y_true, y_pred): # did not give good results
  numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1,2,3))
  denominator = tf.reduce_sum(y_true + y_pred, axis=(1,2,3))

  return 1 - numerator / denominator

# Wighted cross entropy
def weighted_cross_entropy(beta):
  def convert_to_logits(y_pred):
      y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
      return tf.log(y_pred / (1 - y_pred))

  def loss(y_true, y_pred):
    y_pred = convert_to_logits(y_pred)
    loss = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, targets=y_true, pos_weight=beta)

    # or reduce_sum and/or axis=-1
    return tf.reduce_mean(loss)

  return loss

# comined loss
def loss(y_true, y_pred):
    def dice_loss(y_true, y_pred):
        numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1,2,3))
        denominator = tf.reduce_sum(y_true + y_pred, axis=(1,2,3))

        return tf.reshape(1 - numerator / denominator, (-1, 1, 1))

    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
############################################
# Custom metric IoU
def iou_loss_core(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(K.abs(y_true_f * y_pred_f))#, axis=-1
    union = K.sum(y_true_f) + K.sum(y_pred_f)# - intersection
    iou = (intersection + smooth) / ( union + smooth)
    return iou

############################################
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

############################################
#  metrics=[tf.keras.metrics.MeanIoU(num_classes=2)] this is an IOU function bulit in tensorflow

model.compile(optimizer='adam', loss=dice_loss,  metrics=[metrics.binary_accuracy,metrics.mae,metrics.categorical_accuracy])
model.fit(np.array(x_train), np.array(y_train), batch_size=16, epochs=20, validation_split=0.2)
model.summary()
model.save('UNet_augmneted_data_skips_updated_kernal_sizes_updated_New_data_with_custom_loss_fuction_adding_BatchNorm.h5')
#####################################

