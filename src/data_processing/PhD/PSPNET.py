
import numpy as np
from keras.layers import Conv2D, AveragePooling2D, UpSampling2D, Input, GlobalAveragePooling2D, MaxPooling2D, \
    BatchNormalization, Activation, Dropout
from keras.models import Model
from keras.activations import relu, sigmoid
import keras
import gc
import tensorflow as tf
from keras import backend as K
from keras.layers import Reshape

gc.collect()

########################################################################################################################
############################################ Hyper parameters ##########################################################
########################################################################################################################
lr = .0001
rho = 0.995
filters = 32
filterSize = (3, 3)
activation = relu, sigmoid
batch_size = 4
dropout = 0.2
epochs = 100
validation_split = 0.2
dilation_rate = (4, 4)
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
#x, y = shuffle(x, y)
########################################################################################################################
################# Split dataset into training and testing sets and again re-shuffle them ###############################
########################################################################################################################
x_train = x[:1520]
y_train = y[:1520]

test_x_samples = x[1520:1900]
test_y_samples = y[1520:1900]
########################################################################################################################
########################################## Custom loss functions #######################################################
########################################################################################################################
def custom_loss(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(K.abs(y_true_f * y_pred_f))
    return -(2. * intersection + smooth) / (K.sum(K.abs(y_true_f)) + K.sum(K.abs(y_pred_f)) - intersection + smooth)
########################################################################################################################
########################################## Custom metric fuction IoU ###################################################
########################################################################################################################
def iou_metric(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(K.abs(y_true_f * y_pred_f))
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou
########################################################################################################################
##################################################### Model ############################################################
########################################################################################################################
input_x = Input(shape = (512, 512, 1))
########################################################################################################################
Conv1  = Conv2D(filters,(3,3),  padding='same')(input_x)
Conv1 = BatchNormalization()(Conv1)
Conv1 = Activation('relu')(Conv1)

Conv1  = Conv2D(filters,(3,3),  padding='same')(Conv1)
Conv1 = BatchNormalization()(Conv1)
Conv1 = Activation('relu')(Conv1)
########################################################################################################################
Conv1 = MaxPooling2D((2,2),padding='same')(Conv1)
Conv2 = Dropout(0.2)(Conv1)
########################################################################################################################
Conv2 = Conv2D(filters, (3, 3), dilation_rate = (4,4) , padding='same')(Conv1)
Conv2 = BatchNormalization()(Conv2)
Conv2 = Activation('relu')(Conv2)

Conv2 = Conv2D(filters, (3, 3), dilation_rate= (4,4) , padding='same')(Conv2)
Conv2 = BatchNormalization()(Conv2)
Conv2 = Activation('relu')(Conv2)
########################################################################################################################
Conv2 = MaxPooling2D((2, 2), padding='same')(Conv2)
Conv2 = Dropout(0.2)(Conv2)
########################################################################################################################
############################################### Global average Pooling #################################################
########################################################################################################################
global_averge = BatchNormalization()(Conv2)
global_averge = GlobalAveragePooling2D()(Conv2)
global_averge = Reshape((1,1,filters))(global_averge)
global_averge = Conv2D(filters,(1,1), padding='same')(global_averge) #,  activation='relu'
global_averge = Activation('relu')(global_averge)
global_averge = UpSampling2D((512,512), interpolation='bilinear')(global_averge)
########################################################################################################################
########################################### function for Layer creation ################################################
########################################################################################################################
def layer(input,i):
    layer = BatchNormalization()(input)
    layer = MaxPooling2D(pool_size = (i,i),padding='same')(layer)
    layer = Activation('relu')(layer)
    layer = Conv2D(filters,(1,1), padding='same')(layer)
    layer = UpSampling2D((i,i), interpolation='bilinear')(layer)
    print(layer.shape)
    return layer
########################################################################################################################
previous_layer = Conv2
blue = layer(Conv2,2)
green = layer(Conv2,4)
orange = layer(Conv2,8)

#for j in range(1,4):
#    i= 2**j
#    print(i)
#    new_layer =layer(Conv,i)
#    new_layer= keras.layers.Concatenate()([previous_layer, new_layer])
#    previous_layer= new_layer
#gc.collect()
Conv2 = UpSampling2D((4,4))(Conv2)
blue = UpSampling2D((4,4))(blue)
green = UpSampling2D((4,4))(green)
orange = UpSampling2D((4,4))(orange)

new_layer = keras.layers.Concatenate()([Conv2,global_averge,blue,green,orange])

output = Conv2D(filters,(3,3), padding='same')(new_layer)
output = BatchNormalization()(output)
output = Activation('relu')(output)

output = Conv2D(filters,(3,3), padding='same')(output)
output = BatchNormalization()(output)
output = Activation('relu')(output)
########################################################################################################################
output = keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(output)
########################################################################################################################
model = Model(inputs=input_x, outputs=output)
########################################################################################################################
model.compile(optimizer='adam', loss=keras.losses.binary_crossentropy, metrics=[iou_metric])
model.fit(x_train,y_train, batch_size=batch_size, epochs=epochs, validation_split=validation_split, shuffle=True)
score = model.evaluate(test_x_samples, test_y_samples, batch_size=batch_size, verbose=1)
print(score[0], score[1])
model.summary()
model.save('PsPnet_BatchNormalization_Activation.h5')
gc.collect()
