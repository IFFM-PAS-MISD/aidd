import gc
import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.activations import relu, sigmoid
from keras.callbacks import EarlyStopping
from keras.layers import Conv2D, UpSampling2D, Input, GlobalAveragePooling2D, MaxPooling2D, \
    BatchNormalization, Activation
from keras.layers import Reshape
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import to_categorical

gc.collect()

###########################################   memory growing ###########################################################
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)


########################################################################################################################

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
############################################ Hyper parameters ##########################################################
########################################################################################################################
lr = 0.0007
rho = 0.995
filters = 32
filterSize = (3, 3)
activation = relu, sigmoid
batch_size = 4
dropout = 0.0
epochs = 100
validation_split = 0.2
dilation_rate = (2, 2)
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

y_train = to_categorical(y_train)
test_y_samples = to_categorical(test_y_samples)


########################################################################################################################
######################################   ResNet50 as a Backbone   ######################################################
########################################################################################################################
def layer_bb(input_bb_layer, i):
    layer1 = Conv2D(i * 8, (1, 1), padding='same')(input_bb_layer)
    layer1 = Activation('relu')(layer1)
    layer2 = Conv2D(i * 8, (3, 3), padding='same')(layer1)
    layer2 = Activation('relu')(layer2)
    layer3 = Conv2D(i * 8, (1, 1), padding='same')(layer2)
    layer3 = Activation('relu')(layer3)
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
        layerx = Conv2D(64, (3, 3), dilation_rate=(4, 4), padding='same')(layerx)
        layerx = Activation('relu')(layerx)

    return layerx


status = 0
input_x = Input(shape=(512, 512, 1))
########################################################################################################################
Conv1 = Conv2D(16, (7, 7), padding='same')(input_x)
Conv1 = MaxPooling2D((2, 2), padding='same')(Conv1)
########################################################################################################################
B1 = block(Conv1, 3, 1, status)
B1 = MaxPooling2D((2, 2), padding='same')(B1)
########################################################################################################################
B2 = block(B1, 4, 2, status)
B2 = MaxPooling2D((2, 2), padding='same')(B2)
########################################################################################################################
B3 = block(B2, 6, 4, status)
B3 = MaxPooling2D((2, 2), padding='same')(B3)
########################################################################################################################
status = 1
B4 = block(B3, 3, 4, status)
B4 = MaxPooling2D((2, 2), padding='same')(B4)
print(B4.shape)
########################################################################################################################
########################################################################################################################
############################################### Global average Pooling #################################################
########################################################################################################################
global_averge = GlobalAveragePooling2D()(B4)
global_averge = Reshape((1, 1, 64))(global_averge)
global_averge = Conv2D(filters, (1, 1), padding='same')(global_averge)  # ,  activation='relu'
global_averge = Activation('relu')(global_averge)
global_averge = UpSampling2D((512, 512), interpolation='bilinear')(global_averge)


########################################################################################################################
########################################### function for Layer creation ################################################
########################################################################################################################
def layer(input_pspnet_layer, i):
    pspent_layer = MaxPooling2D(pool_size=(i, i), padding='same')(input_pspnet_layer)
    pspent_layer = Activation('relu')(pspent_layer)
    pspent_layer = Conv2D(filters, (1, 1), padding='same', kernel_regularizer=l2(0.0005))(pspent_layer)
    pspent_layer = UpSampling2D((i, i), interpolation='bilinear')(pspent_layer)
    print(pspent_layer.shape)
    return pspent_layer


########################################################################################################################
previous_layer = B4
blue = layer(B4, 2)
green = layer(B4, 4)
orange = layer(B4, 8)

# for j in range(1,4):
#    i= 2**j
#    print(i)
#    new_layer =layer(Conv,i)
#    new_layer= keras.layers.Concatenate()([previous_layer, new_layer])
#    previous_layer= new_layer
# gc.collect()

Conv3 = UpSampling2D((32, 32), interpolation='bilinear')(B4)
blue = UpSampling2D((32, 32), interpolation='bilinear')(blue)
green = UpSampling2D((32, 32), interpolation='bilinear')(green)
orange = UpSampling2D((32, 32), interpolation='bilinear')(orange)

new_layer = keras.layers.Concatenate()([Conv3, global_averge, blue, green, orange])

output = Conv2D(filters, (3, 3), padding='same', kernel_regularizer=l2(0.0005))(new_layer)
output = BatchNormalization()(output)
output = Activation('relu')(output)

output = Conv2D(filters, (3, 3), padding='same', kernel_regularizer=l2(0.0005))(output)
output = BatchNormalization()(output)
output = Activation('relu')(output)
########################################################################################################################
output = keras.layers.Conv2D(2, (1, 1), activation='softmax')(output)
########################################################################################################################
model = Model(inputs=input_x, outputs=output)
########################################################################################################################
############################################### adding earlystoping ####################################################
########################################################################################################################
earlystop = EarlyStopping(monitor='val_iou_metric',
                          # min_delta=1,
                          patience=10,
                          verbose=1,
                          mode="max",
                          restore_best_weights=True)
my_callbacks = [earlystop]
########################################################################################################################
model.compile(optimizer=Adam(lr=lr, ), loss=keras.losses.binary_crossentropy, metrics=[iou_metric])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_split=validation_split,
                    shuffle=True)  # , callbacks=my_callbacks)

score = model.evaluate(test_x_samples,
                       test_y_samples,
                       batch_size=batch_size,
                       verbose=1)

print(score[0], score[1])
model.summary()
model.save('PSPNET_resenet50_1_1_ConvD_softmax.h5')
gc.collect()

########################################################################################################################
############################ Plotting the model loss and acc for training and validation sets ##########################
########################################################################################################################

###############################################################################################################
plt.figure(figsize=(5 / 2.54, 5 / 2.54), dpi=600)
plt.gca().set_axis_off()
plt.axis('off')
plt.subplots_adjust(left=0.0, bottom=0.0, right=1, top=1.0, wspace=0.0, hspace=0.0)
plt.margins(0, 0)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
################################################################################################################

plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='test loss')
plt.legend()
plt.show()

plt.plot(history.history['iou_metric'], label='train iou')
plt.plot(history.history['val_iou_metric'], label='test iou')
plt.legend()
plt.show()

plt.close('all')
