import numpy as np
import cv2
from glob import glob
import gc
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, UpSampling2D, Input, merge, Dropout
from keras.models import Model
from keras.optimizers import adam, rmsprop, sgd

# Force the Garbage Collector to release unreferenced memory
gc.collect ()


x = np.load ('E:/src/Training samples.npy')
x = x.reshape (475,512,512,1)
y = np.load ('E:/src/Ground Truth.npy')
y = y.reshape (475,512,512,1)

x_train = x[:379]
test_x_samples = x[379:475]

y_train = y[0:379]
tests_y_samples = y[379:475]

inputs = Input(shape = (512, 512, 1))

conv1 = Conv2D (filters = 16, kernel_size = (3,3), strides = 1, padding = 'same', activation = 'relu')(inputs)
conv2 = Conv2D (filters = 16, kernel_size = (3,3), strides = 1, padding = 'same', activation = 'relu')(conv1)
conv3 = Conv2D (filters = 16, kernel_size = (3,3), strides = 1, padding = 'same', activation = 'relu')(conv2)

#added1 = keras.layers.Add()([conv1,conv3]) # skip connection, adding conv1  and conv2 --> input for max pooling
added1 = keras.layers.Add()([conv1,conv3])

p1 = (MaxPool2D ((2,2), strides = (2,2)))(conv3)

conv4 = Conv2D (filters = 16, kernel_size = (3,3), strides = 1, padding = 'same', activation = 'relu')(p1)
conv5 = Conv2D (filters = 16, kernel_size = (3,3), strides = 1, padding = 'same', activation = 'relu')(conv4)
conv6 = Conv2D (filters = 16, kernel_size = (3,3), strides = 1, padding = 'same', activation = 'relu')(conv5)

added2 = keras.layers.Add()([conv4,conv6])

p2 = (MaxPool2D ((2,2), strides = (2,2)))(conv6)
conv7 = Conv2D (filters = 16, kernel_size = (3,3), strides = 1, padding = 'same', activation = 'relu')(p2)
conv8 = Conv2D (filters = 16, kernel_size = (3,3), strides = 1, padding = 'same', activation = 'relu')(conv7)
conv9 = Conv2D (filters = 16, kernel_size = (3,3), strides = 1, padding = 'same', activation = 'relu')(conv8)

added3 = keras.layers.Add()([conv7,conv9])
u1 = UpSampling2D((2,2))(conv9)
#added4 = keras.layers.Add()([added1,u1])
conv10 = Conv2D (filters = 16, kernel_size = (3,3), strides = 1, padding = 'same', activation = 'relu')(u1)
conv11 = Conv2D (filters = 16, kernel_size = (3,3), strides = 1, padding = 'same', activation = 'relu')(conv10)
conv12 = Conv2D (filters = 16, kernel_size = (3,3), strides = 1, padding = 'same', activation = 'relu')(conv11)

#added4 = keras.layers.Add()([conv10,conv12])
u2 = UpSampling2D((2,2))(conv12)

conv13 = Conv2D (filters = 16, kernel_size = (3,3), strides = 1, padding = 'same', activation = 'relu')(u2)
conv14 = Conv2D (filters = 16, kernel_size = (3,3), strides = 1, padding = 'same', activation = 'relu')(conv13)
conv15 = Conv2D (filters = 16, kernel_size = (3,3), strides = 1, padding = 'same', activation = 'relu')(conv14)

output = (Conv2D (1, (1, 1), padding ='same', activation='sigmoid'))(conv15)
model = Model(inputs = inputs, outputs = output)

model.compile (optimizer ='adam', loss ='binary_crossentropy', metrics = ['acc'])
model.fit (np.array (x_train), np.array (y_train), batch_size = 16, epochs = 5, validation_split=0.2)
model.summary ()
model.save('Fully_U_Net.h5')




