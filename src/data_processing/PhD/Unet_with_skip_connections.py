import numpy as np
import cv2
from glob import glob
import gc
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, UpSampling2D, Input, merge
from keras.models import Model
from keras.optimizers import adam, rmsprop, sgd

# Force the Garbage Collector to release unreferenced memory
gc.collect ()


x = np.load ('E:/src/Training samples.npy')
x = x.reshape (475,512,512,1)
y = np.load ('E:/src/Ground Truth.npy')
y = y.reshape (475,512,512,1)

x_train = x[:379]
#val_x_train = x[348:379]
test_x_samples = x[379:475]

y_train = y[0:379]
#val_y_train = y[348:379]
tests_y_samples = y[379:475]

inputs = Input(shape = (512, 512, 1))

conv1 = Conv2D (filters = 16, kernel_size = (3,3), strides = 1, padding = 'same', activation = 'relu')(inputs)
conv2 = Conv2D (filters = 16, kernel_size = (3,3), strides = 1, padding = 'same', activation = 'relu')(conv1)

added1 = keras.layers.Add()([conv1,conv2])

p1 = (MaxPool2D ((2,2), strides = (2,2)))(added1)
do1 = keras.layers.Dropout(0.5)(p1)
conv4 = (Conv2D (filters = 16, kernel_size = (3,3), strides = 1, padding = 'same', activation = 'relu'))(do1)
conv5 = (Conv2D (filters = 16 , kernel_size = (3,3), strides = 1, padding = 'same', activation = 'relu'))(conv4)

added2 = keras.layers.Add()([conv4,conv5])

up1 = UpSampling2D ((2,2))(conv5)
conv7 = (Conv2D (filters = 16, kernel_size = (3,3), strides = 1, padding = 'same', activation = 'relu'))(up1)
conv8 = (Conv2D (filters = 16, kernel_size = (3,3), strides = 1, padding = 'same', activation ='relu'))(conv7)

output = (Conv2D (1, (1, 1), padding ='same', activation='sigmoid'))(conv8)

model = Model(inputs = inputs, outputs = output)

model.compile (optimizer ='adam', loss ='binary_crossentropy', metrics = ['acc'])
model.fit (np.array (x_train), np.array (y_train), batch_size = 16, epochs = 8, validation_split=0.2)
model.summary ()
model.save('Unet_skip.h5')