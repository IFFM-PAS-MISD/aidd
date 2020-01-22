import numpy as np
import cv2
from glob import glob
import gc
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, UpSampling2D

# Force the Garbage Collector to release unreferenced memory
gc.collect ()


x = np.load ('E:/src/Training samples.npy')
x = x.reshape (475,512,512,1)

x_train = x[:348]
val_x_train = x[348:379]
test_x_samples = x[379:475]

y = np.load ('E:/src/Ground Truth.npy')
y = y.reshape (475,512,512,1)

y_train = y[0:348]
val_y_train = y[348:379]
tests_y_samples = y[379:475]

model = Sequential ()
model.add (Conv2D (filters = 16, kernel_size = (3,3), input_shape = (512, 512, 1), strides = 1, padding = 'same', activation = 'relu'))
model.add (Conv2D (filters = 16, kernel_size = (3,3), strides = 1, padding = 'same', activation = 'relu'))
model.add (MaxPool2D ((2,2), strides = ( 2,2)))

model.add (Conv2D (filters = 16, kernel_size = (3,3), strides = 1, padding = 'same', activation = 'relu'))
model.add (Conv2D (filters = 16 , kernel_size = (3,3), strides = 1, padding = 'same', activation = 'relu'))

model.add (UpSampling2D ((2,2)))
model.add (Conv2D (filters = 16, kernel_size = (3,3), strides = 1, padding = 'same', activation = 'relu'))
model.add (Conv2D (filters = 16, kernel_size = (3,3), strides = 1, padding = 'same', activation ='relu'))

model.add (Conv2D (1, (1, 1), padding ='same', activation ='sigmoid'))

model.compile (optimizer ='adam', loss ='binary_crossentropy', metrics = ['acc'])
model.fit (np.array (x_train), np.array (y_train), batch_size = 16, epochs = 3, validation_split=0.1)

model.summary ()

model.save ('initialUnet.h5')