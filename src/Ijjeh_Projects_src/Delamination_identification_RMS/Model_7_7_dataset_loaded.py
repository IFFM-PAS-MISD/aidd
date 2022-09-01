import io
import zipfile
from glob import glob
import cv2
import image_slicer
import numpy as np
#from keras import regularizers
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from keras.models import Sequential
from keras.optimizers import RMSprop, adam
from keras.utils import np_utils
#import matplotlib.pyplot as plt
import gc
# garbage collector
gc.collect()



Training_image = np.load('E:/DataSet_aidd/Training_Images_7_7.npy')
Training_image = Training_image.reshape(18522,32,32,1)
Testing_Images = np.load('E:/DataSet_aidd/Testing_Images_7_7.npy')
Testing_Images = Testing_Images.reshape(4753,32,32,1)
Training_Labels = np.load('E:/DataSet_aidd/Training_Labels_7_7.npy')
Testing_Labels = np.load('E:/DataSet_aidd/Testing_Labels_7_7.npy')

# using one-hot-encoding to categorize both training labels and testing labels into two categories
Training_Labels = np_utils.to_categorical(Training_Labels, 2)
Testing_Labels = np_utils.to_categorical(Testing_Labels, 2)

# Configuring the model
model = Sequential()
First_convolution_layer = model.add(Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 1)))
First_Pooling_Layer = model.add(MaxPool2D((2, 2), strides=2))
Second_convolution_layer = model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
Second_Pooling_Layer = model.add(MaxPool2D((2, 2), strides=2))
Flatting_Layer = model.add(Flatten())
First_Dropout_layer = model.add(Dropout(0.5))
First_FC_layer = model.add(Dense(4096, activation='relu'))
Second_Dropout_layer = model.add(Dropout(0.5))
Second_FC_layer = model.add(Dense(1024, activation='relu'))
Final_output = model.add(Dense(2))  # we did not use an activation function

# Compile model using accuracy to measure model performance

model.compile(
    optimizer=adam(lr=0.001),
    loss='mse',
    metrics=['accuracy'])

# Training the model
model.fit(Training_image, Training_Labels, batch_size=98, epochs=10, validation_split= 0.1)
model.summary()

# save model and architecture to single file
model.save("model_7_7.h5")
print("Saved model to disk")




