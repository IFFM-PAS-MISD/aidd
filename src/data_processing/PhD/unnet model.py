from sklearn.utils import shuffle

import cv2
import numpy as np
from keras.models import load_model
import gc
import matplotlib.pyplot as plt
from PIL import Image

# garbage collector
gc.collect()
#####################################
# Loading dataset into x, y arrays and reshape them
#####################################
x = np.load('E:/src/datasets/Segmentation datasets/augemented/Augmented_data_segmentation.npy')
x = x.reshape(1900, 512, 512, 1)
y = np.load('E:/src/datasets/Segmentation datasets/augemented/Augmented_target_segmentation.npy')
y = y.reshape(1900, 512, 512, 1)
#####################################
# SHuffle the data set at random
#####################################
x,y = shuffle(x,y)
#####################################
# Split dataset inti training and testing sets and again re-shuffle them
#####################################
x_train = x[:1520]
y_train = y[0:1520]
x_train,y_train = shuffle(x_train,y_train)

test_x_samples = x[1520:1900]
tests_y_samples = y[1520:1900]
test_x_samples,tests_y_samples = shuffle(test_x_samples,tests_y_samples)
#####################################
# Loading the model
#####################################
model = load_model('E:/aidd/src/models/Nested_UNet_augmneted_data.h5')
model.summary()
#####################################
# Evaluating the model using test set
#####################################
score = model.evaluate(test_x_samples, tests_y_samples, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
#####################################
# Predicting the output of an image
#####################################
m_IoU = 0
for i in range(380):
    prediction = model.predict(test_x_samples, batch_size=1)
    prediction = np.asarray(prediction)
    cv2.imshow('Detected  damage', prediction[i])
    cv2.imshow('GT', tests_y_samples[i])
    cv2.imshow('original image', test_x_samples[i])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#####################################