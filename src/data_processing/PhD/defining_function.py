import cv2
from glob import glob
import numpy as np
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.utils import np_utils
import keras
import matplotlib.pyplot as plt


# In this part we import All images from all outputs_RMS_wave_dataset1_out

def func(path):
    imge_mat = []
    img_names = glob(path)
    for fn in img_names:
        img = cv2.imread(fn, 1)
        # Resize the image to be 512*512 instead of 500*500 so we can split it into four quarters
        width = 128
        height = 128
        dim = (width, height)
        img = cv2.resize(img, dim)
        # Crop the image to get the right upper corner to produce 256*256 image size
        crop_img = img[0:64, 64:128]
        #cv2.imshow("cropped", crop_img)
        #cv2.waitKey(0)
        #print(crop_img.size)
        #print(crop_img.shape)
        #cv2.imshow('image',img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        imge_mat.append(crop_img)
    return imge_mat


# Reads the training images and store then in Img_train
img_mask = 'E:/TestDataset/raw/num/train_data/*/RMS_flat_shell_Vz_*_500x500top.png'
Img_train = func(img_mask)                  
Training_IMG = np.asarray(Img_train)
Training_IMG = Training_IMG.astype('float32')
Training_IMG = Training_IMG/255 -0.5
print('Training Images: ', Training_IMG.shape)

# Reads the labels images for train images and store then in Train_label
###img_mask = 'E:/TestDataset/raw/num/Dataset_Project/train_labels/*.jpg'
###Train_label = func(img_mask)
###Label_Train = np.asarray(Train_label)
# Labels are put from 1-473 as the number of the images and then One_Hot_key is applied
Label_Train = np.arange(0,473,1)
print('Label images :', Label_Train.shape)

# Reads test data and store them in Img_test
img_mask = 'E:/TestDataset/raw/num/Dataset_Project/test_data/*/RMS_flat_shell_velocities_in_plane_*_500x500top.png'
Img_test = func(img_mask)
Test_IMG = np.asarray(Img_test)
Test_IMG = Test_IMG.astype('float32')
Test_IMG = Test_IMG /255 -0.5
print("Test Image : ", Test_IMG.shape)

# Reads the labels images for test images and store then in Test_label
##img_mask = 'E:/TestDataset/raw/num/Dataset_Project/test_labels/*.jpg'
##Test_label = func(img_mask)
##Label_Test = np.asarray(Test_label)
# Labels are put from 1-473 as the number of the images and then One_Hot_key is applied
Label_Test = np.arange(0,49,1)
print("Label test", Label_Test.shape)

#cv2.imshow('image', Img_train[100])
#cv2.imshow('image', Train_label[100])
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#print('training Image :', Img_train[100].shape)

# Starting the Model
#  We need to ‘one-hot-encode’ our target variable.
#  This means that a column will be created for each
#  output category and a binary variable is inputted for each category.

Label_Train = np_utils.to_categorical(Label_Train,473)
print(Label_Train.shape)
Label_Test = np_utils.to_categorical(Label_Test,473)
print(Label_Test.shape)


# Create the model

model = Sequential()

#add model layers

model.add(Conv2D(64, kernel_size=3, activation='relu',input_shape=(64,64,3),padding='same'))
model.add(Conv2D(32, kernel_size=3, activation='relu',padding='same'))
model.add(Conv2D(32, kernel_size=3, activation='relu',padding='same'))
model.add(Conv2D(32, kernel_size=3, activation='relu',padding='same'))
model.add(Conv2D(32, kernel_size=3, activation='relu',padding='same'))
model.add(Conv2D(32, kernel_size=3, activation='relu',padding='same'))
model.add(Conv2D(32, kernel_size=3, activation='relu',padding='same'))
model.add(Flatten())
model.add(Dense(1000, activation= 'relu'))
model.add(Dense(1000, activation= 'relu'))
model.add(Dense(1000, activation= 'relu'))
model.add(Dense(1000, activation= 'relu'))
model.add(Dense(1000, activation= 'relu'))
model.add(Dense(1000, activation= 'relu'))
model.add(Dense(1000, activation= 'relu'))
model.add(Dense(1000, activation= 'relu'))
model.add(Dense(1000, activation= 'relu'))
model.add(Dense(473,activation='softmax'))

# Compile model using accuracy to measure model performance
model.compile(
    optimizer='adam',
    loss = 'binary_crossentropy',
    metrics=['accuracy'])

# Training the model
model.fit(Training_IMG,Label_Train, validation_data=(Test_IMG,Label_Test), epochs= 30)

model.summary()

# evaluate the model and print the results
score = model.evaluate(Test_IMG, Label_Test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])




