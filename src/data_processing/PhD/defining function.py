import cv2
from glob import glob
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K


# In this part we import All images from all outputs_RMS_wave_dataset1_out

def func(path):
    imge_mat = []
    img_names = glob(path)
    for fn in img_names:
        img = cv2.imread(fn, 1)
        # Resize the image to be 512*512 instead of 500*500 so we can split it into four quarters
        width = 512
        height = 512
        dim = (width, height)
        img = cv2.resize(img, dim)
        # Crop the image to get the right upper corner
        crop_img = img[0:255, 256:511]
        # cv2.imshow("cropped", crop_img)
        # cv2.waitKey(0)
        # print(crop_img.size)
        # print(crop_img.shape)
        # cv2.imshow('image',img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        imge_mat.append(crop_img)
    return imge_mat


# Reads the training images and store then in Img_train
img_mask = " Dataset_Project/train_data/*_output/*Vz_*_500x500top.png"
Img_train = func(img_mask)
Training_IMG = np.asarray(Img_train)
print('Training Images: ', Training_IMG.shape)

# Reads the labels images for train images and store then in Train_label
img_mask = "Dataset_Project/train_data/*.png"
Train_label = func(img_mask)
Label_Train = np.asarray(Train_label)
print('Label images :', Label_Train.shape)

# Reads test data and store them in Img_test
img_mask = 'Dataset_Project/test_data/*/*Vz_*_500x500top.png'
Img_test = func(img_mask)
Test_IMG = np.asarray(Img_test)
print("Test Image : ", Test_IMG.shape)                      

# Reads the labels images for test images and store then in Test_label
img_mask = 'Dataset_Project/test_labels/*.png'
Test_label = func(img_mask)
Label_Test = np.asarray(Test_label)
print("Label test", Label_Test.shape)

#cv2.imshow('image', Img_train[100])
#cv2.imshow('image', Train_label[100])
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#print('training Image :', Img_train[100].shape)
