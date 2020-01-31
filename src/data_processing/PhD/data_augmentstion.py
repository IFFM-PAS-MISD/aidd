import random
from random import shuffle
import pandas as pd
import numpy as np
import cv2
from glob import glob
import gc
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, UpSampling2D, Input, merge
from keras.models import Model
from keras.optimizers import adam, rmsprop, sgd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split




# Force the Garbage Collector to release unreferenced memory
gc.collect()
#####################################
# mirroring of quarters of images
#####################################
def Data_Augmentation(path):
    img_names = glob(path)
    Augmented = []
    # resizing
    for fn in img_names:
        print(fn)
        images = cv2.imread(fn, 0)
        images = cv2.resize(images, (512, 512))
        #horizontal_img = cv2.flip(images, 0)
        #vertical_img = cv2.flip(images, 1)
        #diagonal_img = cv2.flip(horizontal_img,1)
        Augmented.append(images)
        #Augmented.append(horizontal_img)
        #Augmented.append(vertical_img)
        #Augmented.append(diagonal_img)
        loaded_data = np.asarray(Augmented)
    return loaded_data


path1 = 'E:/Project_DataSet/PhD_PROJECT_DATA/data/raw/exp/*.png'
#path2 = 'E:/Project_DataSet/PhD_PROJECT_DATA/data/raw/images/dataset1_labels_out/m_*_delam*_position_no_*_a_*mm_b_*mm_angle_*.png'
x = Data_Augmentation(path1)

print(x.shape)
#print(y.shape)

#np.save('Augmented_data_segmentation',x)
#np.save('Augmented_target_segmentation',y)
np.save("test_images",x)

x = np.load('E:/src/datasets/Segmentation datasets/augemented/Augmented_data_segmentation.npy')
x = x.reshape(46, 512, 512, 1)
#y = np.load('E:/src/datasets/Segmentation datasets/augemented/Augmented_target_segmentation.npy')
#y = y.reshape(1900, 512, 512, 1)
#image, target = shuffle(x, y)


#for i in range(1900):
#    cv2.imshow('image', image[i])
#    cv2.imshow('target', target[i])
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()