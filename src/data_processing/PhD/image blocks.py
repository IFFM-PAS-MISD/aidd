import os
import cv2
import numpy as np
from glob import glob
import tensorflow as tf
import keras
import image_slicer
import cv2


# config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 5} )
# sess = tf.Session(config=config)
# keras.backend.set_session(sess)

def func(path):
    imge_mat = []
    img_names = glob(path)
    directory = r'E:\testslicer'

    for fn in img_names:
        filename = 'm_13_delam1_position_no_56_a_10mm_b_10mm_angle_0'
        print(fn)
        img = cv2.imread(fn, 1)
        # Resize the image to be 224*224 instead of 500*500
        width = 448
        height = 448
        dim = (width, height)
        img = cv2.resize(img, dim)
        img = img[0:224, 224:448]
        cv2.imshow('image',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        os.chdir(directory)
        print(os.listdir(directory))
        cv2.imwrite(filename,img)
        return img
    


# def crop_image(image):
#    slices =[]
#    for i in range(7):
#        for j in range(7):
#            cropped = image[j*32:(j+1)*32, i*32:(i+1)*32]
#            slices.append(cropped)
#    return np.asarray(slices)

path = 'E:/TestDataset/raw/num/Dataset_Project/*/m_*_delam*_position_no_*_a_10mm_b_*mm_angle_0.jpg'
testImage = func(path)
cv2.imshow('image',testImage)
cv2.waitKey(0)
cv2.destroyAllWindows()
#tiles = image_slicer.slice(testImage, 49)
#image_slicer.save_tiles(tiles, directory='E:/testslicer', prefix='slice', format='jpg')
