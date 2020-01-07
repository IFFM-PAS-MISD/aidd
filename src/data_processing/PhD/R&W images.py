import os
import numpy as np
import cv2
from glob import glob
import image_slicer


def slicer(path):
    image_mat = []
    img_names = glob(path)
    for fn in img_names:
        img = cv2.imread(fn, 1)
        width = 448
        height = 448
        dim = (width, height)
        img = cv2.resize(img, dim)
        tiles = image_slicer.slice(img, 49)
        image_slicer.save_tiles(tiles, directory='E:/DataSet_1_2_2020/sliced')
        image_mat.append(tiles)
    return image_mat


img_mask = 'E:/TestDataset/raw/num/train_data/*/RMS_flat_shell_Vz_*_500x500top.png'
Img_train = slicer(img_mask)
os.chdir('E:\DataSet_1_2_2020\Training Images')
for i in range(len(Img_train)):
    cv2.imwrite('RMS_flat_shell_Vz_' + str(i) + '_500x500top.png', Img_train[i])

# calling crop_image function to slice the cropped iamge into 49 equal squares if size 32*32


img_mask = 'E:/TestDataset/raw/num/Dataset_Project/test_data/*/RMS_flat_shell_velocities_in_plane_*_500x500top.png'
Img_test = slicer(img_mask)
os.chdir('E:/DataSet_1_2_2020/Test Images')
for i in range(len(Img_train)):
    cv2.imwrite('RMS_flat_shell_velocities_in_plane_' + str(i) + '_500x500top.png', Img_train[i])

img_mask = 'E:/TestDataset/raw/num/Dataset_Project/train_labels/*.jpg'
Train_label = slicer(img_mask)
os.chdir('E:\DataSet_1_2_2020\Training Label')
for i in range(len(Img_train)):
    cv2.imwrite(str(i) + '.jpg', Img_train[i])
