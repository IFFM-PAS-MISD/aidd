import os
from glob import glob
import cv2
import numpy as np
import image_slicer


path = 'E:/PhD_PROJECT_DATA/data/raw/num/RMS_wavefield_dataset1_out/*/RMS_flat_shell_Vz_*_500x500top.png'

def slicer(path):
    img_names = glob(path)
    path1= 'E:/DataSet_Second_Version'
    for fn in img_names:
        print(fn)
        img = cv2.imread(fn, 1)
        width = 448
        height = 448
        dim = (width, height)
        img = cv2.resize(img, dim)
        os.chdir(path1)
        cv2.imwrite('E:/DataSet_Second_Version/*.png',img)
        #tiles = image_slicer.slice(fn , 49)
        #image_slicer.save_tiles(tiles, directory='E:/DataSet_1_2_2020/sliced')


slicer(path)

