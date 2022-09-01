import matplotlib.pyplot as plt
import os
import cv2
import numpy as np

for i in range(1, 476):
    os.chdir('/aijjeh_odroid_sensors/aidd/data/raw/num/dataset2_labels_out/')
    data = cv2.imread('m1_rand_single_delam_%d.png' % i, 0)
    img = cv2.resize(data, (512, 512), interpolation=cv2.INTER_CUBIC)
    os.chdir('/home/aijjeh/aijjeh_rexio_share/reports/figures/comparative_study/Thesis/New_2_March_2022/GT_labels')
    cv2.imwrite('m1_rand_single_delam_%d.png' % i, img)
