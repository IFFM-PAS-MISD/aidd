from sklearn.model_selection import train_test_split
import numpy as np
import os

image_dir = "exp1/dataset2/"
labels_dir = "exp1/Labels_dataset2/"
image_stem = 'CFRP_teflon_3o_375_375p_50kHz_5HC_x12_15Vpp/'
label_stem = 'label_CFRP_teflon_3o_375_375p_50kHz_5HC_x12_15Vpp'

sample_list = os.listdir(image_dir)
data_list = []
label_list = []

sample_data =  image_dir + image_stem
sample_lable = labels_dir + label_stem + '.png'
data_list.append(sample_data)
label_list.append(sample_lable)

test_list = open('exp_test_list.txt', 'w')

for ts in range(len(data_list)):
    test_list.write(data_list[ts] + ' ' + label_list[ts]+ '\n')

test_list.close()
