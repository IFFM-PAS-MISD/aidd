from sklearn.model_selection import train_test_split
import numpy as np
import os

image_dir = "dataset2/wavefield_dataset2_bottom_out/"
labels_dir = "dataset2/dataset2_labels_out/"
image_stem = '_output/'
label_stem = 'm1_rand_single_delam_'

sample_list = os.listdir(image_dir)
data_list = []
label_list = []
for smp in range(len(sample_list)):
    sample_data =  image_dir + str(smp+1) + image_stem
    sample_lable = labels_dir + label_stem + str(smp+1) + '.png'
    data_list.append(sample_data)
    label_list.append(sample_lable)

x_train, x_test, y_train, y_test = train_test_split(data_list, label_list, test_size=0.2, random_state=5)

train_list = open('train_list.txt', 'w')
test_list = open('test_list.txt', 'w')

for tr in range(len(x_train)):
    train_list.write(x_train[tr] + ' ' + y_train[tr] + '\n')

for ts in range(len(x_test)):
    test_list.write(x_test[ts] + ' ' + y_test[ts]+ '\n')

train_list.close()
test_list.close()
