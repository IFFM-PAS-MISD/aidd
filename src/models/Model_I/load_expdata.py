import numpy as np
import os
import cv2

expdata_file = 'exp_test_list.txt'
#image_stem = 'frame_'
image_suffix = '_CFRP_teflon_3o_375_375p_50kHz_5HC_x12_15Vpp.png'
images_to_select = 64 #Images from each subfolder in the training data.
Hight = 500
Width = 500


def read_images(x_path, y_path):
    '''
    :param path: Path to train image folder
    :param folder_num: Number of the folder (2_output----2)
    :return: return ndarray (channels, Hight, width)
    '''
    image_array = np.zeros((0, Hight, Width), dtype='float16')
    #label_array = np.zeros((0, Hight, Width), dtype='float16')
    num_images = os.listdir(x_path)
    step = int(len(num_images)/images_to_select)
    image_counter = step
    while image_counter <= len(num_images):
        # image_path = x_path + image_stem + str(image_counter) + image_suffix
        image_path = x_path + str(image_counter) + image_suffix
        image = cv2.imread(image_path, 0)
        image = cv2.resize(image, (Hight, Width))
        image_array = np.vstack([image_array, image.reshape(1, Hight, Width)])
        image_counter = image_counter + step

    label_image = cv2.imread(y_path, 0)
    label_image = cv2.resize(label_image, (Hight, Width))

    #label_array = np.repeat(label_image[np.newaxis, :, :], image_array.shape[0], axis=0)
    return image_array, label_image.reshape(1, Hight, Width)


def get_data():
    test_data = np.zeros((0, images_to_select, Hight, Width), dtype='float16')
    test_labels = np.zeros((0, Hight, Width), dtype='float16')

    with open(expdata_file, 'r') as file:
        lines=file.readlines()
        for line in lines:
            x = line.split(' ')[0]
            y = line.split(' ')[1][:-1]
            image_data, label_data = read_images(x, y)
            test_data = np.vstack([test_data, image_data.reshape(1, images_to_select, Hight, Width)])
            test_labels = np.vstack([test_labels, label_data])

    return test_data, test_labels
