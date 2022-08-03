import csv
import gc
import os
from pathlib import Path
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.models import load_model
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from tensorflow.python.client import device_lib
from keras import Model
from mpmath import norm
from matplotlib import cm, gridspec
import math
from sklearn.utils import shuffle

########################################################################################################################
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
print(device_lib.list_local_devices())
########################################################################################################################
# garbage collector
gc.collect()

########################################################################################################################
# Loading dataset into x, y arrays and reshape them
########################################################################################################################
dataset = np.load('/home/aijjeh/aijjeh_rexio_share/PhD/NPY_arrays/delamination_dataset.npy', mmap_mode='r+')

samples = dataset[:, :, :, 0]
labels = dataset[:, :, :, 1]

Train_x, test_x_samples, Train_label, test_y_samples = train_test_split(samples, labels, test_size=0.2, shuffle=False)

test_x_samples = np.expand_dims(test_x_samples, axis=3)
test_y_samples = np.expand_dims(test_y_samples, axis=3)
print(test_x_samples.shape)
print(test_y_samples.shape)
print(np.max(test_x_samples))
print(np.max(test_y_samples))
########################################################################################################################
# Loading experimental images
########################################################################################################################

experimental = np.load('/home/aijjeh/aijjeh_rexio_share/PhD/NPY_arrays/exp_ERMS.npy')
experimental_label = np.load('/home/aijjeh/aijjeh_rexio_share/PhD/NPY_arrays/exp_label.npy')
experimental = experimental / 255.0
experimental_label = experimental_label / 255.0
experimental = experimental.reshape(23, 512, 512, 1)
experimental_label = experimental_label.reshape(23, 512, 512, 1)

########################################################################################################################
# Loading the Model
########################################################################################################################

os.chdir('/home/aijjeh/aijjeh_rexio_share/reports/figures/comparative_study/h5_models/FCN_denseNet_models/')
model_name = 'updated_fcn_densenet_kfold_trial_4.h5'
model = load_model(model_name, compile=False)
model.summary()

########################################################################################################################
# reading cmap for plotting
########################################################################################################################

path_to_csv = '/home/aijjeh/aijjeh_rexio_share/PhD/cmap_flipped_jet256.csv'
cmap = matplotlib.colors.ListedColormap(["blue", "green", "red"], name=path_to_csv, N=None)


########################################################################################################################
# Visualizing kernels weights and intermediate outputs
########################################################################################################################

def plot_filters(layer, name_layer):
    if "conv2d_" in layer_name:
        filters = layer.get_weights()[0]
        filters = np.asarray(filters)
        filters = np.reshape(filters, (filters.shape[3], filters.shape[0], filters.shape[1], filters.shape[2]))
        print('convolution filter size', filters.shape)

        length = filters.shape[0]
        print(length)
        image = np.zeros((filters.shape[1], filters.shape[2]))
        print(image.shape)
        fig = plt.figure(figsize=(12, 12))
        gs1 = gridspec.GridSpec(math.ceil(np.sqrt(length)), math.ceil(np.sqrt(length)))
        gs1.update(wspace=0.0, hspace=0.02)

        for j in range(0, length):
            for depth in range(0, filters.shape[3]):
                image = image + filters[j, :, :, depth]
            # image = image / (depth + 1)
            ax = fig.add_subplot(gs1[j])
            ax.imshow(image, cmap='gray')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
            # plt.tight_layout()
            # plt.show()
        plt.savefig(name_layer + str('_kernel_weights_') + str(j + 1))
        plt.close('all')
    else:
        print("NO filters for this layer")


def intermediate_outputs(name, n):
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer(name).output)
    intermediate_output = intermediate_layer_model.predict(test_x_samples[n:n + 1])
    print('layer shape', intermediate_output.shape)
    length_ = intermediate_output.shape[3]
    ################################################################################################################
    plt.figure(figsize=(5 / 2.54, 5 / 2.54), dpi=600)
    plt.gca().set_axis_off()
    plt.axis('off')
    plt.subplots_adjust(left=0.0, bottom=0.0, right=1, top=1.0, wspace=0.0, hspace=0.0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    ################################################################################################################
    # print(intermediate_output[0, :, :, i])
    fig = plt.figure(figsize=(10, 10))
    for j in range(1, (length_ + 1)):
        ax = fig.add_subplot(math.ceil(np.sqrt(length_)), math.ceil(np.sqrt(length_)), j)
        ax.imshow(intermediate_output[0, :, :, (j - 1)], cmap='gray')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(name)
    plt.close('all')


def get_intermediates(img_num):
    os.chdir('/home/aijjeh/aijjeh_rexio_share/reports/figures/comparative_study/intermediate_outputs/')
    Path(str(input_image)).mkdir(parents=True, exist_ok=True)
    os.chdir(
        '/home/aijjeh/aijjeh_rexio_share/reports/figures/comparative_study/intermediate_outputs/' + str(input_image))

    for layers_count in range(len(model.layers)):
        print(layers_count, '/', len(model.layers))
        layer_name = model.layers[layers_count].get_config()
        layer_name = layer_name['name']
        print(layer_name)
        if 'global_average_pooling2d_' not in layer_name:
            plot_filters(model.layers[layers_count], layer_name)
            intermediate_outputs(layer_name, ((input_image - 381) * 4))


input_image = 400  # image number (381,475)
# get_intermediates(input_image)


########################################################################################################################
# Calculates the Intersection over Union metric for the predicted and label images using bitwise OR & AND
########################################################################################################################

def IoU(predicted_image, truth_img):
    InterSectionArray = cv2.bitwise_and(predicted_image.astype(np.float32), truth_img.astype(np.float32))
    UnionArray = cv2.bitwise_or(predicted_image.astype(np.float32), truth_img.astype(np.float32))
    I1 = np.count_nonzero(InterSectionArray)
    U = np.count_nonzero(UnionArray)
    IoU1 = I1 / U
    return IoU1


########################################################################################################################
# Save the IoU values to a csv file
########################################################################################################################
csv_files = '/home/aijjeh/aijjeh_rexio_share/reports/figures/comparative_study/RMS_image_Segmentation_Thesis/New_2_March_2022/csv_files/'

unet_num_csv = 'FCN_denseNet_kfold_num_softmax_iou.csv'
unet_exp_csv = 'FCN_denseNet_kfold_exp_softmax_iou.csv'


#######################################################################################################################

def append_list_as_row(cvs_file_name, list_of_iou, image_num):
    os.chdir(csv_files)
    with open(cvs_file_name, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerows([image_num])
        writer.writerows([list_of_iou])
        gc.collect()


########################################################################################################################
# Plotting the prediction for test images
########################################################################################################################

def Testing(csv_file_num, img):
    prediction = model.predict(test_x_samples, batch_size=1)
    for i in range(0, 380, 4):
        damage = (prediction[i])
        damage = np.argmax(damage, axis=2)
        damage = damage.astype('float32')
        original = np.squeeze(test_x_samples[i], axis=2)
        label = np.squeeze(test_y_samples[i], axis=2)
        ################################################################################################################
        plt.figure(figsize=(1, 1), dpi=512)
        plt.gca().set_axis_off()
        plt.axis('off')
        plt.subplots_adjust(left=0.0, bottom=0.0, right=1, top=1.0, wspace=0.0, hspace=0.0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        ################################################################################################################
        plt.imshow(damage, cmap=cmap)
        plt.axis('off')
        plt.savefig(img + str(i + 1))
        I_o_U = IoU(damage, label)
        image_number.append(i + 1)
        IoU_list.append(I_o_U)
        plt.close('all')
        print(I_o_U, ' ', (i + 1))
    # append_list_as_row(csv_file_num, IoU_list, image_number)


########################################################################################################################

image_number_exp = []  # holds the image number in the loop
IoU_list_exp = []  # hold the IoU values for certain threshold
image_number = []  # holds the image number in the loop
IoU_list = []  # hold the IoU values for certain threshold


########################################################################################################################
# Plotting the prediction for experimental images
########################################################################################################################

def exp(csv_file_exp):
    prediction = model.predict(experimental, batch_size=1)
    prediction = np.asarray(prediction)
    for i in range(23):
        damage = prediction[i]
        damage = np.argmax(damage, axis=2)
        damage = damage.astype('float32')
        label = np.squeeze(experimental_label[i], axis=2)
        ################################################################################################################
        plt.figure(figsize=(1, 1), dpi=512)
        plt.gca().set_axis_off()
        plt.axis('off')
        plt.subplots_adjust(left=0.0, bottom=0.0, right=1, top=1.0, wspace=0.0, hspace=0.0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        ################################################################################################################
        I_o_U = IoU(damage, label)
        plt.imshow(damage, cmap=cmap)
        plt.savefig('Fig_%d.png' % (i + 1))
        plt.close('all')
        print(I_o_U, ' ', (i + 1))
        image_number_exp.append(i + 1)
        IoU_list_exp.append(I_o_U)
    append_list_as_row(csv_file_exp, IoU_list_exp, image_number_exp)
    gc.collect()


########################################################################################################################
# paths to folders to save the output NUM images
########################################################################################################################
Thesis_new_path = '/home/aijjeh/aijjeh_rexio_share/reports/figures/comparative_study/RMS_image_Segmentation_Thesis/New_2_March_2022/'
os.chdir(Thesis_new_path)
path_FCN_DenseNet_Num_figs_softmax = 'FCN_DenseNet/num'

########################################################################################################################
# paths to folders to save the output Exp images
########################################################################################################################
path_FCN_DenseNet_Exp_figs_softmax = 'FCN_DenseNet/exp'

########################################################################################################################
# Running functions
########################################################################################################################
"""
1 - Select the num and exp paths for plotting images
2 - Select the csv files for num and exp results
3 - Select the image names to be saved
"""

img_numerical = 'FCN_DenseNet_'

if __name__ == '__main__':
    os.chdir(Thesis_new_path + path_FCN_DenseNet_Num_figs_softmax)
    Testing(unet_num_csv, img_numerical)

    # os.chdir(Thesis_new_path + path_FCN_DenseNet_Exp_figs_softmax)
    # exp(unet_exp_csv)
