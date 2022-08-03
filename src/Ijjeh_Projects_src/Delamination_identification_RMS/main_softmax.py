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

# from keras import Model
# from mpmath import norm
# from matplotlib import cm, gridspec
# import math
# from sklearn.utils import shuffle

########################################################################################################################
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

########################################################################################################################
# garbage collector
gc.collect()

########################################################################################################################
# Loading dataset into x, y arrays and reshape them
########################################################################################################################
dataset = np.load('delamination_dataset.npy')

samples = dataset[:, :, :, 0]
labels = dataset[:, :, :, 1]

Train_x, test_x_samples, Train_label, test_y_samples = train_test_split(samples, labels, test_size=0.2, shuffle=False)

test_x_samples = np.expand_dims(test_x_samples, axis=3)
test_y_samples = np.expand_dims(test_y_samples, axis=3)

########################################################################################################################
# Loading experimental images
########################################################################################################################

# experimental = np.load('/home/aijjeh/Desktop/new_exp/ERMS_new_exp.npy')
experimental = np.load('WRMS_stringer_exp.npy')
# experimental_label = np.load('/home/aijjeh/Desktop/new_exp/label_new_exp.npy')
experimental_label = np.load('label_stringer_exp.npy')
experimental = experimental / 255.0
experimental_label = experimental_label / 255.0
experimental = experimental.reshape(3, 512, 512, 1)
experimental_label = experimental_label.reshape(3, 512, 512, 1)

########################################################################################################################
# Loading the model
# Models
########################################################################################################################

# FCN_softmax_100_epoches #tr = 1
# VGG16_100_epochs_softmax
# Unet_100_epoches_softmax

# PsPnet_BatchNormalization_add__to_globalaverage_Activation
# PSPNET_resenet50_1_1_ConvD_softmax

########################################################################################################################
# paths for the model source files
########################################################################################################################

# fcn_path = 'E:/backup/models/FCN_DenseNet_models/Softmax/'
# unet_path = 'E:/backup/models/UNet_models/Softmax/'
# vgg16_path = 'E:/backup/models/SegNet_models/Softmax/'
# PsPnet_BatchNormalization_add__to_globalaverage_Activation_softmax
# model_name = fcn_path + 'FCN_softmax_100_epoches.h5'  # 'PSPNET_resenet50_1_1_ConvD_softmax.h5'  #

# os.chdir('/home/aijjeh/aijjeh_rexio_share/reports/figures/comparative_study/h5_models/GCN_models/')
model_name = 'GCN_model_K_7_fold_5.h5'
# model_name = 'GCN_model_K_7_fold_5.h5'
model = load_model(model_name, compile=False)
model.summary()

########################################################################################################################
# reading cmap for plotting
########################################################################################################################

path_to_csv = '/home/aijjeh/aijjeh_rexio_share/PhD/cmap_flipped_jet256.csv'
cmap = matplotlib.colors.ListedColormap(["blue", "green", "red"], name=path_to_csv, N=None)
# m = cm.ScalarMappable(norm=norm, cmap=cmap)

########################################################################################################################
# Visualizing kernels weights and intermediate outputs
########################################################################################################################

# def plot_filters(layer, name_layer):
#     if "conv2d_" in layer_name:
#         filters = layer.get_weights()[0]
#         filters = np.asarray(filters)
#         filters = np.reshape(filters, (filters.shape[3], filters.shape[0], filters.shape[1], filters.shape[2]))
#         print('convolution filter size', filters.shape)
#
#         length = filters.shape[0]
#         print(length)
#         image = np.zeros((filters.shape[1], filters.shape[2]))
#         print(image.shape)
#         fig = plt.figure(figsize=(12, 12))
#         gs1 = gridspec.GridSpec(math.ceil(np.sqrt(length)), math.ceil(np.sqrt(length)))
#         gs1.update(wspace=0.0, hspace=0.02)
#
#         for j in range(0, length):
#             for depth in range(0, filters.shape[3]):
#                 image = image + filters[j, :, :, depth]
#             # image = image / (depth + 1)
#             ax = fig.add_subplot(gs1[j])
#             ax.imshow(image, cmap='gray')
#             ax.set_xticklabels([])
#             ax.set_yticklabels([])
#             ax.set_aspect('equal')
#             plt.xticks(np.array([]))
#             plt.yticks(np.array([]))
#             # plt.tight_layout()
#             # plt.show()
#         plt.savefig(name_layer + str('_kernel_weights_') + str(j + 1))
#         plt.close('all')
#     else:
#         print("NO filters for this layer")
#
#
# def intermediate_outputs(name, n):
#     intermediate_layer_model = Model(inputs=model.input,
#                                      outputs=model.get_layer(name).output)
#     intermediate_output = intermediate_layer_model.predict(test_x_samples[n:n + 1])
#     print('layer shape', intermediate_output.shape)
#     length = intermediate_output.shape[3]
#     # intermediate_output = np.asarray(intermediate_output)
#     ################################################################################################################
#     plt.figure(figsize=(5 / 2.54, 5 / 2.54), dpi=600)
#     plt.gca().set_axis_off()
#     plt.axis('off')
#     plt.subplots_adjust(left=0.0, bottom=0.0, right=1, top=1.0, wspace=0.0, hspace=0.0)
#     plt.margins(0, 0)
#     plt.gca().xaxis.set_major_locator(plt.NullLocator())
#     plt.gca().yaxis.set_major_locator(plt.NullLocator())
#     ################################################################################################################
#     # print(intermediate_output[0, :, :, i])
#     fig = plt.figure(figsize=(10, 10))
#     for j in range(1, (length + 1)):
#         ax = fig.add_subplot(math.ceil(np.sqrt(length)), math.ceil(np.sqrt(length)), j)
#         ax.imshow(intermediate_output[0, :, :, (j - 1)], cmap='gray')
#         ax.set_xticklabels([])
#         ax.set_yticklabels([])
#         ax.set_aspect('equal')
#         plt.axis('off')
#     plt.tight_layout()
#     # plt.show()
#     plt.savefig(name)
#     plt.close('all')


# input_image = 466  # image number (381,475)
# os.chdir('home/aijjeh/aijjeh_rexio_share/reports/figures/comparative_study/intermediate_outputs')
# Path(str(input_image)).mkdir(parents=True, exist_ok=True)
# os.chdir('home/aijjeh/aijjeh_rexio_share/reports/figures/comparative_study/intermediate_outputs/' + str(input_image))
#
# for layers_count in range(len(model.layers)):
#     print(layers_count, '/', len(model.layers))
#     layer_name = model.layers[layers_count].get_config()
#     layer_name = layer_name['name']
#     print(layer_name)
#     if 'global_average_pooling2d_' not in layer_name:
#         plot_filters(model.layers[layers_count], layer_name)
#         intermediate_outputs(layer_name, ((input_image - 381) * 4))

########################################################################################################################


# Organize figures in folders based on the used threshold

os.chdir('/home/aijjeh/aijjeh_rexio_share/reports/figures/')

path_fcn_figures_src = 'FCN_DenseNet/Num/FCN_Dense_image_number_'
path_fcn_figures_dst = 'FCN_DenseNet/Num/Figure_'

path_unet_figuers_src = 'UNet/Num/Unet_image_number_'
path_unet_figures_dst = 'UNet/Num/Figure_'

path_segnet_figuers_src = '/home/aijjeh/aijjeh_rexio_share/reports/figures/VGG_encoder_decoder/Num/Segnet_image_number_'
path_segnet_figures_dst = '/home/aijjeh/aijjeh_rexio_share/reports/figures/VGG_encoder_decoder/Num/Figure_'


########################################################################################################################
# Thresholding function convert the image values to 0 or 1 based on the used threshold
########################################################################################################################

def thresholding(predicted_img, threshold):
    predicted_img[predicted_img >= threshold] = 1
    predicted_img[predicted_img < threshold] = 0
    gc.collect()
    return predicted_img


########################################################################################################################
# Calculates the Intersection over Union metric for the predicted and label images using bitwise OR & AND
########################################################################################################################

def IoU(predicted_image, truth_img):
    # predicted_image = predicted_image.astype('float32')
    InterSectionArray = cv2.bitwise_and(predicted_image, truth_img)
    UnionArray = cv2.bitwise_or(predicted_image, truth_img)
    I1 = np.count_nonzero(InterSectionArray)
    U = np.count_nonzero(UnionArray)
    IoU1 = I1 / U
    gc.collect()
    return IoU1


########################################################################################################################
# Saves the IoU values to a csv file
########################################################################################################################

# file_name = 'IoU_FCN_DenseNets_softmax.csv'
# file_name = 'IoU_UNet_softmax.csv'
# file_name = 'IoU_VGG16_encoder_decoder_softmax.csv'
file_name = 'PSPNET_resenet50_1_1_ConvD_softmax.csv'

########################################################################################################################
# creates different folders in side each corresponding paths for each detected image for all threshold values
########################################################################################################################

path_fcn_figures_folder = '/home/aijjeh/aijjeh_rexio_share/reports/figures/FCN_DenseNet/Num/Figure_'
path_unet_figures_folder = '/home/aijjeh/aijjeh_rexio_share/reports/figures/UNet/Num/Figure_'
path_segnet_figuers_folder = '/home/aijjeh/aijjeh_rexio_share/reports/figures/VGG_encoder_decoder/Num/Figure_'


########################################################################################################################
# Plots the original delamination, predicted and the label figures
########################################################################################################################

def ploting(original, predict, ground, tr, image_nr):
    fig = plt.figure(figsize=(16, 9))
    fig.suptitle('Delamination, prediction and the mask')
    ax1 = fig.add_subplot(1, 3, 1)
    plt.imshow(original, cmap='gist_yarg')
    ax2 = fig.add_subplot(1, 3, 2)
    plt.imshow(predict, cmap='viridis')
    ax3 = fig.add_subplot(1, 3, 3)
    plt.imshow(ground, cmap='gist_yarg')
    ax1.title.set_text('Delamination damage')
    ax2.title.set_text('Detected damage')
    ax3.title.set_text('Ground Truth')
    fig.tight_layout()
    # plt.show()

    # creates folder to contain different IoU values for different threshold values for the same sample
    Path(path_segnet_figuers_folder + str(image_nr + 1)).mkdir(parents=True, exist_ok=True)

    plt.savefig(
        path_segnet_figuers_src + str(image_nr + 1) + '_threshold_' + str(tr) + '_.png')  # Threshold_'+str(tr)+'
    plt.close('all')
    gc.collect()


########################################################################################################################
csv_files = '/home/aijjeh/aijjeh_rexio_share/reports/figures/comparative_study/csv_files/softmax'

unet_num_csv = 'unet_kfold_num_softmax_iou.csv'
unet_exp_csv = 'unet_kfold_exp_softmax_iou.csv'

vgg16_num_csv = 'vgg16_encoder_decoder_kfold_num_softmax_iou.csv'
vgg16_exp_csv = 'vgg16_encoder_decoder_kfold_exp_softmax_iou.csv'

fcn_densenet_num_csv = 'fcn_densenet_kfold_num_softmax_iou.csv'
fcn_densenet_exp_csv = 'fcn_densenet_kfold_exp_softmax_iou.csv'

pspnet_num_csv = 'pspnet_kfold_num_softmax_iou.csv'
pspnet_exp_csv = 'pspnet_kfold_exp_softmax_iou.csv'

GCN_num_csv = 'GCN_kfold_num_softmax_iou.csv'
GCN_exp_csv = 'GCN_kfold_exp_softmax_iou.csv'


#######################################################################################################################

def append_list_as_row(cvs_file_name, list_of_iou, image_num):
    os.chdir(csv_files)
    with open(cvs_file_name, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerows([image_num])
        writer.writerows([list_of_iou])
        gc.collect()


########################################################################################################################
# Main loop
########################################################################################################################

def main_loop():
    # loop for different threshold values
    end = 100  # parameter for the end step of the threshold counter
    step = 1  # counter step
    for z in range(0, end, step):
        layer_outputs = [layer.output for layer in model.layers[:]]
        activation = model.predict(test_x_samples, batch_size=1)
        last_layer_activation = activation[:]
        image_num_ = []  # holds the image number in the loop
        iou_list = []  # hold the IoU values for certain threshold
        threshold_list = []  # holds the different thresholds for different rounds
        tr = (z + 1) / end  # threshold
        for i in range(380):  # calculating IoU for all figures for a certain threshold
            Predict_Img = thresholding(last_layer_activation[i], threshold=tr)
            Predict_Img = np.asarray(Predict_Img)
            Truth_Img = np.asarray(test_y_samples[i])
            I_o_U = IoU(Predict_Img, Truth_Img)
            threshold_list.append(tr)
            image_num_.append(i + 1)
            iou_list.append(I_o_U)
            original = np.squeeze(test_x_samples[i], axis=2)
            mask = np.squeeze(Truth_Img, axis=2)
            Predict_Img = np.squeeze(last_layer_activation[i], axis=2)
            ploting(original, Predict_Img, mask, tr, i)
        # append_list_as_row(file_name, iou_list, image_num_, threshold_list)


########################################################################################################################

# Path(path_PSPNet_Numerical_figures_softmax).mkdir(parents=True, exist_ok=True)

image_number = []  # holds the image number in the loop
IoU_list = []  # hold the IoU values for certain threshold


########################################################################################################################
# Ploting the prediction for test images
########################################################################################################################

def Testing(csv_file_num, img):
    prediction = model.predict(test_x_samples, batch_size=1)
    for i in range(380):
        damage = (prediction[i])
        damage = np.argmax(damage, axis=2)
        damage = damage.astype('float32')
        original = np.squeeze(test_x_samples[i], axis=2)
        label = np.squeeze(test_y_samples[i], axis=2)
        ################################################################################################################
        plt.figure(figsize=(5 / 2.54, 5 / 2.54), dpi=600)
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
        # plt.show()
        ################################################################################################################
        # plt.imshow(original, cmap='Greys') plt.axis('off') plt.savefig(
        # 'E:/aidd_new/aidd/reports/figures/PSPNet/Num/unthreshoding/FCN_DenseNet_original_454_softmax' + str(i+1))
        # plt.imshow(mask, cmap='gist_gray') plt.axis('off') plt.savefig(
        # 'E:/aidd_new/aidd/reports/figures/PSPNet/Num/unthreshoding/FCN_DenseNet_GT_454_softmax' + str(i+1))
        # plt.show()
        image_number.append(i + 1)
        IoU_list.append(I_o_U)
        plt.close('all')
        print(I_o_U, ' ', (i + 1))
    append_list_as_row(csv_file_num, IoU_list, image_number)


########################################################################################################################

image_number_exp = []  # holds the image number in the loop
IoU_list_exp = []  # hold the IoU values for certain threshold


########################################################################################################################
# Ploting the prediction for experimental images
########################################################################################################################

def exp(csv_file_exp, img):
    prediction = model.predict(experimental, batch_size=1)
    prediction = np.asarray(prediction)
    for i in range(3):
        damage = prediction[i]
        damage = np.argmax(damage, axis=2)
        damage = damage.astype('float64')
        original = np.squeeze(experimental[i], axis=2)
        label = np.squeeze(experimental_label[i], axis=2)
        ################################################################################################################
        plt.figure(figsize=(5 / 2.54, 5 / 2.54), dpi=600)
        plt.gca().set_axis_off()
        plt.axis('off')
        plt.subplots_adjust(left=0.0, bottom=0.0, right=1, top=1.0, wspace=0.0, hspace=0.0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        ################################################################################################################
        I_o_U = IoU(damage, label)
        plt.imshow(original, cmap='Greys')
        plt.imshow(damage, cmap=cmap, alpha=0.3)
        plt.imshow(label, cmap='Reds', alpha=0.1)

        plt.show()
        # plt.imshow(original, cmap='Greys')
        # plt.axis('off')
        # plt.show()
        # plt.imshow(label, cmap=cmap)
        # plt.axis('off')
        # plt.savefig(path_pspnet_figures_exp + '/Fig_' + str(i + 1))
        # plt.show()
        plt.close('all')
        print(I_o_U, ' ', (i + 1))

        image_number_exp.append(i + 1)
        IoU_list_exp.append(I_o_U)
    # append_list_as_row(csv_file_exp, IoU_list_exp, image_number_exp)
    gc.collect()


########################################################################################################################
# paths to folders to save the output NUM images
########################################################################################################################
comparative_study_path = '/home/aijjeh/aijjeh_rexio_share/reports/figures/comparative_study/'

os.chdir(comparative_study_path)

path_vgg16_num_figuers_softmax = 'vgg16_encoder_decoder/softmax/num/'
path_FCN_DenseNet_Numerical_figuers_softmax = 'fcn_densenet/softmax/num/'
path_unet_Numerical_figuers_softmax = 'unet/softmax/num'
path_PSPNet_Numerical_figures_softmax = 'pspnet/softmax/num/'
path_GCN_Numerical_figures_softmax = 'GCN/num/'

########################################################################################################################
# paths to folders to save the output Exp images
########################################################################################################################

path_vgg16_exp_figuers_softmax = 'vgg16_encoder_decoder/softmax/exp/'
path_unet_Exp_figuers_softmax = 'unet/softmax/exp'
path_FCNDenseNet_Exp_figuers_softmax = 'fcn_densenet/softmax/exp/'
path_PSPNet_Exp_figures_softmax = 'pspnet/softmax/exp/'
path_GCN_Exp_figures_softmax = 'GCN/exp/'

########################################################################################################################
# Running functions
########################################################################################################################

# select the num and exp paths for plotting images
# select the csv files for num and exp results
# select the image names to be saved

img_numerical = 'pspnet_num_'
img_exp = "GCN_stringer_exp_"

if __name__ == '__main__':
    os.chdir(comparative_study_path + path_PSPNet_Numerical_figures_softmax)
    # Testing(pspnet_num_csv, img_numerical)

    os.chdir(comparative_study_path + path_GCN_Exp_figures_softmax)
    exp(GCN_exp_csv, img_exp)

    # main_loop()

########################################################################################################################
gc.collect()
