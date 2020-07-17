import csv
import gc
from pathlib import Path
import cv2
import matplotlib
import matplotlib.pyplot as plt
# from sklearn.utils import shuffle
import numpy as np
import tensorflow as tf
from keras.models import load_model
from matplotlib import cm
from matplotlib.colors import ListedColormap
from mpmath import norm

########################################################################################################################
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
########################################################################################################################
# garbage collector
gc.collect()
########################################################################################################################
############################## Loading dataset into x, y arrays and reshape them #######################################
########################################################################################################################
x = np.load('E:/backup/datasets/Segmentation datasets/augemented/June_training_dataset.npy')
x = x.reshape(1900, 512, 512, 1)
x = x / 255.0  # normalizing x,y to (0-1) range
y = np.load('E:/backup/datasets/Segmentation datasets/augemented/June_labels.npy')
y = y.reshape(1900, 512, 512, 1)
y = y / 255.0

########################################################################################################################
###################################### Shuffle the data set at random ##################################################
########################################################################################################################
# x, y = shuffle(x, y)
########################################################################################################################
################# Split dataset into training and testing sets and again re-shuffle them ###############################
########################################################################################################################
x_train = x[:1520]
y_train = y[:1520]

# n = 433 # image number
# test_x_samples = x[(n-1)*4:n*4]
# test_y_samples = y[(n-1)*4:n*4]
test_x_samples = x[1520:1900]
test_y_samples = y[1520:1900]

########################################################################################################################
######################################## Loading experimental images ###################################################
########################################################################################################################

experimental = np.load('exp_ERMS.npy')
experimental_label = np.load('exp_label.npy')
experimental = experimental / 255.0
experimental_label = experimental_label / 255.0
experimental = experimental.reshape(23, 512, 512, 1)
experimental_label = experimental_label.reshape(23, 512, 512, 1)

########################################################################################################################
#################################################### Loading the model #################################################
########################################################## Models#######################################################
# FCN_softmax_100_epoches #tr = 1
# VGG16_100_epochs_softmax
# Unet_100_epoches_softmax

# PsPnet_BatchNormalization_add__to_globalaverage_Activation
# PSPNET_resenet50_1_1_ConvD_softmax
########################################################################################################################
################################################ paths for the model srcs ##############################################

fcn_path = 'E:/backup/models/FCN_DenseNet_models/Softmax/'
unet_path = 'E:/backup/models/UNet_models/Softmax/'
vgg16_path = 'E:/backup/models/SegNet_models/Softmax/'
# PsPnet_BatchNormalization_add__to_globalaverage_Activation_softmax
model_name = 'PSPNET_resenet50_1_1_ConvD_softmax.h5'  # fcn_path+'FCN_softmax_100_epoches.h5'
model = load_model(model_name, compile=False)
model.summary()

########################################################################################################################
############################################## reading Cmap for plotting ###############################################

path_to_csv = "E:/aidd_new/aidd/src/data_processing/PhD/cmap_flipped_jet256.csv"
cmap = matplotlib.colors.ListedColormap(["blue", "green", "red"], name=path_to_csv, N=None)
m = cm.ScalarMappable(norm=norm, cmap=cmap)

# Organize figures in folders based on the used threshold

path_fcn_figures_src = 'E:/aidd_new/aidd/reports/figures/FCN_DenseNet/Num/FCN_Dense_image_number_'
path_fcn_figures_dst = 'E:/aidd_new/aidd/reports/figures/FCN_DenseNet/Num/Figure_'

path_unet_figuers_src = 'E:/aidd_new/aidd/reports/figures/UNet/Num/Unet_image_number_'
path_uent_figures_dst = 'E:/aidd_new/aidd/reports/figures/UNet/Num/Figure_'

path_segnet_figuers_src = 'E:/aidd_new/aidd/reports/figures/VGG_encoder_decoder/Num/Segnet_image_number_'
path_segnet_figures_dst = 'E:/aidd_new/aidd/reports/figures/VGG_encoder_decoder/Num/Figure_'


########################################################################################################################
###################### Thresholding function convert the image values to 0 or 1 based on the used threshold ############
########################################################################################################################

def thresholding(predicted_img, threshold):
    predicted_img[predicted_img >= threshold] = 1
    predicted_img[predicted_img < threshold] = 0
    gc.collect()
    return predicted_img


########################################################################################################################
######## Calculates the Intersection over Union metric for the predicted and label images using bitwise OR & AND #######
########################################################################################################################
def IoU(predicted_image, truth_img):
    ######################################
    predicted_image = predicted_image.astype('float64')
    InterSectionArray = cv2.bitwise_and(predicted_image, truth_img)
    UnionArray = cv2.bitwise_or(predicted_image, truth_img)
    I1 = np.count_nonzero(InterSectionArray)
    U = np.count_nonzero(UnionArray)
    IoU1 = I1 / U
    print(IoU1)
    gc.collect()
    return IoU1


########################################################################################################################
########################################### Saves the IoU values to a csv file #########################################
# file_name = 'IoU_FCN_DenseNets_softmax.csv'
# file_name = 'IoU_UNet_softmax.csv'
# file_name = 'IoU_VGG16_encoder_decoder_softmax.csv'
file_name = 'PSPNET_resenet50_1_1_ConvD_softmax.csv'


def append_list_as_row(cvs_file_name, list_of_iou, image_num):
    with open(cvs_file_name, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerows([image_num])
        writer.writerows([list_of_iou])
        gc.collect()


########################################################################################################################
##### creates different folders in side each corrospoding paths for each detected image for all threshold values########
########################################################################################################################
path_fcn_figures_folder = 'E:/aidd_new/aidd/reports/figures/FCN_DenseNet/Num/Figure_'
path_unet_figures_folder = 'E:/aidd_new/aidd/reports/figures/UNet/Num/Figure_'
path_segnet_figuers_folder = 'E:/aidd_new/aidd/reports/figures/VGG_encoder_decoder/Num/Figure_'


########################################################################################################################
####################### Plots the original delamination, predicted and the label figures ###############################
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

    # creates folder to contain different IoU values for diffferent threshold values for the same sample
    Path(path_segnet_figuers_folder + str(image_nr + 1)).mkdir(parents=True, exist_ok=True)

    plt.savefig(
        path_segnet_figuers_src + str(image_nr + 1) + '_threshold_' + str(tr) + '_.png')  # Threshold_'+str(tr)+'
    plt.close('all')
    gc.collect()


########################################################################################################################
#################################################### Main loop #########################################################
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
        append_list_as_row(file_name, iou_list, image_num_, threshold_list)


########################################################################################################################
##################################### paths to folders to save the output NUM images ###################################
########################################################################################################################
path_segnet_figuers_folder_no_threshold = 'E:/aidd_new/aidd/reports/figures/VGG_encoder_decoder/Num/Fig_unthreshold_softmax'
path_fcn_figuers_folder_no_threshold = 'E:/aidd_new/aidd/reports/figures/FCN_DenseNet/Num/Fig_unthreshold_softmax'
path_unet_figuers_folder_no_threshold = 'E:/aidd_new/aidd/reports/figures/UNet/Num/Fig_unthreshold_softmax'
path_PSPNET_figures_softmax = 'E:/aidd_new/aidd/reports/figures/PSPNet/Num/softmax/'

Path(path_PSPNET_figures_softmax).mkdir(parents=True, exist_ok=True)  # create folder for the original images

image_number = []  # holds the image number in the loop
IoU_list = []  # hold the IoU values for certain threshold

file_name_test = 'IoU_PSPnet_resnet_softmax_num.csv'


########################################################################################################################
############################################ Ploting the prediction for test images ####################################
########################################################################################################################
def Testing():
    prediction = model.predict(test_x_samples, batch_size=1)
    for i in range(380):
        if i % 4 == 0:
            damage = (prediction[i])
            damage = np.argmax(damage, axis=2)

            original = np.squeeze(test_x_samples[i], axis=2)
            mask = np.squeeze(test_y_samples[i], axis=2)
            ###############################################################################################################
            plt.figure(figsize=(5 / 2.54, 5 / 2.54), dpi=600)
            plt.gca().set_axis_off()
            plt.axis('off')
            plt.subplots_adjust(left=0.0, bottom=0.0, right=1, top=1.0, wspace=0.0, hspace=0.0)
            plt.margins(0, 0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            ###############################################################################################################
            plt.imshow(damage, cmap=cmap)
            plt.axis('off')
            plt.savefig(path_PSPNET_figures_softmax + 'PSPnet_Pred__softmax' + str(i + 1))

            plt.imshow(original, cmap='Greys')
            plt.axis('off')
            # plt.savefig('E:/aidd_new/aidd/reports/figures/PSPNet/Num/unthreshoding/FCN_DenseNet_original_454_softmax' + str(i+1))

            plt.imshow(mask, cmap='gist_gray')
            plt.axis('off')
            # plt.savefig('E:/aidd_new/aidd/reports/figures/PSPNet/Num/unthreshoding/FCN_DenseNet_GT_454_softmax' + str(i+1))
            # plt.show()
            plt.close('all')
            I_o_U = IoU(damage, mask)
            image_number.append(i + 1)
            IoU_list.append(I_o_U)
            print(i + 1)
    append_list_as_row(file_name_test, IoU_list, image_number)


########################################################################################################################
##################################### paths to folders to save the output Exp images ###################################
########################################################################################################################

path_segnet_figuers_folder_no_threshold_exp = 'E:/aidd_new/aidd/reports/figures/VGG_encoder_decoder/Exp/Fig_unthreshold_softmax'
path_unet_figuers_folder_no_threshold_exp = 'E:/aidd_new/aidd/reports/figures/UNet/Exp/Fig_unthreshold_softmax'
path_fcn_figuers_folder_no_threshold_exp = 'E:/aidd_new/aidd/reports/figures/FCN_DenseNet/Exp/Fig_unthreshold_softmax'
path_pspnet_figures_exp = 'E:/aidd_new/aidd/reports/figures/PSPNet/Exp/softmax/'

Path(path_pspnet_figures_exp).mkdir(parents=True, exist_ok=True)  # create folder for the EXP

image_number_exp = []  # holds the image number in the loop
IoU_list_exp = []  # hold the IoU values for certain threshold
file_name_exp = 'IoU_PSPnet_restnet_softmax_exp.csv'


########################################################################################################################
############################################ Ploting the prediction for experimental images ############################
########################################################################################################################
def exp():
    prediction = model.predict(experimental, batch_size=1)
    prediction = np.asarray(prediction)

    for i in range(23):
        damage = (prediction[i])
        # damage = thresholding(damage,0.51)
        damage = np.argmax(damage, axis=2)
        original = np.squeeze(experimental[i], axis=2)
        label = np.squeeze(experimental_label[i], axis=2)
        plt.figure(figsize=(5 / 2.54, 5 / 2.54), dpi=600)
        plt.gca().set_axis_off()
        plt.axis('off')
        plt.subplots_adjust(left=0.0, bottom=0.0, right=1, top=1.0, wspace=0.0, hspace=0.0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.imshow(damage, cmap=cmap)
        plt.axis('off')
        plt.savefig(path_pspnet_figures_exp + '/PSPnet_Pred_softmax' + str(i + 1))
        plt.imshow(original, cmap='Greys')
        plt.axis('off')

        plt.imshow(label, cmap=cmap)
        plt.axis('off')

        # print(i+1)
        # IoU(damage,label)
        # plt.savefig(path_unet_figuers_folder_no_threshold_exp+'/Fig_' + str(i+1))
        # plt.show()
        plt.close('all')
        print(i + 1)
        I_o_U = IoU(damage, label)
        image_number_exp.append(i + 1)
        IoU_list_exp.append(I_o_U)
    append_list_as_row(file_name_exp, IoU_list_exp, image_number_exp)
    gc.collect()


########################################################################################################################
############################################ Running functions  ########################################################
########################################################################################################################
# Testing()
# main_loop()
exp()
########################################################################################################################
gc.collect()
