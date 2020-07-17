import csv
import gc
from pathlib import Path
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.models import load_model
from matplotlib import cm
from matplotlib.colors import ListedColormap
from mpmath import norm

###########################################   memory growing ###########################################################
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
########################################################################################################################
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.9
# session = tf.Session(config=config)
########################################################################################################################
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
# FCN_Dense_net_Precision_BCE_ConvFilter_16_changing_layers_size_  #FCN model # when tr =1 shows good exp output
# FCN_Dense_net_IoU_100_epoches_ new model #iou 100 epochs
# New_data_unet_adding_dropout_latest_100Epoches    #Unet model
# Unet_epoches_100_iou_metric lasest wtih 100 epoches and iou # iou 100 epochs
# SegNet_Upsampling_added_skip_layer_function_10_epoches_3_3_covolution_vgg_layers.h5 #great numerical results
# SegNet_Encoder_decoder_added_skip_100_epoches_3_3_conv_vgg_16_archi_layers # FCN Vgg16 encoder decoder iou 100 epochs

# PSPNET_3_conv_layers
# PSPNET_resenet50_1_1_ConvD_sigmoid
######################################################### EXP ##########################################################
# FCN_Dense_net_Precision_BCE_ConvFilter_16_changing_layers_size_ somehow good # when tr =1 shows good exp output
# New_data_unet_adding_dropout # gave somehow good exp results

# FCN_DsensNets_Semantic_Segmentation_filter_Using_Conv2DTranspose16_epoch_5_kernal_(3, 3)_
# drpout_0.2_batch_size_4_loss_updated_changed_DB _layer_Iou_and_loss_changed     # when tr =0.999 shows good exp output

# SegNet_Upsampling_added_skip_layer_function_10_epoches_7_7_covolution_4_layers
# great model # when tr =.45-.6 shows good exp output


# 'PsPnet_BatchNormalization_add__to_globalaverage_Activation_sigmoid.h5'
########################################################################################################################
################################################ paths for the model srcs ##############################################
fcn_path = 'E:/backup/models/FCN_DenseNet_models/'
unet_path = 'E:/backup/models/UNet_models/'
vgg16_path = 'E:/backup/models/SegNet_models/'
# model_name = fcn_path+'FCN_Dense_net_IoU_100_epoches_.h5'
model_name = 'PSPNET_resenet50_1_1_ConvD_sigmoid.h5'

model = load_model(model_name, compile=False)
model.summary()
########################################################################################################################
############################################## reading Cmap for plotting ###############################################
########################################################################################################################
path_to_csv = "E:/aidd_new/aidd/src/data_processing/PhD/cmap_flipped_jet256.csv"
cmap = matplotlib.colors.ListedColormap(["blue", "green", "red"], name=path_to_csv, N=None)
m = cm.ScalarMappable(norm=norm, cmap=cmap)
########################################################################################################################
####################################### creates folder to contain folders in NUM folder ################################
########################################################################################################################
path_fcn_Unet = 'E:/aidd_new/aidd/reports/figures/UNet/Num/Sigmoid_100_epoches'
path_fcn_DenseNet = 'E:/aidd_new/aidd/reports/figures/FCN_DenseNet/Num/Sigmoid_100_epoches'
path_fcn_vgg16 = 'E:/aidd_new/aidd/reports/figures/VGG_encoder_decoder/Num/Sigmoid_100_epoches'
path_PSPnet = 'E:/aidd_new/aidd/reports/figures/PSPNet/Num/Sigmoid_100_epoches'
########################################################################################################################
Path(path_fcn_DenseNet).mkdir(parents=True, exist_ok=True)


########################################################################################################################
###################### Thresholding function convert the image values to 0 or 1 based on the used threshold ############
########################################################################################################################
def thresholding(predicted_img, threshold):
    predicted_img[predicted_img > threshold] = 1
    predicted_img[predicted_img <= threshold] = 0
    gc.collect()
    return predicted_img


########################################################################################################################
######## Calculates the Intersection over Union metric for the predicted and label images using bitwise OR & AND #######
########################################################################################################################
def IoU(predicted_image, truth_img):
    predicted_image = predicted_image.astype('float64')
    inter_section_array = cv2.bitwise_and(predicted_image, truth_img)
    union_array = cv2.bitwise_or(predicted_image, truth_img)
    i1 = np.count_nonzero(inter_section_array)
    u = np.count_nonzero(union_array)
    IoU1 = i1 / u
    ##################################################
    # I =0
    # P=0
    # G=0
    # for i in range(512):
    #    for j in range(512):
    #        if predicted_image[i,j]==1:
    #            P = P+1
    #        if truth_img[i,j] == 1:
    #            G = G+1
    #        if predicted_image[i,j] >0.9 and truth_img[i,j] == 1:
    #            I = I+1
    # IoU1 = I/(P+G-I)
    # print(I,P,G)
    return IoU1


########################################################################################################################
########################################### Saves the IoU values to a csv file #########################################
########################################################################################################################
# file_name = 'IoU_FCN_DenseNets_unthresholded.csv'
# file_name = 'IoU_UNet.csv'
# file_name = 'IoU_VGG16_encoder_decoder.csv'
file_name = 'IoU_PSPNet_all_thresholds_resnet50_1_1_conv.csv'


########################################################################################################################
def append_list_as_row(csv_file, list_of_iou, image_number):  # , threshold_list):
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        # writer.writerows([threshold_list])
        writer.writerows([image_number])
        writer.writerows([list_of_iou])
        gc.collect()


########################################################################################################################
##### creates different folders in side each corrospoding paths for each detected image for all threshold values########
########################################################################################################################
path_fcn_figures_folder = path_fcn_DenseNet + '/Figure_'
path_unet_figures_folder = path_fcn_Unet + '/Figure_'
path_fcn_vgg16_figuers_folder = path_fcn_vgg16 + '/Figure_'
path_pspnet_figure = path_PSPnet + '/Figure_'


########################################################################################################################
####################### Plots the original delamination, predicted and the label figures ###############################
########################################################################################################################
def plotting(original, predict, ground, tr, image_number, C):
    # creates folder to contain different IoU values
    # for diffferent threshold values for the same sample
    folder_path = path_pspnet_figure + str(image_number + 1)
    Path(folder_path).mkdir(parents=True, exist_ok=True)
    ############################################################################################################
    # plt.figure(figsize=(5 / 2.54, 5 / 2.54), dpi=600)
    # plt.gca().set_axis_off()
    # plt.axis('off')
    # plt.subplots_adjust(left=0.0, bottom=0.0, right=1, top=1.0, wspace=0.0, hspace=0.0)
    # plt.margins(0, 0)
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())
    ###########################################################################################################
    if C == 0:
        plt.imshow(original, cmap='Greys')
        plt.savefig(folder_path + '/Original_Figure_' + str(image_number + 1) + '_.png')
        plt.imshow(ground, cmap='gist_gray')
        plt.savefig(folder_path + '/GT_Figure_' + str(image_number + 1) + '_.png')
    plt.imshow(predict, cmap=cmap)
    plt.savefig(folder_path + '/Predicted_Figure_' + str(image_number + 1) + '_threshold_' + str(tr) + '_.png')
    plt.close('all')
    gc.collect()


########################################################################################################################
###########################  Main Loop for calculating different thresholding values and   #############################
########################################################################################################################
def main_loop():
    # loop for different threshold values
    end = 100  # parameter for the end step of the threshold counter
    step = 1  # counter step
    c = 0  # condition to save the original and GT images for one time
    for z in range(0, end, step):
        layer_outputs = [layer.output for layer in model.layers[:]]
        activation = model.predict(test_x_samples, batch_size=1)
        last_layer_activation = activation[:]
        image_number = []  # holds the image number in the loop
        iou_list = []  # hold the IoU values for certain threshold
        threshold_list = []  # holds the different thresholds for different rounds
        tr = (z + 1) / end  # threshold
        for i in range(380):  # calculating IoU for all figures for a certain threshold
            predict_img = thresholding(last_layer_activation[i], threshold=tr)
            predict_img = np.asarray(predict_img)
            truth_img = np.asarray(test_y_samples[i])
            iou = IoU(predict_img, truth_img)
            threshold_list.append(tr)
            image_number.append(i + 1)
            iou_list.append(iou)
            original = np.squeeze(test_x_samples[i], axis=2)
            mask = np.squeeze(truth_img, axis=2)
            predict_img = np.squeeze(last_layer_activation[i], axis=2)
            # plotting(original, predict_img, mask, tr, i, c)
            print(i, tr)
        append_list_as_row(file_name, iou_list, image_number, threshold_list)
        c = 1
    gc.collect()


########################################################################################################################
##################################### paths to folders to save the output NUM images ###################################
########################################################################################################################
path_segnet_figures_folder_no_threshold = 'E:/aidd_new/aidd/reports/figures/VGG_encoder_decoder/Num/Fig_unthreshold_100_iou_metricc'
path_fcn_figures_folder_no_threshold = 'E:/aidd_new/aidd/reports/figures/FCN_DenseNet/Num/Fig_unthreshold__100_iou_metricc/unthresholding'
path_unet_figures_folder_no_threshold = 'E:/aidd_new/aidd/reports/figures/UNet/Num/Fig_unthreshold_kernel_100_iou_metricc'
pspnet_path_testing = 'E:/aidd_new/aidd/reports/figures/PSPNet/Num/testing/'

# Path(path_fcn_figures_folder_no_threshold).mkdir(parents=True, exist_ok=True) # create folder for the original images
image_number_num = []  # holds the image number in the loop
IoU_list_num = []  # hold the IoU values for certain threshold
########################################################################################################################
############################################ Ploting the prediction for test images ####################################
########################################################################################################################
file_pspnet_test = 'PSPNET_IoU_testing.csv'
file_pspnet_exp = 'PSPNET_IoU_exp.csv'


########################################################################################################################
def testing():
    prediction = model.predict(test_x_samples, batch_size=1)
    prediction = np.asarray(prediction)
    for i in range(380):
        if i % 1 == 0:
            damage = np.squeeze(prediction[i], axis=2)
            damage = thresholding(damage, 0.5)
            original = np.squeeze(test_x_samples[i], axis=2)
            mask = np.squeeze(test_y_samples[i], axis=2)
            ############################################################################################################
            plt.figure(figsize=(5 / 2.54, 5 / 2.54), dpi=600)
            plt.gca().set_axis_off()
            plt.axis('off')
            plt.subplots_adjust(left=0.0, bottom=0.0, right=1, top=1.0, wspace=0.0, hspace=0.0)
            plt.margins(0, 0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            ############################################################################################################
            plt.imshow(damage, cmap=cmap)
            plt.savefig(pspnet_path_testing + 'fig_' + str(i))
            ############################################################################################################
            # plt.imshow(original, cmap='Greys')
            # plt.savefig(path_fcn_figuers_folder_no_threshold+'/FCN_DenseNet_original_454_sigmoid' + str(i+1))
            ############################################################################################################
            # plt.imshow(mask, cmap='gist_gray')
            # plt.savefig(path_fcn_figuers_folder_no_threshold+'/FCN_DenseNet_GT_454_sigmoid' + str(i+1))
            ############################################################################################################
            # plt.show()
            plt.close('all')
            ############################################################################################################
            print(i + 1, IoU(damage, mask))
            I_o_U = IoU(damage, mask)
            image_number_num.append(i + 1)
            IoU_list_num.append(I_o_U)
            ############################################################################################################
    append_list_as_row(file_pspnet_test, IoU_list_num, image_number_num)


########################################################################################################################
##################################### paths to folders to save the output Exp images ###################################
########################################################################################################################
path_Vgg16_seg_figuers_folder_no_threshold_exp = 'E:/aidd_new/aidd/reports/figures/VGG_encoder_decoder/Exp/Fig_unthreshold_100_iou_metricc'
path_unet_figuers_folder_no_threshold_exp = 'E:/aidd_new/aidd/reports/figures/UNet/Exp/Fig_unthreshold_100_iou_metricc'
path_fcn_figuers_folder_no_threshold_exp = 'E:/aidd_new/aidd/reports/figures/FCN_DenseNet/Exp/Fig_unthreshold_100_iou_metricc'
pspnet_path_exp = 'E:/aidd_new/aidd/reports/figures/PSPNet/Exp/experimental/'
Path(path_fcn_figuers_folder_no_threshold_exp).mkdir(parents=True, exist_ok=True)  # create folder for the EXP
file_iou_exp = 'exp_iou.csv'
########################################################################################################################
############################################ Ploting the prediction for experimental images ############################
########################################################################################################################
image_number_exp = []  # holds the image number in the loop
IoU_list_exp = []  # hold the IoU values for certain threshold


########################################################################################################################
def exp():
    prediction = model.predict(experimental, batch_size=1)
    prediction = np.asarray(prediction)
    for i in range(23):
        damage = np.squeeze(prediction[i], axis=2)
        damage = thresholding(damage, 0.5)
        # print(damage)
        original = np.squeeze(experimental[i], axis=2)
        label = np.squeeze(experimental_label[i], axis=2)
        ###############################################################################################################
        plt.figure(figsize=(5 / 2.54, 5 / 2.54), dpi=600)
        plt.gca().set_axis_off()
        plt.axis('off')
        plt.subplots_adjust(left=0.0, bottom=0.0, right=1, top=1.0, wspace=0.0, hspace=0.0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        ################################################################################################################
        plt.imshow(damage, cmap=cmap)
        plt.savefig(pspnet_path_exp + '/Predict_damage_' + str(i + 1) + '_.png', bboox_inches='tight', pad_inches=0)
        ################################################################################################################
        # plt.imshow(original, cmap='Greys')
        # plt.savefig(path_fcn_figuers_folder_no_threshold_exp+'/Original_damage_' + str(i + 1) +'_.png')
        ################################################################################################################
        # plt.imshow(label, cmap='gist_gray')
        # plt.savefig(path_fcn_figuers_folder_no_threshold_exp+'/Mask_' + str(i + 1) +'_.png')
        ################################################################################################################
        # plt.show()
        plt.close('all')
        ################################################################################################################
        iou = IoU(damage, label)
        print(i + 1, iou)
        I_o_U = IoU(damage, label)
        image_number_exp.append(i + 1)
        IoU_list_exp.append(I_o_U)
        gc.collect()
    append_list_as_row(file_pspnet_exp, IoU_list_exp, image_number_exp)


########################################################################################################################
############################################ Running functions  ########################################################
########################################################################################################################
# testing()
# main_loop()
exp()
########################################################################################################################
gc.collect()
