import os
import cv2
from keras.models import load_model
from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt
import csv
import gc
from pathlib import Path
import shutil
import tensorflow as tf
import matplotlib
from matplotlib import cm
from matplotlib.colors import ListedColormap
from mpmath import norm

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

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
#x, y = shuffle(x, y)
########################################################################################################################
################# Split dataset into training and testing sets and again re-shuffle them ###############################
########################################################################################################################
x_train = x[:1520]
y_train = y[0:1520]

test_x_samples = x[1520:1900]
tests_y_samples = y[1520:1900]
########################################################################################################################
######################################## Loading experimental images ###################################################
########################################################################################################################

experimental = np.load('exp_ERMS.npy')
experimental_label = np.load('exp_label.npy')
experimental = experimental / 255.0
experimental_label =experimental_label/ 255.0
experimental = experimental.reshape(23, 512, 512, 1)
experimental_label = experimental_label.reshape(23,512,512,1)

########################################################################################################################
#################################################### Loading the model #################################################
########################################################## Models#######################################################
# FCN_Dense_net_Precision_BCE_ConvFilter_16_changing_layers_size_  #FCN model # when tr =1 shows good exp output
# FCN_Dense_net_IoU_100_epoches_ new model #iou 100 epochs
# New_data_unet_adding_dropout_latest_100Epoches    #Unet model
# Unet_epoches_100_iou_metric lasest wtih 100 epoches and iou # iou 100 epochs
# SegNet_Upsampling_added_skip_layer_function_10_epoches_3_3_covolution_vgg_layers.h5 #great numerical results
# SegNet_Encoder_decoder_added_skip_100_epoches_3_3_conv_vgg_16_archi_layers # FCN Vgg16 encoder decoder iou 100 epochs
######################################################### EXP ##########################################################
#FCN_Dense_net_Precision_BCE_ConvFilter_16_changing_layers_size_ somehow good # when tr =1 shows good exp output
# New_data_unet_adding_dropout # gave somehow good exp results
#FCN_DsensNets_Semantic_Segmentation_filter_Using_Conv2DTranspose16_epoch_5_kernal_(3, 3)_drpout_0.2_batch_size_4_loss_updated_changed_DB _layer_Iou_and_loss_changed     # when tr =0.999 shows good exp output
# SegNet_Upsampling_added_skip_layer_function_10_epoches_7_7_covolution_4_layers # great model # when tr =.45-.6 shows good exp output
########################################################################################################################
################################################ paths for the model srcs ##############################################

fcn_path = 'E:/backup/models/FCN_DenseNet_models/'
unet_path = 'E:/backup/models/UNet_models/'
vgg16_path = 'E:/backup/models/SegNet_models/'

model_name = fcn_path+'FCN_Dense_net_IoU_100_epoches_.h5'
model = load_model(model_name, compile=False)
model.summary()

########################################################################################################################
############################################## reading Cmap for plotting ###############################################

path_to_csv= "E:/aidd_new/aidd/src/data_processing/PhD/cmap_flipped_jet256.csv"
cmap = matplotlib.colors.ListedColormap(["blue","green","red"], name=(path_to_csv), N=None)
m = cm.ScalarMappable(norm=norm, cmap=cmap)
####################################### creates folder to contain folders in NUM folder ################################
# creates folder to contain folders in NUM folder
path_fcn_Unet = 'E:/aidd_new/aidd/reports/figures/UNet/Num/Sigmoid_100_epoches'
path_fcn_DenseNet = 'E:/aidd_new/aidd/reports/figures/FCN_DenseNet/Num/Sigmoid_100_epoches'
path_fcn_vgg16 = 'E:/aidd_new/aidd/reports/figures/VGG_encoder_decoder/Num/Sigmoid_100_epoches'
path_PSPnet  = 'E:/aidd_new/aidd/reports/figures/PSPNet/Num/Sigmoid_100_epoches'

Path(path_fcn_DenseNet).mkdir(parents=True, exist_ok=True)

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
#file_name = 'IoU_FCN_DenseNets.csv'
#file_name = 'IoU_UNet.csv'
file_name = 'IoU_VGG16_encoder_decoder.csv'

def append_list_as_row(file_name, list_of_Iou, image_number,threshold_list):
    with open(file_name, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerows([threshold_list])
        writer.writerows([image_number])
        writer.writerows([list_of_Iou])
        gc.collect()



########################################################################################################################
##### creates different folders in side each corrospoding paths for each detected image for all threshold values########
########################################################################################################################
path_fcn_figures_folder= path_fcn_DenseNet+'/Figure_'
path_unet_figures_folder = path_fcn_Unet+'/Figure_'
path_fcn_vgg16_figuers_folder = path_fcn_vgg16 + '/Figure_'
path_pspnet_figure = path_PSPnet + '/Figure_'
########################################################################################################################
####################### Plots the original delamination, predicted and the label figures ###############################
########################################################################################################################
def plotting(original,predict,ground,tr,image_number,C):
    # creates folder to contain different IoU values
    # for diffferent threshold values for the same sample
    folder_path =path_fcn_figures_folder + str(image_number + 1)
    Path(folder_path).mkdir(parents=True, exist_ok=True)
    if C==0:
        plt.imshow(original, cmap='Greys')
        plt.axis('off')
        #plt.savefig(folder_path+'/Original_Figure_' + str(image_number + 1) +'_.png') #Threshold_'+str(tr)+'
        plt.imshow(ground, cmap='gist_gray')
        plt.axis('off')
        #plt.savefig(folder_path+'/GT_Figure_' + str(image_number + 1) + '_.png') #Threshold_'+str(tr)+'

    plt.imshow(predict, cmap=cmap)
    plt.axis('off')
    #plt.savefig(folder_path+'/Predicted_Figure_' + str(image_number + 1) + '_threshold_' + str(tr) + '_.png') #Threshold_'+str(tr)+'
    plt.close('all')
    gc.collect()

########################################################################################################################
###########################  Main Loop for calculating different thresholding values and   #############################
########################################################################################################################
def main_loop():
    # loop for different threshold values
    end = 100 # parameter for the end step of the threshold counter
    step = 1 # counter step
    C = 0  # condition to save the original and GT images for one time
    for z in range(0, end, step):
        layer_outputs = [layer.output for layer in model.layers[:]]
        activation = model.predict(test_x_samples, batch_size=1)
        last_layer_activation = activation[:]
        image_number = []  # holds the image number in the loop
        IoU_list = []  # hold the IoU values for certain threshold
        threshold_list = []  # holds the different thresholds for different rounds
        tr = (z + 1) / end  # threshold
        for i in range(380): # calculating IoU for all figures for a certain threshold
            Predict_Img = thresholding(last_layer_activation[i], threshold=tr)
            Predict_Img = np.asarray(Predict_Img)
            Truth_Img = np.asarray(tests_y_samples[i])
            I_o_U = IoU(Predict_Img, Truth_Img)
            threshold_list.append(tr)
            image_number.append(i + 1)
            IoU_list.append(I_o_U)
            original = np.squeeze(test_x_samples[i], axis=2)
            mask = np.squeeze(Truth_Img, axis=2)
            Predict_Img = np.squeeze(last_layer_activation[i],axis=2)
            plotting(original, Predict_Img, mask,tr,i,C)

        append_list_as_row(file_name, IoU_list, image_number,threshold_list)
        C =1

    gc.collect()


########################################################################################################################
##################################### paths to folders to save the output NUM images ###################################
########################################################################################################################
path_segnet_figuers_folder_no_threshold = 'E:/aidd_new/aidd/reports/figures/VGG_encoder_decoder/Num/Fig_unthreshold_100_iou_metricc'
path_fcn_figuers_folder_no_threshold = 'E:/aidd_new/aidd/reports/figures/FCN_DenseNet/Num/Fig_unthreshold__100_iou_metricc'
path_unet_figuers_folder_no_threshold = 'E:/aidd_new/aidd/reports/figures/UNet/Num/Fig_unthreshold_kernel_100_iou_metricc'

Path(path_unet_figuers_folder_no_threshold).mkdir(parents=True, exist_ok=True) # create folder for the original images
########################################################################################################################
############################################ Ploting the prediction for test images ####################################
########################################################################################################################
def Testing():
    prediction = model.predict(test_x_samples, batch_size=1)
    prediction = np.asarray(prediction)
    for i in range(380):
        damage = np.squeeze(prediction[i], axis=2)
        damage = thresholding(damage,0.18)
        original = np.squeeze(test_x_samples[i], axis=2)
        mask = np.squeeze(tests_y_samples[i], axis=2)
        fig = plt.figure(figsize=(16, 9))
        ax1 = fig.add_subplot(1, 3, 1)
        plt.imshow(damage, cmap=cmap)
        ax2 = fig.add_subplot(1, 3, 2)
        plt.imshow(original, cmap='Greys')
        ax3 = fig.add_subplot(1, 3, 3)
        plt.imshow(mask, cmap='gist_gray')
        ax1.title.set_text('Detected Damage')
        ax2.title.set_text('Original input Image')
        ax3.title.set_text('Ground Truth / Label')
        plt.savefig('E:/aidd_new/aidd/reports/figures/PSPNet/Num/unthreshoding/Fig_' + str(i+1))
        IoU(damage,mask)
        #plt.show()
        plt.close('all')


########################################################################################################################
##################################### paths to folders to save the output Exp images ###################################
########################################################################################################################
path_Vgg16_seg_figuers_folder_no_threshold_exp = 'E:/aidd_new/aidd/reports/figures/VGG_encoder_decoder/Exp/Fig_unthreshold_100_iou_metricc'
path_unet_figuers_folder_no_threshold_exp = 'E:/aidd_new/aidd/reports/figures/UNet/Exp/Fig_unthreshold_100_iou_metricc'
path_fcn_figuers_folder_no_threshold_exp = 'E:/aidd_new/aidd/reports/figures/FCN_DenseNet/Exp/Fig_unthreshold_100_iou_metricc'

Path(path_fcn_figuers_folder_no_threshold_exp).mkdir(parents=True, exist_ok=True) # create folder for the EXP
file_iou_exp = 'exp_iou.csv'
########################################################################################################################
############################################ Ploting the prediction for experimental images ############################
########################################################################################################################
def exp():
    #prediction = model.predict(experimental, batch_size=1)
    #prediction = np.asarray(prediction)
    for j in range(0,1000):
        prediction = model.predict(experimental, batch_size=1)
        prediction = np.asarray(prediction)
        image_number = []  # holds the image number in the loop
        IoU_list = []  # hold the IoU values for certain threshold
        threshold_list = []  # holds the different thresholds for different rounds
        tr = (j + 1) / 1000  # threshold

        #for i in range(23):
        damage = np.squeeze(prediction[6], axis=2)
        damage = thresholding(damage,tr)
        #print(damage)
        #original = np.squeeze(experimental[i], axis=2)
        label = np.squeeze(experimental_label[6], axis=2)
        #plt.imshow(damage, cmap=cmap)
        #plt.axis('off')
        #plt.savefig(path_fcn_figuers_folder_no_threshold_exp+'/Predicted_damage_' + str(i + 1) +'_.png')
        #plt.imshow(original, cmap='Greys')
        #plt.axis('off')
        #plt.savefig(path_fcn_figuers_folder_no_threshold_exp+'/Original_damage_' + str(i + 1) +'_.png')
        #plt.imshow(label, cmap='gist_gray')
        #plt.axis('off')
        #plt.savefig(path_fcn_figuers_folder_no_threshold_exp+'/Mask_' + str(i + 1) +'_.png')
        print(tr)
        iou = IoU(damage,label)
        #plt.savefig(path_Vgg16_seg_figuers_folder_no_threshold_exp + '/Fig_epx' + str(i + 1))
        #plt.show()
        plt.close('all')
        gc.collect()
        threshold_list.append(tr)
        image_number.append(6 + 1)
        IoU_list.append(iou)
        append_list_as_row(file_iou_exp,IoU_list,image_number,threshold_list)




########################################################################################################################
############################################ Running functions  ########################################################
########################################################################################################################
#Testing()
#main_loop()
exp()
########################################################################################################################
gc.collect()

