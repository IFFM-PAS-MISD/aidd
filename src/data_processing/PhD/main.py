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

# garbage collector
gc.collect()
#####################################
# Loading dataset into x, y arrays and reshape them
#####################################
x = np.load('E:/backup/datasets/Segmentation datasets/augemented/June_training_dataset.npy')
x = x.reshape(1900, 512, 512, 1)
x = x / 255.0  # normalizing x,y to (0-1) range
y = np.load('E:/backup/datasets/Segmentation datasets/augemented/June_labels.npy')
y = y.reshape(1900, 512, 512, 1)
y = y / 255.0

#####################################
# Shuffle the data set at random
#####################################
#x, y = shuffle(x, y)
#####################################
# Split dataset into training and testing sets and again re-shuffle them
#####################################
x_train = x[:1520]
y_train = y[0:1520]

test_x_samples = x[1520:1900]
tests_y_samples = y[1520:1900]
#####################################
experimental = np.load('Experimental_test_images.npy')
experimental = experimental / 255.0
experimental = experimental.reshape(44, 512, 512, 1)

#####################################
# Loading the model
############################################

#FCN_Dense_net_Precision_BCE_ConvFilter_16_changing_layers_size_  #FCN model
# New_data_unet_adding_dropout_latest_100Epoches    #Unet model

model_name = 'E:/backup/models/SegNet_models/SegNet_Upsampling_added_skip_layer_function.h5'
model = load_model(model_name, compile=False)
#model.summary()

# Organize figures in folders based on the used threshold

path_fcn_figures_src = 'E:/aidd_new/aidd/reports/figures/FCN_DenseNet/Num/FCN_Dense_image_number_'
path_fcn_figures_dst = 'E:/aidd_new/aidd/reports/figures/FCN_DenseNet/Num/Figure_'

path_unet_figuers_src = 'E:/aidd_new/aidd/reports/figures/UNet/Num/Unet_image_number_'
path_uent_figures_dst = 'E:/aidd_new/aidd/reports/figures/UNet/Num/Figure_'


path_segnet_figuers_src = 'E:/aidd_new/aidd/reports/figures/SegNet/Num/Segnet_image_number_'
path_segnet_figures_dst = 'E:/aidd_new/aidd/reports/figures/SegNet/Num/Figure_'



def file_organizer(k,step):

    for i in range(380):
        for j in range(0, k, step):
            z = (j + 1) / k
            shutil.move(path_segnet_figuers_src + str(i + 1) + '_threshold_' + str(z) + '_.png',
            path_segnet_figures_dst + str(i + 1) + '/Segnet_image_number_' + str(i + 1) + '_threshold_' + str(z) + '_.png')
            gc.collect()



# Thresholding function convert the image values to 0 or 1 based on the used threshold
def thresholding(predicted_img, threshold):
    predicted_img[predicted_img >= threshold] = 1
    predicted_img[predicted_img < threshold] = 0
    # nested loop for thresholding
    # for i in range (512):
    #    for j in range(512):
    #        if predicted_img[i,j,0] < threshold:
    #            predicted_img[i,j,0] = 0
    #        else:
    #            predicted_img[i,j,0] =1
    gc.collect()
    return predicted_img

# Calculates the Intersection over Union metric for the predicted and label images using bitwise OR & AND
def IoU(predicted_image, truth_img):
    ######################################
    predicted_image = predicted_image.astype('float64')
    InterSectionArray = cv2.bitwise_and(predicted_image, truth_img)
    UnionArray = cv2.bitwise_or(predicted_image, truth_img)
    I1 = np.count_nonzero(InterSectionArray)
    U = np.count_nonzero(UnionArray)
    IoU1 = I1 / U
    print(IoU1)
    ######################################  nested loop for calculating IoU
    # y_predict = 0
    # y_truth = 0
    # I = 0
    # for i in range (512):
    #    for j in range (512):
    #        if truth_img[i,j,0] == 1:
    #            y_truth = y_truth+1
    #        if predicted_image[i,j,0] ==1:
    #            y_predict = y_predict+1
    #        if predicted_image[i,j,0] == truth_img[i,j,0] == 1:
    #            I = I+1
    # IoU = I/(y_predict+y_truth - I)
    # print('for loop',IoU)
    gc.collect()
    return IoU1

# Saves the IoU values to a csv file

#file_name_FCN = 'IoU_FCN_DenseNets.csv'
#file_name = 'IoU_UNet.csv'
file_name = 'IoU_Segnet.csv'

def append_list_as_row(file_name, list_of_Iou, image_number,threshold_list):
    with open(file_name, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerows([threshold_list])
        writer.writerows([image_number])
        writer.writerows([list_of_Iou])
        gc.collect()

path_fcn_figures_folder= 'E:/aidd_new/aidd/reports/figures/FCN_DenseNet/Num/Figure_'
path_unet_figures_folder = 'E:/aidd_new/aidd/reports/figures/UNet/Num/Figure_'
path_segnet_figuers_folder = 'E:/aidd_new/aidd/reports/figures/SegNet/Num/Figure_'

# Plots the original delamination, predicted and the label figures
def ploting(original,predict,ground,tr,image_number):
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
    #plt.show()

    # creates folder to contain different IoU values for diffferent threshold values for the same sample
    Path(path_segnet_figuers_folder+str(image_number+1)).mkdir(parents=True, exist_ok=True)

    plt.savefig(path_segnet_figuers_src+ str(image_number+1)+'_threshold_'+str(tr)+'_.png') #Threshold_'+str(tr)+'
    plt.close('all')
    gc.collect()


def main_loop():
    # loop for different threshold values
    end = 10 # parameter for the end step of the threshold counter
    step = 5 # counter step
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
            ploting(original, Predict_Img, mask,tr,i)
        append_list_as_row(file_name, IoU_list, image_number,threshold_list)

    file_organizer(end,step)  # sort all images based on their thresholds in folders
    gc.collect()


######################################
# Predicting the output of an image
#####################################

def Testing():
    prediction = model.predict(test_x_samples, batch_size=1)
    prediction = np.asarray(prediction)

    for i in range(380):
        damage = np.squeeze(prediction[i], axis=2)
        original = np.squeeze(test_x_samples[i], axis=2)
        mask = np.squeeze(tests_y_samples[i], axis=2)
        fig = plt.figure(figsize=(16, 9))
        ax1 = fig.add_subplot(1, 3, 1)
        plt.imshow(damage, cmap='tab20c')
        ax2 = fig.add_subplot(1, 3, 2)
        plt.imshow(original, cmap='gist_yarg')
        ax3 = fig.add_subplot(1, 3, 3)
        plt.imshow(mask, cmap='gist_gray')
        ax1.title.set_text('Detected Damage')
        ax2.title.set_text('Original input Image')
        ax3.title.set_text('Ground Truth / Label')
        plt.savefig('E:/aidd_new/aidd/reports/figures/SegNet/Num/SegNet_' + str(i+1))
        #plt.show()
        plt.close('all')



def exp():
    prediction = model.predict(experimental, batch_size=1)
    prediction = np.asarray(prediction)

    for i in range(44):
        damage = np.squeeze(prediction[i], axis=2)
        original = np.squeeze(experimental[i], axis=2)
        fig = plt.figure(figsize=(16, 9))
        ax1 = fig.add_subplot(1, 3, 1)
        plt.imshow(damage, cmap='cool')
        ax2 = fig.add_subplot(1, 3, 2)
        plt.imshow(original, cmap='gist_yarg')
        ax3 = fig.add_subplot(1, 3, 3)
        plt.imshow(original, cmap='gist_yarg')
        plt.imshow(damage, alpha=.65, cmap='gist_yarg')
        ax3.title.set_text('Original Image with mask')
        ax1.title.set_text('Detected Damage')
        ax2.title.set_text('Original input Image')
        plt.savefig('E:/aidd_new/aidd/reports/figures/SegNet/Exp/SegNet_' + str(i+1))
        #plt.show()
        plt.close('all')
        gc.collect()




#Testing()
#main_loop()
#exp()
gc.collect()

