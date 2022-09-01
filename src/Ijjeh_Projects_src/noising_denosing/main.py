# ============================================
__title__ = 'semantic segmentation call function'
__author__ = "Abdalraheem A. Ijjeh"
__maintainer__ = "Abdalraheem A. Ijjeh"
__email__ = "aijjeh@imp.gda.pl"

# ============================================
import csv
import gc
import cv2
import matplotlib
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import neptune
from tensorflow.python.client import device_lib
from PIL import Image
import os
from decouple import config

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

print(device_lib.list_local_devices())


########################################################################################################################
# link to neptune ao
########################################################################################################################
# neptune.init(project_qualified_name='abdalraheem.ijjeh/aidd', api_token=config('NEPTUNE_API_TOKEN'))
# neptune.create_experiment('ESPNet-model-testing')
# neptune.append_tag('Testing numerical samples')


########################################################################################################################
# load dataset
########################################################################################################################
def load_dataset():
    # os.chdir('/home/aijjeh/Desktop/Phd_Project/Full_wavefield_frames_time_series_project/Datasets/')
    # training_set = np.load('training_set/training_consecutive_448_30_consecutive_frames_not_normalised.npy',
    # mmap_mode='r')
    # training_set = training_set.reshape((475, 30, 448, 448, 1))
    # training_set = training_set.astype('float32')
    # training_set = training_set / 255.0
    # labels = np.load('label_set/GT_labels_thresholded_448_only_475_labels.npy')
    # labels = labels.reshape((475, 448, 448, 1))
    # labels = labels.astype('float32')
    # train_x = training_set[380:]
    # train_label = labels[380:]
    # # train_label = to_categorical(train_label, 2)
    # return train_x, train_label
    os.chdir('/home/aijjeh/Desktop/Phd_Project/Full_wavefield_frames_time_series_project/Datasets/')

    training_set = np.load('training_set/training_consecutive_448_30_consecutive_frames_not_normalised.npy',
                           mmap_mode='r')
    # training_set = np.load('training_set/training_consecutive_128_64_consecutive_frames_normalised_100kHz.npy',
    #                        mmap_mode='r')
    # training_set = np.load('training_set/CFRP_teflon_3o_375_375p_50kHz_5HC_x12_15Vpp_exp_sample.npy')
    training_set = training_set.reshape((475, 30, 448, 448, 1))
    training_set = training_set.astype('float32')
    training_set = training_set / 255.0

    labels = np.load('label_set/GT_labels_thresholded_448_only_475_labels.npy')
    # labels = np.load('label_set/GT_labels_thresholded_128_only_475_labels_100kHz.npy')

    # labels = np.load('label_set/label_CFRP_teflon_3o_375_375p_50kHz_5HC_x12_15Vpp.png.npy')
    labels = labels.reshape((475, 448, 448, 1))
    labels = labels.astype('float32')

    Test_x = training_set[380:]
    Test_y = labels[380:]
    return Test_x, Test_y


########################################################################################################################
# load model
########################################################################################################################

os.chdir('/home/aijjeh/Desktop/Phd_Project/Upscaling_downscaling_denoising')
model_name = 'h5_models/AE_time_distributed_filters_16_depth_3_kernel_5_50kHz_softmax.h5'
model = load_model(model_name, compile=False)
model.summary()

########################################################################################################################
# load cmap color
########################################################################################################################

path_to_csv = '/home/aijjeh/aijjeh_rexio_share/PhD/cmap_flipped_jet256.csv'
cmap = matplotlib.colors.ListedColormap(["blue", "green", "red"], name=path_to_csv, N=None)

########################################################################################################################
# output as csv file
########################################################################################################################
csv_files = 'csv_files'


def append_list_as_row(cvs_file_name, list_of_iou, image_num):
    os.chdir('/home/aijjeh/Desktop/Phd_Project/Upscaling_downscaling_denoising/csv_files')
    with open(cvs_file_name, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerows([image_num])
        writer.writerows([list_of_iou])
        gc.collect()


########################################################################################################################
# calculate IoU
########################################################################################################################

def calc_IoU(predicted_image, truth_img):
    predicted_image = predicted_image.astype('float64')
    # ret, predicted_image = cv2.threshold(predicted_image, 0.49, 1, cv2.THRESH_BINARY)
    predicted_image = np.where(predicted_image > .49, 1, 0)  # [0 if i_ > 0.5 else 1 for i_ in predicted_image]
    predicted_image = predicted_image.astype('float64')

    truth_img = truth_img.astype('float64')
    InterSectionArray = cv2.bitwise_and(predicted_image, truth_img)
    UnionArray = cv2.bitwise_or(predicted_image, truth_img)
    I1 = np.count_nonzero(InterSectionArray)
    U = np.count_nonzero(UnionArray)
    IoU1 = I1 / U
    gc.collect()
    return IoU1


########################################################################################################################
# testing function
########################################################################################################################

image_number = []  # holds the image number in the loop
IoU_list = []  # hold the IoU values for certain threshold


def Test_sample():
    Test_x, Test_y = load_dataset()
    print(Test_x.shape)
    prediction = model.predict(Test_x, batch_size=1)
    print('whole prediction shape ', prediction.shape)
    mean_iou = 0
    for i in range(len(Test_x)):
        damage = prediction[i]
        label = Test_y[i]
        # print('prediction shape before argmax', damage.shape)
        # damage = np.argmax(damage, axis=2)
        # print('prediction shape after argmax', damage.shape)
        ############################################################################################################
        # plt.figure(figsize=(5 / 2.54, 5 / 2.54), dpi=600)
        # plt.gca().set_axis_off()
        # plt.axis('off')
        # plt.subplots_adjust(left=0.0, bottom=0.0, right=1, top=1.0, wspace=0.0, hspace=0.0)
        # plt.margins(0, 0)
        # plt.gca().xaxis.set_major_locator(plt.NullLocator())
        # plt.gca().yaxis.set_major_locator(plt.NullLocator())
        # ############################################################################################################
        plt.figure(figsize=(10, 5))
        ax = plt.subplot(1, 2, 1)
        ax.imshow(damage, cmap=cmap)
        ax.set_title('Prediction')
        plt.axis('off')
        ax = plt.subplot(1, 2, 2)
        ax.set_title('Ground truth')
        ax.imshow(label)
        plt.axis('off')
        image_number.append(i + 380)
        os.chdir('/home/aijjeh/Desktop/Phd_Project/Upscaling_downscaling_denoising/num_res/')
        plt.savefig('predicted_%d' % (i + 381))
        # plt.show()
        plt.close('all')

        iou = calc_IoU(damage, label)
        IoU_list.append(iou)
        print('IoU', iou, 'number_%d' % (i + 381))
        mean_iou = mean_iou + iou
    append_list_as_row('RNN_UNet_IoU.csv', IoU_list, image_number)
    print('mean iou = ', mean_iou / len(Test_x))


########################################################################################################################
# main function
########################################################################################################################

if __name__ == '__main__':
    Test_sample()
