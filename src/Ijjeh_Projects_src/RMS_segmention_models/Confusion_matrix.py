import csv
import cv2
import os
import numpy as np
from keras.models import load_model
import gc
import matplotlib.pyplot as plt
from keras.utils import np_utils
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

labels_plot = ['Healthy', 'Damaged']
labels = [0, 1]
predictions = []
for i in range(1, 381, 4):
    os.chdir('/home/aijjeh/aijjeh_rexio_share/reports/figures/comparative_study/Thesis/New_2_March_2022/UNet/num/')
    pred = cv2.imread('UNet_num_%d.png' % i, 0)

    pred = np.asarray(pred)
    pred = pred-np.min(pred)
    pred = pred / np.max(pred)
    pred[pred < 0.5] = 0
    pred[pred >= 0.5] = 1
    predictions.append(pred)
    # plt.imshow(pred)
    # plt.show()
predictions = np.asarray(predictions)

for j in range(95):
    os.chdir('/home/aijjeh/aijjeh_rexio_share/reports/figures/comparative_study/Thesis/New_2_March_2022/GT_labels')
    gt = cv2.imread('m1_rand_single_delam_%d.png' % (j + 381), 0)

    gt = np.asarray(gt)
    gt = gt - np.min(gt)
    gt = gt / np.max(gt)

    gt[gt < 0.5] = 0
    gt[gt >= 0.5] = 1

    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(gt)

    ax2 = plt.subplot(1, 2, 2)
    ax2.imshow(predictions[j])
    
    pred_ = (predictions[j].flatten())
    gt = (gt.flatten())

    cm = confusion_matrix(gt, pred_, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_plot)
    disp.plot(cmap='plasma')
    plt.title('prediction_%d' % ((j * 4)+1))
    plt.show()
