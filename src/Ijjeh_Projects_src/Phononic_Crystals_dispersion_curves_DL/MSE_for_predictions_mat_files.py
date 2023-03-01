import os
import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy.io import savemat, loadmat
import mat73
from sklearn.metrics import mean_squared_error

os.chdir('/home/aijjeh/Desktop/Phd_Projects/PC_uint_cell_dispersion_curves/num_results/')
file = open('MSE_with_GT_split_frequency.csv', 'w')
writer = csv.writer(file)

for i in range(300):
    os.chdir('/home/aijjeh/Desktop/Phd_Projects/PC_uint_cell_dispersion_curves/num_results/num_results/mat_files')
    img_mat = loadmat('prediction_img_mat_%d.mat' % (i + 6701))
    img_vector = np.zeros((1464, 2))
    img_vector[:, 0] = np.reshape(img_mat['F'], (1464,))
    img_vector[:, 1] = np.reshape(img_mat['K'], (1464,))
    os.chdir('/home/aijjeh/Desktop/Phd_Projects/PC_uint_cell_dispersion_curves/num_results/vectors_inputs/mat_files')
    polygon_mat = loadmat('prediction_polygon_mat_%d.mat' % (i + 6701))
    polygon_vector = np.zeros((1464, 2))
    polygon_vector[:, 0] = np.reshape(polygon_mat['F'], (1464,))
    polygon_vector[:, 1] = np.reshape(polygon_mat['K'], (1464,))

    os.chdir('/aijjeh_odroid_laser/BOHEME/cavities/PC_comsol/')
    gt_mat = mat73.loadmat('out_lines_%d_a8_h3.mat' % (i + 6701))
    gt_vector = np.zeros((1464, 2))
    gt_vector[:, 0] = np.reshape(gt_mat['F'], (1464,))
    gt_vector[:, 1] = np.reshape(gt_mat['K'], (1464,))
    ###########################################################
    img_H_fr = np.zeros((1464, 2))
    img_L_fr = np.zeros((1464, 2))
    for j in range(1464):
        if img_vector[j, 0] > np.max(gt_vector) // 2:
            img_H_fr[j, 0] = img_vector[j, 0]
            img_H_fr[j, 1] = img_vector[j, 1]
        else:
            img_L_fr[j, 0] = img_vector[j, 0]
            img_L_fr[j, 1] = img_vector[j, 1]
    ###########################################################
    ###########################################################
    poly_H_fr = np.zeros((1464, 2))
    poly_L_fr = np.zeros((1464, 2))
    for j in range(1464):
        if polygon_vector[j, 0] > np.max(gt_vector) // 2:
            poly_H_fr[j, 0] = polygon_vector[j, 0]
            poly_H_fr[j, 1] = polygon_vector[j, 1]
        else:
            poly_L_fr[j, 0] = polygon_vector[j, 0]
            poly_L_fr[j, 1] = polygon_vector[j, 1]
        ###########################################################
        ###########################################################
    GT_H_fr = np.zeros((1464, 2))
    GT_L_fr = np.zeros((1464, 2))
    for j in range(1464):
        if gt_vector[j, 0] > np.max(gt_vector) // 2:
            GT_H_fr[j, 0] = gt_vector[j, 0]
            GT_H_fr[j, 1] = gt_vector[j, 1]
        else:
            GT_L_fr[j, 0] = gt_vector[j, 0]
            GT_L_fr[j, 1] = gt_vector[j, 1]
    ###########################################################
    # fig, [ax1, ax2, ax3, ax4, ax5, ax6, ax7] = plt.subplots(nrows=1, ncols=7, figsize=(9, 21))
    # ax1.scatter(img_vector[:, 1], img_vector[:, 0], marker='.')
    # ax4.scatter(img_H_fr[:, 1], img_H_fr[:, 0], marker='.')
    # ax6.scatter(poly_H_fr[:, 1], poly_H_fr[:, 0], marker='.')
    # ax5.scatter(img_L_fr[:, 1], img_L_fr[:, 0], marker='.')
    # ax7.scatter(poly_L_fr[:, 1], poly_L_fr[:, 0], marker='.')
    # ax2.scatter(polygon_vector[:, 1], polygon_vector[:, 0], marker='.')
    # ax3.scatter(gt_vector[:, 1], gt_vector[:, 0], marker='.')
    # plt.show()
    # print('MSE image_input to GT Case:%d' % (i + 6701), mean_squared_error(gt_vector, img_vector))
    # print('MSE polygon_input to GT Case:%d' % (i + 6701), mean_squared_error(gt_vector, polygon_vector))

    data = ['image_High_fr: %d' % (i + 6701), mean_squared_error(GT_H_fr, img_H_fr), 'polygon_High_fr: %d' % (i + 6701), mean_squared_error(GT_H_fr, poly_H_fr), 'image_Low_fr: %d' % (i + 6701), mean_squared_error(GT_L_fr, img_L_fr), 'polygon_Low_fr: %d' % (i + 6701), mean_squared_error(GT_L_fr, poly_L_fr)]
    writer.writerow(data)

file.close()
