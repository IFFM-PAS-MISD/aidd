import os

import matplotlib.pyplot as plt
import numpy as np
import cv2
import scipy.io

os.chdir(
    '/home/aijjeh/Desktop/Phd_Projects/Sequence_prediction/Full_wavefield_frames_time_series_project/Datasets/label_set')
file_frame = np.load('frames_initial.npy')


def gen_delamination_GT():
    os.chdir('/aijjeh_odroid_sensors/aidd/data/raw/num/dataset2_labels_out')
    temp_origin = np.zeros((475, 256, 256), dtype="float16")
    temp_90 = np.zeros((475, 256, 256), dtype="float16")
    temp_180 = np.zeros((475, 256, 256), dtype="float16")
    temp_270 = np.zeros((475, 256, 256), dtype="float16")
    for i in range(1, 476):
        print(i)
        img = cv2.imread('m1_rand_single_delam_%d.png' % i, 0)
        img = img / 255.0
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
        temp_origin[i - 1] = img
        img_90 = np.rot90(img, k=1)
        temp_90[i - 1] = img_90
        img_180 = np.rot90(img, k=2)
        temp_180[i - 1] = img_180
        img_270 = np.rot90(img, k=3)
        temp_270[i - 1] = img_270
    temp = np.concatenate([temp_origin, temp_90, temp_180, temp_270], axis=0)
    temp = np.expand_dims(temp, axis=1)
    temp = np.repeat(temp, 32, axis=1)
    os.chdir('/home/aijjeh/Desktop/Phd_Projects/Modeling_guided_waves/Model_1_2_3_datasets/Model_2_dataset')
    return temp


def gen_healthy_ref():
    temp = np.zeros((475, 32, 256, 256), dtype="float16")
    os.chdir('/aijjeh_odroid_sensors/aidd/data/raw/num/wavefield_dataset_undelam_bottom_out/1_output')

    for i in range(475):
        count = 0
        for j in range((file_frame[i] - 6), (file_frame[i] + 26)):
            print(i, j)
            img = cv2.imread('%d_flat_shell_Vz_1_500x500bottom.png' % j, 0)
            img = img / 255.0
            img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
            temp[i][count] = img
            count += 1
    os.chdir('/home/aijjeh/Desktop/Phd_Projects/Modeling_guided_waves/Model_1_2_3_datasets/Model_2_dataset')
    return temp


def pred_skip_connection_of_model_1():
    exp = 7
    for i in range(6):
        os.chdir('/home/aijjeh/Desktop/Phd_Projects/Modeling_guided_waves/Models_1_2_3_predictions/Model_1_predictions')
        temp = np.zeros((475, 32, 2 ** exp, 2 ** exp, 16 * (i + 1)), dtype=np.float16)
        for j in range(475):
            print(i, j)
            mat = scipy.io.loadmat('skip_connection_%i_%d.mat' % (i, j))
            temp[i] = mat['skip_connection_%d' % i]
        exp -= 1
        os.chdir('/home/aijjeh/Desktop/Phd_Projects/Modeling_guided_waves/Model_1_2_3_datasets/Model_2_dataset')
        np.save('predicted_skip_connection_%d' % i, temp)
    return


def pred_LS_of_model_1():
    os.chdir('/home/aijjeh/Desktop/Phd_Projects/Modeling_guided_waves/Models_1_2_3_predictions/Model_1_predictions')
    temp = np.zeros((475, 32, 64, 64, 32), dtype=np.float16)
    for i in range(475):
        print(i)
        mat = scipy.io.loadmat('encoded_latent_space_%d.mat' % i)
        temp[i] = mat['encoded_latent_space']
    os.chdir('/home/aijjeh/Desktop/Phd_Projects/Modeling_guided_waves/Model_1_2_3_datasets/Model_2_dataset')
    return temp


def pred_skip_connection_of_model_2():
    exp = 7
    for i in range(6):
        os.chdir('/home/aijjeh/Desktop/Phd_Projects/Modeling_guided_waves/Models_1_2_3_predictions/Model_2_predictions')
        temp = np.zeros((475, 32, 2 ** exp, 2 ** exp, 16 * (i + 1)), dtype=np.float16)
        for j in range(475):
            print(i, j)
            mat = scipy.io.loadmat('skip_connection_%i_%d.mat' % (i, j))
            temp[i] = mat['skip_connection_%d' % i]
        exp -= 1
        os.chdir('/home/aijjeh/Desktop/Phd_Projects/Modeling_guided_waves/Model_1_2_3_datasets/Model_3_dataset')
        np.save('predicted_skip_connection_%d_model_2' % i, temp)
    return


def pred_LS_of_model_2():
    os.chdir('/home/aijjeh/Desktop/Phd_Projects/Modeling_guided_waves/Models_1_2_3_predictions/Model_2_predictions')
    temp = np.zeros((475, 32, 4, 4, 128), dtype=np.float16)
    for i in range(475):
        print(i)
        mat = scipy.io.loadmat('output_%d.mat' % i)
        temp[i] = mat['output']
    os.chdir('/home/aijjeh/Desktop/Phd_Projects/Modeling_guided_waves/Model_1_2_3_datasets/Model_3_dataset')
    return temp


def den_model_3_sample():
    os.chdir('/home/aijjeh/Desktop/Phd_Projects/Modeling_guided_waves/Model_2_mapping_GT_to_LS/predictions')
    temp = np.zeros((475, 12, 1, 1, 80))

    for i in range(475):
        mat = scipy.io.loadmat('mapped_latenet_space_%d.mat' % i)
        temp[i] = mat['LS']
    os.chdir('/home/aijjeh/Desktop/Phd_Projects/Modeling_guided_waves/Model_2_mapping_GT_to_LS')
    return temp


# def input_model_3_LS_skips():
#     os.chdir('/home/aijjeh/Desktop/Phd_Projects/Modeling_guided_waves/Model_2_mapping_GT_to_LS/Predictions_model_2')
#     temp = np.zeros((475, 960))
#     for i in range(475):
#         print(i)
#         mat = scipy.io.loadmat('latent_space_%d.mat' % i)
#         temp[i] = mat['latent_space']
#     np.save('latent_space', temp)
#
#     exp = 8
#     for i in range(6):
#         temp = np.zeros((475, 12, 2 ** exp, 2 ** exp, 8 * (i + 1)), dtype=np.float16)
#         for j in range(475):
#             print(i, j)
#             mat = scipy.io.loadmat('skip_connection_%i_%d.mat' % (i, j))
#             temp[i] = mat['skip_connection_%d' % i]
#         exp -= 1
#         np.save('skip_connection_%d' % i, temp)


def get_full_wavefields_with_damage():
    temp_origin = np.zeros((475, 32, 256, 256), dtype="float16")
    temp_90 = np.zeros((475, 32, 256, 256), dtype="float16")
    temp_180 = np.zeros((475, 32, 256, 256), dtype="float16")
    temp_270 = np.zeros((475, 32, 256, 256), dtype="float16")
    for i in range(475):
        os.chdir('/aijjeh_odroid_sensors/aidd/data/raw/num/wavefield_dataset2_bottom_out/%d_output' % (i + 1))
        count = 0
        for j in range((file_frame[i] - 6), (file_frame[i] + 26)):
            print(i, j)
            img = cv2.imread('%d_flat_shell_Vz_%d_500x500bottom.png' % (j, (i + 1)), 0)
            img = img / 255.0
            img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
            temp_origin[i][count] = img
            img_90 = np.rot90(img, k=1)
            temp_90[i][count] = img_90
            img_180 = np.rot90(img, k=2)
            temp_180[i][count] = img_180
            img_270 = np.rot90(img, k=3)
            temp_270[i][count] = img_270

            count += 1
    os.chdir('/home/aijjeh/Desktop/Phd_Projects/Modeling_guided_waves/Model_1_2_3_datasets/Model_1_dataset')
    temp = np.concatenate([temp_origin, temp_90, temp_180, temp_270], axis=0)
    return temp


if __name__ == "__main__":
    # np.save('predicted_Latent_space', pred_LS_of_model_1())
    # pred_skip_connection_of_model_1()

    # np.save('predicted_Latent_space_model_2', pred_LS_of_model_2())
    # pred_skip_connection_of_model_2()

    # np.save('delamination_ground_truths', gen_delamination_GT())

    # np.save('health_full_wave_fields', gen_healthy_ref())

    # np.save('samples_model_3', samples_model_3())

    # np.save('GT_full_wavefields', get_full_wavefields_with_damage())

    os.chdir('/home/aijjeh/Desktop/Phd_Projects/Modeling_guided_waves/Model_1_2_3_datasets/Model_2_dataset')
    temp = np.load('health_full_wave_fields.npy', mmap_mode='r')
    temp_90 = np.rot90(temp, k=1, axes=(2, 3))
    temp_180 = np.rot90(temp, k=2, axes=(2, 3))
    temp_270 = np.rot90(temp, k=3, axes=(2, 3))
    temp_total = np.concatenate([temp, temp_90, temp_180, temp_270], axis=0)
    np.save('health_full_wave_fields', temp_total)
