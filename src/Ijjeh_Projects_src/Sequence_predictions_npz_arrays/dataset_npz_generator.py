import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

path = '/home/aijjeh/Desktop/Phd_Project/Full_wavefield_frames_time_series_project/Datasets/Signals_512_448_448/'
os.chdir(path)

signal_array_out = []

for i in range(3):
    folder = 'output_%d' % (i + 1)
    os.chdir(os.path.join(path, folder))
    signal_array = []
    for x in range(3):
        for y in range(3):
            print(i, ':', x, ':', y)
            signal_array.append(np.load('output_%d_signal_%d_%d.npy' % ((i + 1), x, y)))
    signal_array_out.append(signal_array)
for k in range(len(signal_array_out)):
    arr_ = np.stack([signal_array_out], axis=0)
print(arr_.shape)

os.chdir('/home/aijjeh/Desktop/Phd_Project/Sequence_predictions_npz/Dataset_npz')
np.save('Signals_training_set.npy', arr_)
