import csv
import gc
import cv2
from tensorflow import keras
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import neptune
from tensorflow.python.client import device_lib
from PIL import Image
import tensorflow as tf
import os
from keras.layers import LeakyReLU
from pathlib import Path
from decouple import config
import matplotlib
from matplotlib import cm
import itertools
from pathlib import Path
from scipy.io import savemat
import scipy.io
import numpy as np

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
# print(device_lib.list_local_devices())

os.chdir('/home/aijjeh/Desktop/Phd_Projects/compressive_sensing_project')
mat = scipy.io.loadmat('CNNpreparedData_normalizedInputs.mat')
x_test1 = mat['inputs_testing']
x_test1 = np.transpose(x_test1, [0, -1, 1])
x_test2 = mat['inputs_testing_HS1']
x_test2 = np.transpose(x_test2, [0, -1, 1])

x_test = x_test1 / (np.abs(x_test2) + .001)
for i in range(x_test2.shape[0]):
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(x_test1[i])
    ax2 = plt.subplot(3, 1, 2)
    ax2.plot(x_test2[i])
    ax3 = plt.subplot(3, 1, 3) 
    ax3.plot(x_test[i])
    plt.show()
exit()
print(x_test.shape)
y_test = mat['S1_testing']
print(y_test.shape)

os.chdir('/home/aijjeh/Desktop/Phd_Projects/compressive_sensing_project/signal_prediction/h5_models/')

model_name = 'ann_model_signal_prediction_VAE_6.h5'  # ann_model_signal_prediction.h5
model = load_model(model_name, compile=False)
model.summary()

prediction = model.predict(x_test, batch_size=1)
prediction = np.asarray(prediction)
savemat('prediction_CNN_model_6.mat', {'pred': prediction})
plt.figure(figsize=(20, 15))
for i in range(x_test.shape[0]):
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(y_test[i])
    ax1.set_title('GT')
    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(prediction[i], label='Prediction')
    ax2.set_title('predictions')
plt.show()

os.chdir('/home/aijjeh/Desktop/Phd_Projects/compressive_sensing_project/signal_prediction/')
plt.savefig('CNN_predicted_output_')
