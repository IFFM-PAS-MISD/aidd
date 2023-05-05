import os
import numpy as np
import tensorflow as tf
import keras
from tensorflow.python.client import device_lib
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from keras import backend as K
from keras.callbacks import Callback
import mat73
from keras.models import load_model
from sklearn.model_selection import train_test_split

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

os.chdir('/home/aijjeh/Desktop/Phd_Projects/PC_uint_cell_dispersion_curves/dataset/')
x_train = np.load('train_x.npy')
y_train = np.load('train_y_GT_images.npy')

x_test = x_train[450:]
y_test = y_train[450:]

os.chdir('/home/aijjeh/Desktop/Phd_Projects/PC_uint_cell_dispersion_curves/h5_models/')
model_name = 'VAE_ANN_PC_uint_cell_img_to_img.h5'

model = load_model(model_name, compile=False)
model.summary()

os.chdir('/home/aijjeh/Desktop/Phd_Projects/PC_uint_cell_dispersion_curves/num_results/')
prediction = model.predict(x_test, batch_size=1)
# prediction = prediction * 5.2667e5
for i in range(x_test.shape[0]):
    plt.figure(figsize=(4, 12))
    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(y_test[i], label='Reference')
    ax1.set_title('GT')
    # plt.set_ylim([np.min(prediction[:][0]), 500000])

    ax2 = plt.subplot(1, 2, 2)
    ax2.imshow(prediction[i], label='Prediction')
    ax2.legend(['GT', 'Prediction'])
    # ax2.set_title('predictions')
    # ax2.set_ylim([np.min(prediction[:][0]), 500000])

    plt.savefig('predicted_dispersion_cure_img_img_case_%d.png' % (i+451), bbox_inches='tight', transparent="True",
                pad_inches=0)
    # plt.show()
    plt.close()
