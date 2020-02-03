import cv2
from sklearn.utils import shuffle
# import cv2
import numpy as np
from keras.models import load_model
import gc
import matplotlib.pyplot as plt
from keras import backend as K

import tensorflow as tf

run_opts = tf.RunOptions(report_tensor_allocations_upon_oom=True)

# garbage collector
gc.collect()
#####################################
# Loading dataset into x, y arrays and reshape them
#####################################
x = np.load('E:/backup/datasets/Segmentation datasets/augemented/Augmneted_train_new_data_RMS_flat_shell_bottom.npy')
x = x.reshape(1900, 512, 512, 1)
y = np.load('E:/backup/datasets/Segmentation datasets/augemented/Augmented_target_segmentation.npy')
y = y.reshape(1900, 512, 512, 1)
#####################################
experimental = np.load('Experimental_test_images.npy')
print(experimental.shape)
experimental = experimental.reshape(276, 512, 512, 1)
#####################################
# Shuffle the data set at random
#####################################
x, y = shuffle(x, y)
#####################################
# Split dataset into training and testing sets and again re-shuffle them
#####################################
x_train = x[:1520]
y_train = y[0:1520]
x_train, y_train = shuffle(x_train, y_train)

test_x_samples = x[1520:1900]
tests_y_samples = y[1520:1900]
test_x_samples, tests_y_samples = shuffle(test_x_samples, tests_y_samples)
#####################################
# Loading the model
#####################################
model = load_model('E:/backup/models/FCN_DsensNets_Semantic_Segmentation_filter16_epoch5_kernal(3, 3)_drpout0.2_batch_size_4.h5')
model.summary()
#####################################
# Evaluating the model using test set
#####################################
score = model.evaluate(test_x_samples, tests_y_samples, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
#####################################
# Predicting the output of an image
#####################################
m_IoU = 0
count = 0
#####################################
def Training():
    prediction = model.predict(tests_y_samples, batch_size=1)
    prediction = np.asarray(prediction)
    #####################################
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
        plt.show()
        #####################################

def exp():
    prediction = model.predict(experimental, batch_size=1)
    prediction = np.asarray(prediction)
    #####################################
    for i in range(380):
        damage = np.squeeze(prediction[i], axis=2)
        original = np.squeeze(experimental[i], axis=2)
        fig = plt.figure(figsize=(16, 9))
        ax1 = fig.add_subplot(1, 2, 1)
        plt.imshow(damage, cmap='cool')
        ax2 = fig.add_subplot(1, 2, 2)
        plt.imshow(original, cmap='gist_yarg')
        ax1.title.set_text('Detected Damage')
        ax2.title.set_text('Original input Image')
        plt.show()
        #####################################

#Training()
exp()
gc.collect()