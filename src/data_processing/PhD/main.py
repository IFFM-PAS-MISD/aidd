import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#####################################
from keras import backend as K
import cv2
from keras.models import load_model
from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt
import gc

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
# Shuffle the data set at random
#####################################
x, y = shuffle(x, y)
#####################################
# Split dataset into training and testing sets and again re-shuffle them
#####################################
x_train = x[:1520]
y_train = y[0:1520]
x_train, y_train = shuffle(x_train, y_train)
#####################################
test_x_samples = x[1520:1900]
tests_y_samples = y[1520:1900]
test_x_samples, tests_y_samples = shuffle(test_x_samples, tests_y_samples)

experimental = np.load('Experimental_test_images.npy')
print(experimental.shape)
experimental = experimental / 255.
experimental = experimental.reshape(44, 512, 512, 1)

#####################################
# Loading the model
############################################
model_name = 'E:/backup/models/FCN_DenseNet_models/' \
             'FCN_DsensNets_Semantic_Segmentation_filter_Using_Conv2DTranspose16_epoch_20_kernal_(3, 3)_drpout_0.2_batch_size_4_loss_updated_changed_DB _layer'
model = load_model(model_name + '.h5', compile=False)
model.summary()
#####################################
# Evaluating the model using test set
#####################################
#score = model.evaluate(test_x_samples, tests_y_samples, verbose=0)
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])
######################################
# Predicting the output of an image
#####################################

m_IoU = 0
count = 0


#####################################
def Testing():
    prediction = model.predict(tests_y_samples, batch_size=1)
    prediction = np.asarray(prediction)

    for i in range(50):
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
        plt.savefig('E:/aidd_new/aidd/reports/figures/FCN_DenseNet/Num/FCN_DsensNets_' + str(i))
        # plt.show()


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
        plt.savefig('E:/aidd_new/aidd/reports/figures/FCN_DenseNet/Exp/FCN_DsensNets_' + str(i))
        # plt.show()



Testing()
exp()
gc.collect()
