import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#####################################
import cv2
from keras.models import load_model
from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt
import csv
import gc

# garbage collector

gc.collect()
#####################################
# Loading dataset into x, y arrays and reshape them
#####################################
x = np.load('E:/backup/datasets/Segmentation datasets/augemented/June_training_dataset.npy')
x = x.reshape(1900, 512, 512, 1)
x = x / 255.0 # normalizing x,y to (0-1) range
y = np.load('E:/backup/datasets/Segmentation datasets/augemented/June_labels.npy')
y = y.reshape(1900, 512, 512, 1)
y= y / 255.0

#####################################
# Shuffle the data set at random
#####################################
x, y = shuffle(x, y)
#####################################
# Split dataset into training and testing sets and again re-shuffle them
#####################################
x_train = x[:1520]
y_train = y[0:1520]
#####################################
test_x_samples = x[1520:1900]
tests_y_samples = y[1520:1900]
test_x_samples, tests_y_samples = shuffle(test_x_samples, tests_y_samples)

experimental = np.load('Experimental_test_images.npy')
print(experimental.shape)
experimental = experimental / 255.0
experimental = experimental.reshape(44, 512, 512, 1)

#####################################
# Loading the model
############################################
model_name = 'New_data_unet_adding_dropout_latest.h5'
model = load_model(model_name, compile=False)
#######################################
layer_outputs = [layer.output for layer in model.layers[:]]
#activation_model  = models.Model(input = model.input, outputs= layer_outputs)
activation = model.predict(test_x_samples, batch_size=1)
last_layer_activation = activation[:]
print(last_layer_activation.shape)

image_number =[]
IoU_list = []
for i in range(380):
    intersection = 0
    y_pred = 0
    y_true = 0
    for j in range(512):
        for k in range(512):
            for l in range(1):
                if tests_y_samples[i,j,k,l] == 1:
                    y_true =y_true+1
                if last_layer_activation[i,j,k,l] < 0.2:
                    last_layer_activation[i,j,k,l] =0
                else:
                    last_layer_activation[i,j,k,l] =1
                    y_pred = y_pred+1
                if last_layer_activation[i,j,k,l] == tests_y_samples[i,j,k,l] ==1:
                    intersection = intersection+1

    #print('intersection',intersection)
    #print("y pred", y_pred)
    #print('y true', y_true)
    union = abs(y_pred + y_true - intersection)
    IoU = intersection / union
    #print("IoU", IoU)
    #print(i)
    #fig = plt.figure(figsize=(16, 9))
    #original = np.squeeze(test_x_samples[i], axis=2)
    #mask = np.squeeze(tests_y_samples[i], axis=2)
    #
    #plt.subplot(131)
    #plt.imshow(original, cmap='gist_yarg')
    #plt.subplot(132)
    #plt.imshow(last_layer_activation[i, :, :, 0], cmap='viridis')
    #plt.subplot(133)
    #plt.imshow(mask, cmap='gist_yarg')
    #
    #plt.show()
    image_number.append(i+1)
    IoU_list.append(IoU)

#model.summary()

with open('IoU_UNet.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows([image_number])
    writer.writerows([IoU_list])

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
        plt.savefig('E:/aidd_new/aidd/reports/figures/FCN_DenseNet/Num/FCN_Dense_' + str(i))
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
        plt.savefig('E:/aidd_new/aidd/reports/figures/FCN_DenseNet/Exp/FCN_Dense_' + str(i))
        #plt.show()



#Testing()
#exp()
gc.collect()
