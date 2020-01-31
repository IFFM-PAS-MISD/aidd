import cv2
from sklearn.utils import shuffle
# import cv2
import numpy as np
from keras.models import load_model
import gc
import matplotlib.pyplot as plt

# garbage collector
gc.collect()
#####################################
# Loading dataset into x, y arrays and reshape them
#####################################
x = np.load('E:/src/datasets/Segmentation datasets/augemented/Augmented_data_segmentation.npy')
x = x.reshape(1900, 512, 512,1)
y = np.load('E:/src/datasets/Segmentation datasets/augemented/Augmented_target_segmentation.npy')
y = y.reshape(1900, 512, 512,1)
#####################################
experimental = np.load('test_images.npy')
experimental = experimental.reshape(46,512,512,1)
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
model = load_model('FCN_DsensNets_Semantic_Segmentation.h5')
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
for i in range(380):
    prediction = model.predict(test_x_samples, batch_size=1)
    prediction = np.asarray(prediction)
    #####################################
    damage = np.squeeze(prediction[i], axis=2)
    original = np.squeeze(test_x_samples[i], axis=2)
    #original = np.squeeze(experimental[i], axis= 2)
    mask = np.squeeze(tests_y_samples[i], axis=2)
    fig = plt.figure(figsize=(16,9))
    ax1 = fig.add_subplot(1, 3, 1)
    plt.imshow(damage,cmap='tab20c')
    ax2 = fig.add_subplot(1, 3, 2)
    plt.imshow(original,cmap='gist_yarg')
    ax3 = fig.add_subplot(1, 3, 3)
    plt.imshow(mask,cmap='gist_gray')
    ax1.title.set_text('Detected Damage')
    ax2.title.set_text('Original input Image')
    ax3.title.set_text('Ground Truth / Label')
    plt.show()
    #####################################
    #cv2.imshow('Detected  damage', prediction[i])
    #cv2.imshow('GT', tests_y_samples[i])
    #cv2.imshow('original image', test_x_samples[i])
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #####################################
