import numpy as np
from tensorflow.keras.models import load_model

import matplotlib.pyplot as plt
import os

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['CUDA_VISIBLE_DEVICES'] = "2"

X_train = np.load('train_data.npy')
y_train = np.load('train_labels.npy')
X_test = np.load('test_data.npy')
y_test = np.load('test_labels.npy')

X_train = X_train.reshape(X_train.shape[0], 128, 128, 128, 1).astype('float16')
X_test = X_test.reshape(X_test.shape[0], 128, 128, 128, 1).astype('float16')

y_train = y_train.reshape(y_train.shape[0], 128, 128, 1).astype('float16')
y_test = y_test.reshape(y_test.shape[0], 128, 128, 1).astype('float16')

from PIL import Image as im

# load the saved model
saved_model = load_model('full_WF.h5')

# evaluate the model
_, train_acc = saved_model.evaluate(X_train, y_train, verbose=0)
_, test_acc = saved_model.evaluate(X_test, y_test, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

predict = saved_model.predict(X_test)
# print(predict)

# import cv2
# import numpy as np
# color_image = np.zeros((512,512,3),np.uint8)
# bw_image = np.zeros((512,512),np.uint8)
# cv2.imshow("Color Image",color_image)
# cv2.imshow("BW Image",bw_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()