import cv2
import numpy as np
import os
import gc
import train
import tensorflow as tf
import keras.backend as K
import matplotlib.pyplot as plt
import matplotlib

from tensorflow.keras.models import load_model
#from keras.utils import to_categorical

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from tensorflow.python.client import device_lib
devices = device_lib.list_local_devices()

#   memory growing

gc.collect()

config = tf.compat.v1.ConfigProto(device_count={'GPU': 1, 'CPU': 64})
config.gpu_options.allow_growth = True
config.log_device_placement = True
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

q_replay_memory_size=int(1e4)


# X_train = np.load('exp_data_GFRP_nr6_100kHz_5HC_8Vpp_x20_10avg_110889.npy')
# y_train = np.load('exp_label_Alu_2_77841p_35kHz_5T_x30_moneta.npy')
X_test = np.load('test_data.npy')
y_test = np.load('test_labels.npy')

# X_train = X_train.reshape(X_train.shape[0], 128, 128, 128, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 64, 500, 500, 1).astype('float32')

# y_train = y_train.reshape(y_train.shape[0], 128, 128, 1).astype('float32')
y_test = y_test.reshape(y_test.shape[0], 500, 500, 1).astype('float32')

X_test = X_test / 255.
y_test = y_test / 255.

print(X_test.shape)
print(y_test.shape)

#y_test = to_categorical(y_test)

iou_metric = train.iou_metric

dependencies = {
    'iou_metric': iou_metric
}

# load the saved model
saved_model = load_model('full_WF.h5', custom_objects=dependencies)

prediction = saved_model.predict(X_test[4:8])
y_test = y_test[4:8]

mIoU = 0

def iou(pred, gt):
    ret, pred = cv2.threshold(pred, 0.5, 1.0, cv2.THRESH_BINARY)
    intersection = np.count_nonzero(cv2.bitwise_and(pred, gt))
    union = np.count_nonzero(cv2.bitwise_or(pred, gt))
    IoU = intersection / union
    mIoU = + IoU
    print(IoU)
    return IoU

    mIoU = mIoU / len(y_test)
    print(mIoU)

path_to_csv = 'cmap_jet256.csv'
cmap = matplotlib.colors.ListedColormap(["blue", "green", "red"], name=path_to_csv, N=None)

for i in range(len(prediction)):
    gt = y_test[i]
    pred = prediction[i]
    iou(pred, gt)

    fig, (ax1, ax2) = plt.subplots(1, 2)

    # showing image
    ax1.imshow(gt, cmap = cmap) 
    ax1.set_title("Ground Truth")
    ax1.axis('off')

    # showing image
    ax2.imshow(pred, cmap = cmap)  
    ax2.set_title("Predicted Image")
    ax2.axis('off')

    plt.figure(figsize=(5 / 2.54, 5 / 2.54), dpi=600)
    plt.gca().set_axis_off()
    plt.axis('off')
    plt.subplots_adjust(left=0.0, bottom=0.0, right=1, top=1.0, wspace=0.0, hspace=0.0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    ############################################################################################################
    #plt.imshow(img, cmap=cmap)
    #plt.savefig(i + '_' + '_%d')
    # plt.show()
    plt.close('all')


    #plt.axis('off')
    plt.show()
    plt.close('all')

