import itertools
import numpy as np
import os
import cv2
import csv
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, f1_score
from sklearn.metrics import ConfusionMatrixDisplay

labels = [0, 1]
arr_size = 512
labels_plot = ['undamaged', 'damaged']
mat = (np.random.randint(0, len(labels), (arr_size, arr_size), dtype=np.uint))
mat = mat.flatten()
GT = (np.random.randint(0, len(labels), (arr_size, arr_size), dtype=np.uint))
GT = GT.flatten()
f1score = f1_score(GT, mat, labels=labels, average='macro')

print(mat)
print(GT)
mcm = multilabel_confusion_matrix(GT, mat, labels=labels)
cm = confusion_matrix(GT, mat, labels=labels)

print('multi cm\n', mcm)
print('sum ', sum(sum(mcm[0, :, :])))
print(np.flip(mcm[0]))
print(np.flip(mcm[1]))

print('-----------------------------------------------------------')
print('confusion matrix \n', cm)
print(cm[0, 0] + cm[1, 1])
print('CM ', cm.shape)

print('f1 score ', f1score)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_plot)
disp.plot()
plt.show()


def plot_confusion_matrix(con_mat, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.winter):
    plt.imshow(con_mat, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        con_mat = con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = con_mat.max() / 2.
    for i, j in itertools.product(range(con_mat.shape[0]), range(con_mat.shape[1])):
        plt.text(j, i, con_mat[i, j],
                 horizontalalignment="center",
                 color="white" if con_mat[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


plot_confusion_matrix(cm, labels_plot)
