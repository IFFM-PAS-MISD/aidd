from __future__ import absolute_import, division, print_function, unicode_literals

import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import pandas as pd
import xlrd
import xlwt
import xlsxwriter

n = 476 # number of output labels
excel_file = 'E:/TestDataset/raw/num/dataset1_labels_out/output_labels.xlsx'
readings = pd.read_excel(excel_file, index_col= 0)
readings.head()
print(readings.shape)
print(readings.tail(n))

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

tf.debugging.set_log_device_placement(True)

# Create some tensors
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
c = tf.matmul(a, b)

print(c)

