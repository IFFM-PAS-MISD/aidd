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