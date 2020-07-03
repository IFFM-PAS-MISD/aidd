import matplotlib.pyplot as plt
import csv





file1 ='max_iou_unet.csv'
file2 ='average_iou_unet.csv'

file3='max_iou_vgg16.csv'
file4='average_iou_vgg16.csv'

file5='max_iou_fcn_densenet.csv'
file6='averge_iou_fcn_densenet.csv'

def read_csv_file(file):
    x=[]
    y=[]
    with open('E:/backup/IoU_FCNs_Models/'+file, 'r') as csvfile:
        plots= csv.reader(csvfile, delimiter=',')
        for row in plots:
            x.append(float(row[0]))
            y.append(float(row[1]))
    return x,y

x,y = read_csv_file(file1)
plt.plot(x,y, marker='+',label = 'UNet-based model')
x,y = read_csv_file(file3)
plt.plot(x,y, marker='*', label = 'FCN-VGG16')
x,y = read_csv_file(file5)
plt.plot(x,y, marker='o', label = 'FCN-DenseNet')

plt.title('Max IoU')
plt.grid(True)
plt.xlabel('Threshold')
plt.ylabel('IoU')
plt.legend()
plt.show()
