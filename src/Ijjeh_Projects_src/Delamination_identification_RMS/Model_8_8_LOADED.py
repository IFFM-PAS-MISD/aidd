import cv2
import numpy as np
from keras.models import load_model
import gc
import matplotlib.pyplot as plt
from PIL import Image
# garbage collector
from keras.utils import np_utils
gc.collect()

#####################################
# Loading the model
#####################################
model = load_model('model_8_8.h5')
model.summary()
#####################################
# Loading dataset into x, y arrays and reshape them
#####################################
Training_image = np.load('E:/src/datasets/bounding box/Training_Images_8_8.npy')
Training_image = Training_image.reshape(24192,32,32,1)
Testing_Images = np.load('E:/src/datasets/bounding box/Testing_Images_8_8.npy')
Testing_Images = Testing_Images.reshape(6208,32,32,1)
Training_Labels = np.load('E:/src/datasets/bounding box/Training_Labels_8_8.npy')
Testing_Labels = np.load('E:/src/datasets/bounding box/Testing_Labels_8_8.npy')
#####################################
# using one-hot-encoding to categorize both training labels and testing labels into two categories
#####################################
Testing_Labels = np_utils.to_categorical(Testing_Labels, 2)
#####################################
# Evaluating the model using test set
#####################################
score = model.evaluate(Testing_Images,Testing_Labels, verbose =0)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
#####################################
# Calculations of Intersection over Union global variable
#####################################
IoU_Total = 0
cout = 0
for i in range (97):
    #####################################
    ## Visualizing the predicted output and compare it to the ground truth image
    #####################################
    p = 379+i  # Testing data index
    output = []
    predicted_output = model.predict(Testing_Images[64 * (p - 379):64 * (p - 378)])
    for i in range(64):
        print('0 undamaged, 1 damaged :', np.argmax(predicted_output[i]))
        output.append(np.argmax(predicted_output[i]))
    output = np.asarray(output)
    output = output.reshape((8, 8))
    print(output.shape)
    print(output)
    #####################################
    # creating an array to store data of the predicted values
    #####################################
    final = np.zeros((256, 256))
    #####################################
    # rescaling the output array to 224*224,
    # by populating final array with output array values 1,0 so it could be plotted.
    #####################################
    for j in range(8):
        for k in range(8):
            r=output[j,k]
            if r==0:
                a=np.zeros((32, 32))
            else:
                a=np.ones((32, 32))
            #print(r)
            final[(j)*32:(j)*32+32, (k)*32:(k)*32+32] = a
    #####################################
    GroundTruthPath = 'E:/Project_DataSet/PhD_PROJECT_DATA/data/raw/images/Dataset_8_8/GT_Labels_updated/m_ ('+str(476-p)+').png'
    GroundTruth = cv2.imread(GroundTruthPath, 0)
    GroundTruth = np.asarray(GroundTruth)
    GroundTruth = GroundTruth / 255
    #####################################
    InterSectionArray = cv2.bitwise_and(final, GroundTruth)
    UnionArray = cv2.bitwise_or(final, GroundTruth)
    #####################################
    I = np.count_nonzero(InterSectionArray)
    U = np.count_nonzero(UnionArray)
    print(I,U)
    IoU = I / U
    if IoU > 0:
        IoU_Total += IoU
        cout += 1
    print(" IoU = ",IoU)
    print('InterSectionArray',InterSectionArray.shape)
    print('UnionArray', UnionArray.shape)
    #####################################
    test_path = 'E:/Project_DataSet/PhD_PROJECT_DATA/data/raw/images/Dataset_8_8/Testing_images/'+str(p)+'_output/RMS_flat_shell_Vz_'+str(p)+'_500x500top.png'
    test_path = str(test_path)
    test_image = cv2.imread(test_path,0)
    #####################################
    test_image = np.asarray(test_image)
    print(test_image.shape)
    test_image = test_image/255
    #####################################
    # Drawing Border around the damage
    #####################################
    and_array = np.ones((256,256))
    for x in range (255):
        for y in range(255):
            if (final[x,y] == 1) and (final[x-1,y] == 1) and (final[x-1,y] == 1) and (final[x,y+1] ==1) and (final[x,y-1] == 1) \
                    and (final[x-1,y-1] == 1) and (final[x-1,y+1] == 1) and (final[x+1,y-1] == 1) and (final[x+1,y+1]):
                and_array[x,y] = 0

    final = cv2.bitwise_and(final,and_array)
    final_output = cv2.bitwise_or(final,test_image)
    #####################################
    plt.imshow(final_output,cmap='gist_yarg')
    plt.title('Localized damage')
    plt.show()

    #####################################
print('Inter section over Union fo all samples = ', IoU_Total/cout)