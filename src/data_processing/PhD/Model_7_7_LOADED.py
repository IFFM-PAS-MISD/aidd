import cv2
import numpy as np
from keras.models import load_model
import gc
import matplotlib.pyplot as plt
from PIL import Image
# garbage collector
from keras.utils import np_utils
gc.collect()

#loading the Model_7_7
model = load_model('E:/src/model_7_7.h5')
# Summerize model
model.summary()

#load dataset
Training_image = np.load('E:/DataSet_aidd/Training_Images_7_7.npy')
Training_image = Training_image.reshape(18522,32,32,1)
Testing_Images = np.load('E:/DataSet_aidd/Testing_Images_7_7.npy')
Testing_Images = Testing_Images.reshape(4753,32,32,1)
Training_Labels = np.load('E:/DataSet_aidd/Training_Labels_7_7.npy')
Testing_Labels = np.load('E:/DataSet_aidd/Testing_Labels_7_7.npy')
# using one-hot-encoding to categorize both training labels and testing labels into two categories
Testing_Labels = np_utils.to_categorical(Testing_Labels, 2)
# evaluate the model
score = model.evaluate(Testing_Images,Testing_Labels, verbose =0)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

IoU_Total = 0
cout = 0
for i in range (97):
    ## Visualizing the predicted output and compare it to the ground truth image
    p = 379+i  # Testing data index
    output = []
    predicted_output = model.predict(Testing_Images[49 * (p - 379):49 * (p - 378)])
    for i in range(49):
        output.append(np.argmax(predicted_output[i]))

    output = np.asarray(output)
    output = output.reshape((7, 7))
    print(output.shape)
    print(output)
    # creating a tensor to store data of the predicted values
    final = np.zeros((224, 224))
    # rescaling the output array to 224*224,
    # by populating final array with output array values 1,0 so it could be plotted.
    for j in range(7):
        for k in range(7):
            r = output[j, k]
            if r == 0:
                a = np.zeros((32, 32))
            else:
                a = np.ones((32, 32))
            final[(j)*32: (j)*32+32, (k)*32: (k)*32+32] = a

    ################
    GroundTruthPath = 'E:/Project_DataSet/PhD_PROJECT_DATA/data/raw/images/Dataset_7_7/GT_Labels_updated/m_ ('+str(476-p)+').png'
    GroundTruth = cv2.imread(GroundTruthPath, 0)
    GroundTruth = np.asarray(GroundTruth)
    GroundTruth = GroundTruth / 255
    #cv2.imshow('Ground Truth',GroundTruth)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    InterSectionArray = cv2.bitwise_and(final, GroundTruth)
    UnionArray = cv2.bitwise_or(final, GroundTruth)
    #cv2.imshow('Intersect',UnionArray)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    I = np.count_nonzero(InterSectionArray)
    U = np.count_nonzero(UnionArray)
    print(I,U)
    IoU = I / U
    if IoU>0:
        IoU_Total += IoU
        cout +=1
    print(" IoU = ",IoU)
    print('InterSectionArray',InterSectionArray.shape)
    print('UnionArray', UnionArray.shape)


    ################


    path = 'E:/Project_DataSet/PhD_PROJECT_DATA/data/raw/images/Dataset_7_7/Testing_images/'+str(p)+'_output/RMS_flat_shell_Vz_'+str(p)+'_500x500top.png'
    #path = str(path)
    Test_image = cv2.imread(path, 0)
    Test_image = np.asarray(Test_image)
    print('shape of the test sample',Test_image.shape)
    Test_image = Test_image / 255
    # Drawing Border around the damage
    and_array = np.ones((224,224))
    for x in range (223):
        for y in range(223):
            if (final[x,y] == 1) and (final[x-1,y] == 1) and (final[x-1,y] == 1) and (final[x,y+1] ==1) and (final[x,y-1] == 1) \
                    and (final[x-1,y-1] == 1) and (final[x-1,y+1] == 1) and (final[x+1,y-1] == 1) and (final[x+1,y+1]):
                and_array[x,y] = 0

    final = cv2.bitwise_and(final,and_array)
    final_output = cv2.bitwise_or(final, Test_image)

    cv2.imshow('Local damage', final_output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #Display_Image = Image.fromarray(final_output)
    #plt.imshow(final_output,cmap='gist_yarg', interpolation='nearest')
    #plt.show()
    imagepath ='E:/aidd/data/processed/model_7_7/'+str(p)+'.png'
    #plt.savefig(imagepath)
    cv2.imwrite(imagepath, final_output)

print('Inter section over Union fo all samples = ', IoU_Total/cout)