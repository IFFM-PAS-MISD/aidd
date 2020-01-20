import cv2
import numpy as np
from keras.models import load_model
import gc
from keras.utils import np_utils
gc.collect()
model = load_model('model_8_8.h5') # loading the Model_7_7
model.summary() # Summarize model
# load dataset
Training_image = np.load('E:/DataSet_aidd/Training_Images_8_8.npy')
Training_image = Training_image.reshape(24192,32,32,1)
Testing_Images = np.load('E:/DataSet_aidd/Testing_Images_8_8.npy')
Testing_Images = Testing_Images.reshape(6208,32,32,1)
Training_Labels = np.load('E:/DataSet_aidd/Training_Labels_8_8.npy')
Testing_Labels = np.load('E:/DataSet_aidd/Testing_Labels_8_8.npy')
# using one-hot-encoding to categorize both training labels and testing labels into two categories
Testing_Labels = np_utils.to_categorical(Testing_Labels, 2)
# evaluate the model
score = model.evaluate(Testing_Images,Testing_Labels, verbose =0)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
# Visualizing the predicted output and compare it to the ground truth image
p = 444  # Testing data index
output = []
predicted_output = model.predict(Testing_Images[64 * (p - 379):64 * (p - 378)])
for i in range(64):
    print('0 undamaged, 1 damaged :', np.argmax(predicted_output[i]))
    output.append(np.argmax(predicted_output[i]))
output = np.asarray(output)
output = output.reshape((8, 8))
print(output.shape)
print(output)
# creating a tensor to store data of the predicted values
final = np.zeros((256, 256))
# rescaling the output array to 256*256,
# by populating final array with output array values 1,0 so it could be plotted.
for j in range(8):
    for k in range(8):
        r=output[j,k]
        if r==0:
            a=np.zeros((32, 32))
        else:
            a=np.ones((32, 32))
        #print(r)
        final[(j)*32:(j)*32+32, (k)*32:(k)*32+32] = a

################
GroundTruthPath = 'E:/Project_DataSet/PhD_PROJECT_DATA/data/raw/images/Dataset_8_8/GT_Labels_updated/m_ ('+str(476-p)+').png'
GroundTruth = cv2.imread(GroundTruthPath, 0)
GroundTruth = np.asarray(GroundTruth)
GroundTruth = GroundTruth / 255
cv2.imshow('GroundTruth',GroundTruth)
cv2.waitKey(0)
cv2.destroyAllWindows()

InterSectionArray = cv2.bitwise_and(final, GroundTruth)
UnionArray = cv2.bitwise_or(final, GroundTruth)
cv2.imshow('Intersection',UnionArray)
cv2.waitKey(0)
cv2.destroyAllWindows()
I = np.count_nonzero(InterSectionArray)
U = np.count_nonzero(UnionArray)
print(I,U)
IoU = I / U
print(" IoU = ",IoU)
print('InterSectionArray',InterSectionArray.shape)
print('UnionArray', UnionArray.shape)
################


test_path = 'E:/Project_DataSet/PhD_PROJECT_DATA/data/raw/images/Dataset_8_8/Testing_images/'+str(p)+'_output/RMS_flat_shell_Vz_'+str(p)+'_500x500top.png'
test_path = str(test_path)
test_image = cv2.imread(test_path,0)
#cv2.imshow('imag',test_image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
test_image = np.asarray(test_image)
print(test_image.shape)
test_image = test_image/255

# Drawing Border around the damage
and_array = np.ones((256,256))
for x in range (255):
    for y in range(255):
        if (final[x,y] == 1) and (final[x-1,y] == 1) and (final[x-1,y] == 1) and (final[x,y+1] ==1) and (final[x,y-1] == 1) \
                and (final[x-1,y-1] == 1) and (final[x-1,y+1] == 1) and (final[x+1,y-1] == 1) and (final[x+1,y+1]):
            and_array[x,y] = 0

final = cv2.bitwise_and(final,and_array)
final_output = cv2.bitwise_or(final,test_image)
cv2.imshow('Local damage', final_output)
cv2.waitKey(0)
cv2.destroyAllWindows()