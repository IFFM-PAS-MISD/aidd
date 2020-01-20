import cv2
import numpy as np
from keras.models import load_model
import gc
# garbage collector
from keras.utils import np_utils

gc.collect()

#loading the Model_7_7
model = load_model('model_8_8.h5')
# Summerize model
model.summary()
#load dataset
Training_image = np.load('Training Images_8_8.npy')
Training_image = Training_image.reshape(24192,32,32,1)
Testing_Images = np.load('Testing Images_8_8.npy')
Testing_Images = Testing_Images.reshape(6080,32,32,1)
Training_Labels = np.load('Training Labesl_8_8.npy')
Testing_Labels = np.load('Testing Labels_8_8.npy')

Testing_Labels = np_utils.to_categorical(Testing_Labels, 2)
# evaluate the model
score = model.evaluate(Testing_Images,Testing_Labels, verbose =0)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

p = 445  # Testing data index
output = []
predicted_output = model.predict(Testing_Images[64 * (p - 379):64 * (p - 378)])
for i in range(64):
    print('0 undamaged, 1 damaged :', np.argmax(predicted_output[i]))
    output.append(np.argmax(predicted_output[i]))

output = np.asarray(output)
output = output.reshape((8, 8))
print(output.shape)
print(output)

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

test_path = 'E:/Project_DataSet/PhD_PROJECT_DATA/data/raw/num/Dataset_No_2_8_8_slices/Testing_images/'+str(p)+'_output/RMS_flat_shell_Vz_'+str(p)+'_500x500top.png'
test_path = str(test_path)
test_image = cv2.imread(test_path,0)
test_image = np.asarray(test_image)
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
cv2.imshow('local damage', final_output)
cv2.waitKey(0)
cv2.destroyAllWindows()
