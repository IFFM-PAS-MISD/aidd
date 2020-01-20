import io
import zipfile
from glob import glob
import cv2
import image_slicer
import numpy as np
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from keras.models import Sequential
from keras.optimizers import RMSprop, adam
from keras.utils import np_utils
import matplotlib.pyplot as plt
import gc
# garbage collector
gc.collect()

Training_image = np.load('Training_Images_8_8.npy')
Training_image = Training_image.reshape(24192,32,32,1)
Testing_Images = np.load('Testing_Images_8_8.npy')
Testing_Images = Testing_Images.reshape(6208,32,32,1)
Training_Labels = np.load('Training_Labels_8_8.npy')
Testing_Labels = np.load('Testing_Labels_8_8.npy')

# resize and crop images( resize image to 448*448 and crop the upper right side (224*224),
# and slice the image into 49 tiles

# using one-hot-encoding to categorize both training labels and testing labels into two categories
Training_Labels = np_utils.to_categorical(Training_Labels, 2)
Testing_Labels = np_utils.to_categorical(Testing_Labels, 2)

# Configuring the model
model = Sequential()
First_convolution_layer = model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 1)))
First_Pooling_Layer = model.add(MaxPool2D((2, 2), strides=2))
Second_convolution_layer = model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
Second_Pooling_Layer = model.add(MaxPool2D((2, 2), strides=2))
Flatting_Layer = model.add(Flatten())
First_Dropout_layer = model.add(Dropout(0.5))
First_FC_layer = model.add(Dense(4096, activation='relu'))
Second_Dropout_layer = model.add(Dropout(0.5))
Second_FC_layer = model.add(Dense(1024, activation='relu'))
Final_output = model.add(Dense(2))  # we did not use an activation function

# Compile model using accuracy to measure model performance

model.compile(
    optimizer=adam(lr=0.001),
    loss='mse',
    metrics=['accuracy'])

# Training the model
model.fit(Training_image, Training_Labels, batch_size=128, epochs=10)
model.summary()

# validation_data=(Testing_Images, Testing_Labels)
# evaluate the model and print the results
score = model.evaluate(Testing_Images, Testing_Labels, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

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
# save model and architecture to single file
model.save("model_8_8.h5")
print("Saved model to disk")

