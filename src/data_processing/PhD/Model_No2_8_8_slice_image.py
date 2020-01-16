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


# reszie and crop images( reszie image to 448*448 and crop the upper right side (224*224),
# and slice the image into 49 tiles

def Resize_Crop_Slice(path):
    global zip
    img_names = glob(path)
    # resizing and cropping
    for fn in img_names:
        images = cv2.imread(fn, 1)
        height = 512
        width = 512
        dim = (height, width)
        images = cv2.resize(images, dim)
        images = images[0:256, 256:512]
        cv2.imwrite(fn, images)

    # slicing the image into 64 tiles
    size = range(np.size(img_names))
    sliced_images = []
    zipped = zip(size, img_names)
    for i, fn in zipped:
        print(fn)
        tiles = image_slicer.slice(fn, 64)
        sliced_images.append(tiles)
        with zipfile.ZipFile('tiles.zip' + str(i), 'w') as zip:
            for tile in tiles:
                with io.BytesIO() as data:
                    tile.save(data)
                    zip.writestr(tile.generate_filename(path=None),
                                 data.getvalue())


# Loading images data and labels from their paths
def load_data(path):
    images_mat = []
    for fn in glob(path):
        image = cv2.imread(fn, 0)
        images_mat.append(image)
    loaded_data = np.asarray(images_mat)
    return loaded_data


path1 = 'E:/Project_DataSet/PhD_PROJECT_DATA/data/raw/num/Dataset_No_2_8_8_slices/Training_images/*_output/RMS_flat_shell_Vz_*_500x500top_*_*.png'

path2 = 'E:/Project_DataSet/PhD_PROJECT_DATA/data/raw/num/Dataset_No_2_8_8_slices/Traning_labels/m_*_delam*_position_no_*_a_*mm_b_*mm_angle_*_*_*.png'

path3 = 'E:/Project_DataSet/PhD_PROJECT_DATA/data/raw/num/Dataset_No_2_8_8_slices/Testing_images/*_output/RMS_flat_shell_Vz_*_500x500top_*_*.png'

path4 = 'E:/Project_DataSet/PhD_PROJECT_DATA/data/raw/num/Dataset_No_2_8_8_slices/Testing_labels/m_*_delam*_position_no_*_a_*mm_b_*mm_angle_*_*_*.png'

#Resize_Crop_Slice(path1)
#Resize_Crop_Slice(path2)
#Resize_Crop_Slice(path3)
#Resize_Crop_Slice(path4)


Training_image = load_data(path1)
print(Training_image.shape)
Training_image = Training_image.reshape(24192, 32, 32, 1)
# Normalizing the input data
Training_image = Training_image.astype('float64')
mean = np.mean(Training_image, axis=0)
Training_image -= mean
std = np.std(Training_image, axis=0)
Training_image /= std

print('Training Images shape :', Training_image.shape)
print(Training_image[1].shape)

Testing_Images = load_data(path3)
print(Testing_Images.shape)
Testing_Images = Testing_Images.reshape(6080, 32, 32, 1)
Testing_Images = Testing_Images.astype('float64')
Testing_Images -= mean
Testing_Images /= std

print('Testing Images shape :', Testing_Images.shape)

Training_Labels = load_data(path2)
print('Training Labels shape :', Training_Labels.shape)

Testing_Labels = load_data(path4)
print('Testing Labels shape :', Testing_Labels.shape)

Label = []
# Labels must be in one and zero, change label to array of 18522*1 dim instead of 18522*32*32
for i in range(0, Training_Labels.shape[0]):
    Label.append(np.max(Training_Labels[i]))

Training_Labels = np.asarray(Label) /255
print(list(enumerate(Training_Labels)), '\n')  # TO view Labels in order
print('Training Labels new shape:', Training_Labels.shape)

Label = []
# Labels must be in one and zero, change label to array of 4655*1 dim instead of 4655*32*32
for i in range(Testing_Labels.shape[0]):
    Label.append(np.max(Testing_Labels[i]))

Testing_Labels = np.asarray(Label) / 255
print(list(enumerate(Testing_Labels)), '\n')
print('Testing Labels new shape:', Testing_Labels.shape)

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
model.fit(Training_image, Training_Labels, batch_size=128, epochs=5)
model.summary()

# validation_data=(Testing_Images, Testing_Labels)
# evaluate the model and print the results
score = model.evaluate(Testing_Images, Testing_Labels, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# save model and architecture to single file
#model.save("model2.h5")
#print("Saved model to disk")


p = 470  # Testing data index
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

print(final.shape)
cv2.imshow('image',final)
cv2.waitKey(0)
cv2.destroyAllWindows()

