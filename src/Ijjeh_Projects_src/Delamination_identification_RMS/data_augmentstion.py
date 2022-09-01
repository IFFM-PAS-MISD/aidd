import numpy as np
import cv2
from glob import glob
import gc

# Force the Garbage Collector to release unreferenced memory
gc.collect()
#####################################
# mirroring of quarters of images
#####################################
def Data_Augmentation(path):
    img_names = glob(path)
    Augmented = []
    for fn in img_names:
        print(fn)
        images = cv2.imread(fn, 0)
        images = cv2.resize(images, (512, 512))
        #horizontal_img = cv2.flip(images, 0)
        #vertical_img = cv2.flip(images, 1)
        #diagonal_img = cv2.flip(horizontal_img,1)
        Augmented.append(images)
        #Augmented.append(horizontal_img)
        #Augmented.append(vertical_img)
        #Augmented.append(diagonal_img)
        loaded_data = np.asarray(Augmented)
    return loaded_data


path1 = 'E:/Project_DataSet/PhD_PROJECT_DATA/data/raw/exp/*.png'
x = Data_Augmentation(path1)

print(x.shape)
#print(y.shape)

np.save("Experimental_test_images",x)

#x = np.load('E:/src/datasets/Segmentation datasets/augemented/Augmented_data_segmentation.npy')
#x = x.reshape(1900, 512, 512, 1)
#y = np.load('E:/src/datasets/Segmentation datasets/augemented/Augmented_target_segmentation.npy')
#y = y.reshape(1900, 512, 512, 1)
#image, target = shuffle(x, y)


#for i in range(1900):
#    cv2.imshow('image', image[i])
#    cv2.imshow('target', target[i])
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()