import cv2
import numpy as np
from keras.models import load_model
import gc
import matplotlib.pyplot as plt
from PIL import Image
# garbage collector
gc.collect ()

validation =30
TR = 348
TE = 97

x = np.load ('E:/src/Training samples.npy')
x = x.reshape (475,512,512,1)

x_train = x[:348]
val_x_train = x[348:379]
test_x_samples = x[379:]

y = np.load ('E:/src/Ground Truth.npy')
y = y.reshape (475,512,512,1)

y_train = y[0:348]
val_y_train = y[348:379]
tests_y_samples = y[379:475]

model = load_model('Unet_skip.h5')
model.summary()

score = model.evaluate (val_x_train, val_y_train, verbose = 0)
print ('Test loss:', score [0])
print ('Test accuracy:', score [1])

# samples between 0 and 95
p= 33
prediction = model.predict(test_x_samples, batch_size = 1)
prediction = np.asarray (prediction)
print (prediction.shape)
cv2.imshow ('image', prediction[p])
cv2.imshow('GT', tests_y_samples[p])
cv2.waitKey()
cv2.destroyAllWindows()

