import numpy as np
from PIL import Image
import PIL.ImageOps
import cv2
image = np.random.randn(128,128)
inverted_image = np.random.randn(128,128)
dst=cv2.bitwise_or(image, inverted_image)

cv2.imshow('image',inverted_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('image',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
