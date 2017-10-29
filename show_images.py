#!/usr/bin/env python

# Load pickled data
import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import exposure
from skimage import img_as_ubyte
from tqdm import tqdm

def convert_to_gray_and_normalize(x):
    gray_image = img_as_ubyte(exposure.equalize_adapthist(x))
    gray_image = cv2.cvtColor(gray_image, cv2.COLOR_RGB2GRAY)
    return gray_image
#    return np.reshape((gray_image-[128])/[128], [32,32,1])  
#    return np.reshape((cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)-[128])/[128], [32,32,1])  

def convert_to_gray(x):
    gray_image = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
    return gray_image

training_file = "/mnt/raid/projects/udacity/sdc_nd/datasets/traffic-signs-data/train.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']

index_img_18 = np.argwhere(y_train == 18)
index_img_17 = np.argwhere(y_train == 17)
index_img_1 = np.argwhere(y_train == 1)
index_img_4 = np.argwhere(y_train == 4)
index_img_33 = np.argwhere(y_train == 33)

fig, ax = plt.subplots(ncols=5, nrows=3, figsize=(32,32))

ax[0][0].imshow(X_train[index_img_18[0]][0])
ax[0][1].imshow(X_train[index_img_17[0]][0])
ax[0][2].imshow(X_train[index_img_1[0]][0])
ax[0][3].imshow(X_train[index_img_4[0]][0])
ax[0][4].imshow(X_train[index_img_33[0]][0])

ax[1][0].imshow(convert_to_gray_and_normalize(X_train[index_img_18[0]][0]), cmap="gray")
ax[1][1].imshow(convert_to_gray_and_normalize(X_train[index_img_17[0]][0]), cmap="gray")
ax[1][2].imshow(convert_to_gray_and_normalize(X_train[index_img_1[0]][0]), cmap="gray")
ax[1][3].imshow(convert_to_gray_and_normalize(X_train[index_img_4[0]][0]), cmap="gray")
ax[1][4].imshow(convert_to_gray_and_normalize(X_train[index_img_33[0]][0]), cmap="gray")

ax[2][0].imshow(convert_to_gray(X_train[index_img_18[0]][0]), cmap="gray")
ax[2][1].imshow(convert_to_gray(X_train[index_img_17[0]][0]), cmap="gray")
ax[2][2].imshow(convert_to_gray(X_train[index_img_1[0]][0]), cmap="gray")
ax[2][3].imshow(convert_to_gray(X_train[index_img_4[0]][0]), cmap="gray")
ax[2][4].imshow(convert_to_gray(X_train[index_img_33[0]][0]), cmap="gray")

plt.show()

# test_image = X_train[index_img_18[0]][0]
# test_image_resized = cv2.resize(test_image, None, fx=0.8, fy=0.8, interpolation = cv2.INTER_CUBIC)
# test_image_blank = np.zeros([32,32,3], np.uint8)
# test_image_blank[0:test_image_resized.shape[0], 0:test_image_resized.shape[1]] = test_image_resized
# test_image_resized = test_image_blank

# print(test_image_resized.shape)
# plt.imshow(test_image_resized)
# plt.show()





