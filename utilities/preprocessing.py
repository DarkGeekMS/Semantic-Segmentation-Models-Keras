import os
import sys
import random
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

def preprocess_data(im_width, im_height, im_chan, path_train):

############################################ DATA LOADING ###################################################################
    train_ids = next(os.walk(path_train+"images"))[2]

    X_train = np.zeros((len(train_ids), im_height, im_width, im_chan), dtype=np.uint8)
    Y_train = np.zeros((len(train_ids), im_height, im_width, 1), dtype=np.bool)
    print('Getting and resizing train images and masks ... ')
    sys.stdout.flush()
    for n, id_ in enumerate(train_ids):
        path = path_train
        img = load_img(path + '/images/' + id_)
        x = img_to_array(img)[:,:,:]
        x = resize(x, (im_height, im_width, im_chan), mode='constant', preserve_range=True)
        X_train[n] = x
        mask = img_to_array(load_img(path + '/masks/' + id_))[:,:,:]
        Y_train[n] = resize(mask, (im_height, im_width, 1), mode='constant', preserve_range=True)

    print('Done!')

    X_train , X_val , Y_train , Y_val = train_test_split(X_train , Y_train , train_size = 0.9 , random_state = 2019)

############################################ DATA AUGMENTATION ###################################################################

    image_generator = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip = True,
        zoom_range = 0.2,
        width_shift_range = 0.2,
        height_shift_range = 0.2
    )

    mask_generator = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip = True,
        zoom_range = 0.2,
        width_shift_range = 0.2,
        height_shift_range = 0.2
    )

    val_image_generator = ImageDataGenerator(
    )
    val_mask_generator = ImageDataGenerator(
    )

    train_img_gen = image_generator.flow(X_train , seed = 2018 , batch_size = 16)
    train_mask_gen = mask_generator.flow(Y_train , seed = 2018 , batch_size = 16)

    val_img_gen = val_image_generator.flow(X_val , seed = 2018 , batch_size = 16)
    val_mask_gen = val_mask_generator.flow(Y_val , seed = 2018 , batch_size = 16)

    train_gen = zip(train_img_gen , train_mask_gen)
    val_gen = zip(val_img_gen , val_mask_gen)

    return train_gen, val_gen
