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

from keras.models import Model, load_model
from keras.layers import Input , Concatenate
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.layers import *
from keras.initializers import he_normal
from keras.regularizers import l2
from keras.layers import Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
from keras.optimizers import *
from keras.utils import to_categorical
import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from utilities.util_funcs import mean_iou
from utilities.preprocessing import preprocess_data


def train_model(model, img_height, img_width, img_chan, no_classes, train_path, no_epochs):

    train_gen, val_gen = preprocess_data(img_width, img_height, img_chan, train_path)

    if (model == 'native_unet'):
        from models.native_unet import unet_model
        model = unet_model(img_height, img_width, img_chan)
    elif (model == 'resnet_vgg_unet'):
        from models.resnet_vgg_unet import unet_model
        model = unet_model(img_height, img_width, img_chan)
    elif (model == 'resnet_segnet') :
        from models.resnet_segnet import segnet_model
        model = segnet_model(img_height, img_width, img_chan)
    elif (model == 'resnet_vgg_segnet'):
        model = segnet_model(img_height, img_width, img_chan)
    elif (model == 'deeplabv3'):
        from models.deeplabv3 import Deeplabv3
        model = Deeplabv3(input_shape= (img_height, img_width, img_chan), classes= no_classes)
    else:
        raise RuntimeError('The model name you selected not exist.')

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[mean_iou])

    early_stopper = EarlyStopping(monitor= 'val_mean_iou', patience=10, mode='max')
    lr_reducer = ReduceLROnPlateau(monitor= 'val_loss', factor=0.1, patience=5,  min_lr=1e-6, mode='max')
    checkpointer = ModelCheckpoint('best_model.h5', monitor='val_mean_iou', verbose=2, save_best_only=True, save_weights_only = False, mode = 'max')

    results = model.fit_generator(train_gen , steps_per_epoch = 1125 , epochs = no_epochs,
                                  validation_data = val_gen , validation_steps = 25 ,
                                  callbacks=[checkpointer , lr_reducer, early_stopper] , verbose = 1)


model = #Enter Model Name Here
img_height = #Enter Image Height Here
img_width = #Enter Image Width Here
img_chan = #Enter Number of Channels Here
no_classes = #Enter Number of Classes Here
no_epochs = #Enter Number of Epochs Here
train_path = #Enter Path to the Train Images Here

train_model(model, img_height, img_width, img_chan, no_classes, train_path, no_epochs)
