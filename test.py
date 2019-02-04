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

from utilities.util_funcs import mean_iou, iou_metric, iou_metric_batch, RLenc


def run_test(img_height, img_width, img_chan, path_test):

############################################### LOADING TEST DATA #############################################################
    test_ids = next(os.walk(path_test+"images"))[2]

    X_test = np.zeros((len(test_ids), im_height, im_width, im_chan), dtype=np.uint8)
    sizes_test = []
    print('Getting and resizing test images ... ')
    sys.stdout.flush()
    for n, id_ in enumerate(test_ids):
        path = path_test
        img = load_img(path + '/images/' + id_)
        x = img_to_array(img)[:,:,:]
        sizes_test.append([x.shape[0], x.shape[1]])
        x = resize(x, (im_height, im_width, im_chan), mode='constant', preserve_range=True)
        X_test[n] = x

    print('Done!')

############################################### LOADING MODEL #############################################################
    model = load_model("best_model.h5" , custom_objects = {'mean_iou' : mean_iou})

############################################### PERFORMING TEST #############################################################
    preds_test = model.predict(X_test, verbose=1)

    preds_test_t = (preds_test > 0.5).astype(np.uint8)

    preds_test_upsampled = []
    for i in tnrange(len(preds_test)):
        preds_test_upsampled.append(resize(np.squeeze(preds_test[i]),
                                           (sizes_test[i][0], sizes_test[i][1]),
                                           mode='constant', preserve_range=True))


############################################ GETTING BEST THRESHOLD AND CREATING OUTPUT FILE #####################################
    thres = np.linspace(0.25, 0.75, 20)
    thres_ioc = [iou_metric_batch(Y_val, np.int32(preds_val > t)) for t in thres]

    best_thres = thres[np.argmax(thres_ioc)]
    best_thres, max(thres_ioc)

    pred_dict = {fn[:-4]:RLenc(np.round(preds_test_upsampled[i] > best_thres)) for i,fn in enumerate(test_ids)}

    sub = pd.DataFrame.from_dict(pred_dict,orient='index')
    sub.index.names = ['id']
    sub.columns = ['rle_mask']
    sub.to_csv('outputs.csv')


img_height = #Enter Image Height Here
img_width = #Enter Image Width Here
img_chan = #Enter Number of Channels Here
path_test = #Enter Path to the Test Images Here

run_test(img_height, img_width, img_chan, path_test)                 
