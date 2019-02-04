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

def convlayer(inputs , filters , kernel_size , strides , activation = 'relu'):

    X = Conv2D(filters , kernel_size = (kernel_size , kernel_size) , strides = (strides , strides) , padding = 'same')(inputs)
    X = BatchNormalization()(X)
    X = Activation(activation)(X)

    if(activation != 'sigmoid'):
        sq = GlobalAveragePooling2D()(X)
        sq = Dense(int(filters / 4) , activation = 'relu')(sq)
        sq = Dense(filters , activation = 'sigmoid')(sq)
        sq = Reshape((1,1,filters))(sq)
        X = Multiply()([sq , X])

    return X

def resnet_block(inputs , filters , strides):

    identity = Conv2D(filters , kernel_size = (1 , 1) , strides = (strides , strides),
                      kernel_initializer = he_normal(2019) , kernel_regularizer = l2(1e-4), padding = 'same')(inputs)

    conv = Conv2D(filters , kernel_size = (3 , 3) , strides = (strides , strides) ,
                  kernel_initializer = he_normal(2019) , kernel_regularizer = l2(1e-4), padding = 'same')(inputs)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)

    conv = Conv2D(filters , kernel_size = (3 , 3) , strides = (1 , 1),
                  kernel_initializer = he_normal(2019) , kernel_regularizer = l2(1e-4), padding = 'same')(conv)
    conv = BatchNormalization()(conv)

    output = Add()([identity , conv])
    output = Activation('relu')(output)

    sq = GlobalAveragePooling2D()(output)
    sq = Dense(int(filters / 4) , activation = 'relu')(sq)
    sq = Dense(filters , activation = 'sigmoid')(sq)
    sq = Reshape((1,1,filters))(sq)
    ex = Multiply()([sq , output])

    return ex

def unet_model(im_height, im_width, im_chan):

    input_img = Input((im_height, im_width, im_chan), name='img')

    res1 = resnet_block(input_img, filters = 4, strides = 1)

    res1 = resnet_block(res1, filters = 4, strides = 1)

    res2 = resnet_block(res1, filters = 8, strides = 2)

    res2 = resnet_block(res2, filters = 8, strides = 1)

    res3 = resnet_block(res2, filters = 16, strides = 2)

    res3 = resnet_block(res3, filters = 16, strides = 1)

    res4 = resnet_block(res3, filters = 32, strides = 2)

    res4 = resnet_block(res4, filters = 32, strides = 1)

    res5 = resnet_block(res4, filters = 64, strides = 2)

    res5 = resnet_block(res5, filters = 64, strides = 1)

    mid = convlayer(res5 , filters = 128, kernel_size = 3 , strides = 2 , activation = 'relu')
    mid = convlayer(mid , filters = 128, kernel_size = 3 , strides = 1 , activation = 'relu')

    up1 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (mid)
    up1 = concatenate([up1, res5])
    c1 = convlayer(up1 , filters = 64, kernel_size = 3 , strides = 1 , activation = 'relu')
    c1 = convlayer(c1 , filters = 64, kernel_size = 3 , strides = 1 , activation = 'relu')

    up2 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c1)
    up2 = concatenate([up2, res4])
    c2 = convlayer(up2 , filters = 32, kernel_size = 3 , strides = 1 , activation = 'relu')
    c2 = convlayer(c2 , filters = 32, kernel_size = 3 , strides = 1 , activation = 'relu')

    up3 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c2)
    up3 = concatenate([up3, res3])
    c3 = convlayer(up3 , filters = 16, kernel_size = 3 , strides = 1 , activation = 'relu')
    c3 = convlayer(c3 , filters = 16, kernel_size = 3 , strides = 1 , activation = 'relu')

    up4 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (c3)
    up4 = concatenate([up4, res2])
    c4 = convlayer(up4 , filters = 8, kernel_size = 3 , strides = 1 , activation = 'relu')
    c4 = convlayer(c4 , filters = 8, kernel_size = 3 , strides = 1 , activation = 'relu')

    up5 = Conv2DTranspose(4, (2, 2), strides=(2, 2), padding='same') (c4)
    up5 = concatenate([up5, res1], axis = 3)
    c5 = convlayer(up5 , filters = 4, kernel_size = 3 , strides = 1 , activation = 'relu')
    c5 = convlayer(c5 , filters = 4, kernel_size = 3 , strides = 1 , activation = 'relu')

    outputs = convlayer(c5 , filters = 1, kernel_size = 1 , strides = 1 , activation = 'sigmoid')

    model = Model(inputs=[input_img], outputs=[outputs])

    return model
