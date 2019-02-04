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

def downsample(inputs):

    avg_pool = AveragePooling2D(pool_size = (2 , 2) , strides = 2 , padding = 'same')(inputs)
    return avg_pool

def convlayer(inputs , filters , kernel_size , strides , activation = 'relu'):

    if(kernel_size == 3):
        X = Conv2D(filters , kernel_size = kernel_size , strides = strides , padding = 'same')(inputs)
    else:
        X = Conv2D(filters , kernel_size = kernel_size , strides = strides , padding = 'valid')(inputs)
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

    if(strides != 1):
        identity = Conv2D(filters , kernel_size = 1 , strides = 2 , padding = 'same')(inputs)
    else:
        identity = inputs

    conv = Conv2D(filters , kernel_size = 3 , strides = strides , padding = 'same')(inputs)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)

    conv = Conv2D(filters , kernel_size = 3 , strides = 1 , padding = 'same')(conv)
    conv = BatchNormalization()(conv)

    output = Add()([identity , conv])
    output = Activation('relu')(output)

    sq = GlobalAveragePooling2D()(output)
    sq = Dense(int(filters / 4) , activation = 'relu')(sq)
    sq = Dense(filters , activation = 'sigmoid')(sq)
    sq = Reshape((1,1,filters))(sq)
    ex = Multiply()([sq , output])

    return ex

def upsample(inputs):

    X = UpSampling2D(size = (2,2))(inputs)
    return X

def segnet_model(im_height, im_width, im_chan):

    inputs = Input((im_height, im_width, im_chan))

    s = Lambda(lambda x: x / 255) (inputs)

    e1 = resnet_block(s , filters = 8 , strides = 1)

    e2 = resnet_block(e1 , filters = 16 , strides = 2)

    e3 = resnet_block(e2 , filters = 32 , strides = 2)

    e4 = resnet_block(e3 , filters = 64 , strides = 2)

    e5 = resnet_block(e4 , filters = 128 , strides = 2)


    d1 = upsample(e5)
    d1 = convlayer(d1 , filters = 32 , kernel_size = 3 , strides = 1)
    d1 = convlayer(d1 , filters = 32 , kernel_size = 3 , strides = 1)
    d1 = convlayer(d1 , filters = 32 , kernel_size = 3 , strides = 1)

    d2 = upsample(d1)
    d2 = convlayer(d2 , filters = 16 , kernel_size = 3 , strides = 1)
    d2 = convlayer(d2 , filters = 16 , kernel_size = 3 , strides = 1)
    d2 = convlayer(d2 , filters = 16 , kernel_size = 3 , strides = 1)

    d3 = upsample(d2)
    d3 = convlayer(d3 , filters = 8 , kernel_size = 3 , strides = 1)
    d3 = convlayer(d3 , filters = 8 , kernel_size = 3 , strides = 1)

    d4 = upsample(d3)
    d4 = convlayer(d4 , filters = 4 , kernel_size = 12 , strides = 1)
    d4 = convlayer(d4 , filters = 4 , kernel_size = 3 , strides = 1)
    d4 = convlayer(d4 , filters = 1 , kernel_size = 3 , strides = 1 , activation = 'sigmoid')

    model = Model(inputs = inputs , outputs = d4)

    return model
