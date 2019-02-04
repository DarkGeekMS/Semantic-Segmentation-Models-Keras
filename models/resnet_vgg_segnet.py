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

    identity = Conv2D(filters , kernel_size = (1 , 1) , strides = (strides , strides) , padding = 'same')(inputs)

    conv = Conv2D(filters , kernel_size = (3 , 3) , strides = (strides , strides) , padding = 'same')(inputs)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)

    conv = Conv2D(filters , kernel_size = (3 , 3) , strides = (1 , 1), padding = 'same')(conv)
    conv = BatchNormalization()(conv)

    output = Add()([identity , conv])
    output = Activation('relu')(output)

    sq = GlobalAveragePooling2D()(output)
    sq = Dense(int(filters / 4) , activation = 'relu')(sq)
    sq = Dense(filters , activation = 'sigmoid')(sq)
    sq = Reshape((1,1,filters))(sq)
    ex = Multiply()([sq , output])

    return ex

def vgg_block(inputs, filters):

    x = Conv2D(filters, (3, 3), activation='relu', padding='same')(inputs)
    x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)

    if (filters != 4):
        x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    return x

def upsample(inputs):

    X = UpSampling2D(size = (2,2))(inputs)

    return X

def segnet_model(im_height, im_width, im_chan):

    input_img = Input((im_height, im_width, im_chan), name='img')

    inp = BatchNormalization()(input_img)

    res1 = resnet_block(inp, filters = 4, strides = 1)

    res2 = resnet_block(res1, filters = 8, strides = 2)

    res3 = resnet_block(res2, filters = 16, strides = 2)

    res4 = resnet_block(res3, filters = 32, strides = 2)

    res5 = resnet_block(res4, filters = 64, strides = 2)

    vgg1 = vgg_block(inp, filters = 4)

    vgg2 = vgg_block(vgg1, filters = 8)

    vgg3 = vgg_block(vgg2, filters = 16)

    vgg4 = vgg_block(vgg3, filters = 32)

    vgg5 = vgg_block(vgg4, filters = 64)

    mid = concatenate([res5, vgg5])

    up1 = upsample(mid)
    up1 = convlayer(up1 , filters = 32 , kernel_size = 3 , strides = 1)
    up1 = convlayer(up1 , filters = 32 , kernel_size = 3 , strides = 1)
    up1 = convlayer(up1 , filters = 32 , kernel_size = 3 , strides = 1)

    up2 = upsample(up1)
    up2 = convlayer(up2 , filters = 16 , kernel_size = 3 , strides = 1)
    up2 = convlayer(up2 , filters = 16 , kernel_size = 3 , strides = 1)
    up2 = convlayer(up2 , filters = 16 , kernel_size = 3 , strides = 1)

    up3 = upsample(up2)
    up3 = convlayer(up3 , filters = 8 , kernel_size = 3 , strides = 1)
    up3 = convlayer(up3 , filters = 8 , kernel_size = 3 , strides = 1)
    up3 = convlayer(up3 , filters = 8 , kernel_size = 3 , strides = 1)

    up4 = upsample(up3)
    up4 = convlayer(up4 , filters = 4 , kernel_size = 3 , strides = 1)
    up4 = convlayer(up4 , filters = 4 , kernel_size = 3 , strides = 1)
    up4 = convlayer(up4 , filters = 4 , kernel_size = 3 , strides = 1)

    outputs = convlayer(up4 , filters = 1 , kernel_size = 3 , strides = 1 , activation = 'sigmoid')

    model = Model(inputs=[input_img], outputs=[outputs])

    return model             
