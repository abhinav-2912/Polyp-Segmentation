import os
import cv2
import keras
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
import tensorflow

import tensorflow as tf
from tensorflow.keras import *
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Activation, Conv2DTranspose, UpSampling2D, Concatenate
from keras.backend.tensorflow_backend import set_session
import keras, sys, time, warnings
from keras.preprocessing.image import ImageDataGenerator

from keras import optimizers
import pandas as pd 
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
import imageio
from tensorflow.python.client import device_lib
from keras.models import model_from_json
from keras.models import load_model
import segmentation_models as sm

'''
typ 0: UpSampling2D
typ 1: Conv2DTranspose
'''

def get_model(input_shape=(224, 224, 3), typ=0):
    
    input_layer = Input(shape=input_shape)
    
    conv1 = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(input_layer)
    conv1_bn = BatchNormalization()(conv1)
    conv1_relu = Activation('relu')(conv1_bn)

    pool1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(conv1_relu)

    ## branch 1
    conv1_1 = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(pool1)
    conv1_1_bn = BatchNormalization()(conv1_1)
    conv1_1_relu = Activation('relu')(conv1_1_bn)

    ## branch 2
    conv2_1 = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(pool1)
    conv2_1_bn = BatchNormalization()(conv2_1)
    conv2_1_relu = Activation('relu')(conv2_1_bn)

    pool2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(conv2_1_relu)

    ## branch 3
    conv3_1 = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(pool2)
    conv3_1_bn = BatchNormalization()(conv3_1)
    conv3_1_relu = Activation('relu')(conv3_1_bn)

    ## branch 4
    conv4_1 = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(pool2)
    conv4_1_bn = BatchNormalization()(conv4_1)
    conv4_1_relu = Activation('relu')(conv4_1_bn)

    ## dilation branch
    dil1 = Conv2D(128, (3, 3), dilation_rate=(2, 2), padding='same')(conv4_1_relu)
    dil2 = Conv2D(128, (3, 3), dilation_rate=(4, 4), padding='same')(dil1)
    dil3 = Conv2D(128, (3, 3), dilation_rate=(8, 8), padding='same')(dil2)

    ## Add 1
    added1 = Add()([conv3_1_relu, dil3])
    if typ == 0:
        deconv1 = UpSampling2D((2, 2))(added1)
    else:
        deconv1 = Conv2DTranspose(128, (1, 1), strides=(2, 2), padding='valid')(added1)
    deconv1 = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(deconv1)
    deconv1 = BatchNormalization()(deconv1)
    deconv1 = Activation('relu')(deconv1)

    ## Add 2
    added2 = Add()([conv1_1_relu, deconv1])
    if typ == 0:
        deconv2 = UpSampling2D((2, 2))(added2)
    else:
        deconv2 = Conv2DTranspose(128, (1, 1), strides=(2, 2), padding='valid')(added2)
    deconv2 = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(deconv2)
    deconv2 = BatchNormalization()(deconv2)
    deconv2 = Activation('relu')(deconv2)

    conv_final = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(deconv2)
    conv_final = BatchNormalization()(conv_final)
    conv_final = Activation('relu')(conv_final)

    pred = Conv2D(2, (3, 3), strides=(1, 1), padding='same')(conv_final)
    pred = Activation('softmax')(pred)

    model = Model(inputs=input_layer, outputs=pred)

    return model

def get_model_depth_separable(input_shape=(224, 224, 3), typ=0):
    
    input_layer = Input(shape=input_shape)
    
    conv1 = SeparableConv2D(64, (3, 3), strides=(1, 1), padding='same')(input_layer)
    conv1_bn = BatchNormalization()(conv1)
    conv1_relu = Activation('relu')(conv1_bn)

    pool1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(conv1_relu)

    ## branch 1
    conv1_1 = SeparableConv2D(128, (3, 3), strides=(1, 1), padding='same')(pool1)
    conv1_1_bn = BatchNormalization()(conv1_1)
    conv1_1_relu = Activation('relu')(conv1_1_bn)

    ## branch 2
    conv2_1 = SeparableConv2D(128, (3, 3), strides=(1, 1), padding='same')(pool1)
    conv2_1_bn = BatchNormalization()(conv2_1)
    conv2_1_relu = Activation('relu')(conv2_1_bn)

    pool2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(conv2_1_relu)

    ## branch 3
    conv3_1 = SeparableConv2D(128, (3, 3), strides=(1, 1), padding='same')(pool2)
    conv3_1_bn = BatchNormalization()(conv3_1)
    conv3_1_relu = Activation('relu')(conv3_1_bn)

    ## branch 4
    conv4_1 = SeparableConv2D(128, (3, 3), strides=(1, 1), padding='same')(pool2)
    conv4_1_bn = BatchNormalization()(conv4_1)
    conv4_1_relu = Activation('relu')(conv4_1_bn)

    ## dilation branch
    dil1 = Conv2D(128, (3, 3), dilation_rate=(2, 2), padding='same')(conv4_1_relu)
    dil2 = Conv2D(128, (3, 3), dilation_rate=(4, 4), padding='same')(dil1)
    dil3 = Conv2D(128, (3, 3), dilation_rate=(8, 8), padding='same')(dil2)

    ## Add 1
    added1 = Add()([conv3_1_relu, dil3])
    if typ == 0:
        deconv1 = UpSampling2D((2, 2))(added1)
    else:
        deconv1 = Conv2DTranspose(128, (1, 1), strides=(2, 2), padding='valid')(added1)
    deconv1 = SeparableConv2D(128, (3, 3), strides=(1, 1), padding='same')(deconv1)
    deconv1 = BatchNormalization()(deconv1)
    deconv1 = Activation('relu')(deconv1)

    ## Add 2
    added2 = Add()([conv1_1_relu, deconv1])
    if typ == 0:
        deconv2 = UpSampling2D((2, 2))(added2)
    else:
        deconv2 = Conv2DTranspose(128, (1, 1), strides=(2, 2), padding='valid')(added2)
    deconv2 = SeparableConv2D(128, (3, 3), strides=(1, 1), padding='same')(deconv2)
    deconv2 = BatchNormalization()(deconv2)
    deconv2 = Activation('relu')(deconv2)

    conv_final = SeparableConv2D(64, (3, 3), strides=(1, 1), padding='same')(deconv2)
    conv_final = BatchNormalization()(conv_final)
    conv_final = Activation('relu')(conv_final)

    pred = SeparableConv2D(2, (3, 3), strides=(1, 1), padding='same')(conv_final)
    pred = Activation('softmax')(pred)

    model = Model(inputs=input_layer, outputs=pred)

    return model

def u_net1(input_shape=(96, 96, 3)):
    input_layer = Input(shape=input_shape)
    
    conv1 = Conv2D(32, (3, 3), strides=(1, 1), padding='same')(input_layer)
    conv1 = Activation('relu')(conv1)
    conv1 = Conv2D(32, (3, 3), strides=(1, 1), padding='same')(conv1)
    conv1 = Activation('relu')(conv1)

    pool1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(conv1) # 112

    conv2 = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(pool1)
    conv2 = Activation('relu')(conv2)
    conv2 = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(conv2)
    conv2 = Activation('relu')(conv2)

    pool2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(conv2) # 56

    conv3 = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(pool2)
    conv3 = Activation('relu')(conv3)
    conv3 = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(conv3)
    conv3 = Activation('relu')(conv3)

    pool3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(conv3) # 28

    conv4 = Conv2D(256, (3, 3), strides=(1, 1), padding='same')(pool3)
    conv4 = Activation('relu')(conv4)
    conv4 = Conv2D(256, (3, 3), strides=(1, 1), padding='same')(conv4)
    conv4 = Activation('relu')(conv4)

    convd3 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(conv4)
    convd3 = Concatenate()([convd3, conv3])

    convd3 = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(convd3)
    convd3 = Activation('relu')(convd3)
    convd3 = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(convd3)
    convd3 = Activation('relu')(convd3)

    convd2 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(convd3)
    convd2 = Concatenate()([convd2, conv2])

    convd2 = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(convd2)
    convd2 = Activation('relu')(convd2)
    convd2 = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(convd2)
    convd2 = Activation('relu')(convd2)

    convd1 = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(convd2)
    convd1 = Concatenate()([convd1, conv1])

    convd1 = Conv2D(32, (3, 3), strides=(1, 1), padding='same')(convd1)
    convd1 = Activation('relu')(convd1)
    convd1 = Conv2D(32, (3, 3), strides=(1, 1), padding='same')(convd1)
    convd1 = Activation('relu')(convd1)

    pred = Conv2D(2, (1, 1), strides=(1, 1), padding='same', activation='softmax')(convd1)

    model = Model(inputs=input_layer, outputs=pred)

    return model

def base_net(input_shape=(96, 96, 3)):
    input_layer = Input(shape=input_shape)
    
    conv1 = Conv2D(32, (3, 3), strides=(1, 1), padding='same')(input_layer)
    conv1 = Activation('relu')(conv1)
    conv1 = Conv2D(32, (3, 3), strides=(1, 1), padding='same')(conv1)
    conv1 = Activation('relu')(conv1)

    pred = Conv2D(2, (1, 1), strides=(1, 1), padding='same', activation='softmax')(conv1)

    model = Model(inputs=input_layer, outputs=pred)

    return model

    