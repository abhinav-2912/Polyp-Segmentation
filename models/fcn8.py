import os
import cv2
import keras
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
import tensorflow as tf

from tensorflow.keras import *
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
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
from data aug import get_training_augmentation, get_validation_augmentation, get_preprocessing
from data_utils import Dataset, Dataloder
from models import fcn8, u-nets

import segmentation_models as sm

BATCH_SIZE = 8
CLASSES = ['non-polyp', 'polyp']
LR = 0.0001
EPOCHS = 25
IMAGE_ORDERING = 'channels_last'
n_classes = 2

# crop o1 wrt o2
def crop( o1 , o2 , i  ):
    o_shape2 = Model( i  , o2 ).output_shape
#     print("In crop: o1.shape: {}, o2.shape: {}, i.shape: {}, o_shape2.shape: {}".format(o1, o2, i, o_shape2))

    if IMAGE_ORDERING == 'channels_first':
        output_height2 = o_shape2[2]
        output_width2 = o_shape2[3]
    else:
        output_height2 = o_shape2[1]
        output_width2 = o_shape2[2]

    o_shape1 = Model( i  , o1 ).output_shape
    if IMAGE_ORDERING == 'channels_first':
        output_height1 = o_shape1[2]
        output_width1 = o_shape1[3]
    else:
        output_height1 = o_shape1[1]
        output_width1 = o_shape1[2]

    cx = abs( output_width1 - output_width2 )
    cy = abs( output_height2 - output_height1 )

    if output_width1 > output_width2:
        o1 = Cropping2D( cropping=((0,0) ,  (  0 , cx )), data_format=IMAGE_ORDERING  )(o1)
    else:
        o2 = Cropping2D( cropping=((0,0) ,  (  0 , cx )), data_format=IMAGE_ORDERING  )(o2)

    if output_height1 > output_height2 :
        o1 = Cropping2D( cropping=((0,cy) ,  (  0 , 0 )), data_format=IMAGE_ORDERING  )(o1)
    else:
        o2 = Cropping2D( cropping=((0, cy ) ,  (  0 , 0 )), data_format=IMAGE_ORDERING  )(o2)

    return o1 , o2 

def fcn_8( n_classes, input_height=224, input_width=224 ):

    img_input_shape = (224, 224, 3)
    model = tf.keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_tensor=None, input_shape=img_input_shape, pooling=None, classes=1000)

#     for layer in model.layers:
#         layer.trainable = False

    f3 = model.get_layer('block3_pool').output
    f4 = model.get_layer('block4_pool').output
    f5 = model.get_layer('block5_pool').output

    img_input = model.input

    o = f5

    o = ( Conv2D( 4096, ( 7 , 7 ), activation='relu', padding='same', data_format=IMAGE_ORDERING))(o)
    o = Dropout(0.5)(o)
    o = (Conv2D( 4096 , ( 1 , 1 ) , activation='relu' , padding='same', data_format=IMAGE_ORDERING))(o)
    o = Dropout(0.5)(o)

    o = (Conv2D(n_classes, (1, 1), kernel_initializer='glorot_uniform', data_format=IMAGE_ORDERING))(o)
    o = Conv2DTranspose(n_classes, kernel_size=(4,4), strides=(2,2), use_bias=False, data_format=IMAGE_ORDERING)(o)
#     print(o.get_shape())

    o2 = f4
    o2 = (Conv2D(n_classes, (1, 1), kernel_initializer='glorot_uniform', data_format=IMAGE_ORDERING))(o2)
#     print(o2.get_shape())

    o, o2 = crop(o, o2, img_input)
#     print(o.get_shape())
#     print(o2.get_shape())

    o = Add()([o, o2])
#     print(o.get_shape())

    o = Conv2DTranspose(n_classes, kernel_size=(4,4), strides=(2,2), use_bias=False, data_format=IMAGE_ORDERING )(o)
    o2 = f3 
    o2 = (Conv2D(n_classes, (1, 1), kernel_initializer='glorot_uniform', data_format=IMAGE_ORDERING))(o2)
    o2 , o = crop(o2, o, img_input)
    o  = Add()([o2, o])

    o = Conv2DTranspose(n_classes, kernel_size=(16, 16), strides=(8, 8), use_bias=False, data_format=IMAGE_ORDERING)(o)
    o = (Conv2D(n_classes, (1, 1), kernel_initializer='glorot_uniform', data_format=IMAGE_ORDERING))(o)
    o, o2 = crop(o, img_input, img_input)
#     print(o.get_shape())

#     dim_list = o.get_shape().as_list()
#     output_height = dim_list[1]
#     output_width = dim_list[2]

#     o = (Reshape((output_height * output_width, -1)))(o)
    prediction = (Activation('softmax'))(o)
    print(prediction.get_shape())

    final_model = Model(inputs=img_input, outputs=prediction)
    print(final_model.summary())
    return final_model