import sys
sys.path.insert(0, './models/')

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

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

x_train_path = '/home/mat/ug/16123001/data/x_train/folder/'
y_train_path = '/home/mat/ug/16123001/data/y_train/folder/'
x_val_path = '/home/mat/ug/16123001/data/x_val/folder/'
y_val_path = '/home/mat/ug/16123001/data/y_val/folder/'

# classes for data loading and preprocessing
with tf.device('/gpu:3'):

    dataset = Dataset(x_train_path, y_train_path, classes=['non-polyp', 'polyp'], augmentation=get_training_augmentation())
    
    BATCH_SIZE = 8
    CLASSES = ['non-polyp', 'polyp']
    LR = 0.0001
    EPOCHS = 25
    IMAGE_ORDERING = 'channels_last'
    n_classes = 2
    
    # SOTA
    BACKBONE = 'resnet34'
    # define model
    model = sm.Unet(BACKBONE, encoder_weights='imagenet')
    model = sm.Linknet(BACKBONE, encoder_weights='imagenet')
    model = sm.FPN(BACKBONE, encoder_weights='imagenet')
    model = sm.PSPNet(BACKBONE, encoder_weights='imagenet')
    
    model = fcn_8.fcn_8(2)

    optim = tf.keras.optimizers.Adam(LR)

    # Segmentation models losses can be combined together by '+' and scaled by integer or float factor
    # set class weights for dice_loss (car: 1.; pedestrian: 2.; background: 0.5;)
    dice_loss = sm.losses.DiceLoss(class_weights=np.array([0.5, 1])) 
    focal_loss = sm.losses.BinaryFocalLoss() 
    # if n_classes == 1 else sm.losses.CategoricalFocalLoss()
    total_loss = dice_loss + (1 * focal_loss)

    # actulally total_loss can be imported directly from library, above example just show you how to manipulate with losses
    # total_loss = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss 

    metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

    # compile keras model with defined optimozer, loss and metrics
    model.compile(optim, total_loss, metrics)

    train_dataset = Dataset(
        x_train_path, 
        y_train_path, 
        classes=CLASSES, 
        augmentation=get_training_augmentation())

    # Dataset for validation images
    valid_dataset = Dataset(
        x_val_path, 
        y_val_path, 
        classes=CLASSES, 
        augmentation=get_validation_augmentation())

    train_dataloader = Dataloder(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_dataloader = Dataloder(valid_dataset, batch_size=1, shuffle=False)

    # check shapes for errors
    assert train_dataloader[0][0].shape == (BATCH_SIZE, 224, 224, 3)
    assert train_dataloader[0][1].shape == (BATCH_SIZE, 224, 224, n_classes)

    # define callbacks for learning rate scheduling and best checkpoints saving
    callbacks = [
        keras.callbacks.ModelCheckpoint('./best_model.h5', save_weights_only=True, save_best_only=True, mode='min'),
        keras.callbacks.ReduceLROnPlateau(),
    ]

    history = model.fit_generator(
        train_dataloader, 
        steps_per_epoch=int(50000/8), 
        epochs=EPOCHS, 
        callbacks=callbacks, 
        validation_data=valid_dataloader, 
        validation_steps=int(5000),
        verbose=1
    )
    
    print(history)