#!/usr/bin/env python
#  encoding: utf-8

""" test the 2D U-Net model.
    @author: Han Yang
    @version: keras2.0, tensorflow1.0
"""
import cv2, os
import numpy as np
from keras.models import Model
from keras import backend as K
from keras.layers.merge import Concatenate
from keras.layers import Input, Conv2D, MaxPool2D, UpSampling2D, BatchNormalization
from keras.optimizers import Adam
from util import get_testdata, generate_arrays_from_file, normalize, readdcm

K.set_image_dim_ordering('th')

smooth = 1.

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_np(y_true,y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def get_unet():
    inputs = Input((1, 512, 512))

    conv1 = Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    conv1 = Conv2D(32, (3, 3), padding='same', activation='relu')(conv1)
    conv1 = BatchNormalization(axis=1)(conv1)
    pool1 = MaxPool2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), padding='same', activation='relu')(pool1)
    conv2 = Conv2D(64, (3, 3), padding='same', activation='relu')(conv2)
    conv2 = BatchNormalization(axis=1)(conv2)
    pool2 = MaxPool2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), padding='same', activation='relu')(pool2)
    conv3 = Conv2D(128, (3, 3), padding='same', activation='relu')(conv3)
    conv3 = BatchNormalization(axis=1)(conv3)
    pool3 = MaxPool2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), padding='same', activation='relu')(pool3)
    conv4 = Conv2D(256, (3, 3), padding='same', activation='relu')(conv4)
    conv4 = BatchNormalization(axis=1)(conv4)
    pool4 = MaxPool2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), padding='same', activation='relu')(pool4)
    conv5 = Conv2D(512, (3, 3), padding='same', activation='relu')(conv5)
    conv5 = BatchNormalization(axis=1)(conv5)

    up6 = UpSampling2D(size=(2, 2))(conv5)
    up6 = Concatenate(axis=1)([up6, conv4])
    conv6 = Conv2D(256, (3, 3), padding='same', activation='relu')(up6)
    conv6 = Conv2D(256, (3, 3), padding='same', activation='relu')(conv6)
    conv6 = BatchNormalization(axis=1)(conv6)

    up7 = UpSampling2D(size=(2, 2))(conv6)
    up7 = Concatenate(axis=1)([up7, conv3])
    conv7 = Conv2D(128, (3, 3), padding='same', activation='relu')(up7)
    conv7 = Conv2D(128, (3, 3), padding='same', activation='relu')(conv7)
    conv7 = BatchNormalization(axis=1)(conv7)

    up8 = UpSampling2D(size=(2, 2))(conv7)
    up8 = Concatenate(axis=1)([up8, conv2])
    conv8 = Conv2D(64, (3, 3), padding='same', activation='relu')(up8)
    conv8 = Conv2D(64, (3, 3), padding='same', activation='relu')(conv8)
    conv8 = BatchNormalization(axis=1)(conv8)

    up9 = UpSampling2D(size=(2, 2))(conv8)
    up9 = Concatenate(axis=1)([up9, conv1])
    conv9 = Conv2D(32, (3, 3), padding='same', activation='relu')(up9)
    conv9 = Conv2D(32, (3, 3), padding='same', activation='relu')(conv9)
    conv9 = BatchNormalization(axis=1)(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)
    #model = Model(outputs=conv10(inputs=inputs))

    model.compile(optimizer=Adam(lr=1.0e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model

def TestOne():
    model = get_unet()
    model.load_weights('./models/ggo_model.h5')
    images = normalize(np.load('/home/yanghan/data/ggo/train/a5d7909f14d43f01f44cdcaabed27b84/image.npy')) * 255
    #images = normalize(readdcm('/home/yanghan/data/stage1/0a38e7597ca26f9374f8ea2770ba870d')) * 255
    for i in range(images.shape[2]):
        image = images[:, :, i]
        pred = model.predict(image.reshape([1, 1, 512, 512]))
        print np.max(pred)
        cv2.imwrite('./test/{}.bmp'.format(i), image)
        cv2.imwrite('./test/{}.png'.format(i), pred[0][0] * 255)
        print i

def TestAll(datapath):
    model = get_unet()
    model.load_weights('./models/ggo_model.h5')
    for dir in os.listdir(datapath):
        path = datapath + dir
        images = normalize(np.load(path + '/image.npy')) * 255
        preds = []
        for i in range(images.shape[2]):
            image = images[:, :, i]
            pred = model.predict(image.reshape([1, 1, 512, 512]))
            preds.append(pred[0][0])
        preds = np.moveaxis(preds, 0, -1)
        print preds.shape
        np.save(path + '/pred', preds)

TestAll('/home/yanghan/data/ggo/train/')
# pred = np.load('/home/yanghan/data/ggo/train/1.3.6.1.4.1.32722.99.99.150787728253687380458859635290571741615/pred.npy')
# print pred.shape
# for i in range(pred.shape[2]):
#     cv2.imwrite('./test/{}.bmp'.format(i), pred[:,:,i]*255)