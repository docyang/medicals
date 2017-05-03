# --*-- encoding: UTF-8 --*--

import cv2
import numpy as np
import tensorflow as tf
import SimpleITK as sitk
from scipy.ndimage.interpolation import zoom

Wb = {
    'W1': tf.get_variable('W1', [5, 5, 5, 1, 32], tf.float32),
    'b1': tf.Variable(tf.zeros([32])),
    'W2': tf.get_variable('W2', [3, 3, 3, 32, 24], tf.float32),
    'b2': tf.Variable(tf.zeros([24])),
    'W3': tf.get_variable('W3', [3, 3, 3, 24, 24], tf.float32),
    'b3': tf.Variable(tf.zeros([24])),
    'W4': tf.get_variable('W4', [3, 3, 3, 24, 32], tf.float32),
    'b4': tf.Variable(tf.zeros([32])),
    'fcw1-conv': tf.get_variable('fcw1-conv', [4, 4, 4, 32, 16], tf.float32),
    'fcb1-conv': tf.Variable(tf.zeros([16])),
    'fcw2-conv': tf.get_variable('fcw2-conv', [1, 1, 1, 16, 2], tf.float32),
    'fcb2-conv': tf.Variable(tf.zeros([2]))
}


def model(data, keep_prob):
    with tf.variable_scope('conv1') as scope:
        tf.add_to_collection('data', data)
        conv1 = tf.nn.conv3d(data, Wb['W1'], strides=[1, 1, 1, 1, 1], padding='VALID')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, Wb['b1']))
        pool1 = tf.nn.max_pool3d(relu1, ksize=[1, 2, 2, 2, 1],
                                 strides=[1, 2, 2, 2, 1], padding='VALID')
    with tf.variable_scope('conv2') as scope:
        conv2 = tf.nn.conv3d(pool1, Wb['W2'], strides=[1, 1, 1, 1, 1], padding='VALID')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, Wb['b2']))
        pool2 = tf.nn.max_pool3d(relu2, ksize=[1, 2, 2, 2, 1],
                                 strides=[1, 2, 2, 2, 1], padding='VALID')
    with tf.variable_scope('conv3') as scope:
        conv3 = tf.nn.conv3d(pool2, Wb['W3'], strides=[1, 1, 1, 1, 1], padding='VALID')
        relu3 = tf.nn.relu(tf.nn.bias_add(conv3, Wb['b3']))
    with tf.variable_scope('conv4') as scope:
        conv4 = tf.nn.conv3d(relu3, Wb['W4'], strides=[1, 1, 1, 1, 1], padding='VALID')
        relu4 = tf.nn.relu(tf.nn.bias_add(conv4, Wb['b4']))
    with tf.variable_scope('fc1-conv'):
        conv5 = tf.nn.conv3d(relu4, Wb['fcw1-conv'], strides=[1, 1, 1, 1, 1], padding='SAME')
        relu5 = tf.nn.relu(tf.nn.bias_add(conv5, Wb['fcb1-conv']))
    with tf.variable_scope('dropout'):
        hidden = tf.nn.dropout(relu5, keep_prob)
        tf.add_to_collection('keep_prob', keep_prob)
    with tf.variable_scope('fc2-conv'):
        conv6 = tf.nn.conv3d(hidden, Wb['fcw2-conv'], strides=[1, 1, 1, 1, 1], padding='SAME')
        out = tf.nn.bias_add(conv6, Wb['fcb2-conv'])
        tf.add_to_collection('out', out)
    return out

def normalizePlanes(npzarray):
    maxHU = 400.
    minHU = -1000.
    npzarray = (npzarray - minHU) / (maxHU - minHU)
    npzarray[npzarray>1] = 1.
    npzarray[npzarray<0] = 0.
    return npzarray

def interpolatefilter(inputimage):
    origin = inputimage.GetOrigin()
    spacing = inputimage.GetSpacing()
    direction = inputimage.GetDirection()
    outputspacing = (spacing[0], spacing[0], spacing[0])
    size = inputimage.GetSize()
    tmp = []
    for i in range(3):
        s = int(spacing[i] * size[i] / outputspacing[i])
        if s % 2 != 0:
            s = s - 1
        tmp.append(s)
    outputsize = (tmp[0], tmp[1], tmp[2])
    resamplefilter = sitk.ResampleImageFilter()
    resamplefilter.SetOutputDirection(direction)
    resamplefilter.SetSize(outputsize)
    resamplefilter.SetOutputOrigin(origin)
    resamplefilter.SetOutputSpacing(outputspacing)
    outputimage = resamplefilter.Execute(inputimage)
    numpyImage = sitk.GetArrayFromImage(outputimage)
    numpyImage = normalizePlanes(numpyImage)
    return numpyImage

saver = tf.train.Saver(
        {'W1': Wb['W1'], 'Variable': Wb['b1'], 'W2': Wb['W2'], 'Variable_1': Wb['b2'], 'W3': Wb['W3'],
         'Variable_2': Wb['b3'], 'W4': Wb['W4'], 'Variable_3': Wb['b4'], 'Variable_4': Wb['fcw1-conv'],
         'Variable_5': Wb['fcb1-conv'], 'Variable_6': Wb['fcw2-conv'], 'Variable_7': Wb['fcb2-conv']})

with tf.Session() as sess:
    # load the model
    modelPath = './models_conv/model.ckpt'
    saver.restore(sess, modelPath)
    size = 224
    data_path = '/home/yanghan/data/stage1/006b96310a37b36cccb2ab48d10b49a3/'
    seriesreader = sitk.ImageSeriesReader()
    gdcmnames = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(data_path)
    seriesreader.SetFileNames(gdcmnames)
    simage = seriesreader.Execute()
    image_raw = normalizePlanes(sitk.GetArrayFromImage(simage))
    #image = interpolatefilter(simage)
    #image = np.moveaxis(normalizePlanes(np.load(data_path + 'image.npy')), -1, 0).astype(np.float32)
    zoom_scale = (float(size)/image_raw.shape[0], float(size)/image_raw.shape[1], float(size)/image_raw.shape[2])
    data = zoom(image_raw, zoom_scale).reshape([1, size, size, size, 1]).astype(np.float32)
    #data = tf.random_normal([1, 120, 128, 128, 1])
    logit = model(data, 1.0)
    prob =  sess.run(logit)
    print prob.shape
    images = prob[0][:,:,:,1]
    zoom_scale1 = (float(image_raw.shape[0])/images.shape[0], float(image_raw.shape[1])/images.shape[1],
                   float(image_raw.shape[2])/images.shape[2])
    probs = zoom(images, zoom_scale1)
    for i in range(probs.shape[0]):
        prob = probs[i]*255
        img = image_raw[i]*255
        #image = cv2.resize(image, (64, 64))
        #print image
        cv2.imwrite('./test/{}.png'.format(i), prob)
        cv2.imwrite('./test/{}.bmp'.format(i), img)























