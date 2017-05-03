#!/usr/local/bin/python
# -*- coding: utf-8 -*-
'''LUNA basic functions.

   Author: Kong Haiyang, Zhang Minshu & Fang Cheng
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import math
import numpy as np
from six.moves import xrange
import tensorflow as tf
import csv
import SimpleITK as sitk
import cv2
from tensorflow.contrib.layers import batch_norm
# from tf.contrib.layers.python.layers import batch_norm


class FLAGS():
  num_labels = 2
  NUM_EPOCHS = 4
  IMAGE_SIZE = 32
  CHANNEL_NUMBER = 1
  LABEL_NUMBER = [3,3]
  BATCH_SIZE = 128
  EVAL_BATCH_SIZE = 64
  SEED = 66478
  NUM_GPU = 1
  TOWER_NAME = 'JP'
  NUM_LABEL = 1
  NUM_PREPROCESS_THREADS = 12
  NUM_IMAGE = IMAGE_SIZE ** 3
  PIXEL_LENGTH = 4
  CUT_SIZE = 64
  SAVE_SIZE = 40
  BACK_SIZE = 800
  MIN_HU = -1000
  MAX_HU = 400
  #LIDC_PATH = '/home/kong/4T/nodule_project'
  #VIEW_PATH = './models-73_train-malignancy-32-0.04241/'
  VIEW_PATH = './models-73_train-malignancy-32-0.04241/'
  #CSV_FILE = '/home/yanghan/lidc_kaggle_aug/train/train_shuffle.bin/set_aug_num_train_labeled_statistic_lidc_kaggle.csv'
  # BIN_FILE = '/home/kong/4T/lung_cancer_merged41/train_shuffle.bin'
  BIN_FILE = './data/train.bin'
  MODEL_PATH = None
  SAVE_MODEL = True
  USE_OFFICIAL = True

XAVIER_INIT = tf.contrib.layers.xavier_initializer(seed=FLAGS.SEED)


def batch_norm_diy(inputs, is_training, decay=0.999, epsilon=0.001):
  shape = inputs.get_shape()
  beta = tf.Variable(tf.zeros([shape[-1]]))
  scale = tf.Variable(tf.ones([shape[-1]]))
  pop_mean = tf.Variable(tf.zeros([shape[-1]]), trainable=False)
  pop_var = tf.Variable(tf.ones([shape[-1]]), trainable=False)

  def bn_training():
    batch_mean, batch_var = tf.nn.moments(inputs, list(range(len(shape) - 1)))

    # update_moving_mean = moving_averages.assign_moving_average(batch_mean, mean, 0.9997)
    # update_moving_var = moving_averages.assign_moving_average(moving_var, var, 0.9997)

    train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
    train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
    with tf.control_dependencies([train_mean, train_var]):
      return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, scale, epsilon)

  return tf.cond(is_training, bn_training,
                 lambda: tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta,
                                                   scale, epsilon))


def bn(x, n_out, phase_train, scope='bn'):
  with tf.variable_scope(scope):
    beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                       name='beta', trainable=True)
    gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                        name='gamma', trainable=True)
    batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2, 3], name='moments')
    ema = tf.train.ExponentialMovingAverage(decay=0.5)

    def mean_var_with_update():
      ema_apply_op = ema.apply([batch_mean, batch_var])
      with tf.control_dependencies([ema_apply_op]):
        return tf.identity(batch_mean), tf.identity(batch_var)

    mean, var = tf.cond(phase_train,
                        mean_var_with_update,
                        lambda: (ema.average(batch_mean), ema.average(batch_var)))
    normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
  return normed


def batch_norm(input, isTraining, bias=True, scale=True, scope=None):
  return tf.cond(isTraining,
                 lambda: batch_norm(input, activation_fn=tf.nn.relu, reuse=None,
                                    is_training=True, center=bias, scale=scale,
                                    updates_collections=None, scope=scope.name),
                 lambda: batch_norm(input, activation_fn=tf.nn.relu, reuse=True,
                                    is_training=False, center=bias, scale=scale,
                                    updates_collections=None, scope=scope.name))


def eval_batch(data, sess, eval_prediction, eval_data, isTraining, isBN=False):
  size = data.shape[0]
  if size < FLAGS.EVAL_BATCH_SIZE:
    FLAGS.EVAL_BATCH_SIZE = size
  predictions = np.ndarray(shape=(size, FLAGS.LABEL_NUMBER), dtype=np.float32)
  if isBN:
    for begin in xrange(0, size, FLAGS.EVAL_BATCH_SIZE):
      end = begin + FLAGS.EVAL_BATCH_SIZE
      if end <= size:
        predictions[begin:end, :] = sess.run(eval_prediction, feed_dict={
            eval_data: data[begin:end, ...], isTraining: False})
      else:
        batch_predictions = sess.run(eval_prediction, feed_dict={
            eval_data: data[-FLAGS.EVAL_BATCH_SIZE:, ...], isTraining: False})
        predictions[begin:, :] = batch_predictions[begin - size:, :]
  else:
    for begin in xrange(0, size, FLAGS.EVAL_BATCH_SIZE):
      end = begin + FLAGS.EVAL_BATCH_SIZE
      if end <= size:
        predictions[begin:end, :] = sess.run(eval_prediction, feed_dict={
            eval_data: data[begin:end, ...]})
      else:
        batch_predictions = sess.run(eval_prediction, feed_dict={
            eval_data: data[-FLAGS.EVAL_BATCH_SIZE:, ...]})
        predictions[begin:, :] = batch_predictions[begin - size:, :]
  return predictions


def eval_result(ve, te, error, size, data_node, label_node, sess, eval_op, name):
  if ve <= error and te <= error:
    st = time.time()
    label_total = []
    prediction_total = []
    for vi in xrange(size // FLAGS.BATCH_SIZE):
      val_data, val_label = sess.run([data_node, label_node])
      predictions = eval_batch(val_data, sess, eval_op, eval_data, isTraining)
      label_total.extend(val_label)
      prediction_total.extend(predictions)
    label_total = np.array(label_total)
    prediction_total = np.array(prediction_total)
    er = error_rate(prediction_total, label_total)
    print('{} error: {:.3f}%'.format(name, er))
    print('{} costs {:.2f} seconds.'.format(name, time.time() - st))


def error_rate(predictions, labels):
  """Return the error rate based on dense predictions and sparse labels."""
  return 100.0 - (100.0 * np.sum(np.argmax(predictions, 1) == labels) /
                  predictions.shape[0])


def readCSV(filename):
  '''read lines from a csv file.
  '''
  lines = []
  with open(filename, "rb") as f:
    csvreader = csv.reader(f)
    for line in csvreader:
      lines.append(line)
  return lines


def init_bin_file():
  bin_file_name = [FLAGS.BIN_FILE]
  for f in bin_file_name:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)
  fqb = tf.train.string_input_producer(bin_file_name)
  record_bytes = (FLAGS.num_labels + FLAGS.NUM_IMAGE) * FLAGS.PIXEL_LENGTH
  rb = tf.FixedLengthRecordReader(record_bytes=record_bytes)
  return fqb, rb


def init_csv_file():
  csv_file_name = [FLAGS.CSV_FILE]
  for f in csv_file_name:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)
  fqc = tf.train.string_input_producer(csv_file_name)
  rc = tf.TextLineReader(skip_header_lines=True)
  return fqc, rc


def _get_data(fqb, rb):
  key, value = rb.read(fqb)
  record_bytes = tf.decode_raw(value, tf.float32)
  label = tf.cast(tf.slice(record_bytes, [0], [FLAGS.num_labels]), tf.float32)
  image = tf.reshape(tf.slice(record_bytes, [FLAGS.num_labels], [FLAGS.NUM_IMAGE]),
                     shape=[FLAGS.IMAGE_SIZE, FLAGS.IMAGE_SIZE, FLAGS.IMAGE_SIZE, 1])
  return label, image


def get_train_data(fqb, rb):
  label, image = _get_data(fqb, rb)
  min_queue_examples = FLAGS.BATCH_SIZE * 100
  labels, images = tf.train.shuffle_batch([label,image],FLAGS.BATCH_SIZE,capacity=min_queue_examples+386,min_after_dequeue=min_queue_examples)
  labels = tf.reshape(labels, [-1, FLAGS.num_labels])
  return labels, images

def readTestData():
  length = 2 * 4 + 32 * 32 * 32 * 4
  imgBuf = labelBuf = ''
  with open('./data/test.bin', 'rb') as ftest:
    buf = ftest.read(length)
    i = 0
    while buf:
      i += 1
      print(i)
      imgBuf += buf[8:]
      labelBuf += buf[:8]
      buf = ftest.read(length)
  test_data = (np.frombuffer(imgBuf, np.float32)).reshape((-1, 32, 32, 32, 1))
  test_label = np.frombuffer(labelBuf, np.float32).astype(np.int64)
  if FLAGS.num_labels > 1:
    test_labels = []
    for i in range(FLAGS.num_labels):
      label = []
      j = i
      while (j < len(test_label)):
        label.append(test_label[j])
        j += FLAGS.num_labels
        test_labels.append(label)
  else:
    test_labels = test_label
  return test_data, test_labels

def get_size():
  with open(FLAGS.CSV_FILE) as f:
    csvreader = csv.reader(f)
    size = 0
    for line in csvreader:
      if line[0] != 'candidateID':
        size += 1
  return size


def save3DSlice(image, cutCenter, path):
  if not os.path.exists(path):
    os.makedirs(path)
  imageTemp = image[cutCenter[0] - FLAGS.CUT_SIZE:cutCenter[0] + FLAGS.CUT_SIZE, ...]
  for index in np.arange(imageTemp.shape[0]):
    sliceTemp = imageTemp[index, :, :].copy()
    cv2.rectangle(sliceTemp, (cutCenter[2] - FLAGS.CUT_SIZE, cutCenter[1] - FLAGS.CUT_SIZE),
                  (cutCenter[2] + FLAGS.CUT_SIZE, cutCenter[1] + FLAGS.CUT_SIZE), (255, 255, 255))
    cv2.imwrite(path + str(index) + '.png', sliceTemp * 255)


def normalizePlanes(npzarray):
  npzarray = (npzarray - FLAGS.MIN_HU) / (FLAGS.MAX_HU - FLAGS.MIN_HU)
  npzarray[npzarray > 1] = 1.
  npzarray[npzarray < 0] = 0.
  return npzarray


def worldToVoxelCoord(worldCoord, origin, outputSpacing):
  stretchedVoxelCoord = np.absolute(worldCoord - origin)
  voxelCoord = stretchedVoxelCoord / outputSpacing
  return voxelCoord


def interpolatefilter(inputpath):
  inputimage = sitk.ReadImage(inputpath)
  origin = inputimage.GetOrigin()
  spacing = inputimage.GetSpacing()
  direction = inputimage.GetDirection()
  outputspacing = (spacing[0], spacing[0], spacing[0])
  size = inputimage.GetSize()
  tmp = int(spacing[2] * size[2] / spacing[0])
  if tmp % 2 != 0:
    tmp = tmp - 1

  outputsize = (size[0], size[1], tmp)
  resamplefilter = sitk.ResampleImageFilter()
  resamplefilter.SetOutputDirection(direction)
  resamplefilter.SetSize(outputsize)
  resamplefilter.SetOutputOrigin(origin)
  resamplefilter.SetOutputSpacing(outputspacing)
  outputimage = resamplefilter.Execute(inputimage)
  numpyImage = sitk.GetArrayFromImage(outputimage)
  numpyImage = normalizePlanes(numpyImage)
  return numpyImage, list(outputsize), spacing, outputspacing, origin


def createImageBorder(numpyImage, outputsize):
  BackImage = np.zeros(((FLAGS.BACK_SIZE, FLAGS.BACK_SIZE, FLAGS.BACK_SIZE)))
  BackImage[int(FLAGS.BACK_SIZE / 2 - outputsize[2] / 2):int(FLAGS.BACK_SIZE / 2 + outputsize[2] / 2),
            int(FLAGS.BACK_SIZE / 2 - outputsize[1] / 2):int(FLAGS.BACK_SIZE / 2 + outputsize[1] / 2),
            int(FLAGS.BACK_SIZE / 2 - outputsize[0] / 2):int(FLAGS.BACK_SIZE / 2 + outputsize[0] / 2)] = numpyImage
  return BackImage
