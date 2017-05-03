#!/usr/local/bin/python
# -*- coding: utf-8 -*-
'''3D convolutional neural network trained
   to reduce the False Positive Rate for the LUNA datasets.
   The LUNA datasets are organized in the CIFAR architecture.

   Author: Kong Haiyang
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
from luna import FLAGS
from luna import error_rate, readCSV
from luna import get_size
from luna import init_bin_file, init_csv_file
from luna import get_train_data
from luna import readTestData
# import tflearn
from tensorflow.contrib.layers import variance_scaling_initializer
from tensorflow.contrib.layers import l2_regularizer, apply_regularization
from tensorflow.contrib.losses import log_loss
from tensorflow.contrib.metrics import accuracy
#from tensorflow.contrib.layers import batch_norm

FLAGS.NUM_EPOCHS = 200
FLAGS.LABEL_NUMBER = 2
XAVIER_INIT = tf.contrib.layers.xavier_initializer(seed=FLAGS.SEED)
#XAVIER_INIT = tf.truncated_normal_initializer(stddev=0.1)
RELU_INIT = variance_scaling_initializer()
training = tf.placeholder_with_default(True, shape=())

Wb = {
    'W1': tf.get_variable('W1', [5, 5, 5, FLAGS.CHANNEL_NUMBER, 32], tf.float32, XAVIER_INIT),
    'b1': tf.Variable(tf.zeros([32])),
    'W2': tf.get_variable('W2', [3, 3, 3, 32, 48], tf.float32, XAVIER_INIT),
    'b2': tf.Variable(tf.zeros([48])),
    'W3': tf.get_variable('W3', [3, 3, 3, 48, 64], tf.float32, XAVIER_INIT),
    'b3': tf.Variable(tf.zeros([64])),
    'W4': tf.get_variable('W4', [3, 3, 3, 64, 128], tf.float32, XAVIER_INIT),
    'b4': tf.Variable(tf.zeros([128])),
    #'W5': tf.get_variable('W5', [3, 3, 3, 48, 64], tf.float32, XAVIER_INIT),
    #'b5': tf.Variable(tf.zeros([64])),
    'fcw1': tf.get_variable('fcw1', [2**3 * 128, 16], tf.float32, XAVIER_INIT),
    'fcb1': tf.Variable(tf.zeros([16])),
    'fcw2': tf.get_variable('fcw2', [16, FLAGS.LABEL_NUMBER], tf.float32, XAVIER_INIT),
    'fcb2': tf.Variable(tf.zeros([FLAGS.LABEL_NUMBER]))
}
'''
Wb = {
    'W1': tf.get_variable('W1', [5, 5, 5, FLAGS.CHANNEL_NUMBER, 16], tf.float32, XAVIER_INIT),
    'b1': tf.Variable(tf.zeros([16])),
    'W2': tf.get_variable('W2', [3, 3, 3, 16, 24], tf.float32, XAVIER_INIT),
    'b2': tf.Variable(tf.zeros([24])),
    'W3': tf.get_variable('W3', [3, 3, 3, 24, 32], tf.float32, XAVIER_INIT),
    'b3': tf.Variable(tf.zeros([32])),
    'W4': tf.get_variable('W4', [3, 3, 3, 32, 48], tf.float32, XAVIER_INIT),
    'b4': tf.Variable(tf.zeros([48])),
    'W5': tf.get_variable('W5', [3, 3, 3, 48, 64], tf.float32, XAVIER_INIT),
    'b5': tf.Variable(tf.zeros([64])),
    'fcw1': tf.get_variable('fcw1', [2**3 * 64, 32], tf.float32, XAVIER_INIT),
    'fcb1': tf.Variable(tf.zeros([32])),
    'fcw2': tf.get_variable('fcw2', [32, FLAGS.LABEL_NUMBER], tf.float32, XAVIER_INIT),
    'fcb2': tf.Variable(tf.zeros([FLAGS.LABEL_NUMBER]))
}

def spp_layer(input_, levels=[2,1], name='SPP_layer'):
  shape = input_.get_shape().as_list()
  print(shape)
  with tf.variable_scope(name):
    pool_outputs = []
    for l in levels:
      pool = tf.nn.max_pool3d(input_, ksize=[1, np.ceil(shape[1]*1./l).astype(np.int32),
                                             np.ceil(shape[2]*1./l).astype(np.int32),
                                             np.ceil(shape[3]*1./l).astype(np.int32), 1],
                              strides=[1, np.ceil(shape[1] * 1. / l).astype(np.int32),
                                       np.ceil(shape[2] * 1. / l ).astype(np.int32),
                                       np.ceil(shape[3] * 1. / l ).astype(np.int32), 1],
                              padding='SAME'
                              )
      print ('pool level {:}: shape {:}'.format(l, pool.get_shape().as_list()))
      pool_outputs.append(tf.reshape(pool, [tf.shape(input_)[0], -1]))
    spp_pool = tf.concat(1, pool_outputs)
  return spp_pool
  '''

def model(data, keep_prob):
  with tf.variable_scope('conv1') as scope:
    conv1 = tf.nn.conv3d(data, Wb['W1'], strides=[1, 1, 1, 1, 1], padding='VALID')
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, Wb['b1']))
    #tf.histogram_summary(scope.name + '/activations', relu1)
    #tf.histogram_summary(scope.name + '/weights', Wb['W1'])
    pool1 = tf.nn.max_pool3d(relu1, ksize=[1, 2, 2, 2, 1],
                            strides=[1, 2, 2, 2, 1], padding='VALID')
  with tf.variable_scope('conv2') as scope:
    conv2 = tf.nn.conv3d(pool1, Wb['W2'], strides=[1, 1, 1, 1, 1], padding='VALID')
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, Wb['b2']))
    #tf.histogram_summary(scope.name + '/activations', relu2)
    #tf.histogram_summary(scope.name + '/weights', Wb['W2'])
    pool2 = tf.nn.max_pool3d(relu2, ksize=[1, 2, 2, 2, 1],
                            strides=[1, 2, 2, 2, 1], padding='VALID')
  with tf.variable_scope('conv3') as scope:
    conv3 = tf.nn.conv3d(pool2, Wb['W3'], strides=[1, 1, 1, 1, 1], padding='VALID')
    relu3 = tf.nn.relu(tf.nn.bias_add(conv3, Wb['b3']))
    #tf.histogram_summary(scope.name + '/activations', relu3)
    #tf.histogram_summary(scope.name + '/weights', Wb['W3'])
    #pool3 = tf.nn.max_pool3d(relu3, ksize=[1, 2, 2, 2, 1],
                            #strides=[1, 2, 2, 2, 1], padding='VALID')
  with tf.variable_scope('conv4') as scope:
    conv4 = tf.nn.conv3d(relu3, Wb['W4'], strides=[1, 1, 1, 1, 1], padding='VALID')
    relu4 = tf.nn.relu(tf.nn.bias_add(conv4, Wb['b4']))
    #tf.histogram_summary(scope.name + '/activations', relu4)
    #tf.histogram_summary(scope.name + '/weights', Wb['W4'])
    #pool4 = tf.nn.max_pool3d(relu4, ksize=[1, 2, 2, 2, 1],
                            #strides=[1, 2, 2, 2, 1], padding='VALID')
  #with tf.variable_scope('conv5') as scope:
    #conv5 = tf.nn.conv3d(pool4, Wb['W5'], strides=[1, 1, 1, 1, 1], padding='SAME')
    #relu5 = tf.nn.relu(tf.nn.bias_add(conv5, Wb['b5']))
    #tf.histogram_summary(scope.name + '/activations', relu5)
    #tf.histogram_summary(scope.name + '/weights', Wb['W5'])

  #with tf.variable_scope('spp3') as scope:
    #spp3 = spp_layer(relu5, levels=[3,2,1])
  with tf.variable_scope('reshape'):
    ps = relu4.get_shape().as_list()
    reshape = tf.reshape(relu4, [-1, ps[1] * ps[2] * ps[3] * ps[4]])
  with tf.variable_scope('fc1'):
    hidden = tf.nn.relu(tf.matmul(reshape, Wb['fcw1']) + Wb['fcb1'])
  with tf.variable_scope('dropout'):
    hidden = tf.nn.dropout(hidden, keep_prob, seed=FLAGS.SEED)
  with tf.variable_scope('fc2'):
    out = tf.matmul(hidden, Wb['fcw2']) + Wb['fcb2']
  return out

def eval_in_batches(data, sess, eval_prediction, eval_data, keep_hidden):
  size = data.shape[0]
  if size < FLAGS.EVAL_BATCH_SIZE:
    raise ValueError("batch size for evals larger than dataset: %d" % size)
  predictions = np.ndarray(shape=(size, FLAGS.LABEL_NUMBER), dtype=np.float32)
  for begin in xrange(0, size, FLAGS.EVAL_BATCH_SIZE):
    end = begin + FLAGS.EVAL_BATCH_SIZE
    if end <= size:
      predictions[begin:end, :] = sess.run(eval_prediction, feed_dict={
          eval_data: data[begin:end, ...], keep_hidden: 1})
    else:
      batch_predictions = sess.run(eval_prediction, feed_dict={
          eval_data: data[-FLAGS.EVAL_BATCH_SIZE:, ...], keep_hidden: 1})
      predictions[begin:, :] = batch_predictions[begin - size:, :]
  return predictions


def lunaTrain(train_size, test_size):
  #define the parameters
  TRAIN_FREQUENCY = TEST_FREQUENCY = train_size // FLAGS.BATCH_SIZE
  SAVE_FREQUENCY = 10 * train_size // FLAGS.BATCH_SIZE

  #write the begining time
  st = time.time()

  #get train and test data
  fqbt, rbt = init_bin_file()
  train_label_node, train_data_node = get_train_data(fqbt, rbt)
  test_data, test_label = readTestData()

  #define placeholders
  data_node = tf.placeholder(tf.float32, shape=(None, FLAGS.IMAGE_SIZE, FLAGS.IMAGE_SIZE, FLAGS.IMAGE_SIZE, FLAGS.CHANNEL_NUMBER))
  labels_node = tf.placeholder(tf.int64, shape=(None, FLAGS.LABEL_NUMBER))
  keep_hidden = tf.placeholder(tf.float32)

  #start inference
  logits = model(data_node, keep_hidden)

  #compute the loss
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels_node), name='loss')
  tf.scalar_summary('loss', loss)

  #define the decay of the learing rate
  global_step = tf.Variable(0, name='global_step', trainable=False)
  learning_rate = tf.train.exponential_decay(0.1, global_step, 1000, 0.5, staircase=True)

  #define the optimizer
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)

  gradient_all = optimizer.compute_gradients(loss)
  # grads_vars = [v for (g, v) in gradient_all if g is not None]
  for i in range(len(gradient_all)):
    print(i)
    print(gradient_all[i][1])
    print(gradient_all[i][0])

  tf.summary.histogram('/grid_conv1', gradient_all[0][0])
  tf.summary.histogram('/grid_conv5', gradient_all[8][0])

  train_op = optimizer.minimize(loss, global_step=global_step)

  #compute the softmax for the output of model
  eval_predictions = tf.nn.softmax(logits)

  #define the summary merged op
  summary_op = tf.merge_all_summaries()

  #define the saver to save and restore the model
  saver = tf.train.Saver(tf.global_variables())

  with tf.Session() as sess:

    #initialize all variables
    init = tf.global_variables_initializer()
    sess.run(init)
    '''path = './models-73_train-malignancy-32-0.04241/'
    ckpt = tf.train.get_checkpoint_state(path)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
      saver.restore(sess, os.path.join(path, ckpt_name))
      print('Loading success, global_step is %s' % global_step)'''

    summary_writer = tf.summary.FileWriter(FLAGS.VIEW_PATH, sess.graph)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    try:
      while not coord.should_stop():
        start_time = time.time()
        for step in range(int(FLAGS.NUM_EPOCHS * train_size) // FLAGS.BATCH_SIZE):
          train_data, train_label = sess.run([train_data_node, train_label_node])
          train_label = np.where(train_label == [0], np.array([1, 0]), np.array([0, 1]))
          feed_dict = {data_node: train_data, labels_node: train_label, keep_hidden: 0.5}
          _, l, lr = sess.run([train_op, loss, learning_rate], feed_dict = feed_dict)
          if step % 20 == 0:
            summary_str = sess.run(summary_op, feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, step)

          if step != 0 and step % TRAIN_FREQUENCY == 0:
            et = time.time() - start_time
            print('Step %d (epoch %.2f), %.1f ms' %
                  (step, float(step) * FLAGS.BATCH_SIZE / train_size, 1000 * et / TRAIN_FREQUENCY))
            print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
            start_time = time.time()
          if step != 0 and step % TEST_FREQUENCY == 0:
            st = time.time()
            test_label_total = []
            prediction_total = []
            for ti in range(test_size // FLAGS.EVAL_BATCH_SIZE):
              offset = (ti * FLAGS.EVAL_BATCH_SIZE) % (test_size - FLAGS.EVAL_BATCH_SIZE)
              batch_data = test_data[offset:(offset + FLAGS.EVAL_BATCH_SIZE), ...]
              batch_labels = test_label[offset:(offset + FLAGS.EVAL_BATCH_SIZE)]
              predictions = eval_in_batches(
                  batch_data, sess, eval_predictions, data_node, keep_hidden)
              test_label_total.extend(batch_labels)
              prediction_total.extend(predictions)
            test_label_total = np.array(test_label_total)
            prediction_total = np.array(prediction_total)
            test_error = error_rate(prediction_total, test_label_total)
            print('Test error: %.3f%%' % test_error)
            print('Test costs {:.2f} seconds.'.format(time.time() - st)) #'''step % SAVE_FREQUENCY == 0'''
          if step % TEST_FREQUENCY == 0 and step != 0:
            if FLAGS.SAVE_MODEL:
              checkpoint_path = os.path.join(FLAGS.VIEW_PATH, 'model.ckpt')
              saver.save(sess, checkpoint_path, global_step=step)
        else:
          if FLAGS.SAVE_MODEL:
            checkpoint_path = os.path.join(FLAGS.VIEW_PATH, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step)
          coord.request_stop()
    except tf.errors.OutOfRangeError:
      print('Done training -- epoch limit reached')
    finally:
      pass
    coord.join(threads)
  print('All costs {:.2f} seconds...'.format(time.time() - st))
  train_data = train_labels = 0


def main(_):
  train_size = 7994 #{7994:536, 8480:90, 4146:112,}
  test_size = 536
  lunaTrain(train_size, test_size)

if __name__ == '__main__':
  tf.app.run()
