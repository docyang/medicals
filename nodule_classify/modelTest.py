# --*-- encoding: UTF-8 --*--

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
tf.GraphKeys.VARIABLES = tf.GraphKeys.GLOBAL_VARIABLES
from luna import FLAGS, batch_norm
from tensorflow.contrib.layers import variance_scaling_initializer

FLAGS.LABEL_NUMBER = 2
XAVIER_INIT = tf.contrib.layers.xavier_initializer(seed=FLAGS.SEED)
RELU_INIT = variance_scaling_initializer()

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
    'fcw1': tf.get_variable('fcw1', [36 * 64, 32], tf.float32, XAVIER_INIT),
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
def model(data, keep_prob):
  with tf.variable_scope('conv1') as scope:
    conv1 = tf.nn.conv3d(data, Wb['W1'], strides=[1, 1, 1, 1, 1], padding='SAME')
    #conv1 = batch_norm(conv1, False)
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, Wb['b1']))
    #tf.histogram_summary(scope.name + '/activations', relu1)
    #tf.histogram_summary(scope.name + '/weights', Wb['W1'])
    pool1 = tf.nn.max_pool3d(relu1, ksize=[1, 2, 2, 2, 1],
                            strides=[1, 2, 2, 2, 1], padding='VALID')
  with tf.variable_scope('conv2') as scope:
    conv2 = tf.nn.conv3d(pool1, Wb['W2'], strides=[1, 1, 1, 1, 1], padding='SAME')
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, Wb['b2']))
    #tf.histogram_summary(scope.name + '/activations', relu2)
    #tf.histogram_summary(scope.name + '/weights', Wb['W2'])
    pool2 = tf.nn.max_pool3d(relu2, ksize=[1, 2, 2, 2, 1],
                            strides=[1, 2, 2, 2, 1], padding='VALID')
  with tf.variable_scope('conv3') as scope:
    conv3 = tf.nn.conv3d(pool2, Wb['W3'], strides=[1, 1, 1, 1, 1], padding='SAME')
    relu3 = tf.nn.relu(tf.nn.bias_add(conv3, Wb['b3']))
    #tf.histogram_summary(scope.name + '/activations', relu3)
    #tf.histogram_summary(scope.name + '/weights', Wb['W3'])
    #pool3 = tf.nn.max_pool3d(relu3, ksize=[1, 2, 2, 2, 1],
                            #strides=[1, 2, 2, 2, 1], padding='VALID')
  with tf.variable_scope('conv4') as scope:
    conv4 = tf.nn.conv3d(relu3, Wb['W4'], strides=[1, 1, 1, 1, 1], padding='SAME')
    relu4 = tf.nn.relu(tf.nn.bias_add(conv4, Wb['b4']))
    #tf.histogram_summary(scope.name + '/activations', relu4)
    #tf.histogram_summary(scope.name + '/weights', Wb['W4'])
    #pool4 = tf.nn.max_pool3d(relu4, ksize=[1, 2, 2, 2, 1],
                            #strides=[1, 2, 2, 2, 1], padding='VALID')
  with tf.variable_scope('conv5') as scope:
    conv5 = tf.nn.conv3d(relu4, Wb['W5'], strides=[1, 1, 1, 1, 1], padding='SAME')
    relu5 = tf.nn.relu(tf.nn.bias_add(conv5, Wb['b5']))
    #tf.histogram_summary(scope.name + '/activations', relu5)
    #tf.histogram_summary(scope.name + '/weights', Wb['W5'])

  with tf.variable_scope('spp3') as scope:
    spp3 = spp_layer(relu5, levels=[3,2,1])
  #with tf.variable_scope('reshape'):
    #ps = relu5.get_shape().as_list()
    #reshape = tf.reshape(relu5, [-1, ps[1] * ps[2] * ps[3] * ps[4]])
  with tf.variable_scope('fc1'):
    hidden = tf.nn.relu(tf.matmul(spp3, Wb['fcw1']) + Wb['fcb1'])
  with tf.variable_scope('dropout'):
    hidden = tf.nn.dropout(hidden, keep_prob, seed=FLAGS.SEED)
  with tf.variable_scope('fc2'):
    out = tf.matmul(hidden, Wb['fcw2']) + Wb['fcb2']
    out = tf.nn.softmax(out)
  return out

datadir = './test_data/'
path = './models-73_train-malignancy-32-0.04241/'
saver = tf.train.Saver()
with tf.Session() as sess:
  ckpt = tf.train.get_checkpoint_state(path)
  if ckpt and ckpt.model_checkpoint_path:
    ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    saver.restore(sess, os.path.join(path, ckpt_name))
    print('Loading success, global_step is %s' % global_step)
  sum = 0
  count =0
  for dir1 in os.listdir(datadir):
    for dir2 in os.listdir(datadir + dir1):
      sum = sum + 1
      image = np.load(datadir + dir1 + '/' + dir2)
      shape = image.shape
      data = image.reshape([1, shape[0], shape[1], shape[2], 1])
      data = tf.cast(data, tf.float32)
      logit = model(data, 1.0)
      logit = sess.run(tf.nn.softmax(logit))[0].tolist()
      print (logit)
      if (dir1 == str(logit.index(max(logit)))):
        count = count + 1
      else:
        continue
  print (count, sum)
  print("test accuracy: {}".format(count/sum))