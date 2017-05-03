# --*-- encoding: UTF-8 --*--

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from util import *
import tensorflow as tf
from sklearn.metrics import roc_curve, auc
from sklearn.feature_selection import chi2
import matplotlib.pyplot as plt
from tensorflow.contrib.layers import variance_scaling_initializer

LABEL_NUMBER = 2
XAVIER_INIT = tf.contrib.layers.xavier_initializer(seed=66478)
RELU_INIT = variance_scaling_initializer()

Wb = {
    'W1': tf.get_variable('W1', [5, 5, 5, 1, 32], tf.float32, XAVIER_INIT),
    'b1': tf.Variable(tf.zeros([32])),
    'W2': tf.get_variable('W2', [3, 3, 3, 32, 24], tf.float32, XAVIER_INIT),
    'b2': tf.Variable(tf.zeros([24])),
    'W3': tf.get_variable('W3', [3, 3, 3, 24, 24], tf.float32, XAVIER_INIT),
    'b3': tf.Variable(tf.zeros([24])),
    'W4': tf.get_variable('W4', [3, 3, 3, 24, 32], tf.float32, XAVIER_INIT),
    'b4': tf.Variable(tf.zeros([32])),
    'fcw1': tf.get_variable('fcw1', [2**3 * 32, 16], tf.float32, XAVIER_INIT),
    'fcb1': tf.Variable(tf.zeros([16])),
    'fcw2': tf.get_variable('fcw2', [16, LABEL_NUMBER], tf.float32, XAVIER_INIT),
    'fcb2': tf.Variable(tf.zeros([LABEL_NUMBER]))
}

def model(data, keep_prob):
    with tf.variable_scope('conv1') as scope:
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

    with tf.variable_scope('reshape'):
        ps = relu4.get_shape().as_list()
        reshape = tf.reshape(relu4, [-1, ps[1] * ps[2] * ps[3] * ps[4]])
    with tf.variable_scope('fc1'):
        hidden = tf.nn.relu(tf.matmul(reshape, Wb['fcw1']) + Wb['fcb1'])
    with tf.variable_scope('dropout'):
        hidden = tf.nn.dropout(hidden, keep_prob, seed=66478)

    with tf.variable_scope('fc2'):
        out = tf.matmul(hidden, Wb['fcw2']) + Wb['fcb2']
    return out

saver = tf.train.Saver(
        {'W1': Wb['W1'], 'Variable': Wb['b1'], 'W2': Wb['W2'], 'Variable_1': Wb['b2'], 'W3': Wb['W3'],
         'Variable_2': Wb['b3'], 'W4': Wb['W4'], 'Variable_3': Wb['b4'], 'fcw1': Wb['fcw1'],
         'Variable_4': Wb['fcb1'], 'fcw2': Wb['fcw2'], 'Variable_5': Wb['fcb2']})

with tf.Session() as sess:
    # load the model
    modelPath = '/home/yanghan/projects/20170317/models-malignancy-32-0.04492/model.ckpt-12341'
    saver.restore(sess, modelPath)

    batch_size = 32

    testpath = ['/home/yanghan/projects/20170317/data/test1.bin', '/home/yanghan/projects/20170317/data/test.bin']
    test_data = []
    test_label = []
    for test_path in testpath:
        length = 4 + 32 * 32 * 32 * 4
        imgBuf = labelBuf = ''
        with open(test_path, 'rb') as ftest:
            buf = ftest.read(length)
            i = 0
            while buf:
                i += 1
                print(i)
                imgBuf += buf[4:]
                labelBuf += buf[:4]
                buf = ftest.read(length)
        test_data.extend(list((np.frombuffer(imgBuf, np.float32)).reshape((-1, 32, 32, 32, 1))))
        test_label.extend(list(np.frombuffer(labelBuf, np.float32).astype(np.int64)))
    test_data = np.array(test_data)

    count = 0
    prob = []
    for j in range(int(len(test_label) / batch_size + 1)):
        if j == int(len(test_label) / batch_size):
            data = test_data[batch_size * j:]
            label = test_label[batch_size * j:]
        else:
            data = test_data[batch_size * j:batch_size * (j + 1)]
            label = test_label[batch_size * j:batch_size * (j + 1)]
        data = tf.cast(tf.reshape(np.array(data), [-1, 32, 32, 32, 1]), tf.float32)
        logit = model(data, 1.0)
        logit = sess.run(tf.nn.softmax(logit))
        for k in range(logit.shape[0]):
            prob.append(logit[k][1])

    FPR, TPR, thresholds = roc_curve(test_label, prob)
    roc_auc = auc(FPR, TPR)

    RightIndex = (TPR + (1 - FPR) - 1).tolist()
    print (RightIndex)
    print (thresholds)
    m = RightIndex.index(max(RightIndex))
    RightIndexVal = RightIndex[m]
    tpr_val = TPR[m]
    fpr_val = FPR[m]
    thresholds_val = thresholds[m]
    print(thresholds_val)

    print (roc_auc)
    print (count)

    plt.title('Receiver Operating Characteristic')
    plt.plot(FPR, TPR, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1.0])
    plt.ylim([0, 1.0])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()