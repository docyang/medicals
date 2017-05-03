# --*-- encoding: UTF-8 --*--

import numpy as np
import tensorflow as tf


def ModelView(sess):
    new_saver = tf.train.import_meta_graph('models/model.ckpt-30000.meta')
    new_saver.restore(sess, 'models/model.ckpt-30000')
    all_vars = tf.trainable_variables()
    print sess.run(all_vars)
    for v in all_vars:
        print v.name

def GetWeight(sess):
    modelPath = 'models/'
    saver2 = tf.train.import_meta_graph(modelPath + 'model.ckpt-30000.meta')
    saver2.restore(sess, modelPath + 'model.ckpt-30000')
    all_vars = tf.trainable_variables()
    variables = sess.run(all_vars)
    fb2 = variables[-1]
    fw2 = variables[-2]
    fb1 = variables[-3]
    fw1 = variables[-4]
    fw2_conv = fw2.reshape([1, 1, 1, 16, 2])
    fw1_conv = fw1.reshape([4, 4, 4, 32, 16])
    value = np.array([fw1_conv, fb1, fw2_conv, fb2])
    np.save('./value', value)

def TransforModel(sess):
    value = np.load('value.npy')
    Wb = {
        'W1': tf.get_variable('W1', [5, 5, 5, 1, 32], tf.float32),
        'b1': tf.Variable(tf.zeros([32])),
        'W2': tf.get_variable('W2', [3, 3, 3, 32, 24], tf.float32),
        'b2': tf.Variable(tf.zeros([24])),
        'W3': tf.get_variable('W3', [3, 3, 3, 24, 24], tf.float32),
        'b3': tf.Variable(tf.zeros([24])),
        'W4': tf.get_variable('W4', [3, 3, 3, 24, 32], tf.float32),
        'b4': tf.Variable(tf.zeros([32])),
        #'fcw1': tf.get_variable('fcw1', [2**3 * 32, 16], tf.float32),
        'fcw1-conv': tf.Variable(value[0]),
        'fcb1-conv': tf.Variable(value[1]),
        #'fcw2': tf.get_variable('fcw2', [16, 2], tf.float32),
        'fcw2-conv': tf.Variable(value[2]),
        'fcb2-conv': tf.Variable(value[3])
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

    saver = tf.train.Saver()
    saver1 = tf.train.Saver(
        {'W1': Wb['W1'], 'Variable': Wb['b1'], 'W2': Wb['W2'], 'Variable_1': Wb['b2'], 'W3': Wb['W3'],
         'Variable_2': Wb['b3'], 'W4': Wb['W4'], 'Variable_3': Wb['b4']})
    init = tf.global_variables_initializer()
    sess.run(init)
    saver1.restore(sess, './models/model.ckpt-30000')
    saver.save(sess, './models_conv/model.ckpt')

def main():
    with tf.Session() as sess:
        #ModelView(sess)
        #GetWeight(sess)
        TransforModel(sess)
if __name__ == '__main__':
    main()

