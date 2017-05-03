# --*-- encoding: UTF-8 --*--

import os
from util import *
import tensorflow as tf

#Get the input data of the model.
def get_data(image, mask):
    image, mask = interpolatefilter(image, mask)
    label_img, CC = scipy_connect_components(mask)
    point, box = get_information(CC)
    samples = get_sample_images(image, point, box)
    return samples, label_img

#Get the prediction of the malignancy
def malignancy_infer(samples, mini_bs=30, pred=0.511212):
    with tf.Session() as sess:
        #load the model
        modelPath = './models/malignancy/'
        saver = tf.train.import_meta_graph(modelPath + 'model.ckpt-4186.meta')
        saver.restore(sess, modelPath+ 'model.ckpt-4186')
        keep_prob = tf.get_collection("keep_prob")[0]
        input = tf.get_collection("data")[0]
        logit = tf.get_collection("out")[0]
        flag = 1
        pos_sample = {}
        for j in range(len(samples)/mini_bs+1):
            if j == len(samples)/mini_bs:
                data = samples[mini_bs*j:]
            else:
                data = samples[mini_bs*j:mini_bs*(j+1)]
            data = sess.run(tf.cast(tf.reshape(np.array(data), [-1, 32, 32, 32, 1]), tf.float32))
            logit = sess.run(tf.nn.softmax(logit), feed_dict={input:data, keep_prob:1.0})
            for k in range(logit.shape[0]):
                if logit[k][1] >= pred:
                    pos_sample[flag*(k+1)] = 1
                else:
                    pos_sample[flag*(k+1)] = 0
            flag = flag + 1
        return pos_sample


def lobulation_infer(samples, mini_bs=30, pred=0.511212):
    with tf.Session() as sess:
        #load the model
        modelPath = './models/lobulation/'
        saver = tf.train.import_meta_graph(modelPath + 'model.ckpt-6138.meta')
        saver.restore(sess, modelPath + 'model.ckpt-6138')
        keep_prob = tf.get_collection("keep_prob")[0]
        input = tf.get_collection("data")[0]
        logit = tf.get_collection("out")[0]
        flag = 1
        pos_sample = {}
        for j in range(len(samples)/mini_bs+1):
            if j == len(samples)/mini_bs:
                data = samples[mini_bs*j:]
            else:
                data = samples[mini_bs*j:mini_bs*(j+1)]
            data = sess.run(tf.cast(tf.reshape(np.array(data), [-1, 32, 32, 32, 1]), tf.float32))
            logit = sess.run(tf.nn.softmax(logit), feed_dict={input: data, keep_prob: 1.0})
            for k in range(logit.shape[0]):
                if logit[k][1] >= pred:
                    pos_sample[flag*(k+1)] = 1
                else:
                    pos_sample[flag*(k+1)] = 0
            flag = flag + 1
        return  pos_sample

def spiculation_infer(samples, mini_bs=30, pred=0.511212):
    with tf.Session() as sess:
        #load the model
        modelPath = './models/spiculation/'
        saver = tf.train.import_meta_graph(modelPath + 'model.ckpt-7776.meta')
        saver.restore(sess, modelPath + 'model.ckpt-7776')
        keep_prob = tf.get_collection("keep_prob")[0]
        input = tf.get_collection("data")[0]
        logit = tf.get_collection("out")[0]
        flag = 1
        pos_sample = {}
        for j in range(len(samples)/mini_bs+1):
            if j == len(samples)/mini_bs:
                data = samples[mini_bs*j:]
            else:
                data = samples[mini_bs*j:mini_bs*(j+1)]
            data = sess.run(tf.cast(tf.reshape(np.array(data), [-1, 32, 32, 32, 1]), tf.float32))
            logit = sess.run(tf.nn.softmax(logit), feed_dict={input: data, keep_prob: 1.0})
            for k in range(logit.shape[0]):
                if logit[k][1] >= pred:
                    pos_sample[flag*(k+1)] = 1
                else:
                    pos_sample[flag*(k+1)] = 0
            flag = flag + 1
        return pos_sample


def read_dicoms(path, series_id, f):
    seriesreader = sitk.ImageSeriesReader()
    gdcmnames = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(path)
    seriesreader.SetFileNames(gdcmnames)
    simage = seriesreader.Execute()

    mask = np.load('./mask/{}.npy'.format(series_id)) * 1
    print mask.shape
    samples, label_img = get_data(simage, mask)
    for i in range(len(samples)):
        np.save('./samples/{}_{}'.format(series_id, i+1), samples[i])
    probs = malignancy_infer(samples)
    f.write(series_id + str(probs) + '\n')
    print probs
    return probs


if __name__ == '__main__':
    f = open('./prob.txt', 'wb')
    for dir in os.listdir('./mask/'):
        series_id = dir.split('.')[0]
        #series_id = '1.3.6.1.4.1.14519.5.2.1.6279.6001.511347030803753100045216493273'
        probs = read_dicoms('/home/yanghan/data/stage1/' + series_id, series_id, f)
    f.close()
