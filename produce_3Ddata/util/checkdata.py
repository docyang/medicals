import numpy as np

num_labels = 3
length = 3*4 + 32* 32* 32* 4
imgBuf = labelBuf = ''
with open('./data/train/train.bin', 'rb') as ftest:
  buf = ftest.read(length)
  i = 0
  while buf:
    i += 1
    print(i)
    imgBuf += buf[12:]
    labelBuf += buf[:12]
    buf = ftest.read(length)
test_data = (np.frombuffer(imgBuf, np.float32)).reshape((-1, 32, 32, 32, 1))
test_label = np.frombuffer(labelBuf, np.float32).astype(np.int64)
print  test_label
labels = []
for i in range(num_labels):
    label = []
    j = i
    while(j < len(test_label)):
        label.append(test_label[j])
        j += 3
    labels.append(label)
print labels


'''import tensorflow as tf

sess = tf.Session()
new_saver = tf.train.import_meta_graph('/home/yanghan/projects/SPP/models/model.ckpt.meta')
new_saver.restore(sess, '/home/yanghan/projects/SPP/models/model.ckpt')
#new_saver = tf.train.import_meta_graph('/home/yanghan/projects/ModelUse/models/malignancy/model.ckpt-4186.meta')
#new_saver.restore(sess, '/home/yanghan/projects/ModelUse/models/malignancy/model.ckpt-4186')
all_vars = tf.trainable_variables()
print sess.run(all_vars)
for v in all_vars:
    print v.name
'''

'''
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
    return probs


if __name__ == '__main__':
    f = open('./prob.txt', 'wb')
    for dir in os.listdir('./mask/'):
        series_id = dir.split('.')[0]
        #series_id = '1.3.6.1.4.1.14519.5.2.1.6279.6001.511347030803753100045216493273'
        probs = read_dicoms('/home/yanghan/data/stage1/' + series_id, series_id, f)
    f.close()
'''
