#!/usr/bin/env python
#  encoding: utf-8
import os
import cv2
import numpy as np
from six.moves import xrange
from util import readCSV, normalize

csvpath = './csv_file/ggo.csv'
datapath = '../data/ggo/train/'

count = 1
lines = readCSV(csvpath)
for line in lines[1:]:
    series_uid = line[0]
    center_z = int(np.round(eval(line[1])[2]))
    num_z = int(line[-1])
    #print series_uid, center_z, num_z
    absolute_path = datapath + series_uid
    images = normalize(np.load(absolute_path + "/image.npy")) * 255
    print images.shape
    masks = np.load(absolute_path + "/ggo_mask.npy").astype(np.int)
    print masks.shape
    if num_z <= 3:
        seed_image = images[:, :, center_z]
        seed_mask = masks[:, :, center_z]
        np.save("./data/image/{}".format(count), seed_image)
        np.save("./data/mask/{}".format(count), seed_mask)
        count = count + 1
    else:
        for i in xrange(center_z-int(num_z/2)+1, center_z+int(num_z/2)):
            seed_image = images[:, :, i]
            seed_mask = masks[:, :, i]
            np.save("./data/image/{}".format(count), seed_image)
            np.save("./data/mask/{}".format(count), seed_mask)
            count = count + 1
    print count

# image = np.load("./data/1.3.6.1.4.1.14519.5.2.1.6279.6001.487268565754493433372433148666_0606_1002_0078.npy")
# mask = np.load("./data/1.3.6.1.4.1.14519.5.2.1.6279.6001.487268565754493433372433148666_0606_1002_0078_o.npy")*255
# cv2.imwrite("test/3.bmp", image)
# cv2.imwrite("test/3.png", mask)