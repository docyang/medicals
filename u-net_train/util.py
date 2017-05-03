from __future__ import division
import csv
import os
import numpy as np
import SimpleITK as sitk
from scipy import ndimage

def readCSV(filename):
    lines = []
    with open(filename, 'rb') as fr:
        reader = csv.reader(fr)
        for line in reader:
            lines.append(line)
    return lines

def normalize(image, max_hu=400, min_hu=-1000):
    image = (image - min_hu) / (max_hu - min_hu)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image

def readdcm(path):
    reader = sitk.ImageSeriesReader()
    filenames = reader.GetGDCMSeriesFileNames(path)
    reader.SetFileNames(filenames)
    simage = reader.Execute()
    images = sitk.GetArrayFromImage(simage)
    return images

def get_testdata():
    rootdir = './data/'
    test_image = []
    test_mask = []
    for dir1 in os.listdir(rootdir + 'test'):
        if dir1 == 'image':
            data = test_image
        else:
            data = test_mask
        for dir2 in os.listdir(rootdir + 'test' + '/' + dir1):
            data.append(np.load(rootdir + 'test' + '/' + dir1 + '/' + dir2).reshape([1, 512, 512]))
    return np.array(test_image), np.array(test_mask)

def generate_arrays_from_file(path):
    while 1:
        for sample in os.listdir(path):
            image = np.load(path + sample).reshape([1, 1, 512, 512])
            mask = np.load(path.replace('image', 'mask') + sample).reshape([1, 1, 512, 512])
            yield (image, mask)

def scipy_connect_components(mask, num=None, area_min=None):
    label_img, cc_num = ndimage.label(mask)
    cc_areas = ndimage.sum(mask, label_img, range(cc_num + 1))
    area_mask = np.zeros_like(cc_areas, np.bool)

    if area_min is not None:
        area_mask = (cc_areas < area_min)
    if num is not None:
        top_num_indice = cc_areas.argsort()[-num:][::-1]
        top_num_mask = np.zeros_like(area_mask, np.bool)
        top_num_mask[top_num_indice] = True
        area_mask = np.logical_or(area_mask, np.logical_not(top_num_mask))

    label_img[area_mask[label_img]] = 0
    labels = np.unique(label_img)
    label_img = np.searchsorted(labels, label_img)
    CC = ndimage.find_objects(label_img)
    return label_img, CC

def get_information(CC):
    point = [[(CC[i][0].stop+CC[i][0].start)/2, (CC[i][1].stop+CC[i][1].start)/2, (CC[i][2].stop+CC[i][2].start)/2]
           for i in range(len(CC))]
    box = [[CC[i][0].stop-CC[i][0].start, CC[i][1].stop-CC[i][1].start, CC[i][2].stop-CC[i][2].start]
           for i in range(len(CC))]
    return point, box
