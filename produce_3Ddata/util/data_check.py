import SimpleITK as sitk
import numpy as np
import os
import cv2

def normalizePlanes(npzarray):
    maxHU = 400.
    minHU = -1000.
    npzarray = (npzarray - minHU) / (maxHU - minHU)
    npzarray[npzarray > 1] = 1.
    npzarray[npzarray < 0] = 0.
    return npzarray

path = '/home/yanghan/data/DOI/LUNGx-CT056/'
for dir1 in os.listdir(path):
    for dir2 in os.listdir(path + dir1):
        absolute_path = path + dir1 + '/' + dir2
        reader = sitk.ImageSeriesReader()
        filenames = reader.GetGDCMSeriesFileNames(absolute_path)
        reader.SetFileNames(filenames)
        sitk_image = reader.Execute()
        #print sitk_image.GetDirection()
        print sitk_image
        images = sitk.GetArrayFromImage(sitk_image)
        print np.min(images)
        images = normalizePlanes(images)
        image = images[211]*255
        cv2.imwrite('./test.png', image)
        roi = image[218-20:218+20, 374-20:374+20]
        cv2.imwrite('roi.png', roi)
