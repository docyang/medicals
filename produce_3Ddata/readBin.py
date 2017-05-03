#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''This file is used to read the binary file
   which is organized in the format of CIFAR or MNIST.
   Then the image data will be writen to the files
   and the corresponding label will be printed on the Terminal.

   Attension: the label length is one Byte.
   If the length is not one Byte, you need modify the function accordingly.

   Author: Kong Haiyang
'''
import struct
import time
import numpy as np
import cv2
import os


def readCIFAR(filename, columns=512, rows=512, pixelTypeOfImg='byte',
              numOfImg=100, startNo=0, bufFile=True):
  '''Read the binary file of the CIFAR format.

  filename: the filename and path of the binary file
  columns, rows: the columns and rows of the image
  numOfImg: the number of images to be read and write out
  startNo: the start number of images to be read out and write
           start from 0 to read from the start (the first image)
  pixelTypeOfImg: the pixel type of the image, only support:
                  byte: length is 1 Byte, using '>B'
                  int32: length is 4 Bytes, using '>I'
                  float32: length is 4 Bytes, using '<f'
  bufFile: Boolean to indicate to write to file (True) or return buffer (False)
  '''
  f = open(filename, 'rb')
  if pixelTypeOfImg == 'byte':
    pixelLength = 1
    pattern = '>%dB'
  elif pixelTypeOfImg == 'int32':
    pixelLength = 4
    pattern = '>%dI'
  elif pixelTypeOfImg == 'float32':
    pixelLength = 4
    pattern = '<%df'
  else:
    raise TypeError('Only support the type of byte, int32 and float32.')
  length = 1 + columns * rows * pixelLength
  if bufFile:
    seekIndex = startNo * length
    f.seek(seekIndex)
    labelsum = []
    if not os.path.exists('test'):
      os.makedirs('test')
    bufTemp = f.read(length)
    count = 0
    while bufTemp:
      label = struct.unpack_from('>B', bufTemp, 0)[0]
      labelsum.append(label)

      im = struct.unpack_from(pattern % (columns * rows), bufTemp, 1)
      im = np.array(im).reshape(columns, rows)
      # im = np.rot90(np.fliplr(im))

      print 'save image {0} with label {1}...'.format(count, label)
      cv2.imwrite('test/read{}.png'.format(count), im * 255)
      bufTemp = f.read(length)
      count += 1
      if count == numOfImg:
        break

    f.close()

    with open('test/label.txt', 'w') as fl:
      for item in labelsum:
        fl.write('{0}\n'.format(item))
  else:
    bufLabel = bufImg = ''
    bufTemp = f.read(length)
    while bufTemp:
      bufLabel += bufTemp[0]
      bufImg += bufTemp[1:]
      bufTemp = f.read(length)
    f.close()
    return bufImg, bufLabel


def readCIFAR3D(filename, columns=32, rows=32, heights=32,
                numOfImg=1, startNo=0, bufFile=True):
  '''Read the binary file of the CIFAR format.

  filename: the filename and path of the binary file
  columns, rows: the columns and rows of the image
  numOfImg: the number of images to be read and write out
  startNo: the start number of images to be read out and write
           start from 0 to read from the start (the first image)
  bufFile: Boolean to indicate to write to file (True) or return buffer (False)
  '''
  pixelLength = 4

  def saveImageSlice(image, path):
    if not os.path.exists(path):
      os.makedirs(path)
    for index in np.arange(image.shape[0]):
      sliceTemp = image[index, :, :]
      cv2.imwrite(path + '/{}-{}.png'.format(startNo, index), sliceTemp * 255)
  f = open(filename, 'rb')
  length = 4 + columns * rows * heights * pixelLength
  if bufFile:
    seekIndex = startNo * length
    f.seek(seekIndex)
    bufTemp = f.read(length)
    count = 0
    while bufTemp:
      label = np.frombuffer(bufTemp[0:4], dtype=np.float32).astype(np.float32)
      im = (np.frombuffer(
          bufTemp[4:], dtype=np.float32)).reshape(columns, rows, heights)
      print 'save image {0} with label {1}...'.format(count, label)
      saveImageSlice(im, 'test/')
      bufTemp = f.read(length)
      count += 1
      if count == numOfImg:
        break
    f.close()
  else:
    bufLabel = bufImg = ''
    bufTemp = f.read(length)
    while bufTemp:
      bufLabel += bufTemp[0]
      bufImg += bufTemp[1:]
      bufTemp = f.read(length)
    f.close()
    return bufImg, bufLabel


def readMNIST(filename, labelname, pixelTypeOfImg='byte', numOfImg=100):
  '''Read the binary file of the MNIST format.

  filename: the filename and path of the binary file
  columns, rows: the columns and rows of the image
  numOfImg: the number of images to be read out and write
  pixelTypeOfImg: the pixel type of the image, only support:
                  byte: length is 1 Byte, using '>B'
                  int32: length is 4 Byte, using '>I'
                  float32: length is 4 Byte, using '<f'
  '''
  f = open(filename, 'rb')
  fl = open(labelname, 'rb')
  if pixelTypeOfImg == 'byte':
    pixelLength = 1
    pattern = '>%dB'
  elif pixelTypeOfImg == 'int32':
    pixelLength = 4
    pattern = '>%dI'
  elif pixelTypeOfImg == 'float32':
    pixelLength = 4
    pattern = '<%df'
  else:
    raise TypeError('Only support the type of byte, int32 and float32.')
  buf = f.read(16)
  _, imgNo, rows, columns = struct.unpack_from('>IIII', buf, 0)
  bufl = fl.read(8)
  _, _ = struct.unpack_from('>II', bufl, 0)
  labelsum = []
  length = rows * columns * pixelLength
  if not os.path.exists('test'):
    os.makedirs('test')
  bufl = fl.read(1)
  buf = f.read(length)
  count = 0
  while bufl:
    label = struct.unpack_from('>B', bufl, 0)[0]
    im = struct.unpack_from(pattern % (columns * rows), buf, 0)
    im = np.array(im).reshape(columns, rows)
    # im = np.rot90(np.fliplr(im))
    labelsum.append(label)
    print 'save image {0} with label {1}'.format(count, label)
    cv2.imwrite('test/read{}.png'.format(count), im * 255)
    bufl = fl.read(1)
    buf = f.read(length)
    count += 1
    if count == numOfImg:
      break
  f.close()
  fl.close()
  with open('test/label.txt', 'w') as fl:
    for item in labelsum:
      fl.write('{0}\n'.format(item))


def readMNIST2(filename, labelname, pixelTypeOfImg='byte', numOfImg=100):
  '''Read the binary file of the MNIST format without the starting numbers.

  filename: the filename and path of the binary file
  columns, rows: the columns and rows of the image
  numOfImg: the number of images to be read out and write
  pixelTypeOfImg: the pixel type of the image, only support:
                  byte: length is 1 Byte, using '>B'
                  int32: length is 4 Byte, using '>I'
                  float32: length is 4 Byte, using '<f'
  '''
  columns = 64
  rows = 64
  f = open(filename, 'rb')
  fl = open(labelname, 'rb')
  if pixelTypeOfImg == 'byte':
    pixelLength = 1
    pattern = '>%dB'
  elif pixelTypeOfImg == 'int32':
    pixelLength = 4
    pattern = '>%dI'
  elif pixelTypeOfImg == 'float32':
    pixelLength = 4
    pattern = '<%df'
  else:
    raise TypeError('Only support the type of byte, int32 and float32.')
  length = rows * columns * pixelLength
  if not os.path.exists('test'):
    os.makedirs('test')
  for i in xrange(numOfImg):
    bufl = fl.read(1)
    label = struct.unpack_from('>B', bufl, 0)[0]
    buf = f.read(length)
    im = struct.unpack_from(pattern % (columns * rows), buf, 0)
    im = np.array(im).reshape(columns, rows)
    # im = np.rot90(np.fliplr(im))
    print 'save image {0} with label {1}'.format(i, label)
    cv2.imwrite('test/{}.png'.format(i), im * 255)
  f.close()
  fl.close()


def main():
  # filename = '/home/kong/400G/LUNA_CM/LUNA.bin'
  # readCIFAR(filename, 64, 64, 'float32', numOfImg=200, startNo=0)
  # filename = '/home/kong/400G/official3D/view1.bin'
  filename = './data/train/com0.bin'
  #filename = '/home/yanghan/projects/20170317/data/test.bin'
  # filename = '/home/kong/4T/official3D_110W/shuffle3D64.bin'
  readCIFAR3D(filename, numOfImg=1, startNo=100000)
  # filename = '/home/kong/400G/LUNAimage.bin'
  # labelname = '/home/kong/400G/LUNAlabel.bin'
  # filename = '/home/kong/400G/LUNA_CM/LUNAimage.bin'
  # labelname = '/home/kong/400G/LUNA_CM/LUNAlabel.bin'
  # readMNIST(filename, labelname, pixelTypeOfImg='float32', numOfImg=200)

if __name__ == '__main__':
  main()
