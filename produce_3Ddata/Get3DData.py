# --*-- coding:utf-8 --*--

import os
import sys
import csv
import time
import struct
import pprocess
import numpy as np
import SimpleITK as sitk
from scipy.ndimage.interpolation import zoom

CUT_SIZES = [24, 40, 60, 80]
SAVE_SIZE = 32

def ReadCSV(filename):
    lines = []
    with open(filename, 'rb') as fr:
        reader = csv.reader(fr)
        for line in reader:
            lines.append(line)
    return lines

def WriteCSV(filename, lines):
    with open(filename, 'wb') as fw:
        writer = csv.writer(fw)
        writer.writerows(lines)

def view_bar(args, num, total):
      rate = 1.0 * num / total
      rate_num = int(rate * 100)
      r = '\rProcess {} task percentage (%): '.format(args) + '{}'.format(rate_num)
      sys.stdout.write(r)
      sys.stdout.flush()

def ReadDcmFiles(path):
    reader = sitk.ImageSeriesReader()
    filenames = reader.GetGDCMSeriesFileNames(path)
    reader.SetFileNames(filenames)
    try:
        simage = reader.Execute()
    except:
        print "There is not this file"
    spacing = simage.GetSpacing()
    return simage, spacing

def normalizePlanes(npzarray):
    maxHU = 400.
    minHU = -1000.
    npzarray = (npzarray - minHU) / (maxHU - minHU)
    npzarray[npzarray > 1] = 1.
    npzarray[npzarray < 0] = 0.
    return npzarray

def worldToVoxelCoord(worldCoord, origin, outputSpacing):
    stretchedVoxelCoord = np.absolute(worldCoord - origin)
    voxelCoord = stretchedVoxelCoord / outputSpacing
    return voxelCoord

def createImageBorder(numpyImage, outputsize):
    back_shape = (1200, 620, 620)
    back_image = np.zeros(back_shape)
    slices = [slice((back_shape[i] - int(outputsize[2 - i])) / 2,(back_shape[i] + int(outputsize[2 - i])) / 2)
              for i in range(3)]
    back_image[slices] = numpyImage
    return back_image

def createPatchCoodinates(centralCoor, size):
    topMostPoint = np.array([centralCoor[0] - size / 2, centralCoor[1] - size / 2, centralCoor[2] - size / 2])
    pointOne = np.array([centralCoor[0] + size / 2, centralCoor[1] - size / 2, centralCoor[2] - size / 2])
    pointTwo = np.array([centralCoor[0] - size / 2, centralCoor[1] + size / 2, centralCoor[2] - size / 2])
    pointThree = np.array([centralCoor[0] - size / 2, centralCoor[1] - size / 2, centralCoor[2] + size / 2])
    coordinates = np.vstack((topMostPoint, pointOne, pointTwo, pointThree))
    return coordinates

def translate(coordinates, translation):
    translation = np.vstack((translation, translation, translation, translation))
    coordinates = coordinates + translation
    return coordinates

def rotate(coordinates, rotateCenter, rotateAngle):
      rotation = sitk.Euler3DTransform()
      rotation.SetCenter(rotateCenter)
      rotation.SetRotation(rotateAngle[0], rotateAngle[1], rotateAngle[2])
      coordinates = np.array(coordinates, np.int32)
      for i in range(coordinates.shape[0]):
            point = np.array(coordinates, np.float)
            point = tuple(point[i, :])
            rotatePoint = rotation.TransformPoint(point)
            coordinates[i, :] = np.array(rotatePoint)
      return coordinates

def zoom_diy(coordinates, zoomCenter, scale):
      for i in range(coordinates.shape[0]):
            coordinates[i, :] = (coordinates[i, :] - zoomCenter) * scale + zoomCenter
      return coordinates

def pointBoardcastToMatrix(coordinates, patchSize):
      topMostPoint, pointOne, pointTwo, pointThree = coordinates[0], coordinates[1], coordinates[2], coordinates[3]
      point1VectorZ = np.linspace(0, pointOne[0] - topMostPoint[0], patchSize).reshape(patchSize, 1, 1)
      point1VectorY = np.linspace(0, pointOne[1] - topMostPoint[1], patchSize).reshape(patchSize, 1, 1)
      point1VectorX = np.linspace(0, pointOne[2] - topMostPoint[2], patchSize).reshape(patchSize, 1, 1)
      point2VectorZ = np.linspace(0, pointTwo[0] - topMostPoint[0], patchSize).reshape(patchSize, 1)
      point2VectorY = np.linspace(0, pointTwo[1] - topMostPoint[1], patchSize).reshape(patchSize, 1)
      point2VectorX = np.linspace(0, pointTwo[2] - topMostPoint[2], patchSize).reshape(patchSize, 1)
      point3VectorZ = np.linspace(0, pointThree[0] - topMostPoint[0], patchSize)
      point3VectorY = np.linspace(0, pointThree[1] - topMostPoint[1], patchSize)
      point3VectorX = np.linspace(0, pointThree[2] - topMostPoint[2], patchSize)
      xCoorMatrix = point1VectorX + point2VectorX + point3VectorX + topMostPoint[2]
      yCoorMatrix = point1VectorY + point2VectorY + point3VectorY + topMostPoint[1]
      zCoorMatrix = point1VectorZ + point2VectorZ + point3VectorZ + topMostPoint[0]
      coordinates = [zCoorMatrix, yCoorMatrix, xCoorMatrix]
      return coordinates

def neighbourInterpolate(coordinates, oriImage):
      coordinates = np.round(coordinates).astype(np.int16)
      zCoorMatrix, yCoorMatrix, xCoorMatrix = coordinates[0], coordinates[1], coordinates[2]
      shape = zCoorMatrix.shape
      zCoorLine = tuple(zCoorMatrix.reshape(1, -1)[0])
      yCoorLine = tuple(yCoorMatrix.reshape(1, -1)[0])
      xCoorLine = tuple(xCoorMatrix.reshape(1, -1)[0])
      totalCoor = [zCoorLine, yCoorLine, xCoorLine]
      interpolatedImage = oriImage[totalCoor].reshape(shape)
      return interpolatedImage

def aug3d(borderedImage, rotateCenter, rotateAngle, displacement, zoomScale, patchSize):
    coordinates = createPatchCoodinates(rotateCenter, patchSize)
    coordinates = translate(coordinates, displacement)
    coordinates = rotate(coordinates, rotateCenter, rotateAngle)
    coordinates = zoom_diy(coordinates, rotateCenter, zoomScale)
    coordinates = pointBoardcastToMatrix(coordinates, patchSize)
    interpolatedImage = neighbourInterpolate(coordinates, borderedImage)
    return interpolatedImage

def interpolatefilter(inputimage):
    origin = inputimage.GetOrigin()
    spacing = inputimage.GetSpacing()
    direction = inputimage.GetDirection()
    outputspacing = (spacing[0], spacing[0], spacing[0])
    size = inputimage.GetSize()
    tmp = []
    for i in range(3):
        s = int(spacing[i] * size[i] / outputspacing[i])
        if s % 2 != 0:
            s = s - 1
        tmp.append(s)
    outputsize = (tmp[0], tmp[1], tmp[2])

    resamplefilter = sitk.ResampleImageFilter()
    resamplefilter.SetOutputDirection(direction)
    resamplefilter.SetSize(outputsize)
    resamplefilter.SetOutputOrigin(origin)
    resamplefilter.SetOutputSpacing(outputspacing)
    outputimage = resamplefilter.Execute(inputimage)
    outputsize = outputimage.GetSize()
    numpyImage = sitk.GetArrayFromImage(outputimage)
    numpyImage = normalizePlanes(numpyImage)
    return numpyImage, list(outputsize), spacing, outputspacing, origin

def write_csv(processorID, line_start, line_end, lidc_dict, csvlines, auglines, datapath, single_size=True):
    uidTemp = -1
    flag = False
    fBin = open(os.path.join(datapath + '/' + 'com{}'.format(processorID) + '.bin'), 'wb')
    for i in xrange(line_start, line_end):
        cand = csvlines[i]
        rotateAngle = np.array(auglines[int(cand[-1])][1:4], np.float)[::-1]
        displacement = np.array(auglines[int(cand[-1])][4:7], np.int)[::-1]
        zoomScale = float(auglines[int(cand[-1])][-1])
        if single_size:
            CUT_SIZE = 56
        else:
            bounding_box = eval(cand[6])
            max_side = max(bounding_box)
            if max_side <= 16:
                CUT_SIZE = CUT_SIZES[0]
            elif max_side <= 32:
                CUT_SIZE = CUT_SIZES[1]
            elif max_side <= 52:
                CUT_SIZE = CUT_SIZES[2]
            else:
                CUT_SIZE = CUT_SIZES[3]
        uidFlag = cand[2] != uidTemp
        if uidFlag or flag:
            uidTemp = cand[2]
            mhdOriginalPath = lidc_dict[uidTemp]
        if uidFlag:
            simage, spacing = ReadDcmFiles(mhdOriginalPath)
        if spacing[2] > 5:
            flag = True
            continue
        interpolatedImage, outputsize, spacing, outputspacing, origin = interpolatefilter(simage)
        BackImage = createImageBorder(interpolatedImage, outputsize)
        flag = False
        print '\n{} {}'.format(cand[2], CUT_SIZE)
        rotateCenter = np.round(np.array(cand[4 + 1:1 + 1:-1], np.float))
        rotateCenter = worldToVoxelCoord(rotateCenter, origin[::-1], outputspacing)
        rotateCenter[0] += (600 - outputsize[2] / 2)
        rotateCenter[1] += (310 - outputsize[1] / 2)
        rotateCenter[2] += (310 - outputsize[0] / 2)
        augImage = aug3d(BackImage, rotateCenter, rotateAngle, displacement, zoomScale, CUT_SIZE)
        augImage = zoom(augImage, float(SAVE_SIZE) / CUT_SIZE)
        fBin.write(struct.pack('<f', float(cand[-2])))
        #np.save(datapath + '{}'.format(cand[0]))
        augImage.astype('float32').tofile(fBin)

        if (i - line_start) % 10 == 0:
            view_bar(1, i - line_start, line_end - line_start - 1)
        elif i == line_end - line_start - 1:
            view_bar(1, i - line_start, line_end - line_start - 1)
    fBin.close()

def GetAbsolutePath(lidcpath, KAGGLEPath, csvlines):
    csvLines = csvlines
    path_dict = {}
    case_names = os.listdir(lidcpath)
    for case in case_names:
        absolute_path = os.path.join(lidcpath, case)
        for root, dir_names, file_names in os.walk(absolute_path):
            for file_name in file_names:
                if file_name.endswith('.xml'):
                    series_uid = os.path.split(root)[-1]
                    path_dict[series_uid] = root
    kaggleDict = {}
    for line in csvLines:
        if not line[2].startswith('1.'):
            kaggleDict[line[2]] = os.path.join(KAGGLEPath, line[2][:32])
    path_dict.update(kaggleDict)
    return path_dict

def Get3DCube(csvpath, augpath, datapath):
    csvLines = ReadCSV(csvpath)[1:]
    augLines = ReadCSV(augpath)[1:]
    LIDCPath = '/home/yanghan/data/LIDC-IDRI'
    KAGGLEPath = '/home/yanghan/data/stage1'
    lidc_dict = GetAbsolutePath(LIDCPath, KAGGLEPath, csvLines)
    write_csv(0, 0, len(csvLines), lidc_dict, csvLines, augLines, datapath)

def Get3DCube_multiprocess(csvpath, augpath, datapath):
    PPNum = 6
    augLines = ReadCSV(augpath)[1:]
    csvLines = ReadCSV(csvpath)[1:]
    LIDCPath = '/home/yanghan/data/LIDC-IDRI'
    KAGGLEPath = '/home/yanghan/data/stage1'
    lidc_dict = GetAbsolutePath(LIDCPath, KAGGLEPath, csvLines)
    list_pp = range(PPNum)
    group_num = len(csvLines) // PPNum
    cutPoint = np.empty([PPNum, 2], dtype=int)
    for row in range(PPNum):
        # start point
        cutPoint[row, 0] = row * group_num
        if row == PPNum - 1:
            # stop point
            cutPoint[row, 1] = len(csvLines)
        else:
            # stop point
            cutPoint[row, 1] = row * group_num + group_num - 1 + 1

    # starting parallel reading
    st = time.time()
    results = pprocess.Map()
    parallel_function = results.manage(pprocess.MakeParallel(write_csv(0, 0, len(csvLines),
                                                                       lidc_dict, csvLines, augLines, datapath, 1)))
    for args in list_pp:
        parallel_function(args, cutPoint[args, 0], cutPoint[args, 1], lidc_dict)
    print('\nStarting Parallel time {:.2f} seconds...'.format(time.time() - st))

    st = time.time()
    results[:]
    # parallel_results = results[:]
    print('\nParallel costs {:.2f} seconds...'.format(time.time() - st))