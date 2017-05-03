import csv
import os
import numpy as np
import SimpleITK as sitk
import time
import sys
import pprocess
import struct
from scipy.ndimage.interpolation import zoom
from scipy import ndimage

CUT_SIZES = [32, 40, 60, 80]
SAVE_SIZE = 20


def createPatchCoodinates(centralCoor, size):
    topMostPoint = np.array(
        [centralCoor[0] - size / 2, centralCoor[1] - size / 2, centralCoor[2] - size / 2])
    pointOne = np.array([centralCoor[0] + size / 2, centralCoor[1] -
                         size / 2, centralCoor[2] - size / 2])
    pointTwo = np.array([centralCoor[0] - size / 2, centralCoor[1] +
                         size / 2, centralCoor[2] - size / 2])
    pointThree = np.array(
        [centralCoor[0] - size / 2, centralCoor[1] - size / 2, centralCoor[2] + size / 2])

    coordinates = np.vstack((topMostPoint, pointOne, pointTwo, pointThree))
    return coordinates


def translate(coordinates, translation):
    translation = np.vstack((translation, translation, translation, translation))
    coordinates = coordinates + translation
    return coordinates


def zoom_diy(coordinates, zoomCenter, scale):
    for i in range(coordinates.shape[0]):
        coordinates[i, :] = (coordinates[i, :] - zoomCenter) * scale + zoomCenter
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


def pointBoardcastToMatrix(coordinates, patchSize):
    topMostPoint, pointOne, pointTwo, pointThree = coordinates[
                                                       0], coordinates[1], coordinates[2], coordinates[3]

    point1VectorZ = np.linspace(
        0, pointOne[0] - topMostPoint[0], patchSize).reshape(patchSize, 1, 1)
    point1VectorY = np.linspace(
        0, pointOne[1] - topMostPoint[1], patchSize).reshape(patchSize, 1, 1)
    point1VectorX = np.linspace(
        0, pointOne[2] - topMostPoint[2], patchSize).reshape(patchSize, 1, 1)
    point2VectorZ = np.linspace(
        0, pointTwo[0] - topMostPoint[0], patchSize).reshape(patchSize, 1)
    point2VectorY = np.linspace(
        0, pointTwo[1] - topMostPoint[1], patchSize).reshape(patchSize, 1)
    point2VectorX = np.linspace(
        0, pointTwo[2] - topMostPoint[2], patchSize).reshape(patchSize, 1)
    point3VectorZ = np.linspace(0, pointThree[0] - topMostPoint[0], patchSize)
    point3VectorY = np.linspace(0, pointThree[1] - topMostPoint[1], patchSize)
    point3VectorX = np.linspace(0, pointThree[2] - topMostPoint[2], patchSize)

    xCoorMatrix = point1VectorX + point2VectorX + point3VectorX + topMostPoint[2]
    yCoorMatrix = point1VectorY + point2VectorY + point3VectorY + topMostPoint[1]
    zCoorMatrix = point1VectorZ + point2VectorZ + point3VectorZ + topMostPoint[0]

    coordinates = [zCoorMatrix, yCoorMatrix, xCoorMatrix]
    return coordinates


def neighbourInterpolate( coordinates, ori_image,isMask=False):
    if isMask:
        interpolated_image = ndimage.map_coordinates(ori_image, coordinates, order=0, cval=0)
    else:
        interpolated_image = ndimage.map_coordinates(ori_image, coordinates, order=1, cval=0)
    return interpolated_image
'''def neighbourInterpolate(coordinates, oriImage):
      coordinates = np.round(coordinates).astype(np.int16)
      zCoorMatrix, yCoorMatrix, xCoorMatrix = coordinates[0], coordinates[1], coordinates[2]
      shape = zCoorMatrix.shape
      zCoorLine = tuple(zCoorMatrix.reshape(1, -1)[0])
      yCoorLine = tuple(yCoorMatrix.reshape(1, -1)[0])
      xCoorLine = tuple(xCoorMatrix.reshape(1, -1)[0])
      totalCoor = [zCoorLine, yCoorLine, xCoorLine]
      interpolatedImage = oriImage[totalCoor].reshape(shape)
      return interpolatedImage'''


def readCSV(filename):
    '''read lines from a csv file.
    '''
    lines = []
    with open(filename, "rb") as f:
        csvreader = csv.reader(f)
        for line in csvreader:
            lines.append(line)
    return lines


def normalizePlanes(npzarray):
    maxHU = 400.
    minHU = -1000.
    npzarray = (npzarray - minHU) / (maxHU - minHU)
    npzarray[npzarray > 1] = 1.
    npzarray[npzarray < 0] = 0.
    return npzarray


# def saveImageSlice(image, path):
#   if not os.path.exists(path):
#     os.makedirs(path)
#   for index in np.arange(image.shape[0]):
#     sliceTemp = image[index, :, :]
#     cv2.imwrite(path + str(index) + '.png', sliceTemp * 255)


def worldToVoxelCoord(worldCoord, origin, outputSpacing):
    stretchedVoxelCoord = np.absolute(worldCoord - origin)
    voxelCoord = stretchedVoxelCoord / outputSpacing
    return voxelCoord


def view_bar(args, num, total):
    rate = 1.0 * num / total
    rate_num = int(rate * 100)
    r = '\rProcess {} task percentage (%): '.format(args) + '{}'.format(rate_num)
    sys.stdout.write(r)
    sys.stdout.flush()


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
    tmp = int(spacing[2] * size[2] / spacing[0])
    if tmp % 2 != 0:
        tmp = tmp - 1

    outputsize = (size[0], size[1], tmp)

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


def createImageBorder(numpyImage, outputsize):
    '''  padding too hard'''
    back_shape = (1200, 820, 820)
    back_image = np.zeros(back_shape)
    slices = [slice((back_shape[i] - int(outputsize[2 - i])) / 2,
                    (back_shape[i] + int(outputsize[2 - i])) / 2) for i in range(3)]
    back_image[slices] = numpyImage
    # BackImage=np.zeros(((1200, 600, 600)), np.float32)
    # BackImage[600 - int(outputsize[2] / 2):600 + int(outputsize[2] / 2),
    #           300 - int(outputsize[1] / 2):300 + int(outputsize[1] / 2),
    #           300 - int(outputsize[0] / 2):300 + int(outputsize[0] / 2)]=numpyImage
    return back_image


def read_dcm_files(file_dir):
    reader = sitk.ImageSeriesReader()
    filenames = reader.GetGDCMSeriesFileNames(file_dir)
    reader.SetFileNames(filenames)
    sitk_image = reader.Execute()
    spacing = sitk_image.GetSpacing()
    return sitk_image, spacing


def write_csv(processorID, line_start, line_end, lidc_dict):
    fBin = open(os.path.join(binOutputPath + '/' +
                             'com{}'.format(processorID) + '.bin'), 'wb')
    uidTemp = -1
    flag = False
    for line in xrange(line_start, line_end):
        cand = csvLines[line]
        rotateAngle = np.array(eval(cand[-3]),np.float)[::-1]
        displacement = np.array(eval(cand[-2]),np.int)[::-1]
        zoomScale = float(cand[-1])
        # rotateAngle = np.array(augLines[int(cand[-1])][1:4], np.float)[::-1]
        # displacement = np.array(augLines[int(cand[-1])][4:7], np.int)[::-1]
        # zoomScale = float(augLines[int(cand[-1])][-1])
	CUT_SIZE = 32
        '''if single_size:
            CUT_SIZE = 40
        else:
            bounding_box = eval(cand[5])
            max_side = max(bounding_box)
            if max_side <= 16:
                CUT_SIZE = CUT_SIZES[0]
            elif max_side <= 32:
                CUT_SIZE = CUT_SIZES[1]
            elif max_side <= 52:
                CUT_SIZE = CUT_SIZES[2]
            else:
                CUT_SIZE = CUT_SIZES[3]'''
        uidFlag = cand[1] != uidTemp
        if uidFlag or flag:
            uidTemp = cand[1]
	    if uidTemp not in lidc_dict.keys():
		continue
	    else:
            	mhdOriginalPath = lidc_dict[uidTemp]
            if uidFlag:
                inputimage, spacing = read_dcm_files(mhdOriginalPath)
            if spacing[2] > 5:
                errorList.append(uidTemp)
                flag = True
                continue
            interpolatedImage, outputsize, spacing, outputspacing, origin = interpolatefilter(
                inputimage)
            BackImage = createImageBorder(interpolatedImage, outputsize)  # padding, BUT too too HARD
            flag = False
            print '\n{} {}'.format(cand[1], CUT_SIZE)

        rotateCenter = np.round(np.array(cand[4:1:-1], np.float))
        rotateCenter = worldToVoxelCoord(rotateCenter, origin[::-1], outputspacing)
        # rotateCenter += (400 - np.array(outputsize[::-1]) / 2)
        rotateCenter[0] += (600 - outputsize[2] / 2)  # ?
        rotateCenter[1] += (410 - outputsize[1] / 2)  # ?
        rotateCenter[2] += (410 - outputsize[0] / 2)  # ?
        augImage = aug3d(BackImage, rotateCenter, rotateAngle,
                         displacement, zoomScale, CUT_SIZE)
        augImage = zoom(augImage, float(SAVE_SIZE) / CUT_SIZE)
        # saveImageSlice(augImage, 'test/')
        # imageCompare = BackImage[rotateCenter[0] - 20:rotateCenter[0] + 20,
        #                          rotateCenter[1] - 20:rotateCenter[1] + 20,
        #                          rotateCenter[2] - 20:rotateCenter[2] + 20]
        # saveImageSlice(imageCompare, 'test1/')
        # fBin.write(chr(int(cand[-2])))
        # labels = eval(cand[-2])
        # for item in labels:
        #   fBin.write(struct.pack('<f', item))
        fBin.write(struct.pack('<f', float(cand[-5])))
        augImage.astype('float32').tofile(fBin)

        # real time plot the task progress
        if (line - line_start) % 10 == 0:
            view_bar(1, line - line_start, line_end - line_start - 1)
        elif line == line_end - line_start - 1:
            view_bar(1, line - line_start, line_end - line_start - 1)

    fBin.close()


# Global parameters

binOutputPath = './data/train/'
csvName = 'train_nodule_lobulation.csv'
#binOutputPath = '/home/admin6/yh/dataTransform/data/test/'
#csvName = 'test_nodule_lobulation.csv'
LIDCPath = '/mnt/1T/LIDC-IDRI'
kagglePath = '/mnt/1T/bowl/stage1'
workspace = './CSV/'
csvLines = readCSV(os.path.join(workspace,csvName))


ProcessorNum = 6
errorList = []


def get_series_uid_path_dict(lidc_idri_path):
    path_dict = {}
    case_names = os.listdir(lidc_idri_path)
    for case in case_names:
        absolute_path = os.path.join(lidc_idri_path, case)
        for root, dir_names, file_names in os.walk(absolute_path):
            for file_name in file_names:
                if file_name.endswith('.xml'):
                    series_uid = os.path.split(root)[-1]
                    path_dict[series_uid] = root
    kaggleDict = {}
    for line in csvLines:
        if not line[1].startswith('1.'):
            kaggleDict[line[1]] = os.path.join(kagglePath, line[1][:32])
    path_dict.update(kaggleDict)
    return path_dict


def main():
    lidc_dict = get_series_uid_path_dict(LIDCPath)
    list_of_args = range(ProcessorNum)
    group_num = len(csvLines) // ProcessorNum
    cutPoint = np.empty([ProcessorNum, 2], dtype=int)
    for row in range(ProcessorNum):
        # start point
        cutPoint[row, 0] = row * group_num
        if row == ProcessorNum - 1:
            # stop point
            cutPoint[row, 1] = len(csvLines)
        else:
            # stop point
            cutPoint[row, 1] = row * group_num + group_num - 1 + 1

    # starting parallel reading
    st = time.time()
    results = pprocess.Map()
    parallel_function = results.manage(pprocess.MakeParallel(write_csv))
    for args in list_of_args:
        parallel_function(args, cutPoint[args, 0], cutPoint[args, 1], lidc_dict)
    print('\nStarting Parallel time {:.2f} seconds...'.format(time.time() - st))

    st = time.time()
    results[:]
    # parallel_results = results[:]
    print('\nParallel costs {:.2f} seconds...'.format(time.time() - st))


def main1():
    lidc_dict = get_series_uid_path_dict(LIDCPath)
    write_csv(0, 0, len(csvLines), lidc_dict)
    print(errorList)


if __name__ == '__main__':
    main()
    print 'finished'
