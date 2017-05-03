import os
import csv
import fnmatch
import SimpleITK as sitk
# from scipy.misc import imsave
import numpy as np
import xml.dom.minidom
import cv2
import inspect
from scipy.ndimage.morphology import binary_fill_holes

def file_search(dir, dataFormat='mhd', resultIsDir=False):
    resultList=[]
    for root, dirnames, filenames in os.walk(dir):
        for filename in fnmatch.filter(filenames, '*.' + dataFormat):
            if resultIsDir:
                resultList.append(root)
            else:
                resultList.append(os.path.join(root, filename))
    return resultList

def getDicomDir(dir, uid, resultList, dataFormat='mhd', isRecursive=True):
    for root, dirnames, filenames in os.walk(dir):
        for filename in fnmatch.filter(filenames, '*.' + dataFormat):
            if dataFormat.lower() == 'mhd':
                if (filename == uid + '.mhd'):
                    resultList.append(os.path.join(dir, filename))
                    break
            elif dataFormat.lower() == 'dcm':
                if (os.path.basename(dir) == uid):
                    resultList.append(dir)
                break
        if isRecursive:
            for dirname in dirnames:
                getDicomDir(os.path.join(dir, dirname), uid, resultList, dataFormat)


def getMetaInfo(filepath,dataFormat='dcm'):
    if dataFormat.lower() == 'mhd':
        inputimage = sitk.ReadImage(filepath)
        filenames = filepath
    elif dataFormat.lower() == 'dcm':
        reader = sitk.ImageSeriesReader()
        filenames = reader.GetGDCMSeriesFileNames(filepath)
        reader.SetFileNames(filenames)
        inputimage = reader.Execute()
    else:
        print "Unsupport file type: %s. Code line %d in function getSpacingInfo()." % (dataFormat, inspect.stack()[0][2])
        exit()
    spacingInfo={}
    spacingInfo['spacing'] = inputimage.GetSpacing()
    spacingInfo['origin'] = inputimage.GetOrigin()
    spacingInfo['direction'] = inputimage.GetDirection()
    spacingInfo['depth'] = inputimage.GetDepth()
    spacingInfo['width'] = inputimage.GetWidth()
    spacingInfo['height'] = inputimage.GetHeight()
    spacingInfo['filenames'] = filenames
    spacingInfo['image'] = inputimage
    return spacingInfo



def getAnnotationPoints(xmlfile,spacingInfo):
    # xmlfile = '/home/bb/Downloads/LIDC/LIDC-IDRI-0001/1.3.6.1.4.1.14519.5.2.1.6279.6001.298806137288633453246975630178/1.3.6.1.4.1.14519.5.2.1.6279.6001.179049373636438705059720603192/069.xml'
    dom = xml.dom.minidom.parse(xmlfile)

    slice_annot = {}
    reading_sessions = dom.getElementsByTagName('readingSession')
    sessionID = 0
    malignancy_data_set = []
    for session in reading_sessions:
        sessionID += 1
        nodules = session.getElementsByTagName('unblindedReadNodule')
        for nodule in nodules:
            characteristics = nodule.getElementsByTagName('characteristics')
            if characteristics:
                features = {}
                for characteristic in characteristics:
                # cha_children = characteristics[0].childNodes
                    malignancy = characteristic.getElementsByTagName('malignancy')
                    malignancy_data = malignancy[0].firstChild.data
                    malignancy_data = 50*sessionID+int(malignancy_data)
    #                 malignancy_data_set.append(malignancy_data)
    # malignancy_data_set_array = np.asarray(malignancy_data_set)
    # malignancy_data_set_array.shape = (len(malignancy_data_set_array), 1)
    # label_feature_array_t = np.transpose(malignancy_data_set_array)
    # with open(malignancy_csv, 'a') as f:
    #     descriptionwriter = csv.writer(f, delimiter=',')
    #     for line in label_feature_array_t:
    #         descriptionwriter.writerow(line)


            #     for cha_child in cha_children:
            #         if cha_child.nodeType == 1:

                rois = nodule.getElementsByTagName('roi')
                for roi in rois:
                    z_positions = roi.getElementsByTagName('imageZposition')
                    z_index = int(abs((float(z_positions[0].firstChild.data) - spacingInfo['origin'][2])/(spacingInfo['spacing'][2])))
                    assert z_index - abs((float(z_positions[0].firstChild.data) - spacingInfo['origin'][2]) / (spacingInfo['spacing'][2])) < 0.1
                    xcoords = roi.getElementsByTagName('xCoord')
                    if len(xcoords) < 5: continue
                    ycoords = roi.getElementsByTagName('yCoord')
                    if not slice_annot.has_key(z_index):
                        slice_annot[z_index] = [[int(y.firstChild.data), int(x.firstChild.data),sessionID,int(malignancy_data)] for x, y in
                                             zip(xcoords, ycoords)]
                    else:
                        slice_annot[z_index] += [[int(y.firstChild.data), int(x.firstChild.data),sessionID,int(malignancy_data)] for x, y in
                                             zip(xcoords, ycoords)]
    return slice_annot

def normalizePlanes(nparray, scale = 255):
    maxHU = 350.
    minHU = -1150.
    nparray[nparray > maxHU] = maxHU
    nparray[nparray < minHU] = minHU
    nparray = (nparray - minHU) / (maxHU - minHU)
    nparray *= scale
    return nparray

def get_mask_from_lidc(dir,outDir,uid =None,bAddOriImage = False,bOutputPNG =False,bOutputMHD=True,maxZSpacing = 3.1):
    # dir = '/home/cb/mnt/2T/data/LIDC-IDRI/LIDC-IDRI-0001/1.3.6.1.4.1.14519.5.2.1.6279.6001.298806137288633453246975630178/1.3.6.1.4.1.14519.5.2.1.6279.6001.179049373636438705059720603192'
    # outDir = '/home/cb/mnt/2T/data/Nodule_mask/ori_mhd'
    if uid is None: uid = os.path.basename(dir)
    if bOutputMHD:  outfilename = uid+'.mhd'
    else: outfilename =uid+'.dcm'
    if outDir is None:
        outDir = os.path.join(dir,'annotation_mask')
    outMaskFile = os.path.join(outDir,outfilename)
    if os.path.exists(outMaskFile):
        return 1
    if not os.path.exists(outDir):
        os.mkdir(outDir)
    for root, dirnames, filenames in os.walk(dir):
        for filename in fnmatch.filter(filenames, '*.xml'):
            xmlfile = os.path.join(dir, filename)

    if xmlfile is None:
        print "xml file is none! Directory: %s" % dir
        return 0

    metaInfo = getMetaInfo(dir)

    assert metaInfo.has_key('spacing')
    if metaInfo['spacing'][2] > maxZSpacing:
        print "Z Spacing is greater than %s. Case is discarded. Dir: %s" % (maxZSpacing,dir)
        return 0

    annotPoints = getAnnotationPoints(xmlfile,metaInfo)
    nodules_mask_array = np.zeros([metaInfo['depth'], metaInfo['height'], metaInfo['width']], dtype=np.byte)

    for i in range(1, metaInfo['depth']+1):
        dcmFileName = metaInfo['filenames'][i-1]
        if bAddOriImage:
            dcmImage = sitk.ReadImage(dcmFileName)
            image_array = sitk.GetArrayFromImage(dcmImage)
            image_array = normalizePlanes(image_array, 200)
            image_array = np.squeeze(image_array)
        else:
            image_array = np.zeros([metaInfo['width'], metaInfo['height']])
        if annotPoints.has_key(i-1):
            #read session information stastics
            sessionIDs = np.zeros(len(annotPoints[i - 1]))
            k = 0
            for point in annotPoints[i - 1]:
                sessionIDs[k] = point[2]
                k += 1
            unique, counts = np.unique(sessionIDs, return_counts=True)
            # 3 or 4 radiologist identified it as a nodule
            if len(counts)>2:
                mask_array = np.zeros([metaInfo['height'],metaInfo['width']])
                for point in annotPoints[i - 1]:
                # mask_array[point[0], point[1]] = 250 + point[2]
                    mask_array[point[0], point[1]] = 250
                mask_array = binary_fill_holes(mask_array)
                if bAddOriImage:
                    image_array *= mask_array == 0
                    image_array += mask_array*127
                else:
                    image_array = mask_array * 127
            else:
                if not bAddOriImage:
                    image_array[:]=0
                pass
                # print len(counts), dcmBaseName
        if (np.sum(image_array) == 0): continue
        nodules_mask_array[i-1,...] = image_array
        if bOutputPNG:
            dcmBaseName = os.path.basename(dcmFileName)
            dcmMainName, _ = os.path.splitext(dcmBaseName)
            # outDir='/home/cb/mnt/2T/data/Nodule_mask'
            outPNGFile = os.path.join(outDir, dcmMainName + '.png')
            cv2.imwrite(outPNGFile, image_array)

    outMaskImage = sitk.GetImageFromArray(nodules_mask_array)
    outMaskImage.CopyInformation(metaInfo['image'])
    outMaskImage.SetDirection(metaInfo['direction'])
    outMaskImage.SetOrigin(metaInfo['origin'])
    outMaskImage.SetSpacing(metaInfo['spacing'])
    sitk.WriteImage(outMaskImage, outMaskFile, True)
    return 1

def get_mask_sample(dir):
    dir = '/home/bb/Downloads/LIDC/LIDC-IDRI-0001/1.3.6.1.4.1.14519.5.2.1.6279.6001.298806137288633453246975630178/1.3.6.1.4.1.14519.5.2.1.6279.6001.179049373636438705059720603192'

    outDir = os.path.join(dir,'sample')
    outImageFile = os.path.join(outDir,'image.img')
    outMaskFile = os.path.join(outDir, 'mask.img')

    for root, dirnames, filenames in os.walk(dir):
        for filename in fnmatch.filter(filenames, '*.xml'):
            xmlfile = os.path.join(dir, filename)

    if xmlfile is None:
        print "xml file is none! Directory: %s" % dir
        return 0

    if not os.path.exists(outDir):
        os.mkdir(outDir)

    metaInfo = getMetaInfo(dir)

    assert metaInfo.has_key('spacing')
    if metaInfo['spacing'][2] > 2.5:
        print "Z Spacing is greater than 2.5. Case is discarded. Dir: %s" % dir
        return 0

    annotPoints = getAnnotationPoints(xmlfile, metaInfo)

    fp_img = open(outImageFile,mode='wb')
    fp_msk = open(outMaskFile, mode='wb')
    # for i in range(1,metaInfo['depth']+1):
    for i in range(60, 95):
        dcmFileName = metaInfo['filenames'][i-1]
        dcmBaseName = os.path.basename(dcmFileName)
        dcmMainName, _ = os.path.splitext(dcmBaseName)
        dcmImage = sitk.ReadImage(dcmFileName)
        image_array = sitk.GetArrayFromImage(dcmImage)
        # image_array = normalizePlanes(image_array, 200)
        image_array = np.squeeze(image_array)
        mask_array = np.zeros([metaInfo['width'], metaInfo['height']],dtype=np.byte)
        if annotPoints.has_key(i-1):
            #read session information stastics
            sessionIDs = np.zeros(len(annotPoints[i - 1]))
            k = 0
            for point in annotPoints[i - 1]:
                sessionIDs[k] = point[2]
                k += 1
            unique, counts = np.unique(sessionIDs, return_counts=True)
            # 3 or 4 radiologist identified it as a nodule
            if len(counts)>2:
                for point in annotPoints[i - 1]:
                    mask_array[point[0], point[1]] = 1
                mask_array = binary_fill_holes(mask_array)
            else:
                pass
        image_array.tofile(fp_img)
        mask_array.tofile(fp_msk)
    fp_msk.close()
    fp_img.close()
    return 1

def get_mask_from_lidc_batch():
    dir = '/home/orientier7/Datesets/nodule/LIDC-IDRI-0001'


    outDir = '/home/orientier7/Datesets/nodule/Nodule_mask/ori_dcm'
    # malignancy_csv = '/home/orientier7/Datesets/nodule/malignancy_results.csv'
    # for i in range(100, 1014):
    #     individual_dir = os.path.join(dir, str('LIDC-IDRI-0')+str(i))
    resultList = file_search(dir, dataFormat='xml', resultIsDir=True)
    k = 0
    for subdir in resultList:
        k+=get_mask_from_lidc(subdir,outDir,bOutputMHD=False,bAddOriImage=True)
    print "%s processed!" % k

def interpolateSeries():
    return

def getSample(dir):
    get_mask_sample(dir)
    return 1

def thisMain():
    get_mask_from_lidc_batch()
    # get_mask_from_lidc('', bAddOriImage=False, bOutputPNG=False)
    # get_mask_from_lidc('')
    # getSample(dir)
    # get_mask_from_lidc_batch()

if __name__ == '__main__':
    thisMain()
# A=0
# B=0
# C=0
# D=0
# E=0
# F=0
# dir = '/home/orientier7/Datesets/nodule/'
# for i in range(1, 101):
#     xmlfiledir = os.path.join(dir, str(i))
#     a,b,c,d,e,f = getAnnotationPoints(xmlfiledir,spacingInfo=None)
#     A+=a
#     B+=b
#     C+=c
#     D+=d
#     E+=e
#     F+=f
# print A,B,C,D,E,F