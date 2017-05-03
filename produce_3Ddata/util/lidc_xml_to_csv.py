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
from scipy import ndimage


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
    # slice_annot = {}
    slice_sessions_annot = {}
    reading_sessions = dom.getElementsByTagName('readingSession')
    sessionID = 0
    for session in reading_sessions:
        sessionID += 1
        nodules = session.getElementsByTagName('unblindedReadNodule')
        noduleID = 0
        slice_nodule_annot = {}
        # slice_annot = {}
        for nodule in nodules:
            # slice_nodule_annot = {}
            slice_annot = {}
            characteristics = nodule.getElementsByTagName('characteristics')
            if characteristics:
                noduleID +=1
                """calcification', 'internalStructure',
                    'lobulation', 'malignancy', 'margin',
                    'sphericity', 'subtlety', 'texture', 'doctorId']"""
                features = {}
                for characteristic in characteristics:

                    calcification = characteristic.getElementsByTagName('calcification')
                    calcification_data = calcification[0].firstChild.data
                    features['calcification'] = int(calcification_data)

                    internalStructure = characteristic.getElementsByTagName('internalStructure')
                    internalStructure_data = internalStructure[0].firstChild.data
                    features['internalStructure'] = int(internalStructure_data)

                    lobulation = characteristic.getElementsByTagName('lobulation')
                    lobulation_data = lobulation[0].firstChild.data
                    features['lobulation'] = int(lobulation_data)

                    malignancy = characteristic.getElementsByTagName('malignancy')
                    malignancy_data = malignancy[0].firstChild.data
                    features['malignancy'] = int(malignancy_data)

                    margin = characteristic.getElementsByTagName('margin')
                    margin_data = margin[0].firstChild.data
                    features['margin'] = int(margin_data)

                    sphericity = characteristic.getElementsByTagName('sphericity')
                    sphericity_data = sphericity[0].firstChild.data
                    features['sphericity'] = int(sphericity_data)

                    subtlety = characteristic.getElementsByTagName('subtlety')
                    subtlety_data = subtlety[0].firstChild.data
                    features['subtlety'] = int(subtlety_data)

                    texture = characteristic.getElementsByTagName('texture')
                    texture_data = texture[0].firstChild.data
                    features['texture'] = int(texture_data)

                    spiculation = characteristic.getElementsByTagName('spiculation')
                    spiculation_data = spiculation[0].firstChild.data
                    features['spiculation'] = int(spiculation_data)



                rois = nodule.getElementsByTagName('roi')
                for roi in rois:
                    z_positions = roi.getElementsByTagName('imageZposition')
                    z_index = int(abs((float(z_positions[0].firstChild.data) - spacingInfo['origin'][2])/(spacingInfo['spacing'][2])))
                    assert z_index - abs((float(z_positions[0].firstChild.data) - spacingInfo['origin'][2]) / (spacingInfo['spacing'][2])) < 0.1
                    xcoords = roi.getElementsByTagName('xCoord')
                    if len(xcoords) < 5:continue
                    ycoords = roi.getElementsByTagName('yCoord')
                    if not slice_annot.has_key(z_index):
                        slice_annot[z_index] = [[int(y.firstChild.data), int(x.firstChild.data),features] for x, y in
                                             zip(xcoords, ycoords)]
                    else:
                        slice_annot[z_index] += [[int(y.firstChild.data), int(x.firstChild.data),features] for x, y in
                                             zip(xcoords, ycoords)]
                slice_nodule_annot[noduleID] = slice_annot
        slice_sessions_annot[sessionID] = slice_nodule_annot
    return slice_sessions_annot

def normalizePlanes(nparray, scale = 255):
    maxHU = 350.
    minHU = -1150.
    nparray[nparray > maxHU] = maxHU
    nparray[nparray < minHU] = minHU
    nparray = (nparray - minHU) / (maxHU - minHU)
    nparray *= scale
    return nparray

def get_mask_from_lidc(dir,outDir,output_path,uid =None,bAddOriImage = False,bOutputPNG =False,bOutputMHD=True,maxZSpacing = 3.1):
    # dir = '/home/cb/mnt/2T/data/LIDC-IDRI/LIDC-IDRI-0001/1.3.6.1.4.1.14519.5.2.1.6279.6001.298806137288633453246975630178/1.3.6.1.4.1.14519.5.2.1.6279.6001.179049373636438705059720603192'
    # outDir = '/home/cb/mnt/2T/data/Nodule_mask/ori_mhd'
    dcm_metaInfo = []
    if uid is None: uid = os.path.basename(dir)
    dcm_metaInfo.append(uid)
    # if bOutputMHD:  outfilename = uid+'.mhd'
    # else: outfilename =uid+'.dcm'
    # if outDir is None:
    #     outDir = os.path.join(dir,'annotation_mask')
    # outMaskFile = os.path.join(outDir,outfilename)
    # if os.path.exists(outMaskFile):
    #     return 1
    # if not os.path.exists(outDir):
    #     os.mkdir(outDir)
    for root, dirnames, filenames in os.walk(dir):
        for filename in fnmatch.filter(filenames, '*.xml'):
            xmlfile = os.path.join(dir, filename)

    if xmlfile is None:
        print "xml file is none! Directory: %s" % dir
        return 0

    metaInfo = getMetaInfo(dir)
    dcm_metaInfo.append(str(metaInfo['direction']))
    dcm_metaInfo.append(str(metaInfo['origin']))
    dcm_metaInfo.append(str(metaInfo['spacing']))
    dcm_metaInfo.append(str([metaInfo['height'], metaInfo['width'],metaInfo['depth']]))

    assert metaInfo.has_key('spacing')
    if metaInfo['spacing'][2] > maxZSpacing:
        print "Z Spacing is greater than %s. Case is discarded. Dir: %s" % (maxZSpacing,dir)
        return 0

    annotPoints = getAnnotationPoints(xmlfile,metaInfo)
    nodules_mask_array = np.zeros([metaInfo['depth'], metaInfo['height'], metaInfo['width']], dtype=np.byte)
    # doc1_nodules_mask_array = np.zeros([metaInfo['depth'], metaInfo['height'], metaInfo['width']], dtype=np.byte)
    # doc2_nodules_mask_array = np.zeros([metaInfo['depth'], metaInfo['height'], metaInfo['width']], dtype=np.byte)
    # doc3_nodules_mask_array = np.zeros([metaInfo['depth'], metaInfo['height'], metaInfo['width']], dtype=np.byte)
    # doc4_nodules_mask_array = np.zeros([metaInfo['depth'], metaInfo['height'], metaInfo['width']], dtype=np.byte)


    # for i in range(1, metaInfo['depth']+1):
        # dcmFileName = metaInfo['filenames'][i-1]
        # if bAddOriImage:
        #     dcmImage = sitk.ReadImage(dcmFileName)
        #     image_array = sitk.GetArrayFromImage(dcmImage)
        #     image_array = normalizePlanes(image_array, 200)
        #     image_array = np.squeeze(image_array)
        # else:
        #     image_array = np.zeros([metaInfo['width'], metaInfo['height']])
    doc_nodule_info = {}
    for j in range(1,len(annotPoints)+1):
        nodule_info = {}
        for k in range(1,len(annotPoints[j])+1):
            sum_z_coordinates = 0
            # sum_x_coordinates = 0
            # sum_y_coordinates = 0
            doc1_nodules_mask_array = np.zeros([metaInfo['depth'], metaInfo['height'], metaInfo['width']], dtype=np.byte)
            for key,value in annotPoints[j][k].items():
                # sum_z_coordinates += i-1
                for point in value:
                    # if point[2] == 1:
                    doc1_nodules_mask_array[key, point[0], point[1]] = 1
                    # sum_x_coordinates+=point[0]
                    # sum_y_coordinates+=point[1]
                doc1_nodules_mask_array[key, :, :]  = \
                    binary_fill_holes(doc1_nodules_mask_array[key, :, :]).astype(np.int)
            labeled_components_mask1, bounding_box = scipy_connect_components(doc1_nodules_mask_array)
            center_z_coordinates = bounding_box[0][0].start + int((bounding_box[0][0].stop - bounding_box[0][0].start - 1)/2)
            center_x_coordinates = bounding_box[0][1].start + int((bounding_box[0][1].stop - bounding_box[0][1].start - 1)/2)
            center_y_coordinates = bounding_box[0][2].start + int((bounding_box[0][2].stop - bounding_box[0][2].start - 1)/2)
            nodule_radius = np.sqrt((center_z_coordinates - bounding_box[0][0].start)**2+(center_x_coordinates-bounding_box[0][1].start)**2+\
                            (center_y_coordinates - bounding_box[0][2].start)**2)
            nodule_info[k] = [int(center_z_coordinates),int(center_x_coordinates),int(center_y_coordinates),nodule_radius,value[2][2],j]
        doc_nodule_info[j] = nodule_info
    results = {}
    for doc_key, doc_value in doc_nodule_info.items():
        if doc_key == 1:
            first_doc = []
            for nodule_key, nodule_value in doc_value.items():
                first_doc.append(nodule_value)
                results[nodule_key] = [nodule_value]
            continue
        for nodule_key2, nodule_value2 in doc_value.items():
            j = len(first_doc)
            k = 0
            for nodule_key3, nodule_value3 in results.items():
            # for i in xrange(0, len(first_doc)):

                if np.sqrt((nodule_value3[0][0]-nodule_value2[0])**2+(nodule_value3[0][1]-nodule_value2[1])**2+ \
                                (nodule_value3[0][2] - nodule_value2[2])**2) < nodule_value3[0][3]+nodule_value2[3]:
                    results[nodule_key3].append(nodule_value2)
                    k+=1
                    break
                    # continue
                    # del doc_value[nodule_value2]
                elif k == len(results):
                    j+=1
                    results[j] = nodule_value2


    with open(output_path, "a") as f:
        writer = csv.writer(f)
        # attributes = []
        for nodule_key, nodule_value in results.items():
            # attributes = []
            for i in range(0,len(nodule_value)):
                attributes = []
                attributes.extend(dcm_metaInfo)
                attributes.append(str(nodule_key))
                attributes.append(str(nodule_value[i][0:3]))
                attributes.append(str(nodule_value[i][3]))
                for nodule_key4, nodule_value4 in nodule_value[i][4].items():
                    attributes.append(str(nodule_value4))
                attributes.append(str(nodule_value[i][5]))
                # dcm_metaInfo.extend(attributes)
                writer.writerow(attributes)
















                # num_of_points = np.sum(doc1_nodules_mask_array)
                # center_x_coordinates = sum_x_coordinates/num_of_points
                # center_y_coo

                        # elif point[2] == 2:
                        #     doc2_nodules_mask_array[i-1, point[0], point[1]] = 1
                        # elif point[2] == 3:
                        #     doc3_nodules_mask_array[i-1, point[0], point[1]] = 1
                        # elif point[2] == 4:
                        #     doc4_nodules_mask_array[i-1, point[0], point[1]] = 1
                    # doc1_nodules_mask_array[i-1, :, :] = binary_fill_holes(doc1_nodules_mask_array[i-1, :, :]).astype(np.int)
                    # doc2_nodules_mask_array[i-1, :, :] = binary_fill_holes(doc2_nodules_mask_array[i-1, :, :]).astype(np.int)
                    # doc3_nodules_mask_array[i-1, :, :] = binary_fill_holes(doc3_nodules_mask_array[i-1, :, :]).astype(np.int)
                    # doc4_nodules_mask_array[i-1, :, :] = binary_fill_holes(doc4_nodules_mask_array[i-1, :, :]).astype(np.int)

    # labeled_components_mask1,bounding_box = scipy_connect_components(doc1_nodules_mask_array)
    # for i in range(1,np.max(point[3])):
    #     if point[3] == i:


    # doc1_nodules_itk_mask_array = sitk.GetImageFromArray(doc1_nodules_mask_array)
    # doc1_nodules_itk_labeled_mask_array = sitk.ConnectedComponent(doc1_nodules_itk_mask_array)
    # doc1_nodules_mask_array = sitk.GetArrayFromImage(doc1_nodules_itk_labeled_mask_array)


                    # sessionIDs[k] = point[2]
                # k += 1
            # unique, counts = np.unique(sessionIDs, return_counts=True)
            # 3 or 4 radiologist identified it as a nodule
            # if len(counts)>2:
    #             mask_array = np.zeros([metaInfo['height'],metaInfo['width']])
    #             for point in annotPoints[i - 1]:
    #             # mask_array[point[0], point[1]] = 250 + point[2]
    #                 mask_array[point[0], point[1]] = 250
    #             mask_array = binary_fill_holes(mask_array)
    #             if bAddOriImage:
    #                 image_array *= mask_array == 0
    #                 image_array += mask_array*127
    #             else:
    #                 image_array = mask_array * 127
    #         else:
    #             if not bAddOriImage:
    #                 image_array[:]=0
    #             pass
    #             # print len(counts), dcmBaseName
    #     if (np.sum(image_array) == 0): continue
    #     nodules_mask_array[i-1,...] = image_array
    #     if bOutputPNG:
    #         dcmBaseName = os.path.basename(dcmFileName)
    #         dcmMainName, _ = os.path.splitext(dcmBaseName)
    #         # outDir='/home/cb/mnt/2T/data/Nodule_mask'
    #         outPNGFile = os.path.join(outDir, dcmMainName + '.png')
    #         cv2.imwrite(outPNGFile, image_array)
    #
    # outMaskImage = sitk.GetImageFromArray(nodules_mask_array)
    # outMaskImage.CopyInformation(metaInfo['image'])
    # outMaskImage.SetDirection(metaInfo['direction'])
    # outMaskImage.SetOrigin(metaInfo['origin'])
    # outMaskImage.SetSpacing(metaInfo['spacing'])
    # sitk.WriteImage(outMaskImage, outMaskFile, True)
    return 1

class NoduleCSVWriter(object):
    def __init__(self, output_path):
        self.merged_nodule_list = None
        self.output_path = output_path
        self.header = None

    def set_nodule_list(self, merged_nodule_list):
        self.merged_nodule_list = merged_nodule_list

    def write_first_row(self):
        self.header = ['series_uid', 'spacing', 'origin', 'size',
                       'noduleId', 'center(y,x,z)', 'boundingBox(h,w,d)',
                       'calcification', 'internalStructure',
                       'lobulation', 'malignancy', 'margin',
                       'sphericity', 'subtlety', 'texture', 'doctorId']
        with open(self.output_path, "w") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(self.header)

    def write(self):
        with open(self.output_path, "a") as f:
            writer = csv.writer(f)
            for i, real_single_nodule in enumerate(self.merged_nodule_list):
                for single_doctor in real_single_nodule:
                    attributes = []
                    for key in self.header:
                        if key == 'center(y,x,z)':
                            value = single_doctor.box.center
                        elif key == 'boundingBox(h,w,d)':
                            value = single_doctor.box.bounding_shape
                        else:
                            value = single_doctor.metadata[key]
                        attributes.append(str(value))
                    writer.writerow(attributes)

def scipy_connect_components(mask, num=None, area_min=None):
    '''this function can return connect component with the area size of area_min and number of num
    Parameters
    ---------
          mask is the 0-1 image
          num is the Number of connected domains you want to keep
          area_min is the area that you want to keep bigger than it
    Returns
    -------
    label_img:
              labeled image
    cc:
              bounding box
    '''

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


def get_mask_from_lidc_batch():
    dir = '/home/orientier7/Datesets/LIDC'
    outDir = '/home/orientier7/Datesets/nodule/Nodule_mask/ori_dcm'
    origin_path = '/home/orientier7/LIDC/LIDC-IDRI'
    header = ['uid', 'direction', 'origin', 'spacing', 'size', 'noduleId',
              'center(y,x,z)', 'boundingBox(h,w,d)', 'calcification',
              'internalStructure', 'lobulation', 'malignancy', 'margin',
              'sphericity', 'subtlety', 'texture', 'spiculation', 'doctorId']
    output_path = '/home/orientier7/nodule_10_xml.csv'
    with open(output_path, "w") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(header)

    resultList = file_search(dir, dataFormat='xml', resultIsDir=True)
    k = 0
    for subdir in resultList:
        k+=get_mask_from_lidc(subdir,outDir,output_path,bOutputMHD=False,bAddOriImage=True)
    print "%s processed!" % k

if __name__ == '__main__':
    get_mask_from_lidc_batch()

