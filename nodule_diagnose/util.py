# --*-- encoding: UTF-8 --*--

import SimpleITK as sitk
import numpy as np
from scipy import ndimage
from scipy.ndimage.interpolation import zoom

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

def get_sample_images(image, point, box):
    samples = []
    for i in range(len(box)):
        size = max(box[i])
        if size <= 16:
            size = 32
        elif size <= 32:
            size = 40
        elif size <= 52:
            size = 60
        else:
            size = 80
        sample_image = get_patch_with_padding(image, point[i], size=size)
        if size != 32:
            sample_image = zoom(sample_image, float(32) / size)
        sample_image = normalizePlanes(sample_image)
        samples.append(sample_image)
    return samples

def get_patch_with_padding(simage, center, shape=None, size=None):
    #image_array = sitk.GetArrayFromImage(simage)
    dimension = simage.ndim
    image_size = simage.shape
    if shape and size:
        raise ValueError('only one is optional in shape and size')
    if shape:
        if np.ndim(simage) != len(shape):
            raise ValueError('dimension of image and shape must be same')
    if size and isinstance(size, int):
        shape = [size] * dimension
    if (not shape and not size):
        raise ValueError('only one is optional in shape and size')
    center = np.array(center, np.int)
    slices = [slice(center[i] - shape[i] // 2, center[i] + shape[i] // 2)
        for i in range(dimension)]
    patch = simage[slices]
    if patch.shape != tuple(shape):
        npad = ((max(0, shape[0] // 2 - center[0]), max(0, center[0] + shape[0] // 2 - image_size[0])),
            (max(0, shape[1] // 2 - center[1]), max(0, center[1] + shape[1] // 2 - image_size[1])),
            (max(0, shape[2] // 2 - center[2]), max(0, center[2] + shape[2] // 2 - image_size[2])))
        image_pad = np.pad(simage, pad_width=npad, mode='constant', constant_values=0)
        slices = [slice(center[i] - shape[i] // 2 + npad[i][0] - npad[i][1],
                center[i] + shape[i] // 2 + +npad[i][0] - npad[i][1])
                for i in range(dimension)]
        patch = image_pad[slices]
    return patch

def normalizePlanes(npzarray):
    maxHU = 400.
    minHU = -1000.
    npzarray = (npzarray - minHU) / (maxHU - minHU)
    npzarray[npzarray > 1] = 1.
    npzarray[npzarray < 0] = 0.
    return npzarray

def interpolatefilter(inputimage, mask=None):
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
    resamplefilter.SetInterpolator(sitk.sitkNearestNeighbor)

    mask = sitk.GetImageFromArray(mask)
    mask.SetDirection(direction)
    mask.SetOrigin(origin)
    mask.SetSpacing(spacing)
    outputimage1 = resamplefilter.Execute(inputimage)
    outputimage2 = resamplefilter.Execute(mask)
    numpyImage1 = sitk.GetArrayFromImage(outputimage1)
    numpyImage2 = sitk.GetArrayFromImage(outputimage2)
    return numpyImage1, numpyImage2
