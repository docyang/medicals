import SimpleITK as sitk
import numpy as np
from scipy import ndimage
from misc import time_it


def creat_18_connectivity_kernel():
    shape = [3] * 3
    kernel = np.zeros(shape, np.int)
    kernel[1, 1, 1] = 1
    kernel = ndimage.binary_dilation(kernel, iterations=2)
    return kernel.astype(np.int)


# @time_it
def lyBWareaopen(mask, area_min):
    structure = creat_18_connectivity_kernel()
    label_img, cc_num = ndimage.label(mask, structure=structure)
    cc_size = ndimage.sum(mask, label_img, range(cc_num + 1))
    area_mask = (cc_size < area_min)
    label_img[area_mask[label_img]] = 0
    labels = np.unique(label_img)
    label_img = np.searchsorted(labels, label_img)
    connect_comp = ndimage.find_objects(label_img)
    return label_img, connect_comp


def get_connect_component(bw_image):
    structure = creat_18_connectivity_kernel()
    label_img, cc_num = ndimage.label(bw_image, structure=structure)
    CC = ndimage.find_objects(label_img)
    return CC


def lyBWcoordinate(Map):
    structure = creat_18_connectivity_kernel()
    label_img, cc_num = ndimage.label(Map, structure=structure)
    coor_list = ndimage.center_of_mass(Map, label_img, range(1, cc_num + 1))
    return coor_list


@time_it
def fc_bwareaopen(mask, area_min):
    '''slower than lyBWareaopen(), abort it'''
    sitk_mask = sitk.GetImageFromArray(mask.astype(np.int))
    ccf = sitk.ConnectedComponentImageFilter()
    ccf.SetFullyConnected(False)
    sitk_label_image = ccf.Execute(sitk_mask)
    label_image = sitk.GetArrayFromImage(sitk_label_image)
    label_statistic_filter = sitk.LabelStatisticsImageFilter()
    label_statistic_filter.Execute(sitk_mask, sitk_label_image)
    labels = list(label_statistic_filter.GetLabels())
    '''label 0 is background'''
    labels.remove(0)
    volumns = [label_statistic_filter.GetCount(label) for label in labels]
    preserve_label_list = []
    for i, value in enumerate(volumns):
        if value > area_min:
            preserve_label_list.append(i)
    mask_copy = np.zeros_like(mask)
    for preserve_label in preserve_label_list:
        mask_copy[label_image == preserve_label] = 1
    return mask_copy


def test_several_gaussian_filter():
    '''result:these are almost the same'''
    a = np.zeros((5, 5),np.float32)
    a[2, 2] = 1
    import vigra
    from skimage.filter import gaussian_filter

    print gaussian_filter(a,1)
    print vigra.gaussianSmoothing(a, 1)
    print ndimage.gaussian_filter(a, 1)


def test_close_ndiamge():
    a = np.zeros((5, 5,5), dtype=np.int)
    # a[1:-1, 1:-1,1:-1] = 1
    a[2, 2,2] = 1

    b=np.zeros((3,3,3))
    x,y,z=np.indices(b.shape)
    b[abs(x-1)+abs(y-1)+abs(z-1)<3]=1

    print ndimage.binary_dilation(a)