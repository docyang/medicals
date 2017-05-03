import numpy as np
from connectivity import *
from scipy import ndimage
from select_max_area_slice import select_max_area_slice
from shape_feature import shape_feature
from misc import get_mask_coordinate,time_it,GaussianWorker
from pca import pca
import pprocess
import multiprocessing as mp
import ctypes
import copy


@time_it
def filter_regions(multi_level_mask, thresholds_list, parameter_dict, one_label):
    one_threshold = thresholds_list[one_label]
    mask = (multi_level_mask > one_label)
    min_size = parameter_dict['small_vol_threshold']
    label_image, bounding_box_slices = lyBWareaopen(mask, min_size)

    region_result = [False]
    '''single thread'''
    # for label_number in range(1, np.max(label_image) + 1):
    #     single_result = region_task(label_number, label_image, bounding_box_slices, one_threshold, parameter_dict)
    #     region_result.append(single_result)

    '''multi thread'''
    results = pprocess.Map(limit=pprocess.get_number_of_cores())
    calc = results.manage(pprocess.MakeParallel(region_task))

    for label_number in range(1, np.max(label_image) + 1):
        calc(label_number, label_image, bounding_box_slices, one_threshold, parameter_dict)

    for i, result in enumerate(results):
        region_result.append(result)

    region_result = np.array(region_result, np.bool)
    tuild_result = np.logical_not(region_result)
    label_image[tuild_result[label_image]] = 0

    return (label_image > 0)


def region_task(label_number, label_image, bounding_box_slices, one_threshold, parameter_dict):
    bound_slice = bounding_box_slices[label_number - 1]
    region_object, region_volumn = get_label_patch(label_image, bound_slice, label_number)

    if is_like_a_nodule(region_object, region_volumn, bound_slice, one_threshold, parameter_dict):
        return True
    else:
        return False


def is_like_a_nodule(region_object, region_volume, bound_slice, one_threshold, parameter_dict):
    '''3d condition filter'''
    if _is_boundingbox_size_too_small(bound_slice, \
                                      parameter_dict['diameTMin'], parameter_dict['pixel_spacing']):
        return False
    if _is_boundingbox_size_too_big(bound_slice, \
                                    parameter_dict['diameTMax'], parameter_dict['pixel_spacing']):
        return False
    if _is_region_like_vessel(bound_slice, region_volume, one_threshold, parameter_dict):
        return False
    # if _is_satisfied_with_elongation(region_object, parameter_dict['pixel_spacing'], \
    #                                  parameter_dict['elongationTMax'], parameter_dict['volTMin']):
    #     return False

    '''2d condition filter'''
    max_area_slice = select_max_area_slice(region_object)
    area, compateness = shape_feature(max_area_slice, parameter_dict['pixel_spacing'])
    if _max_area_slice_correspond_with_conditions(area, compateness, parameter_dict['areaTMin'],
                                                  parameter_dict['areaTMax']):
        return True


def get_label_patch(label_image, bound_slice, label_number):
    temp_image = label_image == label_number
    region_object = temp_image[bound_slice]
    region_volume = np.sum(region_object)
    return region_object, region_volume


@time_it
def lyExtractROI(vol, lungmask, area_min=3, back_hu=-2000):
    lung_3d = np.zeros_like(vol)
    un_care_slice_depth = vol.shape[2] / 4
    total_depth = vol.shape[2]
    for depth in range(total_depth):
        slice = lungmask[:, :, depth]
        if depth < un_care_slice_depth and np.sum(slice) < area_min:
            lung_3d[:, :, depth] = back_hu
        else:
            lung_3d[:, :, depth] = vol[:, :, depth] * slice
            lung_3d[:, :, depth] += (1 - slice) * back_hu

            # slice_result[slice == 0] = back_hu
            # slice_result[slice == 1] = vol[:, :, depth][slice == 1]
    return lung_3d


@time_it
def ly_gaussian_and_multithreshold_MT(numpy_image, thresholds, sigma=1.5):
    queue=mp.Queue()
    event=mp.Event()
    th_num=len(thresholds)
    worker=GaussianWorker(numpy_image,event,queue,th_num)
    worker.start()

    vol_roi = np.zeros(numpy_image.shape, np.int8)
    for i in range(th_num):
        threshold=thresholds[i]
        event.set()
        vol=queue.get()
        closed_mask = ndimage.binary_closing(vol > threshold)
        vol_roi[closed_mask] = i + 1
    worker.terminate()
    return vol_roi

@time_it
def ly_gaussian_and_multithreshold(numpy_image, thresholds, sigma=1.5):
    vol = copy.deepcopy(numpy_image)
    vol_roi = np.zeros(numpy_image.shape, np.int8)
    for i, threshold in enumerate(thresholds):
        vol = ndimage.gaussian_filter(vol, sigma)
        closed_mask = ndimage.binary_closing(vol > threshold)
        vol_roi[closed_mask] = i + 1
    return vol_roi


def cast_unsuitable_regions_by_label(multi_level_mask, thresholds_list, parameter_dict):
    '''cast small region as noise and big region as vessel'''
    nodule_mask = np.zeros_like(multi_level_mask)
    multi_image_labels = range(1, len(np.unique(multi_level_mask)))
    multi_image_labels.reverse()
    for one_label in multi_image_labels:
        temp_mask = filter_regions(multi_level_mask, thresholds_list, parameter_dict, one_label - 1)
        nodule_mask = np.logical_or(temp_mask, nodule_mask)
    return nodule_mask


def cast_unsuitable_regions_by_label_MT(multi_level_mask, thresholds_list, parameter_dict):
    '''cast small region as noise and big region as vessel'''
    nodule_mask = np.zeros_like(multi_level_mask, np.int8)
    multi_image_labels = range(int(np.max(multi_level_mask)))
    multi_image_labels.reverse()
    loop_times = len(multi_image_labels)
    # TODO map style parallezision
    shared_array_s = mp.Array(ctypes.c_int8, loop_times * np.size(nodule_mask))
    shared_array = np.frombuffer(shared_array_s.get_obj(), dtype=np.int8).reshape((loop_times,) + nodule_mask.shape)

    num_of_proc = pprocess.get_number_of_cores()
    results = pprocess.Map(limit=num_of_proc / 2)
    para_func = results.manage(pprocess.MakeParallel(put_result_into_shared_memory))

    for i in range(loop_times):
        one_label = multi_image_labels[i]
        para_func(shared_array, multi_level_mask, thresholds_list, parameter_dict, one_label)
    results.finish()

    for num_of_loop in range(shared_array.shape[0]):
        nodule_mask = np.logical_or(nodule_mask, shared_array[num_of_loop, ...])

    datastate = shared_array_s.get_obj()._wrapper._state
    arenaobj = datastate[0][0]
    arenaobj.buffer.close()
    mp.heap.BufferWrapper._heap = mp.heap.Heap()

    return nodule_mask


def put_result_into_shared_memory(shared_array, multi_level_mask, thresholds_list, parameter_dict, one_label):
    # print "Process %d start!" % one_label
    temp_result = filter_regions(multi_level_mask, thresholds_list, parameter_dict, one_label)
    shared_array[one_label, ...] = temp_result
    # print 'hehe'


def _is_boundingbox_size_too_small(coordinates_slice, diameTMin, pixel_spacing):
    x_coor_range = coordinates_slice[0]
    y_coor_range = coordinates_slice[1]
    x_spacing = pixel_spacing[0]
    y_spacing = pixel_spacing[1]
    if (x_coor_range.stop - x_coor_range.start) * x_spacing < diameTMin or \
                            (y_coor_range.stop - y_coor_range.start) * y_spacing < diameTMin:
        return True
    else:
        return False


def _is_boundingbox_size_too_big(coordinates_slice, diameTMax, pixel_spacing):
    x_coor_range = coordinates_slice[0]
    y_coor_range = coordinates_slice[1]
    x_spacing = pixel_spacing[0]
    y_spacing = pixel_spacing[1]
    if (x_coor_range.stop - x_coor_range.start) * x_spacing > diameTMax or \
                            (y_coor_range.stop - y_coor_range.start) * y_spacing > diameTMax:
        return True
    else:
        return False


def _is_region_like_vessel(coordinates_slice, region_volume, one_threshold, p_dict):
    vessel_threshold = p_dict['vessel_hu']
    region_max = p_dict['volTMax']
    vol_unit = p_dict['volunit']
    vol = region_volume / vol_unit
    boundingbox_shape = [single_slice.stop - single_slice.start for single_slice in coordinates_slice]
    boundingbox_size = reduce(lambda x, y: x * y, boundingbox_shape)
    cir_rate = vol / boundingbox_size
    half_percent = 0.5
    if (vol > region_max) or (one_threshold > vessel_threshold and cir_rate < half_percent):
        return True
    else:
        return False


def _is_satisfied_with_elongation(region_object, spacing, elongation_max, vol_min):
    region_volumn = np.sum(region_object)
    coor_matrix = get_mask_coordinate(region_object)
    assert (coor_matrix.shape[-1] == len(spacing))
    for dimension in range(len(spacing)):
        coor_matrix[:, dimension] = coor_matrix[:, dimension] * spacing[dimension]

    number_first_eig = 3
    result_coor_matrix, _ = pca(coor_matrix, number_first_eig)
    axis_length = np.max(result_coor_matrix, 0) - np.min(result_coor_matrix, 0)
    principle_axis_length = axis_length[0]
    end_axis_length = axis_length[-1]
    elongation = principle_axis_length / (end_axis_length + np.finfo(float).eps)

    if elongation > elongation_max and region_volumn > vol_min:
        return True
    else:
        return False


def _max_area_slice_correspond_with_conditions(area, compateness, areaTMin, areaTMax):
    compateness_threshold = 0.2
    if compateness > compateness_threshold and area > areaTMin and area < areaTMax:
        return True
    else:
        return False


def __test_pac_filter():
    x, y, z = np.indices((10, 10, 10))
    slant_image = np.logical_and(np.abs(x - y) < 2, np.abs(y - z) < 2)
    slant_image = np.array(slant_image, np.int)
    print slant_image
    result = _is_satisfied_with_elongation(slant_image, (0.5, 0.5, 0.5), 3, 10)
    print result
