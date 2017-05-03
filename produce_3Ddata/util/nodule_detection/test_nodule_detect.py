from lymultithreshold import *
from sccore.src.io.imageio import DcmIo, PngIo
from sccore.src.utils.miscs import itk_zyx_to_yxz
from sccore.src.utils.visualize import *
from sccore.src.imageproc.thresholding import rescale
from misc import time_it
import copy


def get_parameter_dict(outputSpacing):
    parameter_dict = {}
    parameter_dict['pixel_spacing'] = outputSpacing
    parameter_dict['diameTMin'] = 3
    parameter_dict['diameTMax'] = 40
    parameter_dict['areaunit'] = outputSpacing[0] * outputSpacing[1]
    parameter_dict['volunit'] = parameter_dict['areaunit'] * outputSpacing[2]
    parameter_dict['areaTMin'] = ((parameter_dict['diameTMin'] / 2.) ** 2) * np.pi
    parameter_dict['areaTMax'] = ((parameter_dict['diameTMax'] / 2.) ** 2) * np.pi
    parameter_dict['volTMin'] = ((parameter_dict['diameTMin'] / 2.) ** 3) * np.pi * (4. / 3.)
    parameter_dict['volTMax'] = ((parameter_dict['diameTMax'] / 2.) ** 3) * np.pi * (4. / 3.)
    parameter_dict['small_vol_threshold'] = round(parameter_dict['volTMin'] / parameter_dict['volunit'])
    parameter_dict['vessel_hu'] = 0
    parameter_dict['elongationTMax'] = 3
    return parameter_dict


@time_it
def get_region_candidate_center(simage, lung_mask):
    '''input image must be y-x-z'''
    spacing = simage.get_header().spacing

    lung_image = simage.get_pixel_data()
    parameter_dict = get_parameter_dict(spacing)

    lung_3d = lyExtractROI(lung_image, lung_mask, area_min=parameter_dict['areaTMin'])
    # output_path='/home/fc/fc/test/temp'
    # save_image(normalize_image(lung_3d),output_path)

    muti_thresholds = range(-600, 1, 100)
    multi_level_mask = ly_gaussian_and_multithreshold(lung_3d, muti_thresholds, sigma=1)
    # output_path = '/home/fc/fc/test/temp1'
    # normalized_mask=normalize_image(multi_level_mask,multi_level_mask.min(),multi_level_mask.max())
    # save_image(normalized_mask,output_path)

    nodule_mask = cast_unsuitable_regions_by_label_MT(multi_level_mask, muti_thresholds, parameter_dict)
    # output_path = '/home/fc/fc/test/temp1'
    # normalized_image = normalize_image(lung_image)
    # save_boundary_images(normalized_image, [nodule_mask], output_path)
    return nodule_mask


def get_nodule_mask(numpy_image, lung_mask, spacing, output_path=None):
    '''input image must be y-x-z'''
    # save_boundary_images(normalize_image(numpy_image), [lung_mask], output_path)

    # import scipy.io as sio
    # mat_path = '/home/fc/dcm_image.mat'
    # matlab_lung3d = sio.loadmat(mat_path)['orgVolume']
    # cha = numpy_image.astype(np.int) - matlab_lung3d
    # save_image(normalize_image(cha),output_path)


    parameter_dict = get_parameter_dict(spacing)

    lung_3d = lyExtractROI(numpy_image, lung_mask, area_min=parameter_dict['areaTMin'])
    # import scipy.io as sio
    # mat_path='/home/fc/lung3d.mat'
    # matlab_lung3d = sio.loadmat(mat_path)['lung_3d']
    # cha=lung_3d-matlab_lung3d
    # save_image(normalize_image(cha), output_path)

    muti_thresholds = range(-600, 1, 100)
    multi_level_mask = ly_gaussian_and_multithreshold(lung_3d, muti_thresholds, sigma=0.7)
    # save_image(rescale(multi_level_mask,0,255), output_path)

    # output_path = '/home/fc/fc/test/temp1'
    # normalized_mask=normalize_image(multi_level_mask,multi_level_mask.min(),multi_level_mask.max())
    # save_image(normalized_mask,output_path)

    nodule_mask = cast_unsuitable_regions_by_label_MT(multi_level_mask, muti_thresholds, parameter_dict)
    output_path = '/home/fc/fc/test/temp1'
    normalized_image = normalize_image(numpy_image)
    save_boundary_images(normalized_image, [nodule_mask], output_path)
    return nodule_mask


if __name__ == '__main__':
    input_path = '/home/fc/fc/dcm2/ori_dcm'
    mask_path = '/home/fc/fc/dcm2/lung_mask'
    output_path = '/home/fc/fc/test/original'

    dcm_io = DcmIo()
    png_io = PngIo()
    numpy_image, spacing, _, _ = dcm_io.read_to_numpy_image(input_path)
    numpy_image = itk_zyx_to_yxz(numpy_image)
    lung_mask = png_io.read(mask_path)
    lung_mask = lung_mask > 0

    nodule_mask = get_nodule_mask(numpy_image, lung_mask, spacing, output_path)
