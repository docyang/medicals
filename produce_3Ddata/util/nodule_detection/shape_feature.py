import numpy as np
from skimage import transform, morphology


def shape_feature(slice_image, pixel_spacing):
    zoom_shape = [single_shape * 2 for single_shape in slice_image.shape]
    zoomed_image = transform.resize(slice_image, zoom_shape, order=0)
    bordered_image_shape = [single_shape + 4 for single_shape in zoom_shape]
    bordered_image = np.zeros(bordered_image_shape)
    coor_slices = [slice(2, 2 + side_width) for side_width in zoom_shape]
    bordered_image[coor_slices] = zoomed_image
    # bordered_image[bordered_image>0.5]=1
    # bordered_image[bordered_image!=1]=0

    strel = morphology.disk(1)
    dilated_image = morphology.binary_dilation(bordered_image, strel)
    region_slice_border_image = dilated_image - bordered_image

    if np.sum(bordered_image) < 2:
        return None

    region_slice_area = np.sum(bordered_image) * pixel_spacing[0] * pixel_spacing[1] / 4
    region_slice_perimeter = np.sum(region_slice_border_image) / 2 * pixel_spacing[0]
    compactness = 4 * np.pi * region_slice_area / (region_slice_perimeter ** 2)

    return region_slice_area, compactness


def test_func():
    x, y = np.indices((10, 10))
    centerx, centery = 5, 5
    mask_circle = (x - centerx) ** 2 + (y - centery) ** 2 < 16
    spaceing = np.array([0.5, 0.5])
    area, com = shape_feature(mask_circle, spaceing)
    print com
    print area

if __name__ == '__main__':
    test_func()
