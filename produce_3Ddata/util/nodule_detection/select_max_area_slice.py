import numpy as np


def select_max_area_slice(region_image):
    if np.ndim(region_image) == 2:
        return region_image
    areas = np.sum(np.sum(region_image, 0), 0)
    max_number_slice = np.where(areas == np.max(areas))[0][0]
    # image_depth=region_image.shape[2]
    # max_number,max_number_slice=0,0
    # for depth in range(image_depth):
    #     image_piece=region_image[:,:,depth]
    #     slice_area = np.sum(image_piece)
    #     if slice_area >max_number:
    #         max_number_slice=depth
    #         max_number= slice_area
    return region_image[:, :, max_number_slice]


def test_max_slice():
    x, y, z = np.indices((10, 10, 10))
    ball = (x - 5) ** 2 + (y - 5) ** 2 + (z - 5) ** 2 < 25
    print select_max_area_slice(ball.astype(np.int8))


if __name__ == '__main__':
    test_max_slice()
