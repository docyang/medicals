import cv2
import numpy as np

def create_ball_patch(ball_center, radius, spacing=None):
    """create a np_array of a ball in small patch
    inputs
    ball_center: list or np_array, 2 dim are 3 dim all available
    radius: float
    spacing: optional. if set may return a ellipse
    returns
    ball: np_array of ball mask
    origin: top most point coordinate of patch in original image
    """
    dim = len(ball_center)
    if spacing is None:
        spacing = np.array([1, ] * dim)

    real_radius = int(np.ceil(np.max(radius / np.array(spacing))))
    max_bound = real_radius * 2 + 1
    patch_shape = [max_bound] * dim
    patch_center = [real_radius] * dim
    indices = np.indices(patch_shape)

    temp_square = [((indices[i] - patch_center[i]) * spacing[i]) ** 2 for i in range(dim)]
    patch = np.sqrt(reduce(lambda a, b: a + b, temp_square)) < radius
    patch = np.array(patch, np.int8)
    origin = np.array(ball_center) - np.array(patch_center)
    origin = np.array(origin, np.int)
    return patch, origin

def read_gtd_to_mask(input_path):
    """this func is to make a mask from gtd marked by doctors,
    input_path is one dcm case directory
    """

    for image, annotations in ETL.read_images_annotations(input_path):
        spacing = image.get_header().spacing
        mask_shape = image.get_pixel_data().shape
        mask = np.zeros_like(image.get_pixel_data(), np.int8)
        for anno in annotations:
            if anno.get_header().metadata['category'] == 'nodule':
                print input_path.split('/')[-3]

                for roi_points in anno.coordinates:
                    z_index = roi_points[0][2]
                    roi_point_list = []
                    for point in roi_points:
                        roi_point_list.append([point[1], point[0]])
                    points = np.array([roi_point_list], np.int32)

                    if len(roi_points) < 5:
                        # continue
                        center_point = list(np.mean(points[0], 0))
                        center_point.append(z_index)
                        patch, origin = create_ball_patch(center_point, 3, spacing=spacing)
                        slices = [slice(origin[i], origin[i] + patch.shape[i]) for i in range(len(mask_shape))]
                        mask[slices] += patch
                    else:

                        image_temp = mask[:, :, z_index]
                        # print np.sum(image_temp)
                        cv2.fillPoly(image_temp, [points], 1)
                        mask[:, :, z_index] = image_temp

        return mask