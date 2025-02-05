import numpy as np
from neutompy.preproc.preproc import correction_COR


def center_of_rotation_and_tilt(config_model, data_array):
    if not config_model.calculate_center_of_rotation:
        print(f"skipped center of rotation and tilt calculation!")
        return data_array
    
    print(f"center of rotation and tilt calculation ...", end="")
    [top_slice, bottom_slice] = config_model.range_of_slices_for_center_of_rotation
    list_of_angles = config_model.list_of_angles
    image_0_degree, image_180_degree = get_0_and_180_degrees_images(list_of_angles=list_of_angles,
                                                                    data_array=data_array)

    mid_point = int(np.mean([top_slice, bottom_slice]))
    rois = ((top_slice, mid_point+1), (mid_point, bottom_slice))

    corrected_images = correction_COR(data_array,
                                      image_0_degree,
                                      image_180_degree,
                                      rois=rois)
    
    print(f" done!")
    return corrected_images


def get_0_and_180_degrees_images(list_of_angles, data_array):
    angles_minus_180 = [float(_value) - 180 for _value in list_of_angles]
    abs_angles_minus_180 = np.abs(angles_minus_180)
    minimum_value = np.min(abs_angles_minus_180)

    index_0_degree = 0
    index_180_degree = np.where(minimum_value == abs_angles_minus_180)[0][0]

    image_0_degree = data_array[index_0_degree]
    image_180_degree = data_array[index_180_degree]

    return image_0_degree, image_180_degree
