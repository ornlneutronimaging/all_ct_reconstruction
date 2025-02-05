import numpy as np

from __code.workflow.chips_correction import ChipsCorrection
from __code.config import chips_offset


def correct_data(data_array):
    print("chips correction ...", end="")
    offset = list(chips_offset)

    normalized_images = np.array(data_array)
    normalized_images_axis_swap = np.moveaxis(normalized_images, 0, 2)  # y, x, angle
    corrected_images = ChipsCorrection.correct_alignment(normalized_images_axis_swap,
                                                         offsets=offset)
    corrected_images = np.moveaxis(corrected_images, 2, 0)  

    print(f" done!")
    return corrected_images
