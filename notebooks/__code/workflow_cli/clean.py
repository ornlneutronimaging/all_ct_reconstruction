import logging
import numpy as np

from imars3d.backend.corrections.gamma_filter import gamma_filter

from __code import OperatingMode
from __code import CleaningAlgorithm
from __code.utilities.images import replace_pixels
from __code.config import clean_paras


def clean_by_histogram(config_model, master_data):

    print("cleaning by histogram ... ", end="")
    histogram_cleaning_settings = config_model.histogram_cleaning_settings
    nbr_bins = histogram_cleaning_settings.nbr_bins
    nbr_bins_to_exclude = histogram_cleaning_settings.bins_to_exclude

    if nbr_bins != 0:
        # low_gate = config_model.image_cleaner.low_gate
        # high_gate = config_model.image_cleaner.high_gate
        correct_radius = clean_paras['correct_radius']

        for _data_type in master_data.keys():
            cleaned_data = []
            for _data in master_data[_data_type]:
                _cleaned_data = replace_pixels(im=_data,
                                               low_gate=nbr_bins_to_exclude,
                                               high_gate=nbr_bins - nbr_bins_to_exclude,
                                               nbr_bins=nbr_bins,
                                               correct_radius=correct_radius)
                cleaned_data.append(_cleaned_data)
            master_data[_data_type] = cleaned_data

    print("done!")
    return master_data


def clean_by_threshold(config_model, master_data):

    print("cleaning by threshold ... ", end="")
    for _data_type in master_data.keys():
        _data = np.array(master_data[_data_type])
        _cleaned_data = gamma_filter(arrays=_data)
        master_data[_data_type] = _cleaned_data

    print("done!")
    return master_data


def clean_images(config_model, master_data):

    list_clean_algorithm = config_model.list_clean_algorithm

    if CleaningAlgorithm.histogram in list_clean_algorithm:
        master_data = clean_by_histogram(config_model, master_data)

    if CleaningAlgorithm.threshold in list_clean_algorithm:
        master_data = clean_by_threshold(config_model, master_data)

    return master_data
