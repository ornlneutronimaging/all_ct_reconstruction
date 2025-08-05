"""
Image Cleaning Module for CLI-Based CT Reconstruction.

This module provides command-line interface compatible functions for cleaning
CT projection images using various filtering and correction algorithms. It handles
outlier pixel removal, gamma filtering, and histogram-based cleaning to improve
data quality before reconstruction.

Key Functions:
    - clean_images: Main cleaning function orchestrating multiple algorithms
    - clean_by_histogram: Histogram-based outlier pixel cleaning
    - clean_by_threshold: Gamma filter-based threshold cleaning

Key Features:
    - Multiple cleaning algorithms (histogram, threshold-based)
    - Outlier pixel detection and replacement
    - Gamma filtering for noise reduction
    - Configurable parameters for different cleaning methods
    - Support for various data types (sample, OB, DC)

Cleaning Algorithms:
    1. Histogram cleaning: Uses pixel intensity distribution analysis
    2. Threshold cleaning: Applies gamma filtering for outlier removal

Dependencies:
    - imars3d: Gamma filter implementation for threshold cleaning
    - numpy: Numerical computing and array operations

Author: CT Reconstruction Pipeline Team
Created: Part of CLI-based CT reconstruction workflow
"""

import logging
from typing import Dict, List, Any
import numpy as np
from numpy.typing import NDArray

from imars3d.backend.corrections.gamma_filter import gamma_filter

from __code import OperatingMode
from __code import CleaningAlgorithm
from __code.utilities.images import replace_pixels
from __code.config import clean_paras


def clean_by_histogram(config_model: Any, 
                      master_data: Dict[Any, List[NDArray[np.floating]]]) -> Dict[Any, List[NDArray[np.floating]]]:
    """
    Clean images using histogram-based outlier detection and replacement.
    
    This function analyzes pixel intensity distributions and replaces outlier
    pixels that fall outside specified histogram bins. It's effective for
    removing hot pixels, dead pixels, and other intensity-based artifacts.
    
    Args:
        config_model: Configuration object containing cleaning parameters
                     Must have histogram_cleaning_settings with:
                     - nbr_bins: Number of histogram bins for analysis
                     - bins_to_exclude: Number of bins to exclude from each end
        master_data: Dictionary containing image data arrays for different types
        
    Returns:
        Cleaned image data with outlier pixels replaced
        
    Note:
        Uses replace_pixels function with configurable radius for neighbor-based
        pixel replacement. If nbr_bins is 0, no cleaning is performed.
    """

    print("cleaning by histogram ... ", end="")
    histogram_cleaning_settings: Any = config_model.histogram_cleaning_settings
    nbr_bins: int = histogram_cleaning_settings.nbr_bins
    nbr_bins_to_exclude: int = histogram_cleaning_settings.bins_to_exclude

    if nbr_bins != 0:
        # low_gate = config_model.image_cleaner.low_gate
        # high_gate = config_model.image_cleaner.high_gate
        correct_radius: int = clean_paras['correct_radius']

        for _data_type in master_data.keys():
            cleaned_data: List[NDArray[np.floating]] = []
            for _data in master_data[_data_type]:
                _cleaned_data: NDArray[np.floating] = replace_pixels(im=_data,
                                               low_gate=nbr_bins_to_exclude,
                                               high_gate=nbr_bins - nbr_bins_to_exclude,
                                               nbr_bins=nbr_bins,
                                               correct_radius=correct_radius)
                cleaned_data.append(_cleaned_data)
            master_data[_data_type] = cleaned_data

    print("done!")
    return master_data


def clean_by_threshold(config_model: Any, 
                      master_data: Dict[Any, List[NDArray[np.floating]]]) -> Dict[Any, List[NDArray[np.floating]]]:
    """
    Clean images using gamma filter-based threshold cleaning.
    
    This function applies gamma filtering to remove outlier pixels based on
    threshold analysis. The gamma filter is effective at removing noise and
    correcting for detector non-uniformities while preserving image structure.
    
    Args:
        config_model: Configuration object (parameters extracted from arrays)
        master_data: Dictionary containing image data arrays for different types
        
    Returns:
        Cleaned image data with gamma filter applied
        
    Note:
        The gamma filter automatically determines appropriate thresholds based
        on the input data characteristics and applies corrections accordingly.
    """

    print("cleaning by threshold ... ", end="")
    for _data_type in master_data.keys():
        _data: NDArray[np.floating] = np.array(master_data[_data_type])
        _cleaned_data: NDArray[np.floating] = gamma_filter(arrays=_data)
        master_data[_data_type] = _cleaned_data

    print("done!")
    return master_data


def clean_images(config_model: Any, 
                master_data: Dict[Any, List[NDArray[np.floating]]]) -> Dict[Any, List[NDArray[np.floating]]]:
    """
    Apply multiple image cleaning algorithms to CT projection data.
    
    This function orchestrates the application of various cleaning algorithms
    based on the configuration settings. It processes the data sequentially
    through the selected cleaning methods to improve image quality.
    
    Args:
        config_model: Configuration object containing:
                     - list_clean_algorithm: List of cleaning algorithms to apply
        master_data: Dictionary containing image data arrays for different types
        
    Returns:
        Cleaned image data with selected algorithms applied
        
    Available Cleaning Methods:
        - CleaningAlgorithm.histogram: Histogram-based outlier cleaning
        - CleaningAlgorithm.threshold: Gamma filter-based threshold cleaning
        
    Note:
        Algorithms are applied in the order they appear in the configuration.
        If no cleaning algorithms are specified, returns original data unchanged.
    """

    list_clean_algorithm: List[CleaningAlgorithm] = config_model.list_clean_algorithm

    if CleaningAlgorithm.histogram in list_clean_algorithm:
        master_data = clean_by_histogram(config_model, master_data)

    if CleaningAlgorithm.threshold in list_clean_algorithm:
        master_data = clean_by_threshold(config_model, master_data)

    return master_data
