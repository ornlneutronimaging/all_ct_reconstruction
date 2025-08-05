"""
Normalization Module for CLI-Based CT Reconstruction.

This module provides command-line interface compatible functions for normalizing
CT projection data using various correction methods including proton charge,
frame number, and ROI-based normalization. It implements the standard CT
normalization workflow with coefficient calculations and open beam processing.

Key Functions:
    - normalize: Main normalization function with multiple correction methods
    - combine_obs: Combine multiple open beam measurements
    - update_normalize_coeff_by_pc: Proton charge normalization coefficients
    - update_normalize_coeff_by_frame_number: Frame number normalization coefficients
    - update_normalize_coeff_by_roi: ROI-based normalization coefficients

Key Features:
    - Multiple normalization methods (PC, frame number, ROI)
    - Open beam combination and processing
    - Zero-value handling with median filtering
    - Coefficient-based normalization approach
    - Support for various normalization settings

Mathematical Background:
    Standard CT normalization: normalized = (sample / open_beam) * coefficient
    Where coefficient incorporates proton charge, frame number, and ROI corrections.

Dependencies:
    - scipy: Scientific computing and filtering functions
    - numpy: Numerical computing and array operations

Author: CT Reconstruction Pipeline Team
Created: Part of CLI-based CT reconstruction workflow
"""

import numpy as np
from typing import List, Tuple, Any, Dict
from scipy.ndimage import median_filter
from numpy.typing import NDArray

from __code import NormalizationSettings, DataType


def update_normalize_coeff_by_pc(config_model: Any, 
                                list_coeff: List[float], 
                                mean_ob_proton_charge: float) -> List[float]:
    """
    Update normalization coefficients based on proton charge correction.
    
    This function adjusts normalization coefficients to account for differences
    in proton charge between sample and open beam measurements. This correction
    is essential for maintaining consistent neutron flux normalization across
    different measurement conditions.
    
    Args:
        config_model: Configuration object containing sample proton charge data
        list_coeff: Current normalization coefficients for each sample
        mean_ob_proton_charge: Mean proton charge of open beam measurements
        
    Returns:
        Updated normalization coefficients with proton charge correction applied
        
    Formula:
        coeff_new = coeff_old * (mean_ob_pc / sample_pc)
    """
    print(f"updating coeff by proton charge ...", end="")
    list_sample_pc: List[float] = config_model.list_of_sample_pc
    for _index, _sample_pc in enumerate(list_sample_pc):
        list_coeff[_index] *= mean_ob_proton_charge / _sample_pc

    print(f" done!")
    return list_coeff


def update_normalize_coeff_by_frame_number(config_model: Any, 
                                          list_coeff: List[float], 
                                          mean_ob_frame_number: float) -> List[float]:
    """
    Update normalization coefficients based on frame number correction.
    
    This function adjusts normalization coefficients to account for differences
    in frame numbers (exposure counts) between sample and open beam measurements.
    This correction ensures consistent normalization across measurements with
    different exposure durations.
    
    Args:
        config_model: Configuration object containing sample frame number data
        list_coeff: Current normalization coefficients for each sample
        mean_ob_frame_number: Mean frame number of open beam measurements
        
    Returns:
        Updated normalization coefficients with frame number correction applied
        
    Formula:
        coeff_new = coeff_old * (mean_ob_frames / sample_frames)
    """
    print(f"updating coeff by frame number ...", end="")
    list_sample_frame: List[int] = config_model.list_of_sample_frame_number
    for _index, _sample_frame in enumerate(list_sample_frame):
        list_coeff[_index] *= mean_ob_frame_number / _sample_frame

    print(f" done!")
    return list_coeff


def update_normalize_coeff_by_roi(config_model: Any, 
                                 list_coeff: List[float], 
                                 data_array: Dict[DataType, NDArray[np.floating]]) -> List[float]:
    """
    Update normalization coefficients based on ROI (Region of Interest) correction.
    
    This function adjusts normalization coefficients based on intensity ratios
    in a user-defined region of interest. This correction method is useful for
    compensating for beam intensity variations and ensuring consistent
    normalization across the field of view.
    
    Args:
        config_model: Configuration object containing ROI coordinates
                     Must have normalization_roi with top, bottom, left, right attributes
        list_coeff: Current normalization coefficients for each sample
        data_array: Dictionary containing sample and OB data arrays
        
    Returns:
        Updated normalization coefficients with ROI-based correction applied
        
    Formula:
        coeff_new = coeff_old * (ob_roi_sum / sample_roi_sum)
    """
    print(f"update coeff by ROI ...", end="")
    _top: int = config_model.normalization_roi.top
    _bottom: int = config_model.normalization_roi.bottom
    _left: int = config_model.normalization_roi.left
    _right: int = config_model.normalization_roi.right

    ob_roi_counts: float = np.sum(data_array[DataType.ob][_top: _bottom+1, _left: _right+1])
    for _index, _sample_data in enumerate(data_array[DataType.sample]):
        sample_roi_counts: float = np.sum(_sample_data[_top: _bottom+1, _left: _right+1])
        list_coeff[_index] *= ob_roi_counts / sample_roi_counts

    print(f" done!")
    return list_coeff


def combine_obs(config_model: Any = None, 
               data_ob: List[NDArray[np.floating]] = None) -> Tuple[NDArray[np.floating], float, float]:
    """
    Combine multiple open beam measurements into a single reference.
    
    This function processes multiple open beam measurements to create a single
    combined reference for normalization. It handles zero-value pixels using
    median filtering and calculates mean proton charge and frame number values
    for subsequent normalization steps.
    
    Args:
        config_model: Configuration object containing normalization settings
        data_ob: List of open beam data arrays to combine
        
    Returns:
        Tuple containing:
        - obs_combined: Combined open beam reference array
        - mean_proton_charge: Mean proton charge across OB measurements (-1 if not used)
        - mean_frame_number: Mean frame number across OB measurements (-1 if not used)
        
    Note:
        Zero values in the combined OB are replaced with median-filtered values
        to prevent division by zero during normalization.
    """
    print(f"Combining OBs ...", end="")
    if len(data_ob) == 1:
        obs_combined: NDArray[np.floating] = np.array(data_ob[0])
    else:
        obs_combined = np.mean(data_ob, axis=0)
    
    temp_obs_combined: NDArray[np.floating] = median_filter(obs_combined, size=2)
    index_of_zero: Tuple[NDArray[np.integer], ...] = np.where(obs_combined == 0)
    obs_combined[index_of_zero] = temp_obs_combined[index_of_zero]

    mean_proton_charge: float = -1
    if NormalizationSettings.pc in config_model.list_normalization_settings:
        list_pc: List[float] = config_model.list_of_ob_pc
        mean_proton_charge = np.mean(list_pc)

    mean_frame_number: float = -1
    if NormalizationSettings.frame_number in config_model.list_normalization_settings:
        list_frame_number: List[int] = config_model.list_of_ob_frame_number
        mean_frame_number = np.mean(list_frame_number)

    print(f" done!")
    return obs_combined, mean_proton_charge, mean_frame_number


def normalize(config_model: Any, 
             data_array: Dict[DataType, List[NDArray[np.floating]]]) -> List[NDArray[np.floating]]:
    """
    Perform complete normalization of CT projection data.
    
    This function orchestrates the complete normalization workflow including
    open beam combination, coefficient calculation, and final normalization.
    It applies various correction methods based on the configuration settings
    to produce properly normalized projection data ready for reconstruction.
    
    Args:
        config_model: Configuration object containing normalization settings
                     Must have list_normalization_settings attribute
        data_array: Dictionary containing sample and OB data arrays
                   Format: {DataType.sample: [arrays...], DataType.ob: [arrays...]}
        
    Returns:
        List of normalized projection arrays, one for each sample measurement
        
    Normalization Process:
        1. Combine open beam measurements
        2. Calculate normalization coefficients (PC, frame number, ROI)
        3. Apply normalization: normalized = (sample / ob) * coefficient
    """
    
    print(f"running normalization ...")
    mean_ob_proton_charge: float
    mean_ob_frame_number: float
    data_array[DataType.ob], mean_ob_proton_charge, mean_ob_frame_number = combine_obs(config_model=config_model, 
                                                                                       data_ob=data_array[DataType.ob])

    list_normalization_settings_selected: List[NormalizationSettings] = config_model.list_normalization_settings

    list_coeff: List[float] = np.ones(len(data_array[DataType.sample]))
    if NormalizationSettings.pc in list_normalization_settings_selected:
        list_coeff = update_normalize_coeff_by_pc(config_model, list_coeff, mean_ob_proton_charge)

    if NormalizationSettings.frame_number in list_normalization_settings_selected:
        list_coeff = update_normalize_coeff_by_frame_number(config_model, list_coeff, mean_ob_frame_number)

    if NormalizationSettings.roi in list_normalization_settings_selected:
        list_coeff = update_normalize_coeff_by_roi(config_model, list_coeff, data_array)

    normalized_data_array: List[NDArray[np.floating]] = []

    for _coeff, _sample_data, _ob_data in zip(list_coeff, data_array[DataType.sample], data_array[DataType.ob]):
        _normalized: NDArray[np.floating] = np.divide(_sample_data, _ob_data) * _coeff
        normalized_data_array.append(_normalized)

    print(f"running normalization ... done!")
    return normalized_data_array
