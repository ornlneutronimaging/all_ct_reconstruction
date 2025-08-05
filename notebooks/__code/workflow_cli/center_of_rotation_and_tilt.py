"""
Center of Rotation and Tilt Correction for CLI-Based CT Reconstruction.

This module provides command-line interface compatible functions for determining
and correcting center of rotation (COR) and tilt angles in computed tomography
reconstruction. It implements automated correction algorithms using 0° and 180°
projection pairs for accurate geometric alignment.

Key Functions:
    - center_of_rotation_and_tilt: Main correction function for COR and tilt
    - get_0_and_180_degrees_images: Extract 0° and 180° projection pairs

Key Features:
    - Automatic COR and tilt correction using neutompy library
    - 0° and 180° projection pair identification and extraction
    - ROI-based correction for improved accuracy
    - Integration with CLI configuration management
    - Support for slice-range based correction

Mathematical Background:
    Center of rotation correction is essential for accurate CT reconstruction.
    The algorithm uses paired 0° and 180° projections to determine geometric
    misalignments and applies corrections to minimize ring artifacts and
    improve reconstruction quality.

Dependencies:
    - neutompy: Neutron tomography preprocessing functions
    - numpy: Numerical computing and array operations

Author: CT Reconstruction Pipeline Team
Created: Part of CLI-based CT reconstruction workflow
"""

import numpy as np
from typing import Tuple, List, Any, Union
from numpy.typing import NDArray
from neutompy.preproc.preproc import correction_COR


def center_of_rotation_and_tilt(config_model: Any, data_array: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Apply center of rotation and tilt correction to CT data.
    
    This function performs automatic center of rotation and tilt correction
    using paired 0° and 180° projections. The correction is applied to
    specified slice ranges to improve reconstruction accuracy and reduce
    ring artifacts.
    
    Args:
        config_model: Configuration object containing correction parameters
                     Must have attributes:
                     - calculate_center_of_rotation: bool flag to enable/disable
                     - range_of_slices_for_center_of_rotation: [top, bottom] slice range
                     - list_of_angles: List of projection angles in degrees
        data_array: 3D projection data array (projections x height x width)
        
    Returns:
        Corrected projection data with COR and tilt adjustments applied
        
    Note:
        If calculate_center_of_rotation is False, returns original data unchanged.
        The correction uses ROI-based approach with top and bottom slice regions.
    """
    if not config_model.calculate_center_of_rotation:
        print(f"skipped center of rotation and tilt calculation!")
        return data_array
    
    print(f"center of rotation and tilt calculation ...", end="")
    top_slice: int
    bottom_slice: int
    [top_slice, bottom_slice] = config_model.range_of_slices_for_center_of_rotation
    list_of_angles: List[float] = config_model.list_of_angles
    image_0_degree: NDArray[np.floating]
    image_180_degree: NDArray[np.floating]
    image_0_degree, image_180_degree = get_0_and_180_degrees_images(list_of_angles=list_of_angles,
                                                                    data_array=data_array)

    mid_point: int = int(np.mean([top_slice, bottom_slice]))
    rois: Tuple[Tuple[int, int], Tuple[int, int]] = ((top_slice, mid_point+1), (mid_point, bottom_slice))

    corrected_images: NDArray[np.floating] = correction_COR(data_array,
                                      image_0_degree,
                                      image_180_degree,
                                      rois=rois)
    
    print(f" done!")
    return corrected_images


def get_0_and_180_degrees_images(list_of_angles: List[float], 
                                data_array: NDArray[np.floating]) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Extract 0° and 180° projection images from the dataset.
    
    This function identifies and extracts the projection images closest to
    0° and 180° angles from the provided dataset. These paired projections
    are essential for center of rotation and tilt correction algorithms.
    
    Args:
        list_of_angles: List of projection angles in degrees
        data_array: 3D projection data array (projections x height x width)
        
    Returns:
        Tuple containing:
        - image_0_degree: Projection image closest to 0°
        - image_180_degree: Projection image closest to 180°
        
    Algorithm:
        1. Calculate absolute differences between each angle and 180°
        2. Find the angle with minimum difference to 180°
        3. Use first projection (index 0) as 0° reference
        4. Extract corresponding projection images
    """
    angles_minus_180: List[float] = [float(_value) - 180 for _value in list_of_angles]
    abs_angles_minus_180: NDArray[np.floating] = np.abs(angles_minus_180)
    minimum_value: float = np.min(abs_angles_minus_180)

    index_0_degree: int = 0
    index_180_degree: int = np.where(minimum_value == abs_angles_minus_180)[0][0]

    image_0_degree: NDArray[np.floating] = data_array[index_0_degree]
    image_180_degree: NDArray[np.floating] = data_array[index_180_degree]

    return image_0_degree, image_180_degree
