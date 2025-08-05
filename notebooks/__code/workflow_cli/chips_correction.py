"""
Chip Correction Module for CLI-Based CT Reconstruction.

This module provides command-line interface compatible functions for correcting
alignment issues between detector chips in multi-chip detector systems. It applies
geometric corrections to compensate for mechanical misalignments between detector
elements that can cause artifacts in reconstructed images.

Key Functions:
    - correct_data: Main chip correction function for CLI workflow

Key Features:
    - Multi-chip detector alignment correction
    - Axis transformation handling for different data formats
    - Integration with ChipsCorrection class from workflow module
    - Predefined chip offset configurations
    - Support for various detector geometries

Dependencies:
    - numpy: Numerical computing and array operations
    - ChipsCorrection: Main correction algorithm implementation

Author: CT Reconstruction Pipeline Team
Created: Part of CLI-based CT reconstruction workflow
"""

import numpy as np
from typing import List
from numpy.typing import NDArray

from __code.workflow.chips_correction import ChipsCorrection
from __code.config import chips_offset


def correct_data(data_array: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Apply chip alignment correction to CT projection data.
    
    This function corrects alignment issues between multiple detector chips
    by applying geometric transformations based on predefined offset values.
    The correction compensates for mechanical misalignments that can cause
    image artifacts and improve overall reconstruction quality.
    
    Args:
        data_array: 3D projection data array (projections x height x width)
                   Input format: (angle, y, x)
        
    Returns:
        Corrected projection data with chip alignment adjustments applied
        Same shape as input: (projections x height x width)
        
    Note:
        The function performs axis transformations to match the expected
        input format for the ChipsCorrection algorithm (y, x, angle) and
        then transforms back to the original format (angle, y, x).
        
    Process:
        1. Convert from (angle, y, x) to (y, x, angle) format
        2. Apply chip alignment correction with predefined offsets
        3. Convert back to (angle, y, x) format
    """
    print("chips correction ...", end="")
    offset: List[float] = list(chips_offset)

    normalized_images: NDArray[np.floating] = np.array(data_array)
    normalized_images_axis_swap: NDArray[np.floating] = np.moveaxis(normalized_images, 0, 2)  # y, x, angle
    corrected_images: NDArray[np.floating] = ChipsCorrection.correct_alignment(normalized_images_axis_swap,
                                                         offsets=offset)
    corrected_images = np.moveaxis(corrected_images, 2, 0)  

    print(f" done!")
    return corrected_images
