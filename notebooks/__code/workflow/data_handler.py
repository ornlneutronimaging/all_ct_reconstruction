"""
Data Handling Utilities for CT Reconstruction Pipeline.

This module provides utility functions for data cleaning and preprocessing 
in computed tomography (CT) reconstruction workflows. It handles common 
data quality issues such as negative values and zero values in normalized 
datasets using TomoPy library functions.

Key Functions:
    - remove_negative_values: Removes negative values from normalized CT data
    - remove_0_values: Removes zero values from normalized CT data

Dependencies:
    - tomopy: Core tomographic reconstruction library
    - numpy: Numerical computing support
    - logging: Progress tracking and debugging

Author: CT Reconstruction Pipeline Team
Created: Part of CT reconstruction development workflow
"""

import logging
from typing import Union
import tomopy
import numpy as np
from numpy.typing import NDArray

from __code.config import NUM_THREADS


def remove_negative_values(normalized_data: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Remove negative values from normalized CT data using TomoPy correction.
    
    This function checks for negative values in normalized tomographic data
    and replaces them with zeros using TomoPy's remove_neg function. Negative
    values can occur during normalization and can cause issues in reconstruction
    algorithms, particularly with logarithmic operations.
    
    Parameters
    ----------
    normalized_data : NDArray[np.floating]
        3D normalized tomographic data array with shape (angles, height, width).
        Expected to contain floating-point values from normalization process.
    
    Returns
    -------
    NDArray[np.floating]
        Cleaned data array with negative values replaced by zeros.
        Returns copy of original data if no negative values found.
    
    Notes
    -----
    - Uses TomoPy's parallel processing with NUM_THREADS cores
    - Logs progress and results for debugging purposes
    - Returns array copy to avoid modifying original data
    
    Examples
    --------
    >>> import numpy as np
    >>> data = np.array([[[1.0, -0.5], [0.8, 0.2]]])
    >>> cleaned = remove_negative_values(data)
    >>> # Negative values replaced with 0.0
    """
    logging.info("Checking for negative values in normalized_data:")
    if normalized_data.any() < 0:
        logging.info("\tRemoving negative values")
        cleaned_data: NDArray[np.floating] = tomopy.misc.corr.remove_neg(
            normalized_data, val=0.0, ncore=NUM_THREADS
        )
        logging.info("Negative values removed!")
        return cleaned_data[:]
    else:
        return normalized_data[:]
    
def remove_0_values(normalized_data: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Remove zero values from normalized CT data using TomoPy correction.
    
    This function checks for zero values in normalized tomographic data
    and replaces them with NaN values using TomoPy's remove_zero function.
    Zero values can cause division by zero errors and infinite values in
    logarithmic operations during reconstruction.
    
    Parameters
    ----------
    normalized_data : NDArray[np.floating]
        3D normalized tomographic data array with shape (angles, height, width).
        Expected to contain floating-point values from normalization process.
    
    Returns
    -------
    NDArray[np.floating]
        Cleaned data array with zero values replaced by NaN.
        Returns copy of original data if no zero values found.
    
    Notes
    -----
    - Uses TomoPy's parallel processing with NUM_THREADS cores
    - Replaces zeros with np.NaN rather than small positive values
    - Logs progress and results for debugging purposes
    - Returns array copy to avoid modifying original data
    
    Examples
    --------
    >>> import numpy as np
    >>> data = np.array([[[1.0, 0.0], [0.8, 0.2]]])
    >>> cleaned = remove_0_values(data)
    >>> # Zero values replaced with np.NaN
    
    See Also
    --------
    remove_negative_values : Remove negative values from normalized data
    """
    logging.info("Checking for 0 values in normalized_data:")
    if normalized_data.any() == 0:
        logging.info("\tRemoving 0 values")
        cleaned_data: NDArray[np.floating] = tomopy.misc.corr.remove_zero(
            normalized_data, val=np.NaN, ncore=NUM_THREADS
        )
        logging.info("0 values removed!")
        return cleaned_data[:]
    else:
        return normalized_data[:]
    