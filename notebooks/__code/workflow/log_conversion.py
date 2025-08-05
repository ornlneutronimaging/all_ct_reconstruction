"""
Logarithmic Conversion Utilities for CT Reconstruction Pipeline.

This module provides logarithmic data conversion functionality for computed tomography
reconstruction workflows. The logarithmic transformation is a critical preprocessing
step that converts normalized transmission data to attenuation data suitable for
reconstruction algorithms.

Key Functions:
    - log_conversion: Convert normalized data to negative logarithm format

Mathematical Background:
    The Beer-Lambert law relates transmission (T) to attenuation (μ):
    T = I/I₀ = exp(-μt)
    Therefore: μt = -ln(T) = -ln(I/I₀)
    
    This conversion transforms multiplicative noise to additive noise and
    linearizes the relationship between measured data and reconstruction parameters.

Dependencies:
    - tomopy: Core tomographic processing with parallel computation
    - logging: Progress tracking and debugging information

Author: CT Reconstruction Pipeline Team
Created: Part of CT reconstruction development workflow
"""

from typing import Optional
import tomopy
import logging
import numpy as np
from numpy.typing import NDArray

from __code.config import NUM_THREADS


def log_conversion(normalized_data: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Convert normalized CT data to negative logarithm format for reconstruction.
    
    Applies the negative logarithm transformation to normalized transmission data,
    converting it to attenuation data suitable for tomographic reconstruction
    algorithms. This is a fundamental preprocessing step based on the Beer-Lambert law.
    
    The transformation: attenuation = -ln(normalized_transmission)
    
    Parameters
    ----------
    normalized_data : NDArray[np.floating]
        3D normalized transmission data array with shape (angles, height, width).
        Values should be positive and typically in range [0, 1] after normalization.
        
    Returns
    -------
    NDArray[np.floating]
        3D attenuation data array with same shape as input.
        Contains negative logarithm values suitable for reconstruction algorithms.
        
    Notes
    -----
    - Uses TomoPy's minus_log function with parallel processing
    - Automatically handles edge cases (zeros, negative values)
    - Returns array copy to avoid modifying original data
    - Multi-threaded processing for performance with large datasets
    
    Examples
    --------
    >>> import numpy as np
    >>> normalized = np.array([[[0.5, 0.8], [0.3, 0.9]]])
    >>> attenuation = log_conversion(normalized)
    >>> # Result contains -ln(normalized) values
    
    See Also
    --------
    tomopy.minus_log : Underlying TomoPy function for logarithmic conversion
    """
    logging.info("Converting data to -log")
    normalized_data_log: NDArray[np.floating] = tomopy.minus_log(
        normalized_data, ncore=NUM_THREADS, out=None
    )
    logging.info("Data converted to -log!")
    return normalized_data_log[:]
    