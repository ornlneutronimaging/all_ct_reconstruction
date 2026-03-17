"""
Image Processing Utilities for CT Reconstruction Pipeline

This module provides image processing functions for cleaning and preprocessing
images in the CT reconstruction workflow. Functions include outlier pixel
replacement and noise reduction techniques commonly needed for neutron imaging
data preparation.

Functions:
    replace_pixels: Replace outlier pixels using median filtering

Dependencies:
    - numpy: Numerical computations
    - scipy.ndimage: Image filtering operations

Author: CT Reconstruction Development Team
"""

import logging
import os

import numpy as np
import tomopy
from scipy.ndimage import median_filter
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def replace_pixels(im: NDArray[np.floating], 
                   nbr_bins: int = 0, 
                   low_gate: int = 1, 
                   high_gate: int = 9, 
                   correct_radius: int = 1) -> NDArray[np.floating]:
    """
    Replace outlier pixels in an image using median filtering.
    
    Identifies pixels that fall outside specified histogram bins and replaces
    them with values from a median-filtered version of the image. This is
    commonly used to remove hot pixels, dead pixels, and other imaging
    artifacts in neutron CT data.
    
    Args:
        im: Input image array
        nbr_bins: Number of histogram bins for threshold calculation  
        low_gate: Lower histogram bin index for threshold
        high_gate: Upper histogram bin index for threshold
        correct_radius: Radius for median filter correction
        
    Returns:
        Image array with outlier pixels replaced
        
    Example:
        >>> import numpy as np
        >>> # Create test image with outliers
        >>> image = np.random.normal(100, 10, (256, 256))
        >>> image[50, 50] = 1000  # Hot pixel
        >>> image[100, 100] = 0   # Dead pixel
        >>> corrected = replace_pixels(image, nbr_bins=100)
        
    Note:
        The function uses histogram analysis to identify thresholds and
        replaces outlier pixels with median-filtered values to preserve
        local image structure while removing artifacts.
    """

    _, bin_edges = np.histogram(im.flatten(), bins=nbr_bins, density=False)
    thres_low = bin_edges[low_gate]
    thres_high = bin_edges[high_gate]

    y_coords, x_coords = np.nonzero(np.logical_or(im <= thres_low, 
                                                  im > thres_high))

    full_median_filter_corr_im = median_filter(im, size=correct_radius)
    for y, x in zip(y_coords, x_coords):
        im[y, x] = full_median_filter_corr_im[y, x]

    return im


def gamma_filter(
    arrays: NDArray[np.floating],
    threshold: int = -1,
    median_kernel: int = 5,
    axis: int = 0,
    max_workers: int = 0,
    selective_median_filter: bool = True,
    diff_tomopy: float = -1,
) -> NDArray[np.floating]:
    """Replace near-saturated pixels (from gamma radiation) with median values.

    Ported from imars3d.backend.corrections.gamma_filter. Uses
    tomopy.remove_outlier for the underlying median filtering.

    Args:
        arrays: 3D array of images (first dimension is rotation angle).
        threshold: Saturation threshold. -1 uses dtype max - 5.
        median_kernel: Size of the median filter kernel.
        axis: Axis along which to chunk for parallel filtering.
        max_workers: Number of cores (0 = all available minus 2).
        selective_median_filter: If True, only replace pixels above threshold.
        diff_tomopy: Outlier detection threshold for tomopy. Negative values
            use 20% of saturation intensity.

    Returns:
        Corrected 3D array of images.
    """
    if max_workers <= 0:
        max_workers = max(1, os.cpu_count() - 2)

    try:
        saturation_intensity = np.iinfo(arrays.dtype).max
    except ValueError:
        saturation_intensity = 65535

    if threshold == -1:
        threshold = saturation_intensity - 5

    if diff_tomopy < 0:
        diff_tomopy = 0.2 * saturation_intensity

    arrays_filtered = tomopy.remove_outlier(
        arrays,
        dif=diff_tomopy,
        size=median_kernel,
        axis=axis,
        ncore=max_workers,
    )

    if selective_median_filter:
        arrays_filtered = np.where(
            arrays > threshold,
            arrays_filtered,
            arrays,
        )

    return arrays_filtered
