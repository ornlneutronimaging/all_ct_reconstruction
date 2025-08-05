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

import numpy as np
from scipy.ndimage import median_filter
from typing import Union
from numpy.typing import NDArray


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
