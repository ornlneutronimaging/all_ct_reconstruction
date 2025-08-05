"""
Image and Data Saving Utilities for CT Reconstruction Pipeline

This module provides functions for saving image data in various formats commonly
used in scientific imaging and CT reconstruction workflows. Supports TIFF and
FITS formats with metadata preservation capabilities.

Functions:
    make_tiff: Save data as TIFF image with optional metadata
    make_fits: Save data as FITS file for astronomical/scientific data

Author: CT Reconstruction Development Team
"""

from typing import Optional, Any
import numpy as np
from PIL import Image
from astropy.io import fits
from numpy.typing import NDArray


def make_tiff(data: NDArray[Any] = None, 
              filename: str = '', 
              metadata: Optional[dict] = None) -> None:
    """
    Save numpy array data as a TIFF image file.
    
    Creates a TIFF image from numpy array data with optional metadata
    preservation. Commonly used for saving reconstructed CT slices,
    processed images, and intermediate results in the reconstruction pipeline.
    
    Args:
        data: Numpy array containing image data to save
        filename: Output TIFF file path
        metadata: Optional TIFF metadata dictionary to embed in the file
        
    Raises:
        ValueError: If data is None or filename is empty
        IOError: If file cannot be written to specified location
        
    Example:
        >>> import numpy as np
        >>> data = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        >>> make_tiff(data, "output.tiff")
        >>> # With metadata
        >>> meta = {"description": "CT reconstruction slice"}
        >>> make_tiff(data, "slice_001.tiff", metadata=meta)
    """
    if data is None:
        raise ValueError("Data array cannot be None")
    if not filename:
        raise ValueError("Filename cannot be empty")
        
    new_image = Image.fromarray(np.array(data))
    if metadata:
        new_image.save(filename, tiffinfo=metadata)
    else:
        new_image.save(filename)


def make_fits(data: NDArray[Any] = None, filename: str = "") -> None:
    """
    Save numpy array data as a FITS (Flexible Image Transport System) file.
    
    Creates a FITS file from numpy array data. FITS is commonly used in
    astronomical and scientific applications for storing image data with
    comprehensive metadata support.
    
    Args:
        data: Numpy array containing data to save
        filename: Output FITS file path
        
    Raises:
        ValueError: If data is None or filename is empty
        IOError: If file cannot be written to specified location
        
    Example:
        >>> import numpy as np
        >>> data = np.random.random((256, 256)).astype(np.float32)
        >>> make_fits(data, "reconstruction.fits")
        
    Note:
        This function overwrites existing files without warning.
        FITS format preserves data precision and supports metadata headers.
    """
    if data is None:
        raise ValueError("Data array cannot be None")
    if not filename:
        raise ValueError("Filename cannot be empty")
        
    fits.writeto(filename, data, overwrite=True)