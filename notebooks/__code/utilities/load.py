"""
Image loading utilities for CT reconstruction pipeline.

This module provides functions for loading TIFF images both sequentially and
using multiprocessing. It handles various data types and formats commonly
used in neutron imaging and CT reconstruction workflows.
"""

from skimage.io import imread
import numpy as np
import os
import multiprocessing as mp 
import dxchange
import logging
from typing import List, Optional, Union
from numpy.typing import NDArray

# from NeuNorm.normalization import Normalization

from __code.utilities.files import retrieve_list_of_tif
from __code import LOAD_DTYPE


def _worker(fl: str) -> NDArray[np.floating]:
    """
    Worker function for multiprocessing image loading.
    
    Args:
        fl: File path to TIFF image
        
    Returns:
        Loaded image data with swapped axes and converted to LOAD_DTYPE
    """
    return (imread(fl).astype(LOAD_DTYPE)).swapaxes(0, 1)


def load_data_using_multithreading(list_tif: List[str], combine_tof: bool = False) -> NDArray[np.floating]:
    """
    Load TIFF images using multiprocessing for improved performance.
    
    Args:
        list_tif: List of TIFF file paths to load
        combine_tof: If True, sum all images along the first axis (for ToF data)
        
    Returns:
        3D numpy array of loaded images, or 2D if combine_tof is True
        
    Note:
        Uses 40 parallel processes for loading. Consider adjusting based on
        system capabilities and memory constraints.
    """
    with mp.Pool(processes=40) as pool:
        data = pool.map(_worker, list_tif)

    if combine_tof:
        return np.array(data).sum(axis=0)
    else:
        return np.array(data)


def load_list_of_images(list_of_images: List[str], dtype: Optional[np.dtype] = None) -> NDArray[np.generic]:  
    """
    Load a list of TIFF files into a 3D numpy array sequentially.
    
    This function is more memory-efficient than multiprocessing for large datasets
    and provides better control over memory usage. It pre-allocates the output
    array based on the first image dimensions.
    
    Args:
        list_of_tiff: List of TIFF file paths to load
        dtype: Data type for the output array (default: np.uint16)
        
    Returns:
        3D numpy array with shape (n_images, height, width)
    """
    
    # find file extension
    [base, extension] = os.path.splitext(list_of_images[0])
    if extension.lower() == '.tif' or extension.lower() == '.tiff':
        return load_list_of_tif(list_of_images, dtype=dtype)
    else:
        return load_list_of_fits(list_of_images, dtype=dtype)


def load_list_of_tif(list_of_tiff: List[str], dtype: Optional[np.dtype] = None) -> NDArray[np.generic]:
    """
    Load a list of TIFF files into a 3D numpy array sequentially.
    
    This function is more memory-efficient than multiprocessing for large datasets
    and provides better control over memory usage. It pre-allocates the output
    array based on the first image dimensions.
    
    Args:
        list_of_tiff: List of TIFF file paths to load
        dtype: Data type for the output array (default: np.uint16)
        
    Returns:
        3D numpy array with shape (n_images, height, width)
        
    Note:
        The first image is loaded twice - once to determine array size,
        then again as part of the full loading process.
    """
    if dtype is None:
        dtype = np.uint16

    # init array
    logging.info(f"loading first image to determine size of 3D array")
    first_image: NDArray[np.generic] = dxchange.read_tiff(list_of_tiff[0])
    size_3d: List[int] = [len(list_of_tiff), np.shape(first_image)[0], np.shape(first_image)[1]]
    data_3d_array: NDArray[np.generic] = np.empty(size_3d, dtype=dtype)

    # load stack of tiff
    logging.info(f"loading {len(list_of_tiff)} images into 3D array of shape {size_3d}")
    for _index, _file in enumerate(list_of_tiff):
        _array: NDArray[np.generic] = dxchange.read_tiff(_file)
        data_3d_array[_index] = _array
    return data_3d_array


def load_list_of_fits(list_of_fits: List[str], dtype: Optional[np.dtype] = None) -> NDArray[np.generic]:
    """
    Load a list of FITS files into a 3D numpy array sequentially.
    
    This function is more memory-efficient than multiprocessing for large datasets
    and provides better control over memory usage. It pre-allocates the output
    array based on the first image dimensions.
    
    Args:
        list_of_fits: List of FITS file paths to load
        dtype: Data type for the output array (default: np.uint16)
        
    Returns:
        3D numpy array with shape (n_images, height, width) 
    """
    if dtype is None:
        dtype = np.uint16

    # init array
    logging.info(f"loading first image to determine size of 3D array")
    first_image: NDArray[np.generic] = dxchange.read_fits(list_of_fits[0])
    size_3d: List[int] = [len(list_of_fits), np.shape(first_image)[0], np.shape(first_image)[1]]
    data_3d_array: NDArray[np.generic] = np.empty(size_3d, dtype=dtype)

    # load stack of fits
    logging.info(f"loading {len(list_of_fits)} images into 3D array of shape {size_3d}")
    for _index, _file in enumerate(list_of_fits):
        _array: NDArray[np.generic] = dxchange.read_fits(_file)
        data_3d_array[_index] = _array
    return data_3d_array


# def load_tiff(tif_file_name):
#     o_norm = Normalization()
#     o_norm.load(tif_file_name)
#     return np.squeeze(o_norm.data['sample']['data'])


# def load_data_using_imread(folder):
#     list_tif = retrieve_list_of_tif(folder)
#     data = []
#     for _file in list_tif:
#         data.append(_hype_loader_sum)
#         data.append((imread(_file).astype(np.float32)))
#     return data
