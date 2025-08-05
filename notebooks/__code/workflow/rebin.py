"""
Rebinning Utilities for CT Reconstruction Pipeline.

This module provides rebinning (pixel binning) functionality for computed tomography
data processing. Rebinning is used to reduce image resolution by combining adjacent
pixels, which can improve signal-to-noise ratio and reduce computational load during
reconstruction, especially for high-resolution datasets.

Key Classes:
    - Rebin: Main class for image rebinning operations

Key Features:
    - Interactive rebinning factor selection (1-9x)
    - Support for rebinning before normalization (raw data)
    - Support for rebinning after normalization (processed data)
    - Separate handling of sample, open beam, and dark current data
    - Automatic backup of original data before rebinning
    - Memory-efficient block reduction using scikit-image

Rebinning Methods:
    - Before normalization: Uses np.sum to preserve photon counts
    - After normalization: Uses np.mean to preserve normalized values

Mathematical Background:
    Rebinning reduces image dimensions by combining adjacent pixels in blocks.
    For a rebinning factor of N, each NxN pixel block is reduced to a single pixel.
    This operation reduces noise by sqrt(N²) = N times while reducing resolution.

Dependencies:
    - ipywidgets: Interactive widget controls
    - IPython.display: Widget display functionality
    - skimage.measure: Block reduction algorithms
    - numpy: Numerical array operations
    - logging: Progress tracking and debugging

Author: CT Reconstruction Pipeline Team
Created: Part of CT reconstruction development workflow
"""

from ipywidgets import interactive
from IPython.display import display
from IPython.display import HTML
import ipywidgets as widgets
import logging
from skimage.measure import block_reduce
import numpy as np
from typing import Optional
from numpy.typing import NDArray

from __code.parent import Parent
from __code import DataType


class Rebin(Parent):
    """
    Handles rebinning (pixel binning) operations for CT reconstruction data.
    
    This class provides functionality to reduce image resolution by combining adjacent
    pixels in blocks, which improves signal-to-noise ratio and reduces computational
    requirements for reconstruction. Supports rebinning both before and after 
    normalization with appropriate aggregation methods.
    
    Attributes:
        rebin_value (widgets.Dropdown): Interactive widget for selecting rebinning factor
        
    Methods:
        set_rebinning(): Creates interactive widget for rebinning factor selection
        execute_binning_before_normalization(): Applies rebinning to raw data
        execute_binning_after_normalization(): Applies rebinning to normalized data
    """

    # rebinning_method = np.sum

    def set_rebinning(self) -> None:
        """
        Create interactive widget for selecting rebinning factor.
        
        Displays a dropdown widget allowing users to select rebinning factor
        from 1 (no rebinning) to 9. The selected factor determines how many
        adjacent pixels are combined into a single pixel.
        
        Returns:
            None: Creates and displays widget interface
            
        Side Effects:
            - Creates self.rebin_value widget
            - Displays widget in notebook interface
        """
        display(widgets.Label('Rebinning factor:'))
        self.rebin_value: widgets.Dropdown = widgets.Dropdown(options=np.arange(1, 10), value=2)
        display(self.rebin_value)
     
    def execute_binning_before_normalization(self) -> None:
        """
        Apply rebinning to raw CT data before normalization.
        
        Performs block reduction on sample, open beam, and dark current images
        using summation to preserve photon count statistics. Creates backup of
        original data before applying rebinning transformation.
        
        The rebinning operation combines adjacent pixels in NxN blocks where N
        is the rebinning factor. For raw data, summation preserves the total
        photon count while reducing spatial resolution.
        
        Returns:
            None: Modifies parent data arrays in place
            
        Side Effects:
            - Backs up original data in self.parent.before_rebinning
            - Modifies self.parent.master_3d_data_array[DataType.sample]
            - Modifies self.parent.master_3d_data_array[DataType.ob] if available
            - Modifies self.parent.master_3d_data_array[DataType.dc] if available
            - Logs progress and rebinning parameters
            
        Notes:
            - Uses np.sum aggregation to preserve photon statistics
            - Skips rebinning if factor is 1
            - Block size is (1, rebin_factor, rebin_factor) to preserve projections
        """
        logging.info(f"Rebinning using factor {self.rebin_value.value} ...")
        rebin_value: int = self.rebin_value.value
        if rebin_value == 1:
            logging.info(f"Rebinning factor is 1, skipping rebinning ...")
            return

        block_size: tuple[int, int, int] = (1, rebin_value, rebin_value)
        logging.info(f"\t{block_size = }")

        master_3d_data_array: dict = self.parent.master_3d_data_array
        self.parent.before_rebinning = master_3d_data_array[DataType.sample][:]
        
        sample_raw_images: NDArray[np.floating] = master_3d_data_array[DataType.sample]
        ob_raw_images: Optional[NDArray[np.floating]] = master_3d_data_array[DataType.ob]
        dc_raw_images: Optional[NDArray[np.floating]] = master_3d_data_array[DataType.dc]
                
        logging.info(f"\rebinning raw data ...")
        dtype: np.dtype = sample_raw_images.dtype
        _sample_data_rebinned: NDArray[np.floating] = block_reduce(sample_raw_images, 
                                    block_size=block_size, 
                                    func=np.sum,
                                    func_kwargs={'dtype': dtype})
        self.parent.master_3d_data_array[DataType.sample] = _sample_data_rebinned[:]
        logging.info(f"\rebinning raw data ... Done!")

        if ob_raw_images is not None:
            logging.info(f"\rebinning ob data ...")
            _ob_data_rebinned: NDArray[np.floating] = block_reduce(ob_raw_images, 
                                        block_size=block_size, 
                                        func=np.sum,
                                        func_kwargs={'dtype': dtype})
            self.parent.master_3d_data_array[DataType.ob] = _ob_data_rebinned[:]
            logging.info(f"\rebinning ob data ... Done!")

        if dc_raw_images is not None:
            logging.info(f"\rebinning dc data ...")
            _dc_data_rebinned: NDArray[np.floating] = block_reduce(dc_raw_images, 
                                        block_size=block_size, 
                                        func=np.sum,
                                        func_kwargs={'dtype': dtype})
            self.parent.master_3d_data_array[DataType.dc] = _dc_data_rebinned[:]
            logging.info(f"\rebinning dc data ... Done!")


    def execute_binning_after_normalization(self) -> None:
        """
        Apply rebinning to normalized CT data after normalization.
        
        Performs block reduction on normalized projection images using mean
        aggregation to preserve normalized intensity values. Creates backup of
        original normalized data before applying rebinning transformation.
        
        The rebinning operation combines adjacent pixels in NxN blocks where N
        is the rebinning factor. For normalized data, mean aggregation preserves
        the intensity scale while reducing spatial resolution and noise.
        
        Returns:
            None: Modifies parent normalized_images array in place
            
        Side Effects:
            - Backs up original data in self.parent.before_rebinning
            - Modifies self.parent.normalized_images array
            - Logs progress and rebinning parameters
            
        Notes:
            - Uses np.mean aggregation to preserve normalized intensities
            - Skips rebinning if factor is 1
            - Block size is (1, rebin_factor, rebin_factor) to preserve projections
            - Preserves original data type for memory efficiency
        """
        logging.info(f"Rebinning using factor {self.rebin_value.value} ...")
        rebin_value: int = self.rebin_value.value
        if rebin_value == 1:
            logging.info(f"Rebinning factor is 1, skipping rebinning ...")
            return
        
        block_size: tuple[int, int, int] = (1, rebin_value, rebin_value)
        logging.info(f"\t{block_size = }")

        normalized_images: NDArray[np.floating] = self.parent.normalized_images
        self.parent.before_rebinning = normalized_images[:]

        logging.info(f"\rebinning normalized data ...")
        dtype: np.dtype = normalized_images.dtype
        # print(f"\t{data_raw.shape = }")
        # print(f"\t(type(_data_raw) = {type(data_raw)}")
        # print(f"\t{block_size = }")
        # print(f"\t{dtype = }")
        _data_rebinned: NDArray[np.floating] = block_reduce(normalized_images, 
                                    block_size=block_size, 
                                    func=np.mean,
                                    func_kwargs={'dtype': dtype})
        self.parent.normalized_images = _data_rebinned[:]

        logging.info(f"\rebinning normalized data ... Done!")
      