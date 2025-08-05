"""
Open Beam and Dark Current Combination Utilities for CT Reconstruction.

This module provides functionality to combine multiple open beam (OB) and dark current (DC)
images into single representative images for normalization. Combining multiple reference
images improves signal-to-noise ratio and reduces the impact of temporal fluctuations
in beam intensity and detector noise.

Key Classes:
    - CombineObDc: Main class for OB/DC image combination

Key Features:
    - Median-based combination to reduce noise and outliers
    - Support for OB-only combination (ignoring DC)
    - Automatic handling of single-image cases
    - Data type preservation and memory optimization
    - Comprehensive logging of combination process

Mathematical Background:
    Multiple reference images are combined using median aggregation:
    combined_image = median(image_stack, axis=0)
    
    Median combination is preferred over mean because it is more robust
    to outliers such as cosmic rays or detector artifacts.

Dependencies:
    - numpy: Numerical array operations and statistical functions
    - logging: Progress tracking and debugging
    - Parent: Base class providing common functionality

Author: CT Reconstruction Pipeline Team
Created: Part of CT reconstruction development workflow
"""

import numpy as np
import logging
from typing import Optional, List
from numpy.typing import NDArray

from __code.parent import Parent
from __code import DataType


class CombineObDc(Parent):
    """
    Handles combination of multiple open beam and dark current images.
    
    This class provides functionality to combine multiple reference images
    (open beam and dark current) into single representative images using
    median aggregation. This process improves signal-to-noise ratio and
    provides more stable reference data for normalization.
    
    Methods:
        run(ignore_dc=False): Execute the combination process
    """

    def run(self, ignore_dc: bool = False) -> None:
        """
        Combine multiple open beam and dark current images using median aggregation.
        
        Processes multiple reference images by computing the median across the image
        stack, which provides robust combination resistant to outliers and artifacts.
        Handles both open beam and dark current data, with option to ignore DC.
        
        Args:
            ignore_dc (bool, optional): If True, only combine OB images and skip DC.
                                      Defaults to False (combine both OB and DC).
                                      
        Returns:
            None: Modifies parent data arrays in place
            
        Side Effects:
            - Updates self.parent.master_3d_data_array[DataType.ob] with combined OB
            - Updates self.parent.master_3d_data_array[DataType.dc] with combined DC
            - Converts sample data to numpy array format
            - Logs combination progress and array shapes
            
        Notes:
            - Uses median aggregation to reduce noise and outlier sensitivity
            - Skips combination if only one image is available per data type
            - Preserves data type as np.ushort for memory efficiency
            - Handles missing data gracefully with appropriate warnings
        """

        if self.parent.master_3d_data_array[DataType.ob] is None:
            logging.warning(f"Combine obs: No ob data found, skipping combination.")
            return

        if ignore_dc:
            logging.info(f"Combine obs:")
            list_to_combine: List[DataType] = [DataType.ob]
        else:
            logging.info(f"Combine obs and dcs:")
            list_to_combine: List[DataType] = [DataType.ob, DataType.dc]

        master_3d_data_array: dict = self.parent.master_3d_data_array
        self.parent.master_3d_data_array[DataType.sample] = np.array(self.parent.master_3d_data_array[DataType.sample])

        for _data_type in list_to_combine:
#           if self.parent.list_of_images[_data_type] is not None:
            if master_3d_data_array[_data_type] is not None:
                logging.info(f"\t{_data_type} -> {np.shape(master_3d_data_array[_data_type])}")
                logging.info(f"\t             -> data type: {master_3d_data_array[_data_type].dtype}")
                # if len(self.parent.list_of_images[_data_type]) == 1: # only 1 image
                if len(master_3d_data_array[_data_type]) == 1: # only 1 image
                    continue
                else:
                    _combined_array: NDArray[np.ushort] = np.median(np.array(master_3d_data_array[_data_type]), axis=0).astype(np.ushort)
                    master_3d_data_array[_data_type] = _combined_array[:]
                    logging.info(f"\t{_data_type} -> {np.shape(master_3d_data_array[_data_type])}")
            else:
                logging.info(f"\t{_data_type} skipped!")

        self.parent.master_3d_data_array = master_3d_data_array

        if ignore_dc:
            logging.info(f"Combined obs!")    
        else:
            logging.info(f"Combined obs and dcs done !")    
