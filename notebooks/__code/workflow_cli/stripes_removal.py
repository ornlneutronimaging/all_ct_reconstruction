"""
Stripe Removal Module for CLI-Based CT Reconstruction.

This module provides command-line interface compatible functions for removing
stripe artifacts from CT sinograms. Stripe artifacts appear as horizontal bands
in sinograms and manifest as ring artifacts in reconstructed images. Multiple
algorithms are available to address different types of stripe patterns.

Key Functions:
    - stripes_removal: Main stripe removal function for CLI workflow
    - retrieve_options: Extract algorithm-specific parameters from configuration

Key Classes:
    - StripesRemovalHandler: Alternative handler for stripe removal operations

Key Features:
    - Multiple stripe removal algorithms (FW, TI, SF, sorting, filtering, etc.)
    - Algorithm-specific parameter handling and validation
    - Support for various stripe types (small, large, dead pixels)
    - Multi-threading support for performance optimization
    - Error handling for numerical stability issues

Available Algorithms:
    - Fourier-Wavelet (FW): Advanced wavelet-based stripe removal
    - Titarenko (TI): Statistical approach for ring artifact detection
    - Smoothing Filter (SF): Simple smoothing-based removal
    - Sorting-based: Median filtering approach
    - Filtering-based: Frequency domain filtering
    - Fitting-based: Polynomial fitting approach
    - Interpolation-based: Gap filling with interpolation

Dependencies:
    - tqdm: Progress tracking for processing operations
    - numpy: Numerical computing and array operations

Author: CT Reconstruction Pipeline Team
Created: Part of CLI-based CT reconstruction workflow
"""

from tqdm import tqdm
import logging
from typing import Dict, List, Any, Union, Tuple
import numpy as np
from numpy.typing import NDArray

from __code import RemoveStripeAlgo
from __code.config import NUM_THREADS
from __code.utilities.general import retrieve_parameters
from __code.workflow.remove_strips import RemoveStrips


def retrieve_options(config_model: Any, algorithm: RemoveStripeAlgo) -> Dict[str, Any]:
    """
    Retrieve algorithm-specific parameters for stripe removal.
    
    This function extracts and processes configuration parameters for the
    specified stripe removal algorithm. It handles parameter validation,
    type conversion, and special cases for different algorithm requirements.
    
    Args:
        config_model: Configuration object containing algorithm parameters
        algorithm: Stripe removal algorithm enum value
        
    Returns:
        Dictionary containing processed parameters for the specified algorithm
        
    Note:
        Some algorithms require special parameter processing:
        - FW algorithm: 'None' level parameter is removed
        - Sorting/filtering: 'None' size parameter is removed
        - Fitting: sigma parameter is converted from string to tuple
    """

    print(f"-> {algorithm =}")

    if algorithm == RemoveStripeAlgo.remove_stripe_fw:
        param: Dict[str, Any] = retrieve_parameters(config_model.remove_stripe_fw_options)
        if param['level'] == 'None':
            del param['level']
        return param
    if algorithm == RemoveStripeAlgo.remove_stripe_ti:
        return retrieve_parameters(config_model.remove_stripe_ti_options)
    if algorithm == RemoveStripeAlgo.remove_stripe_sf:
        return retrieve_parameters(config_model.remove_stripe_sf_options)
    if algorithm == RemoveStripeAlgo.remove_stripe_based_sorting:
        param = retrieve_parameters(config_model.remove_stripe_based_sorting_options)
        if param['size'] == 'None':
            del param['size']
        return param
    if algorithm == RemoveStripeAlgo.remove_stripe_based_filtering:
        param = retrieve_parameters(config_model.remove_stripe_based_filtering_options)
        if param['size'] == 'None':
            del param['size']
        return param
    if algorithm == RemoveStripeAlgo.remove_stripe_based_fitting:
        param = retrieve_parameters(config_model.remove_stripe_based_fitting_options)
        left_value: str
        right_value: str
        left_value, right_value = param['sigma'].split(",")
        param['sigma'] = (int(left_value), int(right_value))
        return param
    if algorithm == RemoveStripeAlgo.remove_large_stripe:
        return retrieve_parameters(config_model.remove_large_stripe_options)
    if algorithm == RemoveStripeAlgo.remove_dead_stripe:
        return retrieve_parameters(config_model.remove_dead_stripe_options)
    if algorithm == RemoveStripeAlgo.remove_all_stripe:
        return retrieve_parameters(config_model.remove_all_stripe_options)
    if algorithm == RemoveStripeAlgo.remove_stripe_based_interpolation:
        return retrieve_parameters(config_model.remove_stripe_based_interpolation_options)
    return ""

def stripes_removal(config_model: Any, data_array: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Apply stripe removal algorithms to CT projection data.
    
    This function orchestrates the application of multiple stripe removal
    algorithms to CT sinogram data. It processes the list of configured
    algorithms sequentially, applying each with its specific parameters
    to progressively improve image quality by removing stripe artifacts.
    
    Args:
        config_model: Configuration object containing:
                     - list_clean_stripes_algorithm: List of algorithms to apply
                     - Algorithm-specific option dictionaries
        data_array: 3D projection data array (projections x height x width)
        
    Returns:
        Processed projection data with stripe artifacts removed
        
    Note:
        If no stripe removal algorithms are configured, returns original data.
        Algorithms are applied sequentially in the order specified.
    """
    list_clean_stripes_algorithm: List[RemoveStripeAlgo] = config_model.list_clean_stripes_algorithm

    if len(list_clean_stripes_algorithm) == 0:
        print(f"skipping any stripes removal!")
        return data_array
    
    print("stripes removal:")
    for _algo in list_clean_stripes_algorithm:
        options: Dict[str, Any] = retrieve_options(config_model,
                                   _algo)
        data_array = RemoveStrips.run_algo(RemoveStrips.list_algo[_algo]['function'],
                                           data_array,
                                           **options)
        print(" done!")

    return data_array


class StripesRemovalHandler:
    """
    Alternative handler class for stripe removal operations.
    
    This class provides an alternative interface for stripe removal that can
    be used in different contexts within the CT reconstruction pipeline.
    It includes error handling for numerical stability issues that may occur
    during stripe removal operations.
    """

    @staticmethod
    def remove_stripes(image_array: NDArray[np.floating], config: Dict[str, Any]) -> NDArray[np.floating]:
        """
        Apply stripe removal algorithms on the input image array.
        
        This static method provides an alternative interface for applying
        stripe removal algorithms with comprehensive error handling and
        logging capabilities.
        
        Args:
            image_array: 3D numpy array of shape (angles, slices, pixels)
            config: Configuration dictionary containing parameters for strip removal
                   Expected keys:
                   - 'list_clean_stripes_algorithm': List of algorithm names
                   - 'remove_stripe_{algo}_options': Algorithm-specific parameters
                   
        Returns:
            3D numpy array with stripes removed, same shape as input
            
        Note:
            If no algorithms are specified, returns the original array unchanged.
            Includes error handling for numpy.linalg.LinAlgError exceptions.
        """
        nore: int = NUM_THREADS

        list_algo_to_remove_stripes: List[str] = config.get('list_clean_stripes_algorithm', [])
        if not list_algo_to_remove_stripes:
            logging.info("No strip removal algorithms specified. Returning original image array.")
            return image_array
        
        logging.info(f"Applying strip removal algorithms: {list_algo_to_remove_stripes}")
        try:
            for _algo in list_algo_to_remove_stripes:
                logging.info(f"\t -> Applying {_algo} ...")
                kwargs: Dict[str, Any] = config.get(f'remove_stripe_{_algo.lower()}_options', {})
                kwargs['ncore'] = nore  # Ensure we set the number of threads
                logging.info(f"\t -> Options for {_algo}: {kwargs}")
                image_array = RemoveStrips.run_algo(RemoveStrips.list_algo[_algo]['function'],
                                                    image_array,
                                                    **kwargs)
                logging.info(f"\t -> {_algo} applied successfully.")

        except np.linalg.LinAlgError as e:
            logging.info(f"ERROR: LinAlgError during strip removal: {e} running {_algo}.")
            
        return image_array
