"""
Data Loading Module for CLI-Based CT Reconstruction.

This module provides command-line interface compatible functions for loading
CT projection data including sample and open beam measurements. It supports
multi-threaded loading for performance optimization and handles different
operating modes (TOF and white beam).

Key Functions:
    - load_data: Main data loading function for CLI workflow

Key Features:
    - Multi-threaded TIFF file loading for performance
    - Support for sample and open beam data loading
    - Integration with CLI configuration management
    - Operating mode specific data handling (TOF vs white beam)
    - Progress tracking with visual feedback
    - Comprehensive logging for debugging

Dependencies:
    - tqdm: Progress bar visualization
    - logging: Progress tracking and debugging

Author: CT Reconstruction Pipeline Team
Created: Part of CLI-based CT reconstruction workflow
"""

import os
import logging
from typing import Dict, List, Any
from tqdm import tqdm
from numpy.typing import NDArray
import numpy as np

from __code import DataType, OperatingMode
from __code.utilities.files import retrieve_list_of_tif
from __code.utilities.load import load_data_using_multithreading


def load_data(config_model: Any) -> Dict[DataType, List[NDArray[np.floating]]]:
    """
    Load CT projection data for sample and open beam measurements.
    
    This function loads all required projection data based on the configuration
    model settings. It handles both time-of-flight (TOF) and white beam operating
    modes, loading data from specified run directories with multi-threading
    support for improved performance.
    
    Args:
        config_model: Configuration object containing data loading parameters
                     Must have attributes:
                     - operating_mode: OperatingMode enum (TOF or white_beam)
                     - list_of_sample_runs: List of sample run directory names
                     - list_of_ob_runs: List of open beam run directory names
                     - top_folder.sample: Base path for sample data
                     - top_folder.ob: Base path for open beam data
                     
    Returns:
        Dictionary containing loaded 3D data arrays:
        {DataType.sample: [array1, array2, ...],
         DataType.ob: [array1, array2, ...]}
        Each array has shape (projections x height x width) or 
        (tof_channels x projections x height x width) for TOF mode
        
    Note:
        In white beam mode, TOF channels are combined during loading.
        Progress is displayed via tqdm progress bars.
    """

    logging.info(f"loading the data:")
    print(f"Loading the data ... ")
    operating_mode: OperatingMode = config_model.operating_mode
    combine: bool = operating_mode == OperatingMode.white_beam

    # load sample and ob
    list_sample_runs: List[str] = config_model.list_of_sample_runs
    sample_base_folder_name: str = config_model.top_folder.sample
    logging.info(f"{list_sample_runs = }")
    logging.info(f"{sample_base_folder_name = }")

    list_ob_runs: List[str] = config_model.list_of_ob_runs
    ob_base_folder_name: str = config_model.top_folder.ob
    logging.info(f"{list_ob_runs = }")
    logging.info(f"{ob_base_folder_name = }")

    list_of_runs: Dict[DataType, List[str]] = {DataType.sample: [os.path.join(sample_base_folder_name, _file) for _file in list_sample_runs],
                    DataType.ob: [os.path.join(ob_base_folder_name, _file) for _file in list_ob_runs]}

    logging.info(f"{list_of_runs =}")

    master_3d_data_array: Dict[DataType, List[NDArray[np.floating]]] = {DataType.sample: [],
                            DataType.ob: []}
    for _data_type in list_of_runs.keys():
        logging.info(f"\tloading {_data_type}:")
        for _full_path_run in tqdm(list_of_runs[_data_type]):
            logging.info(f"\t{os.path.basename(_full_path_run)}")
            list_tif: List[str] = retrieve_list_of_tif(_full_path_run)
            master_3d_data_array[_data_type].append(load_data_using_multithreading(list_tif,
                                                               combine_tof=combine))

    print(f"done loading the data!")
    return master_3d_data_array
