"""
Logging utilities for CT reconstruction pipeline.

This module provides functions for setting up logging configuration and
logging array statistics commonly used in image processing workflows.
It handles log file creation, formatting, and provides specialized logging
for numpy array analysis.
"""

import logging
import os
import numpy as np
from typing import Optional
from numpy.typing import NDArray


def setup_logging(basename_of_log_file: str = "") -> str:
    """
    Set up logging configuration for the CT reconstruction pipeline.
    
    Creates a log file with the user's name and script name, configures
    logging format and level. Attempts to use a shared log directory
    but falls back to user's home directory if needed.
    
    Args:
        basename_of_log_file: Base name for the log file (usually script name)
        
    Returns:
        Full path to the created log file
        
    Note:
        Log files are created with write mode ('w'), so they overwrite
        existing logs from the same script.
    """
    USER_NAME: str = os.getlogin()  # add user name to the log file name

    default_path: str = "/SNS/VENUS/shared/log/"
    if os.path.exists(default_path) is False:
        # user home folder
        default_path = os.path.join(os.path.expanduser("~"), "log")
    if not os.path.exists(default_path):
        os.makedirs(default_path)

    log_file_name: str = os.path.join(default_path, f"{USER_NAME}_{basename_of_log_file}.log")
    logging.basicConfig(filename=log_file_name,
                        filemode='w',
                        format='[%(levelname)s] - %(asctime)s - %(message)s',
                        level=logging.INFO)
    logging.info(f"*** Starting a new script {basename_of_log_file} ***")

    print(f"logging file: {log_file_name}")

    return log_file_name
    

def logging_3d_array_infos(message: str = "", array: Optional[NDArray[np.generic]] = None) -> None:
    """
    Log statistical information about a 3D numpy array.
    
    This function logs comprehensive statistics about an array including
    min/max values, NaN count, and infinity count. Useful for debugging
    image processing operations and monitoring data quality.
    
    Args:
        message: Descriptive message to include in the log
        array: Numpy array to analyze and log statistics for
        
    Note:
        If array is None, only the message will be logged.
    """
    logging.info(f"{message}")
    if array is not None:
        logging.info(f"{np.min(array) = }")
        logging.info(f"{np.max(array) = }")
        logging.info(f"Number of nan: {np.count_nonzero(np.isnan(array))}")
        logging.info(f"Number of inf: {np.count_nonzero(np.isinf(array))}")
                 