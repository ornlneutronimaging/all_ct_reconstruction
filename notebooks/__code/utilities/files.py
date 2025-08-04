"""
File and folder utility functions for CT reconstruction pipeline.

This module provides utility functions for file and folder operations including:
- Finding and listing TIFF files
- JSON file operations
- Folder creation and management
- File path operations
- Angle extraction from file names

Functions are used throughout the CT reconstruction workflow for data management.
"""

import glob
import os
import json
import shutil
from typing import List, Optional, Union, Any


def retrieve_list_of_files_from_folders(list_folders: List[str]) -> List[str]:
    """
    Retrieve all TIFF files from a list of folders.
    
    Args:
        list_folders: List of folder paths to search for TIFF files
        
    Returns:
        Sorted list of full paths to all TIFF files found in the folders
    """
    list_files: List[str] = []
    for _folder in list_folders:
        _tiff_files: List[str] = glob.glob(os.path.join(_folder, "*.tif*"))
        list_files = [*list_files, *_tiff_files]

    list_files.sort()
    return list_files


def retrieve_list_of_runs(top_folder: str) -> List[str]:
    """
    Retrieve all 'Run_*' folders from a top-level directory.
    
    Args:
        top_folder: Path to the top-level directory to search
        
    Returns:
        Sorted list of full paths to Run_* folders
    """
    list_runs: List[str] = glob.glob(os.path.join(top_folder, "Run_*"))
    list_runs.sort()
    return list_runs


def retrieve_list_of_tif(folder: str) -> List[str]:
    """
    Retrieve all TIFF files from a specific folder.
    
    Args:
        folder: Path to the folder to search for TIFF files
        
    Returns:
        Sorted list of full paths to TIFF files in the folder
    """
    list_tif: List[str] = glob.glob(os.path.join(folder, "*.tif*"))
    list_tif.sort()
    return list_tif


def get_number_of_tif(folder: str) -> int:
    """
    Count the number of TIFF files in a folder.
    
    Args:
        folder: Path to the folder to count TIFF files in
        
    Returns:
        Number of TIFF files found in the folder
    """
    return len(retrieve_list_of_tif(folder))


def load_json(json_file_name: str) -> Optional[Any]:
    """
    Load data from a JSON file.
    
    Args:
        json_file_name: Path to the JSON file to load
        
    Returns:
        Parsed JSON data if file exists, None otherwise
    """
    if not os.path.exists(json_file_name):
        return None

    with open(json_file_name) as json_file:
        data = json.load(json_file)

    return data


def save_json(json_file_name: str, json_dictionary: Optional[Union[dict, str]] = None) -> None:
    """
    Save data to a JSON file.
    
    Args:
        json_file_name: Path where the JSON file should be saved
        json_dictionary: Data to save as JSON (dict or JSON string)
    """
    with open(json_file_name, 'w') as outfile:
        json.dump(json_dictionary, outfile)


def make_or_reset_folder(folder_name: str) -> None:
    """
    Create a folder, removing it first if it already exists.
    
    Args:
        folder_name: Path to the folder to create/reset
    """
    if os.path.exists(folder_name):
         shutil.rmtree(folder_name)
    os.makedirs(folder_name)


def remove_folder(folder_name: str) -> None:
    """
    Remove a folder if it exists.
    
    Args:
        folder_name: Path to the folder to remove
    """
    if not os.path.exists(folder_name):
        return
    shutil.rmtree(folder_name)


def make_folder(folder_name: str) -> None:
    """
    Create a folder if it doesn't already exist.
    
    Args:
        folder_name: Path to the folder to create
    """
    if os.path.exists(folder_name):
        return
    os.makedirs(folder_name)
    

def get_angle_value(run_full_path: Optional[str] = None) -> Optional[str]:
    """
    Extract rotation angle value from TIFF file names.
    
    Parses file names with format:
    Run_####_20240927_date_..._148_443_######_<file_index>.tif
    
    Args:
        run_full_path: Path to folder containing TIFF files
        
    Returns:
        Angle value as string in format "148.443", or None if no files found
    """
    list_tiff: List[str] = retrieve_list_of_tif(run_full_path)
    if len(list_tiff) == 0:
        return None
    
    first_tiff: str = list_tiff[0]
    list_part: List[str] = first_tiff.split("_")
    return f"{list_part[-4]}.{list_part[-3]}"
