"""
Folder and directory utilities for CT reconstruction pipeline.

This module provides functions for folder operations including existence
checking, permission validation, and path resolution. These utilities
are essential for ensuring the reconstruction pipeline can access and
write to the required directories.
"""

from pathlib import Path
import os
from typing import Union


def find_first_real_dir(start_dir: Union[str, Path] = "./") -> Path:
    """
    Find the first existing directory by traversing up the directory tree.
    
    This function is useful for finding a valid working directory when
    the specified path might not exist. It walks up the directory tree
    until it finds a directory that actually exists.
    
    Args:
        start_dir: Starting directory path (string or Path object)
        
    Returns:
        Path object representing the first existing directory found
        
    Example:
        >>> find_first_real_dir("/path/that/might/not/exist")
        PosixPath('/path/that/might')  # if /path/that/might exists
    """
    if type(start_dir) is str:
        start_dir = Path(start_dir)

    if start_dir.exists():
        return start_dir

    dir_path: Path = start_dir
    while not Path(dir_path).exists():
        dir_path = Path(dir_path).parent

    return dir_path


def check_folder_write_permission(folder_path: str) -> bool:
    """
    Check if the current user has write permission to the specified folder.
    
    This function is essential for validating that output directories can
    be used for saving reconstruction results, configuration files, and
    temporary data.
    
    Args:
        folder_path: The path to the folder to check
        
    Returns:
        True if write permission is granted, False otherwise
        
    Note:
        This function checks the actual filesystem permissions, not just
        the existence of the folder. The folder must exist for the check
        to be meaningful.
    """
    return os.access(folder_path, os.W_OK)
