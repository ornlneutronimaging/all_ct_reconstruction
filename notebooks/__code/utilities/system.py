"""
System Utilities for CT Reconstruction Pipeline

This module provides system-level utilities for monitoring memory usage, 
managing system resources, and retrieving system information. These functions
are essential for performance monitoring and resource management during
CT reconstruction workflows.

Functions:
    get_memory_usage: Calculate total memory usage of current process and children
    print_memory_usage: Print formatted memory usage information
    retrieve_memory_usage: Get formatted memory usage string
    delete_array: Safely delete array objects and trigger garbage collection
    get_user_name: Retrieve current system user name
    get_instrument_generic_name: Map instrument codes to generic names

Author: CT Reconstruction Development Team
"""

import os
import psutil
import gc
from typing import Optional, Any


def get_memory_usage() -> float:
    """
    Calculate the total memory usage of the current process and its children.

    Returns
    -------
    float
        Total memory usage in MB.
    """
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    mem_usage_mb = mem_info.rss / (1024 ** 2)  # Convert bytes to MB

    # Include memory usage of child processes
    for child in process.children(recursive=True):
        try:
            mem_info = child.memory_info()
            mem_usage_mb += mem_info.rss / (1024 ** 2)
        except psutil.NoSuchProcess:
            continue

    return mem_usage_mb


def print_memory_usage(message: str = "", end: str = "\n") -> None:
    """
    Print formatted memory usage information.
    
    Displays current total memory usage with appropriate units (MB or GB)
    and optional custom message prefix.
    
    Args:
        message: Optional prefix message to display
        end: String appended after the last value, default newline
        
    Example:
        >>> print_memory_usage("After loading data")
        After loading data: total memory usage = 245.67MB
    """
    mem_usage = get_memory_usage()

    if mem_usage < 999:
        units = 'MB'
    else:
        mem_usage /= 1000
        units = 'GB'
    print(f"{message}: total memory usage = {mem_usage:.2f}{units}", end=end)


def retrieve_memory_usage() -> str:
    """
    Retrieve formatted memory usage as a string.
    
    Returns memory usage with appropriate units (MB or GB) as a formatted
    string suitable for logging or display purposes.
    
    Returns:
        Formatted memory usage string (e.g., "245.67MB" or "1.23GB")
        
    Example:
        >>> usage = retrieve_memory_usage()
        >>> print(f"Current usage: {usage}")
        Current usage: 512.34MB
    """

    mem_usage = get_memory_usage()

    if mem_usage < 999:
        units = 'MB'
    else:
        mem_usage /= 1000
        units = 'GB'

    return f"{mem_usage:.2f}{units}"


def delete_array(object_array: Optional[Any] = None) -> None:
    """
    Safely delete array object and trigger garbage collection.
    
    Sets the array object to None and explicitly calls garbage collection
    to free memory. Useful for managing memory during large data processing.
    
    Args:
        object_array: Array or object to delete from memory
        
    Example:
        >>> import numpy as np
        >>> large_array = np.zeros((1000, 1000, 1000))
        >>> delete_array(large_array)  # Frees memory
    """
    object_array = None
    gc.collect()
    

def get_user_name() -> str:
    """
    Retrieve the current system user name.
    
    Returns the login name of the current user, commonly used for
    log file naming and user-specific file operations.
    
    Returns:
        Current user's login name
        
    Example:
        >>> user = get_user_name()
        >>> log_file = f"reconstruction_{user}.log"
    """
    return os.getlogin()


def get_instrument_generic_name(instrument: str) -> str:
    """
    Map instrument codes to standardized generic names.
    
    Converts specific instrument identifiers to their generic names
    used throughout the neutron scattering facility. Supports common
    beamline and instrument codes.
    
    Args:
        instrument: Instrument identifier or beamline code
        
    Returns:
        Generic instrument name (MARS, SNAP, VENUS) or original string
        if not recognized
        
    Example:
        >>> get_instrument_generic_name("CG1D")
        'MARS'
        >>> get_instrument_generic_name("BL3_SNAP")
        'SNAP'
        >>> get_instrument_generic_name("BL10")
        'VENUS'
        >>> get_instrument_generic_name("unknown")
        'unknown'
    """
    instrument = instrument.lower()  
    
    if instrument.startswith("cg1d"):
        return "MARS"
    elif instrument.startswith("bl3"):
        return "SNAP"
    elif instrument.startswith("bl10"):
        return "VENUS"
    else:
        return instrument  # Return as is if not recognized
    