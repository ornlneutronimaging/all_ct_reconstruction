"""
NeXus File Utilities for CT Reconstruction Pipeline

This module provides functions for reading and extracting metadata from NeXus
files, which are the standard data format used at neutron scattering facilities.
These utilities extract essential experimental parameters such as proton charge
and frame numbers needed for proper CT reconstruction normalization and analysis.

Functions:
    get_proton_charge: Extract proton charge from NeXus files
    get_frame_number: Extract frame number from detector logs

Dependencies:
    - h5py: HDF5 file operations
    - logging: Error logging and debugging

Author: CT Reconstruction Development Team
"""

import h5py
import os
import logging
from typing import Optional, Union


def get_proton_charge(nexus: Optional[str], units: str = 'pc') -> Optional[float]:
    """
    Extract proton charge from NeXus file.
    
    Reads the proton charge value from the NeXus file, which is essential
    for normalizing neutron count data in CT reconstruction. The proton
    charge indicates the intensity of the neutron beam during data collection.
    
    Args:
        nexus: Path to NeXus file, or None
        units: Units for proton charge ('pc' for picocoulombs, 'c' for coulombs)
        
    Returns:
        Proton charge value in specified units, or None if file cannot be read
        
    Example:
        >>> charge_pc = get_proton_charge("data.nxs", units='pc')
        >>> charge_c = get_proton_charge("data.nxs", units='c')  # Converted to coulombs
        
    Note:
        Default units are picocoulombs (pc). Use 'c' for conversion to coulombs.
        Returns None if file doesn't exist or proton charge data is unavailable.
    """
    if nexus is None:
        return None
    
    try:
        with h5py.File(nexus, 'r') as hdf5_data:
            proton_charge = hdf5_data["entry"]["proton_charge"][0]
            if units == 'c':
                return proton_charge/1e12
            return proton_charge
    except FileNotFoundError:
        return None
    

def get_frame_number(nexus: Optional[str]) -> Optional[int]:
    """
    Extract frame number from NeXus file detector logs.
    
    Retrieves the frame number from the detector acquisition logs within
    the NeXus file. This information is crucial for understanding the
    temporal sequence of data collection in time-resolved measurements.
    
    Args:
        nexus: Path to NeXus file, or None
        
    Returns:
        Frame number from detector logs, or None if unavailable
        
    Example:
        >>> frame_num = get_frame_number("measurement.nxs")
        >>> if frame_num is not None:
        ...     print(f"Last frame number: {frame_num}")
        
    Note:
        Specifically designed for BL10 (VENUS) detector systems.
        Returns None if file doesn't exist or frame number data is unavailable.
        Logs errors for missing files to aid in debugging.
    """
    if nexus is None:
        return None

    if os.path.exists(nexus) is False:
        logging.error(f"NeXus file {nexus} does not exist!")
        return None

    try:
        with h5py.File(nexus, 'r') as hdf5_data:
            frame_number = hdf5_data["entry"]['DASlogs']['BL10:Det:PIXELMAN:ACQ:NUM']['value'][:][-1]
            return frame_number
    except KeyError:
        return None
    