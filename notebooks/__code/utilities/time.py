"""
Time and date utility functions for CT reconstruction pipeline.

This module provides functions for time formatting and conversion used
throughout the reconstruction workflow for timestamping files, logging,
and displaying processing times in human-readable formats.
"""

import numpy as np
import datetime
from typing import Union


def convert_time_s_in_time_hr_mn_s(time_s: Union[int, float]) -> str:
    """
    Convert time in seconds to a formatted hour:minute:second string.
    
    Args:
        time_s: Time duration in seconds
        
    Returns:
        Formatted time string in format "HHhr:MMmn:SSs"
        
    Example:
        >>> convert_time_s_in_time_hr_mn_s(3665)
        '01hr:01mn:05s'
    """
    time_s_only: int = int(np.mod(time_s, 60))

    time_hr_mn: float = np.floor(time_s / 60)
    time_mn: int = int(np.mod(time_hr_mn, 60))
    time_hr: int = int(np.floor(time_hr_mn / 60))
    return f"{time_hr:02d}hr:{time_mn:02d}mn:{time_s_only:02d}s"


def get_current_time_in_special_file_name_format() -> str:
    """
    Format the current date and time for use in filenames.
    
    Creates a timestamp string suitable for use in filenames that sorts
    chronologically and avoids special characters that might cause issues
    in file systems.
    
    Returns:
        Formatted timestamp string in format "MMm_DDd_YYYYy_HHh_MMmn"
        
    Example:
        Returns something like "04m_07d_2022y_08h_06mn" for April 7, 2022 at 8:06 AM
    """
    current_time: str = datetime.datetime.now().strftime("%mm_%dd_%Yy_%Hh_%Mmn")
    return current_time
