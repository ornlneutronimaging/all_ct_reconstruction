"""
Mathematical utilities for CT reconstruction pipeline.

This module provides mathematical functions commonly used in CT reconstruction
including angle conversions, statistical analysis, and geometric calculations.
Functions support both degree and radian angle systems and provide utilities
for optimal angle selection in tomographic reconstructions.
"""

import numpy as np
from collections import Counter
from typing import List, Union
from numpy.typing import NDArray


def convert_deg_in_rad(array_of_angles_in_deg: List[Union[int, float]]) -> List[float]:
    """
    Convert a list of angles from degrees to radians.
    
    Args:
        array_of_angles_in_deg: List of angles in degrees
        
    Returns:
        List of angles converted to radians
        
    Example:
        >>> convert_deg_in_rad([0, 90, 180, 270])
        [0.0, 1.5707963267948966, 3.141592653589793, 4.71238898038469]
    """
    array_of_angles_in_rad: List[float] = []
    for angle in array_of_angles_in_deg:
        angle_rad: float = np.deg2rad(angle)
        array_of_angles_in_rad.append(angle_rad)
    return array_of_angles_in_rad


def calculate_most_dominant_int_value_from_list(list_value: List[Union[int, float]]) -> int:
    """
    Find the most frequently occurring integer value in a list.
    
    Rounds all values to integers first, then finds the mode.
    Useful for determining the most common discrete value from
    noisy measurements.
    
    Args:
        list_value: List of numeric values
        
    Returns:
        The most frequently occurring integer value
        
    Example:
        >>> calculate_most_dominant_int_value_from_list([1.1, 1.9, 2.1, 1.8, 2.2])
        2
    """
    round_list: List[int] = [round(_value) for _value in list_value]
    count: Counter = Counter(round_list)
    max_value: int = 0
    max_number: int = 0
    for _key, _value in count.items():
        if _value > max_number:
            max_value = _key
            max_number = _value
    return max_value


def calculate_most_dominant_float_value_from_list(list_value: List[Union[int, float]]) -> float:
    """
    Find the most frequently occurring float value in a list.
    
    Uses exact matching to find the mode. Useful for determining
    the most common value from repeated measurements.
    
    Args:
        list_value: List of numeric values
        
    Returns:
        The most frequently occurring float value
        
    Example:
        >>> calculate_most_dominant_float_value_from_list([1.0, 1.0, 2.0, 1.0, 2.0])
        1.0
    """
    count: Counter = Counter(list_value)
    max_value: float = 0.0
    max_number: int = 0
    for _key, _value in count.items():
        if _value > max_number:
            max_value = _key
            max_number = _value
    return max_value


def angular_distance(a: Union[int, float], b: Union[int, float], max_coverage: Union[int, float] = 360) -> Union[int, float]:
    """
    Calculate the minimum angular distance between two angles on a circle.
    
    This function accounts for the circular nature of angles, choosing
    the shorter path around the circle.
    
    Args:
        a: First angle
        b: Second angle  
        max_coverage: Maximum angle coverage (360 for full circle, 180 for half)
        
    Returns:
        Minimum angular distance between the two angles
        
    Example:
        >>> angular_distance(10, 350, 360)
        20  # Goes the short way around the circle
    """
    return min(abs(a - b), max_coverage - abs(a - b))


def farthest_point_sampling(angles: Union[List[Union[int, float]], NDArray[np.floating]], n: int) -> List[Union[int, float]]:
    """
    Select n angles from the provided list to maximize angular coverage.
    
    Uses a farthest-point sampling strategy to select angles that provide
    optimal coverage for tomographic reconstruction. This is particularly
    useful when reducing the number of projection angles while maintaining
    reconstruction quality.
    
    Args:
        angles: Input list of angles (0° to 360° or 0° to 180°)
        n: Number of angles to select
        
    Returns:
        Selected angles that maximize coverage, sorted in ascending order
        
    Algorithm:
        1. Start with the first angle
        2. Iteratively select the angle that is farthest from all previously selected angles
        3. Continue until n angles are selected
        
    Note:
        Automatically detects whether to use 180° or 360° coverage based on
        the maximum angle in the input list.
    """
    angles_array: NDArray[np.floating] = np.array(sorted(angles))
    selected: List[Union[int, float]] = []

    max_coverage: Union[int, float]
    if angles_array.max() > 180:
        max_coverage = 360
    else:
        max_coverage = 180

    # Start by selecting the first angle arbitrarily (e.g., first angle)
    selected.append(angles_array[0])
    unselected: List[Union[int, float]] = list(angles_array[1:])

    while len(selected) < n:
        max_min_dist: Union[int, float] = -1
        next_angle: Union[int, float, None] = None
        
        for candidate in unselected:
            # Find distance to nearest selected point
            min_dist: Union[int, float] = min(angular_distance(candidate, sel, max_coverage=max_coverage) for sel in selected)
            
            # Choose the candidate with largest minimum distance
            if min_dist > max_min_dist:
                max_min_dist = min_dist
                next_angle = candidate
        
        selected.append(next_angle)
        unselected.remove(next_angle)
    
    return sorted(selected)
