import numpy as np
from collections import Counter


def convert_deg_in_rad(array_of_angles_in_deg):
    array_of_angles_in_rad = []
    for angle in array_of_angles_in_deg:
        angle_rad = np.deg2rad(angle)
        array_of_angles_in_rad.append(angle_rad)
    return array_of_angles_in_rad


def calculate_most_dominant_int_value_from_list(list_value):
    round_list = [round(_value) for _value in list_value]
    count = Counter(round_list)
    max_value = 0
    max_number = 0
    for _key, _value in count.items():
        if _value > max_number:
            max_value = _key
            max_number = _value
    return max_value


def angular_distance(a, b, max_coverage=360):
    """Calculate the minimum angular distance on a circle."""
    return min(abs(a - b), max_coverage - abs(a - b))


def farthest_point_sampling(angles, n):
    """
    Selects n angles from the provided list to maximize coverage.
    
    Parameters:
        angles (list or array): Input list of angles (0° to 360°).
        n (int): Number of angles to select.
    
    Returns:
        selected (list): Selected angles maximizing coverage.
    """
    angles = np.array(sorted(angles))
    selected = []

    if angles.max() > 180:
        max_coverage = 360
    else:
        max_coverage = 180

    # Start by selecting the first angle arbitrarily (e.g., first angle)
    selected.append(angles[0])
    unselected = list(angles[1:])

    while len(selected) < n:
        max_min_dist = -1
        next_angle = None
        
        for candidate in unselected:
            # Find distance to nearest selected point
            min_dist = min(angular_distance(candidate, sel, max_coverage=max_coverage) for sel in selected)
            
            # Choose the candidate with largest minimum distance
            if min_dist > max_min_dist:
                max_min_dist = min_dist
                next_angle = candidate
        
        selected.append(next_angle)
        unselected.remove(next_angle)
    
    return sorted(selected)
