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
