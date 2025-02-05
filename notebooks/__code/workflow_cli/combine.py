import logging
import numpy as np

from __code import OperatingMode


def combine_tof_data(config_model, master_data):
    
    operating_mode = config_model.operating_mode
    if operating_mode == OperatingMode.white_beam:
        logging.info(f"white mode, all TOF data have already been combined!")
        return master_data
       
    # tof mode
    print(f"combining data in TOF ...", end="")
    [left_tof_index, right_tof_index] = config_model.range_of_tof_to_combine[0]
    logging.info(f"combining TOF from index {left_tof_index} to index {right_tof_index}")
    for _data_type in master_data.keys():
        _new_master_data = []
        for _data in master_data[_data_type]:
            _new_master_data.append(np.mean(_data[left_tof_index: right_tof_index+1, :, :], axis=0))
        master_data[_data_type] = _new_master_data
    print(f"done!")

    return master_data
