import logging
import numpy as np

from __code import OperatingMode
from __code.parent import Parent
from __code import DataType, Run


class CombineTof(Parent):

    def run(self):      
        self.update_list_of_runs_status()
        self.load_data()

        # combine the data in TOF



    def update_list_of_runs_status(self):

        # update list of runs to reject
        list_of_runs = self.parent.list_of_runs
        logging.info(f"list_of_runs = {list_of_runs}")

        list_ob_to_reject = self.parent.list_of_ob_runs_to_reject_ui.value
        for _run in list_ob_to_reject:
            list_of_runs[DataType.ob][_run][Run.use_it] = False

        list_sample_to_reject = self.parent.list_of_sample_runs_to_reject_ui.value
        for _run in list_sample_to_reject:
            list_of_runs[DataType.sample][_run][Run.use_it] = False

        self.parent.list_of_runs = list_of_runs

    def load_data(self):
        list_angles_deg_vs_runs_dict = self.parent.list_angles_deg_vs_runs_dict
        list_angles = list(list_angles_deg_vs_runs_dict.keys())
        list_angles.sort()


        ## work in progress
        for _angle in list_angles:
            _runs = list_angles_deg_vs_runs_dict[_angle]
            logging.info(f"Working with angle {_angle} degrees")
            logging.info(f"\t{_runs}")

            # load data
            for _run in _runs:
                logging.info(f"Working with run {_run}")
                _data = self.load_data_for_a_run(run=_run)
                logging.info(f"\t{_data.shape}")











def combine_all_tof(data):
    pass



def combine_tof_data_range(config_model, master_data):
    
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
