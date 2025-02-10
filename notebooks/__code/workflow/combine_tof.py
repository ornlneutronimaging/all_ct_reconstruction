import logging
import numpy as np
from tqdm import tqdm

from __code import OperatingMode
from __code.parent import Parent
from __code import DataType, Run
from __code.utilities.load import load_list_of_tif, load_data_using_multithreading
from __code.utilities.files import retrieve_list_of_tif


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

        logging.info(f"loading data ...")

        # for sample
        logging.info(f"\tworking with sample")
        list_angles_deg_vs_runs_dict = self.parent.list_angles_deg_vs_runs_dict
        list_angles = list(list_angles_deg_vs_runs_dict.keys())
        list_angles.sort()

        list_of_runs = self.parent.list_of_runs

        list_of_angles_of_runs_to_keep = []
        master_3d_data_array = {DataType.sample: None,
                                DataType.ob: None,
                                DataType.dc: None}

        list_sample_data = []

        for _angle in tqdm(list_angles):
            _runs = list_angles_deg_vs_runs_dict[_angle]
            logging.info(f"Working with angle {_angle} degrees")
            logging.info(f"\t{_runs}")

            use_it = list_of_runs[DataType.sample][_runs][Run.use_it]
            if use_it:
                logging.info(f"\twe keep that runs!")
                list_of_angles_of_runs_to_keep.append(_angle)
                logging.info(f"\tloading run {_runs} ...")
                _data = self.load_data_for_a_run(run=_runs)
                logging.info(f"\t{_data.shape}")
                # # combine all tof
                # _data = np.sum(_data, axis=0)
                list_sample_data.append(_data)
            else:
                logging.info(f"\twe reject that runs!")

        master_3d_data_array[DataType.sample] = np.array(list_sample_data)

        # for ob
        logging.info(f"\tworking with ob")

        list_ob_data = []

        for _run in tqdm(list_of_runs[DataType.ob]):
            use_it = list_of_runs[DataType.ob][_run][Run.use_it]
            if use_it:
                logging.info(f"\twe keep that runs!")
                logging.info(f"\tloading run {_run} ...")
                _data = self.load_data_for_a_run(run=_run, data_type=DataType.ob)
                logging.info(f"\t{_data.shape}")
                list_ob_data.append(_data)
            else:
                logging.info(f"\twe reject that runs!")

        master_3d_data_array[DataType.ob] = np.array(list_ob_data)

        self.parent.master_3d_data_array = master_3d_data_array
        self.parent.final_list_of_angles = list_of_angles_of_runs_to_keep

    def load_data_for_a_run(self, run=None, data_type=DataType.sample):

        full_path_to_run = self.parent.list_of_runs[data_type][run][Run.full_path]

        # get list of tiff
        list_tif = retrieve_list_of_tif(full_path_to_run)

        # load data
        # data = load_list_of_tif(list_tif)
        data = load_data_using_multithreading(list_tif, combine_tof=True)

        return data


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
