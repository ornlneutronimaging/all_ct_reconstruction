import os
import logging
from tqdm import tqdm

from __code import DataType, OperatingMode
from __code.utilities.files import retrieve_list_of_tif
from __code.utilities.load import load_data_using_multithreading


def load_data(config_model):

    logging.info(f"loading the data:")
    print(f"Loading the data ... ")
    operating_mode = config_model.operating_mode
    combine = operating_mode == OperatingMode.white_beam

    # load sample and ob
    list_sample_runs = config_model.list_of_sample_runs
    sample_base_folder_name = config_model.top_folder.sample
    logging.info(f"{list_sample_runs = }")
    logging.info(f"{sample_base_folder_name = }")

    list_ob_runs = config_model.list_of_ob_runs
    ob_base_folder_name = config_model.top_folder.ob
    logging.info(f"{list_ob_runs = }")
    logging.info(f"{ob_base_folder_name = }")

    list_of_runs = {DataType.sample: [os.path.join(sample_base_folder_name, _file) for _file in list_sample_runs],
                    DataType.ob: [os.path.join(ob_base_folder_name, _file) for _file in list_ob_runs]}

    logging.info(f"{list_of_runs =}")

    master_3d_data_array = {DataType.sample: [],
                            DataType.ob: []}
    for _data_type in list_of_runs.keys():
        logging.info(f"\tloading {_data_type}:")
        for _full_path_run in tqdm(list_of_runs[_data_type]):
            logging.info(f"\t{os.path.basename(_full_path_run)}")
            list_tif = retrieve_list_of_tif(_full_path_run)
            master_3d_data_array[_data_type].append(load_data_using_multithreading(list_tif,
                                                               combine_tof=combine))

    print(f"done loading the data!")
    return master_3d_data_array
