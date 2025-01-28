import logging
import os
import numpy as np

def setup_logging(basename_of_log_file=""):

    log_file_name = f"/SNS/VENUS/shared/log/{basename_of_log_file}.log"
    logging.basicConfig(filename=log_file_name,
                        filemode='w',
                        format='[%(levelname)s] - %(asctime)s - %(message)s',
                        level=logging.INFO)
    logging.info(f"*** Starting a new script {basename_of_log_file} ***")

    print(f"logging file: {log_file_name}")
    

def logging_3d_array_infos(message="", array=None):
    logging.info(f"{message}")
    logging.info(f"{np.min(array) = }")
    logging.info(f"{np.max(array) = }")
    logging.info(f"Number of nan: {np.count_nonzero(np.isnan(array))}")
    logging.info(f"Number of inf: {np.count_nonzero(np.isinf(array))}")
                 