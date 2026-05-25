import argparse
import logging
import os
import shutil
import h5py
import json

from __code.utilities.logging import setup_logging
from __code.workflow_cli.fbp_white_beam import  FbpCliHandler
from __code.workflow_cli.svmbir_white_beam import SvmbirCliHandler
from __code.utilities.configuration_file import ReconstructionAlgorithm

file_name, ext = os.path.splitext(os.path.basename(__file__))
full_log_file_name = setup_logging(file_name)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run the specify reconstruction algorithm specified in the config file in white beam mode")
    parser.add_argument('hdf5_file', type=str, nargs=1, help="HDF5 file containing the data to reconstruct")
    args = parser.parse_args()

    hdf5_file = args.hdf5_file[0]
    with h5py.File(hdf5_file, "r") as f:
        config_json = f["metadata/config"][()] if "metadata/config" in f else None
    configuration = json.loads(config_json) if config_json is not None else None

    list_reconstruction_algorithm = list(configuration['reconstruction_algorithm'])

    if ReconstructionAlgorithm.svmbir in list_reconstruction_algorithm:
        logging.info(f"about to call SvmbirCliHandler.run_reconstruction_from_pre_data_mode:")
        logging.info(f"\t{hdf5_file = }")
        SvmbirCliHandler.run_reconstruction_from_pre_data_mode(hdf5_file=hdf5_file)
        #SvmbirCliHandler.run_reconstruction_from_pre_data_mode_for_ai_evaluation(config_json_file=config_json_file)
        list_reconstruction_algorithm.remove(ReconstructionAlgorithm.svmbir)
        logging.info(f"Svmbir reconstruction done!")
    
    if ReconstructionAlgorithm.mbirjax in list_reconstruction_algorithm:
        logging.info(f"about to call SvmbirCliHandler.run_reconstruction_from_pre_data_mode using Mbirjax mode:")
        logging.info(f"\t{hdf5_file = }")
        SvmbirCliHandler.run_reconstruction_from_pre_data_mode(hdf5_file=hdf5_file, mbirjax=True)
        list_reconstruction_algorithm.remove(ReconstructionAlgorithm.mbirjax)
        logging.info(f"Mbirjax reconstruction done!")

    if len(list_reconstruction_algorithm) > 0:
        logging.info(f"about to call FbpCliHandler.run_reconstruction_from_pre_data_mode:")
        logging.info(f"\t{hdf5_file = }")
        FbpCliHandler.run_reconstruction_from_pre_data_mode(hdf5_file=hdf5_file)
        logging.info(f"Fbp reconstruction done!")

    logging.info(f"All reconstructions are done!")

    logging.info(f"Copying the log file to the output folder")
    output_folder = configuration['output_folder']
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    shutil.copy(full_log_file_name, output_folder)
    print(f"Reconstruction is done!")
    print(f"Log file copied to {output_folder}")
    