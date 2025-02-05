import argparse
import logging

from __code.utilities.logging import setup_logging
from __code.utilities.configuration_file import loading_config_file_into_model

from __code.workflow_cli.load import load_data
from __code.workflow_cli.combine import combine_tof_data
from __code.workflow_cli.clean import clean_images
from __code.workflow_cli.normalization import normalize
from __code.workflow_cli.chips_correction import correct_data
from __code.workflow_cli.stripes_removal import stripes_removal
from __code.workflow_cli.center_of_rotation_and_tilt import center_of_rotation_and_tilt
from __code.workflow_cli.svmbir_reconstruction import svmbir_reconstruction
from __code.workflow_cli.export_slices import export_slices

setup_logging("svmbir_reconstruction_cli")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run the full svmbir workflow using the provided config file")
    parser.add_argument('config_file', type=str, nargs=1, help="JSON config file created by notebook")
    args = parser.parse_args()

    print(f"check log file at /SNS/VENUS/shared/log/svmbir_reconstruction_cli.log")

    config_file_name = args.config_file[0]
    logging.info(f"{config_file_name}")
    config_model = loading_config_file_into_model(config_file_path=config_file_name)
    logging.info(f"loading config file name: {config_file_name}")

    # loading only the runs and ob of interest
    # {'sample': 4d_data_array, 'ob': 4d_data_array}  with 4d_data_array being [projections angles, tof, y, x]
    master_3d_data_array = load_data(config_model)

    # combining the data in tof
    # {'sample': 3d_data_array', 'ob': 3d_data_array}   with 3d_data_array being [projections angles, y, x]
    master_3d_data_array = combine_tof_data(config_model, master_3d_data_array)

    # clean images (histogram, threshold)
    # {'sample': 3d_data_array', 'ob': 3d_data_array}   with 3d_data_array being [projections angles, y, x]
    master_3d_data_array = clean_images(config_model, master_3d_data_array)

    # normalization
    # 3d_data_array with [projections angles, y, x]
    master_3d_data_array = normalize(config_model, master_3d_data_array)

    # chips correction
    # 3d_data_array with [projections angles, y, x]
    master_3d_data_array = correct_data(master_3d_data_array)

    # stripes removal
    # 3d_data_array with [projections angles, y, x]
    master_3d_data_array = stripes_removal(config_model, master_3d_data_array)

    # center of rotation and tilt
    # 3d_data_array with [projections angles, y, x]
    master_3d_data_array = center_of_rotation_and_tilt(config_model, master_3d_data_array)
    
    # reconstruction
    # 3d_data_array with [z, y, x]
    master_3d_data_array = svmbir_reconstruction(config_model, master_3d_data_array)

    # export
    export_slices(config_model, master_3d_data_array)

    print("svmbir reconstruction is done!")
    