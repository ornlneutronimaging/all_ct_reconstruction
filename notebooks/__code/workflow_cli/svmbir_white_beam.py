"""
SVMBIR White Beam Reconstruction Module for CLI-Based CT Workflow.

This module provides command-line interface compatible functions for performing
Sparse-View Model-Based Iterative Reconstruction (SVMBIR) on white beam CT data.
It supports both traditional SVMBIR and JAX-accelerated mbirjax implementations
for high-performance reconstruction.

Key Classes:
    - SvmbirCliHandler: Main handler class for SVMBIR reconstruction operations

Key Features:
    - SVMBIR and mbirjax reconstruction algorithms
    - White beam mode reconstruction
    - Stripe removal integration
    - System matrix management and cleanup
    - Parallel reconstruction with slice merging
    - Comprehensive parameter configuration

Algorithms:
    1. SVMBIR: Traditional sparse-view model-based iterative reconstruction
    2. mbirjax: JAX-accelerated implementation for GPU/TPU acceleration

Dependencies:
    - svmbir: Core SVMBIR reconstruction library
    - mbirjax: JAX-accelerated model-based iterative reconstruction
    - jax: Just-in-time compilation and GPU acceleration
    - numpy: Numerical computing and array operations

Author: CT Reconstruction Pipeline Team
Created: Part of CLI-based CT reconstruction workflow
"""

import numpy as np
import os
import glob
import logging
from typing import List, Dict, Any, Tuple, Optional, Union
import svmbir
import time
import jax.numpy as jnp
import mbirjax as mj
from numpy.typing import NDArray

from __code import WhenToRemoveStripes
from __code.workflow.export import Export
from __code.utilities.logging import setup_logging
from __code.utilities.files import make_or_reset_folder, remove_folder
from __code.config import NUM_THREADS, SVMBIR_LIB_PATH, SVMBIR_LIB_PATH_BACKUP, SVMBIR_LIB_PATH_BACKUP_2
from __code.utilities.json import load_json_string
from __code.utilities.load import load_data_using_multithreading, load_list_of_tif
from __code.utilities.time import get_current_time_in_special_file_name_format
from __code.workflow_cli.merge_reconstructed_slices import merge_reconstructed_slices
from __code.workflow_cli.stripes_removal import StripesRemovalHandler


class SvmbirCliHandler:
    """
    Handler class for SVMBIR reconstruction operations in CLI workflow.
    
    This class provides static methods for performing model-based iterative
    reconstruction using SVMBIR and mbirjax algorithms. It handles system
    matrix management, reconstruction execution, and result processing.
    """

    @staticmethod
    def run_reconstruction_from_pre_data_mode(config_json_file: str, mbirjax: bool = False) -> None:
        """
        Execute SVMBIR reconstruction from preprocessed data configuration.
        
        This method performs SVMBIR reconstruction using configuration data
        loaded from a JSON file. It supports both traditional SVMBIR and
        JAX-accelerated mbirjax implementations.
        
        Args:
            config_json_file: Path to JSON configuration file containing reconstruction parameters
            mbirjax: Whether to use JAX-accelerated implementation (default: False)
            
        Note:
            When using traditional SVMBIR, the system matrix folder is cleared
            to prevent permission errors and ensure clean reconstruction state.
        """

        if not mbirjax:
            logging.info(f"clearing {SVMBIR_LIB_PATH}/sysmatrix folder")
            # removing all the files in the sysmatrix folder the owner owns
            # this is needed to avoid the error "PermissionError: [Errno 13] Permission denied: '/fastdata/sysmatrix/...' when running svmbir
            if os.path.exists(os.path.join(SVMBIR_LIB_PATH, 'sysmatrix')):
                list_files = glob.glob(os.path.join(SVMBIR_LIB_PATH, 'sysmatrix', '*'))
                for _file in list_files:
                    if os.path.isfile(_file):
                        try:
                            os.remove(_file)
                        except PermissionError as e:
                            logging.error(f"PermissionError: {e} for file {_file}")
                    else:
                        logging.info(f"skipping {_file} as it is not a file") 

        method_used = "mbirjax" if mbirjax else "svmbir"

        config = load_json_string(config_json_file)
        logging.info(f"config = {config}")

        input_data_folder = config["projections_pre_processing_folder"]
        base_output_folder = config['output_folder']

        list_tiff = glob.glob(os.path.join(input_data_folder, '*.tiff'))
        list_tiff.sort()
        print(f"loading {len(list_tiff)} images ... ", end="")
        logging.info(f"loading {len(list_tiff)} images ... ")
        corrected_array_log = load_list_of_tif(list_tiff, dtype=np.float32)
        print(f"done!")
        logging.info(f"loading {len(list_tiff)} images ... done")
      
        logging.info(f"when to remove stripes: {config['when_to_remove_stripes']}")

        # this is where we will apply the strip removal algorithms if requested
        if config['when_to_remove_stripes'] == WhenToRemoveStripes.out_notebook:
            print("Applying strip removal algorithms ...", end="")
            logging.info("Applying strip removal algorithms ...")
            corrected_array_log = StripesRemovalHandler.remove_stripes(corrected_array_log,
                                                                       config=config,
                                                                      )
            logging.info("Strip removal done!")
            print(" done!")
 
        list_of_angles_rad = np.array(config['list_of_angles'])
        width = np.shape(corrected_array_log)[2]

        center_of_rotation = config['center_of_rotation']
        center_offset = -(width / 2 - center_of_rotation)  # it's Shimin's formula

        sharpness = config['svmbir_config']['sharpness']
        snr_db = config['svmbir_config']['snr_db']
        positivity = config['svmbir_config']['positivity']
        max_iterations = config['svmbir_config']['max_iterations']
        verbose = config['svmbir_config']['verbose']
        
        # check if SVMBIR_LIB_PATH is accessible (write permission), otherwise use the backup
        if os.access(SVMBIR_LIB_PATH, os.W_OK):
            svmbir_lib_path = SVMBIR_LIB_PATH
        elif os.access(SVMBIR_LIB_PATH_BACKUP, os.W_OK):
            svmbir_lib_path = SVMBIR_LIB_PATH_BACKUP
        elif os.access(SVMBIR_LIB_PATH_BACKUP_2, os.W_OK):
            svmbir_lib_path = SVMBIR_LIB_PATH_BACKUP_2
        else:
            raise PermissionError(f"None of the SVMBIR library paths are writable: {SVMBIR_LIB_PATH}, {SVMBIR_LIB_PATH_BACKUP}, {SVMBIR_LIB_PATH_BACKUP_2}")
        
        max_resolutions = config['svmbir_config']['max_resolutions']
        list_of_slices_to_reconstruct = config['list_of_slices_to_reconstruct']

        if (len(list_of_slices_to_reconstruct) == 1) and (list_of_slices_to_reconstruct[0][0] == 0) and (list_of_slices_to_reconstruct[0][1] == -1):
            list_of_slices_to_reconstruct = []
            logging.info(f"reconstructing all slices at once")

        top_slice = config['crop_region']['top']

        logging.info(f"Shape of corrected_array_log:")
        logging.info(f"{np.shape(corrected_array_log) = }")

        logging.info(f"{list_of_angles_rad = }")
        logging.info(f"{center_offset = }")
        logging.info(f"{sharpness = }")
        logging.info(f"{snr_db = }")
        logging.info(f"{positivity = }")
        logging.info(f"{max_iterations = }")
        logging.info(f"{max_resolutions = }")
        logging.info(f"{verbose = }")
        logging.info(f"{svmbir_lib_path = }")
        logging.info(f"{input_data_folder = }")
        logging.info(f"{base_output_folder = }")
        logging.info(f"{list_of_slices_to_reconstruct = }")
        
        if mbirjax:
            output_data_folder = os.path.join(base_output_folder, f"mbirjax_reconstructed_data_{get_current_time_in_special_file_name_format()}")
        else:
            output_data_folder = os.path.join(base_output_folder, f"svmbir_reconstructed_data_{get_current_time_in_special_file_name_format()}")
        logging.info(f"{output_data_folder = }")

        # make_or_reset_folder(output_data_folder)
        make_or_reset_folder(output_data_folder)

        list_of_output_folders = []
        start_time = time.time()
        if list_of_slices_to_reconstruct:

            for [index, [top_slice_index, bottom_slice_index]] in enumerate(list_of_slices_to_reconstruct):
                print(f"working with set of slices #{index}: from {top_slice_index} to {bottom_slice_index-1}. ", end="") 
                logging.info(f"working with set of slices #{index}: from {top_slice_index} to {bottom_slice_index-1}")
                
                if mbirjax:
                    print(f"launching mbirjax #{index} ... ", end="")
                    logging.info(f"launching mbirjax #{index} ...")
                else:
                    print(f"launching svmbir #{index} ... ", end="")
                    logging.info(f"launching svmbir #{index} ...")
        
                _sino = corrected_array_log[:, top_slice_index:bottom_slice_index, :]
                logging.info(f"\t{_sino.shape = }")
                    
                if mbirjax:
                    sinogram_shape = _sino.shape

                    ct_model_for_recon = mj.ParallelBeamModel(sinogram_shape,
                                                             list_of_angles_rad)
                    ct_model_for_recon.scale_recon_shape(row_scale=1.1, col_scale=1.1) # overide the region removed to avoid flashes around the region of reconstruction
                    
                    ct_model_for_recon.set_params(sharpness=sharpness,
                                                  verbose=verbose,
                                                  delta_det_channel=center_offset,
                                                  snr_db=snr_db,
                    )
                    reconstruction_array, recond_dict = ct_model_for_recon.recon(_sino,
                                                                    print_logs=False,
                                                                    weights=None,
                                                                    )
                    logging.info(f"\t{recond_dict = }")
                    logging.info(f"\treconstruction_array shape: {np.shape(reconstruction_array)} ")
                    reconstruction_array = np.swapaxes(reconstruction_array, 0, 2)  # swap axes to match SVMBIR output
                    del recond_dict  

                else:

                    reconstruction_array = svmbir.recon(sino=_sino,
                                                        angles=list_of_angles_rad,
                                                        num_rows = _sino.shape[2],  # height,
                                                        num_cols = _sino.shape[2],  # width,
                                                        center_offset = center_offset,
                                                        max_resolutions = max_resolutions,
                                                        sharpness = sharpness,
                                                        snr_db = snr_db,
                                                        positivity = positivity,
                                                        max_iterations = max_iterations,
                                                        num_threads = NUM_THREADS,
                                                        verbose = verbose,
                                                        roi_radius=3000,
                                                        svmbir_lib_path = svmbir_lib_path,
                                                        )
                    logging.info(f"\treconstruction_array shape: {np.shape(reconstruction_array)} ")
                
                print(f"done with #{index}!")
                logging.info(f"done with #{index}!")
                _index = f"{index:03d}"
                print(f"exporting reconstructed slices set #{_index} ... ", end="")
                logging.info(f"\t{np.shape(reconstruction_array) = }")
                logging.info(f"exporting reconstructed data set #{_index} ...")

                _output_data_folder = os.path.abspath(os.path.join(output_data_folder, f"set_{_index}"))
                logging.info(f"making or resetting folder {_output_data_folder}")
                list_of_output_folders.append(_output_data_folder)
                make_or_reset_folder(_output_data_folder)
                o_export = Export(image_3d=reconstruction_array,
                                  output_folder=_output_data_folder)
                o_export.run()
                print(f"done!")

            merge_reconstructed_slices(output_data_folder=output_data_folder, 
                                       top_slice=top_slice,
                                       list_of_output_folders=list_of_output_folders,
                                       list_of_slices_to_reconstruct=list_of_slices_to_reconstruct)

        else:

            if mbirjax:
                sinogram_shape = corrected_array_log.shape

                ct_model_for_recon = mj.ParallelBeamModel(sinogram_shape,
                                                            list_of_angles_rad)
                ct_model_for_recon.set_params(sharpness=sharpness,
                                                verbose=verbose,
                                                det_channel_offset=center_offset,
                                                snr_db=snr_db,
                )
                reconstruction_array, recond_dict = ct_model_for_recon.recon(corrected_array_log,
                                                                print_logs=False,
                                                                weights=None,
                                                                )
               
                logging.info(f"{recond_dict = }")
                logging.info(f"reconstruction_array shape: {np.shape(reconstruction_array)} ")
                del recond_dict

            else:

                print(f"launching svmbir with all slices ... ", end="")
                logging.info(f"launching svmbir with all slices ...")

                logging.info(f"{corrected_array_log.shape = }")
                logging.info(f"{list_of_angles_rad.shape = }")
                





                reconstruction_array = svmbir.recon(sino=corrected_array_log,
                                                    angles=list_of_angles_rad,
                                                    num_rows = corrected_array_log.shape[2],  # height,
                                                    num_cols = corrected_array_log.shape[2],  # width,
                                                    center_offset = center_offset,
                                                    max_resolutions = max_resolutions,
                                                    sharpness = sharpness,
                                                    snr_db = snr_db,
                                                    positivity = positivity,
                                                    max_iterations = max_iterations,
                                                    num_threads = NUM_THREADS,
                                                    verbose = verbose,
                                                    # roi_radius=3000,
                                                    svmbir_lib_path = svmbir_lib_path,
                                                    )
                logging.info(f"reconstruction_array shape: {np.shape(reconstruction_array)} ")

            print(f"done! ")
            end_time = time.time()
            elapsed_time = end_time - start_time
            logging.info(f">>>> Reconstruction took {elapsed_time:.2f} seconds using {method_used} <<<<")

            logging.info(f"{np.shape(reconstruction_array) = }")
            logging.info(f"exporting reconstructed data in {output_data_folder} ...")
            print(f"exporting reconstructed data to {output_data_folder} ... ", end="")
            o_export = Export(image_3d=reconstruction_array,
                            output_folder=output_data_folder)
            o_export.run()
            print(f"done!")

        logging.info(f"exporting reconstructed data ... done!")
        logging.info(f"")

    @staticmethod
    def run_reconstruction_from_pre_data_mode_for_ai_evaluation(config_json_file):

        logging.info(f"run_reconstruction_from_pre_data_mode_for_ai_evaluation")

        config = load_json_string(config_json_file)
        logging.info(f"config = {config}")

        input_data_folder = config["projections_pre_processing_folder"]
        base_output_folder = config['output_folder']

        list_tiff = glob.glob(os.path.join(input_data_folder, '*.tiff'))
        list_tiff.sort()
        print(f"loading {len(list_tiff)} images ... ", end="")
        logging.info(f"loading {len(list_tiff)} images ... ")
        # corrected_array_log = load_data_using_multithreading(list_tiff)
        corrected_array_log = load_list_of_tif(list_tiff, dtype=np.float32)
        print(f"done!")
        logging.info(f"loading {len(list_tiff)} images ... done")

        # this is where we will apply the strip removal algorithms if requested
        if config['when_to_remove_stripes'] == WhenToRemoveStripes.out_notebook:
            print("Applying strip removal algorithms ...", end="")
            logging.info("Applying strip removal algorithms ...")
            corrected_array_log = StripesRemovalHandler.remove_stripes(corrected_array_log,
                                                                       config=config,
                                                                      )
            logging.info("Strip removal done!")
            print(" done!")

        list_of_angles_rad = np.array(config['list_of_angles'])
        width = np.shape(corrected_array_log)[2]
        
        center_of_rotation = config['center_of_rotation']
        center_offset = -(width / 2 - center_of_rotation)  # it's Shimin's formula

        sharpness = config['svmbir_config']['sharpness']
        snr_db = config['svmbir_config']['snr_db']
        positivity = config['svmbir_config']['positivity']
        max_iterations = config['svmbir_config']['max_iterations']
        verbose = config['svmbir_config']['verbose']
        svmbir_lib_path = SVMBIR_LIB_PATH
        max_resolutions = config['svmbir_config']['max_resolutions']
        list_of_slices_to_reconstruct = config['list_of_slices_to_reconstruct']
        top_slice = config['crop_region']['top']

        logging.info(f"Before switching y and x coordinates:")
        logging.info(f"{np.shape(corrected_array_log) = }")   # angles, y, x

        logging.info(f"{list_of_angles_rad = }")
        # logging.info(f"{height = }")
        logging.info(f"{width = }")
        logging.info(f"{center_offset = }")
        logging.info(f"{sharpness = }")
        logging.info(f"{snr_db = }")
        logging.info(f"{positivity = }")
        logging.info(f"{max_iterations = }")
        logging.info(f"{max_resolutions = }")
        logging.info(f"{verbose = }")
        logging.info(f"{svmbir_lib_path = }")
        logging.info(f"{input_data_folder = }")
        logging.info(f"{base_output_folder = }")
        logging.info(f"{list_of_slices_to_reconstruct = }")
        
        output_data_folder = os.path.join(base_output_folder, f"svmbir_reconstructed_data_{get_current_time_in_special_file_name_format()}")
        logging.info(f"{output_data_folder = }")

        # make_or_reset_folder(output_data_folder)
        make_or_reset_folder(output_data_folder)

        list_of_output_folders = []

          # start with 15 projections, then 30, then 45 .... and so on
        number_of_projections = np.arange(15, len(list_of_angles_rad), 15)
        selected_indices = np.array([], dtype=int)

        starting_number_of_projections = 360

        from_slice = list_of_slices_to_reconstruct[0][0]
        to_slice = list_of_slices_to_reconstruct[0][1]

        for _iter, _nbr_projections in enumerate(number_of_projections):

            logging.info(f"iteration #{_iter} with {_nbr_projections} projections")
            new_random_indices = np.random.choice(np.setdiff1d(np.arange(len(list_of_angles_rad)), selected_indices), 15, replace=False)
            selected_indices = np.concatenate((selected_indices, new_random_indices))
            selected_indices.sort()

            if _nbr_projections < starting_number_of_projections:
                logging.info(f"we didn't reach the starting number of projections yet")
                logging.info(f"\t{_nbr_projections =}")
                logging.info
                continue

            logging.info(f"\t{selected_indices}")
            logging.info(f"\tangles: {list_of_angles_rad[selected_indices]}")

            _subset_list_of_angles_rad = list_of_angles_rad[selected_indices]
            _subset_corrected_array_log = corrected_array_log[selected_indices, from_slice: to_slice, :]

            logging.info(f"\t{np.shape(_subset_list_of_angles_rad) = }")
            logging.info(f"\t{np.shape(_subset_corrected_array_log) = }")
            logging.info(f"\tlaunching reconstruction of iteration {_iter} ... ")

            height = np.shape(_subset_corrected_array_log)[1]

            reconstruction_array = svmbir.recon(_subset_corrected_array_log,
                                                angles=_subset_list_of_angles_rad,
                                                num_rows = np.shape(_subset_corrected_array_log)[2],
                                                num_cols = np.shape(_subset_corrected_array_log)[2],
                                                center_offset = center_offset,
                                                max_resolutions = max_resolutions,
                                                sharpness = sharpness,
                                                snr_db = snr_db,
                                                positivity = positivity,
                                                max_iterations = max_iterations,
                                                num_threads = NUM_THREADS,
                                                verbose = verbose,
                                                svmbir_lib_path = svmbir_lib_path,
                                                roi_radius=3000,
                                                delta_pixel=2
                                                )

            logging.info(f"\treconstruction of iteration {_iter} done!")

            print(f"exporting reconstructed slices using {_nbr_projections} projections ... ", end="")
            logging.info(f"{np.shape(reconstruction_array) = }")
            logging.info(f"exporting reconstructed data of iteration {_iter} ...")

            _output_data_folder = os.path.join(output_data_folder, f"set_with_{_nbr_projections}_projections")
            logging.info(f"making or resetting folder {_output_data_folder}")
            list_of_output_folders.append(_output_data_folder)
            make_or_reset_folder(_output_data_folder)

            o_export = Export(image_3d=reconstruction_array,
                                output_folder=_output_data_folder)
            o_export.run()

            del reconstruction_array
