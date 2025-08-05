"""
Filtered Back Projection (FBP) Reconstruction Module for CLI-Based CT Workflow.

This module provides command-line interface compatible functions for performing
FBP reconstruction on white beam CT data. It supports multiple reconstruction
algorithms and handles the complete workflow from data preparation to final
slice export.

Key Classes:
    - FbpCliHandler: Main handler class for FBP reconstruction operations

Key Features:
    - Multiple FBP algorithms (TomoPy, Algotom, SVMBIR)
    - White beam mode reconstruction
    - Stripe removal integration
    - Multi-threading support for performance
    - Automatic slice merging and export
    - Comprehensive logging and progress tracking

Supported Algorithms:
    1. TomoPy FBP: Standard filtered back projection
    2. Algotom FBP: Advanced FBP with enhanced filters
    3. SVMBIR: Model-based iterative reconstruction

Dependencies:
    - tomopy: Core tomographic reconstruction library
    - algotom: Advanced tomographic algorithms
    - svmbir: Sparse-view model-based iterative reconstruction
    - numpy: Numerical computing and array operations

Author: CT Reconstruction Pipeline Team
Created: Part of CLI-based CT reconstruction workflow
"""

import numpy as np
import os
import glob
import logging
from typing import List, Dict, Any, Tuple, Optional
import svmbir
import tomopy
from tomopy.prep import stripe
from numpy.typing import NDArray

# from imars3d.backend.reconstruction import recon
from tomopy import recon as tomopy_recon
import algotom.rec.reconstruction as rec

from __code import WhenToRemoveStripes
from __code.workflow.export import Export
from __code.config import NUM_THREADS
from __code.utilities.logging import setup_logging
from __code.utilities.files import make_or_reset_folder, make_folder
from __code.config import SVMBIR_LIB_PATH
from __code.utilities.json import load_json_string
from __code.utilities.load import load_data_using_multithreading, load_list_of_tif
from __code.utilities.time import get_current_time_in_special_file_name_format
from __code.workflow_cli.merge_reconstructed_slices import merge_reconstructed_slices
from __code.utilities.configuration_file import ReconstructionAlgorithm
from __code.workflow.remove_strips import RemoveStrips
from __code.workflow_cli.stripes_removal import StripesRemovalHandler


class FbpCliHandler:
    """
    Handler class for FBP reconstruction operations in CLI workflow.
    
    This class provides static methods for performing filtered back projection
    reconstruction using various algorithms. It handles data preparation,
    reconstruction execution, and result processing for white beam CT data.
    """

    @staticmethod
    def _run_reconstruction(projections: NDArray[np.floating], 
                           center_of_rotation: float, 
                           list_of_angles_rad: NDArray[np.floating], 
                           algorithm: ReconstructionAlgorithm, 
                           max_workers: int) -> NDArray[np.floating]:
        """
        Execute FBP reconstruction using the specified algorithm.
        
        This method performs the actual reconstruction computation using one
        of the supported FBP algorithms. It handles algorithm-specific parameter
        configuration and execution.
        
        Args:
            projections: 3D projection data array (angles x height x width)
            center_of_rotation: Center of rotation value in pixels
            list_of_angles_rad: Array of projection angles in radians
            algorithm: Reconstruction algorithm to use
            max_workers: Number of threads for parallel processing
            
        Returns:
            3D reconstructed volume array (slices x height x width)
            
        Supported Algorithms:
            - algotom_fbp: Algotom filtered back projection
            - tomopy_fbp: TomoPy filtered back projection  
            - svmbir: Model-based iterative reconstruction
        """
        
        logging.info(f"\t -> {np.shape(projections) = }")
        logging.info(f"\t -> {center_of_rotation = }")
        logging.info(f"\t -> {list_of_angles_rad = }")
        logging.info(f"\t -> {len(list_of_angles_rad) = }")
        logging.info(f"\t -> {algorithm = }")
        logging.info(f"\t -> launching reconstruction using {algorithm} ...")

        if algorithm == ReconstructionAlgorithm.algotom_fbp:

            reconstruction_array = rec.fbp_reconstruction(projections,
                                                          center_of_rotation,
                                                          angles=list_of_angles_rad,
                                                          apply_log=False,
                                                          ramp_win=None,
                                                          filter_name='hann',
                                                          pad=None,
                                                          pad_mode='edge',
                                                          ncore=max_workers,
                                                          gpu=False,
                                                          )
            reconstruction_array = np.swapaxes(reconstruction_array, 0, 1)
        
        elif algorithm == ReconstructionAlgorithm.algotom_gridrec:

            reconstruction_array = rec.gridrec_reconstruction(projections,
                                                              center_of_rotation,
                                                              angles=list_of_angles_rad,
                                                              apply_log=False,
                                                              ratio=1.0,
                                                              filter_name='shepp',
                                                              pad=100,
                                                              ncore=max_workers,
                                                              )
            reconstruction_array = np.swapaxes(reconstruction_array, 0, 1)
            
        elif algorithm == ReconstructionAlgorithm.astra_fbb:

            reconstruction_array = rec.astra_reconstruction(projections,
                                                            center_of_rotation,
                                                            angles=list_of_angles_rad,
                                                            apply_log=False,
                                                            method='SIRT_CUDA',
                                                            ratio=1.0,
                                                            filter_name='hann',
                                                            pad=None,
                                                            num_iter=300,
                                                            ncore=max_workers,
                                                            )
            reconstruction_array = np.swapaxes(reconstruction_array, 0, 1)

        elif algorithm == ReconstructionAlgorithm.tomopy_fbp:

            reconstruction_array = tomopy_recon(tomo=projections,
                                                theta=list_of_angles_rad,
                                                center=center_of_rotation,
                                                sinogram_order=False,
                                                # apply_log=False,
                                                algorithm='fbp',
                                                filter_name='hann',
                                                ncore=max_workers)

        else:
            raise NotImplementedError(f"Algorithm {algorithm} is not implemented yet!")

        return reconstruction_array
    
    @staticmethod
    def run_reconstruction_from_pre_data_mode(config_json_file):

        config = load_json_string(config_json_file)
        logging.info(f"config = {config}")

        input_data_folder = config["projections_pre_processing_folder"]
        base_output_folder = config['output_folder']

        list_tiff = glob.glob(os.path.join(input_data_folder, '*.tiff'))
        list_tiff.sort()
        print(f"loading {len(list_tiff)} images ... ", end="")
        logging.info(f"loading {len(list_tiff)} images ... ")
        #corrected_array_log = load_data_using_multithreading(list_tiff)
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
        
        list_of_slices_to_reconstruct = config['list_of_slices_to_reconstruct']
        top_slice = config['crop_region']['top']
        
        center_of_rotation = config['center_of_rotation']
        if center_of_rotation == -1:
            center_of_rotation = None

        # logging.info(f"before swapping I have (angle, y, x): {np.shape(corrected_array_log) = }")
        logging.info(f"{np.shape(corrected_array_log) = }")
        nbr_angles, nbr_slices, nbr_pixels_wide = np.shape(corrected_array_log)
        logging.info(f"{nbr_angles = }, {nbr_slices = }, {nbr_pixels_wide = }")

        # corrected_array_log = np.swapaxes(corrected_array_log, 0, 1)
        # logging.info(f"after swapping I should have (y, angle, x): {np.shape(corrected_array_log) = }")
        
        # corrected_array_log = np.swapaxes(corrected_array_log, 0, 1)
        # logging.info(f"after swapping I should have (y, angles, x): {np.shape(corrected_array_log) = }")

        logging.info(f"{list_of_angles_rad = }")
        logging.info(f"{input_data_folder = }")
        logging.info(f"{base_output_folder = }")
        logging.info(f"{list_of_slices_to_reconstruct = }")
        logging.info(f"{NUM_THREADS = }")
        logging.info(f"{center_of_rotation}")
            
        list_algorithm = config['reconstruction_algorithm']
        for _algo in list_algorithm: 
        
            if _algo == ReconstructionAlgorithm.svmbir:
                continue  # handled by SvmbirCliHandler

            logging.info(f"Reconstruction using {_algo} ...")
            print(f"Reconstruction using {_algo} ...")
            output_data_folder = os.path.join(base_output_folder, f"{_algo}_reconstructed_data_{get_current_time_in_special_file_name_format()}")
            logging.info(f"\t{output_data_folder = }")

            # make_or_reset_folder(output_data_folder)
            make_or_reset_folder(output_data_folder)

            if (len(list_of_slices_to_reconstruct) == 1) and (list_of_slices_to_reconstruct[0][0] == 0) and \
            (list_of_slices_to_reconstruct[0][1] == -1):
                list_of_slices_to_reconstruct = None

            list_of_output_folders = []
            if list_of_slices_to_reconstruct is not None:

                for [index, [top_slice_index, bottom_slice_index]] in enumerate(list_of_slices_to_reconstruct):
                    print(f"working with set of slices #{index}: from {top_slice_index} to {bottom_slice_index-1}. ", end="") 
                    logging.info(f"working with set of slices #{index}: from {top_slice_index} to {bottom_slice_index-1}")
                    print(f"launching reconstruction using {_algo} #{index} ... ", end="")
                    logging.info(f"launching reconstruction using {_algo} #{index} ...")
            
                    projections = corrected_array_log[:, top_slice_index:bottom_slice_index, :]
                    #_sino = corrected_array_log[top_slice_index:bottom_slice_index, :, :]   # [y, angles, x]
            
                    center_of_rotation = nbr_pixels_wide // 2

                    # projections = np.swapaxes(_sino, 0, 1)  # [angles, y, x]

                    reconstruction_array = FbpCliHandler._run_reconstruction(projections=projections,
                                                                             center_of_rotation=center_of_rotation,
                                                                             list_of_angles_rad=list_of_angles_rad,
                                                                             algorithm=_algo,
                                                                             max_workers=NUM_THREADS)
                              
                    print(f"done!")
                    logging.info(f"done with #{index}!")
                    _index = f"{index:03d}"
                    print(f"exporting reconstructed slices set #{_index} ... ", end="")
                    logging.info(f"\t{np.shape(reconstruction_array) = }")
                    logging.info(f"exporting reconstructed data set #{_index} ...")

                    _output_data_folder = os.path.join(output_data_folder, f"set_{_index}")
                    logging.info(f"making or resetting folder {_output_data_folder}")
                    list_of_output_folders.append(_output_data_folder)
                    make_or_reset_folder(_output_data_folder)
                    o_export = Export(image_3d=reconstruction_array,
                                    output_folder=_output_data_folder)
                    o_export.run()
                    print(f"done!")

                    logging.info(f"Cleaning up ...")
                    del reconstruction_array
                    # del _sino

                merge_reconstructed_slices(output_data_folder=output_data_folder, 
                                        top_slice=top_slice,
                                        list_of_output_folders=list_of_output_folders,
                                        list_of_slices_to_reconstruct=list_of_slices_to_reconstruct)
        
            else:

                print(f"launching reconstruction using {_algo} with all slices ... ", end="")
                logging.info(f"launching reconstruction using {_algo} with all slices ...")
                
                reconstruction_array = FbpCliHandler._run_reconstruction(projections=corrected_array_log,
                                                                        center_of_rotation=center_of_rotation,
                                                                        list_of_angles_rad=list_of_angles_rad,
                                                                        algorithm=_algo,
                                                                        max_workers=NUM_THREADS)

                print(f"done!")
                logging.info(f"done!")

                print(f"exporting reconstructed slices ... ", end="")
                logging.info(f"{np.shape(reconstruction_array) = }")
                logging.info(f"exporting reconstructed data ...")
                o_export = Export(image_3d=reconstruction_array,
                                output_folder=output_data_folder)
                o_export.run()
                print(f"done!")

            logging.info(f"exporting reconstructed data ... done!")
