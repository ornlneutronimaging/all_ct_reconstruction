import numpy as np
import os
import glob
import logging
import svmbir
import tomopy

# from imars3d.backend.reconstruction import recon
from tomopy.recon.algorithm import recon as tomopy_recon
import algotom.rec.reconstruction as rec

from __code.workflow.export import Export
from __code.config import NUM_THREADS
from __code.utilities.logging import setup_logging
from __code.utilities.files import make_or_reset_folder, make_folder
from __code.config import SVMBIR_LIB_PATH
from __code.utilities.json import load_json_string
from __code.utilities.load import load_data_using_multithreading
from __code.utilities.time import get_current_time_in_special_file_name_format
from __code.workflow_cli.merge_reconstructed_slices import merge_reconstructed_slices
from __code.utilities.configuration_file import ReconstructionAlgorithm


class FbpCliHandler:

    @staticmethod
    def _run_reconstruction(_sino, center_of_rotation, list_of_angles_rad, algorithm, max_workers):
        
        logging.info(f"\t -> {np.shape(_sino) = }")
        logging.info(f"\t -> {center_of_rotation = }")
        logging.info(f"\t -> {list_of_angles_rad = }")
        logging.info(f"\t -> {len(list_of_angles_rad) = }")
        logging.info(f"\t -> {algorithm = }")
        logging.info(f"\t -> launching reconstruction using {algorithm} ...")

        if algorithm == ReconstructionAlgorithm.algotom_fbp:

            reconstruction_array = rec.fbp_reconstruction(_sino,
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

            reconstruction_array = rec.gridrec_reconstruction(_sino,
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

            reconstruction_array = rec.astra_reconstruction(_sino,
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

            # reconstruction_array = tomopy_recon(tomo=_sino,
            #                             # center=center_of_rotation,
            #                             theta=list_of_angles_rad,
            #                             sinogram_order=True,
            #                             # apply_log=False,
            #                             algorithm='fbp',
            #                             filter_name='hann')
            #                             # ncore=NUM_THREADS)

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
        corrected_array_log = load_data_using_multithreading(list_tiff)
        print(f"done!")
        logging.info(f"loading {len(list_tiff)} images ... done")
      
        list_of_angles_rad = np.array(config['list_of_angles'])
        
        list_of_slices_to_reconstruct = config['list_of_slices_to_reconstruct']
        top_slice = config['crop_region']['top']
        
        center_of_rotation = config['center_of_rotation']
        if center_of_rotation == -1:
            center_of_rotation = None

        logging.info(f"before swapping I have (angle, x, y): {np.shape(corrected_array_log) = }")
        nbr_angles, nbr_pixels_wide, nbr_slices = np.shape(corrected_array_log)
        logging.info(f"{nbr_angles = }, {nbr_slices = }, {nbr_pixels_wide = }")
        corrected_array_log = np.swapaxes(corrected_array_log, 1, 2)
        logging.info(f"after swapping I should have (angles, y, x): {np.shape(corrected_array_log) = }")
        corrected_array_log = np.swapaxes(corrected_array_log, 0, 1)
        logging.info(f"after swapping I should have (y, angles, x): {np.shape(corrected_array_log) = }")

        logging.info(f"{list_of_angles_rad = }")
        logging.info(f"{input_data_folder = }")
        logging.info(f"{base_output_folder = }")
        logging.info(f"{list_of_slices_to_reconstruct = }")
        logging.info(f"{NUM_THREADS = }")
        logging.info(f"{center_of_rotation}")
            
        list_algorithm = config['reconstruction_algorithm']
        for _algo in list_algorithm: 

            # if _algo == 'gridrec':
            #     continue




            logging.info(f"Reconstruction using {_algo} ...")
            output_data_folder = os.path.join(base_output_folder, f"{_algo}_reconstructed_data_{get_current_time_in_special_file_name_format()}")
            logging.info(f"\t{output_data_folder = }")

            # make_or_reset_folder(output_data_folder)
            make_or_reset_folder(output_data_folder)

            if (len(list_of_slices_to_reconstruct) == 1) and (list_of_slices_to_reconstruct[0][0] == 0) and \
            (list_of_slices_to_reconstruct[0][1] == nbr_slices):
                list_of_slices_to_reconstruct = None

            list_of_output_folders = []
            if list_of_slices_to_reconstruct:

                for [index, [top_slice_index, bottom_slice_index]] in enumerate(list_of_slices_to_reconstruct):
                    print(f"working with set of slices #{index}: from {top_slice_index} to {bottom_slice_index-1}. ", end="") 
                    logging.info(f"working with set of slices #{index}: from {top_slice_index} to {bottom_slice_index-1}")
                    print(f"launching reconstruction using {_algo} #{index} ... ", end="")
                    logging.info(f"launching reconstruction using {_algo} #{index} ...")
            
#                    _sino = corrected_array_log[:, top_slice_index:bottom_slice_index, :]
                    _sino = corrected_array_log[top_slice_index:bottom_slice_index, :, :]   # [y, angles, x]
            
                    center_of_rotation = nbr_pixels_wide // 2

                    _sino = np.swapaxes(_sino, 0, 1)  # [angles, y, x]

                    reconstruction_array = FbpCliHandler._run_reconstruction(_sino=_sino,
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
                    del _sino

                merge_reconstructed_slices(output_data_folder=output_data_folder, 
                                        top_slice=top_slice,
                                        list_of_output_folders=list_of_output_folders,
                                        list_of_slices_to_reconstruct=list_of_slices_to_reconstruct)
        
            else:

                print(f"launching reconstruction using {_algo} with all slices ... ", end="")
                logging.info(f"launching reconstruction using {_algo} with all slices ...")
                
                reconstruction_array = FbpCliHandler._run_reconstruction(_sino=corrected_array_log,
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
