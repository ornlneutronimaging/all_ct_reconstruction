import numpy as np
import os
import glob
import logging
import svmbir
import tomopy

from imars3d.backend.reconstruction import recon

from __code.workflow.export import Export
from __code.config import NUM_THREADS
from __code.utilities.logging import setup_logging
from __code.utilities.files import make_or_reset_folder, make_folder
from __code.config import SVMBIR_LIB_PATH
from __code.utilities.json import load_json_string
from __code.utilities.load import load_data_using_multithreading
from __code.utilities.time import get_current_time_in_special_file_name_format
from __code.workflow_cli.merge_reconstructed_slices import merge_reconstructed_slices


class FbpCliHandler:

    @staticmethod
    def _run_reconstruction(_sino, center_of_rotation, list_of_angles_rad, algorithm, max_workers):
        
        logging.info(f"\t -> {np.shape(_sino) = }")
        logging.info(f"\t -> {center_of_rotation = }")
        logging.info(f"\t -> {list_of_angles_rad = }")
        logging.info(f"\t -> {algorithm = }")
        logging.info(f"\t -> recon(array=_sino, center=center_offset, theta=list_of_angles_rad, algorithm=algorithm)")

        if algorithm == 'fbp':
            filter_name = "hann"
        else:
            filter_name = "none"

        reconstruction_array = recon(arrays=_sino,
                                     center=center_of_rotation,
                                     theta=list_of_angles_rad,
                                     filter_name="none",
                                     algorithm=algorithm,
                                     max_workers=NUM_THREADS)

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
        nbr_angles, nbr_slices, nbr_pixels_wide = np.shape(corrected_array_log)
        logging.info(f"{nbr_angles = }, {nbr_slices = }, {nbr_pixels_wide = }")
        # corrected_array_log = np.swapaxes(corrected_array_log, 1, 2)
        logging.info(f"after swapping I should have (angles, y, x): {np.shape(corrected_array_log) = }")

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
                    _sino = corrected_array_log[top_slice_index:bottom_slice_index, :, :]
 
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
