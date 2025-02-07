import numpy as np
import os
import glob
import logging
import svmbir

from __code.workflow.export import Export
from __code.utilities.logging import setup_logging
from __code.utilities.files import make_or_reset_folder, make_folder
from __code.config import NUM_THREADS, SVMBIR_LIB_PATH
from __code.utilities.json import load_json_string
from __code.utilities.load import load_data_using_multithreading, load_list_of_tif
from __code.utilities.time import get_current_time_in_special_file_name_format
from __code.workflow_cli.merge_reconstructed_slices import merge_reconstructed_slices


class SvmbirCliHandler:

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
        corrected_array_log = load_list_of_tif(list_tiff, dtype=np.float32)
        print(f"done!")
        logging.info(f"loading {len(list_tiff)} images ... done")
      
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
        logging.info(f"{np.shape(corrected_array_log) = }")

        logging.info(f"{list_of_angles_rad = }")
        # logging.info(f"{height = }")
        # logging.info(f"{width = }")
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
        if list_of_slices_to_reconstruct:

            for [index, [top_slice_index, bottom_slice_index]] in enumerate(list_of_slices_to_reconstruct):
                print(f"working with set of slices #{index}: from {top_slice} to {bottom_slice_index-1}. ", end="") 
                logging.info(f"working with set of slices #{index}: from {top_slice} to {bottom_slice_index-1}")
                print(f"launching svmbir #{index} ... ", end="")
                logging.info(f"launching svmbir #{index} ...")
        
                _sino = corrected_array_log[:, top_slice_index:bottom_slice_index, :]
        
                logging.info(f"\t{np.shape(_sino) = }")
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
                print(f"done with #{index}!")
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

            merge_reconstructed_slices(output_data_folder=output_data_folder, 
                                       top_slice=top_slice,
                                       list_of_output_folders=list_of_output_folders,
                                       list_of_slices_to_reconstruct=list_of_slices_to_reconstruct)

        else:

            print(f"launching svmbir with all slices ... ", end="")
            logging.info(f"launching svmbir with all slices ...")

            reconstruction_array = svmbir.recon(sino=corrected_array_log,
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

            print(f"done! ")
            logging.info(f"done with!")

            print(f"exporting reconstructed slices ... ", end="")
            logging.info(f"{np.shape(reconstruction_array) = }")
            logging.info(f"exporting reconstructed data ...")
            o_export = Export(image_3d=reconstruction_array,
                            output_folder=output_data_folder)
            o_export.run()
            print(f"done!")

        logging.info(f"exporting reconstructed data ... done!")

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
            logging.info(f"exporting reconstructed data of iteratin {_iter} ...")

            _output_data_folder = os.path.join(output_data_folder, f"set_with_{_nbr_projections}_projections")
            logging.info(f"making or resetting folder {_output_data_folder}")
            list_of_output_folders.append(_output_data_folder)
            make_or_reset_folder(_output_data_folder)

            o_export = Export(image_3d=reconstruction_array,
                                output_folder=_output_data_folder)
            o_export.run()

            del reconstruction_array
