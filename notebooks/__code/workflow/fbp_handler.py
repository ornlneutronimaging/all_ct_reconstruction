import numpy as np
import os
from IPython.display import display
import ipywidgets as widgets
from IPython.core.display import HTML
import matplotlib.pyplot as plt
from ipywidgets import interactive
import logging
from tqdm import tqdm
import tomopy
import glob

import svmbir

from __code.workflow.export import Export
# from __code.utilities.configuration import Configuration
from __code.utilities.files import make_or_reset_folder
from __code.utilities.configuration_file import SvmbirConfig
from __code.parent import Parent
from __code import DataType, Run
from __code.config import NUM_THREADS, SVMBIR_LIB_PATH
from __code.utilities.save import make_tiff
from __code.utilities.time import get_current_time_in_special_file_name_format
from __code.utilities.json import save_json, load_json
from __code.utilities.load import load_data_using_multithreading


class FbpHandler(Parent):

    def export_pre_reconstruction_data(self):

        logging.info(f"Preparing reconstruction data to export json and projections")

        normalized_images_log = self.parent.normalized_images_log
        height, width = np.shape(normalized_images_log[0])

        list_of_angles = np.array(self.parent.final_list_of_angles)
        list_of_angles_rad = np.array([np.deg2rad(float(_angle)) for _angle in list_of_angles])

        self.parent.configuration.list_of_angles = list(list_of_angles_rad)

        # corrected_array_log = tomopy.minus_log(corrected_array)
        # where_nan = np.where(np.isnan(corrected_array_log))
        # corrected_array_log[where_nan] = 0

        # corrected_array = corrected_array_log

        logging.info(f"\t{np.min(normalized_images_log) =}")
        logging.info(f"\t{np.max(normalized_images_log) =}")
        logging.info(f"\t{np.mean(normalized_images_log) =}")

        output_folder = self.parent.working_dir[DataType.extra]
        _time_ext = get_current_time_in_special_file_name_format()
        base_sample_folder = os.path.basename(self.parent.working_dir[DataType.sample])
        pre_projections_export_folder = os.path.join(output_folder, f"{base_sample_folder}_projections_pre_data_{_time_ext}")
        os.makedirs(pre_projections_export_folder)
        logging.info(f"\tprojections pre data will be exported to {pre_projections_export_folder}!")
        logging.info(f"\toutput folder: {output_folder}")
        
        self.parent.configuration.projections_pre_processing_folder = pre_projections_export_folder

        full_output_folder = os.path.join(output_folder, f"{base_sample_folder}_reconstructed_{_time_ext}")

        # go from [angle, height, width] to [angle, width, height]
        # corrected_array_log = np.moveaxis(corrected_array_log, 1, 2)  # angle, y, x -> angle, x, y
        logging.info(f"\t{np.shape(normalized_images_log) =}")

        for _index, _data in tqdm(enumerate(normalized_images_log)):

            if _index == 0:
                logging.info(f"\t{np.shape(_data) = }")
                logging.info(f"\t{_data.dtype = }")
                # logging.info(f"\t{top_slice = }")
                # logging.info(f"\t{bottom_slice = }")

            short_file_name = f"pre-reconstruction_{_index:04d}.tiff"
            full_file_name = os.path.join(pre_projections_export_folder, short_file_name)
            # make_tiff(data=_data[top_slice:bottom_slice+1, :], filename=full_file_name)
            make_tiff(data=_data, filename=full_file_name)
        print(f"projections exported in {pre_projections_export_folder}")
        print(f"top output folder: {output_folder}")

    def export_images(self):
        
        logging.info(f"Exporting the reconstructed slices")
        logging.info(f"\tfolder selected: {self.parent.working_dir[DataType.reconstructed]}")

        reconstructed_array = self.parent.reconstruction_array

        master_base_folder_name = f"{os.path.basename(self.parent.working_dir[DataType.sample])}_reconstructed"
        full_output_folder = os.path.join(self.parent.working_dir[DataType.reconstructed],
                                          master_base_folder_name)

        make_or_reset_folder(full_output_folder)

        o_export = Export(image_3d=reconstructed_array,
                          output_folder=full_output_folder)
        o_export.run()
        logging.info(f"\texporting reconstructed images ... Done!")

        # update configuration
        self.parent.configuration.output_folder = full_output_folder
