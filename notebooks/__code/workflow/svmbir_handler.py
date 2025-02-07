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


class SvmbirHandler(Parent):

    def set_settings(self):

        # corrected_array = self.parent.corrected_images
        # nbr_images = len(corrected_array)
        # height, _ = np.shape(corrected_array[0])

        # list_angles = self.parent.final_list_of_angles
        # list_runs = self.parent.list_of_runs_to_use[DataType.sample]

        # display(widgets.HTML("<font size=5>Select range of slices to reconstruct</font"))

        # [default_top, default_bottom] = self.parent.configuration.range_of_slices_for_center_of_rotation
        # if default_bottom == 0:
        #     default_bottom = height - 1

        # if default_top > height:
        #     default_top = 0

        # if default_bottom > height:
        #     default_bottom = height - 1

        # def plot_range(image_index, top_slice, bottom_slice):

        #     fig, axs = plt.subplots(nrows=1, ncols=1)

        #     axs.set_title(f"angle: {list_angles[image_index]}")
        #     axs.imshow(corrected_array[image_index], vmin=0, vmax=1)
        #     axs.axhspan(top_slice, bottom_slice, color='blue', alpha=0.3)
        #     axs.axhline(top_slice, color='red', linestyle='--')
        #     axs.axhline(bottom_slice, color='red', linestyle='--')

        #     plt.tight_layout()
        #     plt.show()

        #     return top_slice, bottom_slice

        # self.display_corrected_range = interactive(plot_range,
        #                                             image_index = widgets.IntSlider(min=0,
        #                                                                             max=nbr_images-1,
        #                                                                             value=0),
        #                                            top_slice = widgets.IntSlider(min=0,
        #                                                                          max=height-1,
        #                                                                          value=default_top),
        #                                             bottom_slice = widgets.IntSlider(min=0,
        #                                                                              max=height-1,
        #                                                                              value=default_bottom),
        #                                             )
        # display(self.display_corrected_range)
        # display(widgets.HTML("<hr>"))
        # display(widgets.HTML("<font size=5>Define reconstruction settings</font"))

        title_label = widgets.HTML("<font size=5 color=blue>Define reconstruction settings for svmBIR</font")

        self.sharpness_ui = widgets.FloatSlider(min=0,
                                           max=1,
                                           value=0,
                                           layout=widgets.Layout(width="50%"),
                                           description="sharpness")
        self.snr_db_ui = widgets.FloatSlider(min=0,
                                        max=100,
                                        value=30.0,
                                        layout=widgets.Layout(width="50%"),
                                        description="snr db")
        self.positivity_ui = widgets.Checkbox(value=False,
                                         description="positivity")
        self.max_iterations_ui = widgets.IntSlider(value=20,
                                              min=10,
                                              max=500,
                                              layout=widgets.Layout(width="50%"),
                                              description="max itera.")
        label = widgets.Label("max resolution (0-high, 4-low):")
        self.max_resolutions_ui = widgets.IntSlider(value=2,
                                                    min=0,
                                                    max=4,
                                                    description="",
                                                    layout=widgets.Layout(width="50%"))      
        self.verbose_ui = widgets.Checkbox(value=True,
                                      description='verbose')
        
        vertical_widgets = widgets.VBox([title_label,
                                         self.sharpness_ui,
                                         self.snr_db_ui,
                                         self.positivity_ui,
                                         self.max_iterations_ui,
                                         widgets.HBox([label, self.max_resolutions_ui]),
                                         self.verbose_ui])
        display(vertical_widgets)

    def display_sinograms(self):

        corrected_array = self.parent.normalized_images_log
        max_value = np.max(corrected_array)
        height, _ = np.shape(corrected_array[0])
        
        def display_sinograms(slice_index):

            fig, axs = plt.subplots(nrows=1, ncols=1)
            axs.imshow(corrected_array[:, slice_index, :])
            plt.tight_layout()
            plt.show()

        display_sinogram = interactive(display_sinograms,
                                       slice_index = widgets.IntSlider(min=0,
                                                                       max=height-1,
                                                                       value=0),
                                        vmin=widgets.IntSlider(min=0,
                                                               max=max_value,
                                                               value=0),
                                        vmax=widgets.IntSlider(min=0,
                                                               max=max_value,
                                                               value=max_value),
                                        )
        display(display_sinogram)

    def _get_list_of_index_of_runs_to_exclude(self):
        """this return the index of the runs selected and that need to be rejected from final reconstruction"""
        try:
            list_options = self.parent.runs_to_exclude_ui.options
            list_value = self.parent.runs_to_exclude_ui.value
            list_index = []
            for _index, _option in enumerate(list_options):
                if _option in list_value:
                    list_index.append(_index)
            return list_index, list_value
        except AttributeError:
            return [], []

    def export_pre_reconstruction_data(self):

        logging.info(f"Preparing reconstruction data to export json and projections")

        corrected_array_log = self.parent.normalized_images_log
        height, width = np.shape(corrected_array_log[0])

        list_of_angles = np.array(self.parent.final_list_of_angles)
        list_of_angles_rad = np.array([np.deg2rad(float(_angle)) for _angle in list_of_angles])
        # list_of_runs_to_use = self.parent.list_of_runs_to_use[DataType.sample]
        # list_of_sample_pc = self.parent.final_dict_of_pc[DataType.sample]
        # list_of_sample_pc_to_use = list_of_sample_pc

        # list_of_sample_frame_number = self.parent.final_dict_of_frame_number[DataType.sample]
        # list_of_sample_frame_number_to_use = list_of_sample_frame_number
        
        # # looking at list of runs to reject
        # list_of_index_of_runs_to_exlude, list_runs_to_exclude = self._get_list_of_index_of_runs_to_exclude()
        # if list_of_index_of_runs_to_exlude:
        #     logging.info(f"\tUser wants to reject the following runs: {list_runs_to_exclude}!")
        #     corrected_array = np.delete(corrected_array, list_of_index_of_runs_to_exlude, axis=0)
        #     list_of_angles_rad = np.delete(list_of_angles_rad, list_of_index_of_runs_to_exlude, axis=0)
        #     list_of_runs_to_use = np.delete(list_of_runs_to_use, list_of_index_of_runs_to_exlude)
        #     list_of_sample_pc_to_use = np.delete(list_of_sample_pc, list_of_index_of_runs_to_exlude)
        #     list_of_sample_frame_number_to_use = np.delete(list_of_sample_frame_number, list_of_index_of_runs_to_exlude)

            # updating configuration
            # self.parent.configuration.list_of_sample_index_to_reject = list_of_index_of_runs_to_exlude

        # else:
        #     logging.info(f"\tNo runs rejected before final reconstruction!")

        # update configuration
        # self.parent.configuration.list_of_sample_runs = list(list_of_runs_to_use)
        self.parent.configuration.list_of_angles = list(list_of_angles_rad)

        # save pc and frame number in configuration
        # self.parent.configuration.list_of_sample_frame_number = list_of_sample_frame_number_to_use
        # self.parent.configuration.list_of_sample_pc = list_of_sample_pc_to_use
        # self.parent.configuration.list_of_ob_pc = self.parent.final_dict_of_pc[DataType.ob]
        # self.parent.configuration.list_of_ob_frame_number = self.parent.final_dict_of_frame_number[DataType.ob]

        # top_slice, bottom_slice = self.display_corrected_range.result
        sharpness = self.sharpness_ui.value
        snr_db = self.snr_db_ui.value
        positivity = self.positivity_ui.value
        max_iterations = self.max_iterations_ui.value
        max_resolutions = self.max_resolutions_ui.value
        verbose = 1 if self.verbose_ui.value else 0

        # update configuration
        svmbir_config = SvmbirConfig()
        svmbir_config.sharpness = sharpness
        svmbir_config.snr_db = snr_db
        svmbir_config.positivity = positivity
        svmbir_config.max_iterations = max_iterations
        svmbir_config.verbose = verbose
        # svmbir_config.top_slice = top_slice
        # svmbir_config.bottom_slice = bottom_slice
        self.parent.configuration.svmbir_config = svmbir_config

        # logging.info(f"\t{top_slice = }")
        # logging.info(f"\t{bottom_slice = }")
        logging.info(f"\t{sharpness = }")
        logging.info(f"\t{snr_db = }")
        logging.info(f"\t{positivity = }")
        logging.info(f"\t{max_iterations = }")
        logging.info(f"\t{max_resolutions = }")
        logging.info(f"\t{verbose = }")
        logging.info(f"\t{list_of_angles_rad = }")
        logging.info(f"\t{width = }")
        logging.info(f"\t{height = }")
        logging.info(f"\t{type(corrected_array_log) = }")
        logging.info(f"\t{np.shape(corrected_array_log) = }")

        # corrected_array_log = tomopy.minus_log(corrected_array)

        where_nan = np.where(np.isnan(corrected_array_log))
        corrected_array_log[where_nan] = 0

        logging.info(f"\t{np.min(corrected_array_log) =}")
        logging.info(f"\t{np.max(corrected_array_log) =}")
        logging.info(f"\t{np.mean(corrected_array_log) =}")

        if not (self.parent.center_of_rotation is None):
            center_offset = self.parent.center_of_rotation - int(width /2)

            # removing left region we cropped to the offset
            crop_left = self.parent.crop_region['left']
            if crop_left:
                center_offset -= crop_left
        else:
            center_offset = 0

        output_folder = os.path.abspath(self.parent.working_dir[DataType.extra])
        _time_ext = get_current_time_in_special_file_name_format()
        base_sample_folder = os.path.basename(os.path.abspath(self.parent.working_dir[DataType.sample]))
        pre_projections_export_folder = os.path.join(output_folder, f"{base_sample_folder}_projections_pre_data_{_time_ext}")
        os.makedirs(pre_projections_export_folder)
        logging.info(f"\tprojections pre data will be exported to {pre_projections_export_folder}!")
        logging.info(f"\toutput folder: {output_folder}")
        
        self.parent.configuration.projections_pre_processing_folder = pre_projections_export_folder

        full_output_folder = os.path.join(output_folder, f"{base_sample_folder}_reconstructed_{_time_ext}")

        # go from [angle, height, width] to [angle, width, height]
        # corrected_array_log = np.moveaxis(corrected_array_log, 1, 2)  # angle, y, x -> angle, x, y
        logging.info(f"\t{np.shape(corrected_array_log) =}")

        for _index, _data in tqdm(enumerate(corrected_array_log)):

            if _index == 0:
                logging.info(f"\t{np.shape(_data) = }")
                # logging.info(f"\t{top_slice = }")
                # logging.info(f"\t{bottom_slice = }")

            short_file_name = f"pre-reconstruction_{_index:04d}.tiff"
            full_file_name = os.path.join(pre_projections_export_folder, short_file_name)
            # make_tiff(data=_data[top_slice:bottom_slice+1, :], filename=full_file_name)
            make_tiff(data=_data, filename=full_file_name)
        print(f"projections exported in {pre_projections_export_folder}")
        print(f"top output folder: {output_folder}")

        # export_dict = Configuration()
        # export_dict.list_of_angles = list(list_of_angles_rad)
        # export_dict.image_size = height
        # export_dict.width = width

        # configuration = self.parent.configuration
        # configuration.list_of_angles = list(list_of_angles_rad)

        # export_dict = {'list_of_angles_rad': list(list_of_angles_rad),
        #                'height': height,
        #                'width': width,
        #                'center_offset': center_offset,
        #                'sharpness': sharpness,
        #                'snr_db': snr_db,
        #                'positivity': positivity,
        #                'max_iterations': max_iterations,
        #                'max_resolutions': max_resolutions,
        #                'verbose': verbose,
        #                'svmbir_lib_path': SVMBIR_LIB_PATH,
        #                'input_folder': pre_projections_export_folder,
        #                'reconstructed_output_folder': output_folder,
        #                'output_folder': full_output_folder,
        #                'projections_pre_processing_folder': pre_projections_export_folder,
        #                }
        
        # self.parent.export_dict = export_dict

        # json_file_name = os.path.join(output_folder, f"projections_pre_metadata_{_time_ext}.json")
        # save_json(json_file_name=json_file_name,
        #           json_dictionary=export_dict)
        # print(f"{json_file_name} exported !")

    # def run_reconstruction(self):

    #     logging.info(f"Running reconstruction:")

    #     corrected_array = self.parent.corrected_images
    #     height, width = np.shape(corrected_array[0])

    #     list_of_angles = np.array(self.parent.final_list_of_angles)
    #     list_of_angles_rad = np.array([np.deg2rad(float(_angle)) for _angle in list_of_angles])
    #     # list_of_runs_to_use = self.parent.list_of_runs_to_use[DataType.sample]
    #     # list_of_sample_pc = self.parent.final_dict_of_pc[DataType.sample]
    #     # list_of_sample_pc_to_use = list_of_sample_pc

    #     # list_of_sample_frame_number = self.parent.final_dict_of_frame_number[DataType.sample]
    #     # list_of_sample_frame_number_to_use = list_of_sample_frame_number
        
    #     # # looking at list of runs to reject
    #     # list_of_index_of_runs_to_exlude, list_runs_to_exclude = self._get_list_of_index_of_runs_to_exclude()
    #     # if list_of_index_of_runs_to_exlude:
    #     #     logging.info(f"\tUser wants to reject the following runs: {list_runs_to_exclude}!")
    #     #     corrected_array = np.delete(corrected_array, list_of_index_of_runs_to_exlude, axis=0)
    #     #     list_of_angles_rad = np.delete(list_of_angles_rad, list_of_index_of_runs_to_exlude, axis=0)
    #     #     list_of_runs_to_use = np.delete(list_of_runs_to_use, list_of_index_of_runs_to_exlude)
    #     #     list_of_sample_pc_to_use = np.delete(list_of_sample_pc, list_of_index_of_runs_to_exlude)
    #     #     list_of_sample_frame_number_to_use = np.delete(list_of_sample_frame_number, list_of_index_of_runs_to_exlude)

    #         # updating configuration
    #         # self.parent.configuration.list_of_sample_index_to_reject = list_of_index_of_runs_to_exlude

    #     # else:
    #     #     logging.info(f"\tNo runs rejected before final reconstruction!")

    #     # update configuration
    #     # self.parent.configuration.list_of_sample_runs = list(list_of_runs_to_use)
    #     self.parent.configuration.list_of_angles = list(list_of_angles_rad)

    #     # save pc and frame number in configuration
    #     # self.parent.configuration.list_of_sample_frame_number = list_of_sample_frame_number_to_use
    #     # self.parent.configuration.list_of_sample_pc = list_of_sample_pc_to_use
    #     # self.parent.configuration.list_of_ob_pc = self.parent.final_dict_of_pc[DataType.ob]
    #     # self.parent.configuration.list_of_ob_frame_number = self.parent.final_dict_of_frame_number[DataType.ob]

    #     top_slice, bottom_slice = self.display_corrected_range.result
    #     sharpness = self.sharpness_ui.value
    #     snr_db = self.snr_db_ui.value
    #     positivity = self.positivity_ui.value
    #     max_iterations = self.max_iterations_ui.value
    #     max_resolutions = self.max_resolutions_ui.value
    #     verbose = 1 if self.verbose_ui.value else 0

    #     # update configuration
    #     svmbir_config = SvmbirConfig()
    #     svmbir_config.sharpness = sharpness
    #     svmbir_config.snr_db = snr_db
    #     svmbir_config.positivity = positivity
    #     svmbir_config.max_iterations = max_iterations
    #     svmbir_config.verbose = verbose
    #     svmbir_config.top_slice = top_slice
    #     svmbir_config.bottom_slice = bottom_slice
    #     self.parent.configuration.svmbir_config = svmbir_config

    #     logging.info(f"\t{top_slice = }")
    #     logging.info(f"\t{bottom_slice = }")
    #     logging.info(f"\t{sharpness = }")
    #     logging.info(f"\t{snr_db = }")
    #     logging.info(f"\t{positivity = }")
    #     logging.info(f"\t{max_iterations = }")
    #     logging.info(f"\t{max_resolutions = }")
    #     logging.info(f"\t{verbose = }")
    #     logging.info(f"\t{list_of_angles_rad = }")
    #     logging.info(f"\t{width = }")
    #     logging.info(f"\t{height = }")
    #     logging.info(f"\t{type(corrected_array) = }")
    #     logging.info(f"\t{np.shape(corrected_array) = }")
    #     logging.info(f"\t launching reconstruction ...")

    #     corrected_array_log = tomopy.minus_log(corrected_array)

    #     where_nan = np.where(np.isnan(corrected_array_log))
    #     corrected_array_log[where_nan] = 0

    #     logging.info(f"\t{np.min(corrected_array_log) =}")
    #     logging.info(f"\t{np.max(corrected_array_log) =}")
    #     logging.info(f"\t{np.mean(corrected_array_log) =}")

    #     if not (self.parent.center_of_rotation is None):
    #         center_offset = self.parent.center_of_rotation - int(width /2)
    #     else:
    #         center_offset = 0

    #     self.parent.reconstruction_array = svmbir.recon(sino=corrected_array_log[:, top_slice: bottom_slice+1, :],
    #                                                     angles=list_of_angles_rad,
    #                                                     # num_rows = height,
    #                                                     # num_cols = width,
    #                                                     center_offset = center_offset,
    #                                                     sharpness = sharpness,
    #                                                     snr_db = snr_db,
    #                                                     positivity = positivity,
    #                                                     max_iterations = max_iterations,
    #                                                     num_threads = NUM_THREADS,
    #                                                     max_resolutions = max_resolutions,
    #                                                     verbose = verbose,
    #                                                     svmbir_lib_path = SVMBIR_LIB_PATH,
    #                                                     )
    #     logging.info(f"\t Done !")

    # def display_slices(self):

    #     reconstruction_array = self.parent.reconstruction_array

    #     height = self.parent.image_size['height']
        
    #     def display_slices(slice_index):

    #         fig, axs = plt.subplots(nrows=1, ncols=1)
    #         im1 = axs.imshow(reconstruction_array[:, slice_index, :])
    #         plt.colorbar(im1, ax=axs, shrink=0.5)
    #         plt.tight_layout()
    #         plt.show()

    #     display_svmbir_slices = interactive(display_slices,
    #                                         slice_index = widgets.IntSlider(min=0,
    #                                                                         max=height-1,
    #                                                                         value=0),
    #                                             )
    #     display(display_svmbir_slices)

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
