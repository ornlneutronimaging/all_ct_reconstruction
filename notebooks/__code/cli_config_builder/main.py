import os
from ipywidgets import interactive
from IPython.display import display
import ipywidgets as widgets
from tqdm import tqdm
import numpy as np

from __code.utilities.file_folder_browser import FileFolderBrowser
from __code import roi
from __code.utilities.json import save_json
from __code.utilities.configuration_file import Configuration
from __code import DataType, OperatingMode, CleaningAlgorithm, NormalizationSettings
from __code.utilities.files import retrieve_list_of_runs, get_angle_value, get_number_of_tif, retrieve_list_of_tif
from __code.utilities.time import get_current_time_in_special_file_name_format
from __code.utilities.nexus import get_proton_charge, get_frame_number
from __code.workflow.remove_strips import RemoveStrips
from __code.utilities.load import load_data_using_multithreading

DEBUG = True


class CliConfigBuilder:

    data_type = DataType.sample
    top_folder = "./"
    full_ipts_folder = None
    nbr_images = 0  # nbr of tof bins, or TIFF images in each run folder
    
    image_size = None

    at_least_one_run_with_no_pc_info = False
    at_least_one_frame_number_not_found = False

    def __init__(self):
        self.configuration = Configuration()

    def select_sample_folder(self, system=None):
        
        self.full_ipts_folder = system.System.get_working_dir()
        
        if DEBUG:
            top_folder = "/SNS/VENUS/IPTS-33699/shared/autoreduce/mcp/September20_2024_PurpleCar_GoldenRatio_CT_5_0_C_Cd_inBeam_Resonance"
        else:
             top_folder = system.System.get_working_dir()
        
        self.instrument = system.System.get_instrument_selected()

        self.data_type = DataType.sample
        self.top_folder = top_folder
        o_file_browser = FileFolderBrowser(working_dir=top_folder,
                                           next_function=self.data_selected)
        o_file_browser.select_input_folder(instruction=f"Select Top Folder of {self.data_type}",
                                           multiple_flag=False)

    def data_selected(self, top_folder):
        if self.data_type == DataType.sample:
            self.configuration.top_folder.sample = top_folder
        else:
            self.configuration.top_folder.ob = top_folder
        print(f"As {self.data_type} folder, you selected: {top_folder} ")

    def select_ob_folder(self):
        if DEBUG:
            self.top_folder = "/SNS/VENUS/IPTS-33699/shared/autoreduce/mcp/September26_2024_PurpleCar_OpenBean_5_0_C_Cd_inBeam_Resonance"
        self.data_type = DataType.ob
        o_file_browser = FileFolderBrowser(working_dir=self.top_folder,
                                           next_function=self.data_selected)
        o_file_browser.select_input_folder(instruction=f"Select Top Folder of {self.data_type}",
                                           multiple_flag=False)

    def select_runs_to_remove(self):
        sample_list_of_runs = retrieve_list_of_runs(self.configuration.top_folder.sample)
        ob_list_list_of_runs = retrieve_list_of_runs(self.configuration.top_folder.ob)

        short_sample_list_of_runs = [os.path.basename(_run) for _run in sample_list_of_runs]
        short_ob_list_of_runs = [os.path.basename(_run) for _run in ob_list_list_of_runs]

        sample_runs = widgets.VBox([
            widgets.Label("Sample"),
            widgets.SelectMultiple(options=short_sample_list_of_runs,
                                    layout=widgets.Layout(height="100%",
                                                            width='100%',
                                                            )),                                                       
        ],
        layout=widgets.Layout(width='200px',
                                height='300px'))
        self.list_of_sample_runs_to_reject_ui = sample_runs.children[1]

        ob_runs = widgets.VBox([
            widgets.Label("OB"),
            widgets.SelectMultiple(options=short_ob_list_of_runs,
                                    layout=widgets.Layout(height="100%",
                                                            width='100%'))
        ],
        layout=widgets.Layout(width='200px',
                                height='300px'))
        self.list_of_ob_runs_to_reject_ui = ob_runs.children[1]

        hori_layout = widgets.HBox([sample_runs, ob_runs])
        display(hori_layout)

    def get_list_of_nexus(self, short_list_of_runs=None, top_folder=None):
        top_nexus_path = os.path.join(self.full_ipts_folder, 'nexus')
        list_of_nexus = []
        for _run in short_list_of_runs:
            _, number = os.path.basename(_run).split("_")
            nexus_path = os.path.join(top_nexus_path, f"{self.instrument}_{number}.nxs.h5")
            if not os.path.exists(nexus_path):
                nexus_path = None
            list_of_nexus.append(nexus_path)
        return list_of_nexus

    def retrieve_metadata(self):
        sample_list_of_runs = retrieve_list_of_runs(self.configuration.top_folder.sample)
        ob_list_list_of_runs = retrieve_list_of_runs(self.configuration.top_folder.ob)

        short_sample_list_of_runs = [os.path.basename(_run) for _run in sample_list_of_runs]
        short_ob_list_of_runs = [os.path.basename(_run) for _run in ob_list_list_of_runs]

        list_of_sample_runs_to_remove = self.list_of_sample_runs_to_reject_ui.value
        list_of_ob_runs_to_remove = self.list_of_ob_runs_to_reject_ui.value

        for _sample in list_of_sample_runs_to_remove:
            short_sample_list_of_runs.remove(_sample)

        for _ob in list_of_ob_runs_to_remove:
            short_ob_list_of_runs.remove(_ob)

        # build list of NeXus
        sample_list_of_nexus = self.get_list_of_nexus(short_sample_list_of_runs, 
                                                      self.configuration.top_folder.sample)
        ob_list_of_nexus = self.get_list_of_nexus(short_ob_list_of_runs,
                                                  self.configuration.top_folder.ob)

        # get proton charge
        sample_list_proton_charge = []
        for _nexus in tqdm(sample_list_of_nexus):
            sample_list_proton_charge.append(get_proton_charge(_nexus, units='c'))

        ob_list_proton_charge = []
        for _nexus in tqdm(ob_list_of_nexus):
            ob_list_proton_charge.append(get_proton_charge(_nexus, units='c'))

        # get frame number
        sample_list_frame_number = []
        for _nexus in tqdm(sample_list_of_nexus):
            sample_list_frame_number.append(get_frame_number(_nexus))

        ob_list_frame_number = []
        for _nexus in tqdm(ob_list_of_nexus):
            ob_list_frame_number.append(get_frame_number(_nexus))

        # angle value
        sample_list_of_angles = []
        for _run in tqdm(short_sample_list_of_runs):
            sample_list_of_angles.append(get_angle_value(os.path.join(self.configuration.top_folder.sample, _run)))

        # get number of images (tof bins)
        self.nbr_images = get_number_of_tif(sample_list_of_runs[0])

        # remove bad runs (no tiff, no nexus)
        list_sample_indexes_with_none_in_proton_charge = [i for i, pc in enumerate(sample_list_proton_charge) if pc is None]
        if len(list_sample_indexes_with_none_in_proton_charge):
            self.at_least_one_run_with_no_pc_info = True

        list_sample_indexes_with_none_in_frame_number = [i for i, fn in enumerate(sample_list_frame_number) if fn is None]
        if len(list_sample_indexes_with_none_in_frame_number):
            self.at_least_one_frame_number_not_found = True

        list_sample_indexes_with_none_in_angles = [i for i, angle in enumerate(sample_list_of_angles) if angle is None]

        size_before = len(sample_list_of_runs)
        bad_list_of_runs = None
        if len(list_sample_indexes_with_none_in_angles) > 0:
            bad_list_of_runs = [os.path.basename(short_sample_list_of_runs[_index]) for 
                                _index in list_sample_indexes_with_none_in_angles]

        for index in list_sample_indexes_with_none_in_angles[::-1]:
            del sample_list_proton_charge[index]
            del sample_list_frame_number[index]
            del sample_list_of_angles[index]
            del sample_list_of_nexus[index]
            del sample_list_of_runs[index]
        size_after = len(sample_list_of_runs)

        # cleanup list
        if None in sample_list_frame_number:
            sample_list_frame_number = []

        if None in ob_list_frame_number:
            ob_list_frame_number = []

        self.configuration.list_of_sample_pc = sample_list_proton_charge
        self.configuration.list_of_sample_frame_number = sample_list_frame_number                                            
        self.configuration.list_of_sample_runs = short_sample_list_of_runs
        self.configuration.list_of_ob_frame_number = ob_list_frame_number
        self.configuration.list_of_angles = sample_list_of_angles
        self.configuration.list_of_ob_pc = ob_list_proton_charge
        self.configuration.list_of_ob_runs = short_ob_list_of_runs

        if not (bad_list_of_runs is None):
            print(f"The program automatically removed {size_before - size_after} bad runs!")
            print(f"{bad_list_of_runs = }")
        else:
            print(f"All selected runs are good to go!")

        # load 1 image to get size of it
        list_of_tif = retrieve_list_of_tif(os.path.join(self.configuration.top_folder.sample, sample_list_of_runs[0]))
        image = load_data_using_multithreading([list_of_tif[0]])
        _, height, width = np.shape(image)
        self.image_size = {'height': height,
                           'width': width}

    def select_mode(self):

        white_beam_layout = widgets.Label("White beam mode selected!")
        self.tof_layout = widgets.VBox([widgets.HTML("<b>Select up to 5 TOF ranges!</b>"),          
                                    widgets.IntRangeSlider(value=[0, self.nbr_images-1],
                                            min=0,
                                            max=self.nbr_images-1,
                                            step=1,
                                            description="Range #1",
                                            layout=widgets.Layout(width="100%")),
                                    widgets.IntRangeSlider(value=[0, 0],
                                            min=0,
                                            max=self.nbr_images-1,
                                            step=1,
                                            description="Range #2",
                                            layout=widgets.Layout(width="100%")),
                                    widgets.IntRangeSlider(value=[0, 0],
                                            min=0,
                                            max=self.nbr_images-1,
                                            step=1,
                                            description="Range #3",
                                            layout=widgets.Layout(width="100%")),
                                    widgets.IntRangeSlider(value=[0, 0],
                                            min=0,
                                            max=self.nbr_images-1,
                                            step=1,
                                            description="Range #4",
                                            layout=widgets.Layout(width="100%")),
                                    widgets.IntRangeSlider(value=[0, 0],
                                            min=0,
                                            max=self.nbr_images-1,
                                            step=1,
                                            description="Range #5",
                                            layout=widgets.Layout(width="100%")),
                                 ])

        self.tabs = widgets.Tab()
        self.tabs.children = [white_beam_layout, self.tof_layout]
        self.tabs.titles = ['White beam', 'TOF']
        display(self.tabs)

    def cleanup_pixels_options(self):
        self.histo_ui = widgets.Checkbox(value=False,
                                         description="Histogram")
        self.nbr_bins = widgets.IntSlider(min=10,
                                    max=1000,
                                    value=10,
                                    description='Nbr bins',
                                    continuous_update=False)
        self.nbr_exclude = widgets.IntSlider(min=0,
                                        max=10,
                                        value=1,
                                        description='Bins to excl.',
                                        continuous_update=False,
                                        )
        vertical_box = widgets.VBox([self.histo_ui,
                                     self.nbr_bins,
                                     self.nbr_exclude])

        hr = widgets.HTML("<hr>")
        self.tomo_ui = widgets.Checkbox(value=False,
                                        description="Threshold")
        v_box = widgets.VBox([vertical_box, hr, self.tomo_ui])
        display(v_box)

    def normalization_settings(self):
        self.use_proton_charge_ui = widgets.Checkbox(value=True,
                                                description='Use proton charge',
                                                disabled=self.at_least_one_run_with_no_pc_info)
        self.use_frames_ui = widgets.Checkbox(value=False,
                                         description='Use frames',
                                         disabled=self.at_least_one_frame_number_not_found,
                                         )
        self.use_roi_ui = widgets.Checkbox(value=True,
                                      description='Use ROI')
        vertical_layout = widgets.VBox([self.use_proton_charge_ui,
                                        self.use_frames_ui,
                                        self.use_roi_ui])
        display(vertical_layout)

        self.left_ui = widgets.IntText(value=roi['left'],
                                       description="left")
        self.right_ui = widgets.IntText(value=roi['right'],
                                        description='right')
        self.top_ui = widgets.IntText(value=roi['top'],
                                      description='top')
        self.bottom_ui = widgets.IntText(value=roi['bottom'],
                                         description='bottom')
        vertical_box = widgets.VBox([self.left_ui,
                                     self.right_ui,
                                     self.top_ui,
                                     self.bottom_ui])
        display(vertical_box)

    def remove_stripes(self):
        self.o_remove = RemoveStrips(parent=self)
        self.o_remove.select_algorithms()

    def define_settings(self):
        self.o_remove.define_settings()

    def center_of_rotation_flag(self):
        self.center_of_rotation_ui = widgets.Checkbox(value=False,
                                                 description="Calculate center of rotation")
        label = widgets.Label("Define slice range to calculate center of rotation and reconstruct")
        self.range = widgets.IntRangeSlider(value=[0,self.image_size['height']-1],
                                            min=0,
                                            max=self.image_size['height']-1,
                                            orientation='vertical',
                                            description='slices range')
        display(widgets.VBox([self.center_of_rotation_ui,
                              widgets.HTML("<hr>"),
                              label,
                              self.range]))

    def set_svmbir_settings(self):
        self.sharpness_ui = widgets.FloatSlider(min=0,
                                        max=1,
                                        value=0,
                                        description="sharpness")
        self.snr_db_ui = widgets.FloatSlider(min=0,
                                        max=100,
                                        value=30.0,
                                        description="snr db")
        self.positivity_ui = widgets.Checkbox(value=True,
                                            description="positivity")
        self.max_iterations_ui = widgets.IntSlider(value=200,
                                                min=10,
                                                max=500,
                                                description="max itera.")
        self.verbose_ui = widgets.Checkbox(value=False,
                                        description='verbose')
        
        vertical_widgets = widgets.VBox([self.sharpness_ui,
                                            self.snr_db_ui,
                                            self.positivity_ui,
                                            self.max_iterations_ui,
                                            self.verbose_ui])
        display(vertical_widgets)

    def select_output_folder(self):
        working_dir = os.path.dirname(self.configuration.top_folder.sample)
        o_file_browser = FileFolderBrowser(working_dir=working_dir,
                                           next_function=self.save_output_folder)
        o_file_browser.select_input_folder(instruction=f"Select output folder",
                                           multiple_flag=False)

    def save_output_folder(self, output_folder):
        self.configuration.output_folder = output_folder
        print(f"Reconstructed data will be exported to {output_folder}")

    def export_configuration(self):
        working_dir = os.path.dirname(self.configuration.top_folder.sample)
        o_file_browser = FileFolderBrowser(working_dir=working_dir,
                                           next_function=self.save_export_configuration)
        o_file_browser.select_input_folder(instruction=f"Select where to save the configuration file",
                                           multiple_flag=False)

    def save_export_configuration(self, output_folder):

        base_folder = os.path.basename(os.path.abspath(self.configuration.top_folder.sample))
        _time = get_current_time_in_special_file_name_format()
        config_file_name = os.path.join(output_folder, f"config_{base_folder}_{_time}.json")

        if self.tabs.selected_index:
            self.configuration.operating_mode = OperatingMode.white_beam
        else:
            self.configuration.operating_mode = OperatingMode.tof
            list_of_ranges = [self.tof_layout.children[1].value,
                              self.tof_layout.children[2].value,
                              self.tof_layout.children[3].value,
                              self.tof_layout.children[4].value,
                              self.tof_layout.children[5].value]
            clean_list_of_ranges = []
            for _range in list_of_ranges:
                if _range[0]==0 and _range[1]==0:
                    continue
                clean_list_of_ranges.append(_range)
            self.configuration.range_of_tof_to_combine = clean_list_of_ranges

        list_clean_algorithm = []
        if self.histo_ui.value:
            list_clean_algorithm.append(CleaningAlgorithm.histogram)
            self.configuration.histogram_cleaning_settings.nbr_bins = self.nbr_bins.value
            self.configuration.histogram_cleaning_settings.bins_to_exclude = self.nbr_exclude.value

        if self.tomo_ui.value:
            list_clean_algorithm.append(CleaningAlgorithm.threshold)

        self.configuration.list_clean_algorithm = list_clean_algorithm

        list_normalization_settings = []
        if (not self.at_least_one_run_with_no_pc_info) and self.use_proton_charge_ui.value:
            list_normalization_settings.append(NormalizationSettings.pc)

        if (not self.at_least_one_frame_number_not_found) and self.use_frames_ui.value:
            list_normalization_settings.append(NormalizationSettings.frame_number)

        if self.use_roi_ui.value:
            list_normalization_settings.append(NormalizationSettings.roi)

        self.configuration.normalization_roi.left = self.left_ui.value
        self.configuration.normalization_roi.right = self.right_ui.value
        self.configuration.normalization_roi.top = self.top_ui.value
        self.configuration.normalization_roi.bottom = self.bottom_ui.value

        # remove stripes
        list_algo_to_use = self.o_remove.list_to_use_widget.options
        if list_algo_to_use:
            for _algo in list_algo_to_use:
                # kwargs = self.get_keyword_arguments(algorithm_name=_algo)
                self.o_remove.saving_configuration(algorithm_name=_algo)

        # center of rotation
        center_of_rotation_flag = self.center_of_rotation_ui.value
        self.configuration.calculate_center_of_rotation = center_of_rotation_flag

        slice_range = self.range.value
        self.configuration.range_of_slices_for_center_of_rotation = list(slice_range)
        
        # svmbir settings
        self.configuration.svmbir_config.sharpness = self.sharpness_ui.value
        self.configuration.svmbir_config.snr_db = self.snr_db_ui.value
        self.configuration.svmbir_config.positivity = self.positivity_ui.value
        self.configuration.svmbir_config.max_iterations = self.max_iterations_ui.value
        self.configuration.svmbir_config.verbose = self.verbose_ui.value
        self.configuration.svmbir_config.top_slice = np.min(slice_range)
        self.configuration.svmbir_config.bottom_slice = np.max(slice_range)

        config_json = self.configuration.model_dump_json()
        save_json(config_file_name, json_dictionary=config_json)
        print(f"config file {config_file_name}")
