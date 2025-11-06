import os
import logging
from loguru import logger as loguru_logging
from collections import OrderedDict
import numpy as np
from IPython.display import display, HTML

from __code import DataType, OperatingMode, DEFAULT_OPERATING_MODE
from __code.utilities.logging import setup_logging
from __code.utilities.configuration_file import Configuration
from __code.config import debugging as DEBUG
from __code.workflow.load import Load
from __code.workflow.combine_ob_dc import CombineObDc
from __code.workflow.checking_data import CheckingData
from __code.workflow.recap_data import RecapData
from __code.workflow.mode_selection import ModeSelection
from __code.workflow.reconstruction_selection import ReconstructionSelection
from __code.workflow.images_cleaner import ImagesCleaner
from __code.workflow.rebin import Rebin
from __code.workflow.normalization import Normalization
from __code.workflow.chips_correction import ChipsCorrection
from __code.workflow.log_conversion import log_conversion
from __code.workflow.data_handler import remove_negative_values, remove_0_values
from __code.workflow.center_of_rotation_and_tilt import CenterOfRotationAndTilt
from __code.workflow.remove_strips import RemoveStrips
from __code.workflow.svmbir_handler import SvmbirHandler
from __code.workflow.fbp_handler import FbpHandler
from __code.workflow.final_projections_review import FinalProjectionsReview
from __code.workflow.export import ExportExtra
from __code.workflow.visualization import Visualization
from __code.workflow.rotate import Rotate
from __code.workflow.crop import Crop
from __code.workflow.test_reconstruction import TestReconstruction
from __code.utilities.configuration_file import ReconstructionAlgorithm
from __code.utilities.logging import logging_3d_array_infos
from __code.utilities.exceptions import MetadataError

LOG_BASENAME_FILENAME, _ = os.path.splitext(os.path.basename(__file__))


class Step1PrepareCcdImages:

    MODE = OperatingMode.white_beam

    working_dir = {
        DataType.sample: "",
        DataType.ob: "",
        DataType.nexus: "",
        DataType.cleaned_images: "",
        DataType.normalized: "",
        DataType.processed: "",
        }
    
    operating_mode = DEFAULT_OPERATING_MODE

    histogram_sample_before_cleaning = None

    center_of_rotation = None
    
    crop_region = {'left': None,
                   'right': None,
                   'top': None,
                   'bottom': None}

    image_size = {'height': None,
                  'width': None}

    spectra_file_full_path = None

    final_dict_of_pc = {}
    final_dict_of_frame_number = {}

    # naming schema convention
    list_states_checkbox = None

    # used to displya profile vs lambda in TOF mode
    # np.shape of y, x, tof
    data_3d_of_all_projections_merged = None

    # will record short_run_number and pc
    # will look like
    # {DataType.sample: {'Run_1234': {Run.full_path: "/SNS/VENUS/.../Run_1344",
    #                                 Run.proton_charge: 5.01,
    #                                 Ru.use_it: True,
    #                                },
    #                    ...,
    #                   },
    # DataType.ob: {...},
    # }
    list_of_images = {DataType.sample: OrderedDict(),
                      DataType.ob: None,
                      DataType.dc: None,
                    }
    
    list_of_runs_checking_data = {DataType.sample: {},
                                   DataType.ob: {},
                                  }

    list_proton_charge_c = {DataType.sample: {},
                            DataType.ob: {},
                           }

    final_list_of_runs = {DataType.sample: {},
                          DataType.ob: {},
                          }

    final_list_of_angles = None
    list_of_runs_to_use = None
    
    # set up in the checking_data. True if at least one of the run doesn't have this metadata in the NeXus
    at_least_one_frame_number_not_found = False

    # dictionary used just after loading the data, not knowing the mode yet
    # or the tiff images in using the white beam notebook
    master_3d_data_array = {DataType.sample: None,  # [angle, y, x]
                            DataType.ob: None,
                            DataType.dc: None}
    
    # each element of the dictionary is an master_3d_data_array of each TOF range
    # {'0': {'use_it': True,
    #         'data': master_3d_data_array, 
    #       },
    # '1': {'use_it': False,
    #       'data': master_3d_data_array,
    #       },
    #  ...
    #}
    # this is the master dictionary used no matter the mode
    master_tof_3d_data_array = None

    # master_3d_data_array_cleaned = {DataType.sample: None,  # [angle, y, x]
    #                                 DataType.ob: None}

    normalized_images = None   # after normalization
    normalized_images_log = None   # after log conversion
    corrected_images = None  # after chips correction
    sinogram_normalized_images_log = None  # sinograms after log conversion
    
    instrument = "VENUS"
    ipts = None

    selection_of_pc = None   # plot that allows the user to select the pc for sample and ob and threshold

    list_of_sample_runs_to_reject_ui = None
    list_of_ob_runs_to_reject_ui = None
    minimum_requirements_met = False

    # created during the combine step to match data index with run number (for normalization)
    list_of_runs_to_use = {DataType.sample: [],
                           DataType.ob:[]}
    
    list_of_angles_to_use_sorted = None
    retrieve_angle_value_from_metadata = False

    strip_corrected_images = None # Array 3D after strip correction

    before_rebinning = None # only if rebin is ran, will change

    # center of rotation
    o_center_and_tilt = None
    # remove strips
    o_remove = None
    # remove outliers
    o_clean = None
    # normalization
    o_norm = None
    # tilt
    o_tilt = None
    # svmbir 
    o_svmbir = None
    # tof mode
    o_tof_range_mode = None

    # widget multi selection - list of runs to exclude before running svmbir
    runs_to_exclude_ui = None

    # reconstructed 3D array with svmbir
    reconstruction_array = None

    def __init__(self, system=None, ):

        self.configuration = Configuration()

        # o_init = Initialization(parent=self)
        # o_init.configuration()

        top_sample_dir = system.System.get_working_dir()
        self.instrument = system.System.get_instrument_selected()
        self.ipts_number = system.System.get_ipts_number()

        setup_logging(basename_of_log_file=LOG_BASENAME_FILENAME)        
        self.working_dir[DataType.ipts] = top_sample_dir
        self.working_dir[DataType.top] = os.path.join(top_sample_dir)
        self.working_dir[DataType.sample] = os.path.join(top_sample_dir, "raw", "ct_scans")
        self.working_dir[DataType.ob] = os.path.join(top_sample_dir, "raw", "ob")
        self.working_dir[DataType.dc] = os.path.join(top_sample_dir, "raw", "dc")
        self.working_dir[DataType.nexus] = os.path.join(top_sample_dir, "nexus")
        self.working_dir[DataType.processed] = os.path.join(top_sample_dir, "shared", "processed_data")
        logging.info(f"working_dir: {self.working_dir}")
        logging.info(f"instrument: {self.instrument}")
        if DEBUG:
            logging.info(f"WARNING!!!! we are running using DEBUG mode!")

    # Selection of data
    def select_top_sample_folder(self):
        """updates: list_of_images[DataType.sample]"""
        o_load = Load(parent=self)
        o_load.select_folder(data_type=DataType.sample)

    def select_ob_images(self):
        """updates: list_of_images[DataType.ob]"""
        o_load = Load(parent=self)
        o_load.select_images(data_type=DataType.ob)

    def select_dc_images(self):
        """updates: list_of_images[DataType.dc]"""
        o_load = Load(parent=self)
        o_load.select_images(data_type=DataType.dc)

    def how_to_retrieve_angle_value(self):
        self.o_load = Load(parent=self)
        self.o_load.how_to_retrieve_angle_value()

    def retrieve_angle_value(self):
        self.o_load.retrieve_angle_value()

    def testing_angle_values(self):
        self.o_load.testing_angle_values()

    # # define naming convention to easily extract angle value
    # def define_naming_schema(self):
    #     self.o_load = Load(parent=self)
    #     self.o_load.define_naming_convention()

    # pecentage of data to use
    def use_all_or_fraction(self):
        try:
            self.o_load.use_all_or_fraction()
        except MetadataError as e:
            logging.error(f"MetadataError: {e.message}")
            display(widgets.HTML(f"<font color='red'><b>ERROR</b>: {e.message}</font>"))
  
    def select_percentage_of_data_to_use(self):
        self.o_load.select_percentage_of_data_to_use()

    # load data
    def load_data(self):
        """creates: master_3d_data_array
           updates: list_of_images
           creates: final_list_of_angles
           creates: final_list_of_angles_rad
        """
        self.o_load.load_white_beam_data()
        
    # visualization
    def how_to_visualize(self):
        self.o_vizu = Visualization(parent=self)
        self.o_vizu.how_to_visualize()

    def visualize_raw_data(self):
        """uses: master_3d_data_array"""
        self.o_vizu.visualize_according_to_selection(mode='raw')

    # pre processing crop
    def pre_processing_crop_settings(self):
        self.o_crop1 = Crop(parent=self)
        self.o_crop1.set_region(before_normalization=True)

    def pre_processing_crop(self):
        """updates: master_3d_data_array"""
        self.o_crop1.run()

    # cleaning low/high pixels - remove outliers
    def clean_images_settings(self):
        self.o_clean = ImagesCleaner(parent=self)
        self.o_clean.settings()

    def clean_images_setup(self):
        self.o_clean.cleaning_setup()

    def clean_images(self):
        """updates: master_3d_data_array"""
        self.o_clean.cleaning()

    def how_to_visualize_after_cleaning(self):
        self.o_vizu = Visualization(parent=self)
        self.o_vizu.how_to_visualize(data_type=DataType.cleaned_images)

    def visualize_cleaned_data(self):
        self.o_vizu.visualize_according_to_selection(mode='cleaned')
 
    # normalization
    def normalization_settings(self):
        self.o_norm = Normalization(parent=self)
        self.o_norm.normalization_settings()

    def normalization_select_roi(self):
        self.o_norm.select_roi()

    def normalization(self):
        """creates: normalized_images"""
        o_combine = CombineObDc(parent=self)
        o_combine.run()
        self.o_norm.normalize()

    def visualization_normalization_settings(self):
        self.o_vizu = Visualization(parent=self)
        self.o_vizu.settings()

    def visualize_normalization(self):
        self.o_vizu.visualize(data_after=self.normalized_images,
                              label_before='cleaned',
                              label_after='normalized',
                              data_before=self.master_3d_data_array[DataType.sample],
                              turn_on_vrange=True)
    
    def select_export_normalized_folder(self):
        o_select = Load(parent=self)
        o_select.select_folder(data_type=DataType.normalized)

    def export_normalized_images(self):
        self.o_norm.export_images()
    # rebin
    def rebin_settings(self):
        self.o_rebin = Rebin(parent=self)
        self.o_rebin.set_rebinning()

    def rebin_before_normalization(self):
        """ modifies: master_3d_data_array"""
        self.o_rebin.execute_binning_before_normalization()

    def rebin_after_normalization(self):
        """ modifies: normalized_images"""
        self.o_rebin.execute_binning_after_normalization()

    def visualize_rebinned_data(self, before_normalization=False):
        if self.before_rebinning is None:
            logging.warning(f"before_rebinning is None, cannot visualize rebinned data!")
            print("No rebinning, nothing to visualize!")
            return

        if before_normalization:
            data_after = self.master_3d_data_array[DataType.sample]
            data_before = self.before_rebinning
           
            self.o_vizu.visualize(data_after=data_after,
                                 label_before='raw',
                                 label_after='rebinned',
                                 data_before=data_before,
                                 turn_on_vrange=True,
            )

        else:
            data_after = self.normalized_images
            data_before = self.before_rebinning
            vmin = 0
            vmax = 1
            vmin_after = 0
            vmax_after = 1
        
            self.o_vizu.visualize(data_after=data_after,
                            label_before='raw',
                            label_after='rebinned',
                            data_before=data_before,
                            turn_on_vrange=True,
                            vmin=vmin,
                            vmax=vmax,
                            vmin_after=vmin_after,
                            vmax_after=vmax_after)

  # crop data
    def crop_settings(self):
        self.o_crop = Crop(parent=self)
        self.o_crop.set_region()

    def crop(self):
        """updates: normalized_images"""
        self.o_crop.run()

    # rotate sample
    def is_rotation_needed(self):
        self.o_rotate = Rotate(parent=self)
        self.o_rotate.is_rotation_needed()

    def rotate_data_settings(self):
        self.o_rotate = Rotate(parent=self)
        self.o_rotate.set_settings()

    def apply_rotation(self):
        """updates: normalized_images"""
        self.o_rotate.apply_rotation()

    def visualize_after_rotation(self):
        o_review = FinalProjectionsReview(parent=self)
        o_review.stack_of_images(array=self.normalized_images[:])

    # log conversion
    def log_conversion_and_cleaning(self):
        """creates: corrected_images_log
        """
        normalized_images_log = log_conversion(self.normalized_images[:])
        o_cleaner = ImagesCleaner(parent=self)
        normalized_images_log = o_cleaner.remove_outliers(normalized_images_log[:])
        normalized_images_log = remove_negative_values(normalized_images_log[:])

        # self.corrected_images_log = normalized_images_log[:]
        self.normalized_images_log = normalized_images_log[:]
        logging_3d_array_infos(array=normalized_images_log, message="normalized_images_log")

    def visualize_images_after_log(self):
        o_vizu = Visualization(parent=self)
        o_vizu.visualize_2_stacks(left=self.normalized_images, 
                                  vmin_left=0, 
                                  vmax_left=1,
                                  right=self.normalized_images_log,
                                  vmin_right=None,
                                  vmax_right=None,)
        
    # strips removal
    def select_range_of_data_to_test_stripes_removal(self):
        """updates: list_of_images[DataType.sample]"""
        self.o_remove = RemoveStrips(parent=self)
        self.o_remove.select_range_of_data_to_test_stripes_removal()

    def select_remove_strips_algorithms(self):
        self.o_remove.select_algorithms()

    def define_settings(self):
        self.o_remove.define_settings()

    def test_algorithms_on_selected_range_of_data(self):
        """updates: strip_corrected_images"""
        self.o_remove.perform_cleaning(test=True)
        self.o_remove.display_cleaning(test=True)

    def when_to_remove_strips(self):
        """updates: normalized_images_log"""
        self.o_remove.when_to_remove_strips()

    def remove_strips(self):
        """updates: normalized_images_log"""
        self.o_remove.perform_cleaning()

    def display_removed_strips(self):
        self.o_remove.display_cleaning()

    # calculate and apply tilt
    def select_sample_roi(self):
        self.o_tilt = CenterOfRotationAndTilt(parent=self)
        self.o_tilt.select_range()

    def perform_tilt_correction(self):
        """updates: normalized_images_log"""
        self.o_tilt.run_tilt_correction()

    # calcualte center of rotation
    def center_of_rotation_settings(self):       
        if self.o_tilt is None:
            self.o_tilt = CenterOfRotationAndTilt(parent=self)
        self.o_tilt.isolate_0_180_360_degrees_images()
        self.o_tilt.center_of_rotation_settings()

    def run_center_of_rotation(self):
        """uses: normalized_images_log"""
        self.o_tilt.run_center_of_rotation()

    def determine_center_of_rotation(self):
        self.o_tilt.calculate_center_of_rotation()

    def display_center_of_rotation(self):
        self.o_tilt.test_center_of_rotation_calculated()

    # sinograms
    def create_sinograms(self):
        """creates: sinogram_normalized_images_log"""
        self.sinogram_normalized_images_log = np.moveaxis(self.normalized_images_log, 1, 0)

    def visualize_sinograms(self):
        if self.sinogram_normalized_images_log is None:
            self.create_sinograms()

        logging.debug(f"sinogram_normalized_images_log shape: {self.sinogram_normalized_images_log.shape}")

        o_vizu = Visualization(parent=self)
        o_vizu.visualize_1_stack(data=self.sinogram_normalized_images_log,
                                 title="Sinograms")
    # test reconstruction using gridrec (fast algorithm)
    def select_slices_to_use_to_test_reconstruction(self):
        """uses: normalized_images_log"""
        self.o_test = TestReconstruction(parent=self)
        self.o_test.select_slices()

    def run_reconstruction_of_slices_to_test(self):
        self.o_test.run_reconstruction()

    # select reconstruction method
    def select_reconstruction_method(self):
        self.o_mode = ReconstructionSelection(parent=self)
        self.o_mode.select()

    # run svmbir
    def reconstruction_settings(self):
        # if self.corrected_images is None:
        #     self.corrected_images = self.normalized_images_log
        
        if (ReconstructionAlgorithm.svmbir in self.configuration.reconstruction_algorithm) or \
           (ReconstructionAlgorithm.mbirjax in self.configuration.reconstruction_algorithm):
            self.o_svmbir = SvmbirHandler(parent=self)
            self.o_svmbir.set_settings()
         
    # takes for ever !
    # def svmbir_display_sinograms(self):
    #     self.o_svmbir.display_sinograms()

    def svmbir_run(self):
        self.o_svmbir.run_reconstruction()
        self.o_svmbir.display_slices()

    # # run the CLI version from the pre-reconstructed data
    # def svmbir_run_cli(self, config_json_file, input_data_folder, output_data_folder):
    #     SvmbirCliHandler.run_reconstruction_from_pre_data_mode(config_json_file, 
    #                                                         input_data_folder, 
    #                                                         output_data_folder)

    # def display_slices(self):
    #     self.o_svmbir.display_slices()

    # export slices
    def select_export_slices_folder(self):
        o_select = Load(parent=self)
        o_select.select_folder(data_type=DataType.reconstructed)

    def export_slices(self):
        self.o_svmbir.export_images()

    # export extra files
    def select_export_extra_files(self):
        o_select = Load(parent=self)
        o_select.select_folder(data_type=DataType.extra,
                               output_flag=True)

    def export_pre_reconstruction_data(self):
        # if self.o_svmbir is None:
        o_fbp = FbpHandler(parent=self)
        o_fbp.export_pre_reconstruction_data()
        # else:
        #     self.o_svmbir.export_pre_reconstruction_data()

    def export_extra_files(self, prefix=""):
        self.export_pre_reconstruction_data()
        o_export = ExportExtra(parent=self)
        o_export.update_configuration()
        o_export.run(base_log_file_name=LOG_BASENAME_FILENAME,
                     prefix=prefix)
        
    @classmethod
    def legend(cls) -> None:
        display(HTML("<hr style='height:2px'/>"))
        display(HTML("<h2>Legend</h2>"))
        display(HTML("<ul>"
                     "<li><b><font color='red'>Mandatory steps</font></b> must be performed to ensure proper data preparation and reconstruction.</li>"
                     "<li><b><font color='orange'>Optional but recommended steps</font></b> are not mandatory but should be performed to ensure proper data preparation and reconstruction.</li>"
                     "<li><b><font color='purple'>Optional steps</font></b> are not mandatory but highly recommended to improve the quality of your reconstruction.</li>"
                     "</ul>"))
        display(HTML("<hr style='height:2px'/>"))
