import os
import logging
from collections import OrderedDict


from __code import DataType, OperatingMode, DEFAULT_OPERATING_MODE, DEBUG
from __code.utilities.logging import setup_logging
from __code.utilities.configuration_file import Configuration

from __code.workflow.load import Load
from __code.workflow.checking_data import CheckingData
from __code.workflow.recap_data import RecapData
from __code.workflow.combine_tof import CombineTof
from __code.workflow.images_cleaner import ImagesCleaner
from __code.workflow.normalization import Normalization
from __code.workflow.chips_correction import ChipsCorrection
from __code.workflow.center_of_rotation_and_tilt import CenterOfRotationAndTilt
from __code.workflow.remove_strips import RemoveStrips
from __code.workflow.svmbir_handler import SvmbirHandler
from __code.workflow.final_projections_review import FinalProjectionsReview
from __code.workflow.export import ExportExtra
from __code.workflow.visualization import Visualization
from __code.workflow.crop import Crop
from __code.workflow.combine_ob_dc import CombineObDc
from __code.workflow.mode_selection import ModeSelection
from __code.workflow.reconstruction_selection import ReconstructionSelection
from __code.workflow.rebin import Rebin
from __code.workflow.log_conversion import log_conversion
from __code.workflow.data_handler import remove_negative_values, remove_0_values
from __code.workflow.fbp_handler import FbpHandler
from __code.workflow.rotate import Rotate
from __code.workflow.test_reconstruction import TestReconstruction
from __code.utilities.configuration_file import ReconstructionAlgorithm
from __code.utilities.logging import logging_3d_array_infos


LOG_BASENAME_FILENAME, _ = os.path.splitext(os.path.basename(__file__))


class Step1PrepareTimePixImages:

    MODE = OperatingMode.tof

    working_dir = {
        DataType.sample: "",
        DataType.ob: "",
        DataType.nexus: "",
        DataType.cleaned_images: "",
        DataType.normalized: "",
        DataType.processed: "",
        }
    
    # {100.000: 'run_1234', 101.000: 'run_1235', ...}
    list_angles_deg_vs_runs_dict = {}

    # final list of angles used and sorted (to be used in reconstruction)
    list_angles_of_data_loaded_deg = None

    image_size = {'height': None,
                  'width': None}

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
    list_of_runs = {DataType.sample: OrderedDict(),
                    DataType.ob: OrderedDict(),
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
    
    # set up in the checking_data. True if at least one of the run doesn't have this metadata in the NeXus
    at_least_one_frame_number_not_found = False

    master_3d_data_array = {DataType.sample: None,  # [angle, y, x]
                            DataType.ob: None}

    master_3d_data_array_cleaned = {DataType.sample: None,  # [angle, y, x]
                                    DataType.ob: None}

    normalized_images = None   # after normalization
    corrected_images = None  # after chips correction

    instrument = "VENUS"

    selection_of_pc = None   # plot that allows the user to select the pc for sample and ob and threshold

    list_of_sample_runs_to_reject_ui = None
    list_of_ob_runs_to_reject_ui = None
    minimum_requirements_met = False

    # created during the combine step to match data index with run number (for normalization)
    list_of_runs_to_use = {DataType.sample: [],
                           DataType.ob:[]}
    list_of_angles_to_use_sorted = None

    strip_corrected_images = None # Array 3D after strip correction

    # center of rotation
    o_center_and_tilt = None
    # remove strips
    o_remove = None
    # normalization
    o_norm = None
    # svmbir 
    o_svmbir = None

    # widget multi selection - list of runs to exclude before running svmbir
    runs_to_exclude_ui = None

    # reconstructed 3D array with svmbir
    reconstruction_array = None

    at_least_one_frame_number_not_found = False
    at_lest_one_proton_charge_not_found = False

    def __init__(self, system=None):

        self.configuration = Configuration()

        top_sample_dir = system.System.get_working_dir()
        self.instrument = system.System.get_instrument_selected()

        setup_logging(basename_of_log_file=LOG_BASENAME_FILENAME)        
        self.working_dir[DataType.ipts] = os.path.basename(top_sample_dir)
        self.working_dir[DataType.top] = os.path.join(top_sample_dir, "shared", "autoreduce", "mcp")
        self.working_dir[DataType.nexus] = os.path.join(top_sample_dir, "nexus")
        self.working_dir[DataType.processed] = os.path.join(top_sample_dir, "shared", "processed_data")
        logging.info(f"working_dir: {self.working_dir}")
        logging.info(f"instrument: {self.instrument}")
        if DEBUG:
            logging.info(f"WARNING!!!! we are running using DEBUG mode!")

    # Selection of data
    def select_top_sample_folder(self):
        o_load = Load(parent=self)
        o_load.select_folder(data_type=DataType.sample)

    def select_top_ob_folder(self):
        o_load = Load(parent=self)
        o_load.select_folder(data_type=DataType.ob)

    # Checking data (proton charge, empty runs ...)
    def load_and_check_data(self):
        try:
            o_checking = CheckingData(parent=self)
            o_checking.run()
        except ValueError:
            logging.info("Check the input folders provided !")

    def recap_data(self):
        o_recap = RecapData(parent=self)
        o_recap.run()

    # def checkin_data_entries(self):
    #     o_check = CheckingData(parent=self)
    #     o_check.checking_minimum_requirements()

    # combine images
    def combine_images(self):
        """creates the master_3d_data_array, final_list_of_runs, final_list_of_angles and final_list_of_angles_rad"""
        o_check = CheckingData(parent=self)
        o_check.checking_minimum_requirements()
        if self.minimum_requirements_met:           
            o_combine = CombineTof(parent=self)
            o_combine.run()
        else:
            o_check.minimum_requirement_not_met()

    # visualization
    def how_to_visualize(self):
        self.o_vizu = Visualization(parent=self)
        self.o_vizu.how_to_visualize()

    def visualize_raw_data(self):
        """uses: master_3d_data_array"""
        self.o_vizu.visualize_timepix_according_to_selection(mode='raw')

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
        self.o_clean.cleaning(ignore_dc=True)

    def how_to_visualize_after_cleaning(self):
        self.o_vizu = Visualization(parent=self)
        self.o_vizu.how_to_visualize(data_type=DataType.cleaned_images)

    def visualize_cleaned_data(self):
        self.o_vizu.visualize_timepix_according_to_selection(mode='cleaned')
   
    # normalization
    def normalization_settings(self):
        self.o_norm = Normalization(parent=self)
        self.o_norm.normalization_settings()

    def normalization_select_roi(self):
        self.o_norm.select_roi()

    def normalization(self):
        """creates: normalized_images"""
        o_combine = CombineObDc(parent=self)
        o_combine.run(ignore_dc=True)
        self.o_norm.normalize(ignore_dc=True)

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

    # chips correction
    def chips_correction(self):
        """updates: normalized_images"""
        o_clean = ImagesCleaner(parent=self)
        self.normalized_images = o_clean.remove_outliers(self.normalized_images[:])

        o_chips = ChipsCorrection(parent=self)
        o_chips.run()

    def visualize_chips_correction(self):
        o_chips = ChipsCorrection(parent=self)
        o_chips.visualize_chips_correction()

    # rebin
    def rebin_settings(self):
        self.o_rebin = Rebin(parent=self)
        self.o_rebin.set_rebinning()

    def rebin_before_normalization(self):
        """ modifies: normalized_images"""
        self.o_rebin.execute_binning_before_normalization()

    def rebin_after_normalization(self):
        """ modifies: normalized_images"""
        self.o_rebin.execute_binning_after_normalization()

    def visualize_rebinned_data(self, before_normalization=False):
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
    def rotate_data_settings(self):
        self.o_rotate = Rotate(parent=self)
        self.o_rotate.set_settings()

    def apply_rotation(self):
        """updates: normalized_images"""
        self.o_rotate.apply_rotation()

    def visualize_after_rotation(self):
        o_review = FinalProjectionsReview(parent=self)
        o_review.single_image(image=self.normalized_images[0])

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
    def select_remove_strips_algorithms(self):
        self.o_remove = RemoveStrips(parent=self)
        self.o_remove.select_algorithms()

    def define_settings(self):
        self.o_remove.define_settings()

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

    def run_center_of_rotation_or_skip_it(self):
        self.o_tilt.calculate_center_of_rotation()

    def display_center_of_rotation(self):
        self.o_tilt.test_center_of_rotation_calculated()

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
        
        if ReconstructionAlgorithm.svmbir in self.configuration.reconstruction_algorithm:
            self.o_svmbir = SvmbirHandler(parent=self)
            self.o_svmbir.set_settings()
       
    def svmbir_run(self):
        self.o_svmbir.run_reconstruction()
        self.o_svmbir.display_slices()

    # export slices
    def select_export_slices_folder(self):
        o_select = Load(parent=self)
        o_select.select_folder(data_type=DataType.reconstructed)

    def export_slices(self):
        self.o_svmbir.export_images()

    # export extra files
    def select_export_extra_files(self):
        o_select = Load(parent=self)
        o_select.select_folder(data_type=DataType.extra)

    def export_pre_reconstruction_data(self):
        if self.o_svmbir is None:
            o_fbp = FbpHandler(parent=self)
            o_fbp.export_pre_reconstruction_data()
        else:
            self.o_svmbir.export_pre_reconstruction_data()

    def export_extra_files(self, prefix=""):
        self.export_pre_reconstruction_data()
        o_export = ExportExtra(parent=self)
        o_export.run(base_log_file_name=LOG_BASENAME_FILENAME,
                     prefix=prefix)
        