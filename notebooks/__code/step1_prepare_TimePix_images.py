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
    list_of_runs_to_use = None
    
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
        o_check = CheckingData(parent=self)
        o_check.checking_minimum_requirements()
        if self.minimum_requirements_met:           
            o_combine = CombineTof(parent=self)
            o_combine.run()
        else:
            o_check.minimum_requirement_not_met()

    # cleaning low/high pixels
    def clean_images_setup(self):
        o_clean = ImagesCleaner(parent=self)
        o_clean.cleaning_setup()

    def clean_images(self):
        o_clean = ImagesCleaner(parent=self)
        o_clean.cleaning()

    def select_export_folder(self):
        o_clean = ImagesCleaner(parent=self)
        o_clean.select_export_folder()

    def export_cleaned_images(self):
        o_clean = ImagesCleaner(parent=self)
        o_clean.export_clean_images()

    # normalization
    def normalization_settings(self):
        self.o_norm = Normalization(parent=self)
        self.o_norm.normalization_settings()

    def normalization_select_roi(self):
        self.o_norm.select_roi()

    def normalization(self):
        self.o_norm.run()

    def visualization_normalization_settings(self):
        self.o_norm.visualization_normalization_settings()

    def visualize_normalization(self):
        self.o_norm.visualize_normalization()

    def select_export_normalized_folder(self):
        o_select = Load(parent=self)
        o_select.select_folder(data_type=DataType.normalized)

    def export_normalized_images(self):
        self.o_norm.export_images()

    # chips correction
    def chips_correction(self):
        o_chips = ChipsCorrection(parent=self)
        o_chips.run()

    def visualize_chips_correction(self):
        o_chips = ChipsCorrection(parent=self)
        o_chips.visualize_chips_correction()

    # strips removal
    def select_remove_strips_algorithms(self):
        self.o_remove = RemoveStrips(parent=self)
        self.o_remove.select_algorithms()

    def define_settings(self):
        self.o_remove.define_settings()

    def remove_strips_and_display(self):
        self.o_remove.run()

    # calculate center of rotation & tilt
    def select_sample_roi(self):
        if self.strip_corrected_images is None:
            # if the remove filter hasn't been ran
            self.strip_corrected_images = self.corrected_images

        self.o_center_and_tilt = CenterOfRotationAndTilt(parent=self)
        self.o_center_and_tilt.select_range()

    def calculate_center_of_rotation_and_tilt(self):
        self.o_center_and_tilt.run()

    # last chance to reject runs
    def final_projections_review(self):
        o_review = FinalProjectionsReview(parent=self)
        o_review.run(array=self.corrected_images)
        o_review.list_runs_to_reject()

    # run svmbir
    def svmbir_settings(self):
        self.o_svmbir = SvmbirHandler(parent=self)
        self.o_svmbir.set_settings()

    def svmbir_display_sinograms(self):
        self.o_svmbir.display_sinograms()

    def svmbir_run(self):
        self.o_svmbir.run_reconstruction()
        self.o_svmbir.display_slices()

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
        o_select.select_folder(data_type=DataType.extra)

    def export_extra_files(self):
        o_export = ExportExtra(parent=self)
        o_export.run(base_log_file_name=LOG_BASENAME_FILENAME)
