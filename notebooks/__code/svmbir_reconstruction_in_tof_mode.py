import os
import logging
from collections import OrderedDict


from __code import DataType, OperatingMode, DEFAULT_OPERATING_MODE
from __code.utilities.logging import setup_logging
from __code.utilities.configuration_file import Configuration

from __code.workflow.load import Load
from __code.workflow.checking_data import CheckingData
from __code.workflow.recap_data import RecapData
from __code.workflow.mode_selection import ModeSelection
from __code.workflow.images_cleaner import ImagesCleaner
from __code.workflow.normalization import Normalization
from __code.workflow.chips_correction import ChipsCorrection
from __code.workflow.center_of_rotation_and_tilt import CenterOfRotationAndTilt
from __code.workflow.remove_strips import RemoveStrips
from __code.workflow.svmbir_handler import SvmbirHandler
from __code.workflow.final_projections_review import FinalProjectionsReview
from __code.workflow.export import ExportExtra
from __code.workflow.visualization import Visualization

LOG_BASENAME_FILENAME = "svmbir_reconstruction_tof_mode"


class SvmbirReconstruction:

    MODE = OperatingMode.tof

    working_dir = {
        DataType.sample: "",
        DataType.ob: "",
        DataType.nexus: "",
        DataType.cleaned_images: "",
        DataType.normalized: "",
        DataType.processed: "",
        }
    
    operating_mode = DEFAULT_OPERATING_MODE

    image_size = {'height': None,
                  'width': None}

    spectra_file_full_path = None

    final_dict_of_pc = {}
    final_dict_of_frame_number = {}

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

    # dictionary used just after loading the data, not knowing the mode yet
    master_3d_data_array = {DataType.sample: None,  # [angle, y, x]
                            DataType.ob: None}
    
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
    # remove outliers
    o_clean = None
    # normalization
    o_norm = None
    # svmbir 
    o_svmbir = None
    # tof mode
    o_tof_range_mode = None

    # widget multi selection - list of runs to exclude before running svmbir
    runs_to_exclude_ui = None

    # reconstructed 3D array with svmbir
    reconstruction_array = None

    def __init__(self, system=None):

        self.configuration = Configuration()

        # o_init = Initialization(parent=self)
        # o_init.configuration()

        top_sample_dir = system.System.get_working_dir()
        self.instrument = system.System.get_instrument_selected()

        setup_logging(basename_of_log_file=LOG_BASENAME_FILENAME)        
        self.working_dir[DataType.ipts] = os.path.basename(top_sample_dir)
        self.working_dir[DataType.top] = os.path.join(top_sample_dir, "shared", "autoreduce", "mcp")
        self.working_dir[DataType.nexus] = os.path.join(top_sample_dir, "nexus")
        self.working_dir[DataType.processed] = os.path.join(top_sample_dir, "shared", "processed_data")
        logging.info(f"working_dir: {self.working_dir}")
        logging.info(f"instrument: {self.instrument}")

    # Selection of data
    def select_top_sample_folder(self):
        o_load = Load(parent=self)
        o_load.select_folder(data_type=DataType.sample)

    def select_top_ob_folder(self):
        o_load = Load(parent=self)
        o_load.select_folder(data_type=DataType.ob)

    # Checking data (proton charge, empty runs ...)
    def checking_data(self):
        try:
            o_checking = CheckingData(parent=self)
            o_checking.run()
        except ValueError:
            logging.info("Check the input folders provided !")

    def recap_data(self):
        o_recap = RecapData(parent=self)
        o_recap.run()

    def checkin_data_entries(self):
        o_check = CheckingData(parent=self)
        o_check.checking_minimum_requirements()

    # mode selection
    def mode_selection(self):
        self.o_mode = ModeSelection(parent=self)
        self.o_mode.select()

    # load data
    def load_data(self):
        self.o_mode.load()
        
    def select_tof_ranges(self):
        if self.operating_mode == OperatingMode.white_beam:
            return

        # self.o_tof_range_mode.select_multi_tof_range()
        self.o_tof_range_mode.select_tof_range()

    def combine_tof_mode_data(self):
        if self.o_tof_range_mode:
            self.o_tof_range_mode.combine_tof_mode_data()

    # cleaning low/high pixels
    def clean_images_settings(self):
        self.o_clean = ImagesCleaner(parent=self)
        self.o_clean.settings()

    def clean_images_setup(self):
        self.o_clean.cleaning_setup()

    def clean_images(self):
        self.o_clean.cleaning()

    def visualization_cleaning_settings(self):
        self.o_vizu = Visualization(parent=self)
        self.o_vizu.settings()

    def visualization_cleaning(self):
        self.o_vizu.visualize(data_after=self.master_3d_data_array_cleaned[DataType.sample],
                              label_after='cleaned',
                              label_before='raw',
                              data_before=self.master_3d_data_array[DataType.sample],
                              turn_on_vrange=False)

    def select_export_folder(self):
        self.o_clean.select_export_folder()

    def export_cleaned_images(self):
        self.o_clean.export_clean_images()

    # normalization
    def normalization_settings(self):
        self.o_norm = Normalization(parent=self)
        self.o_norm.normalization_settings()

    def normalization_select_roi(self):
        self.o_norm.select_roi()

    def normalization(self):
        self.o_norm.run()

    def visualization_normalization_settings(self):
        self.o_vizu = Visualization(parent=self)
        self.o_vizu.settings()

        # self.o_norm.visualization_normalization_settings()

    def visualize_normalization(self):
        self.o_vizu.visualize(data_after=self.normalized_images,
                              label_before='cleaned',
                              label_after='normalized',
                              data_before=self.master_3d_data_array_cleaned[DataType.sample],
                              turn_on_vrange=True)
        # self.o_norm.visualize_normalization()

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
