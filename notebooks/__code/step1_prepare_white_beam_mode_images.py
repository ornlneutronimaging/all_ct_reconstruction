import os
import logging
from collections import OrderedDict


from __code import DataType, OperatingMode, DEFAULT_OPERATING_MODE, DEBUG
from __code.utilities.logging import setup_logging
from __code.utilities.configuration_file import Configuration

from __code.workflow.load import Load
from __code.workflow.combine_ob_dc import CombineObDc
from __code.workflow.checking_data import CheckingData
from __code.workflow.recap_data import RecapData
from __code.workflow.mode_selection import ModeSelection
from __code.workflow.reconstruction_selection import ReconstructionSelection
from __code.workflow.images_cleaner import ImagesCleaner
from __code.workflow.normalization import Normalization
from __code.workflow.chips_correction import ChipsCorrection
from __code.workflow.center_of_rotation_and_tilt import CenterOfRotationAndTilt
from __code.workflow.remove_strips import RemoveStrips
from __code.workflow.svmbir_handler import SvmbirHandler
from __code.workflow.fbp_handler import FbpHandler
from __code.workflow.final_projections_review import FinalProjectionsReview
from __code.workflow.export import ExportExtra
from __code.workflow.visualization import Visualization
from __code.workflow.rotate import Rotate
from __code.workflow.crop import Crop
from __code.utilities.configuration_file import ReconstructionAlgorithm

LOG_BASENAME_FILENAME, _ = os.path.splitext(os.path.basename(__file__))


class Step1PrepareWhiteBeamModeImages:

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
        self.working_dir[DataType.top] = os.path.join(top_sample_dir)
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

    def select_ob_images(self):
        o_load = Load(parent=self)
        o_load.select_images(data_type=DataType.ob)

    def select_dc_images(self):
        o_load = Load(parent=self)
        o_load.select_images(data_type=DataType.dc)

    # define naming convention to easily extract angle value
    def define_naming_schema(self):
        self.o_load = Load(parent=self)
        self.o_load.define_naming_convention()

    # pecentage of data to use
    def select_percentage_of_data_to_use(self):
        self.o_load.select_percentage_of_data_to_use()

    # load data
    def load_data(self):
        self.o_load.load_white_beam_data()
        
    # visualization
    def what_to_visualize(self):
        self.o_vizu = Visualization(parent=self)
        self.o_vizu.what_to_visualize()

    def visualize_raw_data(self):
        self.o_vizu.visualize_according_to_selection()

    # pre processing cropt
    def pre_processing_crop_settings(self):
        self.o_crop1 = Crop(parent=self)
        self.o_crop1.set_region(before_normalization=True)

    def pre_processing_crop(self):
        self.o_crop1.run()

    # cleaning low/high pixels
    def clean_images_settings(self):
        self.o_clean = ImagesCleaner(parent=self)
        self.o_clean.settings()

    def clean_images_setup(self):
        self.o_clean.cleaning_setup()

    def clean_images(self):
        self.o_clean.cleaning()

    # normalization
    def normalization_settings(self):
        self.o_norm = Normalization(parent=self)
        self.o_norm.normalization_settings()

    def normalization_select_roi(self):
        self.o_norm.select_roi()

    def normalization(self):
        o_combine = CombineObDc(parent=self)
        o_combine.run()
        self.o_norm.run()

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

  # crop data
    def crop_settings(self):
        self.o_crop = Crop(parent=self)
        self.o_crop.set_region()

    def crop(self):
        self.o_crop.run()

    # rotate sample
    def rotate_data_settings(self):
        self.o_rotate = Rotate(parent=self)
        self.o_rotate.set_settings()

    def apply_rotation(self):
        self.o_rotate.apply_rotation()

    def visualize_after_rotation(self):
        o_review = FinalProjectionsReview(parent=self)
        o_review.single_image(image=self.normalized_images[0])

    # strips removal
    def select_remove_strips_algorithms(self):
        self.corrected_images = self.normalized_images
        self.o_remove = RemoveStrips(parent=self)
        self.o_remove.select_algorithms()

    def define_settings(self):
        self.o_remove.define_settings()

    def remove_strips_and_display(self):
        self.o_remove.run()

    # calculate center of rotation & tilt
    def select_sample_roi(self):
        if self.corrected_images is None:
            self.corrected_images = self.normalized_images

        self.o_center_and_tilt = CenterOfRotationAndTilt(parent=self)
        self.o_center_and_tilt.select_range()

    def center_of_rotation_and_tilt_settings(self):
        self.o_center_and_tilt.settings()

    def perform_center_of_rotation_and_tilt(self):
        self.o_center_and_tilt.run()

    # def calculate_and_apply_center_of_rotation_and_tilt(self):
    #     self.o_center_and_tilt.run()

    # select reconstruction method
    def select_reconstruction_method(self):
        self.o_mode = ReconstructionSelection(parent=self)
        self.o_mode.select()

    # run svmbir
    def reconstruction_settings(self):
        if self.corrected_images is None:
            self.corrected_images = self.normalized_images
        
        if ReconstructionAlgorithm.svmbir in self.configuration.reconstruction_algorithm:
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
        