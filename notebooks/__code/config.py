"""
Configuration file for CT reconstruction pipeline.

This module contains all configuration parameters, constants, and default values
used throughout the CT reconstruction workflow. It includes settings for:
- Debug and development modes
- File paths and directories
- Reconstruction parameters
- Hardware specifications
- Data processing parameters

The configuration is organized by operating mode (ToF vs White Beam) and
data type (sample, OB, DC, etc.).
"""

from typing import Dict, List, Any, Union
from __code import DetectorType, OperatingMode, DataType
from __code.utilities.system import get_user_name

# Debug and development settings
debugging: bool = True
verbose: bool = True
debugger_username: str = 'j35'
imaging_team: List[str] = ["j35", "gxt"]

# Development folder paths for different users and instruments
debugger_folder: List[str] = ['/Users/j35/HFIR/CG1D/',
                   '/Volumes/JeanHardDrive/HFIR/CG1D/']

debugger_instrument_folder: Dict[str, List[str]] = {
    'CG1D': ["/Users/j35/HFIR/CG1D",
             "/Volumes/JeanHardDrive/HFIR/CG1D/"],
    'SNAP': ["/Users/j35/SNS/SNAP"],
    'VENUS': ["/Users/j35/SNS/VENUS"],
}

# System configuration
analysis_machine: str = 'bl10-analysis1.sns.gov'
project_folder: str = 'IPTS-24863-test-imars3d-notebook'

# Processing parameters
percentage_of_images_to_use_for_roi_selection: float = 0.05
minimum_number_of_images_to_use_for_roi_selection: int = 10

# Default ROI and processing parameters
DEFAULT_CROP_ROI: List[int] = [0, 510, 103, 404]
DEFAULT_BACKROUND_ROI: List[int] = [5, 300, 5, 600]
DEFAULT_TILT_SLICES_SELECTION: List[int] = [103, 602]
STEP_SIZE: int = 50  # for working with bucket of data at a time

# File system paths
HSNT_FOLDER: str = '/data/'
HSNT_SCRIPTS_FOLDER: str = '/data/scripts/'

DEFAULT_NAMING_CONVENTION_INDICES: List[int] = [10, 11]

# Physical constants
PROTON_CHARGE_TOLERANCE_C: float = 0.1  # Coulombs
DISTANCE_SOURCE_DETECTOR: int = 25  # meters (at VENUS)

# Data cleaning parameters
"""
Data cleaning configuration dictionary:

if_clean: Enable/disable cleaning operation (bool)
if_save_clean: Enable/disable saving cleaned TIFF files (bool) 
low_gate: Lower index of image histogram bin edges (int 0-9)
high_gate: Higher index of image histogram bin edges (int 0-9)
correct_radius: Neighbor radius (2r+1 × 2r+1 matrix) for bad pixel replacement (int)
edge_nbr_pixels: Number of edge pixels to consider (int)
nbr_bins: Number of histogram bins (int)
"""
clean_paras: Dict[str, Union[bool, int]] = {
    'if_clean': True, 
    'if_save_clean': False, 
    'low_gate': 1, 
    'high_gate': 9, 
    'correct_radius': 2,
    'edge_nbr_pixels': 10,
    'nbr_bins': 10
}

# Hardware offset parameters
chips_offset: List[int] = [2, 2]  # X and Y axis offsets for detector chips

# Performance and library configuration
NUM_THREADS: int = 60
SVMBIR_LIB_PATH: str = "/fastdata/"
SVMBIR_LIB_PATH_BACKUP: str = "/SNS/VENUS/shared/fastdata/"
SVMBIR_LIB_PATH_BACKUP_2: str = "/tmp/"

# Image processing thresholds
TOMOPY_REMOVE_OUTLIER_THRESHOLD_RATIO: float = 0.1  # Outlier threshold percentage
PERCENTAGE_OF_DATA_TO_USE_FOR_RECONSTRUCTION: int = 50  # Data usage percentage
GAMMA_DIFF: int = 20  # Gamma difference for outlier removal
TOMOPY_DIFF: float = 0.2  # TomoPy difference threshold

# SVMBIR reconstruction parameters
svmbir_parameters: Dict[str, Union[int, float, bool]] = {
    'sharpness': 0,
    'max_resolutions': 2,
    'positivity': False,
    'snr_db': 30,
    'max_iterations': 20,
    'verbose': True,
}

default_detector_type = DetectorType.tpx1_legacy

# Debug data folder configuration organized by operating mode and data type
debug_folder: Dict[DetectorType, Dict[OperatingMode, Dict[DataType, str]]] = {   
                                                                              
    DetectorType.tpx1_legacy: {

        ## old naming convention
        OperatingMode.tof: {
            DataType.sample: "/SNS/VENUS/IPTS-33699/shared/autoreduce/mcp/September20_2024_PurpleCar_GoldenRatio_CT_5_0_C_Cd_inBeam_Resonance",
            DataType.ob: "/SNS/VENUS/IPTS-33699/shared/autoreduce/mcp/September26_2024_PurpleCar_OpenBean_5_0_C_Cd_inBeam_Resonance",
            DataType.cleaned_images: '/SNS/VENUS/IPTS-33699/shared/processed_data/jean_test',
            DataType.normalized: '/SNS/VENUS/IPTS-33699/shared/processed_data/jean_test',
            DataType.reconstructed: '/SNS/VENUS/IPTS-33699/shared/processed_data/jean_test',
            DataType.extra: '/SNS/VENUS/IPTS-33699/shared/processed_data/jean_test',
            DataType.nexus: '/SNS/VENUS/IPTS-33699/nexus/'
        },

        ## new naming convention
        # OperatingMode.tof: {      
        #     DataType.sample: "/SNS/VENUS/IPTS-33531/shared/jean/Crushed_ring/",
        #     DataType.ob: "/SNS/VENUS/IPTS-33531/shared/Crushed_ring_ob",
        #     DataType.cleaned_images: '/SNS/VENUS/IPTS-33531/shared/processed_data/jean',
        #     DataType.normalized: '/SNS/VENUS/IPTS-33531/shared/processed_data/jean',
        #     DataType.reconstructed: '/SNS/VENUS/IPTS-33531/shared/processed_data/jean',
        #     DataType.extra: '/SNS/VENUS/IPTS-33531/shared/processed_data/jean',
        #     DataType.nexus: '/SNS/VENUS/IPTS-33531/nexus/'
        # },
        OperatingMode.white_beam: {
            DataType.sample: "/SNS/VENUS/IPTS-34899/shared/jean_test/white_beam_ct_test/UHP_CT_TIF",
            DataType.ob: "/SNS/VENUS/IPTS-34899/shared/jean_test/white_beam_ct_test/brights_for_CT",
            DataType.dc: "/SNS/VENUS/IPTS-34899/shared/jean_test/white_beam_ct_test/darks_for_CT",
            DataType.cleaned_images: '/SNS/VENUS/IPTS-34899/shared/processed_data/jean_test',
            DataType.normalized: '/SNS/VENUS/IPTS-34899/shared/processed_data/jean_test',
            DataType.reconstructed: '/SNS/VENUS/IPTS-34899/shared/processed_data/jean_test',
            DataType.extra: '/SNS/VENUS/IPTS-34899/shared/processed_data/jean_test',
            # DataType.nexus: '/SNS/VENUS/IPTS-34899/nexus',
        },
    },
    DetectorType.tpx1: {
        OperatingMode.tof: {
            DataType.sample: "/SNS/VENUS/IPTS-33531/shared/Shimin_save/new_data_structure_replica/images/tpx1/raw/ct/20241122_CrashRingB_hyperCT_2_500C_1_800AngsMin_light",
            DataType.ob: "/SNS/VENUS/IPTS-33531/shared/Shimin_save/new_data_structure_replica/images/tpx1/ob/20241122_CrashRingB_hyperCT_2_500C_1_800AngsMin",
            DataType.cleaned_images: '/SNS/VENUS/IPTS-33531/shared/jean',
            DataType.normalized: '/SNS/VENUS/IPTS-33531/shared/jean',
            DataType.reconstructed: '/SNS/VENUS/IPTS-33531/shared/jean',
            DataType.extra: '/SNS/VENUS/IPTS-33531/shared/jean',
            DataType.nexus: '/SNS/VENUS/IPTS-33531/nexus/'
        },
        OperatingMode.white_beam: None,
    },

    DetectorType.tpx3: {
        OperatingMode.tof: None,
        OperatingMode.white_beam: None
    },

    DetectorType.ccd: {
       OperatingMode.white_beam: {
            DataType.sample: "/HFIR/CG1D/IPTS-34899/shared/CT_by_jean/UTK_ORNL_VENUS_collaboration/UHP_CT_TIF",
            DataType.ob: "/HFIR/CG1D/IPTS-34899/shared/CT_by_jean/UTK_ORNL_VENUS_collaboration//brights_for_CT/LiFZnS_OB_1p8A_256x256_best.tif",
            DataType.dc: "",
            DataType.cleaned_images: '/HFIR/CG1D/IPTS-34899/shared/jean_test',
            DataType.normalized: '/HFIR/CG1D/IPTS-34899/shared/processed_data/jean_test',
            DataType.reconstructed: '/HFIR/CG1D/IPTS-34899/shared/processed_data/jean_test',
            DataType.extra: '/HFIR/CG1D/IPTS-34899/shared/processed_data/jean_test',
            # DataType.nexus: '/HFIR/CG1D/IPTS-34899/nexus',
        },

    #  DetectorType.ccd: {
    #    OperatingMode.white_beam: {
    #         DataType.sample: "/HFIR/CG1D/IPTS-33767/shared/PSI_files/09_nct/09_nct_post_2",
    #         DataType.ob: "/HFIR/CG1D/IPTS-33767/shared/PSI_files/09_nct/09_nct_ob",
    #         DataType.dc: "/HFIR/CG1D/IPTS-33767/shared/PSI_files/09_nct/09_nct_dc",
    #         DataType.cleaned_images: '/HFIR/CG1D/IPTS-33767/shared/processed_data/jean_psi_files',
    #         DataType.normalized: '/HFIR/CG1D/IPTS-33767/shared/processed_data/jean_psi_files',
    #         DataType.reconstructed: '/HFIR/CG1D/IPTS-33767/shared/processed_data/jean_psi_files',
    #         DataType.extra: '/HFIR/CG1D/IPTS-33767/shared/processed_data/jean_psi_files',
    #         # DataType.nexus: '/HFIR/CG1D/IPTS-33767/nexus',
    #     },
    },
}

# Region of Interest (ROI) configuration by operating mode
roi: Dict[OperatingMode, Dict[str, int]] = {
    OperatingMode.tof: {
        'left': 0,
        'right': 74,
        'top': 0,
        'bottom': 49
    },
    OperatingMode.white_beam: {
        'left': 155,
        'right': 8989,
        'top': 200,
        'bottom': 464
    },
}

# Crop ROI configuration by operating mode  
crop_roi: Dict[OperatingMode, Dict[str, int]] = {
    OperatingMode.tof: {
        'left': 0,
        'right': 74,
        'top': 0,
        'bottom': 49
    },
    OperatingMode.white_beam: {
        'left': 5,
        'right': -200,
        'top': 5,
        'bottom': -5
    },
}

# Debug mode activation
DEBUG: bool = False
if get_user_name() == debugger_username:
    DEBUG = debugging


default_config_timepix1 = {
    "chip1": {"xoffset": 2.4, "yoffset": 1, "description": "top right chip"},
    "chip2": {"xoffset": 0, "yoffset": 0, "description": "top left and reference chip"},
    "chip3": {"xoffset": 0, "yoffset": 1, "description": "bottom left chip"},
    "chip4": {"xoffset": 2.3, "yoffset": 2.3, "description": "bottom right chip"},
}


default_config_timepix3 = {
    "chip1": {"xoffset": 2.4, "yoffset": 1, "description": "top right chip"},
    "chip2": {"xoffset": 0, "yoffset": 0, "description": "top left and reference chip"},
    "chip3": {"xoffset": 0, "yoffset": 1, "description": "bottom left chip"},
    "chip4": {"xoffset": 2.3, "yoffset": 2.3, "description": "bottom right chip"},
}
