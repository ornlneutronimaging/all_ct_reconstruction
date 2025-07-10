from __code import OperatingMode, DataType
from __code.utilities.system import get_user_name


debugging = True

verbose = True
debugger_username = 'j35'
imaging_team = ["j35", "gxt"]

debugger_folder = ['/Users/j35/HFIR/CG1D/',
                   '/Volumes/JeanHardDrive/HFIR/CG1D/']
debugger_instrument_folder = {'CG1D': ["/Users/j35/HFIR/CG1D",
                                       "/Volumes/JeanHardDrive/HFIR/CG1D/",
                                       ],
                              'SNAP': ["/Users/j35/SNS/SNAP"],
                              'VENUS': ["/Users/j35/SNS/VENUS"],
                              }
# debugger_instrument_folder = {'CG1D': "/Volumes/JeanHardDrive/HFIR/CG1D",
#                               'SNAP': "/Volumes/JeanHardDrive/SNS/SNAP",
#                               'VENUS': "/Volumes/JeanHardDrive/SNS/VENUS"}
analysis_machine = 'bl10-analysis1.sns.gov'
project_folder = 'IPTS-24863-test-imars3d-notebook'
percentage_of_images_to_use_for_roi_selection = 0.05
minimum_number_of_images_to_use_for_roi_selection = 10
DEFAULT_CROP_ROI = [0, 510, 103, 404]
DEFAULT_BACKROUND_ROI = [5, 300, 5, 600]
DEFAULT_TILT_SLICES_SELECTION = [103, 602]
STEP_SIZE = 50  # for working with bucket of data at a time
HSNT_FOLDER = '/data/'
HSNT_SCRIPTS_FOLDER = '/data/scripts/'

DEFAULT_NAMING_CONVENTION_INDICES = [10, 11]

PROTON_CHARGE_TOLERANCE_C = 0.1  # C

# at VENUS
DISTANCE_SOURCE_DETECTOR = 25  # m

# parameters used for data cleaning
"""
if_clean: switch of cleaning operation (bool)
if_save_clean: switch of saving cleaned tiff files (bool)
low_gate: the lower index of the image hist bin edges (int 0-9)
high_gate: the higher index of the image hist bin edges (int 0-9)
correct_radius: the neighbors (2r+1 * 2r+1 matrix) radius used for replacing bad pixels (int) 
"""
clean_paras = {'if_clean': True, 
               'if_save_clean': False, 
               'low_gate': 1, 
               'high_gate': 9, 
               'correct_radius': 1,
               'edge_nbr_pixels': 10,
               'nbr_bins': 10}

# list of offset values along X and Y axis, respectively (X offset, Y offset)
chips_offset = [2, 2]

NUM_THREADS = 60
SVMBIR_LIB_PATH = "/fastdata/"
SVMBIR_LIB_PATH_BACKUP = "/SNS/VENUS/shared/fastdata/"

# if x percent of a pixel value is still above median, remove it
TOMOPY_REMOVE_OUTLIER_THRESHOLD_RATIO = 0.1

# percentage of data to use for the reconstruction
PERCENTAGE_OF_DATA_TO_USE_FOR_RECONSTRUCTION = 50

# gamma diff to use in remove_outlier with tomopy
GAMMA_DIFF = 20

# remove outlier tomopy diff
TOMOPY_DIFF = 0.2

svmbir_parameters = {'sharpness': 0,
                     'max_resolutions': 2,
                     'positivity': False,
                     'snr_db': 30,
                     'max_iterations': 20,
                     'verbose': True,
                    }

debug_folder = {OperatingMode.tof: {DataType.sample: "/SNS/VENUS/IPTS-33699/shared/autoreduce/mcp/September20_2024_PurpleCar_GoldenRatio_CT_5_0_C_Cd_inBeam_Resonance",
                                    DataType.ob: "/SNS/VENUS/IPTS-33699/shared/autoreduce/mcp/September26_2024_PurpleCar_OpenBean_5_0_C_Cd_inBeam_Resonance",
                                    DataType.cleaned_images: '/SNS/VENUS/IPTS-33699/shared/processed_data/jean_test',
                                    DataType.normalized: '/SNS/VENUS/IPTS-33699/shared/processed_data/jean_test',
                                    DataType.reconstructed: '/SNS/VENUS/IPTS-33699/shared/processed_data/jean_test',
                                    DataType.extra: '/SNS/VENUS/IPTS-33699/shared/processed_data/jean_test',
                                    DataType.nexus: '/SNS/VENUS/IPTS-33699/nexus/'
                                    },
                # OperatingMode.white_beam: {DataType.sample: "/SNS/VENUS/IPTS-33531/shared/processed_data/November8_2024_PlantE/",
                #                             DataType.ob: "/SNS/VENUS/IPTS-33531/shared/processed_data/ob_PlantE/",
                #                             DataType.dc: "/SNS/VENUS/IPTS-33531/shared/processed_data/dc/45s/",
                #                             DataType.cleaned_images: '/SNS/VENUS/IPTS-33531/shared/processed_data/jean_test',
                #                             DataType.normalized: '/SNS/VENUS/IPTS-33531/shared/processed_data/jean_test',
                #                             DataType.reconstructed: '/SNS/VENUS/IPTS-33531/shared/processed_data/jean_test',
                #                             DataType.extra: '/SNS/VENUS/IPTS-33531/shared/processed_data/jean_test',
                #                             DataType.nexus: '/SNS/VENUS/IPTS-33531/nexus/',
                #                             },
                # OperatingMode.white_beam: {DataType.sample: "/SNS/VENUS/IPTS-33531/shared/processed_data/November9_2024_Soil_microplastics_1/",
                #                             DataType.ob: "/SNS/VENUS/IPTS-33531/shared/processed_data/ob_PlantE/",
                #                             DataType.dc: "",
                #                             DataType.cleaned_images: '/SNS/VENUS/IPTS-33531/shared/processed_data/jean_test',
                #                             DataType.normalized: '/SNS/VENUS/IPTS-33531/shared/processed_data/jean_test',
                #                             DataType.reconstructed: '/SNS/VENUS/IPTS-33531/shared/processed_data/jean_test',
                #                             DataType.extra: '/SNS/VENUS/IPTS-33531/shared/processed_data/jean_test',
                #                             },
                # OperatingMode.white_beam: {DataType.sample: "/SNS/VENUS/IPTS-33531/shared/processed_data/truck_CT_data/",
                #                             DataType.ob: "/SNS/VENUS/IPTS-33531/shared/processed_data/ob/",
                #                             DataType.dc: "/SNS/VENUS/IPTS-33531/shared/processed_data/dc/45s/",
                #                             DataType.cleaned_images: '/SNS/VENUS/IPTS-33531/shared/processed_data/jean_test',
                #                             DataType.normalized: '/SNS/VENUS/IPTS-33531/shared/processed_data/jean_test',
                #                             DataType.reconstructed: '/SNS/VENUS/IPTS-33531/shared/processed_data/jean_test',
                #                             DataType.extra: '/SNS/VENUS/IPTS-33531/shared/processed_data/jean_test',
                #                             DataType.nexus: '/SNS/VENUs/IPTS-33531/nexus',
                #                             },
                # OperatingMode.white_beam: {DataType.sample: "/SNS/SNAP/IPTS-25265/shared/Jean/moon_rocks_combined_normalized_53_angles_renamed",
                #                             DataType.ob: "",
                #                             DataType.dc: "",
                #                             DataType.cleaned_images: '/SNS/SNAP/IPTS-25265/shared/processed_data/jean_test',
                #                             DataType.normalized: '/SNS/SNAP/IPTS-25265/shared/processed_data/jean_test',
                #                             DataType.reconstructed: '/SNS/SNAP/IPTS-25265/shared/processed_data/jean_test',
                #                             DataType.extra: '/SNS/SNAP/IPTS-25265/shared/processed_data/jean_test',
                #                             DataType.nexus: '/SNS/SNAP/IPTS-25265/nexus',
                #                             },
                                            
                OperatingMode.white_beam: {DataType.sample: "/HFIR/CG1D/IPTS-27829/raw/ct_scans/October15_2021",
                                            DataType.ob: "/HFIR/CG1D/IPTS-27829/raw/ob/October15_2021",
                                            DataType.dc: "",
                                            DataType.cleaned_images: '/HFIR/CG1D/IPTS-27829/shared/processed_data/jean_test',
                                            DataType.normalized: '/HFIR/CG1D/IPTS-27829/shared/processed_data/jean_test',
                                            DataType.reconstructed: '/HFIR/CG1D/IPTS-27829/shared/processed_data/jean_test',
                                            DataType.extra: '/HFIR/CG1D/IPTS-27829/shared/processed_data/jean_test',
                                            DataType.nexus: '/HFIR/CG1D/IPTS-27829/nexus',
                                            DataType.top: '/HFIR/CG1D/IPTS-27829/raw/ct_scans/',
                                            },
}

# roi = {OperatingMode.tof: {'left': 0,
#                             'right': 74,
#                             'top': 0,
#                             'bottom': 49},
#         OperatingMode.white_beam: {'left': 0,
#                                    'right': 549,
#                                    'top': 131,
#                                    'bottom': 8177},
#                                    }

# plant
roi = {OperatingMode.tof: {'left': 0,
                            'right': 74,
                            'top': 0,
                            'bottom': 49},
        OperatingMode.white_beam: {'left': 155,
                                   'right': 8989,
                                   'top': 200,
                                   'bottom': 464},
                                   }

crop_roi = {OperatingMode.tof: {'left': 0,
                            'right': 74,
                            'top': 0,
                            'bottom': 49},
        OperatingMode.white_beam: {'left': 5,
                                   'right': -200,
                                   'top': 5,
                                   'bottom': -5},
                                   }

DEBUG = False
if get_user_name() == debugger_username:
    DEBUG = debugging
