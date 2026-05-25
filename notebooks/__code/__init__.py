import numpy as np
import tomopy
import os

# Patch as_ndarray
def patched_as_ndarray(arr, dtype=None, copy=False):
    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr, dtype=dtype)
    return arr

# Patch as_dtype
def patched_as_dtype(arr, dtype, copy=False):
    arr = patched_as_ndarray(arr)  # ensure it's a NumPy array first
    if arr.dtype != dtype:
        arr = np.asarray(arr, dtype=dtype)  # safer than np.array(..., copy=...)
    return arr

# Apply both patches
tomopy.util.dtype.as_ndarray = patched_as_ndarray
tomopy.util.dtype.as_dtype = patched_as_dtype


# from __code.utilities.system import get_user_name
# from __code.config import debugging, debugger_username


class DetectorType:
    tpx1_legacy = "tpx1 - old naming convention (until July 2025)"
    tpx1 = "tpx1 - new naming convention (from August 2025)"
    tpx3 = "tpx3"
    ccd = "CCD"
    ikonxl = "IkonXL"


class DataType:
    sample = 'sample'
    ob = 'ob'
    dc = 'dc'
    ct_scans = 'ct_scans'
    ipts = 'ipts'
    top = 'top'
    nexus = 'nexus'
    cleaned_images = 'cleaned images'
    normalized = 'normalized'
    reconstructed = 'reconstructed'
    extra = 'extra'
    processed = "processed"
    raw= 'raw'
    hdf5 = 'hdf5'


class Instrument:
    mars = "mars"
    venus = "venus"
    snap = "snap"


class OperatingMode:
    tof = 'tof'
    white_beam = 'white_beam'


# is project in development or not? This variable is used to decide which script to use for reconstruction (development or stable)
# get name of top folder of this project
path_of_this_file = os.path.abspath(__file__)
top_folder_of_this_project = os.path.dirname(os.path.dirname(os.path.dirname(path_of_this_file)))
if "development" in top_folder_of_this_project:
    _root_folder = "/SNS/VENUS/shared/software/git/all_ct_reconstruction_development/notebooks/"
else:
    _root_folder = "/SNS/VENUS/shared/software/git/all_ct_reconstruction/notebooks/"
    
STEP3_SVMBIR_SCRIPTS = os.path.join(_root_folder, "step3_reconstruction_white_beam_mode_images_using_svmbir.py")
STEP3_FPB_SCRIPTS = os.path.join(_root_folder, "step3_reconstruction_white_beam_mode_images_using_fbp.py")
STEP3_SCRIPTS = os.path.join(_root_folder, "step3_reconstruction_images.py")
STEP3_NOTEBOOK = os.path.join(_root_folder, "step3_reconstruct_images.ipynb")

STEP3_SCRIPTS_OFFLINE = "step3_reconstruction_CCD_or_TimePix_images.py"

DEFAULT_OPERATING_MODE = OperatingMode.white_beam
DEFAULT_RECONSTRUCTION_ALGORITHM = ["tomopy_fbp"]
NBR_TOF_RANGES = 3

LOAD_DTYPE = np.uint16
                                
ANGSTROMS = u"\u212b"
LAMBDA = u"\u03bb"

class Run:
    full_path = 'full path'
    proton_charge_c = 'proton charge c'
    use_it = 'use it'
    angle = 'angle'
    frame_number = 'number of frames'
    nexus = 'nexus'


class CleaningAlgorithm:
    in_house = 'histogram'
    tomopy = 'tomopy'
    scipy = 'scipy'


class NormalizationSettings:
    pc = 'proton charge'
    frame_number = 'frame number'
    roi = 'roi'
    sample_roi = 'roi_sample'


class RemoveStripeAlgo:
    remove_stripe_fw = "remove_stripe_fw"
    remove_stripe_ti = "remove_stripe_ti"
    remove_stripe_sf = "remove_stripe_sf"
    remove_stripe_based_sorting = "remove_stripe_based_sorting"
    remove_stripe_based_filtering = "remove_stripe_based_filtering"
    remove_stripe_based_fitting = "remove_stripe_based_fitting"
    remove_large_stripe = "remove_large_stripe"
    remove_all_stripe = "remove_all_stripe"
    remove_dead_stripe = "remove_dead_stripe"
    remove_stripe_based_interpolation = "remove_stripe_based_interpolation"


class WhenToRemoveStripes:
    in_notebook = "in notebook"
    out_notebook = "outside notebook"
    never = "never"
