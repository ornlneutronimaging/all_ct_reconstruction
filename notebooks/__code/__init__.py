import numpy as np
import tomopy

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


class Instrument:
    mars = "mars"
    venus = "venus"
    snap = "snap"
    

class FileNamingConvention:
    old_file = "Old file naming convention (Run_####/)"
    new_file = "New file naming convention (####_Run_####_####/)"


class OperatingMode:
    tof = 'tof'
    white_beam = 'white_beam'


STEP3_SVMBIR_SCRIPTS = "/SNS/VENUS/shared/software/git/all_ct_reconstruction/notebooks/step3_reconstruction_white_beam_mode_images_using_svmbir.py"
STEP3_FPB_SCRIPTS = "/SNS/VENUS/shared/software/git/all_ct_reconstruction/notebooks/step3_reconstruction_white_beam_mode_images_using_fbp.py"

STEP3_SCRIPTS = "/SNS/VENUS/shared/software/git/all_ct_reconstruction/notebooks/step3_reconstruction_CCD_or_TimePix_images.py"
STEP2_NOTEBOOK = "/SNS/VENUS/shared/software/git/all_ct_reconstruction/notebooks/step2_slice_CCD_or_TimePix_images.ipynb"

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
