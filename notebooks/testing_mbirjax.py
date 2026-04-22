import mbirjax as mj
import os
import numpy as np
import json
import matplotlib.pyplot as plt
import glob
from skimage.io import imread

def _normalize_to_float32(img: np.ndarray) -> np.ndarray:
    """
    Convert image to float32 and normalize if it is an integer dtype.

    - If `imgs.dtype` is an integer type, cast to float32 and divide by the max value for that dtype.
    - Otherwise, cast to float32 without scaling.

    Args:
        img (np.ndarray): Input image array.

    Returns:
        np.ndarray: float32 array, normalized to [0, 1] if input was integer.
    """
    if np.issubdtype(img.dtype, np.integer):
        maxval = np.iinfo(img.dtype).max
        return img.astype(np.float32) / maxval
    return img.astype(np.float32)

def read_tif_stack_dir(scan_dir, view_ids=None):
    """Reads a tif stack of scan images from a directory. This function is a subroutine to `load_scans_and_params`.

    Args:
        scan_dir (string): Path to a ConeBeam Scan directory.
            Example: "<absolute_path_to_dataset>/Radiographs"
        view_ids (ndarray of ints, optional, default=None): List of view indices to specify which scans to read.
    Returns:
        ndarray (float): 3D numpy array, (num_views, num_det_rows, num_det_channels). A stack of scan images.
    """

    import tifffile
    # Get the files that are views and check that we have as many as we need
    img_path_list = sorted(glob.glob(os.path.join(scan_dir, '*[0-9].tif')))
    if len(img_path_list) == 0:
        img_path_list = sorted(glob.glob(os.path.join(scan_dir, '*[0-9].tiff')))  # Assume files are '.tif' but check '.tiff' if not

    # if no views are found, raise an error
    if len(img_path_list) == 0:
        raise FileNotFoundError('No scan images found in directory: {}'.format(scan_dir))

    # Set view_idx to be an array corresponding to the views that should be read.
    # This assumes that all the views are labeled sequentially.
    if view_ids is None:
        view_ids = np.arange(len(img_path_list))
    else:
        max_view_id = np.amax(view_ids)
        if max_view_id >= len(img_path_list):
            raise FileNotFoundError('The max view index was given as {}, but there are only {} views in {}'.format(max_view_id, len(img_path_list), scan_dir))
    img_path_list = [img_path_list[idx] for idx in view_ids]

    output_views = tifffile.imread(img_path_list, ioworkers=48, maxworkers=8)
    output_views = _normalize_to_float32(output_views)

    # return shape = num_views x num_det_rows x num_det_channels
    return output_views

if __name__ == "__main__":
    
    data_dir = "/SNS/VENUS/IPTS-37480/shared/Jean_mbirjax/stack_of_attenuation_images"
    assert os.path.exists(data_dir), f"Data directory {data_dir} does not exist"
    
    # load json file and get list of angles
    config_file = os.path.join(os.path.dirname(data_dir), "config.json")
    assert os.path.exists(config_file), f"Config file {config_file} does not exist"

    with open(config_file, "r") as f:
        config = json.load(f)
        
    config_dict = json.loads(config)

    angles = np.array(config_dict["list_of_angles"])
    print(type(angles), len(angles))
    
    sinogram = read_tif_stack_dir(data_dir)
    
    start_slice = 400
    num_slices = 20
    downsample_factor = 1  # Channel downsampling

    end_slice = start_slice + num_slices

    s = sinogram[:, start_slice:end_slice, ::downsample_factor]
    # plt.hist(s.flatten(), bins=400)
    # plt.show()
    mj.slice_viewer(s, slice_axis=0, title='Sinogram, downsampled by {}'.format(downsample_factor))

    ct_model = mj.ParallelBeamModel(s.shape, angles)
    recon0, recon_dict0 = ct_model.recon(s)

    mj.slice_viewer(recon0, slice_axis=2, title='Reconstruction, downsampled by {}'.format(downsample_factor))
