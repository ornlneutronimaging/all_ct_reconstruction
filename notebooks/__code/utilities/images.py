import numpy as np
from scipy.ndimage import median_filter


def replace_pixels(im, nbr_bins=0, low_gate=1, high_gate=9, correct_radius=1):

    _, bin_edges = np.histogram(im.flatten(), bins=nbr_bins, density=False)
    thres_low = bin_edges[low_gate]
    thres_high = bin_edges[high_gate]

    y_coords, x_coords = np.nonzero(np.logical_or(im <= thres_low, 
                                                  im > thres_high))

    full_median_filter_corr_im = median_filter(im, size=correct_radius)
    for y, x in zip(y_coords, x_coords):
        im[y, x] = full_median_filter_corr_im[y, x]

    return im
