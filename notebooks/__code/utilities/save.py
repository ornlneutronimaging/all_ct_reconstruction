from PIL import Image
from astropy.io import fits
import numpy as np


def make_tiff(data=[], filename='', metadata=None):
    new_image = Image.fromarray(np.array(data))
    if metadata:
        new_image.save(filename, tiffinfo=metadata)
    else:
        new_image.save(filename)


def make_fits(data=[], filename=""):
    """create fits file"""
    fits.writeto(filename, data, overwrite=True)