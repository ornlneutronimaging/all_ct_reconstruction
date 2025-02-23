import logging
import tomopy
import numpy as np

from __code.config import NUM_THREADS


def remove_negative_values(normalized_data):
    logging.info("Checking for negative values in normalized_data:")
    if normalized_data.any() < 0:
        logging.info("\tRemoving negative values")
        cleaned_data = tomopy.misc.corr.remove_neg(normalized_data, val=0.0, ncore=NUM_THREADS)
        logging.info("Negative values removed!")
        return cleaned_data[:]
    else:
        return normalized_data[:]
    
def remove_0_values(normalized_data):
    logging.info("Checking for 0 values in normalized_data:")
    if normalized_data.any() == 0:
        logging.info("\tRemoving 0 values")
        cleaned_data = tomopy.misc.corr.remove_zero(normalized_data, val=np.NaN, ncore=NUM_THREADS)
        logging.info("0 values removed!")
        return cleaned_data[:]
    else:
        return normalized_data[:]
    