import logging
import tomopy

from __code.config import NUM_THREADS


def remove_negative_value(normalized_data):
    logging.info("Checking for negative values in normalized_data:")
    if normalized_data.any() < 0:
        logging.info("\tRemoving negative values")
        cleaned_data = tomopy.misc.corr.remove_neg(normalized_data, val=0.0, ncore=NUM_THREADS)
        logging.info("Negative values removed!")
        return cleaned_data
    else:
        return normalized_data
    