import tomopy
import logging

from __code.config import NUM_THREADS


def log_conversion(normalized_data):
    logging.info("Converting data to -log")
    normalized_data_log = tomopy.minus_log(normalized_data, NUM_THREADS, out=None)
    logging.info("Data converted to -log!")
    return normalized_data_log
    