from tqdm import tqdm
import logging
import numpy as np

from __code import RemoveStripeAlgo, NUM_THREADS
from __code.utilities.general import retrieve_parameters
from __code.workflow.remove_strips import RemoveStrips


def retrieve_options(config_model, algorithm):

    print(f"-> {algorithm =}")

    if algorithm == RemoveStripeAlgo.remove_stripe_fw:
        param = retrieve_parameters(config_model.remove_stripe_fw_options)
        if param['level'] == 'None':
            del param['level']
        return param
    if algorithm == RemoveStripeAlgo.remove_stripe_ti:
        return retrieve_parameters(config_model.remove_stripe_ti_options)
    if algorithm == RemoveStripeAlgo.remove_stripe_sf:
        return retrieve_parameters(config_model.remove_stripe_sf_options)
    if algorithm == RemoveStripeAlgo.remove_stripe_based_sorting:
        param = retrieve_parameters(config_model.remove_stripe_based_sorting_options)
        if param['size'] == 'None':
            del param['size']
        return param
    if algorithm == RemoveStripeAlgo.remove_stripe_based_filtering:
        param = retrieve_parameters(config_model.remove_stripe_based_filtering_options)
        if param['size'] == 'None':
            del param['size']
        return param
    if algorithm == RemoveStripeAlgo.remove_stripe_based_fitting:
        param = retrieve_parameters(config_model.remove_stripe_based_fitting_options)
        left_value, right_value = param['sigma'].split(",")
        param['sigma'] = (int(left_value), int(right_value))
        return param
    if algorithm == RemoveStripeAlgo.remove_large_stripe:
        return retrieve_parameters(config_model.remove_large_stripe_options)
    if algorithm == RemoveStripeAlgo.remove_dead_stripe:
        return retrieve_parameters(config_model.remove_dead_stripe_options)
    if algorithm == RemoveStripeAlgo.remove_all_stripe:
        return retrieve_parameters(config_model.remove_all_stripe_options)
    if algorithm == RemoveStripeAlgo.remove_stripe_based_interpolation:
        return retrieve_parameters(config_model.remove_stripe_based_interpolation_options)
    return ""

def stripes_removal(config_model, data_array):    
    list_clean_stripes_algorithm = config_model.list_clean_stripes_algorithm

    if len(list_clean_stripes_algorithm) == 0:
        print(f"skipping any stripes removal!")
        return data_array
    
    print("stripes removal:")
    for _algo in list_clean_stripes_algorithm:
        options = retrieve_options(config_model,
                                   _algo)
        data_array = RemoveStrips.run_algo(RemoveStrips.list_algo[_algo]['function'],
                                           data_array,
                                           **options)
        print(" done!")

    return data_array


class StripesRemovalHandler:

    @staticmethod
    def remove_stripes(image_array, config):
        """
        Apply the strip removal algorithms on the input image array.
        :param image_array: 3D numpy array of shape (angles, slices, pixels)
        :param config: configuration dictionary containing parameters for strip removal
        :param ncore: number of threads to use for parallel processing
        :return: 3D numpy array with stripes removed
        """
        nore = NUM_THREADS

        list_algo_to_remove_stripes = config.get('list_clean_stripes_algorithm', [])
        if not list_algo_to_remove_stripes:
            logging.info("No strip removal algorithms specified. Returning original image array.")
            return image_array
        
        logging.info(f"Applying strip removal algorithms: {list_algo_to_remove_stripes}")
        try:
            for _algo in list_algo_to_remove_stripes:
                logging.info(f"\t -> Applying {_algo} ...")
                kwargs = config.get(f'remove_stripe_{_algo.lower()}_options', {})
                kwargs['ncore'] = nore  # Ensure we set the number of threads
                logging.info(f"\t -> Options for {_algo}: {kwargs}")
                image_array = RemoveStrips.run_algo(RemoveStrips.list_algo[_algo]['function'],
                                                    image_array,
                                                    **kwargs)
                logging.info(f"\t -> {_algo} applied successfully.")

        except np.linalg.LinAlgError as e:
            logging.info(f"ERROR: LinAlgError during strip removal: {e} running {_algo}.")
            
        return image_array
