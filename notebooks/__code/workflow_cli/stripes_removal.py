from tqdm import tqdm

from __code import RemoveStripeAlgo
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
