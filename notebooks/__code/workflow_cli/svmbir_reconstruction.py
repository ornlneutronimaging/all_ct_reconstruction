import numpy as np
import svmbir

from __code.config import NUM_THREADS, SVMBIR_LIB_PATH
from __code.utilities.general import retrieve_parameters


def svmbir_reconstruction(config_model, data_array):
    print("launching the reconstruction ...", end="")
    svmbir_config = config_model.svmbir_config
    dict_parameters = retrieve_parameters(svmbir_config)
    top_slice = dict_parameters['top_slice']
    del dict_parameters['top_slice']
    bottom_slice = dict_parameters['bottom_slice']
    del dict_parameters['bottom_slice']
    list_of_angles_deg = config_model.list_of_angles
    list_of_angles_rad = [np.deg2rad(_angle) for _angle in list_of_angles_deg]
    image_width = config_model.image_size.width
    image_height = config_model.image_size.height

    # make call
    data_array = svmbir.recon(sino=np.array(data_array[:, top_slice: bottom_slice+1, :]),
                              angles=np.array(list_of_angles_rad),
                              num_rows=image_height,
                              num_cols=image_width,
                              center_offset=0,
                              **dict_parameters
                              )
    
    print(" done!")
    return data_array
