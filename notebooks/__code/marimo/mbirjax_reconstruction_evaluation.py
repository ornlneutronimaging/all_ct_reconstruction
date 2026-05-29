import time
import numpy as np
import mbirjax as mj
import os
import logging
import numpy as np

from __code.config import MARIMO_TEST_RECONSTRUCTION_WIDTH
from __code.utilities.logging import setup_logging

LOG_BASENAME_FILENAME, _ = os.path.splitext(os.path.basename(__file__))
setup_logging(basename_of_log_file=LOG_BASENAME_FILENAME)


class MbirjaxReconstructionEvaluation:
    
    def __init__(self, data, list_angles_deg, reconstruction_parameters):
        self.corrected_array_log = data
        self.list_angles_rad = np.deg2rad(list_angles_deg)
        
        self.top_slice = reconstruction_parameters.get("top_slice", 0)
        self.bottom_slice = reconstruction_parameters.get("bottom_slice", self.corrected_array_log.shape[0])
            
        self.positivity = reconstruction_parameters.get("positivity", True)
        self.max_iterations = reconstruction_parameters.get("max_iterations", 100)
        self.sharpness = reconstruction_parameters.get("sharpness", 0.0)
        self.snr_db = reconstruction_parameters.get("snr_db", 30)
        self.det_channel_offset = reconstruction_parameters.get("det_channel_offset", 0)
        self.row_scale = reconstruction_parameters.get("row_scale", 1.0)
        self.col_scale = reconstruction_parameters.get("col_scale", 1.0)
        
        logging.addLevelName(logging.INFO, "MBIRJAX_RECONSTRUCTION_EVALUATION")
        logging.info("Initialized MbirjaxReconstructionEvaluation with parameters:")
        logging.info(f"top_slice={self.top_slice}, bottom_slice={self.bottom_slice}, ")
        logging.info(f"positivity={self.positivity}, max_iterations={self.max_iterations}, ")
        logging.info(f"sharpness={self.sharpness}, snr_db={self.snr_db}, ")
        logging.info(f"det_channel_offset={self.det_channel_offset}, ")
        logging.info(f"row_scale={self.row_scale}, col_scale={self.col_scale}")
        
    def evaluate(self):
                
        _, n_slices, _ = self.corrected_array_log.shape # (n_angles, n_slices, n_det_channels)
        
        top_slice = self.top_slice
        if top_slice < 0 or top_slice >= n_slices:
            top_slice = MARIMO_TEST_RECONSTRUCTION_WIDTH // 2
            
        bottom_slice = self.bottom_slice
        if bottom_slice <= 0 or bottom_slice > n_slices:
            bottom_slice = n_slices - MARIMO_TEST_RECONSTRUCTION_WIDTH // 2
        
        # reconstruction of top slices
        top_reconstruction_slice, top_recond_dict, top_reconstruction_time = self._reconstruct_slices(from_slice=top_slice-MARIMO_TEST_RECONSTRUCTION_WIDTH//2, 
                                                                             to_slice=top_slice+MARIMO_TEST_RECONSTRUCTION_WIDTH//2)
        
        # reconstruction of bottom slices
        bottom_reconstruction_slice, bottom_recond_dict, bottom_reconstruction_time = self._reconstruct_slices(from_slice=bottom_slice-MARIMO_TEST_RECONSTRUCTION_WIDTH//2, 
                                                                                   to_slice=bottom_slice+MARIMO_TEST_RECONSTRUCTION_WIDTH//2)
        
        return top_reconstruction_slice, bottom_reconstruction_slice, top_reconstruction_time, bottom_reconstruction_time
           
    def _reconstruct_slices(self, from_slice, to_slice):
        
        logging.info(f"Reconstructing slices from {from_slice} to {to_slice}...")
        
        _sinogram = self.corrected_array_log[:, from_slice : to_slice, :]
        logging.info(f"\t{np.shape(_sinogram) = }")
        
        sinogram_shape = _sinogram.shape
        logging.info(f"\t{sinogram_shape = }")
        
        logging.info(f"\t{self.list_angles_rad = }")
        
        top_ct_model = mj.ParallelBeamModel(sinogram_shape,
                                            self.list_angles_rad)
        top_ct_model.scale_recon_shape(row_scale=self.row_scale, col_scale=self.col_scale)
        top_ct_model.set_params(sharpness=self.sharpness, 
                                snr_db=self.snr_db, 
                                det_channel_offset=self.det_channel_offset,
                                positivity_flag=self.positivity)
        start_time = time.perf_counter()
        reconstruction_array, recond_dict = top_ct_model.recon(_sinogram, max_iterations=self.max_iterations)
        reconstruction_array = np.swapaxes(reconstruction_array, 0, 2)  # swap rows and cols to match the original orientation
        logging.info(f"\t{reconstruction_array.shape = }")
        elapsed_time = time.perf_counter() - start_time
        logging.info(f"\tReconstruction time: {elapsed_time:.2f} s")
        
        middle_slice = MARIMO_TEST_RECONSTRUCTION_WIDTH // 2
        logging.info(f"\tReconstruction of middle slice {middle_slice} completed.")
        slice = reconstruction_array[middle_slice, :, :]
        logging.info(f"\t{slice.shape = }")
        logging.info(f"\t{type(slice) = }")
        
        return slice, recond_dict, elapsed_time