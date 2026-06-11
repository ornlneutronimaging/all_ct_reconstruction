import time
import numpy as np
import mbirjax as mj
import os
import logging
import numpy as np
from scipy.ndimage import rotate

from __code.config import MARIMO_MBIRJAX_TEST_RECONSTRUCTION_WIDTH
from __code.utilities.logging import setup_logging

LOG_BASENAME_FILENAME, _ = os.path.splitext(os.path.basename(__file__))
setup_logging(basename_of_log_file=LOG_BASENAME_FILENAME)


class MbirjaxReconstructionEvaluation:
    
    def __init__(self, data, list_angles_deg, reconstruction_parameters, init_recon=None):
        self.list_angles_rad = np.deg2rad(list_angles_deg)

        # optional previous reconstruction reused as the starting point for this
        # run; a dict {"top": <full recon volume>, "bottom": <full recon volume>}
        # in mbirjax's native recon orientation, or None for a fresh start
        self.init_recon = init_recon or {}

        # full reconstruction volumes produced by evaluate(), kept so the next
        # run can pass them back in as init_recon
        self.top_full_reconstruction = None
        self.bottom_full_reconstruction = None

        self.top_slice = reconstruction_parameters.get("top_slice", 0)
        self.bottom_slice = reconstruction_parameters.get("bottom_slice", data.shape[0])

        mbirjax_config = reconstruction_parameters.get("mbirjax_config", {})
        self.positivity = mbirjax_config.get("positivity", True)
        self.max_iterations = mbirjax_config.get("max_iterations", 100)
        self.sharpness = mbirjax_config.get("sharpness", 0.0)
        self.snr_db = mbirjax_config.get("snr_db", 30)
        self.det_channel_offset = mbirjax_config.get("det_channel_offset", 0)
        self.row_scale = mbirjax_config.get("row_scale", 1.0)
        self.col_scale = mbirjax_config.get("col_scale", 1.0)
        
        logging.addLevelName(logging.INFO, "MBIRJAX_RECONSTRUCTION_EVALUATION")
        logging.info("Initialized MbirjaxReconstructionEvaluation with parameters:")
        logging.info(f"top_slice={self.top_slice}, bottom_slice={self.bottom_slice}, ")
        logging.info(f"positivity={self.positivity}, max_iterations={self.max_iterations}, ")
        logging.info(f"sharpness={self.sharpness}, snr_db={self.snr_db}, ")
        logging.info(f"det_channel_offset={self.det_channel_offset}, ")
        logging.info(f"row_scale={self.row_scale}, col_scale={self.col_scale}")

        tilt = reconstruction_parameters.get("tilt", 0.0)
        perform_tilt = reconstruction_parameters.get("perform_tilt", False)
        if perform_tilt and tilt != 0.0:
            logging.info(f"Applying tilt correction of {tilt}° to all {data.shape[0]} projections ...")
            self.corrected_array_log = np.array([
                rotate(img, angle=tilt, reshape=False, order=1, mode="nearest")
                for img in data
            ])
            logging.info("Tilt correction applied.")
        else:
            self.corrected_array_log = data
        
    def evaluate(self):
                
        _, n_slices, _ = self.corrected_array_log.shape # (n_angles, n_slices, n_det_channels)
        
        top_slice = self.top_slice
        if top_slice < 0 or top_slice >= n_slices:
            top_slice = MARIMO_MBIRJAX_TEST_RECONSTRUCTION_WIDTH // 2
            
        bottom_slice = self.bottom_slice
        if bottom_slice <= 0 or bottom_slice > n_slices:
            bottom_slice = n_slices - MARIMO_MBIRJAX_TEST_RECONSTRUCTION_WIDTH // 2
        
        # reconstruction of top slices, reusing the previous top reconstruction
        # as the starting point when available
        top_reconstruction_slice, top_full, top_recond_dict, top_reconstruction_time = self._reconstruct_slices(from_slice=top_slice-MARIMO_MBIRJAX_TEST_RECONSTRUCTION_WIDTH//2,
                                                                             to_slice=top_slice+MARIMO_MBIRJAX_TEST_RECONSTRUCTION_WIDTH//2,
                                                                             init_recon=self.init_recon.get("top"))

        # reconstruction of bottom slices, reusing the previous bottom reconstruction
        bottom_reconstruction_slice, bottom_full, bottom_recond_dict, bottom_reconstruction_time = self._reconstruct_slices(from_slice=bottom_slice-MARIMO_MBIRJAX_TEST_RECONSTRUCTION_WIDTH//2,
                                                                                   to_slice=bottom_slice+MARIMO_MBIRJAX_TEST_RECONSTRUCTION_WIDTH//2,
                                                                                   init_recon=self.init_recon.get("bottom"))

        # keep the full volumes so the caller can feed them into the next run
        self.top_full_reconstruction = top_full
        self.bottom_full_reconstruction = bottom_full

        return top_reconstruction_slice, bottom_reconstruction_slice, top_reconstruction_time, bottom_reconstruction_time
           
    def _reconstruct_slices(self, from_slice, to_slice, init_recon=None):

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

        # reuse a previous reconstruction as the starting point, but only when it
        # matches the current reconstruction grid (the grid changes with the
        # row/col scale), otherwise mbirjax would reject the init_recon
        try:
            expected_recon_shape = tuple(top_ct_model.get_params("recon_shape"))
        except Exception:
            expected_recon_shape = None
        init_recon_arg = None
        if init_recon is not None:
            init_shape = tuple(np.shape(init_recon))
            if expected_recon_shape is None or init_shape == expected_recon_shape:
                init_recon_arg = init_recon
                logging.info(f"\tUsing init_recon of shape {init_shape} as starting point")
            else:
                logging.info(f"\tIgnoring init_recon: shape {init_shape} != recon_shape {expected_recon_shape}")

        # when warm-starting from a previous reconstruction, skip ahead to
        # iteration 5 so we do not repeat the early iterations; start at 0 for a
        # fresh reconstruction
        first_iteration = 5 if init_recon_arg is not None else 0
        logging.info(f"\t{first_iteration = }")
        logging.info(f"\tinit_recon_arg.shape = {init_recon_arg.shape if init_recon_arg is not None else 'no init_recon'}")

        start_time = time.perf_counter()
        logging.info(f"\tStarting reconstruction ... ")
        reconstruction_array, recond_dict = top_ct_model.recon(_sinogram,
                                                               max_iterations=self.max_iterations,
                                                               init_recon=init_recon_arg,
                                                               first_iteration=first_iteration)
        
        # native recon orientation (recon_shape), kept for reuse as init_recon
        logging.info(f"\tReconstruction done!")
        full_reconstruction = np.array(reconstruction_array, dtype=np.float32)
        reconstruction_array = np.array(np.swapaxes(full_reconstruction, 0, 2), dtype=np.float32)  # convert JAX array to numpy
        logging.info(f"\treconstruction_array.shape = {reconstruction_array.shape}")
        elapsed_time = time.perf_counter() - start_time
        logging.info(f"\tReconstruction time: {elapsed_time:.2f} s")

        middle_slice = MARIMO_MBIRJAX_TEST_RECONSTRUCTION_WIDTH // 2
        logging.info(f"\tReconstruction of middle slice {middle_slice} completed.")
        result_slice = reconstruction_array[middle_slice, :, :]
        logging.info(f"\t{result_slice.shape = }")
        logging.info(f"\t{type(result_slice) = }")
        logging.info(f"************************************************************************")

        return result_slice, full_reconstruction, recond_dict, elapsed_time