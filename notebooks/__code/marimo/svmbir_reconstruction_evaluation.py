import time
import numpy as np
import os
import logging
from scipy.ndimage import rotate

try:
    import svmbir
    HAS_SVMBIR = True
except ImportError:
    HAS_SVMBIR = False

from __code.config import (
    MARIMO_SVMBIR_TEST_RECONSTRUCTION_WIDTH,
    NUM_THREADS,
    SVMBIR_LIB_PATH,
    SVMBIR_LIB_PATH_BACKUP,
    SVMBIR_LIB_PATH_BACKUP_2,
)
from __code.utilities.logging import setup_logging

LOG_BASENAME_FILENAME, _ = os.path.splitext(os.path.basename(__file__))
setup_logging(basename_of_log_file=LOG_BASENAME_FILENAME)


def _resolve_svmbir_lib_path():
    """Return the first writable svmbir library path, or None if none is."""
    for _path in (SVMBIR_LIB_PATH, SVMBIR_LIB_PATH_BACKUP, SVMBIR_LIB_PATH_BACKUP_2):
        if os.access(_path, os.W_OK):
            return _path
    return None


class SvmbirReconstructionEvaluation:

    def __init__(self, data, list_angles_deg, reconstruction_parameters, init_recon=None):
        self.list_angles_rad = np.deg2rad(list_angles_deg)

        # optional previous reconstruction reused as the starting point for this
        # run; a dict {"top": <full recon volume>, "bottom": <full recon volume>}
        # in svmbir's native recon orientation, or None for a fresh start
        self.init_recon = init_recon or {}

        # full reconstruction volumes produced by evaluate(), kept so the next
        # run can pass them back in as init_recon
        self.top_full_reconstruction = None
        self.bottom_full_reconstruction = None

        self.top_slice = reconstruction_parameters.get("top_slice", 0)
        self.bottom_slice = reconstruction_parameters.get("bottom_slice", data.shape[0])

        svmbir_config = reconstruction_parameters.get("svmbir_config", {})
        self.positivity = svmbir_config.get("positivity", True)
        self.max_iterations = svmbir_config.get("max_iterations", 200)
        self.max_resolutions = svmbir_config.get("max_resolutions", 3)
        self.sharpness = svmbir_config.get("sharpness", 0.0)
        self.snr_db = svmbir_config.get("snr_db", 30)
        self.center_offset = svmbir_config.get("center_offset", 0)
        self.verbose = svmbir_config.get("verbose", False)

        # svmbir caches its system matrix on disk; pick the first writable folder
        self.svmbir_lib_path = _resolve_svmbir_lib_path()

        logging.addLevelName(logging.INFO, "SVMBIR_RECONSTRUCTION_EVALUATION")
        logging.info("Initialized SvmbirReconstructionEvaluation with parameters:")
        logging.info(f"top_slice={self.top_slice}, bottom_slice={self.bottom_slice}, ")
        logging.info(f"positivity={self.positivity}, max_iterations={self.max_iterations}, ")
        logging.info(f"max_resolutions={self.max_resolutions}, sharpness={self.sharpness}, ")
        logging.info(f"snr_db={self.snr_db}, center_offset={self.center_offset}, ")
        logging.info(f"verbose={self.verbose}, svmbir_lib_path={self.svmbir_lib_path}")

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
            top_slice = MARIMO_SVMBIR_TEST_RECONSTRUCTION_WIDTH // 2

        bottom_slice = self.bottom_slice
        if bottom_slice <= 0 or bottom_slice > n_slices:
            bottom_slice = n_slices - MARIMO_SVMBIR_TEST_RECONSTRUCTION_WIDTH // 2

        # reconstruction of top slices, reusing the previous top reconstruction
        # as the starting point when available
        top_reconstruction_slice, top_full, top_recond_dict, top_reconstruction_time = self._reconstruct_slices(from_slice=top_slice-MARIMO_SVMBIR_TEST_RECONSTRUCTION_WIDTH//2,
                                                                             to_slice=top_slice+MARIMO_SVMBIR_TEST_RECONSTRUCTION_WIDTH//2,
                                                                             init_recon=self.init_recon.get("top"))

        # reconstruction of bottom slices, reusing the previous bottom reconstruction
        bottom_reconstruction_slice, bottom_full, bottom_recond_dict, bottom_reconstruction_time = self._reconstruct_slices(from_slice=bottom_slice-MARIMO_SVMBIR_TEST_RECONSTRUCTION_WIDTH//2,
                                                                                   to_slice=bottom_slice+MARIMO_SVMBIR_TEST_RECONSTRUCTION_WIDTH//2,
                                                                                   init_recon=self.init_recon.get("bottom"))

        # keep the full volumes so the caller can feed them into the next run
        self.top_full_reconstruction = top_full
        self.bottom_full_reconstruction = bottom_full

        return top_reconstruction_slice, bottom_reconstruction_slice, top_reconstruction_time, bottom_reconstruction_time

    def _reconstruct_slices(self, from_slice, to_slice, init_recon=None):

        if not HAS_SVMBIR:
            raise ImportError("svmbir is not installed in this environment.")

        logging.info(f"Reconstructing slices from {from_slice} to {to_slice}...")

        _sinogram = self.corrected_array_log[:, from_slice : to_slice, :]
        logging.info(f"\t{np.shape(_sinogram) = }")

        sinogram_shape = _sinogram.shape
        logging.info(f"\t{sinogram_shape = }")

        logging.info(f"\t{self.list_angles_rad = }")

        # svmbir reconstructs onto a (num_slices, num_rows, num_cols) grid; both
        # spatial dimensions are tied to the number of detector channels
        num_cols = sinogram_shape[2]
        expected_recon_shape = (sinogram_shape[1], num_cols, num_cols)

        # reuse a previous reconstruction as the starting point (init_image), but
        # only when it matches the current reconstruction grid (the grid changes
        # with the selected slice range), otherwise svmbir would reject it
        init_image = 0.0
        if init_recon is not None:
            init_shape = tuple(np.shape(init_recon))
            if init_shape == expected_recon_shape:
                init_image = init_recon
                logging.info(f"\tUsing init_recon of shape {init_shape} as starting point")
            else:
                logging.info(f"\tIgnoring init_recon: shape {init_shape} != recon_shape {expected_recon_shape}")

        start_time = time.perf_counter()
        logging.info(f"\tStarting reconstruction ... ")
        reconstruction_array = svmbir.recon(sino=_sinogram,
                                            angles=self.list_angles_rad,
                                            num_rows=num_cols,
                                            num_cols=num_cols,
                                            center_offset=self.center_offset,
                                            max_resolutions=self.max_resolutions,
                                            sharpness=self.sharpness,
                                            snr_db=self.snr_db,
                                            positivity=self.positivity,
                                            max_iterations=self.max_iterations,
                                            num_threads=NUM_THREADS,
                                            verbose=self.verbose,
                                            svmbir_lib_path=self.svmbir_lib_path,
                                            init_image=init_image)

        # native recon orientation (num_slices, num_rows, num_cols), kept for
        # reuse as init_recon on the next run
        logging.info(f"\tReconstruction done!")
        full_reconstruction = np.array(reconstruction_array, dtype=np.float32)
        logging.info(f"\tfull_reconstruction.shape = {full_reconstruction.shape}")
        elapsed_time = time.perf_counter() - start_time
        logging.info(f"\tReconstruction time: {elapsed_time:.2f} s")

        middle_slice = MARIMO_SVMBIR_TEST_RECONSTRUCTION_WIDTH // 2
        logging.info(f"\tReconstruction of middle slice {middle_slice} completed.")
        result_slice = full_reconstruction[middle_slice, :, :]
        logging.info(f"\t{result_slice.shape = }")
        logging.info(f"\t{type(result_slice) = }")
        logging.info(f"************************************************************************")

        return result_slice, full_reconstruction, {}, elapsed_time
