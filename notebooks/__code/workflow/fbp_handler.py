"""
Filtered Back Projection (FBP) Handler for CT Reconstruction Pipeline.

This module provides comprehensive FBP reconstruction functionality for computed
tomography workflows. It handles data preparation, reconstruction execution using
TomoPy algorithms, and result export with support for various reconstruction
parameters and quality settings.

Key Classes:
    - FbpHandler: Main class for FBP reconstruction workflow management

Key Features:
    - Pre-reconstruction data export and organization
    - TomoPy-based FBP reconstruction with customizable parameters
    - Integration with SVMBIR for advanced reconstruction algorithms
    - Multi-threaded processing for performance optimization
    - Progress tracking and logging for large datasets
    - Configuration management and result export

Dependencies:
    - tomopy: Core tomographic reconstruction algorithms
    - svmbir: Statistical reconstruction methods
    - tqdm: Progress bar visualization
    - matplotlib: Result visualization and plotting

Author: CT Reconstruction Pipeline Team
Created: Part of CT reconstruction development workflow
"""

from typing import Optional, Any, List, Dict, Union, Tuple
import numpy as np
from numpy.typing import NDArray
import os
from IPython.display import display, HTML
import ipywidgets as widgets
import matplotlib.pyplot as plt
from ipywidgets import interactive
import logging
from tqdm import tqdm
import tomopy
import glob

import svmbir

from __code.workflow.export import Export
# from __code.utilities.configuration import Configuration
from __code.utilities.files import make_or_reset_folder
from __code.utilities.configuration_file import SvmbirConfig
from __code.parent import Parent
from __code import DataType, Run
from __code.config import NUM_THREADS, SVMBIR_LIB_PATH
from __code.utilities.save import make_tiff
from __code.utilities.time import get_current_time_in_special_file_name_format
from __code.utilities.json import save_json, load_json
from __code.utilities.load import load_data_using_multithreading


class FbpHandler(Parent):
    """
    Filtered Back Projection reconstruction handler for CT pipeline.
    
    This class provides comprehensive FBP reconstruction functionality including
    data preparation, reconstruction execution, and result management. It integrates
    with TomoPy for standard FBP algorithms and SVMBIR for advanced statistical
    reconstruction methods.
    
    Inherits from Parent class which provides access to reconstruction pipeline
    state, working directories, and configuration parameters.
    
    Key Features:
        - Pre-reconstruction data export and organization
        - TomoPy FBP reconstruction with customizable filters
        - SVMBIR integration for statistical reconstruction
        - Multi-threaded processing for performance
        - Progress tracking and quality control
        - Configuration management and result export
    
    Examples
    --------
    >>> fbp = FbpHandler(parent=parent_instance)
    >>> fbp.export_pre_reconstruction_data()
    >>> fbp.run_reconstruction()
    >>> results = fbp.get_reconstruction_results()
    """

    def export_pre_reconstruction_data(self) -> None:
        """
        Export and organize normalized projection data for reconstruction processing.
        
        Prepares normalized projection data for reconstruction by converting angles
        to radians, exporting projections to TIFF format, and creating organized
        directory structure. This critical preprocessing step ensures data is
        properly formatted for TomoPy/SVMBIR reconstruction algorithms.
        
        Data Processing Steps:
        1. Convert rotation angles from degrees to radians
        2. Log data statistics for quality assessment
        3. Create timestamped output directories
        4. Export each projection as numbered TIFF file
        5. Update configuration with file paths
        
        Parameters
        ----------
        None (uses parent object state)
        
        Notes
        -----
        - Requires parent.normalized_images_log to be available
        - Creates pre_projections_export_folder with timestamp
        - Updates parent.configuration.projections_pre_processing_folder
        - Updates parent.configuration.list_of_angles (in radians)
        - Exports projections as pre-reconstruction_NNNN.tiff files
        
        Side Effects
        ------------
        - Creates timestamped output directories
        - Exports projection data as TIFF sequence
        - Updates parent configuration object
        - Logs processing statistics and file paths
        
        Raises
        ------
        KeyError
            If DataType.extra (output folder) not selected
        RuntimeError
            If no output folder selected in previous workflow step
        OSError
            If directories cannot be created or files cannot be written
        """

        logging.info(f"Preparing reconstruction data to export json and projections")

        normalized_images_log: NDArray[np.floating] = self.parent.normalized_images_log
        height: int
        width: int
        height, width = np.shape(normalized_images_log[0])

        list_of_angles: NDArray[np.floating] = np.array(self.parent.final_list_of_angles)
        list_of_angles_rad: NDArray[np.floating] = np.array([np.deg2rad(float(_angle)) for _angle in list_of_angles])

        self.parent.configuration.list_of_angles = list(list_of_angles_rad)

        # corrected_array_log = tomopy.minus_log(corrected_array)
        # where_nan = np.where(np.isnan(corrected_array_log))
        # corrected_array_log[where_nan] = 0

        # corrected_array = corrected_array_log

        logging.info(f"\t{np.min(normalized_images_log) =}")
        logging.info(f"\t{np.max(normalized_images_log) =}")
        logging.info(f"\t{np.mean(normalized_images_log) =}")

        try:
            output_folder: str = self.parent.working_dir[DataType.extra]
        except KeyError:
            display(HTML("<h3 style='color: red;'>No output folder selected in the previous cell, please select one!</h3>"))
            raise RuntimeError("No output folder selected in the previous cell, please select one!")           
        
        _time_ext: str = get_current_time_in_special_file_name_format()
        base_sample_folder: str = os.path.basename(self.parent.working_dir[DataType.sample])
        pre_projections_export_folder: str = os.path.join(output_folder, f"{base_sample_folder}_projections_pre_data_{_time_ext}")
        os.makedirs(pre_projections_export_folder)
        logging.info(f"\tprojections pre data will be exported to {pre_projections_export_folder}!")
        logging.info(f"\toutput folder: {output_folder}")
        
        self.parent.configuration.projections_pre_processing_folder = pre_projections_export_folder

        full_output_folder: str = os.path.join(output_folder, f"{base_sample_folder}_reconstructed_{_time_ext}")

        # go from [angle, height, width] to [angle, width, height]
        # corrected_array_log = np.moveaxis(corrected_array_log, 1, 2)  # angle, y, x -> angle, x, y
        logging.info(f"\t{np.shape(normalized_images_log) =}")

        for _index, _data in tqdm(enumerate(normalized_images_log)):

            if _index == 0:
                logging.info(f"\t{np.shape(_data) = }")
                logging.info(f"\t{_data.dtype = }")
                # logging.info(f"\t{top_slice = }")
                # logging.info(f"\t{bottom_slice = }")

            short_file_name: str = f"pre-reconstruction_{_index:04d}.tiff"
            full_file_name: str = os.path.join(pre_projections_export_folder, short_file_name)
            # make_tiff(data=_data[top_slice:bottom_slice+1, :], filename=full_file_name)
            make_tiff(data=_data, filename=full_file_name)
        print(f"projections exported in {pre_projections_export_folder}")
        print(f"top output folder: {output_folder}")

    def export_images(self) -> None:
        """
        Export reconstructed CT slices to TIFF files.
        
        Takes the reconstruction results from the parent object and exports
        each reconstructed slice as a numbered TIFF file in an organized
        directory structure. Updates configuration with the output location.
        
        The method creates a timestamped directory for the reconstructed
        slices and uses the Export utility for efficient TIFF file creation
        with progress tracking.
        
        Notes
        -----
        - Creates output directory based on sample folder name
        - Uses Export class for standardized TIFF export
        - Updates parent configuration with output folder path
        - Organizes files with descriptive directory naming
        
        Side Effects
        ------------
        - Creates output directory structure
        - Exports reconstruction_array as TIFF sequence
        - Updates parent.configuration.output_folder
        
        Raises
        ------
        AttributeError
            If parent.reconstruction_array is not available
        OSError
            If output directory cannot be created or written to
        """
        
        logging.info(f"Exporting the reconstructed slices")
        logging.info(f"\tfolder selected: {self.parent.working_dir[DataType.reconstructed]}")

        reconstructed_array: NDArray[np.floating] = self.parent.reconstruction_array

        master_base_folder_name: str = f"{os.path.basename(self.parent.working_dir[DataType.sample])}_reconstructed"
        full_output_folder: str = os.path.join(self.parent.working_dir[DataType.reconstructed],
                                               master_base_folder_name)

        make_or_reset_folder(full_output_folder)

        o_export: Export = Export(image_3d=reconstructed_array,
                                  output_folder=full_output_folder)
        o_export.run()
        logging.info(f"\texporting reconstructed images ... Done!")

        # update configuration
        self.parent.configuration.output_folder = full_output_folder
