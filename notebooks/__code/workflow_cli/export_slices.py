"""
Slice Export Module for CLI-Based CT Reconstruction.

This module provides command-line interface compatible functions for exporting
reconstructed CT slices to disk. It handles output folder creation, file naming
with timestamps, and integration with the main Export workflow class.

Key Functions:
    - export_slices: Main slice export function for CLI workflow

Key Features:
    - Automatic output folder generation with timestamps
    - Integration with Export workflow class
    - Configurable output directory structure
    - Progress feedback and completion status
    - Folder cleanup and organization

Dependencies:
    - Export: Main export workflow class for slice writing

Author: CT Reconstruction Pipeline Team
Created: Part of CLI-based CT reconstruction workflow
"""

import os
from typing import Any
from numpy.typing import NDArray
import numpy as np

from __code.utilities.time import get_current_time_in_special_file_name_format
from __code.utilities.files import make_or_reset_folder
from __code.workflow.export import Export


def export_slices(config_model: Any, data_array: NDArray[np.floating]) -> None:
    """
    Export reconstructed CT slices to disk with organized folder structure.
    
    This function creates an organized output directory structure and exports
    the reconstructed CT slices using the main Export workflow class. It
    automatically generates timestamped folder names for unique output locations.
    
    Args:
        config_model: Configuration object containing export parameters
                     Must have attributes:
                     - output_folder: Base output directory path
                     - top_folder.sample: Sample data source folder path
        data_array: 3D reconstructed volume array (slices x height x width)
        
    Output Structure:
        output_folder/
        └── {sample_folder_name}_reconstructed_on_{timestamp}/
            ├── slice_0000.tiff
            ├── slice_0001.tiff
            └── ...
            
    Note:
        The function provides progress feedback and reports the final output
        location upon completion. Existing folders with the same name are reset.
    """
    print(f"exporting ...", end="")
    output_folder: str = config_model.output_folder
    top_folder: str = os.path.basename(config_model.top_folder.sample)

    full_output_folder_name: str = os.path.join(output_folder, top_folder + 
                                          f"_reconstructed_on_{get_current_time_in_special_file_name_format()}")
    
    make_or_reset_folder(full_output_folder_name)
    print(f" -> {full_output_folder_name}")

    o_export: Export = Export(image_3d=data_array,
                          output_folder=full_output_folder_name)
    o_export.run()

    print(" done!")
    print(f"Slices can be found in {full_output_folder_name}!")
    