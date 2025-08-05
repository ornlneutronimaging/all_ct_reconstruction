"""
Reconstructed Slice Merging Module for CLI-Based CT Reconstruction.

This module provides functionality for merging reconstructed CT slices from
multiple output folders into a single organized directory. It handles slice
renaming, duplicate removal, and folder cleanup for parallel reconstruction
workflows.

Key Functions:
    - merge_reconstructed_slices: Main merging function for parallel reconstruction outputs

Key Features:
    - Merges slices from multiple reconstruction output folders
    - Handles slice numbering and renaming for continuous sequences
    - Removes duplicate slices from overlapping reconstructions
    - Cleanup of temporary reconstruction folders
    - Comprehensive logging and error handling

Use Case:
    When reconstruction is performed in parallel on different slice ranges,
    this function combines the results into a single coherent volume with
    proper slice numbering and no gaps or duplicates.

Dependencies:
    - glob: File pattern matching for TIFF file discovery
    - logging: Progress tracking and error reporting

Author: CT Reconstruction Pipeline Team
Created: Part of CLI-based CT reconstruction workflow
"""

import numpy as np
import os
import logging
import glob
from typing import List, Optional, Dict, Tuple


def merge_reconstructed_slices(output_data_folder: Optional[str] = None, 
                              top_slice: int = 0, 
                              list_of_output_folders: Optional[List[str]] = None, 
                              list_of_slices_to_reconstruct: Optional[List[Tuple[int, int]]] = None) -> None:
    """
    Merge reconstructed CT slices from multiple parallel reconstruction outputs.
    
    This function combines reconstruction results from multiple output folders
    into a single organized directory. It handles proper slice numbering,
    removes duplicates from overlapping reconstructions, and cleans up
    temporary directories.
    
    Args:
        output_data_folder: Final output directory for merged slices
        top_slice: Starting slice number for the final sequence
        list_of_output_folders: List of temporary output folders containing reconstructed slices
        list_of_slices_to_reconstruct: List of (start, end) slice index tuples for each folder
        
    Output Structure:
        output_data_folder/
        ├── image_000.tiff
        ├── image_001.tiff
        └── ...
        
    Process:
        1. Scan each output folder for TIFF files
        2. Map slice indices to files based on reconstruction ranges
        3. Remove duplicate slices (keep first occurrence)
        4. Rename and move slices to final output with sequential numbering
        5. Clean up temporary folders
        
    Raises:
        ValueError: If no TIFF files are found in any output folder
        OSError: If folder removal fails (logged but not fatal)
        
    Note:
        Slice numbering starts from top_slice and increments sequentially.
        Duplicate slices are removed to prevent conflicts in overlapping ranges.
    """
    
    final_output_folder: str = output_data_folder

    logging.info(f"merge reconstructed slices ...")
    logging.info(f"\t{output_data_folder = }")
    logging.info(f"\t{top_slice = }")
    logging.info(f"\t{list_of_output_folders = }")
    logging.info(f"\t{list_of_slices_to_reconstruct = }")

    list_folder_tiff: Dict[int, List[str]] = {}
    for _index, _folder in enumerate(list_of_output_folders):
        _list_tiff: List[str] = glob.glob(os.path.join(_folder, '*.tiff'))
        if len(_list_tiff) == 0:
            raise ValueError(f"no tiff files found in {_folder}")
        
        _list_tiff.sort()
        list_folder_tiff[_index] = _list_tiff

    list_slices_already_processed: List[int] = []
    for _index, [top_slice_index, bottom_slice_index] in enumerate(list_of_slices_to_reconstruct):
        logging.info(f"working with folder: {os.path.dirname(list_folder_tiff[_index][0])}")
        logging.info(f"\tfrom slice #{top_slice_index} to slice #{bottom_slice_index}")

        list_slices: np.ndarray = np.arange(top_slice_index, bottom_slice_index)
        for _tiff_index, _slice_index in enumerate(list_slices):
            if _slice_index in list_slices_already_processed:
                os.remove(list_folder_tiff[_index][_tiff_index]) # no need to move that slice, already processed

            else:
                list_slices_already_processed.append(_slice_index)
                logging.info(f"moving slice #{_slice_index} ({os.path.basename(list_folder_tiff[_index][_tiff_index])}) -> #image_{_slice_index + top_slice:03d}.tiff ... ")
                _new_tiff_file: str = os.path.join(final_output_folder, f"image_{_slice_index + top_slice:03d}.tiff")
                os.rename(list_folder_tiff[_index][_tiff_index], _new_tiff_file)
                
        # remove the input folder
        try:
            logging.info(f"removing folder {list_of_output_folders[_index]}!")
            os.rmdir(list_of_output_folders[_index])
        except OSError as e:
            logging.error(f"Error removing folder {list_of_output_folders[_index]}: {e}")
            continue
        
    logging.info(f"all slices merged into {final_output_folder}!")
