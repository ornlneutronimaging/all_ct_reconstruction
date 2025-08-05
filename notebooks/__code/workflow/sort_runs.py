"""
Run Sorting Utilities for CT Reconstruction Pipeline.

This module provides functionality to sort CT projection runs by their rotation
angles in ascending order. Proper angular ordering is essential for tomographic
reconstruction algorithms that expect monotonically increasing projection angles.

Key Classes:
    - SortRuns: Main class for run sorting operations

Key Features:
    - Automatic sorting by rotation angle
    - Maintains correspondence between runs and angles
    - Logging of sorting results for validation
    - Integration with parent data structures

Mathematical Background:
    Tomographic reconstruction requires projection data acquired at known angular
    positions. Most reconstruction algorithms expect these angles to be in
    ascending order for proper sinogram construction and backprojection.

Dependencies:
    - logging: Progress tracking and validation
    - Parent: Base class providing common functionality

Author: CT Reconstruction Pipeline Team
Created: Part of CT reconstruction development workflow
"""

import logging
from typing import List, Union
from __code.parent import Parent
from __code import DataType, Run


class SortRuns(Parent):
    """
    Handles sorting of CT projection runs by rotation angle.
    
    This class provides functionality to sort projection runs in ascending
    order of their rotation angles, which is required for proper tomographic
    reconstruction. It maintains the correspondence between run identifiers
    and their associated angles.
    
    Methods:
        run(): Execute the sorting process
    """

    def run(self) -> None:
        """
        Sort projection runs by ascending rotation angle.
        
        Sorts the list of projection runs based on their associated rotation
        angles in ascending order. This ensures proper angular ordering for
        tomographic reconstruction algorithms.
        
        Returns:
            None: Modifies parent data structures in place
            
        Side Effects:
            - Updates self.parent.list_of_runs_to_use[DataType.sample] with sorted runs
            - Creates self.parent.list_of_angles_to_use_sorted with corresponding angles
            - Logs the sorted run and angle lists for validation
            
        Notes:
            - Uses Python's built-in sorted() with key function for stable sorting
            - Maintains one-to-one correspondence between runs and angles
            - Essential preprocessing step for reconstruction algorithms
        """

        logging.info(f"Sorting the runs by increasing angle value!")
        list_of_angles: List[float] = []
        list_of_runs: List[str] = self.parent.list_of_runs_to_use[DataType.sample]
        for _run in list_of_runs:
            list_of_angles.append(self.parent.list_of_runs[DataType.sample][_run][Run.angle])

        # sort the angles and sort the runs the same way
        index_sorted: List[int] = sorted(range(len(list_of_angles)), key=lambda k: list_of_angles[k])
        list_of_runs_sorted: List[str] = [list_of_runs[_index] for _index in index_sorted]
        list_of_angles_sorted: List[float] = [list_of_angles[_index] for _index in index_sorted]

        logging.info(f"\t{list_of_runs_sorted = }")
        logging.info(f"\t{list_of_angles_sorted = }")

        self.parent.list_of_runs_to_use[DataType.sample] = list_of_runs_sorted
        self.parent.list_of_angles_to_use_sorted = list_of_angles_sorted
        