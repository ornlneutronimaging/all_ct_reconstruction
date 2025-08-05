"""
Rejected Run Removal for CT Reconstruction Pipeline.

This module provides functionality for removing user-selected rejected runs
from the final dataset that will be used for CT reconstruction. It processes
the run rejection selections made by users in the recap interface and updates
the run usage flags accordingly.

Key Classes:
    - RemoveRejectedRuns: Main class for processing run rejection workflow

Key Features:
    - Processing of user-selected rejected runs for sample and OB data
    - Automatic update of run usage flags in the configuration
    - Maintenance of final run lists for reconstruction
    - Comprehensive logging of rejection decisions
    - Integration with parent workflow state management

Dependencies:
    - tqdm: Progress tracking for data processing operations
    - numpy: Numerical computing support

Author: CT Reconstruction Pipeline Team
Created: Part of CT reconstruction development workflow
"""

import logging
import os
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import numpy as np

from __code.parent import Parent
from __code import DataType, Run
from __code.utilities.load import load_data_using_multithreading
from __code.utilities.files import retrieve_list_of_tif


class RemoveRejectedRuns(Parent):
    """
    Rejected run removal for CT reconstruction workflow.
    
    This class processes user-selected run rejections and updates the workflow
    configuration to exclude rejected runs from the final reconstruction dataset.
    It handles both sample and open beam run rejections while maintaining
    consistency in the run usage tracking system.
    
    Methods:
        run: Main execution method for processing rejected runs
    """

    def run(self) -> None:
        """
        Process rejected runs and update run usage flags.
        
        This method processes the lists of rejected runs selected by the user
        in the recap interface and updates the run usage flags accordingly.
        It ensures that rejected runs are excluded from the final reconstruction
        dataset while maintaining proper tracking of which runs are being used.
        
        The method handles:
        - Sample run rejection processing
        - Open beam run rejection processing
        - Run usage flag updates
        - Final run list maintenance
        - Comprehensive logging of rejection decisions
        """
        logging.info(f"removing rejected runs:")
        list_of_sample_runs_to_reject: Tuple[str, ...] = self.parent.list_of_sample_runs_to_reject_ui.value
        list_of_ob_runs_to_reject: Tuple[str, ...] = self.parent.list_of_ob_runs_to_reject_ui.value
        logging.info(f"\tlist of sample runs to reject: {list_of_sample_runs_to_reject}")
        logging.info(f"\tlist of ob runs to reject: {list_of_ob_runs_to_reject}")

        for _run in self.parent.list_of_runs[DataType.sample].keys():
            if self.parent.list_of_runs[DataType.sample][_run][Run.use_it]:
                if _run in list_of_sample_runs_to_reject:
                    logging.info(f"\t\t rejecting sample run {_run}!")
                    self.parent.list_of_runs[DataType.sample][_run][Run.use_it] = False
                else:
                    self.parent.list_of_runs[DataType.sample][_run][Run.use_it] = True
                    self.parent.list_of_runs_to_use[DataType.sample].append(_run)

        for _run in self.parent.list_of_runs[DataType.ob].keys():
            if self.parent.list_of_runs[DataType.ob][_run][Run.use_it]:
                if _run in list_of_ob_runs_to_reject:
                    self.parent.list_of_runs[DataType.ob][_run][Run.use_it] = False
                    logging.info(f"\t\t rejecting ob run {_run}!")
                else:
                    self.parent.list_of_runs[DataType.ob][_run][Run.use_it] = True
                    self.parent.list_of_runs_to_use[DataType.ob].append(_run)
