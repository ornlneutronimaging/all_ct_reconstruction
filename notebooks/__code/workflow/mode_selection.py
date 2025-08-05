"""
Operating Mode Selection for CT Reconstruction Pipeline.

This module provides functionality for selecting and configuring the operating mode
for computed tomography reconstruction workflows. It supports both white beam and
time-of-flight (TOF) reconstruction modes with appropriate data loading and
processing pipelines for each mode.

Key Classes:
    - ModeSelection: Main class for mode selection and workflow configuration

Operating Modes:
    - White Beam Mode: Traditional CT with monochromatic or broad spectrum X-rays
    - Time-of-Flight Mode: Energy-resolved CT using neutron TOF measurements

Key Features:
    - Interactive mode selection interface
    - Automatic workflow configuration based on selected mode
    - Data validation and minimum requirement checking
    - Mode-specific data loading and processing
    - Integration with TOF spectral analysis tools

Dependencies:
    - ipywidgets: Interactive UI components for mode selection
    - numpy: Numerical operations for data processing
    - IPython: Jupyter notebook display functionality

Author: CT Reconstruction Pipeline Team
Created: Part of CT reconstruction development workflow
"""

from typing import Optional, Any
import ipywidgets as widgets
from IPython.display import display, HTML
from ipywidgets import interactive
import logging
import numpy as np
from numpy.typing import NDArray

from __code.parent import Parent
from __code import OperatingMode, DataType
from __code import DEFAULT_OPERATING_MODE

from __code.workflow.checking_data import CheckingData
from __code.workflow.remove_rejected_runs import RemoveRejectedRuns
from __code.workflow.sort_runs import SortRuns
from __code.workflow.load import Load
from __code.workflow.tof_range_mode import TofRangeMode


class ModeSelection(Parent):
    """
    Operating mode selection and workflow configuration for CT reconstruction.
    
    This class provides interactive selection between different CT reconstruction
    modes and automatically configures the appropriate data processing workflow
    for each mode. It handles the critical decision point that determines how
    data will be processed throughout the reconstruction pipeline.
    
    Inherits from Parent class which provides access to reconstruction pipeline
    state, working directories, and configuration parameters.
    
    Operating Modes:
        - White Beam: Traditional CT with broad spectrum or monochromatic radiation
        - Time-of-Flight (TOF): Energy-resolved CT using neutron TOF measurements
    
    Key Features:
        - Interactive mode selection with toggle buttons
        - Automatic workflow configuration based on mode
        - Data validation and minimum requirement checking
        - Mode-specific data loading and processing pipelines
        - Integration with TOF spectral analysis tools
    
    Attributes
    ----------
    mode_selection_ui : ipywidgets.ToggleButtons
        Interactive widget for mode selection
    
    Examples
    --------
    >>> mode_selector = ModeSelection(parent=parent_instance)
    >>> mode_selector.select()  # Display mode selection UI
    >>> mode_selector.load()    # Execute selected workflow
    """

    def select(self) -> None:
        """
        Display interactive mode selection interface.
        
        Creates and displays a toggle button widget allowing users to choose
        between white beam and time-of-flight reconstruction modes. The selection
        determines the subsequent data processing workflow.
        
        Notes
        -----
        - Creates mode_selection_ui toggle button widget
        - Default value from DEFAULT_OPERATING_MODE configuration
        - Options: OperatingMode.white_beam, OperatingMode.tof
        - Widget must be interacted with before calling load()
        
        Side Effects
        ------------
        Creates and displays mode_selection_ui widget for user interaction
        """
        self.mode_selection_ui: widgets.ToggleButtons = widgets.ToggleButtons(
            options=[OperatingMode.white_beam, OperatingMode.tof],
            value=DEFAULT_OPERATING_MODE
        )
        display(self.mode_selection_ui)
    
    def load(self) -> None:
        """
        Execute data loading workflow based on selected operating mode.
        
        Configures and executes the appropriate data processing pipeline based
        on the user's mode selection. Handles data validation, run filtering,
        sorting, and mode-specific loading procedures.
        
        Workflow Steps:
        1. Set operating mode in parent and configuration
        2. Check minimum data requirements
        3. Remove rejected runs and sort remaining runs
        4. Load data with mode-appropriate settings
        5. For TOF mode: merge slices and setup spectral analysis
        
        Mode-Specific Processing:
        - White Beam: Standard data loading with combination
        - TOF: Data loading + slice merging + spectral file loading + TOF range setup
        
        Notes
        -----
        - Requires mode_selection_ui to be created and selected first
        - Updates parent.operating_mode and parent.configuration
        - Stops execution if minimum requirements not met
        - For TOF mode: creates merged projection data and TOF analysis tools
        
        Side Effects
        ------------
        - Updates parent object configuration and state
        - Loads and processes CT data arrays
        - For TOF mode: creates o_tof_range_mode analysis interface
        - May display error messages if requirements not met
        
        Raises
        ------
        AttributeError
            If mode_selection_ui has not been created via select() method
        """
        
        if self.mode_selection_ui.value == OperatingMode.white_beam:
            self.parent.operating_mode = OperatingMode.white_beam
        else:
            self.parent.operating_mode = OperatingMode.tof
        
        self.parent.configuration.operating_mode = self.mode_selection_ui.value

        logging.info(f"Working in {self.mode_selection_ui.value} mode")
        o_check: CheckingData = CheckingData(parent=self.parent)
        o_check.checking_minimum_requirements()
        if self.parent.minimum_requirements_met:

            o_rejected: RemoveRejectedRuns = RemoveRejectedRuns(parent=self.parent)
            o_rejected.run()

            o_sort: SortRuns = SortRuns(parent=self.parent)
            o_sort.run()

            combine_mode: bool = (self.mode_selection_ui.value == OperatingMode.white_beam)
            o_load: Load = Load(parent=self.parent)
            o_load.load_data(combine=combine_mode)

            if self.mode_selection_ui.value == OperatingMode.tof:

                master_3d_data_array: NDArray[np.floating] = self.parent.master_3d_data_array[DataType.sample]
                logging.info(f"combining all the slices:")
                logging.info(f"\t{np.shape(master_3d_data_array) =}")
                merged_all_slices: NDArray[np.floating] = np.sum(master_3d_data_array, axis=0)
                self.parent.data_3d_of_all_projections_merged = merged_all_slices
                logging.info(f"\t{np.shape(merged_all_slices) = }")
        
                o_load.load_spectra_file()

                # will display the profile of the region with lambda as x-axis
                self.parent.o_tof_range_mode = TofRangeMode(parent=self.parent)
                self.parent.o_tof_range_mode.run()

            # else:  # white beam mode
            #     self.parent.master_tof_3d_data_array = {'0': self.parent.master_3d_data_array}

        else:
            o_check.minimum_requirement_not_met()
