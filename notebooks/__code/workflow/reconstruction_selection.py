"""
Reconstruction Algorithm Selection for CT Reconstruction Pipeline.

This module provides functionality for selecting reconstruction algorithms that
will be used in the CT reconstruction process. It presents an interactive interface
for choosing from available algorithms with support for multiple algorithm selection.

Key Classes:
    - ReconstructionSelection: Main class for reconstruction algorithm selection workflow

Key Features:
    - Interactive multi-selection interface for reconstruction algorithms
    - Support for multiple algorithm selection with CTRL+click
    - Integration with configuration management system
    - Real-time feedback and logging of selected algorithms
    - Default algorithm pre-selection for user convenience

Available Algorithms:
    - Filtered Back Projection (FBP)
    - Algebraic Reconstruction Technique (ART)
    - Simultaneous Iterative Reconstruction Technique (SIRT)
    - Maximum Likelihood Expectation Maximization (MLEM)
    - Additional algorithms from TomoPy library

Dependencies:
    - ipywidgets: Interactive widget creation and management
    - IPython: Jupyter notebook display functionality

Author: CT Reconstruction Pipeline Team
Created: Part of CT reconstruction development workflow
"""

import ipywidgets as widgets
from IPython.display import display
from IPython.display import HTML
from ipywidgets import interactive
import logging
from typing import List, Tuple, Any
import numpy as np

from __code.parent import Parent
from  __code.utilities.general import retrieve_list_class_attributes_name
from __code import DEFAULT_RECONSTRUCTION_ALGORITHM
from __code.utilities.configuration_file import ReconstructionAlgorithm


class ReconstructionSelection(Parent):
    """
    Reconstruction algorithm selection for CT reconstruction workflow.
    
    This class provides an interactive interface for selecting one or more
    reconstruction algorithms that will be used in the CT reconstruction process.
    It supports multiple algorithm selection and integrates with the configuration
    management system to store user preferences.
    
    Attributes:
        multi_reconstruction_selection_ui: Multi-selection widget for algorithm choice
    
    Methods:
        select: Display interactive algorithm selection interface
        on_change: Handle algorithm selection changes and update configuration
    """

    def select(self, default_selection: Tuple[str, ...] = None) -> None:   
        """
        Display interactive reconstruction algorithm selection interface.
        
        This method creates and displays a multi-selection widget that allows users
        to choose one or more reconstruction algorithms from the available options.
        The interface supports multiple selection using CTRL+click and provides
        clear instructions for the user.
        
        The selected algorithms are automatically stored in the parent configuration
        and can be used by subsequent reconstruction workflow steps.
        """

        if default_selection is None:
            default_selection = DEFAULT_RECONSTRUCTION_ALGORITHM

        # get all the attribute names of the ReconstructionAlgorithm class
        list_algo: List[str] = retrieve_list_class_attributes_name(ReconstructionAlgorithm)

        display(widgets.HTML("<font size=5 color=blue>Select reconstruction algorithm(s)</font>"))
        display(widgets.HTML("<font size=3 color=black>Multiple selection allowed by <b>CTRL+click</b></font>"))

        self.multi_reconstruction_selection_ui: widgets.SelectMultiple = widgets.SelectMultiple(
                                                                        options=list_algo,
                                                                        rows=len(list_algo),
                                                                        description="",
                                                                        value=default_selection,
        )

        self.multi_reconstruction_selection_ui.observe(self.on_change, names='value')

        display(widgets.VBox([self.multi_reconstruction_selection_ui]))
      
    def on_change(self, change: dict) -> None:
        """
        Handle algorithm selection changes and update configuration.
        
        This method is called whenever the user changes the selected reconstruction
        algorithms. It updates the parent configuration with the new selection and
        logs the change for tracking purposes.
        
        Args:
            change: Widget change event containing new selected values
                   Expected format: {'new': tuple_of_selected_algorithms}
        """
        selected_values: Tuple[str, ...] = change['new']
        self.parent.configuration.reconstruction_algorithm = selected_values
        logging.info(f"selected reconstruction algorithm: {selected_values}")
        