"""
SVMBIR Handler for Advanced CT Reconstruction.

This module provides comprehensive functionality for Super-Voxel Model-Based Iterative
Reconstruction (SVMBIR) in computed tomography workflows. SVMBIR is an advanced
reconstruction algorithm that provides superior image quality compared to traditional
filtered back projection, especially for limited-angle or sparse data scenarios.

Key Classes:
    - SvmbirHandler: Main class for SVMBIR reconstruction workflow

Key Features:
    - Interactive reconstruction parameter configuration
    - Support for both time-of-flight (TOF) and white beam modes
    - Advanced regularization and prior model settings
    - Multi-threading support for performance optimization
    - Progress tracking and quality control
    - Automatic data validation and preprocessing
    - Export functionality for reconstructed volumes

SVMBIR Algorithm:
    SVMBIR uses iterative optimization to solve the reconstruction problem:
    minimize ||Ax - b||² + β*R(x)
    where:
    - A: system matrix (forward projection operator)
    - x: reconstructed volume
    - b: measured projection data
    - β: regularization parameter
    - R(x): regularization function (e.g., total variation)

Dependencies:
    - svmbir: Core SVMBIR reconstruction library
    - tomopy: Preprocessing and utilities
    - numpy: Numerical computing
    - ipywidgets: Interactive controls
    - matplotlib: Visualization
    - logging: Progress tracking

Author: CT Reconstruction Pipeline Team
Created: Part of advanced CT reconstruction development workflow
"""

import numpy as np
import os
from IPython.display import display
import ipywidgets as widgets
import matplotlib.pyplot as plt
from ipywidgets import interactive
import logging
from tqdm import tqdm
from numpy.typing import NDArray


from __code.workflow.export import Export
# from __code.utilities.configuration import Configuration
from __code.utilities.files import make_or_reset_folder
from __code.utilities.configuration_file import SvmbirConfig
from __code.parent import Parent
from __code import DataType
from __code.utilities.save import make_tiff
from __code.utilities.time import get_current_time_in_special_file_name_format


class MbirjaxHandler(Parent):
    """
    """

    def set_settings(self) -> None:
        """
        Initialize interactive settings interface for Mbirjax reconstruction.
        
        Creates and displays interactive widgets for configuring reconstruction
        parameters including regularization, prior models, iteration settings,
        and computational parameters.
        
        Returns:
            None: Creates interactive widget interface
        """

        title_label = widgets.HTML("<font size=5 color=blue>Define reconstruction settings for Mbirjax</font")

        self.sharpness_ui = widgets.FloatSlider(min=-1,
                                           max=3,
                                           value=0,
                                           layout=widgets.Layout(width="50%"),
                                           description="sharpness (higher is sharper)")
        self.snr_db_ui = widgets.FloatSlider(min=0,
                                        max=50,
                                        value=30.0,
                                        layout=widgets.Layout(width="50%"),
                                        description="snr db (higher is sharper)")
        self.positivity_ui = widgets.Checkbox(value=False,
                                         description="positivity")
        self.max_iterations_ui = widgets.IntSlider(value=10,
                                              min=10,
                                              max=100,
                                              layout=widgets.Layout(width="50%"),
                                              description="max itera. (if high artifacts, increase this)")
        self.verbose_ui = widgets.Checkbox(value=True,
                                      description='verbose')
        
        vertical_widgets = widgets.VBox([title_label,
                                         self.sharpness_ui,
                                         self.snr_db_ui,
                                         self.positivity_ui,
                                         self.max_iterations_ui,
                                         self.verbose_ui])
        display(vertical_widgets)
