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
from __code.utilities.configuration_file import MbirjaxConfig
from __code.parent import Parent
from __code import DataType
from __code.utilities.save import make_tiff
from __code.utilities.time import get_current_time_in_special_file_name_format


class MbirjaxHandler(Parent):
    """
   
    """

    def set_settings(self) -> None:
        """
        Initialize interactive settings interface for MBIRJAX reconstruction.
        
        Creates and displays interactive widgets for configuring reconstruction
        parameters including regularization, prior models, iteration settings,
        and computational parameters.
        
        Returns:
            None: Creates interactive widget interface
        """

        mbirjaxConfig = MbirjaxConfig()

        title_label = widgets.HTML("<font size=5 color=blue>Define reconstruction settings for MBIRJAX</font")

        self.sharpness_ui = widgets.FloatSlider(min=-1,
                                           max=3,
                                           value=mbirjaxConfig.sharpness,
                                           layout=widgets.Layout(width="50%"),
                                           description="sharpness")
        self.snr_db_ui = widgets.FloatSlider(min=0,
                                        max=100,
                                        value=mbirjaxConfig.snr_db,
                                        layout=widgets.Layout(width="50%"),
                                        description="snr db")
        self.positivity_ui = widgets.Checkbox(value=mbirjaxConfig.positivity_flag,
                                         description="positivity")
        self.max_iterations_ui = widgets.IntSlider(value=mbirjaxConfig.max_iterations,
                                              min=1,
                                              max=200,
                                              layout=widgets.Layout(width="50%"),
                                              description="max itera.")
        
        vertical_widgets = widgets.VBox([title_label,
                                         self.sharpness_ui,
                                         self.snr_db_ui,
                                         self.positivity_ui,
                                         self.max_iterations_ui])
        display(vertical_widgets)

    def export_pre_reconstruction_data(self):

        logging.info(f"Preparing reconstruction data to export json and projections")

        # top_slice, bottom_slice = self.display_corrected_range.result
        sharpness = self.sharpness_ui.value
        snr_db = self.snr_db_ui.value
        positivity_flag = self.positivity_ui.value
        max_iterations = self.max_iterations_ui.value

        # update configuration
        mbirjax_config = MbirjaxConfig()
        mbirjax_config.sharpness = sharpness
        mbirjax_config.snr_db = snr_db
        mbirjax_config.positivity_flag = positivity_flag
        mbirjax_config.max_iterations = max_iterations
        self.parent.configuration.mbirjax_config = mbirjax_config

        # logging.info(f"\t{top_slice = }")
        # logging.info(f"\t{bottom_slice = }")
        logging.info(f"\t{sharpness = }")
        logging.info(f"\t{snr_db = }")
        logging.info(f"\t{positivity_flag = }")
        logging.info(f"\t{max_iterations = }")
