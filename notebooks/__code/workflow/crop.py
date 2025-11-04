"""
Image cropping utilities for CT reconstruction pipeline.

This module provides functionality for interactively selecting and applying
Region of Interest (ROI) cropping to CT projection images. It supports cropping
both before and after normalization, with visual feedback through matplotlib
widgets and real-time preview of the selected region.

The cropping operation can be applied to:
- Raw projection images (before normalization)
- Normalized images (after normalization)
- Open beam (OB) and dark current (DC) images when available

The module uses interactive widgets to allow users to visually select the
optimal cropping region and immediately see the effects on image visualization.
"""

import numpy as np
import logging
from neutompy.preproc.preproc import correction_COR
import matplotlib.pyplot as plt
from ipywidgets import interactive
from IPython.display import display
import ipywidgets as widgets
from matplotlib.patches import Rectangle
from typing import Tuple, Optional, Dict, Any
from numpy.typing import NDArray

from __code.parent import Parent
from __code.config import DEBUG
from __code.utilities.configuration_file import CropRegion
from __code.config import crop_roi as default_roi
from __code import OperatingMode
from __code import DataType


class Crop(Parent):
    """
    Interactive image cropping class for CT reconstruction pipeline.
    
    This class provides an interactive interface for selecting and applying
    Region of Interest (ROI) cropping to CT projection images. It supports
    both pre-normalization and post-normalization cropping with real-time
    visual feedback.
    
    The cropping operation reduces memory usage, processing time, and focuses
    reconstruction on the relevant portion of the sample. The class maintains
    the crop region settings in the parent configuration for use in subsequent
    processing steps.
    
    Attributes:
        before_normalization: Flag indicating whether cropping is applied before
                            or after normalization
        display_roi: Interactive widget for ROI selection
    """

    before_normalization: bool = False

    def set_region(self, before_normalization: bool = False) -> None:
        """
        Display interactive widget for selecting the crop region.
        
        Creates an interactive plot with sliders to define the ROI boundaries.
        The method automatically detects whether to use debug default values
        or full image dimensions, and provides real-time visualization of the
        selected crop region.
        
        Args:
            before_normalization: If True, crop raw projection images before
                                normalization. If False, crop normalized images.
                                
        Note:
            The visualization uses the minimum projection along the first axis
            to show features clearly. Default crop values are loaded from
            configuration when DEBUG mode is enabled.
        """

        self.before_normalization = before_normalization
    
        _data: NDArray[np.generic]
        if before_normalization:
            _data = self.parent.master_3d_data_array[DataType.sample]
        else:
           _data = self.parent.normalized_images
        integrated_min: NDArray[np.generic] = np.min(_data, axis=0)
        integrated_mean: NDArray[np.generic] = np.mean(_data, axis=0)

        height: int
        width: int
        height, width = integrated_min.shape

        # Set default crop boundaries based on debug mode
        default_left: int
        default_right: int
        default_top: int
        default_bottom: int
        
        if DEBUG:
            default_left = default_roi[OperatingMode.white_beam]['left']
            default_right = default_roi[OperatingMode.white_beam]['right']
            default_top = default_roi[OperatingMode.white_beam]['top']
            default_bottom = default_roi[OperatingMode.white_beam]['bottom'] 
        else:
            default_left = 0
            default_top = 0
            default_right = width - 1
            default_bottom = height - 1

        # Handle negative values (offset from edges)
        if default_right < 0:
            default_right = width - abs(default_right)

        if default_bottom < 0:
            default_bottom = height - abs(default_bottom)

        max_value: float = np.max(integrated_mean)

        vmax_default_value: float
        if before_normalization:
            vmax_default_value = max_value
        else:
            vmax_default_value = 1

        def plot_crop(left_right: list, top_bottom: list, vmin_vmax: list, data_type: str) -> Tuple[int, int, int, int]:
            """
            Inner function to plot the crop region visualization.
            
            Args:
                left: Left boundary of crop region
                right: Right boundary of crop region
                top: Top boundary of crop region
                bottom: Bottom boundary of crop region
                vmin: Minimum value for colormap
                vmax: Maximum value for colormap
                
            Returns:
                Tuple of (left, right, top, bottom) crop boundaries
            """

            left: int = left_right[0]
            right: int = left_right[1]

            top: int = top_bottom[0]
            bottom: int = top_bottom[1]

            vmin: float = vmin_vmax[0]
            vmax: float = vmin_vmax[1]

            if data_type == "Min":
                integrated: NDArray[np.generic] = integrated_min
            else:
                integrated: NDArray[np.generic] = integrated_mean

            fig, axs = plt.subplots(figsize=(7,7)) 
            img = axs.imshow(integrated, vmin=vmin, vmax=vmax)
            plt.colorbar(img, ax=axs, shrink=0.5)

            crop_width: int = right - left + 1
            crop_height: int = bottom - top + 1

            axs.add_patch(Rectangle((left, top), crop_width, crop_height,
                                            edgecolor='yellow',
                                            facecolor='green',
                                            fill=True,
                                            lw=2,
                                            alpha=0.3,
                                            ),
                )     

            return left, right, top, bottom            
        
        self.display_roi = interactive(plot_crop,
                                       left_right=widgets.SelectionRangeSlider(options=list(range(width)),
                                                                                index=(default_left, default_right),
                                                                                description='Left/Right:',
                                                                                layout=widgets.Layout(width="80%"),
                                                                                continuous_update=False,
                                                                                ),

                                        top_bottom=widgets.SelectionRangeSlider(options=list(range(height)),
                                                                                index=(default_top, default_bottom),
                                                                                description='Top/Bottom:',
                                                                                layout=widgets.Layout(width="80%"),
                                                                                continuous_update=False,
                                                                                ),
                                        vmin_vmax=widgets.FloatRangeSlider(description='Vmin/Vmax:',
                                                                           options=list(np.linspace(0, max_value, num=100)),
                                                                           layout=widgets.Layout(width="80%"),
                                                                           min=0,
                                                                           max=max_value,
                                                                           continuous_update=False,
                                                                           value=(0, vmax_default_value)
                                        ),
                                        data_type=widgets.RadioButtons(options=["Min", "Mean"],
                                                                       description='Data type:',
                                                                       disabled=False,
                                                                       value="Min",
                                                                       layout=widgets.Layout(width="50%"))
        )
        display(self.display_roi)

    def run(self) -> None:
        """
        Apply the selected crop region to the image data.
        
        Extracts the crop boundaries from the interactive widget and applies
        the cropping operation to the appropriate image arrays. Updates both
        the parent's crop_region dictionary and configuration object with
        the selected region.
        
        The method handles two scenarios:
        1. Before normalization: Crops sample, OB, and DC images if available
        2. After normalization: Crops only the normalized images
        
        Note:
            The crop region is stored in the parent configuration for use
            in subsequent processing steps and for saving to configuration files.
        """
        left: int
        right: int
        top: int
        bottom: int
        left, right, top, bottom = self.display_roi.result
        
        # Store crop region in parent objects
        self.parent.crop_region = {'left': left, 'right': right, 'top': top, 'bottom': bottom}
        self.parent.configuration.crop_region = CropRegion(left=left, right=right, top=top, bottom=bottom)
        
        if self.before_normalization:
            # Crop raw data arrays before normalization
            self.parent.master_3d_data_array[DataType.sample] = np.array([image[top: bottom+1, left: right+1] 
                                                 for image in self.parent.master_3d_data_array[DataType.sample]])
            
            # Crop OB images if available
            if self.parent.master_3d_data_array[DataType.ob] is not None:
                self.parent.master_3d_data_array[DataType.ob] = np.array([image[top: bottom+1, left: right+1] 
                                                                        for image in self.parent.master_3d_data_array[DataType.ob]])
            
            # Crop DC images if available
            if self.parent.master_3d_data_array[DataType.dc] is not None:
                self.parent.master_3d_data_array[DataType.dc] = np.array([image[top: bottom+1, left: right+1] 
                                                                          for image in self.parent.master_3d_data_array[DataType.dc]])

        else:
            # Crop normalized images
            self.parent.normalized_images = np.array([image[top: bottom+1, left: right+1] 
                                                     for image in self.parent.normalized_images])
        