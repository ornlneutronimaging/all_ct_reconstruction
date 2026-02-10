"""
Normalization Workflow for CT Reconstruction Pipeline.

This module provides comprehensive normalization functionality for computed tomography
data processing. It handles the critical step of normalizing sample projections using
open beam and dark current measurements to produce calibrated transmission data.

Key Classes:
    - Normalization: Main class for CT data normalization workflow

Key Features:
    - Standard normalization using open beam and dark current corrections
    - Interactive ROI selection for normalization parameters
    - Quality control and validation of normalized data
    - Support for various normalization algorithms
    - Progress tracking and logging for large datasets
    - Export functionality for normalized projections

Normalization Formula:
    normalized = (sample - dark) / (open_beam - dark)

Dependencies:
    - tomopy: Core tomographic processing functions
    - scipy: Scientific computing and filtering
    - matplotlib: Interactive plotting and visualization
    - IPython: Jupyter notebook widget integration

Author: CT Reconstruction Pipeline Team
Created: Part of CT reconstruction development workflow
"""

from typing import Optional, Tuple, Any, List, Union
import logging
import numpy as np
from numpy.typing import NDArray
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from ipywidgets import interactive
from IPython.display import display, HTML
import ipywidgets as widgets
from scipy.ndimage import median_filter
import tomopy
from copy import copy

from __code.parent import Parent
from __code import Run, DataType, NormalizationSettings
from __code.workflow.load import Load
from __code.workflow.export import Export
from __code.utilities.files import make_or_reset_folder
from __code.utilities.logging import logging_3d_array_infos
from __code.workflow.final_projections_review import FinalProjectionsReview
from __code.config import NUM_THREADS, DEBUG, roi


# class RectangleSelector:
   
#    def __init__(self, ax):
#       self.ax = ax
#       self.start_point = None
#       self.rect = None
#       self.cid_press = ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
#       self.cid_release = ax.figure.canvas.mpl_connect('button_release_event', self.on_release)
#       self.cid_motion = ax.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)

#    def on_press(self, event):
#       if event.inaxes == self.ax:
#          self.start_point = (event.xdata, event.ydata)
#          self.rect = Rectangle(self.start_point, 0, 0, edgecolor='red', alpha=0.2)
#          self.ax.add_patch(self.rect)

#    def on_motion(self, event):
#       if self.start_point is not None and event.inaxes == self.ax:
#          width = event.xdata - self.start_point[0]
#          height = event.ydata - self.start_point[1]
#          self.rect.set_width(width)
#          self.rect.set_height(height)
#          self.ax.figure.canvas.draw()

#    def on_release(self, event):
#       if self.start_point is not None:
#          # Determine the data points within the rectangle and perform actions as needed
#          selected_data = self.get_data_within_rectangle()
#          print("Selected Data:", selected_data)
#          self.start_point = None
#          self.rect.remove()
#          self.ax.figure.canvas.draw()

#    def get_data_within_rectangle(self):
#       # Placeholder function to determine data points within the rectangle
#       # Implement logic to identify data points based on the rectangle's coordinates
#       return [(1, 2), (3, 4)]  # Example data points
   

class Normalization(Parent):

    obs_combined = None
    mean_ob_proton_charge = None
    mean_ob_frame_number = 1

    enable_frame_number = False

    do_not_run_normalization = False
    
    cbar = None
    rectangle = None
    fig = None
    axs = None
    img = None

    def normalization_settings(self):

        if self.parent.master_3d_data_array[DataType.ob] is None:
            self.do_not_run_normalization = True
            logging.warning(f"Normalization settings: No ob data found, normalization will not be applied.")
            return

        self.use_proton_charge_ui = widgets.Checkbox(value=False,
                                                description='Use proton charge',
                                                disabled=False,
                                                layout=widgets.Layout(width="100%"))
        # self.use_frames_ui = widgets.Checkbox(value=False,
        #                                  description='Use frames',
        #                                  disabled=True,
        #                                  )
        self.use_roi_ui = widgets.Checkbox(value=False,
                                      description='Use beam fluctuation correction (ROI)',
                                      layout=widgets.Layout(width="100%"))
        self.use_sample_roi_ui = widgets.Checkbox(value=False,
                                        description='Use sample ROI normalization',
                                        layout=widgets.Layout(width="100%"))
        vertical_layout = widgets.VBox([# self.use_proton_charge_ui,
                                        # self.use_frames_ui,
                                        self.use_roi_ui,
                                        self.use_sample_roi_ui])
        display(vertical_layout)

    def select_roi(self):

        if self.do_not_run_normalization:
            logging.info(f"Skipping ROI selection for normalization.")
            return
        
        if (not self.use_roi_ui.value) and (not self.use_sample_roi_ui.value):
            logging.info(f"User skipped normalization ROI selection.")
            return

        display(HTML("Note: This is an integrated view of the projections allowing you to see the contours of all the angles!"))

        # integrated_images = np.log(np.min(self.parent.master_3d_data_array[DataType.sample], axis=0))
        sample_images = self.parent.master_3d_data_array[DataType.sample]
        integrated_images = np.mean(sample_images, axis=0)
        height, width = np.shape(integrated_images)

        # self.fig, self.axs = plt.subplots(nrows=1, ncols=1, figsize=(7,7), 
        #                                   num="Normalization ROI Selection")
        # self.img = self.axs.imshow(integrated_images)
        # self.cbar = plt.colorbar(self.img, ax=self.axs, shrink=0.5)

        def plot_roi(left_right, top_bottom):

            self.fig, self.axs = plt.subplots(nrows=1, ncols=1, figsize=(7,7), 
                                            num="Normalization ROI Selection")
            self.img = self.axs.imshow(integrated_images)
            self.cbar = plt.colorbar(self.img, ax=self.axs, shrink=0.5)

            left, right = left_right
            top, bottom = top_bottom

            height = np.abs(bottom - top) + 1
            width = np.abs(right - left) + 1

            self.rectangle = Rectangle((left, top), width, height,
                                        edgecolor='yellow',
                                        facecolor='green',
                                        fill=True,
                                        lw=2,
                                        alpha=0.3,
                                        )
            self.axs.add_patch(self.rectangle)
   
            return left, right, top, bottom                       
    
        if DEBUG:
            default_left = roi[self.MODE]['left']
            default_right = roi[self.MODE]['right']
            default_top = roi[self.MODE]['top']
            default_bottom = roi[self.MODE]['bottom']
        else:
            default_left = default_top = 0
            default_right = default_bottom = 20

        if default_left >= width:
            default_left = 0
        if default_right >= width:
            default_right = width // 10
        if default_top >= height:
            default_top = 0
        if default_bottom >= height:
            default_bottom = height // 10

        self.display_roi = interactive(plot_roi,
                                       left_right=widgets.SelectionRangeSlider(options=list(range(width)),
                                                                                index=(default_left, default_right),
                                                                                description='Left-Right:',
                                                                                orientation='horizontal',
                                                                                layout=widgets.Layout(width='80%')),
                                        top_bottom=widgets.SelectionRangeSlider(options=list(range(height)),
                                                                                index=(default_top, default_bottom),
                                                                                description='Top-Bottom:',
                                                                                orientation='horizontal',
                                                                                layout=widgets.Layout(width='80%')),
                                      )
                                        
        display(self.display_roi)
      
    def normalize(self, ignore_dc=False) -> None:
        master_3d_data = self.parent.master_3d_data_array
        
        if self.parent.master_3d_data_array[DataType.ob] is None:
            self.parent.normalized_images = copy(self.parent.master_3d_data_array[DataType.sample])
            return

        size_data = np.shape(master_3d_data[DataType.sample])
        normalized_data = np.empty(size_data, dtype=np.float32)

        logging.info(f"Normalization:")

        use_proton_charge = self.use_proton_charge_ui.value
        use_roi = self.use_roi_ui.value
        use_sample_roi = self.use_sample_roi_ui.value

        logging.info(f"\tnormalization settings:")
        logging.info(f"\t\t- use_proton_charge: {use_proton_charge}")
        logging.info(f"\t\t- use_roi: {use_roi}")
        logging.info(f"\t\t- use_sample_roi: {use_sample_roi}")

        # roi sample/ob or roi sample only
        if use_roi or use_sample_roi:
            left, right, top, bottom = self.display_roi.result
            self.parent.configuration.normalization_roi.top = top
            self.parent.configuration.normalization_roi.bottom = bottom
            self.parent.configuration.normalization_roi.left = left
            self.parent.configuration.normalization_roi.right = right

        # update configuration
        list_norm_settings = []
        if use_roi:
            list_norm_settings.append(NormalizationSettings.roi)
        
        if use_sample_roi:
            list_norm_settings.append(NormalizationSettings.sample_roi)

        self.parent.configuration.list_normalization_settings = list_norm_settings

        ob_data_combined = np.squeeze(master_3d_data[DataType.ob])
        # dc_data_combined = None if (self.parent.list_of_images[DataType.dc] is None) else np.squeeze(master_3d_data[DataType.dc])
        dc_data_combined = None if (master_3d_data[DataType.dc] is None) else np.squeeze(master_3d_data[DataType.dc])

        for _index, sample_data in enumerate(master_3d_data[DataType.sample]):
          
            sample_data = np.array(master_3d_data[DataType.sample][_index], dtype=np.float32)

            coeff = 1

            if use_roi:
                sample_roi_counts = np.sum(sample_data[top: bottom+1, left: right+1])
                ob_roi_counts = np.sum(ob_data_combined[top: bottom+1, left: right+1])
                coeff *= ob_roi_counts / sample_roi_counts

            if not (dc_data_combined is None):

                normalized_sample = np.divide(np.subtract(sample_data, dc_data_combined, dtype=np.float32),
                                             np.subtract(ob_data_combined, dc_data_combined, dtype=np.float32),
                                             out=np.zeros_like(sample_data, dtype=np.float32),
                                             where=(ob_data_combined - dc_data_combined) != 0,
                                            ) * coeff
            else:
                normalized_sample = np.divide(sample_data, ob_data_combined,
                                              out=np.zeros_like(sample_data, dtype=np.float32),
                                              where=ob_data_combined != 0) * coeff
                
            if use_sample_roi:
                sample_roi_counts = np.median(normalized_sample[top: bottom+1, left: right+1])
                coeff = 1 / sample_roi_counts
                normalized_sample = normalized_sample * coeff

            # remove NaN and Inf
            # logging.info(f"removing NaN and Inf values (nan->0, -inf->0, inf->1)")
            # normalized_sample = tomopy.misc.corr.remove_nan(normalized_sample, val=0, ncore=NUM_THREADS)
            # normalized_sample = np.nan_to_num(normalized_sample, nan=0, posinf=1, neginf=0)

            # logging_3d_array_infos(message="after normalization", array=normalized_sample)
            # normalized_data.append(normalized_sample) 

            normalized_sample[normalized_sample > 1] = 1
            normalized_sample[normalized_sample < 0] = 0
            normalized_data[_index] = normalized_sample[:]

        # remove negative values
        logging.info(f"removing nan and negative values (set to 0)")
        # normalized_data = tomopy.misc.corr.remove_nan(normalized_data, val=0, ncore=NUM_THREADS)
        # normalized_data = tomopy.misc.corr.remove_neg(normalized_data, val=0, ncore=NUM_THREADS)

        self.parent.normalized_images = np.squeeze(np.asarray(normalized_data, dtype=np.float32))
        logging_3d_array_infos(message="normalized images", array=self.parent.normalized_images)

    def export_images(self):
        
        logging.info(f"Exporting the normalized images")
        logging.info(f"\tfolder selected: {self.parent.working_dir[DataType.normalized]}")

        normalized_data = self.parent.normalized_images

        master_base_folder_name = f"{os.path.basename(self.parent.working_dir[DataType.sample])}_normalized"
        full_output_folder = os.path.join(self.parent.working_dir[DataType.normalized],
                                          master_base_folder_name)

        make_or_reset_folder(full_output_folder)

        o_export = Export(image_3d=normalized_data,
                          output_folder=full_output_folder)
        o_export.run()
        logging.info(f"\texporting normalized images ... Done!")
        display(HTML(f"<b> Exported normalized images to folder: </b> {full_output_folder}"))
