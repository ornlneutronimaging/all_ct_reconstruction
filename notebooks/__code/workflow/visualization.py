"""
Visualization Utilities for CT Reconstruction Pipeline.

This module provides comprehensive visualization functionality for computed tomography
data analysis and quality control. It includes interactive plotting tools, statistical
analysis displays, and data cleaning visualization for various stages of the
reconstruction pipeline.

Key Classes:
    - Visualization: Main class for CT data visualization and analysis

Key Features:
    - Interactive data visualization with customizable display modes
    - Statistical analysis and outlier detection visualization
    - Raw and cleaned data comparison displays
    - Progress tracking for large dataset visualization
    - Integration with TomoPy outlier removal algorithms
    - Customizable plotting parameters and gamma correction

Dependencies:
    - matplotlib: Core plotting and visualization functionality
    - IPython: Jupyter notebook widget integration
    - tomopy: Outlier detection and data cleaning algorithms
    - numpy: Numerical operations for data analysis

Author: CT Reconstruction Pipeline Team
Created: Part of CT reconstruction development workflow
"""

from typing import Optional, Union, List, Dict, Any
from IPython.display import display, HTML
import ipywidgets as widgets
import matplotlib.pyplot as plt
from ipywidgets import interactive
import numpy as np
from numpy.typing import NDArray
from tomopy.misc.corr import remove_outlier

from __code.parent import Parent
from __code import DataType
from __code.workflow.final_projections_review import FinalProjectionsReview
from __code.config import clean_paras, GAMMA_DIFF, NUM_THREADS

class Visualization(Parent):
    """
    Visualization and analysis tools for CT reconstruction data.
    
    This class provides comprehensive visualization capabilities for CT data
    analysis including interactive plotting, statistical analysis, and data
    quality assessment. It supports both raw and processed data visualization
    with customizable display parameters.
    
    Inherits from Parent class which provides access to reconstruction pipeline
    state, working directories, and data arrays.
    
    Key Features:
        - Interactive data visualization with widget controls
        - Statistical analysis and outlier detection displays
        - Raw vs cleaned data comparison visualization
        - Customizable gamma correction and display parameters
        - Integration with TomoPy data cleaning algorithms
        - Progress tracking for large dataset analysis
    
    Attributes
    ----------
    mode : str
        Current visualization mode ('raw' or 'cleaned')
    what_to_visualize_ui : ipywidgets.ToggleButtons
        Widget for selecting visualization type
    
    Examples
    --------
    >>> viz = Visualization(parent=parent_instance)
    >>> viz.how_to_visualize(DataType.raw)
    >>> viz.visualize_according_to_selection(mode='cleaned')
    """

    mode: str = 'raw'  # 'cleaned'

    def how_to_visualize(self, data_type: DataType = DataType.raw) -> None:
        """
        Display visualization mode selection interface.
        
        Provides interactive widget for selecting how to visualize CT data,
        offering options between comprehensive image display and statistical
        analysis modes.
        
        Parameters
        ----------
        data_type : DataType, default=DataType.raw
            Type of data to visualize (raw, normalized, etc.)
            
        Notes
        -----
        - Creates toggle button widget for mode selection
        - Displays section header with data type information
        - Sets up UI for subsequent visualization calls
        
        Side Effects
        ------------
        Creates and displays what_to_visualize_ui widget
        """
        display(HTML(f"<hr><h2>How to visualize the {data_type} data?</h2>"))
        self.what_to_visualize_ui = widgets.ToggleButtons(options=['All images', 
                                                                   '1 image at a time',
                                                                   'Statistics'],
                                                          value='Statistics')
        display(self.what_to_visualize_ui)

    def visualize_according_to_selection(self, mode: str = 'cleaned') -> None:
        """
        Execute visualization based on user selection and mode.
        
        Dispatches to appropriate visualization method based on the
        selected visualization type and data processing mode.
        
        Parameters
        ----------
        mode : str, default='cleaned'
            Data processing mode ('raw' or 'cleaned')
            
        Notes
        -----
        - Routes to visualize_all_images_at_once() or visualize_statistics()
        - Based on what_to_visualize_ui widget selection
        - Sets internal mode for subsequent processing
        """
        # for QHY data
        self.mode = mode
        if self.what_to_visualize_ui.value == 'All images':
            self.visualize_all_images_at_once()
        elif self.what_to_visualize_ui.value == 'Statistics':
            self.visualize_statistics()
        elif self.what_to_visualize_ui.value == '1 image at a time':
            self.visualize_1_stack(data=self.parent.master_3d_data_array[DataType.sample],
                                   title=f"{self.mode} data")

    def visualize_timepix_according_to_selection(self, mode='cleaned'):
        # for timepix data
        self.mode = mode
        if self.what_to_visualize_ui.value == 'All images':
            self.visualize_all_images_at_once()
        elif self.what_to_visualize_ui.value == 'Statistics':
            self.visualize_timepix_statistics()
        elif self.what_to_visualize_ui.value == '1 image at a time':
            self.visualize_1_stack(data=self.parent.master_3d_data_array[DataType.sample],
                                   title=f"{self.mode} data")

    def visualize_statistics(self):

        if self.mode in ['raw', 'cleaned']:
            master_3d_data_array = self.parent.master_3d_data_array
        else:
            raise NotImplementedError(f"mode {self.mode} not implemented")
            
        sample_data = master_3d_data_array[DataType.sample]
        ob_data = master_3d_data_array[DataType.ob]
        dc_data = master_3d_data_array[DataType.dc]
        list_of_angles = self.parent.final_list_of_angles
        
        if ob_data is None:
            no_ob_data = True
            no_dc_data = True
        else:
            no_ob_data = False
            if dc_data is None:
                no_dc_data = True
            else:
                no_dc_data = False

        # if no_ob_data:
        #     vmax = sample_data.max()
        # else:
        #     vmax = ob_data.max()
        # vmin = sample_data.min()

        if no_ob_data:
            vmax = sample_data.max()
        else:
            vmax = remove_outlier(ob_data[0], GAMMA_DIFF, ncore=NUM_THREADS).astype(np.ushort).max()
        vmin = remove_outlier(sample_data[0], GAMMA_DIFF, ncore=NUM_THREADS).astype(np.ushort).min()

        # np.min of sample
        sample_proj_min = np.min(sample_data, axis=0)

        if not no_ob_data:
            # np.min of ob
            ob_proj_min = np.min(ob_data, axis=0)

        if not no_dc_data:
            # np.max of dark current
            dc_proj_max = np.max(dc_data, axis=0)

        # projection of first image loaded
        sample_proj_first = sample_data[0]
        
        # projection of last image loaded
        sample_proj_last = sample_data[-1]

        # # ratio firt / last
        # ratio_last_first = sample_proj_last / sample_proj_first
        nrows = 2

        fig, axs = plt.subplots(nrows=nrows, ncols=3, figsize=(15, 15))

        im0 = axs[0, 0].imshow(sample_proj_min, vmin=vmin, vmax=vmax)
        axs[0, 0].set_title("Sample (np.min)")
        plt.colorbar(im0, ax=axs[0, 0], shrink=0.5)

        if not no_ob_data:
            im1 = axs[0, 1].imshow(ob_proj_min, vmin=vmin, vmax=vmax)
            axs[0, 1].set_title("OB (np.min)")
            plt.colorbar(im1, ax=axs[0, 1], shrink=0.5)
        else:
            axs[0, 1].axis('off')

        if not no_dc_data:
            im2 = axs[0, 2].imshow(dc_proj_max, vmin=vmin, vmax=vmin+1000)
            axs[0, 2].set_title("DC (np.max)")
            plt.colorbar(im2, ax=axs[0, 2], shrink=0.5)
        else:
            axs[0, 2].axis('off')

        im3 = axs[1, 0].imshow(sample_proj_first, vmin=vmin, vmax=vmax)
        axs[1, 0].set_title(f"Sample at angle {list_of_angles[0]}")
        plt.colorbar(im3, ax=axs[1, 0], shrink=0.5)

        im4 = axs[1, 1].imshow(sample_proj_last, vmin=vmin, vmax=vmax)
        axs[1, 1].set_title(f"Sample at angle {list_of_angles[-1]}")
        plt.colorbar(im4, ax=axs[1, 1], shrink=0.5)

        # im5 = axs[1, 2].imshow(ratio_last_first, vmin=0.9, vmax=1.1)
        # axs[1, 2].set_title("Ratio last/first")
        # plt.colorbar(im5, ax=axs[1, 2], shrink=0.5)
        axs[1, 2].axis('off')


        # im0 = axs[0, 0].imshow(sample_proj_min, vmin=vmin, vmax=vmax)
        # axs[0, 0].set_title("Sample (np.min)")
        # plt.colorbar(im0, ax=axs[0, 0], shrink=0.5)

        # if no_ob_data:
        #     im1 = axs[0, 1].imshow(ob_proj_min, vmin=vmin, vmax=vmax)   
        #     axs[0, 1].set_title("OB (np.min)")
        #     plt.colorbar(im1, ax=axs[0, 1], shrink=0.5)

        # if not no_dc_data:
        #     im2 = axs[0, 2].imshow(dc_proj_max, vmin=vmin, vmax=vmin+1000)
        #     axs[0, 2].set_title("DC (np.max)")
        #     plt.colorbar(im2, ax=axs[0, 2], shrink=0.5)

        # im3 = axs[1, 0].imshow(sample_proj_first, vmin=vmin, vmax=vmax)
        # axs[1, 0].set_title(f"Sample at angle {list_of_angles[0]}")
        # plt.colorbar(im3, ax=axs[1, 0], shrink=0.5)

        # im4 = axs[1, 1].imshow(sample_proj_last, vmin=vmin, vmax=vmax)
        # axs[1, 1].set_title(f"Sample at angle {list_of_angles[-1]}")
        # plt.colorbar(im4, ax=axs[1, 1], shrink=0.5)

        # im5 = axs[1, 2].imshow(ratio_last_first, vmin=0.9, vmax=1.1)
        # axs[1, 2].set_title("Ratio last/first")
        # plt.colorbar(im5, ax=axs[1, 2], shrink=0.5)
   
        if (self.mode == 'cleaned') and (self.parent.histogram_sample_before_cleaning is not None):

            # display histogram of sample before and after
            fig, axs = plt.subplots(nrows=2, ncols=1)
            
            flatten_raw_histogram = self.parent.histogram_sample_before_cleaning.flatten()
            # _, sample_bin_edges = np.histogram(flatten_raw_histogram, bins=100, density=False)
            axs[0].hist(flatten_raw_histogram, bins=100)
            axs[0].set_title('raw sample histogram')
            axs[0].set_yscale('log')

            edge_nbr_pixels = clean_paras['edge_nbr_pixels']

            corrected_data = np.array(self.parent.master_3d_data_array[DataType.sample])
            histogram_corrected_data = corrected_data.sum(axis=0)[edge_nbr_pixels: -edge_nbr_pixels,
                                                        edge_nbr_pixels: -edge_nbr_pixels]
            flatten_corrected_histogram = histogram_corrected_data.flatten()
            axs[1].hist(flatten_corrected_histogram, bins=100)
            axs[1].set_title('cleaned sample histogram')
            axs[1].set_yscale('log')

            plt.tight_layout()
            plt.show()

    def settings(self):
        self.display_ui = widgets.ToggleButtons(options=['1 image at a time',
                                                         'All images'],
                                                         description="How to plot?",
                                                         )
        display(self.display_ui)

    def visualize(self, data_before=None, data_after=None, label_before="", label_after="", 
                  turn_on_vrange=False, 
                  vmin=None, 
                  vmax=None,
                  vmin_after=None,
                  vmax_after=None):

        if self.display_ui.value == '1 image at a time':

            # list_of_images = self.parent.list_of_images[DataType.sample]
            nbr_images = len(self.parent.master_3d_data_array[DataType.sample])

            if turn_on_vrange:
                
                if vmin is None:
                    vmin_before = np.min(data_before)
                else:
                    vmin_before = vmin

                if vmax is None:
                    vmax_before = np.max(data_before)
                else:
                    vmax_before = vmax

                if vmin_after is None:
                    vmin_after = np.min(data_after)
            
                if vmax_after is None:
                    vmax_after = np.max(data_after)

                def plot_norm(image_index=0, 
                              v_before=None, 
                              v_after=None
                              ):

                    vmin_before, vmax_before = v_before
                    vmin_after, vmax_after = v_after

                    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

                    _norm_data = data_after[image_index]
                    # _run_number = list_of_images[image_index]
                    _raw_data = data_before[image_index]

                    im0 = axs[0].imshow(_raw_data, vmin=vmin_before, vmax=vmax_before)
                    axs[0].set_title(label_before)
                    plt.colorbar(im0, ax=axs[0], shrink=0.5)

                    im1 = axs[1].imshow(_norm_data, vmin=vmin_after, vmax=vmax_after)
                    axs[1].set_title(label_after)
                    plt.colorbar(im1, ax=axs[1], shrink=0.5)
            
                    # fig.set_title(f"{_run_number}")
                    
                    plt.tight_layout()
                    # plt.show()

                display_plot = interactive(plot_norm,
                                        image_index=widgets.IntSlider(min=0,
                                                                        max=nbr_images-1,
                                                                        continuous_update=False,
                                                                        layout=widgets.Layout(width='80%'),
                                                                        value=0),
                                        v_before=widgets.IntRangeSlider(min=vmin_before, 
                                                                        max=vmax_before, 
                                                                        value=[vmin_before, vmax_before],
                                                                        continuous_update=False,
                                                                        layout=widgets.Layout(width='80%')),
                                        v_after=widgets.FloatRangeSlider(min=vmin_after, 
                                                                            max=vmax_after, 
                                                                            value=[vmin_after, vmax_after],
                                                                            continuous_update=False,
                                                                            layout=widgets.Layout(width='80%')),
                                        # vmin_before=widgets.IntSlider(min=vmin_before, 
                                        #                               max=vmax_before, 
                                        #                               value=vmin_before,
                                        #                               continuous_update=False,
                                        #                               layout=widgets.Layout(width='50%')),
                                        # vmax_before=widgets.IntSlider(min=vmin_before, 
                                        #                               max=vmax_before, 
                                        #                               value=vmax_before,
                                        #                               continuous_update=False,
                                        #                               layout=widgets.Layout(width='50%')),
                                        # vmin_after=widgets.FloatSlider(min=vmin_after, 
                                        #                                max=vmax_after,
                                        #                                value=vmin_after,
                                        #                                continuous_update=False,
                                        #                                layout=widgets.Layout(width='50%')),
                                        # vmax_after=widgets.FloatSlider(min=vmin_after, 
                                        #                                max=vmax_after, 
                                        #                                value=vmax_after,
                                        #                                continuous_update=False,
                                        #                                layout=widgets.Layout(width='50%')),
                                                                       )

            else:

                def plot_norm(image_index=0):

                    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

                    _norm_data = data_after[image_index]
                    # _run_number = list_of_images[image_index]
                    _raw_data = data_before[image_index]

                    im0 = axs[0].imshow(_raw_data)
                    axs[0].set_title(label_before)
                    plt.colorbar(im0, ax=axs[0], shrink=0.5)

                    im1 = axs[1].imshow(_norm_data)
                    axs[1].set_title(label_after)
                    plt.colorbar(im1, ax=axs[1], shrink=0.5)
            
                    # fig.set_title(f"{_run_number}")
                    
                    plt.tight_layout()
                    plt.show()

                display_plot = interactive(plot_norm,
                                        image_index=widgets.IntSlider(min=0,
                                                                    max=nbr_images-1,
                                                                    value=0),
                )
                
            display(display_plot)


        else:
            o_review = FinalProjectionsReview(parent=self.parent)
            o_review.run(array=data_after)

    def visualize_all_images_at_once(self):
        
        master_3d_data_array = self.parent.master_3d_data_array

        nbr_cols = 8

        for _data_type in master_3d_data_array.keys():

            if master_3d_data_array[_data_type] is None:
            # if not self.parent.list_of_images[_data_type]:
                continue
            
            display(HTML(f"<b>{_data_type}</b>"))
            array = master_3d_data_array[_data_type]
            # nbr_images = len(self.parent.list_of_images[_data_type])
            nbr_images = len(self.parent.master_3d_data_array[_data_type])
            nbr_rows = int(np.ceil(nbr_images / nbr_cols))

            fig, axs =  plt.subplots(nrows=nbr_rows, ncols=nbr_cols,
                                    figsize=(nbr_cols*2,nbr_rows*2))
            flat_axs = axs.flatten()

            _index = 0
            list_runs_with_infos = []
            for _row in np.arange(nbr_rows):
                for _col in np.arange(nbr_cols):
                    _index = _col + _row * nbr_cols
                    if _index == (nbr_images):
                        break
                    # title = f"{list_runs[_index]}, {list_angles[_index]}"
                    # list_runs_with_infos.append(title)
                    # flat_axs[_index].set_title(title)
                    im1 = flat_axs[_index].imshow(array[_index])
                    plt.colorbar(im1, ax=flat_axs[_index], shrink=0.5)
            
            for _row in np.arange(nbr_rows):
                for _col in np.arange(nbr_cols):
                    _index = _col + _row * nbr_cols
                    flat_axs[_index].axis('off')

            plt.tight_layout()
            plt.show()

    def visualize_normalized_images(self):
     
        normalized_images = self.parent.normalized_images

        def plot_images(image_index=0):

            fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(7, 7))

            _norm_data = normalized_images[image_index]
            
            im = axs.imshow(_norm_data, vmin=0, vmax=1)
            axs.set_title("Normalized data")
            plt.colorbar(im, ax=axs, shrink=0.5)

            plt.tight_layout()
            plt.show()

        display_plot = interactive(plot_images,
                                image_index=widgets.IntSlider(min=0,
                                                              max=len(normalized_images)-1,
                                                              value=0),
        )

        display(display_plot)

    def visualize_2_stacks(self, 
                           left=None, vmin_left=None, vmax_left=None, 
                           right=None, vmin_right=None, vmax_right=None):

        self.vmin_left = vmin_left
        self.vmax_left = vmax_left
        self.vmin_right = vmin_right
        self.vmax_right = vmax_right

        def plot_images(index=0):

            fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

            if self.vmin_left is None:
                self.vmin_left = np.min(left[index])
            if self.vmax_left is None:
                self.vmax_left = np.max(left[index])

            im0 = axs[0].imshow(left[index], vmin=self.vmin_left, vmax=self.vmax_left, )
            axs[0].set_title("normalized")
            plt.colorbar(im0, ax=axs[0], shrink=0.5)

            if self.vmin_right is None:
                self.vmin_right = np.min(right[index])
            if self.vmax_right is None:
                self.vmax_right = np.max(right[index])   

            # im1 = axs[1].imshow(right[index], vmin=self.vmin_right, vmax=self.vmax_right)
            im1 = axs[1].imshow(right[index])
            axs[1].set_title("log(normalized)")
            plt.colorbar(im1, ax=axs[1], shrink=0.5)

            plt.tight_layout()
            plt.show()

        display_plot = interactive(plot_images,
                                index=widgets.IntSlider(min=0,
                                                        max=len(left)-1,
                                                        layout=widgets.Layout(width='50%'),
                                                        value=0),
        )
        display(display_plot)

    def visualize_1_stack(self,
                          data=None, 
                          vmin=None, 
                          vmax=None,
                          title="normalized"):
        
        self.vmin = vmin
        self.vmax = vmax

        def plot_images(index=0):
            fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(7, 7))

            if self.vmin is None:
                self.vmin = np.min(data[index])
            if self.vmax is None:
                self.vmax = np.max(data[index])

            im = axs.imshow(data[index], vmin=self.vmin, vmax=self.vmax)
            axs.set_title(title)
            plt.colorbar(im, ax=axs, shrink=0.5)

            plt.tight_layout()
            
        _display_plot_images = interactive(plot_images,
                                index=widgets.IntSlider(min=0,
                                                        layout=widgets.Layout(width='80%'),
                                                        max=len(data)-1,
                                                        value=0),
        )
        display(_display_plot_images)

    def visualize_timepix_statistics(self):

        if self.mode in ['raw', 'cleaned']:
            master_3d_data_array = self.parent.master_3d_data_array
        else:
            raise NotImplementedError(f"mode {self.mode} not implemented")
            
        sample_data = master_3d_data_array[DataType.sample]
        ob_data = master_3d_data_array[DataType.ob]
        list_of_angles = self.parent.final_list_of_angles
      
        vmax = ob_data.max()
        vmin = sample_data.min()

        vmax = remove_outlier(ob_data[0], GAMMA_DIFF, ncore=NUM_THREADS).astype(np.ushort).max()
        vmin = remove_outlier(sample_data[0], GAMMA_DIFF, ncore=NUM_THREADS).astype(np.ushort).min()

        # np.min of sample
        sample_proj_min = np.min(sample_data, axis=0)

        # np.min of ob
        ob_proj_min = np.min(ob_data, axis=0)

        # projection of first image loaded
        sample_proj_first = sample_data[0]
        
        # projection of last image loaded
        sample_proj_last = sample_data[-1]

        # ratio firt / last
        ratio_last_first = sample_proj_last / sample_proj_first

        fig0, axs0 = plt.subplots(nrows=1, ncols=2, figsize=(15, 7))

        im0 = axs0[0].imshow(sample_proj_min, vmin=vmin, vmax=vmax)
        axs0[0].set_title("Sample (np.min)")
        plt.colorbar(im0, ax=axs0[0], shrink=0.5)

        im1 = axs0[1].imshow(ob_proj_min, vmin=vmin, vmax=vmax)   
        axs0[1].set_title("OB (np.min)")
        plt.colorbar(im1, ax=axs0[1], shrink=0.5)

        fig1, axs1 = plt.subplots(nrows=1, ncols=3, figsize=(15, 7))

        im3 = axs1[0].imshow(sample_proj_first, vmin=vmin, vmax=vmax)
        axs1[0].set_title(f"Sample at angle {list_of_angles[0]}")
        plt.colorbar(im3, ax=axs1[0], shrink=0.5)

        im4 = axs1[1].imshow(sample_proj_last, vmin=vmin, vmax=vmax)
        axs1[1].set_title(f"Sample at angle {list_of_angles[-1]}")
        plt.colorbar(im4, ax=axs1[1], shrink=0.5)

        im5 = axs1[2].imshow(ratio_last_first, vmin=0.9, vmax=1.1)
        axs1[2].set_title("Ratio last/first")
        plt.colorbar(im5, ax=axs1[2], shrink=0.5)

        if (self.mode == 'cleaned') and (self.parent.histogram_sample_before_cleaning is not None):

            # display histogram of sample before and after
            fig, axs = plt.subplots(nrows=2, ncols=1)
            
            flatten_raw_histogram = self.parent.histogram_sample_before_cleaning.flatten()
            # _, sample_bin_edges = np.histogram(flatten_raw_histogram, bins=100, density=False)
            axs[0].hist(flatten_raw_histogram, bins=100)
            axs[0].set_title('raw sample histogram')
            axs[0].set_yscale('log')

            edge_nbr_pixels = clean_paras['edge_nbr_pixels']

            corrected_data = np.array(self.parent.master_3d_data_array[DataType.sample])
            histogram_corrected_data = corrected_data.sum(axis=0)[edge_nbr_pixels: -edge_nbr_pixels,
                                                        edge_nbr_pixels: -edge_nbr_pixels]
            flatten_corrected_histogram = histogram_corrected_data.flatten()
            axs[1].hist(flatten_corrected_histogram, bins=100)
            axs[1].set_title('cleaned sample histogram')
            axs[1].set_yscale('log')

            plt.tight_layout()
            plt.show()
