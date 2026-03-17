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
from ipywidgets import interactive
import numpy as np
from numpy.typing import NDArray
from tomopy.misc.corr import remove_outlier
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
    
    fig0 = None  # Placeholder for matplotlib figure
    axs = None  # Placeholder for matplotlib axes

    axs_before = None
    axs_after = None
    img_before = None
    img_after = None
    cbar_before = None
    cbar_after = None

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
        self.fig0 = None
        self.axs = None
        
        
        
        display(HTML(f"<h2>How to visualize the {data_type} data?</h2>"))
        display(HTML(f"<b>Choose one of the following options and then execute the next cell to display:</b>"))
        
        self.what_to_visualize_ui = widgets.ToggleButtons(options=['All images', 
                                                                   '1 image at a time',
                                                                   'Statistics',
                                                                   'Integrated intensity vs image index'],
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
        elif self.what_to_visualize_ui.value == 'Integrated intensity vs image index':
            self.visualize_integrated_intensity_vs_image_index()

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
        elif self.what_to_visualize_ui.value == 'Integrated intensity vs image index':
            self.visualize_integrated_intensity_vs_image_index()

    def visualize_integrated_intensity_vs_image_index(self):
           
        sample_data = self.parent.master_3d_data_array[DataType.sample]
        integrated_intensity = sample_data.sum(axis=(1,2))

        fig = go.Figure()
        fig.add_trace(go.Scatter(y=integrated_intensity, mode='lines+markers'))
        fig.update_layout(
            title=f"Integrated intensity vs image index ({self.mode} data)",
            xaxis_title="Angle (degrees)",
            yaxis_title="Integrated intensity (a.u.)",
            height=500,
            width=700,
        )
        fig.show()

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

        # Row 1: Sample min, OB min, DC max
        row1_titles = ["Sample (np.min)"]
        row1_data = [sample_proj_min]
        row1_zmin = [vmin]
        row1_zmax = [vmax]

        if not no_ob_data:
            row1_titles.append("OB (np.min)")
            row1_data.append(ob_proj_min)
            row1_zmin.append(vmin)
            row1_zmax.append(vmax)

        if not no_dc_data:
            row1_titles.append("DC (np.max)")
            row1_data.append(dc_proj_max)
            row1_zmin.append(vmin)
            row1_zmax.append(vmin + 1000)

        if row1_data:
            ncols_row1 = len(row1_data)
            fig0 = make_subplots(rows=1, cols=ncols_row1,
                                 horizontal_spacing=0.15,
                                 subplot_titles=tuple(row1_titles))
            for i, (data, zmin_val, zmax_val) in enumerate(zip(row1_data, row1_zmin, row1_zmax), start=1):
                fig0.add_trace(go.Heatmap(z=data, colorscale='Viridis',
                                          zmin=zmin_val, zmax=zmax_val,
                                          coloraxis=f'coloraxis{i}'), row=1, col=i)
            fig0.update_yaxes(autorange='reversed')
            coloraxis_dict = {}
            x_positions = np.linspace(0.3, 1.0, ncols_row1)
            for i, (zmin_val, zmax_val) in enumerate(zip(row1_zmin, row1_zmax), start=1):
                key = f'coloraxis{i}'
                coloraxis_dict[key] = dict(colorscale='Viridis', cmin=zmin_val, cmax=zmax_val,
                                           colorbar=dict(x=float(x_positions[i-1]), len=0.9, thickness=15))
            fig0.update_layout(height=500, width=1000, **coloraxis_dict)
            fig0.show()

        # Row 2: First angle, last angle
        fig1 = make_subplots(rows=1, cols=2,
                             horizontal_spacing=0.15,
                             subplot_titles=(f"Sample at angle {list_of_angles[0]}",
                                             f"Sample at angle {list_of_angles[-1]}"))
        fig1.add_trace(go.Heatmap(z=sample_proj_first, colorscale='Viridis',
                                  zmin=vmin, zmax=vmax,
                                  coloraxis='coloraxis1'), row=1, col=1)
        fig1.add_trace(go.Heatmap(z=sample_proj_last, colorscale='Viridis',
                                  zmin=vmin, zmax=vmax,
                                  coloraxis='coloraxis2'), row=1, col=2)
        fig1.update_yaxes(autorange='reversed')
        fig1.update_layout(height=500, width=1000,
                           coloraxis1=dict(colorscale='Viridis', cmin=vmin, cmax=vmax,
                                           colorbar=dict(x=0.44, len=0.9, thickness=15)),
                           coloraxis2=dict(colorscale='Viridis', cmin=vmin, cmax=vmax,
                                           colorbar=dict(x=1.0, len=0.9, thickness=15)))
        fig1.show()
   
        if (self.mode == 'cleaned') and (self.parent.histogram_sample_before_cleaning is not None):

            # display histogram of sample before and after
            flatten_raw_histogram = self.parent.histogram_sample_before_cleaning.flatten()
            raw_counts, raw_bin_edges = np.histogram(flatten_raw_histogram, bins=100)

            edge_nbr_pixels = clean_paras['edge_nbr_pixels']

            corrected_data = np.array(self.parent.master_3d_data_array[DataType.sample])
            histogram_corrected_data = corrected_data.sum(axis=0)[edge_nbr_pixels: -edge_nbr_pixels,
                                                        edge_nbr_pixels: -edge_nbr_pixels]
            flatten_corrected_histogram = histogram_corrected_data.flatten()
            cleaned_counts, cleaned_bin_edges = np.histogram(flatten_corrected_histogram, bins=100)

            fig_hist = make_subplots(rows=2, cols=1,
                                    subplot_titles=('raw sample histogram', 'cleaned sample histogram'))
            fig_hist.add_trace(go.Bar(x=raw_bin_edges[:-1], y=raw_counts, name='raw'), row=1, col=1)
            fig_hist.add_trace(go.Bar(x=cleaned_bin_edges[:-1], y=cleaned_counts, name='cleaned'), row=2, col=1)
            fig_hist.update_yaxes(type='log', row=1, col=1)
            fig_hist.update_yaxes(type='log', row=2, col=1)
            fig_hist.update_layout(height=600, width=800, showlegend=False)
            fig_hist.show()

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

            # if we combine the images, the index of the image before will match the index of the image after after correction
            nbr_images_before = len(data_before)
            nbr_images_after = len(data_after)
            if nbr_images_before != nbr_images_after:
                coeff = nbr_images_before // nbr_images_after
            else:
                coeff = 1

            # list_of_images = self.parent.list_of_images[DataType.sample]
            nbr_images = len(data_after)

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

                    _norm_data = data_after[image_index]
                    # _run_number = list_of_images[image_index]
                    _raw_data = data_before[image_index * coeff]
                    
                    fig = make_subplots(rows=2, cols=1,
                                        subplot_titles=(label_before, label_after),
                                        vertical_spacing=0.08)
                    fig.add_trace(go.Heatmap(z=_raw_data, colorscale='Viridis',
                                            zmin=vmin_before, zmax=vmax_before,
                                            showscale=True), row=1, col=1)
                    fig.add_trace(go.Heatmap(z=_norm_data, colorscale='Viridis',
                                            zmin=vmin_after, zmax=vmax_after,
                                            showscale=True), row=2, col=1)
                    fig.update_yaxes(autorange='reversed')
                    fig.update_layout(height=800, width=500)
                    fig.show()

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

                    _norm_data = data_after[image_index]
                    # _run_number = list_of_images[image_index]
                    _raw_data = data_before[image_index * coeff]

                    fig = make_subplots(rows=1, cols=2,
                                        subplot_titles=(label_before, label_after))
                    fig.add_trace(go.Heatmap(z=_raw_data, colorscale='Viridis',
                                            showscale=True), row=1, col=1)
                    fig.add_trace(go.Heatmap(z=_norm_data, colorscale='Viridis',
                                            showscale=True), row=1, col=2)
                    fig.update_yaxes(autorange='reversed')
                    fig.update_layout(height=500, width=1000)
                    fig.show()

                display_plot = interactive(plot_norm,
                                        image_index=widgets.IntSlider(min=0,
                                                                    max=nbr_images-1,
                                                                    value=0),
                )
                
            display(display_plot)


        else:
            o_review = FinalProjectionsReview(parent=self.parent)
            o_review.run(array=data_after,
                         auto_vrange=True)

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

            subplot_titles = [f"{i}" for i in range(nbr_images)] + [""] * (nbr_rows * nbr_cols - nbr_images)
            fig = make_subplots(rows=nbr_rows, cols=nbr_cols,
                                subplot_titles=subplot_titles,
                                horizontal_spacing=0.02,
                                vertical_spacing=0.05)

            for _index in range(nbr_images):
                _row = _index // nbr_cols + 1
                _col = _index % nbr_cols + 1
                fig.add_trace(go.Heatmap(z=array[_index], colorscale='Viridis',
                                         showscale=False), row=_row, col=_col)

            fig.update_yaxes(autorange='reversed', showticklabels=False)
            fig.update_xaxes(showticklabels=False)
            fig.update_layout(height=nbr_rows * 200, width=nbr_cols * 150)
            fig.show()

    def visualize_normalized_images(self):
     
        normalized_images = self.parent.normalized_images

        def plot_images(image_index=0):

            _norm_data = normalized_images[image_index]

            fig = go.Figure(go.Heatmap(z=_norm_data, colorscale='Viridis',
                                       zmin=0, zmax=1))
            fig.update_yaxes(autorange='reversed')
            fig.update_layout(title="Normalized data", height=500, width=500)
            fig.show()

        display_plot = interactive(plot_images,
                                image_index=widgets.IntSlider(min=0,
                                                              max=len(normalized_images)-1,
                                                              value=0),
        )

        display(display_plot)

    def visualize_2_stacks(self, 
                           left=None, vmin_left=None, vmax_left=None, 
                           right=None, vmin_right=None, vmax_right=None):

        if vmin_left is None:
            vmin_left = np.min(left)
        if vmax_left is None:             
            vmax_left = np.max(left)
        if vmin_right is None:
            vmin_right = np.min(right)
        if vmax_right is None:
            vmax_right = np.max(right)

        self.vmin_left = vmin_left
        self.vmax_left = vmax_left
        self.vmin_right = vmin_right
        self.vmax_right = vmax_right
        
        # get default vleft and vright ranges        
        default_vmin_left = float(np.percentile(left, 2))
        default_vmax_left = float(np.percentile(left, 98))
        default_vmin_right = float(np.percentile(right, 2))
        default_vmax_right = float(np.percentile(right, 98))
        
        def plot_images(index=0, vrange_before=None, vrange_after=None):

            if vrange_before is not None:
                vmin_left = vrange_before[0]
                vmax_left = vrange_before[1]

            if vrange_after is not None:
                vmin_right = vrange_after[0]
                vmax_right = vrange_after[1]

            fig = make_subplots(rows=2, cols=1,
                                subplot_titles=("normalized", "log(normalized)"),
                                vertical_spacing=0.12)
            fig.add_trace(go.Heatmap(z=left[index], colorscale='Viridis',
                                     zmin=vmin_left, zmax=vmax_left,
                                     coloraxis='coloraxis1'), row=1, col=1)
            fig.add_trace(go.Heatmap(z=right[index], colorscale='Viridis',
                                     zmin=vmin_right, zmax=vmax_right,
                                     coloraxis='coloraxis2'), row=2, col=1)
            fig.update_yaxes(autorange='reversed')
            fig.update_layout(height=700, width=500,
                              coloraxis1=dict(colorscale='Viridis', cmin=vmin_left, cmax=vmax_left,
                                              colorbar=dict(y=0.77, len=0.4)),
                              coloraxis2=dict(colorscale='Viridis', cmin=vmin_right, cmax=vmax_right,
                                              colorbar=dict(y=0.23, len=0.4)))
            fig.show()

        display_plot = interactive(plot_images,
                                index=widgets.IntSlider(min=0,
                                                        max=len(left)-1,
                                                        layout=widgets.Layout(width='50%'),
                                                        value=0),
                                vrange_before=widgets.FloatRangeSlider(min=vmin_left, 
                                                                    max=vmax_left, 
                                                                    value=[default_vmin_left, default_vmax_left],
                                                                    continuous_update=False,
                                                                    layout=widgets.Layout(width='50%')),
                                vrange_after=widgets.FloatRangeSlider(min=vmin_right, 
                                                                    max=vmax_right,
                                                                    value=[default_vmin_right, default_vmax_right],
                                                                    continuous_update=False,
                                                                    layout=widgets.Layout(width='50%'))     
                                
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
            
            if self.vmin is None:
                self.vmin = np.min(data[index])
            if self.vmax is None:
                self.vmax = np.max(data[index])

            fig = go.Figure(go.Heatmap(z=data[index], colorscale='Viridis',
                                       zmin=self.vmin, zmax=self.vmax))
            fig.update_yaxes(autorange='reversed')
            fig.update_layout(title=title, height=500, width=500)
            fig.show()

            
        _display_plot_images = interactive(plot_images,
                                index=widgets.IntSlider(min=0,
                                                        layout=widgets.Layout(width='80%'),
                                                        max=len(data)-1,
                                                        continuous_update=False,
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

        # Figure 1: Sample min and OB min side by side
        fig0 = make_subplots(rows=1, cols=2,
                             horizontal_spacing=0.15,
                             subplot_titles=("Sample (np.min)", "OB (np.min)"))
        fig0.add_trace(go.Heatmap(z=sample_proj_min, colorscale='Viridis',
                                  coloraxis='coloraxis1'), row=1, col=1)
        fig0.add_trace(go.Heatmap(z=ob_proj_min, colorscale='Viridis',
                                  coloraxis='coloraxis2'), row=1, col=2)
        fig0.update_yaxes(autorange='reversed')  # match imshow orientation
        fig0.update_layout(height=500, width=1000,
                           coloraxis1=dict(colorscale='Viridis',
                                           colorbar=dict(x=0.44, len=0.9, thickness=15)),
                           coloraxis2=dict(colorscale='Viridis',
                                           colorbar=dict(x=1.0, len=0.9, thickness=15)))
        fig0.show()

        # Figure 2: First angle, last angle, and ratio
        fig1 = make_subplots(rows=1, cols=3,
                             horizontal_spacing=0.15,
                             subplot_titles=(f"Sample at angle {list_of_angles[0]}",
                                             f"Sample at angle {list_of_angles[-1]}",
                                             "Ratio last/first"))
        fig1.add_trace(go.Heatmap(z=sample_proj_first, colorscale='Viridis',
                                  coloraxis='coloraxis1'), row=1, col=1)
        fig1.add_trace(go.Heatmap(z=sample_proj_last, colorscale='Viridis',
                                  coloraxis='coloraxis2'), row=1, col=2)
        fig1.add_trace(go.Heatmap(z=ratio_last_first, colorscale='Viridis',
                                  coloraxis='coloraxis3'), row=1, col=3)
        fig1.update_yaxes(autorange='reversed')
        fig1.update_layout(height=500, width=1000,
                           coloraxis1=dict(colorscale='Viridis',
                                           colorbar=dict(x=0.24, len=0.9, thickness=15)),
                           coloraxis2=dict(colorscale='Viridis',
                                           colorbar=dict(x=0.61, len=0.9, thickness=15)),
                           coloraxis3=dict(colorscale='Viridis',
                                           cmin=0.9, cmax=1.1,
                                           colorbar=dict(x=1.0, len=0.9, thickness=15)))
        fig1.show()

        if (self.mode == 'cleaned') and (self.parent.histogram_sample_before_cleaning is not None):

            # display histogram of sample before and after
            flatten_raw_histogram = self.parent.histogram_sample_before_cleaning.flatten()
            raw_counts, raw_bin_edges = np.histogram(flatten_raw_histogram, bins=100)

            edge_nbr_pixels = clean_paras['edge_nbr_pixels']

            corrected_data = np.array(self.parent.master_3d_data_array[DataType.sample])
            histogram_corrected_data = corrected_data.sum(axis=0)[edge_nbr_pixels: -edge_nbr_pixels,
                                                        edge_nbr_pixels: -edge_nbr_pixels]
            flatten_corrected_histogram = histogram_corrected_data.flatten()
            cleaned_counts, cleaned_bin_edges = np.histogram(flatten_corrected_histogram, bins=100)

            fig_hist = make_subplots(rows=2, cols=1,
                                    subplot_titles=('raw sample histogram', 'cleaned sample histogram'))
            fig_hist.add_trace(go.Bar(x=raw_bin_edges[:-1], y=raw_counts, name='raw'), row=1, col=1)
            fig_hist.add_trace(go.Bar(x=cleaned_bin_edges[:-1], y=cleaned_counts, name='cleaned'), row=2, col=1)
            fig_hist.update_yaxes(type='log', row=1, col=1)
            fig_hist.update_yaxes(type='log', row=2, col=1)
            fig_hist.update_layout(height=600, width=800, showlegend=False)
            fig_hist.show()
