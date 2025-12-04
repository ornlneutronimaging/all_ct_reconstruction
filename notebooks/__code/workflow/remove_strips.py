"""
Stripe Artifact Removal for CT Reconstruction Pipeline.

This module provides comprehensive functionality for detecting and removing stripe
artifacts in CT sinograms. Stripe artifacts are horizontal bands in sinograms that
appear as ring artifacts in reconstructed images, typically caused by detector
non-uniformities, dead pixels, or calibration issues.

Key Classes:
    - RemoveStrips: Main class for stripe artifact removal operations

Key Features:
    - Multiple stripe removal algorithms from TomoPy library
    - Interactive parameter tuning with real-time preview
    - Support for different stripe types (small, large, combined)
    - Fourier-Wavelet, Titarenko, and Vo methods
    - Sinogram visualization and quality assessment
    - Algorithm comparison and validation tools

Stripe Removal Algorithms:
    1. Fourier-Wavelet (FW): Advanced wavelet-based decomposition
    2. Titarenko: Statistical approach using ring artifact detection
    3. Vo: Robust method for various stripe patterns
    4. Combined approaches for optimal results

Mathematical Background:
    Stripe artifacts manifest as:
    - Horizontal bands in sinograms (projection space)
    - Ring artifacts in reconstructed slices (image space)
    - Frequency domain signatures that can be filtered
    
    Removal methods use various mathematical approaches:
    - Fourier domain filtering
    - Wavelet decomposition and reconstruction
    - Statistical outlier detection and correction

Dependencies:
    - tomopy: Core stripe removal algorithms
    - numpy: Numerical array operations
    - matplotlib: Sinogram visualization
    - ipywidgets: Interactive parameter controls
    - logging: Progress tracking and validation

Author: CT Reconstruction Pipeline Team
Created: Part of CT reconstruction development workflow
"""

import numpy as np
from IPython.display import display
import ipywidgets as widgets
from IPython.display import HTML
from IPython.display import Javascript
from enum import Enum
from tomopy.prep import stripe
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
from ipywidgets import interactive
import webbrowser
from matplotlib.patches import Rectangle
from typing import Optional, List, Dict, Any, Union
from numpy.typing import NDArray

from __code import DataType, RemoveStripeAlgo, OperatingMode, WhenToRemoveStripes
from __code.utilities import configuration_file
from __code.utilities.logging import logging_3d_array_infos
from __code.config import NUM_THREADS


class RemoveStrips:
    """
    Handles removal of stripe artifacts from CT sinograms.
    
    This class provides comprehensive functionality for detecting and removing
    horizontal stripe artifacts in sinograms that appear as ring artifacts in
    reconstructed CT images. It supports multiple algorithms with interactive
    parameter tuning and validation tools.
    
    Key Features:
        - Multiple TomoPy-based stripe removal algorithms
        - Interactive parameter configuration with real-time preview
        - Support for different stripe patterns and intensities
        - Algorithm comparison and validation
        - Sinogram visualization and quality assessment
        
    Attributes:
        sinogram: Current sinogram data for processing
        skip_remove_strips: Flag to bypass stripe removal
        default_list_algo_to_use: Default algorithms for automatic processing
        list_algo: Dictionary of available algorithms with configurations
        
    Methods:
        Various stripe removal methods using different algorithms
        Interactive parameter tuning interfaces
        Validation and comparison tools
        
    Stripe Types Addressed:
        - Small stripes: Narrow detector artifacts
        - Large stripes: Wide non-uniformity patterns  
        - Mixed patterns: Combination of different stripe types
    """

    _roi = None
    fig = None
    ax = None
    img = None
    cbar = None

    sinogram: Optional[NDArray[np.floating]] = None
    skip_remove_strips: bool = True

    default_list_algo_to_use: List[RemoveStripeAlgo] = [RemoveStripeAlgo.remove_stripe_fw]

    list_algo: Dict[RemoveStripeAlgo, Dict[str, Any]] = {RemoveStripeAlgo.remove_stripe_fw: {'help': 'Remove horizontal stripes from sinogram using the Fourier-Wavelet (FW) based method',
                                             'function': stripe.remove_stripe_fw,
                                             'settings': widgets.VBox([
                                                            widgets.Text(value="None",
                                                                            description="level"),
                                                            widgets.Dropdown(options=['haar', 'db5', 'sym5'],
                                                                             value='haar',
                                                                description="wname"
                                                            ),
                                                            widgets.FloatText(value=2,
                                                                            description="sigma"),
                                                            widgets.Checkbox(value=True,
                                                                            description='pad')
                                                        ]),
                },
                 RemoveStripeAlgo.remove_stripe_ti: {'help': "Remove horizontal stripes from sinogram using Titarenko's approach [B13]",
                                             'function': stripe.remove_stripe_ti,
                                              'settings': widgets.VBox([
                                                                        widgets.IntText(value=0,
                                                                            description="nblock"),
                                                                        widgets.FloatText(value=1.5,
                                                                            description="alpha"),
                                                        ]),
                 },
                 RemoveStripeAlgo.remove_stripe_sf: {'help': "Normalize raw projection data using a smoothing filter approach.",
                                             'function': stripe.remove_stripe_sf,
                                              'settings': widgets.VBox([
                                                                        widgets.IntText(value=5,
                                                                            description="size"),
                                                        ]),
                 },
                 RemoveStripeAlgo.remove_stripe_based_sorting: {'help': "Remove full and partial stripe artifacts from sinogram using Nghia Vo's approach [B24] (algorithm 3).",
                                                        'function': stripe.remove_stripe_based_sorting,
                                                         'settings': widgets.VBox([
                                                                        widgets.Text(value="None",
                                                                            description="size"),
                                                                        widgets.Dropdown(options=['1','2'],
                                                                            value='1',
                                                                            description="dim"),
                                                        ]),
                 },
                #  RemoveStripeAlgo.remove_stripe_based_filtering: {'help': "Remove stripe artifacts from sinogram using Nghia Vo's approach [B24] (algorithm 2).",
                #                                           'function': stripe.remove_stripe_based_filtering,
                #                                            'settings': widgets.VBox([
                #                                             widgets.FloatSlider(value=3,
                #                                                                 min=3,
                #                                                                 max=10,
                #                                                             description="sigma"),
                #                                             widgets.Text(value="None",
                #                                                             description="size"),
                #                                             widgets.Dropdown(options=['1','2'],
                #                                                             value='1',
                #                                                             description="dim")
                #                                         ]),
                #  },
                 RemoveStripeAlgo.remove_stripe_based_fitting: {'help': "Remove stripe artifacts from sinogram using Nghia Vo's approach [B24] (algorithm 1).",
                                                        'function': stripe.remove_stripe_based_fitting,
                                                         'settings': widgets.VBox([
                                                            widgets.IntSlider(value=3,
                                                                                min=1,
                                                                                max=5,
                                                                            description="order"),
                                                            widgets.Text(value="5,20",
                                                                            description="sigma"),
                                                        ]),
                 },
                 RemoveStripeAlgo.remove_large_stripe: {'help': "Remove unresponsive and fluctuating stripe artifacts from sinogram using Nghia Vo's approach [B24] (algorithm 6).",
                                                'function': stripe.remove_large_stripe,
                                                'settings': widgets.VBox([
                                                            widgets.FloatText(value=3,
                                                                            description="snr"),
                                                            widgets.IntText(value=51,
                                                                            description="size"),
                                                            widgets.FloatSlider(value=0.1,
                                                                                min=0,
                                                                                max=1,
                                                                            description="drop_ratio"),
                                                            widgets.Checkbox(value=True,
                                                                             description='norm')
                                                        ]),
                 },
                RemoveStripeAlgo.remove_dead_stripe: {'help': "Remove unresponsive and fluctuating stripe artifacts from sinogram using Nghia Vo's approach [B24] (algorithm 6).",
                                              'function': stripe.remove_dead_stripe,
                                              'settings': widgets.VBox([
                                                            widgets.FloatText(value=3,
                                                                            description="snr"),
                                                            widgets.IntText(value=51,
                                                                            description="size"),
                                                            widgets.Checkbox(value=True,
                                                                             description='norm'),
                                                        ]),
                },
                # RemoveStripeAlgo.remove_all_stripe: {'help': "Remove all types of stripe artifacts from sinogram using Nghia Vo's approach [B24] (combination of algorithm 3,4,5, and 6).",
                #                               'function': stripe.remove_all_stripe,
                #                               'settings': widgets.VBox([
                #                                             widgets.FloatText(value=3,
                #                                                             description="snr"),
                #                                             widgets.IntText(value=61,
                #                                                             description="la_size"),
                #                                             widgets.IntText(value=21,
                #                                                             description="sm_size"),
                #                                             widgets.Dropdown(options=[1, 2],
                #                                                             value=1,
                #                                                             description='dim')
                #                                         ]),
                #  },
                 RemoveStripeAlgo.remove_stripe_based_interpolation: {'help': "Remove most types of stripe artifacts from sinograms based on interpolation.",
                                                              'function': stripe.remove_stripe_based_interpolation,
                                                                'settings': widgets.VBox([
                                                                widgets.FloatText(value=3,
                                                                                description="snr"),
                                                                widgets.IntText(value=31,
                                                                                description="size"),
                                                                widgets.FloatText(value=.1,
                                                                                description="drop_ratio"),
                                                                widgets.Checkbox(value=True,
                                                                                description='norm')
                                                            ]),
                 },
                }
    
    list_options = list(list_algo.keys())
    list_options_widgets = None
    list_to_use_widgets = None

    nothing_to_display = True
    
    def __init__(self, parent=None):
        self.parent = parent
        self.define_default_lists()

    def select_range_of_data_to_test_stripes_removal(self):
        normalized_images = self.parent.normalized_images_log
        nbr_projections, height, width = np.shape(normalized_images)

        default_top_slice = height//2 - height//4
        default_bottom_slice = height//2 + height//4 

        default_left_slice = width//2 - width//4
        default_right_slice = width//2 + width//4

        self.fig, self.ax = plt.subplots(figsize=(5, 5), 
                                         num="Select range of data to test stripes removal")
        self.img = self.ax.imshow(normalized_images[0])
        self.cbar = plt.colorbar(self.img, ax=self.ax, shrink=0.5)
        self.image_index = 0

        def select_range_of_data(left_right=[default_left_slice, default_right_slice], 
                                 top_bottom=[default_top_slice, default_bottom_slice],
                                 image_index=0):
                
            left_slice, right_slice = left_right
            top_slice, bottom_slice = top_bottom
                
            if self._roi is not None:
                self._roi.remove()
            
            if self.image_index != image_index:
                
                if self.cbar is not None:
                    self.cbar.remove()
                
                self.image_index = image_index
            
                self.img = self.ax.imshow(normalized_images[image_index])
                self.cbar = plt.colorbar(self.img, ax=self.ax, shrink=0.5)

            width = right_slice - left_slice
            height = bottom_slice - top_slice

            self._roi = Rectangle((left_slice, top_slice), width, height,
                                                edgecolor='yellow',
                                                facecolor='green',
                                                fill=True,
                                                lw=2,
                                                alpha=0.3,
                                                )
            self.ax.add_patch(self._roi)
            
            return left_slice, right_slice, top_slice, bottom_slice

        self.range_to_use_to_test = interactive(select_range_of_data,
                                                left_right = widgets.IntRangeSlider(min=0,
                                                                                     max=width-1,
                                                                                     value=(default_left_slice,
                                                                                            default_right_slice),
                                                                                     description="Left - Right",
                                                                                     layout=widgets.Layout(width='50%')),
                                                top_bottom = widgets.IntRangeSlider(min=0,
                                                                                   max=height-1,
                                                                                   value=(default_top_slice,
                                                                                          default_bottom_slice),
                                                                                   description="Top - Bottom",
                                                                                   layout=widgets.Layout(width='50%')),
                                                image_index=widgets.IntSlider(min=0,
                                                                                max=nbr_projections-1,
                                                                                value=0,
                                                                                layout=widgets.Layout(width='50%')),
        )
        display(self.range_to_use_to_test)

    def define_default_lists(self):
        full_list_options = self.list_options
        default_list_algo_to_use = self.default_list_algo_to_use

        left_list_algo = []
        right_list_algo = default_list_algo_to_use
        for _algo in full_list_options:
            if _algo in default_list_algo_to_use:
                pass
            else:
                left_list_algo.append(_algo)

        self.default_left_list_algo = left_list_algo
        self.default_right_list_algo = right_list_algo

    def calculate_sinogram(self, data_3d):
        return np.moveaxis(data_3d, 1, 0)

    def select_algorithms(self):
        
        display(widgets.HTML("<font size=4><b>Select the algorithms to use (CTRL + CLICK to select more than one)</b></font>"))

        self.list_options = list(self.list_algo.keys())
        self.list_options_widget = widgets.SelectMultiple(
            options = self.default_left_list_algo,
            layout = widgets.Layout(height="200px")
        )
        left_text = widgets.HTML("<b>List of algorithms available</b>")
        left_widget = widgets.VBox([left_text, self.list_options_widget])
        
        button_add = widgets.Button(description=">>>")
        button_remove = widgets.Button(description="<<<")
        second_column = widgets.VBox([widgets.Label(""),
                                      button_add, button_remove])

        self.list_to_use_widget = widgets.SelectMultiple(options=self.default_right_list_algo,
                                                    layout = widgets.Layout(height="200px"))      
        right_text = widgets.HTML("<b>List of algorithms to use</b>")
        right_widget = widgets.VBox([right_text, self.list_to_use_widget])

        all_widget = widgets.HBox([left_widget,
                                   second_column,
                                   right_widget])
        display(all_widget)
        
        help_button = widgets.Button(description='Algorithms documentation ...',
                                     button_style='info',
                                     layout=widgets.Layout(width='max-content'))
        display(help_button)

        button_add.on_click(self.button_add_clicked)
        button_remove.on_click(self.button_remove_clicked)
        help_button.on_click(self.help_button_clicked)

    def define_settings(self):
        list_options_to_use = self.list_to_use_widget.options

        _children = []
        for _option in list_options_to_use:
            _children.append(self.list_algo[_option]['settings'])

        # update configuration
        self.parent.configuration.list_clean_stripes_algorithm = list_options_to_use

        accordion = widgets.Accordion(children=_children,
                                      titles=list_options_to_use)
        display(accordion)

    def button_add_clicked(self, value):
        list_to_add = self.list_options_widget.value
        list_already_added = self.list_to_use_widget.options
        full_list = list_to_add + list_already_added
        list_to_show_on_the_right = []
        list_to_display_on_the_left = []
        for _option in self.list_options:
            if _option in full_list:
                list_to_show_on_the_right.append(_option)
            else:
                list_to_display_on_the_left.append(_option)
        self.list_to_use_widget.options = list_to_show_on_the_right
        self.list_options_widget.options = list_to_display_on_the_left

    def button_remove_clicked(self, value):
        right_list_to_remove = self.list_to_use_widget.value
        right_list = self.list_to_use_widget.options

        new_list_to_use = []
        for _option in right_list:
            if _option in right_list_to_remove:
                pass
            else:
                new_list_to_use.append(_option)
        self.list_to_use_widget.options = new_list_to_use

        full_list_options = self.list_options
        new_left_list = []
        for _option in full_list_options:
            if _option in new_list_to_use:
                pass
            else:
                new_left_list.append(_option)
        self.list_options_widget.options = new_left_list

    def help_button_clicked(self, value):
        self.window_open("https://tomopy.readthedocs.io/en/latest/api/tomopy.prep.stripe.html")

    def when_to_remove_strips(self):
        label = widgets.Label(
            value="Do you want to remove stripes NOW or in the BACKGROUND when running the reconstruction?"
        )
        options = [WhenToRemoveStripes.in_notebook, WhenToRemoveStripes.out_notebook, WhenToRemoveStripes.never]
        self.when_to_remove_widget = widgets.RadioButtons(
            value=WhenToRemoveStripes.out_notebook,  # default value
            options=options,
            disabled=False)
        display(label, self.when_to_remove_widget)
        self.when_to_remove_widget.observe(self.on_change, names='value')

    def on_change(self, change):
        if change['new'] == WhenToRemoveStripes.in_notebook:
            logging.info("Strips removal will be done in the next cell (in-notebook).")
            self.parent.configuration.when_to_remove_stripes = WhenToRemoveStripes.in_notebook
            self.skip_remove_strips = False
        elif change['new'] == WhenToRemoveStripes.out_notebook:
            logging.info("Strips removal will be done in the background just before reconstruction.")
            self.parent.configuration.when_to_remove_stripes = WhenToRemoveStripes.out_notebook
            self.skip_remove_strips = False
        elif change['new'] == WhenToRemoveStripes.never:
            self.parent.configuration.when_to_remove_stripes = WhenToRemoveStripes.never
            logging.info("Strips removal will never be done.")
            self.skip_remove_strips = True
        else:
            logging.error("Unexpected option selected for when to remove stripes.")

    def run(self):
        self.perform_cleaning()
        self.display_cleaning()

    def get_keyword_arguments(self, algorithm_name=None):
        list_widgets = self.list_algo[algorithm_name]['settings'].children
        list_arguments = {}
        for _widget in list_widgets:
            _arg_name = _widget.description
            _arg_value = _widget.value
            list_arguments[_arg_name] = _arg_value

        if algorithm_name == RemoveStripeAlgo.remove_stripe_fw:
            if list_arguments['level'] == 'None':
                del list_arguments['level']
        elif algorithm_name == RemoveStripeAlgo.remove_stripe_based_sorting:
            if list_arguments['size'] == 'None':
                del list_arguments['size']
        elif algorithm_name == RemoveStripeAlgo.remove_stripe_based_filtering:
            if list_arguments['size'] == 'None':
                del list_arguments['size']
        elif algorithm_name == RemoveStripeAlgo.remove_stripe_based_fitting:
            left_value, right_value = list_arguments['sigma'].split(",")
            list_arguments['sigma'] = (int(left_value), int(right_value))

        return list_arguments

    def saving_configuration(self, algorithm_name=RemoveStripeAlgo.remove_stripe_fw):
        list_widgets = self.list_algo[algorithm_name]['settings'].children

        if algorithm_name == RemoveStripeAlgo.remove_stripe_fw:
            my_instance = configuration_file.RemoveStripeFw()
        elif algorithm_name == RemoveStripeAlgo.remove_stripe_ti:
            my_instance = configuration_file.RemoveStripeTi()
        elif algorithm_name == RemoveStripeAlgo.remove_stripe_sf:
            my_instance = configuration_file.RemoveStripeSf()
        elif algorithm_name == RemoveStripeAlgo.remove_stripe_based_sorting:
            my_instance = configuration_file.RemoveStripeBasedSorting()
        elif algorithm_name == RemoveStripeAlgo.remove_stripe_based_filtering:
            my_instance = configuration_file.RemoveStripeBasedFiltering()
        elif algorithm_name == RemoveStripeAlgo.remove_stripe_based_fitting:
            my_instance = configuration_file.RemoveStripeBasedFitting()
        elif algorithm_name == RemoveStripeAlgo.remove_large_stripe:
            my_instance = configuration_file.RemoveLargeStripe()
        elif algorithm_name == RemoveStripeAlgo.remove_dead_stripe:
            my_instance = configuration_file.RemoveDeadStripe()
        elif algorithm_name == RemoveStripeAlgo.remove_all_stripe:
            my_instance = configuration_file.RemoveAllStripe()
        elif algorithm_name == RemoveStripeAlgo.remove_stripe_based_interpolation:
            my_instance = configuration_file.RemoveStripeBasedInterpolation()
        else:
            raise NotImplementedError("filter not implemented")

        for _widget in list_widgets:
            _arg_name = _widget.description
            _arg_value = _widget.value
            setattr(my_instance, _arg_name, _arg_value)

        setattr(self.parent.configuration, f"{algorithm_name}_options", my_instance)

    def perform_cleaning(self, test=False):

        if self.parent.configuration.when_to_remove_stripes == WhenToRemoveStripes.never:
            logging.info("Strips removal was skipped.")
            print("Strips removal was skipped. No cleaning performed now.")
            return

        if not test:
            if self.parent.configuration.when_to_remove_stripes == WhenToRemoveStripes.out_notebook:
                logging.info("Strips removal will be done in the background just before reconstruction.")
                print("Strips removal was set to be done in the background. No cleaning performed now.")
                return

        list_algo_to_use = self.list_to_use_widget.options
        logging.info(f"Strip cleaning:")
       
        if test:
            left_slice, right_slice, top_slice, bottom_slice = self.range_to_use_to_test.result
            data_to_use = np.copy(self.parent.normalized_images_log[:, top_slice:bottom_slice, left_slice:right_slice])
        else:
            data_to_use = self.parent.normalized_images_log[:]    

        tomography_array = np.array(data_to_use)
        self.parent.before_normalized_images = data_to_use

        if list_algo_to_use:

            logging_3d_array_infos(array=data_to_use,
                               message="before removing strips cleaning")
            logging.info(f"\t{type(tomography_array) =}")
            print(f"{np.shape(tomography_array) = }")
            list_algo_that_failed = []
            list_algo_that_worked = []
            try:
                for _algo in tqdm(list_algo_to_use):
                    logging.info(f"\t{_algo} ... running")
                    kwargs = self.get_keyword_arguments(algorithm_name=_algo)
                    # add the number of cores to use
                    kwargs['ncore'] = NUM_THREADS
                    logging.info(f"\t\t{kwargs =}")
                    tomography_array = RemoveStrips.run_algo(self.list_algo[_algo]['function'], 
                                                            tomography_array, 
                                                            **kwargs)
                    self.saving_configuration(algorithm_name=_algo)
                    logging.info(f"\t{_algo} done!")
                    list_algo_that_worked.append(_algo)
            except np.linalg.LinAlgError:
                list_algo_that_failed.append(_algo)

            self.nothing_to_display = False
            
            if test:
                self.parent.test_normalized_images_log = tomography_array
            else:
                self.parent.normalized_images_log = tomography_array
            
            
            if list_algo_that_failed:
                display(HTML("<font color=red><b>List of algo that failed:</b></font>"))
                for _algo in list_algo_that_failed:
                    display(HTML(f"<font color=red> * {_algo}</font>"))
            if list_algo_that_worked:
                display(HTML("<font color=green><b>List of algos that worked:</b></font>"))
                for _algo in list_algo_that_worked:
                    display(HTML(f"<font color=green> * {_algo}</font>"))
        
            logging_3d_array_infos(array=self.parent.before_normalized_images,
                               message="after removing strips cleaning")

        else:
            logging.info(f"\tskipped!")
                  
    @staticmethod
    def run_algo(name_of_algo, array, **kwargs):
        return name_of_algo(array, **kwargs)

    def display_cleaning(self, test=False):

        if not test:
            if self.parent.configuration.when_to_remove_stripes == WhenToRemoveStripes.out_notebook:
                logging.info("Strips removal will be done in the background just before reconstruction. No display.")
                print("Strips removal was set to be done in the background. No display of results.")
                return
        
        if self.nothing_to_display:
            return

        if self.parent.configuration.when_to_remove_stripes == WhenToRemoveStripes.never:
            logging.info("Strips removal was set to never. No display of results.")
            print("Strips removal was set to never. No display of results.")
            return

        normalized_images_before = self.parent.before_normalized_images
        sinogram_before = self.calculate_sinogram(normalized_images_before)

        if test:
            normalized_images_after = self.parent.test_normalized_images_log
        else:
            normalized_images_after = self.parent.normalized_images_log
        sinogram_after = self.calculate_sinogram(normalized_images_after)

        nbr_projections, height, _ = np.shape(normalized_images_after)

        final_list_of_angles = self.parent.final_list_of_angles
        # final_list_of_runs = self.parent.list_of_runs_to_use[DataType.sample]
        final_list_of_runs = self.parent.final_list_of_runs[DataType.sample]

        self.fig, self.axs1 = plt.subplots(nrows=1, ncols=2, figsize=(7, 5), 
                                         num="Comparison before and after stripes removal")
        self.axs1[0].imshow(normalized_images_before[0], vmin=0, vmax=1)
        self.axs1[0].set_title("Before correction")
        self.axs1[0].axhline(0, color='red', linestyle='--')
        
        self.axs1[1].imshow(normalized_images_after[0], vmin=0, vmax=1)
        self.axs1[1].set_title("After correction")
        self.axs1[1].axhline(0, color='red', linestyle='--')
        
        self.fig2, self.axs2 = plt.subplots(nrows=1, ncols=3, figsize=(7, 5),
                                            num="Sinogram comparison before and after stripes removal")

        self.axs2[0].imshow(sinogram_before[0], vmin=0, vmax=1)
        self.axs2[0].set_title("Before correction")
        self.axs2[1].imshow(sinogram_after[0], vmin=0, vmax=1)
        self.axs2[1].set_title("After correction")
        self.axs2[2].imshow(sinogram_before[0] - sinogram_after[0])
        self.axs2[2].set_title("Difference (before - after)")

        self.slice_index = 0

        def plot_result(image_index, slice_index):

            if self.slice_index != slice_index:
                self.slice_index = slice_index
                self.axs1[0].cla()
                self.axs1[1].cla()

            if self.parent.MODE == OperatingMode.tof:
                self.fig.suptitle(f"Run: {final_list_of_runs[image_index]}, Angle: {final_list_of_angles[image_index]}")

            self.axs1[0].imshow(normalized_images_before[image_index], vmin=0, vmax=1)
            self.axs1[0].set_title("Before correction")
            self.axs1[0].axhline(slice_index, color='red', linestyle='--')

            self.axs1[1].imshow(normalized_images_after[image_index], vmin=0, vmax=1)
            self.axs1[1].set_title("After correction")
            self.axs1[1].axhline(slice_index, color='red', linestyle='--')

            self.axs2[0].imshow(sinogram_before[slice_index], vmin=0, vmax=1)
            self.axs2[0].set_title("Before correction")
            self.axs2[1].imshow(sinogram_after[slice_index], vmin=0, vmax=1)
            
            self.axs2[1].set_title("After correction")
            self.axs2[2].imshow(sinogram_before[slice_index] - sinogram_after[slice_index])
            self.axs2[2].set_title("Difference (before - after)")

            plt.tight_layout()
            plt.show()

        display_plot = interactive(plot_result,
                                   image_index=widgets.IntSlider(min=0,
                                                                 max=nbr_projections-1),
                                    slice_index=widgets.IntSlider(min=0,
                                                                  max=height-1),
                                    )
        display(display_plot)
    
    def window_open(self, url):
        webbrowser.open(url)
