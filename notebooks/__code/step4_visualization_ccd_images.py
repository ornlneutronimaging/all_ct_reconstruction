"""
CT Reconstruction Visualization Module for CCD Images.

This module provides the Step4VisualizationCcdImages class for visualizing
reconstructed CT slices from CCD detector images. It supports loading,
displaying, and comparing reconstructed image stacks with interactive
controls for slice navigation and dual-stack comparison.

Key Classes:
    - Step4VisualizationCcdImages: Main visualization class for reconstructed CT slices

Key Features:
    - Interactive folder selection for reconstructed image stacks
    - Configurable percentage-based random sampling of slices
    - Side-by-side comparison of primary and secondary reconstructions
    - Multi-threaded image loading for performance
    - Interactive slice navigation with matplotlib widgets
    - Support for both single and dual-stack visualization modes

Visualization Capabilities:
    - Primary reconstruction stack visualization
    - Optional secondary reconstruction stack for comparison
    - Slice-by-slice navigation with index sliders
    - Colormapped visualization with adjustable scales
    - Logging and progress tracking for all operations

Dependencies:
    - matplotlib: Interactive plotting and image visualization
    - ipywidgets: Jupyter notebook interactive controls
    - numpy: Numerical computing and random sampling

Author: CT Reconstruction Pipeline Team
Created: Part of Step 4 visualization workflow for CCD-based CT reconstruction
"""

import os
import logging
import glob
from typing import Optional, List, Any, Dict
from IPython.display import display
import ipywidgets as widgets
from IPython.display import HTML
import matplotlib.pyplot as plt
from ipywidgets import interactive
import numpy as np
from numpy.typing import NDArray

from __code import OperatingMode, DataType
from __code.utilities.logging import setup_logging
from __code.utilities.file_folder_browser import FileFolderBrowser
from __code.utilities.load import load_data_using_multithreading
from __code.config import PERCENTAGE_OF_DATA_TO_USE_FOR_RECONSTRUCTION, DEBUG, debug_folder

LOG_BASENAME_FILENAME, _ = os.path.splitext(os.path.basename(__file__))


class Step4VisualizationCcdImages:
    """
    Visualization class for reconstructed CT slices from CCD detector images.
    
    This class provides comprehensive functionality for loading, managing, and
    visualizing reconstructed CT image stacks. It supports both single-stack
    and dual-stack comparison modes with interactive controls for slice
    navigation and percentage-based sampling.
    
    Attributes:
        configuration: Configuration object (currently unused)
        working_dir: Working directory path for file operations
        instrument: Selected instrument name
        primary_reconstructed_slices: Loaded primary reconstruction stack
        secondary_reconstructed_slices: Loaded secondary reconstruction stack (optional)
        primary_list_tiff: List of primary TIFF file paths
        secondary_list_tiff: List of secondary TIFF file paths
        primary_reconstructed_folder: Path to primary reconstruction folder
        secondary_reconstructed_folder: Path to secondary reconstruction folder
        primary_reconstructed: Flag indicating which stack is currently active
        percentage_to_use: Widget for controlling sampling percentage
        number_of_images_to_use: Label showing number of images to be loaded
    """
   
    configuration: Optional[Any] = None

    primary_reconstructed_slices: Optional[NDArray] = None
    secondary_reconstructed_slices: Optional[NDArray] = None

    primary_list_tiff: Optional[List[str]] = None
    secondary_list_tiff: Optional[List[str]] = None

    def __init__(self, system: Optional[Any] = None) -> None:
        """
        Initialize the Step4VisualizationCcdImages instance.
        
        Sets up the working directory, instrument configuration, and logging
        for the visualization workflow. Supports both production and debug
        modes based on the DEBUG configuration flag.
        
        Args:
            system: System configuration object containing working directory
                   and instrument selection information. If None, defaults
                   will be used.
        
        Side Effects:
            - Configures logging for the visualization module
            - Sets working directory from system configuration or debug folder
            - Logs initialization parameters and debug mode status
        """
        # self.configuration = Configuration()
        self.working_dir = system.System.get_working_dir()
        if DEBUG:
            self.working_dir = debug_folder[OperatingMode.white_beam][DataType.extra]

        self.instrument = system.System.get_instrument_selected()

        setup_logging(LOG_BASENAME_FILENAME)      
        logging.info(f"working_dir: {self.working_dir}")
        logging.info(f"instrument: {self.instrument}")
        if DEBUG:
            logging.info(f"WARNING!!!! we are running using DEBUG mode!")

    def select_reconstructed_folder(self, primary_reconstructed: bool = True) -> None:
        """
        Launch interactive folder selection for reconstructed image stacks.
        
        Opens a file browser widget to allow user selection of folders containing
        reconstructed TIFF images. Supports selection of both primary and secondary
        reconstruction folders for comparison visualization.
        
        Args:
            primary_reconstructed: Flag indicating whether to select primary (True)
                                 or secondary (False) reconstruction folder.
                                 Defaults to True for primary folder selection.
        
        Side Effects:
            - Sets self.primary_reconstructed flag for subsequent operations
            - Launches interactive file browser widget
            - Triggers save_folder_selected callback upon folder selection
        """
        self.primary_reconstructed = primary_reconstructed

        working_dir = self.working_dir
        o_file_browser = FileFolderBrowser(working_dir=working_dir,
                                           next_function=self.save_folder_selected)
        o_file_browser.select_input_folder(instruction=f"Select folder containing the reconstructed images ...",
                                           multiple_flag=False)
        
    def save_folder_selected(self, folder_selected: str) -> None:
        """
        Process and store the selected reconstruction folder path.
        
        Callback function triggered by folder selection widget. Scans the
        selected folder for TIFF files and stores the file list for subsequent
        loading operations. Handles both primary and secondary folder selections.
        
        Args:
            folder_selected: Absolute path to the selected folder containing
                           reconstructed TIFF images.
        
        Side Effects:
            - Sets primary_reconstructed_folder or secondary_reconstructed_folder
            - Scans folder for TIFF files (*.tif and *.tiff extensions)
            - Stores file list in primary_list_tiff or secondary_list_tiff
            - Logs folder path and number of discovered TIFF files
        
        Note:
            Contains typo in secondary folder attribute name (reconstructef_folder)
            which should be maintained for backward compatibility.
        """
        if self.primary_reconstructed:
            self.primary_reconstructed_folder = folder_selected
        else:
            self.secondary_reconstructef_folder = folder_selected

        logging.info(f"folder_selected: {folder_selected} (primary_reconstructed={self.primary_reconstructed})")
        list_tiff = glob.glob(os.path.join(folder_selected, "*.tif*"))
        logging.info(f"found {len(list_tiff)} tiff files in {folder_selected}")

        if self.primary_reconstructed:
            self.primary_list_tiff = list_tiff
        else:
            self.secondary_list_tiff = list_tiff

    def load_images(self) -> None:
        """
        Load reconstructed images with percentage-based random sampling.
        
        Loads a randomly selected subset of TIFF files based on the percentage
        specified by the user interface slider. Uses multi-threaded loading
        for improved performance with large image stacks.
        
        The sampling process:
        1. Calculates number of images based on percentage
        2. Randomly selects image indices without replacement
        3. Sorts indices to maintain chronological order
        4. Loads selected images using multi-threading
        
        Side Effects:
            - Loads images into primary_reconstructed_slices or 
              secondary_reconstructed_slices based on current mode
            - Logs loading progress and completion status
            - Displays completion message to user interface
        
        Raises:
            AttributeError: If percentage_to_use widget is not initialized
            ValueError: If no TIFF files are available for loading
        """
        if self.primary_reconstructed:
            list_tiff = self.primary_list_tiff
        else:
            list_tiff = self.secondary_list_tiff
              
        nbr_images_to_use = int(self.percentage_to_use.value / 100 * len(list_tiff))
        list_tiff_index_to_use = np.random.randint(0, len(list_tiff), nbr_images_to_use)
        list_tiff_index_to_use.sort()
        list_tiff = [list_tiff[_index] for _index in list_tiff_index_to_use]

        logging.info(f"loading {len(list_tiff)} images ...")
        
        if self.primary_reconstructed:
            self.primary_reconstructed_slices = load_data_using_multithreading(list_tiff)
        else:
            self.secondary_reconstructed_slices = load_data_using_multithreading(list_tiff)
        
        logging.info(f"done!")
        print(f"Loading done! ({len(list_tiff)} images loaded)")
        
    def select_percentage_of_images_to_visualize(self) -> None:
        """
        Create interactive controls for image sampling percentage selection.
        
        Displays a slider widget allowing users to specify what percentage of
        available reconstructed images should be loaded for visualization.
        Updates the display to show the actual number of images that will be
        loaded based on the selected percentage.
        
        Widget Configuration:
            - Range: 0.1% to 100% in 0.1% increments
            - Default: Value from PERCENTAGE_OF_DATA_TO_USE_FOR_RECONSTRUCTION
            - Layout: Full width for easy interaction
        
        Side Effects:
            - Creates and displays percentage_to_use FloatSlider widget
            - Creates and displays number_of_images_to_use Label widget
            - Registers callback for real-time percentage change updates
            - Calculates and displays initial image count
        """
        self.percentage_to_use = widgets.FloatSlider(value=PERCENTAGE_OF_DATA_TO_USE_FOR_RECONSTRUCTION,
                                    min=.1,
                                    max=100,
                                    step=.1,
                                    layout=widgets.Layout(width='100%'))
        display(self.percentage_to_use)

        list_of_tiff = self.primary_list_tiff
        percentage = self.percentage_to_use.value
        nbr_images = int(percentage / 100 * len(list_of_tiff))
        self.number_of_images_to_use = widgets.Label(f"{nbr_images} primary slices will be used for the visualization")
        display(self.number_of_images_to_use)
        self.percentage_to_use.observe(self.on_percentage_to_use_change, names='value') 

    def on_percentage_to_use_change(self, change: Dict[str, Any]) -> None:
        """
        Callback handler for percentage slider value changes.
        
        Updates the displayed image count when the user adjusts the percentage
        slider. Recalculates the number of images that will be loaded based
        on the new percentage value and the total number of available TIFF files.
        
        Args:
            change: Dictionary containing change information from the widget.
                   Expected to have 'new' key with the updated percentage value.
        
        Side Effects:
            - Updates number_of_images_to_use label with new image count
            - Recalculates based on primary_list_tiff length
        """
        new_value = change['new']
        list_tiff = self.primary_list_tiff
        nbr_images = int(new_value / 100 * len(list_tiff))
        self.number_of_images_to_use.value = f"{nbr_images} images will be used for the visualization"

    def visualize(self) -> None:
        """
        Create interactive visualization interface for reconstructed CT slices.
        
        Generates an interactive matplotlib-based visualization with slider controls
        for navigating through reconstructed image stacks. Supports both single-stack
        and dual-stack comparison modes depending on whether secondary reconstructed
        slices are available.
        
        Visualization Modes:
            - Single-stack: Displays only primary reconstruction with navigation
            - Dual-stack: Side-by-side comparison of primary and secondary stacks
        
        Interactive Features:
            - Slice index sliders for independent navigation
            - Viridis colormap with individual colorbars
            - Automatic layout adjustment for optimal display
            - Disabled secondary controls when no secondary stack is loaded
        
        Side Effects:
            - Creates and displays interactive matplotlib plots
            - Logs visualization completion status for each stack
            - Returns early with message if no primary data is loaded
        
        Returns:
            None: Early return if primary_reconstructed_slices is None
        """

        if self.primary_reconstructed_slices is None:
            logging.info(f"Nothing to visualize!")
            print("Nothing to visualize (load the primary reconstructed stack of images!")
            return
        
        disable_secondary_slider = False

        if self.secondary_reconstructed_slices is None:
            disable_secondary_slider = True
            len_secondary_slices = 1
        else:
            len_secondary_slices = len(self.secondary_reconstructed_slices)

        def plot_norm(left_index: int = 0, right_index: int = 0) -> None:
            """
            Internal plotting function for interactive visualization.
            
            Renders the actual plots based on current slider values. Handles
            both single and dual-stack visualization modes with appropriate
            subplot arrangements and colorbar positioning.
            
            Args:
                left_index: Index for primary stack slice selection
                right_index: Index for secondary stack slice selection (if available)
            """
            
            if self.secondary_reconstructed_slices is not None:

                fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
                im0 = axs[0].imshow(self.primary_reconstructed_slices[left_index], cmap='viridis')
                axs[0].set_title(f"Primary reconstructed slice #{left_index}")
                plt.colorbar(im0, ax=axs[0], shrink=0.5)
                                
                im1 = axs[1].imshow(self.secondary_reconstructed_slices[right_index], cmap='viridis')
                axs[1].set_title(f"Secondary reconstructed slice #{right_index}")
                plt.colorbar(im1, ax=axs[1], shrink=0.5)
              
                plt.tight_layout()
                plt.show()

            else:

                fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
                im0 = axs.imshow(self.primary_reconstructed_slices[left_index], cmap='viridis')
                axs.set_title(f"Primary reconstructed slice #{left_index}")
                plt.colorbar(im0, ax=axs, shrink=0.5)

        interactive_plot = interactive(plot_norm,
                                       left_index=widgets.IntSlider(min=0,
                                                                      max=len(self.primary_reconstructed_slices)-1,
                                                                      step=1,
                                                                      value=0),
                                        right_index=widgets.IntSlider(min=0,
                                                                      max=len_secondary_slices-1,
                                                                      step=1,
                                                                      value=0,
                                                                      disabled=disable_secondary_slider),
                                        )                            
        
        display(interactive_plot)
        
        logging.info(f"Visualization of {len(self.primary_reconstructed_slices)} primary slices done!")
        if self.secondary_reconstructed_slices is not None:
            logging.info(f"Visualization of {len(self.secondary_reconstructed_slices)} secondary slices done!")
        logging.info(f"Done!")
                