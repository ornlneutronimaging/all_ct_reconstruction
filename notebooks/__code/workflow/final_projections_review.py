"""
Final Projection Review and Quality Control for CT Reconstruction.

This module provides comprehensive visualization and quality assessment tools for
reviewing processed CT projection data before final reconstruction. It enables
users to inspect projection quality, identify problematic runs, and make informed
decisions about data inclusion for optimal reconstruction results.

Key Classes:
    - FinalProjectionsReview: Main class for projection review and quality control

Key Features:
    - Multi-panel grid visualization for rapid data overview
    - Interactive single-image inspection with rotation axis overlay
    - Stack browsing for detailed projection analysis
    - Run selection interface for quality-based filtering
    - Standardized visualization with consistent contrast settings
    - Integration with workflow data structures

Quality Control Workflow:
    1. Display all projections in organized grid layout
    2. Enable detailed inspection of individual projections
    3. Provide interactive browsing of projection stack
    4. Allow selection of runs to exclude from reconstruction
    5. Validate data quality and consistency

Visualization Features:
    - Automatic grid layout optimization for multiple projections
    - Synchronized contrast settings across all images
    - Rotation axis overlay for geometric validation
    - Color-coded displays with calibrated intensity ranges
    - Interactive controls for detailed examination

Dependencies:
    - numpy: Numerical array operations
    - matplotlib: High-quality scientific visualization
    - ipywidgets: Interactive control interfaces
    - IPython.display: Notebook display functionality

Author: CT Reconstruction Pipeline Team
Created: Part of CT reconstruction development workflow
"""

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
import ipywidgets as widgets
from IPython.display import HTML
from ipywidgets import interactive
from typing import Optional, List, Tuple
from numpy.typing import NDArray

from __code.parent import Parent
from __code import DataType


class FinalProjectionsReview(Parent):
    """
    Provides comprehensive review and quality control for final projection data.
    
    This class enables detailed inspection and quality assessment of processed
    CT projection data before final reconstruction. It offers multiple visualization
    modes and interactive tools for identifying and excluding problematic data.
    
    Key Features:
        - Multi-panel grid display for rapid data overview
        - Interactive single-image detailed inspection
        - Stack browsing with geometric validation overlays
        - Quality-based run selection and filtering
        - Standardized visualization parameters
        
    Attributes:
        list_runs_with_infos: Metadata about runs for selection interface
        
    Methods:
        run(): Create multi-panel overview of all projections
        single_image(): Display individual projection with controls
        stack_of_images(): Interactive browsing of projection stack
        list_runs_to_reject(): Interface for selecting runs to exclude
        
    Quality Control Process:
        1. Grid overview for rapid visual inspection
        2. Detailed single-image analysis for suspicious projections
        3. Stack browsing for geometric consistency validation
        4. Interactive selection of runs to exclude
        5. Final validation before reconstruction
    """
    
    list_runs_with_infos: Optional[List[str]] = None

    def run(self, array: Optional[NDArray[np.floating]] = None, auto_vrange: bool = True) -> None:
        """
        Create multi-panel grid visualization of all projections for quality review.
        
        Displays all projection images in an organized grid layout for rapid
        visual inspection and quality assessment. Automatically calculates
        optimal grid dimensions and provides consistent visualization parameters.
        
        Args:
            array: Array of projection images to display (n_projections, height, width)
            auto_vrange: Flag to enable automatic intensity range calculation
            
        Returns:
            None: Displays matplotlib figure with projection grid
            
        Side Effects:
            - Creates and displays multi-panel matplotlib figure
            - Sets consistent color mapping and intensity range
            - Adds colorbars for quantitative assessment
            - Optimizes layout for visual clarity
            
        Notes:
            - Uses 5-column grid layout for optimal visualization
            - Automatically calculates required number of rows
            - Applies standardized contrast settings (0-1 range)
            - Hides unused subplot panels for clean appearance
        """

        if not list(array):
            return

        nbr_images: int = len(array)

        # list_angles = self.parent.final_list_of_angles
        # list_runs = self.parent.list_of_runs_to_use[DataType.sample]

        nbr_cols: int = 5
        nbr_rows: int = int(np.ceil(nbr_images / nbr_cols))

        fig, axs = plt.subplots(nrows=nbr_rows, ncols=nbr_cols,
                                figsize=(nbr_cols*2,nbr_rows*2))
        flat_axs = axs.flatten()

        if auto_vrange:
            vmin: float = np.min(array)
            vmax: float = np.max(array)
        else:
            vmin = 0.0
            vmax = 1.0

        _index: int = 0
        # list_runs_with_infos = []
        for _row in np.arange(nbr_rows):
            for _col in np.arange(nbr_cols):
                _index = _col + _row * nbr_cols
                if _index == (nbr_images):
                    break
                # title = f"{list_runs[_index]}, {list_angles[_index]}"
                # list_runs_with_infos.append(title)
                # flat_axs[_index].set_title(title)
                im1 = flat_axs[_index].imshow(array[_index], vmin=vmin, vmax=vmax)
                plt.colorbar(im1, ax=flat_axs[_index], shrink=0.5)
           
        for _row in np.arange(nbr_rows):
            for _col in np.arange(nbr_cols):
                _index = _col + _row * nbr_cols
                flat_axs[_index].axis('off')

        # self.list_runs_with_infos = list_runs_with_infos

        plt.tight_layout()
        plt.show()

    def single_image(self, image: Optional[NDArray[np.floating]] = None) -> None:
        """
        Display a single projection image with detailed visualization controls.
        
        Provides detailed visualization of an individual projection image with
        standardized contrast settings and color mapping for quality assessment.
        
        Args:
            image: 2D projection image array to display
            
        Returns:
            None: Displays matplotlib figure with single image
            
        Side Effects:
            - Creates and displays single-panel matplotlib figure
            - Applies standardized intensity range (0-1)
            - Adds colorbar for quantitative intensity assessment
            - Removes axis labels for clean presentation
            
        Notes:
            - Returns early if no image provided
            - Uses consistent visualization parameters with grid display
            - Optimized for detailed inspection of individual projections
        """

        if image is None:
            return

        fig, ax = plt.subplots(num="After rotation")
        im1 = ax.imshow(image, vmin=0, vmax=1)
        plt.colorbar(im1, ax=ax, shrink=0.5)
        ax.axis('off')
        plt.show()

    def stack_of_images(self, array: Optional[NDArray[np.floating]] = None) -> None:
        """
        Create interactive browsing interface for projection stack with geometric validation.
        
        Provides an interactive interface for browsing through the projection stack
        with rotation axis overlay for geometric consistency validation. Essential
        for identifying angular positioning errors or geometric inconsistencies.
        
        Args:
            array: 3D array of projections (n_projections, height, width)
            
        Returns:
            None: Creates interactive widget interface
            
        Side Effects:
            - Creates interactive slider widget for projection browsing
            - Displays matplotlib plots with rotation axis overlay
            - Shows red dashed line indicating expected rotation axis position
            - Updates display dynamically as user browses projections
            
        Notes:
            - Returns early if no array provided
            - Calculates rotation axis position at image center
            - Uses viridis colormap for enhanced contrast
            - Essential for validating geometric consistency across projections
        """
        if array is None:
            return
        
        _, width = array.shape[-2:]
        horizontal_center: int = width // 2

        def plot_images(index: int) -> None:
            """Plot individual projection with rotation axis overlay."""
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
            image: NDArray[np.floating] = array[index]
            ax.imshow(image, cmap='viridis', vmin=0, vmax=1)
            ax.axvline(x=horizontal_center, color='red', linestyle='--', label='Rotation Axis')
            ax.set_title(f"Image {index}")

        display_plot_images = interactive(plot_images, 
                                          index=widgets.IntSlider(min=0, 
                                                                  max=len(array)-1, 
                                                                  step=1, 
                                                                  value=0))
        display(display_plot_images)

    def list_runs_to_reject(self) -> None:
        """
        Create interactive interface for selecting runs to exclude from reconstruction.
        
        Provides a user interface for selecting specific runs to exclude from
        the final reconstruction based on quality assessment. Creates a multi-select
        widget populated with run information for easy identification and selection.
        
        Returns:
            None: Creates and displays interactive selection widget
            
        Side Effects:
            - Creates self.parent.runs_to_exclude_ui widget
            - Displays selection interface with run metadata
            - Enables multiple run selection for exclusion
            - Stores selection in parent object for workflow integration
            
        Notes:
            - Uses run metadata for clear identification
            - Supports multiple selection for batch exclusion
            - Essential for quality-based data filtering
            - Integrates with reconstruction workflow for automatic exclusion
        """
        
        label_ui: widgets.HTML = widgets.HTML("<b>Select runs you want to exclude from the final reconstruction:</b>")
        self.parent.runs_to_exclude_ui = widgets.SelectMultiple(options=self.list_runs_with_infos,
                                                                layout=widgets.Layout(height="300px"))
        display(widgets.VBox([label_ui, self.parent.runs_to_exclude_ui]))
        