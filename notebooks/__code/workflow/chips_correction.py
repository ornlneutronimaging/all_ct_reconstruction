"""
Chip Alignment Correction for Multi-Chip Detector Systems.

This module provides functionality to correct alignment issues in CT projection images
acquired with multi-chip detector systems. Many modern neutron imaging detectors consist
of multiple sensor chips that may have slight misalignments, creating discontinuities
in the projection images that can severely impact reconstruction quality.

Key Classes:
    - ChipsCorrection: Main class for chip alignment correction workflow

Key Features:
    - Automatic correction of chip-to-chip alignment offsets
    - Support for 4-chip detector configurations (2x2 arrangement)
    - Gap filling algorithms to smooth transitions between chips
    - Interactive visualization for before/after comparison
    - Configurable correction parameters from system configuration

Alignment Correction Process:
    1. Divide detector image into 4 quadrants based on chip boundaries
    2. Apply pre-determined offset corrections to each chip
    3. Reconstruct aligned image with proper chip positioning
    4. Fill gaps between chips using interpolation algorithms
    5. Validate correction through visual inspection tools

Mathematical Background:
    The correction applies spatial transformations to each chip quadrant:
    - chip_corrected = translate(chip_original, offset_x, offset_y)
    - Gap filling uses weighted linear interpolation between adjacent regions
    
Dependencies:
    - numpy: Numerical array operations and transformations
    - matplotlib: Visualization and comparison plotting
    - ipywidgets: Interactive control interfaces
    - logging: Progress tracking and validation
    - tqdm: Progress bars for batch processing

Author: CT Reconstruction Pipeline Team
Created: Part of CT reconstruction development workflow
"""

import numpy as np
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
from ipywidgets import interactive
from IPython.display import display
import ipywidgets as widgets
from typing import Optional, List, Tuple, Union
from numpy.typing import NDArray

from __code.parent import Parent
from __code.config import chips_offset
from __code import DataType
from __code.utilities.logging import logging_3d_array_infos


class ChipsCorrection(Parent):
    """
    Handles correction of alignment issues in multi-chip detector systems.
    
    This class provides comprehensive functionality for correcting spatial misalignments
    between detector chips in multi-chip imaging systems. It addresses discontinuities
    at chip boundaries that can create artifacts in reconstructed CT volumes.
    
    The correction process involves:
    1. Applying pre-calibrated spatial offsets to each chip quadrant
    2. Reconstructing the aligned detector image
    3. Filling gaps between chips using interpolation
    4. Providing visualization tools for quality control
    
    Methods:
        run(): Execute the chip alignment correction process
        correct_alignment(): Static method for performing the actual alignment
        visualize_chips_correction(): Interactive comparison of before/after images
    """

    def run(self) -> None:
        """
        Execute chip alignment correction on normalized projection images.
        
        Applies pre-configured chip offset corrections to the normalized projection
        data to correct for detector chip misalignments. The process involves axis
        swapping to optimize processing order, applying corrections, and restoring
        the original axis arrangement.
        
        Returns:
            None: Modifies parent object with corrected images
            
        Side Effects:
            - Creates self.parent.corrected_images with alignment-corrected data
            - Logs array shapes and correction progress
            - Transforms data from (angle, y, x) to (y, x, angle) and back
            
        Notes:
            - Uses pre-configured offsets from system configuration
            - Preserves original normalized_images for comparison
            - Logs detailed shape information for debugging
        """
        
        logging.info(f"Chips correction")
        offset: List[int] = list(chips_offset)

        logging_3d_array_infos(message="before chips correction",
                               array=self.parent.normalized_images)

        normalized_images: NDArray[np.floating] = np.array(self.parent.normalized_images)
        logging.info(f"\t{np.shape(normalized_images) =}")   # angle, y, x
        normalized_images_axis_swap: NDArray[np.floating] = np.moveaxis(normalized_images, 0, 2)  # y, x, angle
        logging.info(f"\t{np.shape(normalized_images_axis_swap) =}")
        corrected_images: NDArray[np.floating] = ChipsCorrection.correct_alignment(normalized_images_axis_swap,
                                                    offsets=offset)
        self.parent.corrected_images = np.moveaxis(corrected_images, 2, 0)  # angle, y, x
        logging.info(f"\tChips correction done!")

        logging_3d_array_infos(message="aftert chips correction",
                               array=self.parent.corrected_images)

    @staticmethod
    def correct_alignment(unaligned_image: Optional[NDArray[np.floating]] = None, 
                         offsets: Optional[List[int]] = None, 
                         center: Optional[List[int]] = None, 
                         fill_gap: bool = True, 
                         num_pix_unused: int = 1, 
                         num_pix_neighbor: int = 1) -> NDArray[np.floating]:
        """
        Correct alignment of detector chip segments in projection images.
        
        This static method performs the core chip alignment correction by dividing
        the detector image into quadrants representing individual chips, applying
        spatial offset corrections, and optionally filling gaps between chips.
        
        Args:
            unaligned_image: 3D projection data array with shape (height, width, n_projections)
            offsets: List containing [x_offset, y_offset] correction values in pixels
            center: Optional list with [center_x, center_y] coordinates of chip junction.
                   If None, uses image center.
            fill_gap: Whether to interpolate values in gaps created by chip movement
            num_pix_unused: Number of border pixels to exclude during gap filling
            num_pix_neighbor: Number of neighboring pixels used for gap interpolation
            
        Returns:
            Aligned projection data array with corrected chip positions
            
        Raises:
            None: Gracefully handles edge cases with logging warnings
            
        Notes:
            - Assumes 2x2 chip arrangement (4 chips total)
            - Uses linear interpolation for gap filling
            - Returns original data if offsets are zero
            - Automatically adjusts center if outside image bounds
            
        Mathematical Details:
            Gap filling uses weighted linear interpolation:
            filled_value = weight * region_1 + (1-weight) * region_2
            where weights vary linearly across the gap
        """
        # Get the offsets
        x_offset: int = offsets[0]
        y_offset: int = offsets[1]

        # Get the center
        if center is not None:
            center_x: int = center[0]
            center_y: int = center[1]

            # Check if the unaligned image contains the alignment center along both axes
            if (center_x < 0) or (center_x > unaligned_image.shape[1]):
                center_x = unaligned_image.shape[1] // 2
                x_offset = 0
            if (center_y < 0) or (center_y > unaligned_image.shape[0]):
                center_y = unaligned_image.shape[0] // 2
                y_offset = 0

        else:
            center_x: int = unaligned_image.shape[1] // 2
            center_y: int = unaligned_image.shape[0] // 2

        # Return the original image if both the offset values are zero
        if (x_offset == 0) and (y_offset == 0):
            warning_message: str = "Alignment correction not performed as both the offset values are zero."
            logging.info(warning_message)

            return unaligned_image

        # Get the chips
        chip_1: NDArray[np.floating] = unaligned_image[:center_y, :center_x]
        chip_2: NDArray[np.floating] = unaligned_image[:center_y, center_x:]
        chip_3: NDArray[np.floating] = unaligned_image[center_y:, :center_x]
        chip_4: NDArray[np.floating] = unaligned_image[center_y:, center_x:]

        # Move the chips and create aligned image
        moved_image: NDArray[np.floating] = np.zeros((unaligned_image.shape[0] + y_offset,
                                unaligned_image.shape[1] + x_offset,
                                unaligned_image.shape[2]))

        moved_image[:center_y, :center_x] = chip_1
        moved_image[:center_y, center_x + x_offset:] = chip_2
        moved_image[center_y + y_offset:, :center_x] = chip_3
        moved_image[center_y + y_offset:, center_x + x_offset:] = chip_4

        if fill_gap is True:
            num_wave: int = unaligned_image.shape[2]
            filled_image: NDArray[np.floating] = np.copy(moved_image)

            # Fill gaps along y-axis
            if y_offset > 0:
                y_upper_bound: int = unaligned_image.shape[0] - num_pix_unused - num_pix_neighbor
                y_lower_bound: int = num_pix_unused + num_pix_neighbor
                if y_upper_bound > center_y >= y_lower_bound:
                    y0_up: int = center_y - num_pix_unused - num_pix_neighbor
                    y1_up: int = center_y - num_pix_unused
                    region_up: NDArray[np.floating] = np.expand_dims(np.mean(filled_image[y0_up:y1_up], axis=0), axis=0)

                    y0_down: int = center_y + y_offset + num_pix_unused
                    y1_down: int = center_y + y_offset + num_pix_unused + num_pix_neighbor
                    region_down: NDArray[np.floating] = np.expand_dims(np.mean(filled_image[y0_down:y1_down], axis=0), axis=0)

                    weights_y: NDArray[np.floating] = np.expand_dims(np.linspace(0, 1, y_offset + 2 * num_pix_unused), axis=1)

                    for wave in range(num_wave):
                        filled_image[center_y - num_pix_unused:center_y + y_offset + num_pix_unused, :, wave] = \
                            weights_y[::-1] @ region_up[:, :, wave] + weights_y @ region_down[:, :, wave]

                else:
                    warning_message: str = "Couldn't fill gaps along y-axis as the center is close to border."
                    logging.info(warning_message)

            # Fill gaps along x-axis
            if x_offset > 0:
                x_upper_bound: int = unaligned_image.shape[1] - num_pix_unused - num_pix_neighbor
                x_lower_bound: int = num_pix_unused + num_pix_neighbor
                if x_upper_bound > center_x >= x_lower_bound:
                    x0_left: int = center_x - num_pix_unused - num_pix_neighbor
                    x1_left: int = center_x - num_pix_unused
                    region_left: NDArray[np.floating] = np.expand_dims(np.mean(filled_image[:, x0_left:x1_left], axis=1), axis=1)

                    x0_right: int = center_x + x_offset + num_pix_unused
                    x1_right: int = center_x + x_offset + num_pix_unused + num_pix_neighbor
                    region_right: NDArray[np.floating] = np.expand_dims(np.mean(filled_image[:, x0_right:x1_right], axis=1), axis=1)

                    weights_x: NDArray[np.floating] = np.expand_dims(np.linspace(0, 1, x_offset + 2 * num_pix_unused), axis=0)

                    for wave in range(num_wave):
                        filled_image[:, center_x - num_pix_unused:center_x + x_offset + num_pix_unused, wave] = \
                            region_left[:, :, wave] @ weights_x[:, ::-1] + region_right[:, :, wave] @ weights_x

                else:
                    warning_message: str = "Couldn't fill gaps along x-axis as the center is close to border."
                    logging.info(warning_message)

            return filled_image

        else:
            return moved_image
        
    def visualize_chips_correction(self) -> None:
        """
        Create interactive visualization comparing before and after chip correction.
        
        Provides an interactive interface for visually inspecting the quality of
        chip alignment correction. Displays side-by-side comparison of original
        normalized images and corrected images with adjustable contrast controls.
        
        Returns:
            None: Creates interactive widget interface
            
        Side Effects:
            - Displays interactive matplotlib plots
            - Creates slider widgets for image browsing and contrast adjustment
            - Shows before/after comparison with synchronized controls
            
        Widget Controls:
            - image_index: Browse through different projection angles
            - vmin/vmax: Adjust contrast and brightness for visualization
            
        Notes:
            - Helps identify residual alignment artifacts
            - Useful for validating correction quality
            - Synchronized contrast controls for fair comparison
        """

        corrected_images: NDArray[np.floating] = self.parent.corrected_images
        # list_of_runs_to_use = self.parent.list_of_runs_to_use[DataType.sample]
        normalized_images: NDArray[np.floating] = self.parent.normalized_images

        def plot_norm(image_index: int = 0, vmin: float = 0, vmax: float = 1) -> None:
            """Plot comparison of uncorrected vs corrected chip alignment."""

            fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

            _norm_data: NDArray[np.floating] = corrected_images[image_index]
            # _run_number = list_of_runs_to_use[image_index]
            _raw_data: NDArray[np.floating] = normalized_images[image_index]

            im0 = axs[0].imshow(_raw_data, vmin=vmin, vmax=vmax)
            axs[0].set_title("Chips uncorrected")
            plt.colorbar(im0, ax=axs[0], shrink=0.5)

            im1 = axs[1].imshow(_norm_data, vmin=vmin, vmax=vmax)
            axs[1].set_title('Chips corrected')
            plt.colorbar(im1, ax=axs[1], shrink=0.5)
    
            # fig.set_title(f"{_run_number}")
            
            plt.tight_layout()
            plt.show()

        display_plot = interactive(plot_norm,
                                  image_index=widgets.IntSlider(min=0,
                                                                max=len(corrected_images) -1,
                                                                value=0),
                                  vmin=widgets.IntSlider(min=0, max=10, value=0),
                                  vmax=widgets.IntSlider(min=0, max=10, value=1))
        display(display_plot)
        