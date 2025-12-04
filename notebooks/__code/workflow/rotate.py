"""
Image Rotation Utilities for CT Reconstruction Pipeline.

This module provides functionality for rotating CT projection images to ensure proper
alignment with the reconstruction coordinate system. CT reconstruction algorithms
typically require the rotation axis to be vertical in the projection images, and this
module provides interactive tools to assess and correct image orientation.

Key Classes:
    - Rotate: Main class for image rotation operations

Key Features:
    - Interactive rotation axis visualization and assessment
    - Support for 90-degree rotations (most common requirement)
    - Real-time preview of rotation effects
    - Efficient array-based rotation using numpy operations
    - Memory-optimized implementation for large datasets

Geometric Background:
    CT reconstruction requires consistent geometric alignment where:
    - The rotation axis must be vertical in projection images
    - Sample rotation corresponds to horizontal movement in projections
    - Proper alignment ensures accurate reconstruction geometry

Mathematical Implementation:
    90-degree rotation is performed using efficient array operations:
    rotated_image = image.swapaxes(-2, -1)[..., ::-1]
    This is equivalent to numpy.rot90 but optimized for large arrays.

Dependencies:
    - ipywidgets: Interactive control interfaces
    - IPython.display: Widget display functionality
    - matplotlib: Image visualization and rotation axis overlay
    - scikit-image: Image transformation algorithms
    - numpy: Efficient array operations
    - logging: Progress tracking

Author: CT Reconstruction Pipeline Team
Created: Part of CT reconstruction development workflow
"""

import ipywidgets as widgets
from IPython.display import display
from IPython.display import HTML
from ipywidgets import interactive
from tqdm import tqdm
import numpy as np
from skimage import transform
import multiprocessing as mp 
import logging
from functools import partial
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Union
from numpy.typing import NDArray

from __code.parent import Parent


def _worker(_data: NDArray[np.floating], angle_value: float) -> NDArray[np.floating]:
    """
    Worker function for rotating individual images.
    
    This function is designed for potential multiprocessing operations,
    though the current implementation uses more efficient array operations.
    
    Args:
        _data: Input image array to rotate
        angle_value: Rotation angle in degrees
        
    Returns:
        Rotated image array
        
    Note:
        Currently not used in favor of more efficient array-based rotation
    """
    data: NDArray[np.floating] = transform.rotate(_data, angle_value)
    print(f"{np.shape(data) = }")
    print(f"{data = }")
    return data


class Rotate(Parent):
    """
    Handles rotation of CT projection images for proper geometric alignment.
    
    This class provides interactive tools for assessing and correcting the orientation
    of CT projection images to ensure the rotation axis is vertical, which is required
    for accurate tomographic reconstruction.
    
    Attributes:
        angle_ui (widgets.RadioButtons): Interactive widget for angle selection
        
    Methods:
        is_rotation_needed(): Interactive assessment of rotation axis alignment
        set_settings(): Configure rotation parameters with preview
        apply_rotation(): Execute the rotation transformation
    """

    def is_rotation_needed(self) -> None:
        """
        Interactive tool to assess if rotation is needed for proper axis alignment.
        
        Creates an interactive visualization showing projection images with a red
        dashed line indicating where the vertical rotation axis should be located.
        Users can browse through different projections to assess alignment.
        
        Returns:
            None: Creates interactive visualization interface
            
        Side Effects:
            - Displays interactive plot with rotation axis overlay
            - Creates slider widget for browsing projections
            - Shows guidance text about required vertical axis alignment
            
        Notes:
            - Red dashed line shows the expected position of rotation axis
            - Rotation axis must be vertical for proper reconstruction
            - If sample appears to rotate around horizontal axis, 90° rotation needed
        """
        
        _, width = self.parent.normalized_images[0].shape[-2:]
        horizontal_center: int = width // 2

        display(HTML(f"<h3>The rotation axis of the sample must be VERTICAL! If it's not, you will need to rotate by 90degrees</h3>"))

        def plot_images(index: int) -> None:
            """Plot individual projection with rotation axis overlay."""
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
            image: NDArray[np.floating] = self.parent.normalized_images[index]
            ax.imshow(image, cmap='viridis', vmin=0, vmax=1)
            ax.axvline(x=horizontal_center, color='red', linestyle='--', label='Rotation Axis')
            ax.set_title(f"Image {index}")

        display_plot_images = interactive(plot_images, 
                                          index=widgets.IntSlider(min=0, 
                                                                  max=len(self.parent.normalized_images)-1, 
                                                                  step=1, 
                                                                  value=0))
        display(display_plot_images)

    def set_settings(self) -> None:
        """
        Configure rotation settings with interactive preview.
        
        Creates an interactive interface allowing users to select rotation angle
        and preview the effects. Shows side-by-side comparison of original and
        rotated images to aid in decision making.
        
        Returns:
            None: Creates interactive settings interface
            
        Side Effects:
            - Creates self.angle_ui widget for angle selection
            - Displays side-by-side preview of rotation options
            - Shows original (0°) and 90° rotated versions
            
        Notes:
            - Most common requirement is 90° rotation
            - Preview helps verify correct rotation direction
            - Radio button interface for easy selection
        """
    
        title_ui: widgets.HTML = widgets.HTML("Select rotation angle")
        self.angle_ui: widgets.RadioButtons = widgets.RadioButtons(options=['90 degrees', '0 degree'],
                                             value='90 degrees',
                                            description='Angle')
        
        vbox: widgets.VBox = widgets.VBox([title_ui, self.angle_ui])
        display(vbox)

        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(5, 10), num="Rotation Preview")

        image_normal: NDArray[np.floating] = self.parent.normalized_images[0]
        #image_rot_plut_90 = transform.rotate(self.parent.normalized_images[0], +90)
        image_rot_plus_90: NDArray[np.floating] = self.parent.normalized_images[0].swapaxes(-2, -1)[..., ::-1]

        axs[0].imshow(image_normal, cmap='viridis', vmin=0, vmax=1)
        axs[0].set_title('0 degree')

        axs[1].imshow(image_rot_plus_90, cmap='viridis', vmin=0, vmax=1)
        axs[1].set_title('+90 degrees')

    def _worker(self, _data: NDArray[np.floating], angle_value: float) -> NDArray[np.floating]:
        """
        Worker method for individual image rotation.
        
        Args:
            _data: Input image array to rotate
            angle_value: Rotation angle in degrees
            
        Returns:
            Rotated image array
            
        Note:
            Alternative implementation using scikit-image transform
        """
        data: NDArray[np.floating] = transform.rotate(_data, angle_value)
        return data

    def apply_rotation(self) -> None:
        """
        Apply the selected rotation to all normalized images.
        
        Executes the rotation transformation based on user selection from the
        settings interface. Uses efficient array operations for 90-degree
        rotation to minimize memory usage and processing time.
        
        Returns:
            None: Modifies parent normalized_images array in place
            
        Side Effects:
            - Modifies self.parent.normalized_images with rotated data
            - Logs rotation progress and array shape changes
            - Updates image dimensions in parent object
            
        Notes:
            - Only applies rotation if '90 degrees' is selected
            - Uses efficient swapaxes + indexing for 90° rotation
            - Preserves data type and memory layout
            - Alternative multiprocessing approach commented out for reference
        """

        logging.info("applying rotation ...")
        str_angle_value: str = self.angle_ui.value
        if str_angle_value == '90 degrees':
            angle_value: int = -90
        else:
            logging.info(f"not applying any rotation to the data!")
            return
    
        logging.info(f"\tangle_value = {angle_value}")

        # worker_with_angle = partial(_worker, angle_value=angle_value)

        # logging.info(f"rotating the normalized_images by {angle_value} ...")        
        # with mp.Pool(processes=5) as pool:
        #      self.parent.normalized_images = pool.map(worker_with_angle, list(self.parent.normalized_images), angle_value)
    
        logging.info(f"\tbefore rotation, {np.shape(self.parent.normalized_images) = }")
        new_array_rotated: NDArray[np.floating] = self.parent.normalized_images.swapaxes(-2, -1)[..., ::-1]
        logging.info(f"\tafter rotation, {np.shape(new_array_rotated) = }")
        # new_array_rotated = []
        # for _data in tqdm(self.parent.normalized_images):
        #     new_array_rotated.append(transform.rotate(_data, angle_value, resize=True))

        self.parent.normalized_images = new_array_rotated[:]
        logging.info(f"rotating the normalized_images ... done!")        
