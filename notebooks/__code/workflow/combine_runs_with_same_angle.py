"""
Open Beam and Dark Current Combination Utilities for CT Reconstruction.

This module provides functionality to combine multiple open beam (OB) and dark current (DC)
images into single representative images for normalization. Combining multiple reference
images improves signal-to-noise ratio and reduces the impact of temporal fluctuations
in beam intensity and detector noise.

Key Classes:
    - CombineObDc: Main class for OB/DC image combination

Key Features:
    - Median-based combination to reduce noise and outliers
    - Support for OB-only combination (ignoring DC)
    - Automatic handling of single-image cases
    - Data type preservation and memory optimization
    - Comprehensive logging of combination process

Mathematical Background:
    Multiple reference images are combined using median aggregation:
    combined_image = median(image_stack, axis=0)
    
    Median combination is preferred over mean because it is more robust
    to outliers such as cosmic rays or detector artifacts.

Dependencies:
    - numpy: Numerical array operations and statistical functions
    - logging: Progress tracking and debugging
    - Parent: Base class providing common functionality

Author: CT Reconstruction Pipeline Team
Created: Part of CT reconstruction development workflow
"""

import numpy as np
import logging
from typing import Optional, List
from numpy.typing import NDArray

from __code.parent import Parent
from __code import DataType


class CombineRunsWithSameAngle(Parent):
    """
   
    """

    def run(self) -> None:
        """
        
        """
        logging.info("Combining runs with same angle if requested:")
       
        if self.parent.how_to_treat_duplicate_angles_ui.value == 'Combine (average)':
            normalized_images = self.parent.normalized_images
            list_angles_deg_vs_runs_dict = self.parent.list_angles_deg_vs_runs_dict
            list_of_angles_deg_to_keep = self.parent.list_of_angles_of_runs_to_keep
            logging.info(f"\tNumber of unique angles: {len(list_angles_deg_vs_runs_dict)}")
            
            if len(list_angles_deg_vs_runs_dict) == len(normalized_images):
                logging.info("\tOnly one run per angle - no combination needed.")
                return
           
            else:
                combined_normalized_images = []
                master_normalized_index = 0
                final_list_of_angles_deg = np.array([], dtype=np.float32)
                for angle in list_of_angles_deg_to_keep:
                    list_runs = list_angles_deg_vs_runs_dict[angle]
                    logging.info(f"\tProcessing angle {angle} deg with {len(list_runs)} runs.")
                    if len(list_runs) == 1:
                        # only one run for this angle - no combination needed
                        combined_normalized_images.append(normalized_images[master_normalized_index])
                        logging.info(f"\t\tAngle {angle} deg: only one run - no combination needed.")
                        
                    else:
                        combined_image = np.mean(np.array([normalized_images[master_normalized_index + i] for i in range(len(list_runs))]), axis=0)
                        combined_normalized_images.append(combined_image)
                        logging.info(f"\t\tAngle {angle} deg: combining {len(list_runs)} runs.")

                    master_normalized_index += len(list_runs)
                    final_list_of_angles_deg = np.append(final_list_of_angles_deg, np.float32(angle))

            self.parent.normalized_images = combined_normalized_images
            self.parent.list_of_angles_deg = final_list_of_angles_deg
            self.parent.list_of_angles_rad = [np.deg2rad(angle) for angle in final_list_of_angles_deg]

            logging.info(f"\tNumber of normalized images after combination: {len(self.parent.normalized_images)}")
            logging.info(f"\t{final_list_of_angles_deg = }")
            logging.info(f"\t{self.parent.list_of_angles_rad = }")
            return

        else:
            logging.info("\tNo combination of runs with same angle requested.")
            return
        