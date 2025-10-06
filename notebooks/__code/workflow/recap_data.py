"""
Data Recapitulation and Run Summary for CT Reconstruction Pipeline.

This module provides functionality for summarizing and validating the final
selection of runs for CT reconstruction. It processes the filtered runs based
on proton charge criteria and provides visual feedback on the final dataset
composition for both sample and open beam measurements.

Key Classes:
    - RecapData: Main class for data recapitulation and run summary workflow

Key Features:
    - Proton charge validation and filtering
    - Final run list preparation and validation
    - Interactive display of selected runs
    - Quality metrics and statistics reporting
    - Integration with parent workflow configuration

Dependencies:
    - matplotlib: Data visualization and plotting
    - IPython: Jupyter notebook widget integration
    - numpy: Numerical computing support

Author: CT Reconstruction Pipeline Team
Created: Part of CT reconstruction development workflow
"""

import logging
import os
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interactive
from IPython.display import display
import ipywidgets as widgets
from IPython.display import HTML, display
from numpy.typing import NDArray

from __code.parent import Parent
from __code import DataType, Run
from __code.config import DEBUG


class RecapData(Parent):
    """
    Data recapitulation and run summary for CT reconstruction workflow.
    
    This class provides comprehensive functionality for summarizing the final
    selection of runs that will be used in the CT reconstruction process. It
    validates proton charge criteria, prepares final run lists, and provides
    visual feedback on dataset composition.
    
    Attributes:
        final_list_of_runs: Dictionary containing final filtered runs for sample and OB data
    
    Methods:
        is_pc_within_range: Static method to check if proton charge is within acceptable range
        run: Main execution method for the recapitulation workflow
        prepare_list_of_runs: Filters runs based on proton charge criteria
        display_list_of_runs: Displays summary of selected runs with statistics
    """

    final_list_of_runs: Dict[DataType, Optional[List[str]]] = {DataType.sample: None,
                          DataType.ob: None}

    @staticmethod
    def is_pc_within_range(pc_value: float = 0, pc_requested: float = 0, threshold: float = 1) -> bool:
        """
        Check if proton charge value is within acceptable range of requested value.
        
        This method validates whether a measured proton charge value falls within
        the specified threshold of the requested proton charge, ensuring data
        quality and consistency across measurements.
        
        Args:
            pc_value: Measured proton charge value
            pc_requested: Target/requested proton charge value
            threshold: Acceptable deviation threshold (default: 1)
            
        Returns:
            True if proton charge is within acceptable range, False otherwise
            
        Example:
            >>> RecapData.is_pc_within_range(10.5, 10.0, 1.0)
            True
            >>> RecapData.is_pc_within_range(12.0, 10.0, 1.0)
            False
        """
        if np.abs(pc_value - pc_requested) < threshold:
            return True
        else:
            return False

    def run(self) -> None:
        """
        Execute the main data recapitulation workflow.
        
        This method orchestrates the complete process of preparing and displaying
        the final list of runs that will be used for CT reconstruction. It applies
        proton charge filtering and generates summary reports.
        """
        self.prepare_list_of_runs()
        self.display_list_of_runs()

    def prepare_list_of_runs(self) -> None:
        """
        Prepare and filter the final list of runs based on proton charge criteria.
        
        This method processes all available runs for both sample and open beam data,
        applying proton charge filtering to ensure data quality and consistency.
        Only runs that meet the proton charge requirements and are marked for use
        will be included in the final reconstruction dataset.
        
        The filtering process considers:
        - User-specified proton charge targets for sample and OB data
        - Acceptable deviation thresholds
        - Run usage flags set by previous workflow steps
        """
        logging.info(f"Preparing list of runs to use:")
        pc_sample_requested: float
        pc_ob_requested: float
        pc_threshold: float
        pc_sample_requested, pc_ob_requested, pc_threshold = self.parent.selection_of_pc.result
        logging.info(f"\t{pc_sample_requested = }")
        logging.info(f"\t{pc_ob_requested = }")
        logging.info(f"\t{pc_threshold = }")

        list_of_runs: Dict[DataType, Dict[str, Dict[Run, Any]]] = self.parent.list_of_runs
        
        final_list_of_sample_runs: List[str] = []
        for _run in list_of_runs[DataType.sample].keys():
            logging.info(f"Working with {DataType.sample}")

            if list_of_runs[DataType.sample][_run][Run.use_it]:
    
                if list_of_runs[DataType.sample][_run][Run.use_it]:
                    _pc: float = list_of_runs[DataType.sample][_run][Run.proton_charge_c]
                    _angle: float = list_of_runs[DataType.sample][_run][Run.angle]
                    if RecapData.is_pc_within_range(pc_value=_pc,
                                                    pc_requested=pc_sample_requested,
                                                    threshold=pc_threshold):
                        final_list_of_sample_runs.append(_run)
                        logging.info(f"\t{_run} with pc of {_pc} ({_angle} degrees) is within the range !")
                        list_of_runs[DataType.sample][_run][Run.use_it] = True
                    else:
                        logging.info(f"\t{_run} with pc of {_pc} ({_angle} degrees) is not within the range !")
                        list_of_runs[DataType.sample][_run][Run.use_it] = False
                
                else:
                    logging.info(f"\t{_run} can not be used!")
                
        self.final_list_of_runs[DataType.sample] = final_list_of_sample_runs

        final_list_of_ob_runs: List[str] = []
        for _run in list_of_runs[DataType.ob].keys():
            logging.info(f"Working with {DataType.ob}")

            if list_of_runs[DataType.ob][_run][Run.use_it]:
                _pc: float = list_of_runs[DataType.ob][_run][Run.proton_charge_c]
                if RecapData.is_pc_within_range(pc_value=_pc,
                                                pc_requested=pc_ob_requested,
                                                threshold=pc_threshold):
                    final_list_of_ob_runs.append(_run)
                    list_of_runs[DataType.ob][_run][Run.use_it] = True
                    logging.info(f"\t{_run} with pc of {_pc} is within the range !")

                else:
                    list_of_runs[DataType.ob][_run][Run.use_it] = False
                    logging.info(f"\t{_run} with pc of {_pc} is not within the range !")

            else:
                logging.info(f"\t{_run} can not be used!")

        self.final_list_of_runs[DataType.ob] = final_list_of_ob_runs
        self.parent.final_list_of_runs = self.final_list_of_runs
        self.parent.list_of_runs = list_of_runs

    def display_list_of_runs(self) -> None:
        """
        Display interactive interface for final run selection and rejection.
        
        This method creates an interactive widget interface that allows users to
        review and optionally reject specific runs from the final dataset. It
        provides separate selection lists for sample and open beam runs with
        the ability to exclude problematic runs from the reconstruction process.
        
        The interface includes:
        - Multi-selection lists for sample and OB runs
        - Debug mode defaults for testing
        - Clear all functionality for easy reset
        """

        if DEBUG:
            default_list_sample: Optional[List[str]] = self.final_list_of_runs[DataType.sample][3:]
            default_list_ob: Optional[List[str]] = self.final_list_of_runs[DataType.ob][1:]
        else:
            default_list_sample = []
            default_list_ob = []

        final_list_of_sample: List[str] = self.final_list_of_runs[DataType.sample][:]
        sample_runs: widgets.VBox = widgets.VBox([
            widgets.Label("Sample"),
            widgets.SelectMultiple(options=final_list_of_sample,
                                   value=default_list_sample,
                                    layout=widgets.Layout(height="100%",
                                                            width='100%',
                                                            )),                                                       
        ],
        layout=widgets.Layout(width='600px',
                                height='300px'))
        self.parent.list_of_sample_runs_to_reject_ui = sample_runs.children[1]

        final_list_of_ob: List[str] = self.final_list_of_runs[DataType.ob][:]
        ob_runs: widgets.VBox = widgets.VBox([
            widgets.Label("OB"),
            widgets.SelectMultiple(options=final_list_of_ob,
                                   value=default_list_ob,
                                    layout=widgets.Layout(height="100%",
                                                            width='100%'))
        ],
        layout=widgets.Layout(width='600px',
                                height='300px'))
        self.parent.list_of_ob_runs_to_reject_ui = ob_runs.children[1]

        title: widgets.HTML = widgets.HTML("<b>Select any run(s) you want to exclude!:")

        hori_layout: widgets.HBox = widgets.HBox([sample_runs, ob_runs])
        verti_layout: widgets.VBox = widgets.VBox([title, hori_layout])
        display(verti_layout)

        clear_all: widgets.Button = widgets.Button(description="Clear All")
        display(clear_all)
        clear_all.on_click(self.clear_all)

        display(HTML("<hr>"))
        display(HTML("<h3>How to treat duplicate angles (this will affect your selection above!):</h3>"))
        how_to_treat_duplicate_angles_ui = widgets.RadioButtons(options=['Keep only one', 'Combine (average)'],
                              value='Keep only one',
                              description='',
                              disabled=False)
        self.parent.how_to_treat_duplicate_angles_ui = how_to_treat_duplicate_angles_ui
        display(how_to_treat_duplicate_angles_ui)
        how_to_treat_duplicate_angles_ui.observe(self.on_change_how_to_treat_duplicate_angles, names='value')


    def on_change_how_to_treat_duplicate_angles(self, change: Dict[str, Any]) -> None:
        """
        Callback for changes in duplicate angle treatment option.
        
        This method updates the parent workflow configuration based on user
        selection of how to handle duplicate angles in the dataset. It ensures
        that the chosen strategy is reflected in subsequent processing steps.
        
        Args:
            change: Dictionary containing change information from the widget
        """
        self.parent.how_to_treat_duplicate_angles = change['new']
        # logging.info(f"How to treat duplicate angles: {change['new']}")
        if change['new'] == 'Keep only one':
            final_list_of_sample: List[str] = self.final_list_of_runs[DataType.sample][:]
            no_duplicate_final_list_of_sample: List[str] = []
            for _run in final_list_of_sample:
                # keep string beyond _Ang_ to identify duplicates
                angle_str: str = _run.split("_Ang_")[-1]
                if not any(angle_str in s for s in no_duplicate_final_list_of_sample):
                    no_duplicate_final_list_of_sample.append(_run)  
            self.parent.list_of_sample_runs_to_reject_ui.value = [run for run in final_list_of_sample if run not in no_duplicate_final_list_of_sample]
            self.parent.list_of_sample_runs_to_reject_ui.disabled = False
        else:
            self.parent.list_of_sample_runs_to_reject_ui.value = []
            self.parent.list_of_sample_runs_to_reject_ui.disabled = True

    def clear_all(self, _: Any) -> None:
        """
        Clear all selected runs from rejection lists.
        
        This method resets both sample and open beam run rejection lists,
        effectively including all filtered runs back into the reconstruction
        dataset. Useful for quickly resetting the selection state.
        
        Args:
            _: Widget callback parameter (unused)
        """
        self.parent.list_of_sample_runs_to_reject_ui.value = []
        self.parent.list_of_ob_runs_to_reject_ui.value = []
        logging.info(f"Clearing all selected runs")
        