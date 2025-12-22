"""
Data Quality Checking and Validation for CT Reconstruction Pipeline.

This module provides comprehensive data validation and quality checking functionality
for computed tomography reconstruction workflows. It validates run completeness,
extracts metadata from NeXus files, analyzes proton charge consistency, and provides
interactive interfaces for data selection and filtering.

Key Classes:
    - CheckingData: Main class for CT data validation and quality control

Key Features:
    - Run discovery and validation in sample/open beam directories
    - Empty run detection and rejection
    - Proton charge extraction and analysis from NeXus files
    - Rotation angle extraction from file naming conventions
    - Frame number validation and extraction
    - Interactive proton charge selection with tolerance thresholds
    - Minimum requirement validation (OB and sample counts)
    - Quality control visualization and reporting

Dependencies:
    - matplotlib: Data visualization and interactive plotting
    - IPython: Jupyter notebook widget integration
    - numpy: Numerical operations for data analysis

Author: CT Reconstruction Pipeline Team
Created: Part of CT reconstruction development workflow
"""

from typing import Optional, Dict, List, Any, Tuple, Union
import logging
import os
import glob
import re
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from ipywidgets import interactive
from IPython.display import display, HTML
import ipywidgets as widgets

from __code.parent import Parent
from __code.config import PROTON_CHARGE_TOLERANCE_C
from __code import DataType, DetectorType, Run
from __code.utilities.files import retrieve_list_of_runs, retrieve_list_of_tif, get_angle_value, load_spectra_file
from __code.utilities.nexus import get_proton_charge, get_frame_number, get_detector_offset
from __code.utilities.math import calculate_most_dominant_int_value_from_list, calculate_most_dominant_float_value_from_list


class CheckingData(Parent):
    """
    Data quality checking and validation for CT reconstruction pipeline.
    
    This class provides comprehensive validation of CT data including run discovery,
    metadata extraction, proton charge analysis, and quality control checks. It ensures
    data integrity before proceeding with reconstruction workflows.
    
    Inherits from Parent class which provides access to reconstruction pipeline
    state, working directories, and configuration parameters.
    
    Key Features:
        - Automatic run discovery in sample and open beam directories
        - Empty run detection and filtering
        - NeXus metadata extraction (proton charge, frame numbers)
        - Rotation angle parsing from file naming conventions
        - Interactive proton charge selection with tolerance bands
        - Minimum requirement validation for reconstruction
        - Quality control visualization and reporting
    
    Attributes
    ----------
    list_of_runs : Dict[DataType, Optional[Any]]
        Dictionary mapping data types to run information
    list_of_metadata : Dict[str, Any]
        Dictionary storing extracted metadata for each run
    list_proton_charge_c : Dict[DataType, List[float]]
        Proton charge values by data type in Coulombs
    min_proton_charge_c : Dict[DataType, Optional[float]]
        Minimum proton charge values for slider ranges
    max_proton_charge_c : Dict[DataType, Optional[float]]
        Maximum proton charge values for slider ranges
    
    Examples
    --------
    >>> checker = CheckingData(parent=parent_instance)
    >>> checker.run()  # Complete validation workflow
    >>> checker.checking_minimum_requirements()
    """

    list_of_runs: Dict[DataType, Optional[Any]] = {DataType.sample: None,
                                                   DataType.ob: None}
    list_of_metadata: Dict[str, Any] = {}

    # def get_angle_value(self, run_full_path=None):
    #     """ extract the rotation angle value from a string name looking like 
    #     Run_####_20240927_date_..._148_443_######_<file_index>.tif
    #     """
    #     list_tiff = retrieve_list_of_tif(run_full_path)
    #     first_tiff = list_tiff[0]
    #     list_part = first_tiff.split("_")
    #     return f"{list_part[-4]}.{list_part[-3]}"

    def run(self) -> None:
        """
        Execute complete data validation workflow.
        
        Performs comprehensive data checking including run discovery,
        empty run rejection, metadata extraction, and quality control
        visualization. This is the main entry point for data validation.
        
        Workflow Steps:
        1. Discover all runs in sample and OB directories
        2. Reject empty runs (no TIFF files)
        3. Extract proton charge from NeXus files
        4. Extract rotation angles from file names
        5. Extract frame numbers from NeXus files
        6. Display interactive quality control interface
        
        Notes
        -----
        Updates parent object with validated run lists and metadata.
        Sets flags for missing metadata that affects reconstruction.
        """

        # retrieve the full list of runs found in the top folder
        self.retrieve_runs()

        # check empty runs
        self.reject_empty_runs()

        # retrieve proton charge of runs
        self.retrieve_proton_charge()

        # retrieve spectra files
        self.retrieve_spectra_file()

        # retrieve detector offset
        self.retrieve_detector_offset()

        # retrieve rotation angle
        self.retrieve_rotation_angle()

        # retrieve frame number
        self.retrieve_frame_number()

        # # # display graph
        self.display_graph()

    def retrieve_frame_number(self) -> None:
        """
        Extract frame numbers from NeXus files for all runs.
        
        Retrieves frame count information from NeXus metadata files
        associated with each run. Frame numbers are used for data
        validation and reconstruction parameter estimation.
        
        Notes
        -----
        - Iterates through all data types (sample, OB)
        - Updates parent.list_of_runs with frame number information
        - Sets parent.at_least_one_frame_number_not_found flag if any missing
        - Logs progress and results for debugging
        
        Side Effects
        ------------
        Updates parent object state with frame number metadata
        """
        logging.info(f"Retrieving frame numbers:")
        if self.parent.detector_type == DetectorType.tpx3:
            logging.info(f"\tframe number retrieval skipped for TPX3 detector")
            return
        
        self.parent.at_least_one_frame_number_not_found = False
        for _data_type in self.parent.list_of_runs.keys():
            logging.info(f"\t{_data_type}:")
            list_of_frame_number: List[Optional[int]] = []
            for _run in self.parent.list_of_runs[_data_type]:
                logging.info(f"\t\t{_run}")
                # _, number = os.path.basename(_run).split("_")
                # nexus_path = os.path.join(top_nexus_path, f"{self.parent.instrument}_{number}.nxs.h5")
                nexus_path: str = self.parent.list_of_runs[_data_type][_run][Run.nexus]
                logging.info(f"\t\t{nexus_path}")
                frame_number: Optional[int] = get_frame_number(nexus_path)
                logging.info(f"\t\t{frame_number}")
                list_of_frame_number.append(frame_number)
                self.parent.list_of_runs[_data_type][_run][Run.frame_number] = frame_number
                if not frame_number:
                    self.parent.at_least_one_frame_number_not_found = True
                
            logging.info(f"\t\t{list_of_frame_number}")

    def retrieve_rotation_angle(self) -> None:
        """
        Extract rotation angles from sample run file names.
        
        Parses rotation angle values from TIFF file naming conventions
        for sample runs. Angles are extracted from standardized file
        name patterns and used for reconstruction geometry setup.
        
        Notes
        -----
        - Only processes sample runs (not OB/DC)
        - Only processes runs marked as 'use_it' = True
        - Updates parent.list_of_runs with angle information
        - Creates angle-to-run mapping dictionary
        - Logs angle extraction results
        
        Side Effects
        ------------
        Updates parent.list_angles_deg_vs_runs_dict mapping
        """
        logging.info(f"Retrieving rotation angles:")
        list_of_sample_runs: Dict[str, Any] = self.parent.list_of_runs[DataType.sample]
        list_angles_deg_vs_runs_dict: Dict[str, list] = {}
        for _run in list_of_sample_runs.keys():
            if list_of_sample_runs[_run][Run.use_it]:
                angle_value: str = get_angle_value(run_full_path=list_of_sample_runs[_run][Run.full_path],
                                                   detector_type=self.parent.detector_type)
                if angle_value not in list_angles_deg_vs_runs_dict:
                    list_angles_deg_vs_runs_dict[str(angle_value)] = []
                list_angles_deg_vs_runs_dict[str(angle_value)].append(_run)
                self.parent.list_of_runs[DataType.sample][_run][Run.angle] = angle_value
                logging.info(f"\t{_run}: {angle_value}")
            else:
                logging.info(f"\t{_run}: not used")
        self.parent.list_angles_deg_vs_runs_dict = list_angles_deg_vs_runs_dict

    def retrieve_runs(self) -> None:
        """
        Discover and catalog all runs in sample and open beam directories.
        
        Scans the working directories for CT data runs and creates a comprehensive
        catalog with metadata placeholders. Validates that required runs exist
        and sets up the data structure for subsequent processing.
        
        Raises
        ------
        ValueError
            If no runs are found in required directories
            
        Notes
        -----
        - Scans both sample and OB directories
        - Creates NeXus file path associations
        - Initializes run metadata structure
        - Validates minimum run count requirements
        - Sets up run processing flags
        
        Side Effects
        ------------
        Populates parent.list_of_runs with discovered run information
        """
        ''' retrieve the full list of runs in the top folder of sample and ob '''
        
        logging.info(f"Retrieving runs:")
        for _data_type in self.list_of_runs.keys():
            logging.info(f"\t{_data_type}:")
            list_of_runs: List[str] = retrieve_list_of_runs(top_folder=self.parent.working_dir[_data_type],
                                                            detector_type=self.parent.detector_type)
            if len(list_of_runs) == 0:
                display(HTML(f"<font color=red>Found 0 {_data_type} runs in {self.parent.working_dir[_data_type]}</font>"))
                raise ValueError("Missing files !")
            
            logging.info(f"\tfound {len(list_of_runs)} {_data_type} runs")
            
            top_nexus_path: str = self.parent.working_dir[DataType.nexus]
            for _run in list_of_runs:
                run_number = self.extract_run_number(_run, self.parent.detector_type)
                # _, number = os.path.basename(_run).split("_")
                nexus_path: str = os.path.join(top_nexus_path, f"{self.parent.instrument}_{run_number}.nxs.h5")
                self.parent.list_of_runs[_data_type][os.path.basename(_run)] = {Run.full_path: _run,
                                                                                Run.proton_charge_c: None,
                                                                                Run.use_it: True,
                                                                                Run.angle: None,
                                                                                Run.nexus: nexus_path,
                                                                                Run.frame_number: None,
                                                                                }

    def extract_run_number(self, run_full_path: str, detector_type: DetectorType) -> Optional[int]:
        """
        Extract the run number from the full path of a run directory.

        Args:
            run_full_path: Full path to the run directory
            detector_type: Type of detector being used

        Returns:
            Extracted run number as integer, or None if not found
        """
        match = re.search(r"Run_(\d+)", run_full_path)

        if match:
            return int(match.group(1))
        return None

    def reject_empty_runs(self) -> None:
        """
        Identify and mark empty runs for exclusion from processing.
        
        Scans each discovered run directory to check for TIFF files.
        Runs without TIFF files are marked as unusable to prevent
        processing errors during reconstruction.
        
        Notes
        -----
        - Checks all data types (sample, OB)
        - Counts TIFF files in each run directory
        - Updates run 'use_it' flag based on file presence
        - Logs rejected runs for debugging
        - Maintains lists of valid and invalid runs
        
        Side Effects
        ------------
        Updates parent.list_of_runs run processing flags
        """

        logging.info(f"Rejecting empty runs:")
        for _data_type in self.parent.list_of_runs.keys():
            list_of_runs: List[str] = list(self.parent.list_of_runs[_data_type].keys())

            list_of_runs_to_keep: List[str] = []
            list_of_runs_to_remove: List[str] = []

            for _run in list_of_runs:

                _run_full_path: str = self.parent.list_of_runs[_data_type][_run][Run.full_path]
                list_tif: List[str] = retrieve_list_of_tif(_run_full_path)
                if len(list_tif) > 0:
                    list_of_runs_to_keep.append(_run_full_path)
                    self.parent.list_of_runs[_data_type][_run][Run.use_it] = True
                else:
                    list_of_runs_to_remove.append(_run)
                    self.parent.list_of_runs[_data_type][_run][Run.use_it] = False

            logging.info(f"\trejected {len(list_of_runs_to_remove)} {_data_type} runs")
            logging.info(f"\t -> {[os.path.basename(_file) for _file in list_of_runs_to_remove]}")

    def retrieve_proton_charge(self) -> None:
        """
        Extract proton charge values from NeXus files for all runs.
        
        Retrieves proton charge measurements from NeXus metadata files for
        data quality assessment and run filtering. Proton charge consistency
        is critical for neutron tomography data normalization and quality.
        
        The method processes both sample and open beam runs, extracting proton
        charge values and calculating statistics for interactive selection
        widgets. Missing proton charge data is flagged for user attention.
        
        Notes
        -----
        - Converts raw proton charge from pC to C (divides by 1e12)
        - Calculates min/max ranges for interactive slider widgets
        - Sets flags for missing proton charge data
        - Stores results in instance attributes for visualization
        - Updates parent object with extracted metadata
        
        Side Effects
        ------------
        - Updates self.list_proton_charge_c with extracted values
        - Updates self.min_proton_charge_c and self.max_proton_charge_c
        - Sets parent.at_lest_one_proton_charge_not_found flag
        - Updates parent.list_of_runs with proton charge metadata
        
        Attributes Updated
        ------------------
        list_proton_charge_c : Dict[DataType, List[float]]
            Proton charge values in Coulombs by data type
        min_proton_charge_c : Dict[DataType, Optional[float]]
            Minimum proton charge for slider range (with buffer)
        max_proton_charge_c : Dict[DataType, Optional[float]]
            Maximum proton charge for slider range (with buffer)
        """
        logging.info(f"Retrieving proton charge values:")
        top_nexus_path: str = self.parent.working_dir[DataType.nexus]
        logging.info(f"\ttop_nexus_path: {top_nexus_path}")

        list_proton_charge_c: Dict[DataType, List[float]] = {DataType.sample: [],
                                                            DataType.ob: []} 
        
        min_proton_charge_c: Dict[DataType, Optional[float]] = {DataType.sample: None,
                                                               DataType.ob: None}
        
        max_proton_charge_c: Dict[DataType, Optional[float]] = {DataType.sample: None,
                                                               DataType.ob: None}

        at_lest_one_proton_charge_not_found: bool = False
        for _data_type in self.parent.list_of_runs.keys():
            _list_proton_charge: List[Optional[float]] = []
            for _run in self.parent.list_of_runs[_data_type]:
                logging.info(f"\t{_run = }")
                nexus_path: str = self.parent.list_of_runs[_data_type][_run][Run.nexus]
                logging.info(f"\t\t{nexus_path = }")
                proton_charge: Optional[float] = get_proton_charge(nexus_path)
                logging.info(f"\t\t{proton_charge = }")
                _list_proton_charge.append(proton_charge)
                self.list_of_metadata[_run] = proton_charge
                if proton_charge is not None:
                    self.parent.list_of_runs[_data_type][_run][Run.proton_charge_c] = proton_charge/1e12
                else:
                    at_lest_one_proton_charge_not_found = True
                    self.parent.list_of_runs[_data_type][_run][Run.proton_charge_c] = None

            if not at_lest_one_proton_charge_not_found:
                list_proton_charge_c[_data_type] = [_pc/1e12 for _pc in _list_proton_charge if _pc is not None]
                min_proton_charge_c[_data_type] = min(list_proton_charge_c[_data_type]) - 1
                max_proton_charge_c[_data_type] = max(list_proton_charge_c[_data_type]) + 1
                logging.info(f"\t{_data_type}: {list_proton_charge_c[_data_type]}")
            else:
                logging.info(f"\t{_data_type}: proton charge not available")

        self.list_proton_charge_c = list_proton_charge_c
        self.min_proton_charge_c = min_proton_charge_c
        self.max_proton_charge_c = max_proton_charge_c
        self.parent.at_lest_one_proton_charge_not_found = at_lest_one_proton_charge_not_found
  
    def retrieve_spectra_file(self) -> None:
        logging.info(f"retrieve_spectra_file")
        list_of_sample_runs: Dict[str, Any] = self.parent.list_of_runs[DataType.sample]
        list_of_runs = list(list_of_sample_runs.keys())
        first_run = list_of_runs[0]
        full_path = list_of_sample_runs[first_run][Run.full_path]
        list_spectra_files_in_that_folder = glob.glob(os.path.join(full_path, "*_Spectra.txt"))
        if list_spectra_files_in_that_folder:
            first_spectra_file = list_spectra_files_in_that_folder[0]
            logging.info(f"\t first_spectra_file: {first_spectra_file}")
            tof_array = load_spectra_file(first_spectra_file)
        else:
            tof_array = None
        
        self.parent.tof_array = tof_array
   
    def retrieve_detector_offset(self) -> None:
        """
        Retrieve detector offset from the first sample run.
        
        Extracts the detector offset value from the NeXus file of the
        first sample run. This offset is used for time-of-flight (TOF)
        calculations and data normalization in neutron tomography.
        
        Notes
        -----
        - Identifies the first sample run from the list of runs
        - Reads the NeXus file to extract detector offset metadata
        - Updates parent.detector_offset attribute with extracted value
        - Logs progress and results for debugging
        
        Side Effects
        ------------
        Sets parent.detector_offset with extracted offset value
        """
        logging.info(f"Retrieving detector offset from first sample run ...")
        list_of_runs = list(self.parent.list_of_runs[DataType.sample].keys())
        first_run = list_of_runs[0]
        nexus_path = self.parent.list_of_runs[DataType.sample][first_run][Run.nexus]
        logging.info(f"\t nexus_path: {nexus_path}")
        detector_offset, detector_offset_units = get_detector_offset(nexus_path)
        logging.info(f"\t detector_offset: {detector_offset} {detector_offset_units}")
        self.parent.detector_offset = detector_offset
        self.parent.detector_offset_units = detector_offset_units
  
    def display_graph(self) -> None:
        """
        Display interactive proton charge analysis and selection interface.
        
        Creates an interactive visualization of proton charge values for both
        sample and open beam runs with adjustable selection thresholds. Users
        can set tolerance bands to filter runs based on proton charge consistency.
        
        The interface includes:
        - Scatter plot of proton charge values by run index
        - Interactive sliders for target proton charge values
        - Tolerance threshold adjustment
        - Visual tolerance bands (shaded regions)
        - Real-time plot updates based on slider values
        
        Notes
        -----
        - Calculates default values using most dominant proton charge
        - Creates interactive matplotlib plot with ipywidgets controls
        - Stores interactive widget in parent.selection_of_pc
        - Provides visual feedback for run selection criteria
        - Tolerance bands help identify outlier runs for exclusion
        
        Side Effects
        ------------
        Creates and displays interactive widget interface for proton charge selection
        """

        default_sample_proton_charge: float = calculate_most_dominant_float_value_from_list(self.list_proton_charge_c[DataType.sample])
        default_ob_proton_charge: float = calculate_most_dominant_float_value_from_list(self.list_proton_charge_c[DataType.ob])

        logging.info(f"-- display graph --")

        def plot_proton_charges(sample_proton_charge_value: float, 
                               ob_proton_charge_value: float, 
                               proton_charge_threshold: float) -> Tuple[float, float, float]:
            """
            Plot proton charge values with interactive tolerance bands.
            
            Parameters
            ----------
            sample_proton_charge_value : float
                Target proton charge for sample runs
            ob_proton_charge_value : float
                Target proton charge for open beam runs  
            proton_charge_threshold : float
                Tolerance threshold for run selection
                
            Returns
            -------
            Tuple[float, float, float]
                Sample proton charge, OB proton charge, threshold values
            """
            fig, axs = plt.subplots(nrows=1, ncols=1)
            axs.set_title("Proton charge (C) of selected runs")
            axs.plot(self.list_proton_charge_c[DataType.sample], 'g+', label=DataType.sample)
            axs.plot(self.list_proton_charge_c[DataType.ob], 'bo', label=DataType.ob)
            axs.set_xlabel("file index")
            axs.set_ylabel("proton charge (C)")
            axs.legend()
    
            axs.axhline(sample_proton_charge_value, linestyle='--', color='green')
            sample_proton_charge_range: List[float] = [sample_proton_charge_value + proton_charge_threshold,
                                                      sample_proton_charge_value - proton_charge_threshold]
            axs.axhspan(sample_proton_charge_range[0], 
                        sample_proton_charge_range[1], facecolor='green', alpha=0.2)

            axs.axhline(ob_proton_charge_value, linestyle='--', color='blue')
            ob_proton_charge_range: List[float] = [ob_proton_charge_value + proton_charge_threshold,
                                                  ob_proton_charge_value - proton_charge_threshold]
            axs.axhspan(ob_proton_charge_range[0], 
                        ob_proton_charge_range[1], facecolor='blue', alpha=0.2)

            # plt.show()

            return sample_proton_charge_value, ob_proton_charge_value, proton_charge_threshold

        self.parent.selection_of_pc = interactive(plot_proton_charges,
                            sample_proton_charge_value = widgets.FloatSlider(min=self.min_proton_charge_c[DataType.sample],
                                                                    max=self.max_proton_charge_c[DataType.sample],
                                                                    value=default_sample_proton_charge,
                                                                    description='sample pc',
                                                                    continuous_update=True),
                            ob_proton_charge_value = widgets.FloatSlider(min=self.min_proton_charge_c[DataType.ob],
                                                                    max=self.max_proton_charge_c[DataType.ob],
                                                                    value=default_ob_proton_charge,
                                                                    description='ob pc',
                                                                    continuous_update=True),
                            proton_charge_threshold = widgets.FloatSlider(min=0.0001,
                                                                        max=1,
                                                                        step=0.01,
                                                                        description='threshold',
                                                                        value=PROTON_CHARGE_TOLERANCE_C,
                                                                        continuous_update=True),
                                                                        )
        display(self.parent.selection_of_pc)

    def checking_minimum_requirements(self) -> None:
        """
        Validate minimum data requirements for CT reconstruction.
        
        Checks that sufficient data is available for reconstruction by
        verifying minimum counts of open beam and sample runs. Sets
        the minimum_requirements_met flag based on validation results.
        
        Requirements:
        - At least 1 open beam (OB) run available
        - At least 1 OB run selected (not rejected)
        - At least 3 sample runs available  
        - At least 1 sample run selected (not rejected)
        
        Notes
        -----
        - Uses UI selection widgets to determine kept/rejected runs
        - Logs validation results for debugging
        - Updates parent.minimum_requirements_met flag
        - Called before proceeding to reconstruction steps
        
        Side Effects
        ------------
        Sets parent.minimum_requirements_met boolean flag
        """
        """at least 1 OB and 3 samples selected"""
        logging.info(f"Checking minimum requirements:")
                
        list_ob: Tuple[str, ...] = self.parent.list_of_ob_runs_to_reject_ui.options
        list_ob_selected: Tuple[str, ...] = self.parent.list_of_ob_runs_to_reject_ui.value

        # at least 1 OB
        if len(list_ob) == 0:
            logging.info(f"\tno OB available. BAD!")
            self.parent.minimum_requirements_met = False
            return
        
        # at least 1 OB to keep
        if len(list_ob) == len(list_ob_selected):
            logging.info(f"\tnot keeping any OB. BAD!")
            self.parent.minimum_requirements_met = False
            return
        
        list_sample: Tuple[str, ...] = self.parent.list_of_sample_runs_to_reject_ui.options
        list_sample_selected: Tuple[str, ...] = self.parent.list_of_sample_runs_to_reject_ui.value
        
        # at least 3 projections
        if len(list_sample) < 3:
            logging.info(f"\tless than 3 sample runs available. BAD!")
            self.parent.minimum_requirements_met = False
            return
        
        if len(list_sample) == len(list_sample_selected):
            logging.info(f"\tnot keeping any sample run. BAD!")
            self.parent.minimum_requirements_met = False
            return
        
        logging.info(f"At least 1 OB and 3 sample runs. GOOD!")
        self.parent.minimum_requirements_met = True

    def minimum_requirement_not_met(self) -> None:
        """
        Display error message when minimum data requirements are not met.
        
        Shows user-friendly error message indicating that insufficient
        data is available for reconstruction. Called when validation
        fails to provide clear feedback about data requirements.
        
        Notes
        -----
        - Displays HTML formatted error message
        - Logs requirement failure for debugging
        - Provides specific guidance on minimum requirements
        """
        display(HTML(f"<font color=red><b>STOP!</b> Make sure you have at least 3 sample and 1 OB selected!</font>"))
        logging.info(f"Minimum requirement not met!")
