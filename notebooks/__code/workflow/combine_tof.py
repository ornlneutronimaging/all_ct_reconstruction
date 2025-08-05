"""
Time-of-Flight (TOF) Data Combination for Neutron CT Reconstruction.

This module provides functionality for combining time-of-flight neutron imaging data
in computed tomography workflows. TOF neutron imaging captures data across multiple
wavelength channels, and this module handles the selection, filtering, and combination
of runs based on angular positions and data quality criteria.

Key Classes:
    - CombineTof: Main class for TOF data combination operations

Key Features:
    - Selective run inclusion/exclusion based on quality criteria
    - Angular-based data organization and sorting
    - Multi-threaded data loading for performance optimization
    - TOF channel combination and wavelength integration
    - Support for sample, open beam, and dark current data types
    - Progress tracking for large datasets

Time-of-Flight Neutron Imaging:
    TOF neutron imaging exploits the wavelength-dependent properties of neutron
    interactions with matter. Different neutron wavelengths provide complementary
    information about material composition and structure. The combination process
    can involve:
    - Wavelength-specific analysis for material identification
    - Spectral integration for enhanced contrast
    - Energy-resolved reconstruction for advanced analysis

Mathematical Background:
    TOF data combination typically involves:
    - Spectral integration: I_combined = ∑(λ) I(λ) * w(λ)
    - Weighted averaging across wavelength channels
    - Angular sorting for proper sinogram construction
    
Dependencies:
    - numpy: Numerical array operations and integration
    - tqdm: Progress tracking for batch processing
    - logging: Process monitoring and debugging
    - Parent: Base class providing common functionality

Author: CT Reconstruction Pipeline Team
Created: Part of neutron CT reconstruction development workflow
"""

import logging
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Union, Any
from numpy.typing import NDArray

from __code import OperatingMode
from __code.parent import Parent
from __code import DataType, Run
from __code.utilities.load import load_list_of_tif, load_data_using_multithreading
from __code.utilities.files import retrieve_list_of_tif


class CombineTof(Parent):
    """
    Handles combination and organization of time-of-flight neutron CT data.
    
    This class provides comprehensive functionality for processing TOF neutron
    imaging data, including run selection based on quality criteria, angular
    organization, and spectral data combination. It manages the complex workflow
    of combining multiple TOF acquisitions into organized datasets suitable for
    tomographic reconstruction.
    
    Key Features:
        - Quality-based run selection and rejection
        - Angular sorting and organization
        - Multi-threaded data loading for performance
        - Support for sample, open beam, and dark current data
        - TOF spectral data combination
        - Progress tracking and logging
        
    Methods:
        run(): Execute the complete TOF combination workflow
        update_list_of_runs_status(): Apply quality-based run filtering
        load_data(): Load and organize data by angular position
        load_data_for_a_run(): Load individual run data with TOF combination
    """

    def run(self) -> None:
        """
        Execute the complete TOF data combination workflow.
        
        Orchestrates the full process of combining TOF neutron CT data,
        including run status updates, data loading, and organization.
        This is the main entry point for TOF data processing.
        
        Returns:
            None: Modifies parent object with combined data
            
        Side Effects:
            - Updates run status based on quality criteria
            - Loads and organizes all selected data
            - Creates master_3d_data_array with combined data
            - Sets up angular and run lists for reconstruction
            
        Workflow:
            1. Update run inclusion/exclusion status
            2. Load data for all selected runs
            3. Organize data by angular position
            4. Create master data arrays for reconstruction
        """      
        self.update_list_of_runs_status()
        self.load_data()
        # self.combine_tof_data_range(self.parent.master_3d_data_array)

    def update_list_of_runs_status(self) -> None:
        """
        Update run inclusion status based on user selection criteria.
        
        Processes user-specified lists of runs to reject for both sample and
        open beam data types. Marks rejected runs as not to be used in the
        final reconstruction workflow.
        
        Returns:
            None: Modifies parent run status in place
            
        Side Effects:
            - Updates self.parent.list_of_runs with rejection flags
            - Marks rejected runs with Run.use_it = False
            - Applies to both sample and open beam data types
            
        Notes:
            - Uses UI selection widgets for run rejection lists
            - Maintains original run metadata while updating status
            - Essential for quality control in TOF data processing
        """

        # update list of runs to reject
        list_of_runs: Dict[DataType, Dict[str, Dict[Run, Any]]] = self.parent.list_of_runs
        logging.info(f"list_of_runs = {list_of_runs}")

        list_ob_to_reject: List[str] = self.parent.list_of_ob_runs_to_reject_ui.value
        for _run in list_ob_to_reject:
            list_of_runs[DataType.ob][_run][Run.use_it] = False

        list_sample_to_reject: List[str] = self.parent.list_of_sample_runs_to_reject_ui.value
        for _run in list_sample_to_reject:
            list_of_runs[DataType.sample][_run][Run.use_it] = False

        self.parent.list_of_runs = list_of_runs

    def load_data(self) -> None:
        """
        Load and organize TOF data for all selected runs by angular position.
        
        Loads TOF neutron imaging data for all non-rejected runs, organizing
        them by angular position for proper tomographic reconstruction. Handles
        both sample and open beam data with progress tracking and logging.
        
        Returns:
            None: Updates parent object with loaded and organized data
            
        Side Effects:
            - Creates self.parent.master_3d_data_array with sample/OB data
            - Sets self.parent.final_list_of_angles with angular positions
            - Creates self.parent.final_list_of_angles_rad in radians
            - Updates self.parent.final_list_of_runs and list_of_images
            - Logs data shapes and organization details
            
        Process:
            1. Sort angles and process sample data by angular position
            2. Load and combine TOF data for each accepted run
            3. Process open beam data with similar methodology
            4. Create master arrays and finalize data organization
            
        Notes:
            - Uses multi-threaded loading for performance
            - Combines TOF channels during loading process
            - Maintains correspondence between angles and runs
        """

        logging.info(f"loading data ...")

        # for sample
        logging.info(f"\tworking with sample")
        list_angles_deg_vs_runs_dict: Dict[str, str] = self.parent.list_angles_deg_vs_runs_dict
        list_angles: List[str] = list(list_angles_deg_vs_runs_dict.keys())
        list_angles.sort()

        list_of_runs: Dict[DataType, Dict[str, Dict[Run, Any]]] = self.parent.list_of_runs

        list_of_angles_of_runs_to_keep: List[str] = []
        master_3d_data_array: Dict[DataType, Optional[NDArray[np.floating]]] = {DataType.sample: None,
                                DataType.ob: None,
                                DataType.dc: None}

        list_sample_data: List[NDArray[np.floating]] = []
        final_list_of_runs: Dict[DataType, List[str]] = {DataType.sample: [], DataType.ob: []}

        for _angle in tqdm(list_angles):
            _runs: str = list_angles_deg_vs_runs_dict[_angle]
            logging.info(f"Working with angle {_angle} degrees")
            logging.info(f"\t{_runs}")

            use_it: bool = list_of_runs[DataType.sample][_runs][Run.use_it]
            if use_it:
                logging.info(f"\twe keep that runs!")
                list_of_angles_of_runs_to_keep.append(_angle)
                logging.info(f"\tloading run {_runs} ...")
                _data: NDArray[np.floating] = self.load_data_for_a_run(run=_runs)
                logging.info(f"\t{_data.shape}")
                # # combine all tof
                # _data = np.sum(_data, axis=0)
                list_sample_data.append(_data)
                final_list_of_runs[DataType.sample].append(_runs)
            else:
                logging.info(f"\twe reject that runs!")

        master_3d_data_array[DataType.sample] = np.array(list_sample_data)

        # for ob
        logging.info(f"\tworking with ob")

        list_ob_data: List[NDArray[np.floating]] = []

        for _run in tqdm(list_of_runs[DataType.ob]):
            logging.info(f"Working with run {_run}")
            use_it: bool = list_of_runs[DataType.ob][_run][Run.use_it]
            if use_it:
                logging.info(f"\twe keep that runs!")
                logging.info(f"\tloading run {_run} ...")
                _data: NDArray[np.floating] = self.load_data_for_a_run(run=_run, data_type=DataType.ob)
                logging.info(f"\t{_data.shape}")
                list_ob_data.append(_data)
                final_list_of_runs[DataType.ob].append(_runs)

            else:
                logging.info(f"\twe reject that runs!")

        master_3d_data_array[DataType.ob] = np.array(list_ob_data)

        self.parent.master_3d_data_array = master_3d_data_array
        self.parent.final_list_of_angles = list_of_angles_of_runs_to_keep
        self.parent.final_list_of_angles_rad = [np.deg2rad(float(_angle)) for _angle in list_of_angles_of_runs_to_keep]
        self.parent.final_list_of_runs = final_list_of_runs
        self.parent.list_of_images = final_list_of_runs

        logging.info(f"{master_3d_data_array[DataType.sample].shape = }")
        logging.info(f"{master_3d_data_array[DataType.ob].shape = }")
        logging.info(f"{list_of_angles_of_runs_to_keep = }")
        logging.info(f"{self.parent.final_list_of_angles_rad = }")
        logging.info(f"{self.parent.list_of_images = }")

    def load_data_for_a_run(self, run: Optional[str] = None, data_type: DataType = DataType.sample) -> NDArray[np.floating]:
        """
        Load TOF data for a specific run with spectral combination.
        
        Loads time-of-flight neutron imaging data for a single run,
        retrieving all TIFF files and combining TOF channels as specified.
        Uses multi-threaded loading for performance optimization.
        
        Args:
            run: Run identifier string for data location
            data_type: Type of data (sample, open beam, dark current)
            
        Returns:
            Combined TOF data array for the specified run
            
        Notes:
            - Uses multi-threaded loading with TOF combination
            - Retrieves full path from run metadata
            - Combines spectral channels during loading process
            - Returns 2D or 3D array depending on combination settings
        """

        full_path_to_run: str = self.parent.list_of_runs[data_type][run][Run.full_path]

        # get list of tiff
        list_tif: List[str] = retrieve_list_of_tif(full_path_to_run)

        # load data
        # data = load_list_of_tif(list_tif)
        data: NDArray[np.floating] = load_data_using_multithreading(list_tif, combine_tof=True)

        return data
    
    # def combine_tof_data_range(self, master_3d_data_array):

    #     # tof mode
    #     logging.info(f"combining TOF ...")
    #     for _data_type in master_3d_data_array.keys():
    #         if _data_type not in [DataType.sample, DataType.ob]:
    #             logging.info(f"skipping {_data_type} data type")
    #             continue
    #         logging.info(f"combining data for {_data_type} ...")
    #         logging.info(f"\tdata shape before combining: {master_3d_data_array[_data_type].shape}")
    #         master_3d_data_array[_data_type] = np.mean(master_3d_data_array[_data_type], axis=0)
    #         logging.info(f"\tdata shape after combining: {master_3d_data_array[_data_type].shape}")
    #     print(f"done!")

# def combine_tof_data_range(config_model, master_data):
    
#     operating_mode = config_model.operating_mode
#     if operating_mode == OperatingMode.white_beam:
#         logging.info(f"white mode, all TOF data have already been combined!")
#         return master_data
       
#     # tof mode
#     print(f"combining data in TOF ...", end="")
#     [left_tof_index, right_tof_index] = config_model.range_of_tof_to_combine[0]
#     logging.info(f"combining TOF from index {left_tof_index} to index {right_tof_index}")
#     for _data_type in master_data.keys():
#         _new_master_data = []
#         for _data in master_data[_data_type]:
#             _new_master_data.append(np.mean(_data[left_tof_index: right_tof_index+1, :, :], axis=0))
#         master_data[_data_type] = _new_master_data
#     print(f"done!")

#     return master_data
