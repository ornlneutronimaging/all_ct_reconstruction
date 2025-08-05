"""
SVMBIR Neutron CT Reconstruction in Time-of-Flight (TOF) Mode.

This module implements the complete neutron computed tomography reconstruction
pipeline specifically for time-of-flight mode using the SVMBIR (Sparse View
Model Based Iterative Reconstruction) algorithm. It provides a comprehensive
workflow from data loading through final slice export.

Key Classes:
    - SvmbirReconstruction: Main reconstruction workflow class for TOF mode

Key Features:
    - Time-of-flight neutron imaging data processing
    - Multi-run data loading and validation with proton charge analysis
    - Comprehensive data cleaning and normalization workflows
    - Center of rotation and tilt angle determination
    - Advanced strip removal and outlier correction algorithms
    - SVMBIR iterative reconstruction with customizable parameters
    - Interactive visualization and quality control interfaces
    - Comprehensive export capabilities for reconstructed slices

TOF Mode Capabilities:
    - Multi-wavelength neutron imaging data handling
    - Spectral analysis and wavelength-dependent reconstruction
    - 3D data array management for angle-wavelength-position data
    - TOF range selection and combination workflows
    - Proton charge normalization for beam intensity variations

Reconstruction Pipeline:
    1. Data loading and validation (sample, open beam, nexus files)
    2. Data quality checking and proton charge analysis
    3. Mode selection and TOF range configuration
    4. Image cleaning (outlier removal, low/high pixel correction)
    5. Normalization using open beam corrections
    6. Chip-specific corrections for detector artifacts
    7. Strip removal using advanced filtering algorithms
    8. Center of rotation and tilt angle determination
    9. Final projection review and run exclusion
    10. SVMBIR iterative reconstruction
    11. Slice export and extra file generation

Dependencies:
    - workflow modules: Complete reconstruction pipeline components
    - utilities: Logging, configuration, and file handling
    - svmbir: Sparse view model-based iterative reconstruction

Author: Neutron Imaging Team
Created: Part of VENUS neutron CT reconstruction pipeline
"""

import os
import logging
from collections import OrderedDict
from typing import Optional, Dict, List, Any, Union
import numpy as np
from numpy.typing import NDArray

from __code import DataType, OperatingMode, DEFAULT_OPERATING_MODE
from __code.utilities.logging import setup_logging
from __code.utilities.configuration_file import Configuration

from __code.workflow.load import Load
from __code.workflow.checking_data import CheckingData
from __code.workflow.recap_data import RecapData
from __code.workflow.mode_selection import ModeSelection
from __code.workflow.images_cleaner import ImagesCleaner
from __code.workflow.normalization import Normalization
from __code.workflow.chips_correction import ChipsCorrection
from __code.workflow.center_of_rotation_and_tilt import CenterOfRotationAndTilt
from __code.workflow.remove_strips import RemoveStrips
from __code.workflow.svmbir_handler import SvmbirHandler
from __code.workflow.final_projections_review import FinalProjectionsReview
from __code.workflow.export import ExportExtra
from __code.workflow.visualization import Visualization

LOG_BASENAME_FILENAME = "svmbir_reconstruction_tof_mode"


class SvmbirReconstruction:
    """
    Complete SVMBIR neutron CT reconstruction workflow for time-of-flight mode.
    
    This class implements the full neutron computed tomography reconstruction
    pipeline specifically optimized for time-of-flight neutron imaging data.
    It manages the entire workflow from raw data loading through final
    reconstructed slice export using the SVMBIR algorithm.
    
    The reconstruction process handles:
    - Multi-run neutron imaging datasets with proton charge normalization
    - Time-of-flight spectral data with wavelength-dependent processing
    - Advanced data cleaning and normalization procedures
    - Geometric corrections including center of rotation and tilt
    - Iterative reconstruction using sparse view model-based methods
    - Comprehensive quality control and visualization tools
    
    Attributes:
        MODE: Operating mode set to OperatingMode.tof
        working_dir: Dictionary of working directory paths for different data types
        operating_mode: Current operating mode (default from configuration)
        image_size: Dictionary containing image dimensions (height, width)
        spectra_file_full_path: Path to neutron wavelength spectra file
        final_dict_of_pc: Final proton charge values for normalization
        final_dict_of_frame_number: Frame number mapping for data organization
        data_3d_of_all_projections_merged: 3D array of merged projection data
        list_of_runs: Organized run information for sample and open beam data
        list_of_runs_checking_data: Data validation results for each run
        list_proton_charge_c: Proton charge values for beam normalization
        final_list_of_runs: Final validated run lists after quality control
        final_list_of_angles: Angular positions for reconstruction geometry
        master_3d_data_array: Primary 3D data arrays for sample and open beam
        master_tof_3d_data_array: TOF-specific 3D data organization
        normalized_images: Normalized projection images after open beam correction
        corrected_images: Images after detector-specific corrections
        strip_corrected_images: Images after strip artifact removal
        reconstruction_array: Final 3D reconstructed volume
    """

    MODE: OperatingMode = OperatingMode.tof

    working_dir: Dict[DataType, str] = {
        DataType.sample: "",
        DataType.ob: "",
        DataType.nexus: "",
        DataType.cleaned_images: "",
        DataType.normalized: "",
        DataType.processed: "",
        }
    
    operating_mode: OperatingMode = DEFAULT_OPERATING_MODE

    image_size: Dict[str, Optional[int]] = {'height': None,
                                           'width': None}

    spectra_file_full_path: Optional[str] = None

    final_dict_of_pc: Dict[Any, Any] = {}
    final_dict_of_frame_number: Dict[Any, Any] = {}

    # used to displya profile vs lambda in TOF mode
    # np.shape of y, x, tof
    data_3d_of_all_projections_merged: Optional[NDArray] = None

    # will record short_run_number and pc
    # will look like
    # {DataType.sample: {'Run_1234': {Run.full_path: "/SNS/VENUS/.../Run_1344",
    #                                 Run.proton_charge: 5.01,
    #                                 Ru.use_it: True,
    #                                },
    #                    ...,
    #                   },
    # DataType.ob: {...},
    # }
    list_of_runs: Dict[DataType, OrderedDict] = {DataType.sample: OrderedDict(),
                                                 DataType.ob: OrderedDict(),
                                                 }
    
    list_of_runs_checking_data: Dict[DataType, Dict] = {DataType.sample: {},
                                                        DataType.ob: {},
                                                        }

    list_proton_charge_c: Dict[DataType, Dict] = {DataType.sample: {},
                                                  DataType.ob: {},
                                                  }

    final_list_of_runs: Dict[DataType, Dict] = {DataType.sample: {},
                                               DataType.ob: {},
                                               }

    final_list_of_angles: Optional[List[float]] = None
    list_of_runs_to_use: Optional[List] = None
    
    # set up in the checking_data. True if at least one of the run doesn't have this metadata in the NeXus
    at_least_one_frame_number_not_found: bool = False

    # dictionary used just after loading the data, not knowing the mode yet
    master_3d_data_array: Dict[DataType, Optional[NDArray]] = {DataType.sample: None,  # [angle, y, x]
                                                               DataType.ob: None}
    
    # each element of the dictionary is an master_3d_data_array of each TOF range
    # {'0': {'use_it': True,
    #         'data': master_3d_data_array, 
    #       },
    # '1': {'use_it': False,
    #       'data': master_3d_data_array,
    #       },
    #  ...
    #}
    # this is the master dictionary used no matter the mode
    master_tof_3d_data_array: Optional[Dict] = None

    master_3d_data_array_cleaned: Dict[DataType, Optional[NDArray]] = {DataType.sample: None,  # [angle, y, x]
                                                                       DataType.ob: None}

    normalized_images: Optional[NDArray] = None   # after normalization
    corrected_images: Optional[NDArray] = None  # after chips correction

    instrument: str = "VENUS"

    selection_of_pc: Optional[Any] = None   # plot that allows the user to select the pc for sample and ob and threshold

    list_of_sample_runs_to_reject_ui: Optional[Any] = None
    list_of_ob_runs_to_reject_ui: Optional[Any] = None
    minimum_requirements_met: bool = False

    # created during the combine step to match data index with run number (for normalization)
    list_of_runs_to_use: Dict[DataType, List] = {DataType.sample: [],
                                                 DataType.ob:[]}
    
    list_of_angles_to_use_sorted: Optional[List[float]] = None

    strip_corrected_images: Optional[NDArray] = None # Array 3D after strip correction

    # center of rotation
    o_center_and_tilt: Optional[Any] = None
    # remove strips
    o_remove: Optional[Any] = None
    # remove outliers
    o_clean: Optional[Any] = None
    # normalization
    o_norm: Optional[Any] = None
    # svmbir 
    o_svmbir: Optional[Any] = None
    # tof mode
    o_tof_range_mode: Optional[Any] = None

    # widget multi selection - list of runs to exclude before running svmbir
    runs_to_exclude_ui: Optional[Any] = None

    # reconstructed 3D array with svmbir
    reconstruction_array: Optional[NDArray] = None

    def __init__(self, system: Optional[Any] = None) -> None:
        """
        Initialize the SVMBIR reconstruction workflow for TOF mode.
        
        Sets up the complete reconstruction environment including working
        directories, instrument configuration, and logging. Configures
        paths for different data types (IPTS, nexus, processed data) based
        on the system configuration.
        
        Args:
            system: System configuration object containing working directory
                   and instrument selection. If None, defaults will be used.
        
        Side Effects:
            - Initializes configuration management
            - Sets up working directory structure for all data types
            - Configures logging for the reconstruction workflow
            - Logs initialization parameters and directory structure
        """

        self.configuration = Configuration()

        # o_init = Initialization(parent=self)
        # o_init.configuration()

        top_sample_dir = system.System.get_working_dir()
        self.instrument = system.System.get_instrument_selected()

        setup_logging(basename_of_log_file=LOG_BASENAME_FILENAME)        
        self.working_dir[DataType.ipts] = os.path.basename(top_sample_dir)
        self.working_dir[DataType.top] = os.path.join(top_sample_dir, "shared", "autoreduce", "mcp")
        self.working_dir[DataType.nexus] = os.path.join(top_sample_dir, "nexus")
        self.working_dir[DataType.processed] = os.path.join(top_sample_dir, "shared", "processed_data")
        logging.info(f"working_dir: {self.working_dir}")
        logging.info(f"instrument: {self.instrument}")

    # Selection of data
    def select_top_sample_folder(self) -> None:
        """
        Launch interactive folder selection for sample data.
        
        Opens a file browser interface to allow user selection of the
        top-level folder containing sample neutron imaging data runs.
        This folder should contain multiple run directories with NeXus files.
        
        Side Effects:
            - Creates Load workflow object for sample data selection
            - Launches interactive folder browser widget
            - Updates working_dir[DataType.sample] upon selection
        """
        o_load = Load(parent=self)
        o_load.select_folder(data_type=DataType.sample)

    def select_top_ob_folder(self) -> None:
        """
        Launch interactive folder selection for open beam data.
        
        Opens a file browser interface to allow user selection of the
        top-level folder containing open beam (background) neutron imaging
        data runs. Open beam data is essential for normalization corrections.
        
        Side Effects:
            - Creates Load workflow object for open beam data selection
            - Launches interactive folder browser widget
            - Updates working_dir[DataType.ob] upon selection
        """
        o_load = Load(parent=self)
        o_load.select_folder(data_type=DataType.ob)

    # Checking data (proton charge, empty runs ...)
    def checking_data(self) -> None:
        """
        Validate and check loaded neutron imaging data quality.
        
        Performs comprehensive data validation including proton charge
        analysis, empty run detection, metadata verification, and data
        integrity checks. Essential for ensuring reconstruction quality.
        
        Validation checks include:
        - Proton charge consistency across runs
        - Frame number metadata availability
        - NeXus file integrity and completeness
        - Run data availability and accessibility
        
        Side Effects:
            - Creates CheckingData workflow object
            - Populates list_of_runs_checking_data with validation results
            - Sets at_least_one_frame_number_not_found flag if metadata missing
            - Logs validation results and any detected issues
        
        Raises:
            ValueError: If input folders are invalid or inaccessible
        """
        try:
            o_checking = CheckingData(parent=self)
            o_checking.run()
        except ValueError:
            logging.info("Check the input folders provided !")

    def recap_data(self) -> None:
        """
        Generate and display comprehensive data summary.
        
        Creates a detailed recap of all loaded data including run counts,
        proton charge statistics, data quality metrics, and validation
        results. Provides overview before proceeding with reconstruction.
        
        Side Effects:
            - Creates RecapData workflow object
            - Displays interactive data summary interface
            - Shows run statistics and quality metrics
        """
        o_recap = RecapData(parent=self)
        o_recap.run()

    def checkin_data_entries(self) -> None:
        """
        Verify minimum data requirements for reconstruction.
        
        Checks that sufficient sample and open beam data are available
        to proceed with reconstruction. Validates that minimum requirements
        for proton charge, run counts, and data quality are met.
        
        Side Effects:
            - Creates CheckingData workflow object for requirement validation
            - Sets minimum_requirements_met flag based on validation results
            - Logs requirement check results
        """
        o_check = CheckingData(parent=self)
        o_check.checking_minimum_requirements()

    # mode selection
    def mode_selection(self) -> None:
        """
        Configure operating mode and TOF range selection.
        
        Sets up the time-of-flight mode selection interface allowing users
        to configure wavelength ranges and spectral parameters for TOF
        neutron imaging reconstruction. Essential for proper wavelength-
        dependent data processing.
        
        Side Effects:
            - Creates ModeSelection workflow object
            - Stores reference in self.o_mode for subsequent operations
            - Launches interactive mode selection interface
            - Configures TOF range parameters
        """
        self.o_mode = ModeSelection(parent=self)
        self.o_mode.select()

    # load data
    def load_data(self) -> None:
        """
        Load neutron imaging data based on selected mode and parameters.
        
        Executes the actual data loading process using the configured
        mode selection parameters. Handles multi-run data loading with
        proper memory management for large TOF datasets.
        
        Side Effects:
            - Loads data using self.o_mode.load() method
            - Populates master_3d_data_array with loaded imaging data
            - Updates data organization structures for TOF processing
        """
        self.o_mode.load()
        
    def select_tof_ranges(self) -> None:
        """
        Configure time-of-flight wavelength ranges for reconstruction.
        
        Provides interface for selecting specific TOF ranges or wavelength
        bins for reconstruction. In white beam mode, this step is skipped.
        Supports both single and multi-range TOF configurations.
        
        Side Effects:
            - Returns early if operating_mode is white_beam
            - Uses self.o_tof_range_mode for TOF range configuration
            - Launches TOF range selection interface
        """
        if self.operating_mode == OperatingMode.white_beam:
            return

        # self.o_tof_range_mode.select_multi_tof_range()
        self.o_tof_range_mode.select_tof_range()

    def combine_tof_mode_data(self) -> None:
        """
        Combine and organize multi-range TOF data for reconstruction.
        
        Merges data from multiple TOF ranges into unified data structures
        suitable for reconstruction processing. Handles wavelength-dependent
        data organization and memory management.
        
        Side Effects:
            - Combines TOF range data using self.o_tof_range_mode
            - Updates master_tof_3d_data_array with combined data
            - Organizes data for subsequent processing steps
        """
        if self.o_tof_range_mode:
            self.o_tof_range_mode.combine_tof_mode_data()

    # cleaning low/high pixels
    def clean_images_settings(self) -> None:
        """
        Configure image cleaning parameters for outlier removal.
        
        Sets up the interface for configuring outlier detection and removal
        parameters. Allows users to define thresholds for low and high
        pixel value outliers that may result from detector artifacts.
        
        Side Effects:
            - Creates ImagesCleaner workflow object
            - Stores reference in self.o_clean for subsequent operations
            - Launches parameter configuration interface
        """
        self.o_clean = ImagesCleaner(parent=self)
        self.o_clean.settings()

    def clean_images_setup(self) -> None:
        """
        Prepare image cleaning algorithms and parameters.
        
        Finalizes the setup of image cleaning algorithms based on user-
        configured parameters. Prepares the cleaning workflow for execution
        on the loaded imaging data.
        
        Side Effects:
            - Completes cleaning algorithm setup using self.o_clean
            - Prepares cleaning parameters for data processing
        """
        self.o_clean.cleaning_setup()

    def clean_images(self) -> None:
        """
        Execute image cleaning and outlier removal.
        
        Applies configured cleaning algorithms to remove outlier pixels
        and artifacts from the loaded imaging data. Updates the cleaned
        data arrays for subsequent processing steps.
        
        Side Effects:
            - Executes cleaning algorithms using self.o_clean
            - Updates master_3d_data_array_cleaned with processed data
            - Logs cleaning progress and results
        """
        self.o_clean.cleaning()

    def visualization_cleaning_settings(self) -> None:
        """
        Configure visualization parameters for cleaning results.
        
        Sets up visualization interface for reviewing image cleaning
        results. Allows users to compare before and after cleaning
        to validate the cleaning process effectiveness.
        
        Side Effects:
            - Creates Visualization workflow object
            - Stores reference in self.o_vizu for visualization operations
            - Configures visualization parameters
        """
        self.o_vizu = Visualization(parent=self)
        self.o_vizu.settings()

    def visualization_cleaning(self) -> None:
        """
        Display before/after comparison of image cleaning results.
        
        Generates interactive visualization comparing raw and cleaned
        imaging data to validate the effectiveness of the cleaning
        process. Essential for quality control.
        
        Side Effects:
            - Displays interactive before/after comparison
            - Shows raw vs cleaned data with configurable display parameters
            - Allows user validation of cleaning effectiveness
        """
        self.o_vizu.visualize(data_after=self.master_3d_data_array_cleaned[DataType.sample],
                              label_after='cleaned',
                              label_before='raw',
                              data_before=self.master_3d_data_array[DataType.sample],
                              turn_on_vrange=False)

    def select_export_folder(self) -> None:
        """
        Select output folder for cleaned image export.
        
        Opens folder selection interface for choosing where to export
        the cleaned imaging data. Useful for saving intermediate results
        or sharing cleaned data with collaborators.
        
        Side Effects:
            - Launches folder selection interface using self.o_clean
            - Updates export path configuration
        """
        self.o_clean.select_export_folder()

    def export_cleaned_images(self) -> None:
        """
        Export cleaned images to selected folder.
        
        Saves the cleaned imaging data to the user-selected folder in
        appropriate file formats. Preserves metadata and data organization
        for future use or external processing.
        
        Side Effects:
            - Exports cleaned images using self.o_clean
            - Creates output files in selected export folder
            - Logs export progress and completion
        """
        self.o_clean.export_clean_images()

    # normalization
    def normalization_settings(self) -> None:
        """
        Configure normalization parameters using open beam data.
        
        Sets up the normalization workflow parameters for correcting
        beam intensity variations using open beam measurements.
        Essential for quantitative neutron imaging reconstruction.
        
        Side Effects:
            - Creates Normalization workflow object
            - Stores reference in self.o_norm for normalization operations
            - Launches normalization parameter configuration interface
        """
        self.o_norm = Normalization(parent=self)
        self.o_norm.normalization_settings()

    def normalization_select_roi(self) -> None:
        """
        Select region of interest for normalization calculations.
        
        Provides interface for selecting specific regions of the detector
        for normalization calculations. This ROI selection helps avoid
        artifacts and ensures proper beam intensity correction.
        
        Side Effects:
            - Launches ROI selection interface using self.o_norm
            - Updates normalization ROI parameters
        """
        self.o_norm.select_roi()

    def normalization(self) -> None:
        """
        Execute normalization using open beam corrections.
        
        Applies open beam normalization to correct for beam intensity
        variations and detector response non-uniformities. Critical
        step for quantitative reconstruction accuracy.
        
        Side Effects:
            - Executes normalization algorithms using self.o_norm
            - Updates normalized_images with corrected data
            - Logs normalization progress and statistics
        """
        self.o_norm.run()

    def visualization_normalization_settings(self) -> None:
        """
        Configure visualization parameters for normalization results.
        
        Sets up visualization interface for reviewing normalization
        results and comparing before/after normalization effects.
        
        Side Effects:
            - Creates Visualization workflow object for normalization review
            - Configures visualization parameters for normalization data
        """
        self.o_vizu = Visualization(parent=self)
        self.o_vizu.settings()

        # self.o_norm.visualization_normalization_settings()

    def visualize_normalization(self) -> None:
        """
        Display before/after comparison of normalization results.
        
        Generates interactive visualization comparing cleaned and
        normalized imaging data to validate normalization effectiveness.
        Includes dynamic range controls for optimal visualization.
        
        Side Effects:
            - Displays interactive before/after normalization comparison
            - Shows cleaned vs normalized data with range controls
            - Allows user validation of normalization quality
        """
        self.o_vizu.visualize(data_after=self.normalized_images,
                              label_before='cleaned',
                              label_after='normalized',
                              data_before=self.master_3d_data_array_cleaned[DataType.sample],
                              turn_on_vrange=True)
        # self.o_norm.visualize_normalization()

    def select_export_normalized_folder(self) -> None:
        """
        Select output folder for normalized image export.
        
        Opens folder selection interface for choosing where to export
        the normalized imaging data. Useful for saving processed data
        at intermediate reconstruction stages.
        
        Side Effects:
            - Creates Load workflow object for folder selection
            - Launches folder browser for normalized data export path
        """
        o_select = Load(parent=self)
        o_select.select_folder(data_type=DataType.normalized)

    def export_normalized_images(self) -> None:
        """
        Export normalized images to selected folder.
        
        Saves the normalized imaging data to the user-selected folder
        with appropriate file formats and metadata preservation.
        
        Side Effects:
            - Exports normalized images using self.o_norm
            - Creates output files in selected export folder
            - Logs export progress and completion
        """
        self.o_norm.export_images()

    # chips correction
    def chips_correction(self) -> None:
        """
        Apply detector chip-specific corrections.
        
        Corrects for detector-specific artifacts and non-uniformities
        that may be present in multi-chip detector systems. Addresses
        gain variations and response differences between detector elements.
        
        Side Effects:
            - Creates ChipsCorrection workflow object
            - Applies chip-specific corrections to normalized data
            - Updates corrected_images with processed data
            - Logs correction progress and statistics
        """
        o_chips = ChipsCorrection(parent=self)
        o_chips.run()

    def visualize_chips_correction(self) -> None:
        """
        Display visualization of chip correction results.
        
        Shows the effects of chip-specific corrections on the imaging
        data, helping users validate the correction effectiveness.
        
        Side Effects:
            - Creates ChipsCorrection workflow object for visualization
            - Displays chip correction comparison interface
        """
        o_chips = ChipsCorrection(parent=self)
        o_chips.visualize_chips_correction()

    # strips removal
    def select_remove_strips_algorithms(self) -> None:
        """
        Configure algorithms for strip artifact removal.
        
        Sets up the interface for selecting and configuring algorithms
        to remove strip artifacts that commonly appear in neutron imaging
        data due to detector or beamline characteristics.
        
        Side Effects:
            - Creates RemoveStrips workflow object
            - Stores reference in self.o_remove for strip removal operations
            - Launches algorithm selection interface
        """
        self.o_remove = RemoveStrips(parent=self)
        self.o_remove.select_algorithms()

    def define_settings(self) -> None:
        """
        Configure strip removal algorithm parameters.
        
        Provides interface for fine-tuning the parameters of selected
        strip removal algorithms based on the specific characteristics
        of the imaging data and detected artifacts.
        
        Side Effects:
            - Launches parameter configuration interface using self.o_remove
            - Updates strip removal algorithm settings
        """
        self.o_remove.define_settings()

    def remove_strips_and_display(self) -> None:
        """
        Execute strip removal and display results.
        
        Applies configured strip removal algorithms to the corrected
        imaging data and displays the results for user validation.
        Updates the strip_corrected_images for subsequent processing.
        
        Side Effects:
            - Executes strip removal algorithms using self.o_remove
            - Updates strip_corrected_images with processed data
            - Displays strip removal results for validation
            - Logs processing progress and effectiveness metrics
        """
        self.o_remove.run()

    # calculate center of rotation & tilt
    def select_sample_roi(self) -> None:
        """
        Select region of interest for center of rotation calculation.
        
        Provides interface for selecting a region of the sample data
        to use for center of rotation and tilt angle determination.
        Uses strip-corrected images if available, otherwise falls back
        to chip-corrected images.
        
        Side Effects:
            - Uses strip_corrected_images if available, else corrected_images
            - Creates CenterOfRotationAndTilt workflow object
            - Stores reference in self.o_center_and_tilt
            - Launches ROI selection interface for geometry calculation
        """
        if self.strip_corrected_images is None:
            # if the remove filter hasn't been ran
            self.strip_corrected_images = self.corrected_images

        self.o_center_and_tilt = CenterOfRotationAndTilt(parent=self)
        self.o_center_and_tilt.select_range()

    def calculate_center_of_rotation_and_tilt(self) -> None:
        """
        Calculate center of rotation and tilt angle for reconstruction.
        
        Executes algorithms to determine the center of rotation and tilt
        angle parameters essential for accurate CT reconstruction geometry.
        These parameters correct for sample positioning and rotation axis
        alignment.
        
        Side Effects:
            - Calculates center of rotation using self.o_center_and_tilt
            - Determines tilt angle compensation parameters
            - Updates reconstruction geometry parameters
            - Logs calculated values and geometry corrections
        """
        self.o_center_and_tilt.run()

    # last chance to reject runs
    def final_projections_review(self) -> None:
        """
        Final quality review and run exclusion interface.
        
        Provides a final opportunity to review all processed projections
        and exclude any problematic runs before proceeding with SVMBIR
        reconstruction. Essential quality control step.
        
        Side Effects:
            - Creates FinalProjectionsReview workflow object
            - Displays corrected_images for final review
            - Launches run exclusion interface
            - Updates runs_to_exclude_ui with user selections
        """
        o_review = FinalProjectionsReview(parent=self)
        o_review.run(array=self.corrected_images)
        o_review.list_runs_to_reject()

    # run svmbir
    def svmbir_settings(self) -> None:
        """
        Configure SVMBIR reconstruction parameters.
        
        Sets up the interface for configuring Sparse View Model Based
        Iterative Reconstruction (SVMBIR) parameters including iteration
        counts, regularization parameters, and convergence criteria.
        
        Side Effects:
            - Creates SvmbirHandler workflow object
            - Stores reference in self.o_svmbir for reconstruction operations
            - Launches SVMBIR parameter configuration interface
        """
        self.o_svmbir = SvmbirHandler(parent=self)
        self.o_svmbir.set_settings()

    def svmbir_display_sinograms(self) -> None:
        """
        Display sinograms for reconstruction preview.
        
        Shows the sinogram data that will be used for SVMBIR reconstruction,
        allowing users to verify data quality and reconstruction parameters
        before executing the computationally intensive reconstruction.
        
        Side Effects:
            - Displays sinogram visualization using self.o_svmbir
            - Shows projection data organized for reconstruction
        """
        self.o_svmbir.display_sinograms()

    def svmbir_run(self) -> None:
        """
        Execute SVMBIR reconstruction and display results.
        
        Runs the complete SVMBIR iterative reconstruction algorithm
        using configured parameters and displays the resulting reconstructed
        slices. This is the main reconstruction computation step.
        
        Side Effects:
            - Executes SVMBIR reconstruction using self.o_svmbir
            - Updates reconstruction_array with 3D reconstructed volume
            - Displays reconstructed slices for immediate review
            - Logs reconstruction progress and completion statistics
        """
        self.o_svmbir.run_reconstruction()
        self.o_svmbir.display_slices()

    # def display_slices(self):
    #     self.o_svmbir.display_slices()

    # export slices
    def select_export_slices_folder(self) -> None:
        """
        Select output folder for reconstructed slice export.
        
        Opens folder selection interface for choosing where to export
        the final reconstructed CT slices. These are the primary output
        of the reconstruction process.
        
        Side Effects:
            - Creates Load workflow object for folder selection
            - Launches folder browser for reconstructed slice export path
        """
        o_select = Load(parent=self)
        o_select.select_folder(data_type=DataType.reconstructed)

    def export_slices(self) -> None:
        """
        Export reconstructed slices to selected folder.
        
        Saves the final reconstructed CT slices to the user-selected
        folder with appropriate file formats and metadata. These are
        the primary deliverable of the reconstruction workflow.
        
        Side Effects:
            - Exports reconstructed slices using self.o_svmbir
            - Creates output files in selected export folder
            - Preserves reconstruction metadata and parameters
            - Logs export progress and completion
        """
        self.o_svmbir.export_images()

    # export extra files
    def select_export_extra_files(self) -> None:
        """
        Select output folder for additional reconstruction files.
        
        Opens folder selection interface for choosing where to export
        additional files such as configuration files, logs, and
        reconstruction metadata.
        
        Side Effects:
            - Creates Load workflow object for folder selection
            - Launches folder browser for extra files export path
        """
        o_select = Load(parent=self)
        o_select.select_folder(data_type=DataType.extra)

    def export_extra_files(self) -> None:
        """
        Export additional reconstruction files and metadata.
        
        Saves configuration files, processing logs, reconstruction
        parameters, and other metadata to the selected folder for
        documentation and reproducibility.
        
        Side Effects:
            - Creates ExportExtra workflow object
            - Exports configuration files, logs, and metadata
            - Uses LOG_BASENAME_FILENAME for log file organization
            - Creates comprehensive reconstruction documentation
        """
        o_export = ExportExtra(parent=self)
        o_export.run(base_log_file_name=LOG_BASENAME_FILENAME)
