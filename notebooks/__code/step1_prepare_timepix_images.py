"""
TimePix Detector Neutron CT Reconstruction Preparation Module.

This module implements the complete neutron computed tomography reconstruction
preparation workflow specifically for TimePix detector systems. It provides
comprehensive data processing from raw TimePix data loading through final
reconstruction preparation, optimized for time-of-flight neutron imaging.

Key Classes:
    - Step1PrepareTimePixImages: Main preparation workflow class for TimePix detectors

Key Features:
    - TimePix detector-specific data loading and processing
    - Comprehensive time-of-flight (TOF) mode support
    - Multi-run data validation and quality control
    - Advanced image cleaning and outlier removal
    - Open beam normalization with detector corrections
    - Geometric corrections (rotation, tilt, center of rotation)
    - Strip artifact removal with multiple algorithms
    - Data rebinning and cropping capabilities
    - Log conversion for enhanced reconstruction quality
    - Support for both SVMBIR and FBP reconstruction methods

TimePix-Specific Capabilities:
    - High-resolution neutron imaging data handling
    - Pixel-level outlier detection and correction
    - Detector chip-specific corrections
    - Time-resolved neutron detection processing
    - Angular position management for CT geometry
    - Proton charge normalization for beam variations

Reconstruction Preparation Pipeline:
    1. Data loading and validation (sample, open beam, dark current)
    2. Multi-run data quality checking and organization
    3. Image combination and TOF processing
    4. Pre-processing cropping and cleaning
    5. Normalization using open beam and dark current
    6. Detector chip corrections and outlier removal
    7. Data rebinning for optimized reconstruction
    8. Geometric corrections (rotation, tilt compensation)
    9. Strip artifact removal using advanced algorithms
    10. Center of rotation determination
    11. Final data preparation for reconstruction algorithms
    12. Export of processed data and reconstruction parameters

Dependencies:
    - workflow modules: Complete reconstruction preparation pipeline
    - utilities: Logging, configuration, and file handling
    - numpy: Numerical computing for image processing
    - matplotlib: Visualization and quality control

Author: Neutron Imaging Team
Created: Part of Step 1 preparation workflow for TimePix-based neutron CT
"""

import os
import logging
import ipywidgets as widgets
from collections import OrderedDict
from typing import Optional, Dict, List, Any, Union
import numpy as np
from numpy.typing import NDArray
from IPython.display import display
from IPython.display import HTML

from __code import DataType, OperatingMode, DEFAULT_OPERATING_MODE, DetectorType
from __code.utilities.logging import setup_logging
from __code.utilities.configuration_file import Configuration
from __code.config import DEBUG, default_detector_type

from __code.workflow.load import Load
from __code.workflow.checking_data import CheckingData
from __code.workflow.recap_data import RecapData
from __code.workflow.combine_tof import CombineTof
from __code.workflow.images_cleaner import ImagesCleaner
from __code.workflow.normalization import Normalization
from __code.workflow.chips_correction import ChipsCorrection
from __code.workflow.center_of_rotation_and_tilt import CenterOfRotationAndTilt
from __code.workflow.remove_strips import RemoveStrips
from __code.workflow.svmbir_handler import SvmbirHandler
from __code.workflow.final_projections_review import FinalProjectionsReview
from __code.workflow.export import ExportExtra
from __code.workflow.visualization import Visualization
from __code.workflow.crop import Crop
from __code.workflow.combine_ob_dc import CombineObDc
from __code.workflow.mode_selection import ModeSelection
from __code.workflow.reconstruction_selection import ReconstructionSelection
from __code.workflow.rebin import Rebin
from __code.workflow.log_conversion import log_conversion
from __code.workflow.data_handler import remove_negative_values, remove_0_values
from __code.workflow.fbp_handler import FbpHandler
from __code.workflow.rotate import Rotate
from __code.workflow.test_reconstruction import TestReconstruction
from __code.utilities.configuration_file import ReconstructionAlgorithm
from __code.utilities.logging import logging_3d_array_infos


LOG_BASENAME_FILENAME, _ = os.path.splitext(os.path.basename(__file__))


class Step1PrepareTimePixImages:
    """
    Complete preparation workflow for TimePix detector neutron CT reconstruction.
    
    This class implements the comprehensive data preparation pipeline specifically
    optimized for TimePix detector systems used in neutron computed tomography.
    It handles the complete workflow from raw data loading through final
    reconstruction preparation, including all necessary corrections and processing.
    
    The preparation process manages:
    - TimePix detector-specific data loading and organization
    - Multi-run neutron imaging datasets with angular positioning
    - Time-of-flight spectral data processing and combination
    - Advanced image cleaning with pixel-level outlier detection
    - Open beam and dark current normalization procedures
    - Geometric corrections including center of rotation and tilt
    - Strip artifact removal using multiple algorithms
    - Data optimization through rebinning and cropping
    - Log conversion for enhanced reconstruction quality
    
    Attributes:
        MODE: Operating mode set to OperatingMode.tof for time-of-flight
        working_dir: Dictionary of working directory paths for different data types
        list_angles_deg_vs_runs_dict: Mapping of angles to run identifiers
        list_angles_of_data_loaded_deg: Final sorted list of angles for reconstruction
        image_size: Dictionary containing image dimensions (height, width)
        list_of_runs: Organized run information for sample and open beam data
        list_of_images: Image data organized by data type (sample, ob, dc)
        master_3d_data_array: Primary 3D data arrays for sample and open beam
        normalized_images: Normalized projection images after corrections
        corrected_images: Images after detector-specific corrections
        strip_corrected_images: Images after strip artifact removal
        reconstruction_array: Final prepared data for reconstruction algorithms
        center_of_rotation: Calculated center of rotation for geometry correction
        configuration: Configuration management for reconstruction parameters
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

    detector_type = DetectorType.tpx1_legacy

    # {100.000: 'run_1234', 101.000: 'run_1235', ...}
    list_angles_deg_vs_runs_dict: Dict[float, str] = {}

    # final list of angles used and sorted (to be used in reconstruction)
    list_angles_of_data_loaded_deg: Optional[List[float]] = None

    image_size: Dict[str, Optional[int]] = {'height': None,
                                           'width': None}

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
    
    list_of_images: Dict[DataType, Optional[Union[OrderedDict, NDArray]]] = {
        DataType.sample: OrderedDict(),
        DataType.ob: None,
        DataType.dc: None,
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
    
    # set up in the checking_data. True if at least one of the run doesn't have this metadata in the NeXus
    at_least_one_frame_number_not_found: bool = False

    master_3d_data_array: Dict[DataType, Optional[NDArray]] = {DataType.sample: None,  # [angle, y, x]
                                                               DataType.ob: None}

    master_3d_data_array_cleaned: Dict[DataType, Optional[NDArray]] = {DataType.sample: None,  # [angle, y, x]
                                                                       DataType.ob: None}

    normalized_images: Optional[NDArray] = None   # after normalization
    corrected_images: Optional[NDArray] = None  # after chips correction
    before_rebinning: Optional[NDArray] = None
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
    center_of_rotation: Optional[float] = None  # center of rotation calculated by the user
    # remove strips
    o_remove: Optional[Any] = None
    # normalization
    o_norm: Optional[Any] = None
    # svmbir 
    o_svmbir: Optional[Any] = None

    # widget multi selection - list of runs to exclude before running svmbir
    runs_to_exclude_ui: Optional[Any] = None

    # reconstructed 3D array with svmbir
    reconstruction_array: Optional[NDArray] = None

    at_least_one_frame_number_not_found: bool = False
    at_lest_one_proton_charge_not_found: bool = False

    def __init__(self, system: Optional[Any] = None) -> None:
        """
        Initialize the TimePix preparation workflow.
        
        Sets up the complete TimePix detector preparation environment including
        working directories, instrument configuration, and logging. Configures
        paths for different data types based on system configuration.
        
        Args:
            system: System configuration object containing working directory
                   and instrument selection. If None, defaults will be used.
        
        Side Effects:
            - Initializes configuration management for reconstruction parameters
            - Sets up working directory structure for all data types
            - Configures logging for the preparation workflow
            - Logs initialization parameters and debug mode status
        """

        self.configuration = Configuration()

        top_sample_dir = system.System.get_working_dir()
        self.instrument = system.System.get_instrument_selected()

        setup_logging(basename_of_log_file=LOG_BASENAME_FILENAME)        
        self.working_dir[DataType.ipts] = os.path.basename(top_sample_dir)
        self.working_dir[DataType.sample] = os.path.join(top_sample_dir, "shared", "autoreduce", "mcp")
        self.working_dir[DataType.ob] = os.path.join(top_sample_dir, "shared", "autoreduce", "mcp")
        self.working_dir[DataType.top] = os.path.join(top_sample_dir, "shared", "autoreduce", "mcp")
        self.working_dir[DataType.nexus] = os.path.join(top_sample_dir, "nexus")
        self.working_dir[DataType.processed] = os.path.join(top_sample_dir, "shared", "processed_data")
        logging.info(f"working_dir: {self.working_dir}")
        logging.info(f"instrument: {self.instrument}")
        if DEBUG:
            logging.info(f"WARNING!!!! we are running using DEBUG mode!")
            _default_detector_type = default_detector_type
        else:
            _default_detector_type = DetectorType.tpx1

        display(HTML("<span style='color:blue; font-size:16px'>Select detector type</span>"))
        self.detector_type_widget = widgets.Dropdown(
            options=[DetectorType.tpx1_legacy, DetectorType.tpx1, DetectorType.tpx3],
            value=_default_detector_type,
            layout=widgets.Layout(width="400px"),
            disabled=False,
        )
        display(self.detector_type_widget)

    # Selection of data
    def select_top_sample_folder(self) -> None:
        """
        Launch interactive folder selection for sample data.
        
        Opens a file browser interface to allow user selection of the
        top-level folder containing sample neutron imaging data runs.
        Essential first step for TimePix data preparation workflow.
        
        Side Effects:
            - Creates Load workflow object for sample data selection
            - Launches interactive folder browser widget
            - Updates working_dir[DataType.sample] upon selection
        """
        self.detector_type = self.detector_type_widget.value
        
        o_load = Load(parent=self)
        o_load.select_folder(data_type=DataType.sample)

    def select_top_ob_folder(self) -> None:
        """
        Launch interactive folder selection for open beam data.
        
        Opens a file browser interface to allow user selection of the
        top-level folder containing open beam (background) neutron imaging
        data runs. Essential for normalization corrections in TimePix workflow.
        
        Side Effects:
            - Creates Load workflow object for open beam data selection
            - Launches interactive folder browser widget
            - Updates working_dir[DataType.ob] upon selection
        """
        o_load = Load(parent=self)
        o_load.select_folder(data_type=DataType.ob)

    # Checking data (proton charge, empty runs ...)
    def load_and_check_data(self) -> None:
        """
        Load and validate TimePix neutron imaging data quality.
        
        Performs comprehensive data loading and validation including proton
        charge analysis, empty run detection, metadata verification, and
        TimePix-specific data integrity checks. Essential for ensuring
        high-quality reconstruction preparation.
        
        Validation checks include:
        - TimePix detector data format validation
        - Proton charge consistency across runs
        - Frame number metadata availability
        - NeXus file integrity and completeness
        - Angular position data validation
        
        Side Effects:
            - Creates CheckingData workflow object
            - Populates list_of_runs_checking_data with validation results
            - Sets flags for missing metadata (frame numbers, proton charge)
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
        Generate and display comprehensive TimePix data summary.
        
        Creates a detailed recap of all loaded TimePix data including run
        counts, angular coverage, proton charge statistics, data quality
        metrics, and validation results. Provides overview before proceeding
        with preparation workflow.
        
        Side Effects:
            - Creates RecapData workflow object
            - Displays interactive data summary interface
            - Shows TimePix-specific metrics and quality indicators
        """
        o_recap = RecapData(parent=self)
        o_recap.run()

    # def checkin_data_entries(self):
    #     o_check = CheckingData(parent=self)
    #     o_check.checking_minimum_requirements()

    # combine images
    def combine_images(self) -> None:
        """
        Combine and organize TimePix images for reconstruction preparation.
        
        Validates minimum requirements and combines multi-run TimePix imaging
        data into organized 3D arrays suitable for CT reconstruction. Creates
        master data structures with proper angular organization and metadata.
        
        This method:
        - Checks minimum data requirements for reconstruction
        - Combines TOF data from multiple runs
        - Creates master_3d_data_array with organized imaging data
        - Establishes final_list_of_runs and final_list_of_angles
        - Organizes angular positions for proper CT geometry
        
        Side Effects:
            - Creates master_3d_data_array for sample and open beam data
            - Populates final_list_of_runs with validated run information
            - Establishes final_list_of_angles for reconstruction geometry
            - Sets minimum_requirements_met flag based on data validation
            - Displays requirement error if insufficient data available
        """
        o_check = CheckingData(parent=self)
        o_check.checking_minimum_requirements()
        if self.minimum_requirements_met:           
            o_combine = CombineTof(parent=self)
            o_combine.run()
        else:
            o_check.minimum_requirement_not_met()

    # visualization
    def how_to_visualize(self) -> None:
        """
        Configure visualization options for TimePix data review.
        
        Sets up the interface for configuring visualization parameters
        for TimePix imaging data. Provides options for different visualization
        modes and display parameters optimized for TimePix detector characteristics.
        
        Side Effects:
            - Creates Visualization workflow object
            - Stores reference in self.o_vizu for visualization operations
            - Launches visualization configuration interface
        """
        self.o_vizu = Visualization(parent=self)
        self.o_vizu.how_to_visualize()

    def visualize_raw_data(self) -> None:
        """
        Display raw TimePix imaging data for quality assessment.
        
        Visualizes the raw combined TimePix imaging data using the master
        3D data array. Essential for initial quality assessment and
        validation of data loading and combination procedures.
        
        Side Effects:
            - Uses master_3d_data_array for raw data visualization
            - Displays TimePix data according to configured visualization settings
            - Shows data in 'raw' mode for initial quality assessment
        """
        self.o_vizu.visualize_timepix_according_to_selection(mode='raw')

    # pre processing crop
    def pre_processing_crop_settings(self) -> None:
        """
        Configure pre-processing crop parameters for TimePix data.
        
        Sets up the interface for defining crop regions before normalization.
        This pre-processing crop helps reduce data volume and focus on
        regions of interest for TimePix detector analysis.
        
        Side Effects:
            - Creates Crop workflow object for pre-processing
            - Stores reference in self.o_crop1 for crop operations
            - Configures crop region settings before normalization
        """
        self.o_crop1 = Crop(parent=self)
        self.o_crop1.set_region(before_normalization=True)

    def pre_processing_crop(self) -> None:
        """
        Execute pre-processing crop on TimePix data arrays.
        
        Applies the configured crop region to the master 3D data arrays
        before normalization processing. Reduces data volume and focuses
        analysis on the region of interest.
        
        Side Effects:
            - Updates master_3d_data_array with cropped data
            - Reduces data dimensions based on configured crop region
            - Logs crop operation results
        """
        self.o_crop1.run()

    # cleaning low/high pixels - remove outliers
    def clean_images_settings(self) -> None:
        """
        Configure TimePix-specific image cleaning parameters.
        
        Sets up the interface for configuring outlier detection and removal
        parameters optimized for TimePix detector characteristics. Handles
        hot pixels, dead pixels, and other detector-specific artifacts.
        
        Side Effects:
            - Creates ImagesCleaner workflow object
            - Stores reference in self.o_clean for cleaning operations
            - Launches TimePix-optimized parameter configuration interface
        """
        self.o_clean = ImagesCleaner(parent=self)
        self.o_clean.settings()

    def clean_images_setup(self) -> None:
        """
        Prepare TimePix image cleaning algorithms and parameters.
        
        Finalizes the setup of image cleaning algorithms based on TimePix
        detector characteristics and user-configured parameters. Prepares
        cleaning workflow for execution on TimePix imaging data.
        
        Side Effects:
            - Completes cleaning algorithm setup using self.o_clean
            - Prepares TimePix-specific cleaning parameters
        """
        self.o_clean.cleaning_setup()

    def clean_images(self) -> None:
        """
        Execute TimePix image cleaning and outlier removal.
        
        Applies configured cleaning algorithms to remove outlier pixels
        and TimePix detector artifacts from the imaging data. Specifically
        handles dark current correction by ignoring DC processing.
        
        Side Effects:
            - Executes cleaning algorithms using self.o_clean
            - Updates master_3d_data_array with cleaned data
            - Ignores dark current processing (ignore_dc=True)
            - Logs cleaning progress and results
        """
        self.o_clean.cleaning(ignore_dc=True)

    def how_to_visualize_after_cleaning(self) -> None:
        """
        Configure visualization options for cleaned TimePix data.
        
        Sets up visualization interface for reviewing TimePix image cleaning
        results. Configures display parameters for cleaned data type.
        
        Side Effects:
            - Creates Visualization workflow object for cleaned data
            - Configures visualization for DataType.cleaned_images
        """
        self.o_vizu = Visualization(parent=self)
        self.o_vizu.how_to_visualize(data_type=DataType.cleaned_images)

    def visualize_cleaned_data(self) -> None:
        """
        Display cleaned TimePix imaging data for quality validation.
        
        Visualizes the cleaned TimePix imaging data to validate cleaning
        effectiveness and ensure proper outlier removal without data
        degradation.
        
        Side Effects:
            - Displays cleaned TimePix data in 'cleaned' visualization mode
            - Shows results of outlier removal and artifact correction
        """
        self.o_vizu.visualize_timepix_according_to_selection(mode='cleaned')
   
    # normalization
    def normalization_settings(self) -> None:
        """
        Configure TimePix normalization parameters using open beam data.
        
        Sets up the normalization workflow parameters for correcting beam
        intensity variations using open beam measurements. Essential for
        quantitative TimePix neutron imaging reconstruction.
        
        Side Effects:
            - Creates Normalization workflow object
            - Stores reference in self.o_norm for normalization operations
            - Launches TimePix-optimized normalization configuration interface
        """
        self.o_norm = Normalization(parent=self)
        self.o_norm.normalization_settings()

    def normalization_select_roi(self) -> None:
        """
        Select region of interest for TimePix normalization calculations.
        
        Provides interface for selecting specific regions of the TimePix
        detector for normalization calculations. This ROI selection helps
        avoid edge effects and ensures proper beam intensity correction.
        
        Side Effects:
            - Launches ROI selection interface using self.o_norm
            - Updates normalization ROI parameters for TimePix geometry
        """
        self.o_norm.select_roi()

    def normalization(self) -> None:
        """
        Execute TimePix normalization using open beam and dark current.
        
        Applies open beam normalization to correct for beam intensity
        variations and TimePix detector response non-uniformities.
        Combines open beam and dark current data for comprehensive correction.
        
        Side Effects:
            - Combines open beam and dark current using CombineObDc
            - Executes normalization with ignore_dc=True for TimePix workflow
            - Creates normalized_images with corrected data
            - Logs normalization progress and statistics
        """
        o_combine = CombineObDc(parent=self)
        o_combine.run(ignore_dc=True)
        self.o_norm.normalize(ignore_dc=True)

    def visualization_normalization_settings(self) -> None:
        """
        Configure visualization parameters for TimePix normalization results.
        
        Sets up visualization interface for reviewing TimePix normalization
        results and comparing before/after normalization effects.
        
        Side Effects:
            - Creates Visualization workflow object for normalization review
            - Configures visualization parameters for TimePix normalized data
        """
        self.o_vizu = Visualization(parent=self)
        self.o_vizu.settings()

    def visualize_normalization(self) -> None:
        """
        Display before/after comparison of TimePix normalization results.
        
        Generates interactive visualization comparing cleaned and normalized
        TimePix imaging data to validate normalization effectiveness.
        Includes dynamic range controls for optimal TimePix data visualization.
        
        Side Effects:
            - Displays interactive before/after normalization comparison
            - Shows cleaned vs normalized TimePix data with range controls
            - Allows user validation of normalization quality
        """
        self.o_vizu.visualize(data_after=self.normalized_images,
                              label_before='cleaned',
                              label_after='normalized',
                              data_before=self.master_3d_data_array[DataType.sample],
                              turn_on_vrange=True)
    
    def select_export_normalized_folder(self) -> None:
        """
        Select output folder for normalized TimePix image export.
        
        Opens folder selection interface for choosing where to export
        the normalized TimePix imaging data. Useful for saving processed
        data at intermediate preparation stages.
        
        Side Effects:
            - Creates Load workflow object for folder selection
            - Launches folder browser for normalized TimePix data export
        """
        o_select = Load(parent=self)
        o_select.select_folder(data_type=DataType.normalized)

    def export_normalized_images(self) -> None:
        """
        Export normalized TimePix images to selected folder.
        
        Saves the normalized TimePix imaging data to the user-selected
        folder with appropriate file formats and metadata preservation.
        
        Side Effects:
            - Exports normalized TimePix images using self.o_norm
            - Creates output files in selected export folder
            - Preserves TimePix-specific metadata and parameters
        """
        self.o_norm.export_images()

    # chips correction
    def chips_correction(self) -> None:
        """
        Apply TimePix detector chip-specific corrections.
        
        Performs comprehensive corrections specific to TimePix detector
        characteristics including additional outlier removal and chip-specific
        gain corrections. Essential for high-quality TimePix reconstruction.
        
        The correction process:
        1. Removes outliers from normalized images using ImagesCleaner
        2. Applies TimePix chip-specific corrections via ChipsCorrection
        
        Side Effects:
            - Removes outliers from normalized_images using ImagesCleaner
            - Applies chip-specific corrections using ChipsCorrection
            - Updates normalized_images with corrected data
            - Logs correction progress and effectiveness metrics
        """
        o_clean = ImagesCleaner(parent=self)
        self.normalized_images = o_clean.remove_outliers(self.normalized_images[:])

        o_chips = ChipsCorrection(parent=self)
        o_chips.run()

    def visualize_chips_correction(self) -> None:
        """
        Display visualization of TimePix chip correction results.
        
        Shows the effects of TimePix chip-specific corrections on the
        imaging data, helping users validate correction effectiveness.
        
        Side Effects:
            - Creates ChipsCorrection workflow object for visualization
            - Displays TimePix chip correction comparison interface
        """
        o_chips = ChipsCorrection(parent=self)
        o_chips.visualize_chips_correction()

    # rebin
    def rebin_settings(self) -> None:
        """
        Configure rebinning parameters for TimePix data optimization.
        
        Sets up the interface for configuring data rebinning parameters
        to optimize TimePix data for reconstruction. Rebinning can improve
        signal-to-noise ratio and reduce computation time.
        
        Side Effects:
            - Creates Rebin workflow object
            - Stores reference in self.o_rebin for rebinning operations
            - Launches rebinning parameter configuration interface
        """
        self.o_rebin = Rebin(parent=self)
        self.o_rebin.set_rebinning()

    def rebin_before_normalization(self) -> None:
        """
        Execute rebinning before normalization processing.
        
        Applies rebinning to the TimePix data before normalization step.
        This can help reduce noise and improve processing efficiency
        for subsequent normalization operations.
        
        Side Effects:
            - Executes rebinning using self.o_rebin before normalization
            - Modifies normalized_images with rebinned data
            - Stores original data in before_rebinning for comparison
        """
        self.o_rebin.execute_binning_before_normalization()

    def rebin_after_normalization(self) -> None:
        """
        Execute rebinning after normalization processing.
        
        Applies rebinning to the TimePix data after normalization step.
        This approach preserves fine details during normalization while
        reducing data volume for subsequent processing.
        
        Side Effects:
            - Executes rebinning using self.o_rebin after normalization
            - Modifies normalized_images with rebinned data
            - Updates data dimensions for subsequent processing steps
        """
        self.o_rebin.execute_binning_after_normalization()

    def visualize_rebinned_data(self, before_normalization: bool = False) -> None:
        """
        Display before/after comparison of rebinning results.
        
        Generates visualization comparing original and rebinned TimePix
        data to validate rebinning effectiveness and parameter selection.
        Supports visualization for both pre- and post-normalization rebinning.
        
        Args:
            before_normalization: If True, shows rebinning applied before
                                normalization. If False, shows post-normalization
                                rebinning results.
        
        Side Effects:
            - Displays interactive before/after rebinning comparison
            - Shows appropriate data based on rebinning timing
            - Returns early with message if no rebinning was performed
        """
        if before_normalization:
            data_after = self.master_3d_data_array[DataType.sample]
            if self.before_rebinning is None:
                display(HTML("No rebinning performed!"))
                return
            
            data_before = self.before_rebinning
           
            self.o_vizu.visualize(data_after=data_after,
                                 label_before='raw',
                                 label_after='rebinned',
                                 data_before=data_before,
                                 turn_on_vrange=True,
            )

        else:
            data_after = self.normalized_images
            data_before = self.before_rebinning
            vmin = 0
            vmax = 1
            vmin_after = 0
            vmax_after = 1
        
            self.o_vizu.visualize(data_after=data_after,
                            label_before='raw',
                            label_after='rebinned',
                            data_before=data_before,
                            turn_on_vrange=True,
                            vmin=vmin,
                            vmax=vmax,
                            vmin_after=vmin_after,
                            vmax_after=vmax_after)

    # crop data
    def crop_settings(self) -> None:
        """
        Configure final crop parameters for TimePix reconstruction data.
        
        Sets up the interface for defining final crop regions on processed
        TimePix data. This crop is applied after normalization to focus
        on the specific region of interest for reconstruction.
        
        Side Effects:
            - Creates Crop workflow object for final processing
            - Stores reference in self.o_crop for crop operations
            - Configures final crop region settings
        """
        self.o_crop = Crop(parent=self)
        self.o_crop.set_region()

    def crop(self) -> None:
        """
        Execute final crop on processed TimePix data.
        
        Applies the configured final crop region to the normalized
        TimePix imaging data, focusing on the region of interest
        for CT reconstruction.
        
        Side Effects:
            - Updates normalized_images with cropped data
            - Reduces data dimensions based on configured crop region
            - Optimizes data size for reconstruction processing
        """
        self.o_crop.run()

    # rotate sample
    def rotate_data_settings(self) -> None:
        """
        Configure rotation parameters for TimePix sample alignment.
        
        Sets up the interface for configuring sample rotation parameters
        to correct for sample misalignment in the TimePix detector setup.
        Essential for proper CT reconstruction geometry.
        
        Side Effects:
            - Creates Rotate workflow object
            - Stores reference in self.o_rotate for rotation operations
            - Launches rotation parameter configuration interface
        """
        self.o_rotate = Rotate(parent=self)
        self.o_rotate.set_settings()

    def apply_rotation(self) -> None:
        """
        Apply rotation correction to TimePix sample data.
        
        Executes the configured rotation correction to align the sample
        properly for CT reconstruction. Corrects for sample positioning
        errors in the TimePix imaging setup.
        
        Side Effects:
            - Updates normalized_images with rotation-corrected data
            - Applies geometric transformation to align sample orientation
            - Logs rotation parameters and correction results
        """
        self.o_rotate.apply_rotation()

    def visualize_after_rotation(self) -> None:
        """
        Display TimePix data after rotation correction.
        
        Shows the first image of the rotation-corrected TimePix dataset
        to validate the effectiveness of the rotation correction.
        
        Side Effects:
            - Creates FinalProjectionsReview object for single image display
            - Shows first normalized image after rotation correction
        """
        o_review = FinalProjectionsReview(parent=self)
        o_review.single_image(image=self.normalized_images[0])

    # log conversion
    def log_conversion_and_cleaning(self) -> None:
        """
        Apply logarithmic conversion and final cleaning to TimePix data.
        
        Performs logarithmic conversion on normalized TimePix data to
        enhance reconstruction quality, followed by comprehensive cleaning
        to remove artifacts introduced by the log transformation.
        
        The process includes:
        1. Log conversion of normalized images
        2. Outlier removal after log transformation
        3. Negative value removal for reconstruction compatibility
        
        Side Effects:
            - Creates normalized_images_log with log-converted data
            - Removes outliers and negative values from log data
            - Logs array information for quality monitoring
            - Prepares data for optimal reconstruction performance
        """
        normalized_images_log = log_conversion(self.normalized_images[:])
        o_cleaner = ImagesCleaner(parent=self)
        normalized_images_log = o_cleaner.remove_outliers(normalized_images_log[:])
        normalized_images_log = remove_negative_values(normalized_images_log[:])

        # self.corrected_images_log = normalized_images_log[:]
        self.normalized_images_log = normalized_images_log[:]
        logging_3d_array_infos(array=normalized_images_log, message="normalized_images_log")

    def visualize_images_after_log(self) -> None:
        """
        Display comparison of TimePix data before and after log conversion.
        
        Generates side-by-side visualization comparing linear and log-converted
        TimePix data to validate the log conversion effectiveness and
        parameter selection.
        
        Side Effects:
            - Creates Visualization object for dual-stack comparison
            - Shows linear data (left) vs log-converted data (right)
            - Uses appropriate value ranges for each data type
        """
        o_vizu = Visualization(parent=self)
        o_vizu.visualize_2_stacks(left=self.normalized_images, 
                                  vmin_left=0, 
                                  vmax_left=1,
                                  right=self.normalized_images_log,
                                  vmin_right=None,
                                  vmax_right=None,)

    # strips removal
    def select_range_of_data_to_test_stripes_removal(self) -> None:
        """
        Select data range for testing strip removal algorithms on TimePix data.
        
        Provides interface for selecting a subset of TimePix projection
        images to test strip removal algorithms before applying to the
        entire dataset. Essential for parameter optimization.
        
        Side Effects:
            - Creates RemoveStrips workflow object
            - Stores reference in self.o_remove for strip removal operations
            - Updates list_of_images[DataType.sample] with test range selection
            - Launches data range selection interface
        """
        self.o_remove = RemoveStrips(parent=self)
        self.o_remove.select_range_of_data_to_test_stripes_removal()

    def select_remove_strips_algorithms(self) -> None:
        """
        Select strip removal algorithms for TimePix artifact correction.
        
        Provides interface for choosing from multiple strip removal
        algorithms optimized for different types of strip artifacts
        commonly found in TimePix neutron imaging data.
        
        Side Effects:
            - Launches algorithm selection interface using self.o_remove
            - Configures strip removal algorithms for TimePix characteristics
        """
        self.o_remove.select_algorithms()

    def define_settings(self) -> None:
        """
        Configure strip removal algorithm parameters for TimePix data.
        
        Provides interface for fine-tuning the parameters of selected
        strip removal algorithms based on TimePix detector characteristics
        and the specific artifacts observed in the data.
        
        Side Effects:
            - Launches parameter configuration interface using self.o_remove
            - Updates strip removal algorithm settings for TimePix optimization
        """
        self.o_remove.define_settings()

    def test_algorithms_on_selected_range_of_data(self) -> None:
        """
        Test strip removal algorithms on selected TimePix data range.
        
        Applies configured strip removal algorithms to the selected test
        range of TimePix data and displays results for parameter validation
        before applying to the complete dataset.
        
        Side Effects:
            - Executes strip removal on test data using self.o_remove
            - Updates strip_corrected_images with test results
            - Displays test results for algorithm validation
        """
        self.o_remove.perform_cleaning(test=True)
        self.o_remove.display_cleaning(test=True)

    def when_to_remove_strips(self) -> None:
        """
        Configure timing for strip removal in TimePix processing workflow.
        
        Provides interface for determining when in the processing workflow
        strip removal should be applied to the log-converted TimePix data.
        
        Side Effects:
            - Configures strip removal timing using self.o_remove
            - Updates normalized_images_log processing workflow
        """
        self.o_remove.when_to_remove_strips()

    def remove_strips(self) -> None:
        """
        Execute strip removal on complete TimePix dataset.
        
        Applies the validated strip removal algorithms to the complete
        log-converted TimePix dataset using the optimized parameters
        determined during testing.
        
        Side Effects:
            - Executes strip removal on full dataset using self.o_remove
            - Updates normalized_images_log with strip-corrected data
            - Logs strip removal progress and effectiveness metrics
        """
        self.o_remove.perform_cleaning()

    def display_removed_strips(self) -> None:
        """
        Display results of strip removal on TimePix data.
        
        Shows before/after comparison of strip removal results on the
        complete TimePix dataset to validate the effectiveness of the
        strip correction process.
        
        Side Effects:
            - Displays strip removal results using self.o_remove
            - Shows before/after comparison for validation
        """
        self.o_remove.display_cleaning()

    # def select_remove_strips_algorithms(self):
    #     self.o_remove = RemoveStrips(parent=self)
    #     self.o_remove.select_algorithms()

    # def define_settings(self):
    #     self.o_remove.define_settings()

    # def when_to_remove_strips(self):
    #     """updates: normalized_images_log"""
    #     self.o_remove.when_to_remove_strips()

    # def remove_strips(self):
    #     """updates: normalized_images_log"""
    #     self.o_remove.perform_cleaning()

    # def display_removed_strips(self):
    #     self.o_remove.display_cleaning()

    # calculate and apply tilt
    def select_sample_roi(self) -> None:
        """
        Select region of interest for tilt correction calculation.
        
        Provides interface for selecting a region of the TimePix sample
        data to use for tilt angle determination and correction. Essential
        for correcting sample misalignment in the rotation axis.
        
        Side Effects:
            - Creates CenterOfRotationAndTilt workflow object
            - Stores reference in self.o_tilt for tilt operations
            - Launches ROI selection interface for tilt calculation
        """
        self.o_tilt = CenterOfRotationAndTilt(parent=self)
        self.o_tilt.select_range()

    def perform_tilt_correction(self) -> None:
        """
        Execute tilt correction on TimePix log-converted data.
        
        Applies calculated tilt correction to compensate for sample
        misalignment in the TimePix imaging setup. Critical for accurate
        CT reconstruction geometry.
        
        Side Effects:
            - Executes tilt correction using self.o_tilt
            - Updates normalized_images_log with tilt-corrected data
            - Logs tilt correction parameters and results
        """
        self.o_tilt.run_tilt_correction()

    # calcualte center of rotation
    def center_of_rotation_settings(self) -> None:
        """
        Configure center of rotation calculation for TimePix reconstruction.
        
        Sets up the interface for center of rotation determination using
        0°, 180°, and 360° TimePix projection images. Isolates specific
        angular positions and configures calculation parameters.
        
        Side Effects:
            - Creates CenterOfRotationAndTilt object if not already initialized
            - Isolates 0°, 180°, and 360° degree images for calculation
            - Launches center of rotation parameter configuration interface
        """
        if self.o_tilt is None:
            self.o_tilt = CenterOfRotationAndTilt(parent=self)
        self.o_tilt.isolate_0_180_360_degrees_images()
        self.o_tilt.center_of_rotation_settings()

    def run_center_of_rotation(self) -> None:
        """
        Execute center of rotation calculation using TimePix log data.
        
        Runs the center of rotation calculation algorithms using the
        log-converted TimePix data and isolated angular positions.
        Essential for accurate CT reconstruction geometry.
        
        Side Effects:
            - Executes center of rotation calculation using self.o_tilt
            - Uses normalized_images_log for calculation
            - Determines optimal center of rotation value
        """
        self.o_tilt.run_center_of_rotation()

    def run_center_of_rotation_or_skip_it(self) -> None:
        """
        Calculate center of rotation or provide skip option.
        
        Provides interface for either calculating the center of rotation
        automatically or allowing the user to skip this step if the
        center is already known or will be determined manually.
        
        Side Effects:
            - Launches center of rotation calculation interface
            - Allows user choice between calculation and manual specification
        """
        self.o_tilt.calculate_center_of_rotation()

    def display_center_of_rotation(self) -> None:
        """
        Display center of rotation calculation results for TimePix data.
        
        Shows the calculated center of rotation overlaid on TimePix
        reconstruction test images to validate the accuracy of the
        calculated center position.
        
        Side Effects:
            - Displays center of rotation test results using self.o_tilt
            - Shows validation images with calculated center overlay
        """
        self.o_tilt.test_center_of_rotation_calculated()

    # test reconstruction using gridrec (fast algorithm)
    def select_slices_to_use_to_test_reconstruction(self) -> None:
        """
        Select TimePix slices for test reconstruction validation.
        
        Provides interface for selecting specific slices from the prepared
        TimePix data to test reconstruction algorithms and parameters
        before running the full reconstruction process.
        
        Side Effects:
            - Creates TestReconstruction workflow object
            - Stores reference in self.o_test for test operations
            - Uses normalized_images_log for slice selection
            - Launches slice selection interface
        """
        self.o_test = TestReconstruction(parent=self)
        self.o_test.select_slices()

    def run_reconstruction_of_slices_to_test(self) -> None:
        """
        Execute test reconstruction on selected TimePix slices.
        
        Runs fast reconstruction algorithms (gridrec) on selected TimePix
        slices to validate preparation workflow and reconstruction parameters
        before full processing.
        
        Side Effects:
            - Executes test reconstruction using self.o_test
            - Displays test reconstruction results for validation
            - Validates preparation workflow effectiveness
        """
        self.o_test.run_reconstruction()

    # select reconstruction method
    def select_reconstruction_method(self) -> None:
        """
        Select reconstruction algorithm for TimePix data processing.
        
        Provides interface for choosing between available reconstruction
        algorithms (SVMBIR, FBP, etc.) based on TimePix data characteristics
        and reconstruction quality requirements.
        
        Side Effects:
            - Creates ReconstructionSelection workflow object
            - Stores reference in self.o_mode for reconstruction selection
            - Launches reconstruction algorithm selection interface
        """
        self.o_mode = ReconstructionSelection(parent=self)
        self.o_mode.select()

    # run svmbir
    def reconstruction_settings(self) -> None:
        """
        Configure reconstruction algorithm parameters for TimePix data.
        
        Sets up reconstruction parameters based on the selected algorithm.
        If SVMBIR is selected, configures SVMBIR-specific parameters
        optimized for TimePix detector characteristics.
        
        Side Effects:
            - Creates SvmbirHandler if SVMBIR is selected in configuration
            - Stores reference in self.o_svmbir for SVMBIR operations
            - Launches reconstruction parameter configuration interface
        """
        if ReconstructionAlgorithm.svmbir in self.configuration.reconstruction_algorithm:
            self.o_svmbir = SvmbirHandler(parent=self)
            self.o_svmbir.set_settings()
       
    def svmbir_run(self) -> None:
        """
        Execute SVMBIR reconstruction on prepared TimePix data.
        
        Runs the complete SVMBIR iterative reconstruction algorithm
        using the prepared and corrected TimePix data. Displays
        reconstructed slices for immediate quality assessment.
        
        Side Effects:
            - Executes SVMBIR reconstruction using self.o_svmbir
            - Updates reconstruction_array with 3D reconstructed volume
            - Displays reconstructed slices for immediate review
            - Logs reconstruction progress and completion statistics
        """
        self.o_svmbir.run_reconstruction()
        self.o_svmbir.display_slices()

    # export slices
    def select_export_slices_folder(self) -> None:
        """
        Select output folder for TimePix reconstructed slice export.
        
        Opens folder selection interface for choosing where to export
        the final reconstructed CT slices from TimePix data. These are
        the primary output of the reconstruction process.
        
        Side Effects:
            - Creates Load workflow object for folder selection
            - Launches folder browser for reconstructed slice export path
        """
        o_select = Load(parent=self)
        o_select.select_folder(data_type=DataType.reconstructed)

    def export_slices(self) -> None:
        """
        Export reconstructed TimePix slices to selected folder.
        
        Saves the final reconstructed CT slices from TimePix data to the
        user-selected folder with appropriate file formats and metadata.
        These are the primary deliverable of the reconstruction workflow.
        
        Side Effects:
            - Exports reconstructed slices using self.o_svmbir
            - Creates output files in selected export folder
            - Preserves TimePix-specific metadata and reconstruction parameters
        """
        self.o_svmbir.export_images()

    # export extra files
    def select_export_extra_files(self) -> None:
        """
        Select output folder for additional TimePix reconstruction files.
        
        Opens folder selection interface for choosing where to export
        additional files such as configuration files, logs, and
        TimePix-specific reconstruction metadata.
        
        Side Effects:
            - Creates Load workflow object for folder selection
            - Launches folder browser for extra files export path
        """
        o_select = Load(parent=self)
        o_select.select_folder(data_type=DataType.extra)

    def export_pre_reconstruction_data(self) -> None:
        """
        Export pre-reconstruction TimePix data and parameters.
        
        Saves processed TimePix data and parameters that will be used
        for reconstruction. Supports both SVMBIR and FBP workflows
        depending on the selected reconstruction algorithm.
        
        Side Effects:
            - Uses FbpHandler if o_svmbir is None (FBP workflow)
            - Uses SvmbirHandler if available (SVMBIR workflow)
            - Exports pre-reconstruction data and parameters
        """
        if self.o_svmbir is None:
            o_fbp = FbpHandler(parent=self)
            o_fbp.export_pre_reconstruction_data()
        else:
            self.o_svmbir.export_pre_reconstruction_data()

    def export_extra_files(self, prefix: str = "") -> None:
        """
        Export additional TimePix reconstruction files and metadata.
        
        Saves configuration files, processing logs, reconstruction
        parameters, and other TimePix-specific metadata to the selected
        folder for documentation and reproducibility.
        
        Args:
            prefix: Optional prefix to add to exported filenames for
                   organization and identification.
        
        Side Effects:
            - Exports pre-reconstruction data first
            - Creates ExportExtra workflow object
            - Exports configuration files, logs, and TimePix metadata
            - Uses LOG_BASENAME_FILENAME and optional prefix for organization
        """
        self.export_pre_reconstruction_data()
        o_export = ExportExtra(parent=self)
        o_export.run(base_log_file_name=LOG_BASENAME_FILENAME,
                     prefix=prefix)
        