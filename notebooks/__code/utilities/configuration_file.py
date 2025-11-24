"""
Configuration Management for Neutron CT Reconstruction Pipeline.

This module provides comprehensive configuration management for neutron computed
tomography reconstruction workflows. It defines Pydantic models for all
reconstruction parameters, ensuring type safety and validation across the
entire CT reconstruction pipeline.

Key Classes:
    - Configuration: Main configuration model with all reconstruction parameters
    - SvmbirConfig: SVMBIR algorithm-specific configuration
    - RemoveStripe*: Various strip removal algorithm configurations
    - ImageSize: Image dimension specifications
    - CropRegion: Image cropping region definitions
    - NormalizationRoi: Region of interest for normalization

Key Features:
    - Type-safe configuration with Pydantic validation
    - Support for multiple reconstruction algorithms
    - Comprehensive strip removal algorithm configurations
    - Flexible data cleaning and normalization parameters
    - JSON-based configuration file loading and validation
    - Interactive file selection for configuration management

Configuration Categories:
    - Instrument and experiment settings (IPTS, instrument type)
    - Reconstruction algorithm selection and parameters
    - Data organization (sample, open beam, dark current paths)
    - Image processing parameters (cleaning, normalization, cropping)
    - Strip removal algorithms with detailed parameter control
    - Center of rotation and geometric correction settings
    - Output folder organization and file management

Reconstruction Algorithms Supported:
    - SVMBIR: Sparse View Model Based Iterative Reconstruction
    - MBIRJAX: JAX-based Model Based Iterative Reconstruction
    - FBP variants: ASTRA, TomoPy, AlgoTom implementations
    - GridRec: Fast grid-based reconstruction algorithm

Dependencies:
    - pydantic: Type validation and model definitions
    - typing: Type hint support for complex data structures
    - JSON handling utilities for configuration persistence

Author: Neutron Imaging Team
Created: Configuration management for CT reconstruction pipeline
"""

from pydantic import BaseModel, Field
from typing import List, Tuple, Optional, Union

from __code.utilities.json import load_json_string
from __code import CleaningAlgorithm, NormalizationSettings, OperatingMode, WhenToRemoveStripes, Instrument
from __code.utilities.file_folder_browser import FileFolderBrowser
from __code.config import SVMBIR_LIB_PATH


class RemoveStripeFwWnameOptions:
    """
    Wavelet name options for Fourier-Wavelet strip removal algorithm.
    
    Defines available wavelet basis functions for the FW (Fourier-Wavelet)
    strip removal method. Each wavelet has different characteristics for
    artifact removal effectiveness.
    
    Attributes:
        haar: Haar wavelet - simple, good for sharp discontinuities
        db5: Daubechies 5 wavelet - balanced smoothness and localization
        sym5: Symlet 5 wavelet - nearly symmetric, good reconstruction
    """
    haar: str = 'haar'
    db5: str = 'db5'
    sym5: str = 'sym5'


class RemoveStripeDim:
    """
    Dimension options for multi-dimensional strip removal algorithms.
    
    Specifies whether strip removal should be applied in 1D (along columns)
    or 2D (considering both spatial dimensions) for various algorithms.
    
    Attributes:
        one: 1D processing - strips removed column-wise
        two: 2D processing - considers spatial correlations
    """
    one: int = 1
    two: int = 2


class ImageSize(BaseModel):
    """
    Image dimension configuration for neutron CT reconstruction.
    
    Defines the width and height dimensions for processing neutron imaging
    data. Used for cropping, rebinning, and reconstruction parameter setup.
    Default values are optimized for typical neutron detector dimensions.
    
    Attributes:
        width: Image width in pixels (default: 512)
        height: Image height in pixels (default: 512)
    """
    width: int = 512
    height: int = 512


class RemoveStripeFw(BaseModel):
    """
    Configuration for Fourier-Wavelet (FW) strip removal algorithm.
    
    Parameters for the FW method which combines Fourier analysis with
    wavelet decomposition to remove strip artifacts. Effective for
    periodic and semi-periodic strip patterns in neutron imaging.
    
    Attributes:
        level: Decomposition level for wavelet transform ("None" for auto)
        wname: Wavelet name from RemoveStripeFwWnameOptions (default: haar)
        sigma: Standard deviation for Gaussian filtering (default: 2)
        pad: Whether to use padding during wavelet transform (default: True)
    """
    level: str = "None"
    wname: str = Field(default=RemoveStripeFwWnameOptions.haar)
    sigma: float = 2
    pad: bool = True


class RemoveStripeTi(BaseModel):
    """
    Configuration for Titarenko (TI) strip removal algorithm.
    
    Parameters for the TI method which uses regularized optimization
    to remove ring artifacts. Particularly effective for systematic
    detector response variations in neutron CT.
    
    Attributes:
        nblock: Number of blocks for processing (0 for automatic)
        alpha: Regularization parameter controlling smoothness (default: 1.5)
    """
    nblock: int = 0
    alpha: float = 1.5


class RemoveStripeSf(BaseModel):
    """
    Configuration for Smoothing Filter (SF) strip removal algorithm.
    
    Parameters for the SF method which applies smoothing operations
    to reduce strip artifacts while preserving image features.
    
    Attributes:
        size: Filter kernel size for smoothing operation (default: 5)
    """
    size: int = 5


class RemoveStripeBasedSorting(BaseModel):
    """
    Configuration for sorting-based strip removal algorithm.
    
    Parameters for strip removal method that uses sorting operations
    to identify and correct outlier values causing strip artifacts.
    
    Attributes:
        size: Window size for sorting operation ("None" for automatic)
        dim: Processing dimension (1D or 2D from RemoveStripeDim)
    """
    size: str = "None"
    dim: int = Field(default=RemoveStripeDim.one)


class RemoveStripeBasedFiltering(BaseModel):
    """
    Configuration for filtering-based strip removal algorithm.
    
    Parameters for strip removal using various filtering approaches
    to suppress artifacts while maintaining image quality.
    
    Attributes:
        sigma: Standard deviation for filtering operations (default: 3)
        size: Filter size ("None" for automatic determination)
        dim: Processing dimension (1D or 2D from RemoveStripeDim)
    """
    sigma: float = 3
    size: str = "None"
    dim: int = Field(default=RemoveStripeDim.one)


class RemoveStripeBasedFitting(BaseModel):
    """
    Configuration for polynomial fitting-based strip removal algorithm.
    
    Parameters for strip removal using polynomial fitting to model
    and subtract systematic variations causing strip artifacts.
    
    Attributes:
        order: Polynomial order for fitting (3-10, default: 3)
        sigma: Sigma range for outlier detection (default: "5,20")
    """
    order: int = Field(default=3, ge=3, le=10)
    sigma: str = "5,20"


class RemoveLargeStripe(BaseModel):
    """
    Configuration for large stripe removal algorithm.
    
    Parameters specifically designed for removing large-scale strip
    artifacts that span significant portions of the image.
    
    Attributes:
        snr: Signal-to-noise ratio threshold (default: 3)
        size: Window size for large stripe detection (default: 51)
        drop_ratio: Fraction of extreme values to exclude (default: 0.1)
        norm: Whether to apply normalization (default: True)
    """
    snr: float = 3
    size: int = 51
    drop_ratio: float = 0.1
    norm: bool = True


class RemoveDeadStripe(BaseModel):
    """
    Configuration for dead stripe removal algorithm.
    
    Parameters for removing artifacts caused by dead or malfunctioning
    detector pixels that create consistent vertical stripes.
    
    Attributes:
        snr: Signal-to-noise ratio threshold (default: 3)
        size: Window size for dead stripe detection (default: 51)
        norm: Whether to apply normalization (default: True)
    """
    snr: float = 3
    size: int = 51
    norm: bool = True


class RemoveAllStripe(BaseModel):
    """
    Configuration for comprehensive stripe removal algorithm.
    
    Parameters for removing all types of stripe artifacts using
    a combination of methods for comprehensive artifact correction.
    
    Attributes:
        snr: Signal-to-noise ratio threshold (default: 3)
        la_size: Large stripe window size (default: 61)
        sm_size: Small stripe window size (default: 21)
        dim: Processing dimension (1D or 2D from RemoveStripeDim)
    """
    snr: float = 3
    la_size: int = 61
    sm_size: int = 21
    dim: int = Field(default=RemoveStripeDim.one)


class RemoveStripeBasedInterpolation(BaseModel):
    """
    Configuration for interpolation-based stripe removal algorithm.
    
    Parameters for strip removal using interpolation methods to
    estimate and correct corrupted pixel values causing artifacts.
    
    Attributes:
        snr: Signal-to-noise ratio threshold (default: 3)
        size: Interpolation window size (default: 31)
        drop_ratio: Fraction of extreme values to exclude (default: 0.1)
        norm: Whether to apply normalization (default: True)
    """
    snr: float = 3
    size: int = 31
    drop_ratio: float = .1
    norm: bool = True


class HistogramCleaningSettings(BaseModel):
    """
    Configuration for histogram-based image cleaning.
    
    Parameters for cleaning algorithms that use histogram analysis
    to identify and remove outlier pixels and artifacts.
    
    Attributes:
        nbr_bins: Number of histogram bins for analysis (default: 10)
        bins_to_exclude: Number of extreme bins to exclude (default: 1)
    """
    nbr_bins: int = 10
    bins_to_exclude: int = 1


class TopFolder(BaseModel):
    """
    Configuration for top-level data folder paths.
    
    Specifies the root directories containing sample and open beam
    data for the neutron CT reconstruction workflow.
    
    Attributes:
        sample: Path to sample data folder (default: empty string)
        ob: Path to open beam data folder (default: empty string)
    """
    sample: str = ""
    ob: str = ""


class NormalizationRoi(BaseModel):
    """
    Configuration for normalization region of interest.
    
    Defines the rectangular region used for normalization calculations.
    Coordinates are specified as fractions (0-1) of the total image dimensions.
    
    Attributes:
        top: Top boundary of ROI (fraction, default: 0)
        bottom: Bottom boundary of ROI (fraction, default: 1)
        left: Left boundary of ROI (fraction, default: 0)
        right: Right boundary of ROI (fraction, default: 1)
    """
    top: int = 0
    bottom: int = 1
    left: int = 0
    right: int = 1


class SvmbirConfig(BaseModel):
    """
    Configuration for SVMBIR (Sparse View Model Based Iterative Reconstruction).
    
    Comprehensive parameter set for the SVMBIR algorithm, which provides
    high-quality reconstruction from sparse-view or limited-angle CT data
    using iterative optimization with regularization.
    
    Attributes:
        sharpness: Edge preservation parameter (default: 0)
        snr_db: Signal-to-noise ratio in decibels (default: 30.0)
        positivity: Enforce positivity constraint (default: True)
        max_iterations: Maximum number of iterations (default: 200)
        max_resolutions: Maximum resolution levels (default: 3)
        verbose: Enable verbose output logging (default: False)
        top_slice: Starting slice for reconstruction (default: 0)
        bottom_slice: Ending slice for reconstruction (default: 1)
    """
    sharpness: float = 0
    snr_db: float = 30.0
    positivity: bool = True
    max_iterations: int = 200
    max_resolutions: int = 3
    verbose: bool = False
    top_slice: int = 0
    bottom_slice: int = 1


class CropRegion(BaseModel):
    """
    Configuration for image cropping region.
    
    Defines the rectangular region to extract from images during processing.
    Coordinates are specified as fractions (0-1) of the total image dimensions.
    
    Attributes:
        left: Left boundary of crop region (fraction, default: 0)
        right: Right boundary of crop region (fraction, default: 1)
        top: Top boundary of crop region (fraction, default: 0)
        bottom: Bottom boundary of crop region (fraction, default: 1)
    """
    left: int = 0
    right: int = 1
    top: int = 0
    bottom: int = 1


class ReconstructionAlgorithm:
    """
    Available reconstruction algorithms for neutron CT processing.
    
    Defines string constants for all supported reconstruction algorithms
    in the neutron CT pipeline. Each algorithm has different characteristics
    regarding speed, quality, and computational requirements.
    
    Iterative Algorithms:
        svmbir: Sparse View Model Based Iterative Reconstruction - high quality
        mbirjax: JAX-based Model Based Iterative Reconstruction - GPU accelerated
    
    Filtered Back Projection (FBP) Algorithms:
        astra_fbp: ASTRA Toolbox FBP implementation - fast, GPU support
        tomopy_fbp: TomoPy FBP implementation - versatile, well-tested
        algotom_fbp: AlgoTom FBP implementation - optimized for large data
        algotom_gridrec: AlgoTom GridRec - very fast, good for quick previews
    
    Note: Commented algorithms are available but not currently active
    """
    svmbir: str = "svmbir"
    mbirjax: str = "mbirjax"
    astra_fbp: str = "astra_fbp"
    tomopy_fbp: str = "tomopy_fbp"
    algotom_fbp: str = "algotom_fbp"
    algotom_gridrec: str = "algotom_gridrec"

    # art: str = "art"
    # bart: str = "bart"
    # gridrec: str = "gridrec"
    # mlem: str = "mlem"
    # osem: str = "osem"  
    # ospml_hybrid: str = "ospml_hybrid"
    # ospml_quad: str = "ospml_quad"  
    # pml_hybrid: str = "pml_hybrid"
    # pml_quad: str = "pml_quad"
    # # sirt: str = "sirt"
    # # tv: str = "tv"
    # grad: str = "grad"
    # tikh: str = "tikh"


class Configuration(BaseModel):
    """
    Master configuration model for neutron CT reconstruction pipeline.
    
    This comprehensive configuration class manages all parameters needed
    for neutron computed tomography reconstruction workflows. It provides
    type-safe parameter management with validation for the entire pipeline
    from data loading through final reconstruction export.
    
    Configuration Categories:
        - Instrument and experiment identification
        - Reconstruction algorithm selection and parameters
        - Data organization and file path management
        - Image processing and cleaning parameters
        - Normalization and correction settings
        - Strip removal algorithm configurations
        - Geometric correction parameters
        - Output folder organization
    
    The configuration supports multiple reconstruction algorithms, comprehensive
    artifact removal, and flexible parameter adjustment for different neutron
    imaging experiments and detector systems.
    """

    instrument: str = Field(default=Instrument.mars, description="Instrument used for the reconstruction: mars, venus, snap.")
    ipts_number: int = Field(default=27829, description="IPTS number for the experiment.")

    reconstruction_algorithm: List[str] = Field(default=[ReconstructionAlgorithm.algotom_gridrec])

    top_folder: TopFolder = Field(default=TopFolder())
    operating_mode: str = Field(default=OperatingMode.tof) 
    image_size: ImageSize = Field(default=ImageSize())
    crop_region: CropRegion = Field(default=CropRegion())

    list_of_angles: List[float] = Field(default=[])
    list_of_sample_runs: List[str] = Field(default=[])
    list_of_sample_frame_number: List[int] = Field(default=[])
    list_of_sample_pc: List[float] = Field(default=[])

    list_of_ob_runs: List[str] = Field(default=[])
    list_of_ob_frame_number: List[int] = Field(default=[])
    list_of_ob_pc: List[float] = Field(default=[])

    range_of_tof_to_combine: List[Tuple[int, int]] = Field(default=[[0, -1]])
    
    list_of_slices_to_reconstruct: List[Tuple[int, int]] = Field(default=[[0, -1]])

    list_clean_algorithm: List[str] = Field(default=[CleaningAlgorithm.tomopy])
    histogram_cleaning_settings: HistogramCleaningSettings = Field(default=HistogramCleaningSettings())
    list_normalization_settings: List[str] = Field(default=[NormalizationSettings.pc, 
                                              NormalizationSettings.frame_number,
                                              NormalizationSettings.sample_roi,
                                              NormalizationSettings.roi])
    normalization_roi: NormalizationRoi = Field(default=NormalizationRoi())
    
    list_clean_stripes_algorithm: List[str] = Field(default=[])
    remove_stripe_fw_options: RemoveStripeFw = Field(default=RemoveStripeFw())
    remove_stripe_ti_options: RemoveStripeTi = Field(default=RemoveStripeTi())
    remove_stripe_sf_options: RemoveStripeSf = Field(default=RemoveStripeSf())
    remove_stripe_based_sorting_options: RemoveStripeBasedSorting = Field(default=RemoveStripeBasedFiltering())
    remove_stripe_based_filtering_options: RemoveStripeBasedFiltering = Field(default=RemoveStripeBasedFiltering())
    remove_stripe_based_fitting_options: RemoveStripeBasedFitting = Field(default=RemoveStripeBasedFitting())
    remove_large_stripe_options: RemoveLargeStripe = Field(default=RemoveLargeStripe())
    remove_dead_stripe_options: RemoveDeadStripe = Field(default=RemoveDeadStripe())
    remove_all_stripe_options: RemoveAllStripe = Field(default=RemoveAllStripe())
    remove_stripe_based_interpolation_options: RemoveStripeBasedInterpolation = Field(default=RemoveStripeBasedInterpolation())
    when_to_remove_stripes: str = Field(
        default=WhenToRemoveStripes.out_notebook,
        description="When to remove stripes: 'in the notebook' or 'outside the notebook'."
    )

    calculate_center_of_rotation: bool = Field(default=False)
    range_of_slices_for_center_of_rotation: List[int] = Field(default=[0, -1])
    center_of_rotation: float = Field(default=-1)
    center_offset: float = Field(default=0)
    
    svmbir_config: SvmbirConfig = Field(default=SvmbirConfig())
    output_folder: str = Field(default="")
    reconstructed_output_folder: str = Field(default="")
    projections_pre_processing_folder: str = Field(default="")


def loading_config_file_into_model(config_file_path: str) -> Configuration:
    """
    Load and validate a JSON configuration file into a Configuration model.
    
    Reads a JSON configuration file from disk and parses it into a validated
    Configuration model instance. Provides type checking and validation of
    all configuration parameters according to the defined schema.
    
    Args:
        config_file_path: Absolute path to the JSON configuration file
                         containing reconstruction parameters.
    
    Returns:
        Configuration: Validated configuration model instance with all
                      parameters loaded and type-checked.
    
    Raises:
        FileNotFoundError: If the configuration file doesn't exist
        ValidationError: If the configuration contains invalid parameters
        JSONDecodeError: If the file contains invalid JSON syntax
    
    Example:
        >>> config = loading_config_file_into_model("/path/to/config.json")
        >>> print(config.instrument)  # Access validated parameters
        'venus'
    """
    config_dictionary = load_json_string(config_file_path)
    my_model = Configuration.parse_obj(config_dictionary)
    return my_model


def select_file(top_folder: Optional[str] = None, next_function: Optional[callable] = None) -> None:
    """
    Launch interactive file selection for configuration files.
    
    Opens a file browser interface allowing users to select JSON configuration
    files for neutron CT reconstruction. Provides filtering to show only
    JSON files and supports callback functions for processing selected files.
    
    Args:
        top_folder: Optional starting directory for file browser. If None,
                   uses current working directory as starting point.
        next_function: Optional callback function to execute after file
                      selection. Function should accept file path as parameter.
    
    Side Effects:
        - Launches interactive file browser widget
        - Filters display to show only JSON files
        - Executes callback function with selected file path
        - Updates GUI with file selection interface
    
    Example:
        >>> def process_config(file_path):
        ...     config = loading_config_file_into_model(file_path)
        ...     print(f"Loaded config for {config.instrument}")
        >>> select_file("/data/configs", process_config)
    """
    o_file = FileFolderBrowser(working_dir=top_folder,
                               next_function=next_function)
    o_file.select_file(instruction="Select configuration file ...",
                       filters={"Json": "*.json"},
                       default_filter="Json")