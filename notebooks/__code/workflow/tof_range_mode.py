"""
Time-of-Flight Range Selection and Spectral Analysis for Neutron CT.

This module provides comprehensive functionality for analyzing and selecting optimal
time-of-flight (TOF) ranges in neutron imaging experiments. It enables interactive
wavelength analysis, spectral profiling, and range selection for enhanced material
characterization and contrast optimization in computed tomography reconstruction.

Key Classes:
    - TofRangeMode: Main class for TOF range analysis and selection

Key Features:
    - Interactive TOF spectral analysis with wavelength conversion
    - Region-of-interest (ROI) selection for spectral profiling
    - Real-time wavelength calculation from TOF measurements
    - Interactive range selection with visual feedback
    - Detector offset calibration and geometry correction
    - Spectral integration for selected wavelength ranges

Scientific Background:
    Time-of-flight neutron imaging exploits the wavelength-dependent properties
    of neutron interactions with matter. Different neutron wavelengths provide
    complementary information about:
    - Crystalline structure (Bragg edges)
    - Material composition (absorption features)
    - Microstructure analysis (scattering characteristics)
    - Phase identification (wavelength-specific signatures)

Mathematical Foundations:
    TOF to wavelength conversion uses the de Broglie relation:
    λ = (h/m_n) × (TOF/L)
    where:
    - λ: neutron wavelength
    - h: Planck constant
    - m_n: neutron mass
    - TOF: time-of-flight
    - L: source-detector distance

    Spectral integration combines selected wavelength ranges:
    I_combined = ∫[λ1→λ2] I(λ) dλ

Dependencies:
    - neutronbraggedge: Specialized neutron wavelength analysis library
    - matplotlib: Scientific visualization and plotting
    - ipywidgets: Interactive control interfaces
    - numpy: Numerical computations and array operations

Author: CT Reconstruction Pipeline Team
Created: Part of neutron CT reconstruction development workflow
"""

import matplotlib.pyplot as plt
from ipywidgets import interactive
from IPython.display import display, HTML
import ipywidgets as widgets
import logging
from matplotlib.patches import Rectangle
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from numpy.typing import NDArray

from neutronbraggedge.experiment_handler.tof import TOF
from neutronbraggedge.experiment_handler.experiment import Experiment

from __code.parent import Parent
from __code import LAMBDA, ANGSTROMS, NBR_TOF_RANGES
from __code.config import DISTANCE_SOURCE_DETECTOR


class TofRangeMode(Parent):
    """
    Handles time-of-flight range selection and spectral analysis for neutron CT.
    
    This class provides comprehensive functionality for analyzing neutron TOF spectra,
    selecting optimal wavelength ranges, and performing spectral integration for
    enhanced material characterization in computed tomography reconstruction.
    
    Key Features:
        - TOF spectrum loading and wavelength conversion
        - Interactive spectral profiling with ROI selection
        - Real-time wavelength calculation and calibration
        - Visual range selection with spectral overlays
        - Detector geometry correction and offset calibration
        - Spectral integration for selected wavelength bands
        
    Attributes:
        tof_array_s: Time-of-flight array in seconds
        lambda_array_angstroms: Corresponding wavelength array in Angstroms
        y_axis: Spectral intensity data for selected ROI
        y_log_scale: Flag for logarithmic intensity scale
        
    Methods:
        run(): Execute complete TOF analysis workflow
        retrieve_tof_array(): Load TOF data from spectra file
        calculate_lambda_array(): Convert TOF to wavelength
        display_widgets(): Create instrument parameter interface
        plot(): Interactive spectral analysis with ROI selection
        select_tof_range(): Wavelength range selection interface
        combine_tof_mode_data(): Apply spectral integration to data
    """
    
    tof_array_s: Optional[NDArray[np.floating]] = None
    lambda_array_angstroms: Optional[NDArray[np.floating]] = None

    def run(self) -> None:
        """
        Execute the complete TOF range analysis workflow.
        
        Orchestrates the full process of TOF spectral analysis including
        data loading, wavelength conversion, instrument parameter display,
        and interactive spectral plotting for range selection.
        
        Returns:
            None: Creates interactive analysis interface
            
        Side Effects:
            - Loads TOF array from spectra file
            - Creates instrument parameter widgets
            - Displays interactive spectral analysis plot
            
        Workflow:
            1. Retrieve TOF array from experimental data
            2. Display instrument configuration parameters
            3. Create interactive spectral analysis interface
            4. Enable wavelength range selection
        """
        
        self.retrieve_tof_array()
        self.display_widgets()
        # self.calculate_lambda_array()
        self.plot()

    def retrieve_tof_array(self) -> None:
        """
        Load time-of-flight array from experimental spectra file.
        
        Reads the TOF array from the neutron spectra file using the neutronbraggedge
        library. This array contains the time channels corresponding to neutron
        flight times from source to detector.
        
        Returns:
            None: Sets self.tof_array_s with loaded TOF data
            
        Side Effects:
            - Loads TOF data from self.parent.spectra_file_full_path
            - Sets self.tof_array_s attribute
            - Logs file path and TOF array information
            
        Notes:
            - Uses neutronbraggedge.TOF handler for data loading
            - TOF array is in seconds
            - Essential for subsequent wavelength calculations
        """
        logging.info(f"retrieving tof array:")
        spectra_file_full_path: str = self.parent.spectra_file_full_path
        logging.info(f"\t{spectra_file_full_path = }")
        _tof_handler: TOF = TOF(filename=spectra_file_full_path)
        self.tof_array_s = _tof_handler.tof_array
        logging.info(f"\ttof_array_s: {self.tof_array_s}")

    def calculate_lambda_array(self, detector_offset: float = 9000) -> None:
        """
        Convert time-of-flight array to neutron wavelength array.
        
        Calculates neutron wavelengths from TOF measurements using the de Broglie
        relation and experimental geometry. Accounts for detector offset and
        source-detector distance for accurate wavelength determination.
        
        Args:
            detector_offset: Detector timing offset in microseconds (default: 9000)
            
        Returns:
            None: Sets self.lambda_array_angstroms with calculated wavelengths
            
        Side Effects:
            - Creates Experiment object with TOF and geometry parameters
            - Calculates wavelength array using neutronbraggedge library
            - Converts wavelengths to Angstroms (× 1e10 from meters)
            - Sets self.lambda_array_angstroms attribute
            
        Mathematical Background:
            λ = (h/m_n) × (TOF - offset)/L
            where h/m_n = 3.956 × 10^-7 m²/s (neutron de Broglie constant)
        """
        _exp: Experiment = Experiment(tof=self.tof_array_s,
            distance_source_detector_m=DISTANCE_SOURCE_DETECTOR,
            detector_offset_micros=detector_offset,
                )
        _lambda_array: NDArray[np.floating] = _exp.lambda_array
        self.lambda_array_angstroms = _lambda_array * 1e10

    def display_widgets(self) -> None:
        """
        Display instrument configuration parameters.
        
        Creates and displays widgets showing key experimental parameters
        including source-detector distance. Provides visual confirmation
        of instrument geometry used in wavelength calculations.
        
        Returns:
            None: Displays widget interface
            
        Side Effects:
            - Creates and displays parameter labels
            - Shows source-detector distance configuration
            
        Notes:
            - Currently displays read-only parameters
            - Commented code includes interactive detector offset controls
            - Part of instrument configuration validation
        """
        distance_uis: widgets.HBox = widgets.HBox([widgets.Label("distance source detector (m):",
                                               layout=widgets.Layout(width="max-contents")),
                                widgets.Label(f"{DISTANCE_SOURCE_DETECTOR}"),
        ])
        display(distance_uis)

    #     detector_offset_uis = widgets.HBox([widgets.Label("Detector offset (micros)",
    #                                            layout=widgets.Layout(width="max-contents")),
    #                                         widgets.FloatText(value=9000,
    #                                                           )])
    #     display(widgets.VBox([distance_uis, detector_offset_uis]))
    #     self.detector_offset_text_ui = detector_offset_uis.children[1]
    #     self.detector_offset_text_ui.observe(self.detector_value_changed, names='value')


    # def detector_value_changed(self, value):
    #     detector_offset = value['new']
    #     self.calculate_lambda_array(detector_offset=detector_offset)
    #     self.plot()

    def plot(self) -> None:
        """
        Create interactive spectral analysis interface with ROI selection.
        
        Generates an interactive plotting interface that allows users to select
        regions of interest on integrated projection data and view corresponding
        neutron spectra. Enables real-time detector offset adjustment and
        provides both linear and logarithmic scale options.
        
        Returns:
            None: Creates and displays interactive plot interface
            
        Side Effects:
            - Creates self.plot_profile interactive widget
            - Displays combined projection/spectrum visualization
            - Enables ROI selection with rectangular overlay
            - Provides real-time spectral analysis feedback
            
        Interactive Controls:
            - Detector offset: Timing calibration adjustment
            - ROI position and size: Spatial region selection
            - Position adjustment: Fine-tuning ROI location
            - Log scale: Spectral intensity scaling option
            
        Features:
            - Integrated projection display with ROI overlay
            - Real-time spectral plotting for selected region
            - Wavelength-dependent analysis capabilities
            - Interactive parameter adjustment
        """
        
        data_3d: NDArray[np.floating] = self.parent.data_3d_of_all_projections_merged
        integrated_data_3d: NDArray[np.floating] = np.sum(data_3d, axis=0)

        width: int = self.parent.image_size['width']
        height: int = self.parent.image_size['height']

        def display_projection_and_profile(det_offset: float = 9000, left: int = 100, top: int = 100, width: int = 100, height: int = 100, 
                            x_move: int = 0, y_move: int = 0, y_log_scale: bool = True) -> None:
            """Display projection with ROI and corresponding spectral profile."""

            self.calculate_lambda_array(detector_offset=det_offset)

            fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

            im0 = axs[0].imshow(integrated_data_3d)
            # plt.colorbar(im0, ax=axs[0], shrink=0.8)

            new_left: int = left + x_move
            new_top: int = top + y_move

            axs[0].add_patch(Rectangle((new_left, new_top), width, height,
                                        edgecolor='yellow',
                                        facecolor='green',
                                        fill=True,
                                        lw=2,
                                        alpha=0.3,
                                        ),
            )

            counts_of_region: NDArray[np.floating] = np.mean(np.mean(data_3d[:, new_top: new_top+height+1, 
                                                       new_left: new_left+width+1], 
                                                       axis=1), axis=1)
            self.y_axis = counts_of_region
            self.y_log_scale = y_log_scale

            if y_log_scale:
                axs[1].semilogy(self.lambda_array_angstroms, counts_of_region)
            else:
                axs[1].plot(self.lambda_array_angstroms, counts_of_region)           
            axs[1].set_xlabel(f"{LAMBDA} ({ANGSTROMS})")
            axs[1].set_ylabel(f"Mean counts of region selected")

            plt.tight_layout()
            plt.show()

        self.plot_profile = interactive(display_projection_and_profile,
                                        det_offset=widgets.FloatText(value=9000),
                                        left=widgets.IntSlider(min=0, max=width-1, value=100),
                                        top=widgets.IntSlider(min=0, max=height-1, value=100),
                                        width=widgets.IntSlider(min=0, max=width, value=100),
                                        height=widgets.IntSlider(min=0, max=height, value=100),
                                        x_move=widgets.IntSlider(min=-width, max=width, value=0),
                                        y_move=widgets.IntSlider(min=-height, max=height, value=0),
                                        y_log_scale=widgets.Checkbox(value=True),
                                        )
        display(self.plot_profile)

    def select_tof_range(self) -> None:
        """
        Create interactive interface for wavelength range selection.
        
        Provides an interactive tool for selecting specific wavelength ranges
        from the neutron spectrum. Displays the spectrum with adjustable range
        markers and annotations showing selected wavelength bounds.
        
        Returns:
            None: Creates interactive range selection interface
            
        Side Effects:
            - Creates self.plot_tof_profile interactive widget
            - Displays spectrum with selectable range overlay
            - Shows wavelength annotations for selected bounds
            - Stores selection results for subsequent processing
            
        Interactive Features:
            - Left/right wavelength bound sliders
            - Real-time wavelength value display
            - Visual range highlighting with transparency
            - Spectrum display with logarithmic/linear scaling
            
        Notes:
            - Uses y_axis data from previous ROI selection
            - Wavelength bounds shown in Angstroms
            - Selection stored in widget.result for data processing
        """

        max_y: float = np.max(self.y_axis)

        def display_profile(left_tof: int, right_tof: int) -> Tuple[int, int]:
            """Display spectrum with selected wavelength range overlay."""

            fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10,10))

            x_axis: NDArray[np.floating] = self.lambda_array_angstroms
            y_axis: NDArray[np.floating] = self.y_axis

            if self.y_log_scale:
                axs.semilogy(x_axis, y_axis)
            else:
                axs.plot(x_axis, y_axis)

            axs.annotate(f"{self.lambda_array_angstroms[left_tof]:0.3f} {ANGSTROMS}",
                        (self.lambda_array_angstroms[left_tof], max_y),
                        color='red',
                        horizontalalignment='right',
                        )

            axs.annotate(f"{self.lambda_array_angstroms[right_tof]:0.3f} {ANGSTROMS}",
                        (self.lambda_array_angstroms[right_tof], max_y),
                        color='red',
                        horizontalalignment='left',
                        )


            axs.axvspan(self.lambda_array_angstroms[left_tof], 
                        self.lambda_array_angstroms[right_tof], 
                        alpha=0.5,
                        linestyle="--",
                        edgecolor='green')
            plt.show()

            return left_tof, right_tof

        self.plot_tof_profile = interactive(display_profile,
                                            left_tof=widgets.IntSlider(min=0,
                                                                        max=len(self.lambda_array_angstroms)-1,
                                                                        value=0),
                                            right_tof=widgets.IntSlider(min=0,
                                                                        max=len(self.lambda_array_angstroms)-1,
                                                                        value=len(self.lambda_array_angstroms)-1),
        )
        display(self.plot_tof_profile)

    def combine_tof_mode_data(self) -> None:
        """
        Apply spectral integration to selected wavelength range.
        
        Combines TOF data within the user-selected wavelength range by averaging
        spectral channels. This process reduces the data to a single integrated
        image while preserving the spectral characteristics of the selected range.
        
        Returns:
            None: Modifies parent master_3d_data_array with integrated data
            
        Side Effects:
            - Retrieves wavelength range from self.plot_tof_profile.result
            - Updates self.parent.configuration.range_of_tof_to_combine
            - Modifies self.parent.master_3d_data_array with averaged data
            - Logs integration parameters and array shape changes
            
        Process:
            1. Extract selected wavelength range indices
            2. Log wavelength bounds and integration parameters
            3. Apply spectral averaging to each data type
            4. Update data arrays with integrated results
            
        Mathematical Operation:
            I_integrated = mean(I[λ_left:λ_right], axis=wavelength)
            
        Notes:
            - Preserves spatial dimensions while reducing spectral dimension
            - Applied to all data types (sample, OB, DC)
            - Essential for wavelength-specific reconstruction
        """
        
        logging.info(f"combining in tof mode:")
        left_tof_index: int
        right_tof_index: int
        left_tof_index, right_tof_index = self.plot_tof_profile.result
        logging.info(f"\tfrom index {left_tof_index} ({self.lambda_array_angstroms[left_tof_index]:0.3f} {ANGSTROMS})")
        logging.info(f"\tto index {right_tof_index} ({self.lambda_array_angstroms[right_tof_index]:0.3f} {ANGSTROMS})")
        self.parent.configuration.range_of_tof_to_combine = [(left_tof_index, right_tof_index)]

        master_3d_data_array: Dict[Any, NDArray[np.floating]] = self.parent.master_3d_data_array

        for _data_type in master_3d_data_array.keys():
            logging.info(f"\tbefore: {np.shape(master_3d_data_array[_data_type]) = }")
            new_master_3d_data_array: List[NDArray[np.floating]] = []
            for _data in master_3d_data_array[_data_type]:
                new_master_3d_data_array.append(np.mean(_data[left_tof_index:right_tof_index+1, :, :], axis=0))
            master_3d_data_array[_data_type] = np.array(new_master_3d_data_array)
            logging.info(f"\tafter: {np.shape(master_3d_data_array[_data_type]) = }")
        self.parent.master_3d_data_array = master_3d_data_array


    # def select_multi_tof_range(self):

    #     self.list_of_tof_ranges_ui = []
    #     list_default_use_it = [False for i in range(NBR_TOF_RANGES)]
    #     list_default_use_it[0] = True
    #     max_y = np.max(self.y_axis)

    #     for i in range(NBR_TOF_RANGES):

    #         def display_profile(use_it, left_tof, right_tof):

    #             fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10,10))

    #             axs.set_title(f"TOF range {i}/{NBR_TOF_RANGES-1}")

    #             x_axis = self.lambda_array_angstroms
    #             y_axis = self.y_axis

    #             if self.y_log_scale:
    #                 axs.semilogy(x_axis, y_axis)
    #             else:
    #                 axs.plot(x_axis, y_axis)

    #             axs.annotate(f"{self.lambda_array_angstroms[left_tof]:0.3f} {ANGSTROMS}",
    #                         (self.lambda_array_angstroms[left_tof], max_y),
    #                         color='red',
    #                         horizontalalignment='right',
    #                         )

    #             axs.annotate(f"{self.lambda_array_angstroms[right_tof]:0.3f} {ANGSTROMS}",
    #                         (self.lambda_array_angstroms[right_tof], max_y),
    #                         color='red',
    #                         horizontalalignment='left',
    #                         )


    #             axs.axvspan(self.lambda_array_angstroms[left_tof], 
    #                         self.lambda_array_angstroms[right_tof], 
    #                         alpha=0.5,
    #                         linestyle="--",
    #                         edgecolor='green')
    #             plt.show()

    #             return use_it, left_tof, right_tof

    #         _plot_tof_profile = interactive(display_profile,
    #                                             use_it=widgets.Checkbox(value=list_default_use_it[i]),
    #                                             left_tof=widgets.IntSlider(min=0,
    #                                                                         max=len(self.lambda_array_angstroms)-1,
    #                                                                         value=0),
    #                                             right_tof=widgets.IntSlider(min=0,
    #                                                                         max=len(self.lambda_array_angstroms)-1,
    #                                                                         value=len(self.lambda_array_angstroms)-1),
    #         )
    #         display(_plot_tof_profile)
    #         display(HTML("<hr style='border-bottom: dotted 1px;'>"))

    #     self.list_of_tof_ranges_ui.append(_plot_tof_profile)
 
    # def combine_tof_mode_data(self):
        
    #     logging.info(f"combining in tof mode:")

    #     logging.info(f"\tlooking at {NBR_TOF_RANGES} potential TOF ranges!")
    #     for _index, _widgets in enumerate(self.list_of_tof_ranges_ui):
    #         _use_it, _left_index, _right_index = _widgets.result
    #         logging.info(f"\ttof range index {_index}:")
    #         logging.info(f"\t\t{_use_it = }")
    #         logging.info(f"\t\t{_left_index = }")
    #         logging.info(f"\t\t{_right_index = }")




        # left_tof_index, right_tof_index = self.plot_tof_profile.result
        # logging.info(f"\tfrom index {left_tof_index} ({self.lambda_array_angstroms[left_tof_index]:0.3f} {ANGSTROMS})")
        # logging.info(f"\tto index {right_tof_index} ({self.lambda_array_angstroms[right_tof_index]:0.3f} {ANGSTROMS})")

        # master_3d_data_array = self.parent.master_3d_data_array

        # for _data_type in master_3d_data_array.keys():
        #     logging.info(f"\tbefore: {np.shape(master_3d_data_array[_data_type]) = }")
        #     new_master_3d_data_array = []
        #     for _data in master_3d_data_array[_data_type]:
        #         new_master_3d_data_array.append(np.mean(_data[left_tof_index:right_tof_index+1, :, :], axis=0))
        #     master_3d_data_array[_data_type] = np.array(new_master_3d_data_array)
        #     logging.info(f"\tafter: {np.shape(master_3d_data_array[_data_type]) = }")
        # self.parent.master_3d_data_array = master_3d_data_array
