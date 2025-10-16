import os
import logging
import glob
import matplotlib.pyplot as plt
from ipywidgets import interactive
from IPython.display import display
import ipywidgets as widgets
import numpy as np
from matplotlib.patches import Rectangle
from IPython.display import HTML
from typing import Optional, Tuple, List, Any
from numpy.typing import NDArray

from __code import OperatingMode, DataType, STEP3_SCRIPTS
from __code.config import DEBUG, debug_folder # , default_file_naming_convention
from __code.utilities.configuration_file import CropRegion
from __code.utilities.configuration_file import select_file, loading_config_file_into_model
from __code.utilities.logging import setup_logging
from __code.workflow.reconstruction_selection import ReconstructionSelection
from __code.utilities.files import retrieve_list_of_tif, make_or_reset_folder
from __code.utilities.create_scripts import create_sh_file, create_sh_hsnt_file
from __code.utilities.load import load_data_using_multithreading, load_list_of_tif
from __code.utilities.time import get_current_time_in_special_file_name_format
from __code.utilities.json import save_json

BASENAME_FILENAME, _ = os.path.splitext(os.path.basename(__file__))


class JsonTypeRequested:
    """
    Constants for JSON configuration file types.
    
    Defines the types of configuration files that can be generated:
    - single: Creates one JSON file for sequential reconstruction
    - multi: Creates multiple JSON files for parallel reconstruction
    - undefined: Default state before user selection
    """
    single = '1 json (reconstruction will run in sequence)'
    multi = 'multi jsons (to run reconstruction in parallel)'
    undefined = 'undefined'


class Step2SliceCcdOrTimePixImages:
    """
    A class for processing CT image slices and generating reconstruction configuration files.
    
    This class handles the second step in the CT reconstruction pipeline:
    - Loading and displaying CT projection images
    - Interactive selection of slice ranges and ROI cropping
    - Generation of JSON configuration files for reconstruction
    - Creation of shell scripts for running reconstructions
    
    Attributes:
        json_type_requested: Type of JSON configuration files to generate
        working_dir: Working directory path
        instrument: Instrument name (CCD or TimePix)
        configuration: Configuration object loaded from JSON
        data: Loaded CT projection data as numpy array
        images_path: Path to projection images
        output_config_file: Output directory for configuration files
    """

    json_type_requested: str = JsonTypeRequested.undefined
    MODE = OperatingMode.white_beam

    def __init__(self, system: Optional[Any] = None) -> None:
        """
        Initialize the Step2SliceCcdOrTimePixImages class.
        
        Args:
            system: System configuration object containing working directory and instrument info
        """

        # self.configuration = Configuration()
        self.working_dir: str = os.path.join(system.System.get_working_dir(), "shared")
        if DEBUG:
            self.working_dir = debug_folder[default_file_naming_convention][OperatingMode.white_beam][DataType.extra]

        self.instrument: str = system.System.get_instrument_selected()

        setup_logging(BASENAME_FILENAME)      

        logging.info(f"working_dir: {self.working_dir}")
        logging.info(f"instrument: {self.instrument}")
        if DEBUG:
            logging.info(f"WARNING!!!! we are running using DEBUG mode!")

    def select_config_file(self) -> None:
        """
        Display file selector widget to choose a configuration JSON file.
        
        Opens a file browser to select the configuration file from the working directory.
        The selected file will be passed to load_config_file method.
        """
        select_file(top_folder=self.working_dir,
                    next_function=self.load_config_file)

    def load_config_file(self, config_file_path: str) -> None:
        """
        Load configuration from the selected JSON file.
        
        Args:
            config_file_path: Path to the JSON configuration file
            
        Sets:
            output_config_file: Directory containing the config file
            configuration: Loaded configuration object
            images_path: Path to projection images from configuration
        """
        self.output_config_file: str = os.path.dirname(config_file_path)
        logging.info(f"configuration file loaded: {config_file_path}")
        self.configuration = loading_config_file_into_model(config_file_path)
        self.images_path: str = self.configuration.projections_pre_processing_folder
        print(f"Configuration file {os.path.basename(config_file_path)} loaded!")
        self.configuration_file_name = os.path.basename(config_file_path)

    def load_and_crop(self) -> None:
        """
        Load projection images and display cropping interface.
        
        This method calls load_images() followed by crop_settings() to provide
        a complete workflow for loading data and setting up ROI selection.
        """
        self.load_images()
        self.crop_settings()
        
    def load_images(self) -> None:
        """
        Load CT projection images from the configured path.
        
        Loads TIFF images from the images_path directory and stores them
        in self.data as a numpy array with shape (n_images, height, width).
        
        Note:
            Currently limited to first 30 images for debugging purposes.
        """
        logging.info(f"images_path: {self.images_path}")
        list_tiff: List[str] = retrieve_list_of_tif(self.images_path)
        logging.info(f"list_tiff: {list_tiff}")

        #DEBUG
        list_tiff = list_tiff[0:30]

        self.data: NDArray[np.float32] = load_list_of_tif(list_tiff, dtype=np.float32)
        # self.data = load_data_using_multithreading(list_tiff)
        # self.data = np.moveaxis(self.data, 1, 2)
        logging.info(f"loading images done!")
        logging.info(f"self.data.shape: {self.data.shape}")

    def select_range_of_slices(self) -> None:
        """
        Display interactive widget for selecting slice ranges for reconstruction.
        
        Creates an interactive plot with sliders to:
        - Select which projection image to view
        - Define top and bottom slice boundaries
        - Set number of slice ranges for parallel processing
        
        The slice ranges are visualized as colored rectangles overlaid on the image.
        Results are stored in self.display_plot_images.result.
        """
        
        left: int
        right: int 
        top: int
        bottom: int
        left, right, top, bottom = self.display_roi.result
        data: NDArray[np.float32] = self.data[:, top:bottom, left:right]

        nbr_images: int
        height: int
        width: int
        nbr_images, height, width = data.shape

        def plot_images(image_index: int, top_slice: int, bottom_slice: int, nbr: int) -> Tuple[int, int, int]:
            """
            Inner function to plot slice ranges on the selected image.
            
            Args:
                image_index: Index of the projection image to display
                top_slice: Starting slice position
                bottom_slice: Ending slice position
                nbr: Number of slice ranges to create
                
            Returns:
                Tuple of (top_slice, bottom_slice, nbr)
            """
            fig, ax = plt.subplots()
            im = ax.imshow(data[image_index], cmap='jet')
            plt.colorbar(im, ax=ax, shrink=0.5)
            
            range_size: int = int((np.abs(top_slice - bottom_slice)) / nbr)

            for _range_index in np.arange(nbr):
                _top_slice: int = top_slice + _range_index * range_size

                ax.add_patch(Rectangle((0, _top_slice), width, range_size,
                                    edgecolor='yellow',
                                    facecolor='green',
                                    fill=True,
                                    lw=2,
                                    alpha=0.3,
                                    ),
                )     

            ax.axhline(top_slice, color='red')
            ax.axhline(bottom_slice, color='red')
                  
            plt.show()

            return top_slice, bottom_slice, nbr

        self.display_plot_images = interactive(plot_images,
                                          image_index=widgets.IntSlider(min=0, max=nbr_images-1, step=1, value=0,
                                                                        layout=widgets.Layout(width='50%')),
                                          top_slice=widgets.IntSlider(min=0, max=height-1, step=1, value=0,
                                                                      layout=widgets.Layout(width='50%')),
                                          bottom_slice=widgets.IntSlider(min=0, max=height-1, 
                                                                         step=1, value=height-1,
                                                                         layout=widgets.Layout(width='50%')),
                                          nbr=widgets.IntSlider(min=1, max=30, step=1, value=1,
                                                                          layout=widgets.Layout(width='50%')))
        display(self.display_plot_images)

    def reconstruction_algorithm_selection(self) -> None:
        self.o_mode = ReconstructionSelection(parent=self)
        self.o_mode.select(default_selection=self.configuration.reconstruction_algorithm)

    def crop_settings(self) -> None:
        """
        Display interactive widget for selecting Region of Interest (ROI) cropping parameters.
        
        Creates an interactive plot with sliders to define:
        - ROI boundaries (left, right, top, bottom)
        - Visualization parameters (vmin, vmax, use_local)
        - Image selection for preview
        
        The ROI is visualized as a colored rectangle overlaid on the selected image.
        Results are stored in self.display_roi.result.
        """

        nbr_images: int
        height: int
        width: int
        nbr_images, height, width = self.data.shape

        master_vmin: float = np.min(self.data)
        master_vmax: float = np.max(self.data)

        def plot_crop(image_index: int, left: int, right: int, top: int, bottom: int, 
                     vmin: float, vmax: float, use_local: bool) -> Tuple[int, int, int, int]:
            """
            Inner function to plot ROI cropping visualization.
            
            Args:
                image_index: Index of the projection image to display
                left: Left boundary of ROI
                right: Right boundary of ROI
                top: Top boundary of ROI
                bottom: Bottom boundary of ROI
                vmin: Minimum value for colormap
                vmax: Maximum value for colormap
                use_local: Whether to use local min/max values for the image
                
            Returns:
                Tuple of (left, right, top, bottom) ROI boundaries
            """
            fig0, axs = plt.subplots(figsize=(7,7))

            if use_local:
                vmin=np.min(self.data[image_index])
                vmax = np.max(self.data[image_index])

            img = axs.imshow(self.data[image_index], vmin=vmin, vmax=vmax)
            plt.colorbar(img, ax=axs, shrink=0.5)

            width = right - left + 1
            height = bottom - top + 1

            axs.add_patch(Rectangle((left, top), width, height,
                                            edgecolor='yellow',
                                            facecolor='green',
                                            fill=True,
                                            lw=2,
                                            alpha=0.3,
                                            ),
                )     

            return left, right, top, bottom
                
        self.display_roi = interactive(plot_crop,
                                       image_index = widgets.IntSlider(min=0, max=nbr_images-1,
                                                                       value=0,
                                                                       layout=widgets.Layout(width="50%")),
                                        left=widgets.IntSlider(min=0,
                                                                max=width-1,
                                                                layout=widgets.Layout(width="50%"),
                                                                value=0),
                                        right=widgets.IntSlider(min=0,
                                                                layout=widgets.Layout(width="50%"),
                                                                max=width-1,
                                                                value=width-1),                      
                                        top=widgets.IntSlider(min=0,
                                                                layout=widgets.Layout(width="50%"),
                                                                max=height-1,
                                                                value=0),
                                        bottom=widgets.IntSlider(min=0,
                                                                layout=widgets.Layout(width="50%"),
                                                                    max=height-1,
                                                                    value=height-1),
                                        vmin=widgets.FloatSlider(min=master_vmin,
                                                                layout=widgets.Layout(width="50%"),
                                                                    max=master_vmax,
                                                                    value=master_vmin),
                                        vmax=widgets.FloatSlider(min=master_vmin,
                                                                layout=widgets.Layout(width="50%"),
                                                                    max=master_vmax,
                                                                    value=master_vmax),
                                        use_local=widgets.Checkbox(value=False),
                                        )
        display(self.display_roi)
  
    def rename_or_not_configuration_files(self) -> None:
        """
        Display option to rename configuration files in the output directory.
        
        Provides a checkbox to choose whether to rename existing configuration
        files in the output directory. If checked, all JSON files matching the
        naming convention will be renamed with a timestamp suffix.
        """
        self.rename_ui = widgets.Checkbox(
            value=False,
            description='Rename base configuration file',
            disabled=False,
            indent=False,
            layout=widgets.Layout(width='50%')
        )
        display(self.rename_ui)
        self.rename_ui.observe(self.on_rename_ui_change, names='value')

        new_name_label = widgets.Label("New basename:",
                                        layout=widgets.Layout(width='100px'))
        self.new_name_ui = widgets.Text(
            value=f"{BASENAME_FILENAME}",
            layout=widgets.Layout(width='800px'),
        )
        hori_layout = widgets.HBox([new_name_label, self.new_name_ui])
        self.new_name_ui.disabled = True
        display(hori_layout)

        old_name_label = widgets.Label("Old basename:",
                                        layout=widgets.Layout(width='100px'))
        old_name = widgets.Label(BASENAME_FILENAME)
        hori_layout2 = widgets.HBox([old_name_label, old_name])
        display(hori_layout2)

    def on_rename_ui_change(self, change: dict) -> None:
        state = change['new']
        if state:
            self.new_name_ui.disabled = False
        else:
            self.new_name_ui.disabled = True

    def select_json_type_you_want_to_create(self) -> None:
        """
        Display options for selecting the type of JSON configuration files to create.
        
        If only one slice range is selected, automatically chooses single JSON.
        If multiple slice ranges are selected, displays radio buttons to choose between:
        - Single JSON file for sequential reconstruction
        - Multiple JSON files for parallel reconstruction
        
        Updates self.json_type_requested based on selection.
        """

        _, _, nbr = self.display_plot_images.result

        if nbr == 1:
            display(HTML("1 single json (config file) will be created for the whole reconstruction"))
            self.json_type_requested = JsonTypeRequested.single

        else:

            self.json_type = widgets.RadioButtons(
                options=[JsonTypeRequested.single,
                        JsonTypeRequested.multi],
                value=JsonTypeRequested.single,
                description='Json types:',
                disabled=False,
                style={'description_width': 'initial'},
                layout=widgets.Layout(width='50%')
            )
            display(self.json_type)
    
    def export_config_file(self) -> None:
        """
        Export configuration files based on the selected JSON type.
        
        Routes to either export_single_config_file() or export_multi_config_file()
        depending on the user's selection from select_json_type_you_want_to_create().
        """
        if self.rename_ui.value:
            self.BASENAME_FILENAME = self.new_name_ui.value
            logging.info(f"Renaming base configuration files to: {self.BASENAME_FILENAME}")
        else:
            self.BASENAME_FILENAME = BASENAME_FILENAME
            logging.info(f"Keeping original base configuration file name: {self.BASENAME_FILENAME}")

        logging.info(f"{self.json_type_requested}")

        if self.json_type_requested == JsonTypeRequested.undefined:
            export_options = self.json_type.value
        else:
            export_options = JsonTypeRequested.single
        
        # update config with reconstruction algorithm selected
        self.configuration.reconstruction_algorithm = list(self.o_mode.multi_reconstruction_selection_ui.value)
        logging.info(f"reconstruction_algorithm selected: {self.configuration.reconstruction_algorithm}")

        if export_options == JsonTypeRequested.single:
            self.export_single_config_file()
        else:
            self.export_multi_config_file()
    
    def export_multi_config_file(self) -> None:
        """
        Export multiple JSON configuration files for parallel reconstruction.
        
        Creates separate configuration files for each slice range, allowing
        parallel processing of different slice ranges. Each configuration file
        includes:
        - Slice range boundaries with overlap
        - ROI cropping parameters
        - Shell scripts for HSNT execution
        
        Displays instructions for copying files to HSNT and running reconstructions.
        """
        top_slice: int
        bottom_slice: int
        nbr: int
        top_slice, bottom_slice, nbr = self.display_plot_images.result
        logging.info(f"Exporting config file:")
        logging.info(f"\ttop_slice: {top_slice}")
        logging.info(f"\tbottom_slice: {bottom_slice}")
        logging.info(f"\tnbr_of_ranges: {nbr}")

        range_size: int = int((np.abs(top_slice - bottom_slice)) / nbr)

        left: int
        right: int
        top: int 
        bottom: int
        left, right, top, bottom = self.display_roi.result
        self.configuration.crop_region = CropRegion(left=left, right=right, top=top, bottom=bottom)

        current_time: str = get_current_time_in_special_file_name_format()
        working_dir: str = self.output_config_file
        sub_folder_for_all_config_files: str = os.path.join(working_dir, f"all_config_files_{current_time}")
        make_or_reset_folder(sub_folder_for_all_config_files)

        ipts_number: str = self.configuration.ipts_number
        instrument: str = self.configuration.instrument
        hsnt_output_json_folder: str = os.path.join("/data", instrument, f"IPTS-{ipts_number}", "all_config_files")
        hsnt_output_folder: str = os.path.join("/data", instrument, f"IPTS-{ipts_number}")

        list_of_sh_hsnt_script_files: List[str] = []
        list_of_json_files: List[str] = []
        current_time: str = get_current_time_in_special_file_name_format()

        for _range_index in np.arange(nbr):

            list_slices: List[Tuple[int, int]] = []

            _top_slice: int = top_slice + _range_index * range_size
            if _top_slice > 0:
                _top_slice -= 1  # to make sure we have a 2 pixels overlap between ranges of slices

            _bottom_slice: int = top_slice + _range_index * range_size + range_size
            if _bottom_slice < (self.data.shape[1] - 1):
                _bottom_slice += 1 # to make sure we have a 2 pixels overlap between ranges of slices

            list_slices.append((_top_slice, _bottom_slice))
            self.configuration.list_of_slices_to_reconstruct = list_slices

            config_file_name: str = f"{self.BASENAME_FILENAME}_{current_time}_from_{_top_slice}_to_{_bottom_slice}.json"
            full_config_file_name: str = os.path.join(sub_folder_for_all_config_files, config_file_name)
            list_of_json_files.append(full_config_file_name)
            config_json: str = self.configuration.model_dump_json()
            save_json(full_config_file_name, json_dictionary=config_json)
            logging.info(f"config file saved: {full_config_file_name} for slice range {_top_slice} to {_bottom_slice}")

            sh_hsnt_script_name: str = create_sh_hsnt_file(configuration=self.configuration,
                                                      json_file_name=full_config_file_name, 
                                                      hstn_output_json_folder=hsnt_output_json_folder,
                                                      prefix=f"from_{_top_slice}_to_{_bottom_slice}_{current_time}")
            list_of_sh_hsnt_script_files.append(sh_hsnt_script_name)

        display(HTML(f"All config files created in <font color='blue'>{sub_folder_for_all_config_files}</font>"))
        display(HTML(f"Instructions:"))
        display(HTML(f"<b>1. Connect to hsnt</b>"))
        display(HTML(f"<b>2. Copy all config files</b> > '<font color='blue'>\'cp {sub_folder_for_all_config_files}/* {hsnt_output_folder}\'</font>"))
        display(HTML(f"<b>3. Copy scripts to run</b> > '<font color='blue'>\'cp {os.path.join(os.path.dirname(sh_hsnt_script_name), '*.sh')}/* {hsnt_output_folder}\'</font>"))
        display(HTML(f"<b>4. Run each of the .sh script</b> from hype.sns.gov</b>"))

    def export_single_config_file(self) -> None:
        """
        Export a single JSON configuration file for sequential reconstruction.
        
        Creates one configuration file containing all slice ranges for sequential
        processing. Includes:
        - All slice ranges with overlap
        - ROI cropping parameters
        - Shell script for local execution
        
        Displays the configuration file name and shell script to run.
        """

        top_slice: int
        bottom_slice: int
        nbr: int
        top_slice, bottom_slice, nbr = self.display_plot_images.result
        logging.info(f"Exporting config file:")
        logging.info(f"\ttop_slice: {top_slice}")
        logging.info(f"\tbottom_slice: {bottom_slice}")
        logging.info(f"\tnbr_of_ranges: {nbr}")

        range_size: int = int((np.abs(top_slice - bottom_slice)) / nbr)

        list_slices: List[Tuple[int, int]] = []
        for _range_index in np.arange(nbr):
            _top_slice: int = top_slice + _range_index * range_size
            if _top_slice > 0:
                _top_slice -= 1  # to make sure we have a 2 pixels overlap between ranges of slices

            _bottom_slice: int = top_slice + _range_index * range_size + range_size
            if _bottom_slice < (self.data.shape[1] - 1):
                _bottom_slice += 1 # to make sure we have a 2 pixels overlap between ranges of slices

            list_slices.append((_top_slice, _bottom_slice))
            self.configuration.list_of_slices_to_reconstruct = list_slices

        left: int
        right: int
        top: int
        bottom: int
        left, right, top, bottom = self.display_roi.result
        self.configuration.crop_region = CropRegion(left=left, right=right, top=top, bottom=bottom)

        working_dir: str = self.output_config_file
        current_time: str = get_current_time_in_special_file_name_format()
        config_file_name: str = f"{self.BASENAME_FILENAME}_{current_time}.json"
        full_config_file_name: str = os.path.join(working_dir, config_file_name)
        config_json: str = self.configuration.model_dump_json()
        save_json(full_config_file_name, json_dictionary=config_json)
        logging.info(f"config file saved: {full_config_file_name}")

        sh_file_name: str = create_sh_file(json_file_name=full_config_file_name,
                                      output_folder=working_dir)
        display(HTML(f"Next and final step. Launch the following script from the command line:"))
        display(HTML(f"config file name: <font color='blue'>{config_file_name}</font>"))
        display(HTML(f"<font color='green'>{sh_file_name}</font>"))
