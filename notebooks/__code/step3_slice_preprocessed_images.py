from email.mime import image
import os
import logging
import glob
import plotly.graph_objects as go
from ipywidgets import interactive
from IPython.display import display
import ipywidgets as widgets
import numpy as np
from IPython.display import HTML
from typing import Optional, Tuple, List, Any, Dict
from numpy.typing import NDArray

try:
    import svmbir
    HAS_SVMBIR = True
except ImportError:
    HAS_SVMBIR = False
    
from __code import OperatingMode, DataType
from __code.config import NUMBER_OF_SLICES_TO_OVERAP # , default_file_naming_convention
from __code.utilities.configuration_file import CropRegion
from __code.utilities.configuration_file import select_file
from __code.utilities.logging import setup_logging
from __code.workflow.reconstruction_selection import ReconstructionSelection
from __code.utilities.files import retrieve_list_of_tif, make_or_reset_folder
from __code.utilities.create_scripts import create_sh_file, create_sh_hsnt_file
from __code.utilities.load import load_data_using_multithreading, load_list_of_tif
from __code.utilities.time import get_current_time_in_special_file_name_format
from __code.utilities.json import save_json
from __code.workflow.checkpoint_hdf5 import CheckpointHdf5
from __code.utilities.configuration_file import Configuration

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


class Step3SlicePreprocessedImages:
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

    # json_type_requested: str = JsonTypeRequested.undefined
    MODE = OperatingMode.white_beam
    SVBMIR_MODE_FLAG = HAS_SVMBIR
    
    hdf5_input_file = None
    
    normalized_images_log = None
   
    working_dir: Dict[DataType, str] = {DataType.sample: "",
                                        DataType.ob: "",
                                        DataType.dc: "",
                                        DataType.ct_scans: "",
                                        DataType.ipts: "",
                                        DataType.top: "",
                                        DataType.nexus: "",
                                        DataType.cleaned_images: "",
                                        DataType.normalized: "",
                                        DataType.reconstructed: "",
                                        DataType.extra: "",
                                        DataType.processed: "",
                                        DataType.raw: "",
                                        DataType.hdf5: "",
                                        }

    master_3d_data_array: Dict[DataType, Optional[NDArray]] = {DataType.sample: None,  # [angle, y, x]
                                                               DataType.ob: None,
                                                               DataType.dc: None}


    def __init__(self, system: Optional[Any] = None) -> None:
        """
        Initialize the Step2SliceCcdOrTimePixImages class.
        
        Args:
            system: System configuration object containing working directory and instrument info
        """

        self.configuration = Configuration()
        setup_logging(BASENAME_FILENAME)      

        self.offline = system.System.offline
        logging.info(f"System offline mode: {self.offline}")

        top_sample_dir = system.System.get_working_dir()
        self.top_sample_dir = top_sample_dir
        self.instrument = "VENUS"  
        self.full_ipts_number = os.path.basename(top_sample_dir) 
        self.ipts_number = self.full_ipts_number.replace("IPTS-", "")
        
        self.update_all_paths()
        logging.info(f"working_dir: {self.working_dir}")
        logging.info(f"instrument: {self.instrument}")
        logging.info(f"full_ipts_number: {self.full_ipts_number}")
        logging.info(f"ipts_number: {self.ipts_number}")
        logging.info(f"offline: {self.offline}")

    def update_all_paths(self) -> None:
                
        if self.offline:
            logging.info("offline mode: Updating all paths.")
            top_sample_dir = os.path.expanduser("~")
            self.working_dir[DataType.ipts] = top_sample_dir
            self.working_dir[DataType.nexus] = top_sample_dir
            self.working_dir[DataType.processed] = top_sample_dir  
            self.working_dir[DataType.normalized] = top_sample_dir
            self.working_dir[DataType.sample] = top_sample_dir
            self.working_dir[DataType.ob] = ""
            self.working_dir[DataType.top] = top_sample_dir
      
        else:
            logging.info("online mode: Updating all paths.")

            top_sample_dir = self.top_sample_dir
            # self.working_dir[DataType.ipts] = os.path.basename(top_sample_dir)
            self.working_dir[DataType.ipts] = top_sample_dir
            self.working_dir[DataType.nexus] = os.path.join(top_sample_dir, "nexus")
            self.working_dir[DataType.processed] = os.path.join(top_sample_dir, "shared", "processed_data")       
            self.working_dir[DataType.normalized] = os.path.join(top_sample_dir, "shared", "processed_data", "normalized_data")
            self.working_dir[DataType.top] = top_sample_dir
            
        logging.info(f"Updates all paths:")
        logging.info(f"  - top_sample_dir: {top_sample_dir}")
        logging.info(f"  - sample: {self.working_dir[DataType.sample]}")
        logging.info(f"  - ob: {self.working_dir[DataType.ob]}")
        logging.info(f"  - nexus: {self.working_dir[DataType.nexus]}")  
        logging.info(f"  - processed: {self.working_dir[DataType.processed]}")
        logging.info(f"  - ipts: {self.working_dir[DataType.ipts]}")
        logging.info(f"  - top: {self.working_dir[DataType.top]}")

    def select_hdf5_file(self) -> None:
        """
        Display file selector widget to choose a configuration HDF5 file.
        
        Opens a file browser to select the configuration file from the working directory.
        The selected file will be passed to load_config_file method.
        """
        o_hdf5_file_selector = CheckpointHdf5(parent=self)
        o_hdf5_file_selector.select_input_file()
       
    # def load_hdf5_file(self, hdf5_file_path: str) -> None:
    #     """
    #     Load configuration from the selected HDF5 file.
        
    #     Args:
    #         hdf5_file_path: Path to the HDF5 configuration file
            
    #     Sets:
    #         output_config_file: Directory containing the config file
    #         configuration: Loaded configuration object
    #         images_path: Path to projection images from configuration
    #     """
    #     self.output_hdf5_file: str = os.path.dirname(hdf5_file_path)
    #     logging.info(f"Loading HDF5 {hdf5_file_path} ...")
    #     o_checkpoint = CheckpointHdf5(parent=self)
    #     o_checkpoint.load(hdf5_file_path)
    #     self.data = self.normalized_images_log
    #     print(f"HDF5 file {os.path.basename(hdf5_file_path)} loaded!")
    #     logging.info(f"Loaded from HDF5 file: {self.configuration}")
    #     self.crop_settings()

    # def load_images(self) -> None:
    #     """
    #     Load CT projection images from the configured path.
        
    #     Loads TIFF images from the images_path directory and stores them
    #     in self.data as a numpy array with shape (n_images, height, width).
        
    #     Note:
    #         Currently limited to first 30 images for debugging purposes.
    #     """
    #     logging.info(f"images_path: {self.images_path}")
    #     list_tiff: List[str] = retrieve_list_of_tif(self.images_path)
    #     logging.info(f"list_tiff: {list_tiff}")

    #     #DEBUG
    #     list_tiff = list_tiff[0:30]

    #     self.data: NDArray[np.float32] = load_list_of_tif(list_tiff, dtype=np.float32)
    #     # self.data = load_data_using_multithreading(list_tiff)
    #     # self.data = np.moveaxis(self.data, 1, 2)
    #     logging.info(f"loading images done!")
    #     logging.info(f"self.data.shape: {self.data.shape}")

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

        master_vmin: float = float(np.min(data))
        master_vmax: float = float(np.max(data))

        nbr_images: int
        height: int
        width: int
        nbr_images, height, width = data.shape

        if height > 1000 or width > 1000:
            coeff = 20
        else:
            coeff = 1

        max_slices = int(height/10)

        def plot_images(image_index: int, top_bottom: Tuple[int, int], nbr: int, vrange: Tuple[float, float]) -> Tuple[int, int, int]:
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
            
            vmin: float = vrange[0]
            vmax: float = vrange[1]

            top_slice: int = top_bottom[0]
            bottom_slice: int = top_bottom[1]

            local_top_slice = int(top_slice/coeff)
            local_bottom_slice = int(bottom_slice/coeff)

            range_size: int = int((np.abs(local_top_slice - local_bottom_slice)) / nbr)

            _data = data[image_index]
            _data = _data[::coeff, ::coeff]
            y_coords = np.arange(_data.shape[0]) * coeff
            x_coords = np.arange(_data.shape[1]) * coeff

            fig = go.Figure(data=go.Heatmap(
                z=_data,
                x=x_coords,
                y=y_coords,
                colorscale='Jet',
                zmin=vmin,
                zmax=vmax,
            ))

            for _range_index in np.arange(nbr):
                _top_slice: int = local_top_slice + _range_index * range_size

                fig.add_shape(
                    type="rect",
                    x0=0, y0=coeff * _top_slice,
                    x1=width - 1, y1=coeff * (_top_slice + range_size),
                    line=dict(color="yellow", width=2),
                    fillcolor="green",
                    opacity=0.3,
                )

            fig.add_hline(y=coeff * top_slice, line_color="red")
            fig.add_hline(y=coeff * bottom_slice, line_color="red")

            fig.update_layout(
                width=700, height=700,
                xaxis=dict(range=[0, width - 1]),
                yaxis=dict(range=[height - 1, 0], scaleanchor='x'),
            )
            fig.show()

            display(HTML(f"Each green range of slices contains {range_size*coeff} slices (with 2 slices of overlap between ranges)"))

            return top_slice, bottom_slice, nbr

        self.display_plot_images = interactive(plot_images,
                                          image_index=widgets.IntSlider(min=0, max=nbr_images-1, step=1, value=0,
                                                                        layout=widgets.Layout(width='50%')),
                                          top_bottom=widgets.IntRangeSlider(min=0, max=height-1, step=1, value=[0, height-1],
                                                                                 layout=widgets.Layout(width='50%')),
                                          nbr=widgets.IntSlider(min=1, max=max_slices, step=1, value=1,
                                                                          layout=widgets.Layout(width='50%')),
                                            vrange = widgets.FloatRangeSlider(min=master_vmin,
                                                                        layout=widgets.Layout(width="50%"),
                                                                           max=master_vmax,
                                                                           value=[master_vmin, master_vmax]),
        )
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

        self.data = self.normalized_images_log

        nbr_images: int
        height: int
        width: int
        nbr_images, height, width = self.data.shape

        if height > 1000 or width > 1000:
            coeff = 20
        else:
            coeff = 1

        master_vmin: float = np.min(self.data)
        master_vmax: float = np.max(self.data)

        def plot_crop(image_index: int, left_right: Tuple[int, int], top_bottom: Tuple[int, int], 
                     vrange: Tuple[float, float], use_local: bool) -> Tuple[int, int, int, int]:
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
            vmin: float = vrange[0]
            vmax: float = vrange[1]

            if use_local:
                vmin = np.min(self.data[image_index])
                vmax = np.max(self.data[image_index])

            left: int = left_right[0]
            right: int = left_right[1]
            top: int = top_bottom[0]
            bottom: int = top_bottom[1]

            local_left = int(left/coeff)
            local_right = int(right/coeff)
            local_top = int(top/coeff)
            local_bottom = int(bottom/coeff)

            if coeff > 1:
                _data = self.data[image_index]
                _data = _data[::coeff, ::coeff]
            else:
                _data = self.data[image_index]

            y_coords = np.arange(_data.shape[0]) * coeff
            x_coords = np.arange(_data.shape[1]) * coeff

            fig = go.Figure(data=go.Heatmap(
                z=_data,
                x=x_coords,
                y=y_coords,
                zmin=vmin,
                zmax=vmax,
            ))

            fig.add_shape(
                type="rect",
                x0=coeff * local_left, y0=coeff * local_top,
                x1=coeff * local_right, y1=coeff * local_bottom,
                line=dict(color="yellow", width=2),
                fillcolor="green",
                opacity=0.3,
            )

            fig.update_layout(
                width=700, height=700,
                xaxis=dict(range=[0, width - 1]),
                yaxis=dict(range=[height - 1, 0], scaleanchor='x'),
            )
            fig.show()

            return left, right, top, bottom
                
        self.display_roi = interactive(plot_crop,
                                       image_index = widgets.IntSlider(min=0, max=nbr_images-1,
                                                                       value=0,
                                                                       layout=widgets.Layout(width="50%")),
                                       left_right = widgets.IntRangeSlider(min=0,
                                                                           max=width-1,
                                                                           layout=widgets.Layout(width="50%"),
                                                                           value=[0, width-1]),
                                       top_bottom = widgets.IntRangeSlider(min=0,
                                                                           max=height-1,
                                                                           layout=widgets.Layout(width="50%"),
                                                                           value=[0, height-1]),
                                       vrange = widgets.FloatRangeSlider(min=master_vmin,
                                                                        layout=widgets.Layout(width="50%"),
                                                                           max=master_vmax,
                                                                           value=[master_vmin, master_vmax]),
                                        use_local=widgets.Checkbox(value=False),
                                        )
        display(self.display_roi)
  
    # def rename_or_not_configuration_files(self) -> None:
    #     """
    #     Display option to rename configuration files in the output directory.
        
    #     Provides a checkbox to choose whether to rename existing configuration
    #     files in the output directory. If checked, all JSON files matching the
    #     naming convention will be renamed with a timestamp suffix.
    #     """
    #     self.rename_ui = widgets.Checkbox(
    #         value=False,
    #         description='Rename base configuration file',
    #         disabled=False,
    #         indent=False,
    #         layout=widgets.Layout(width='50%')
    #     )
    #     display(self.rename_ui)
    #     self.rename_ui.observe(self.on_rename_ui_change, names='value')

    #     new_name_label = widgets.Label("New basename:",
    #                                     layout=widgets.Layout(width='100px'))
    #     self.new_name_ui = widgets.Text(
    #         value=f"{BASENAME_FILENAME}",
    #         layout=widgets.Layout(width='800px'),
    #     )
    #     hori_layout = widgets.HBox([new_name_label, self.new_name_ui])
    #     self.new_name_ui.disabled = True
    #     display(hori_layout)

    #     old_name_label = widgets.Label("Old basename:",
    #                                     layout=widgets.Layout(width='100px'))
    #     old_name = widgets.Label(BASENAME_FILENAME)
    #     hori_layout2 = widgets.HBox([old_name_label, old_name])
    #     display(hori_layout2)

    # def on_rename_ui_change(self, change: dict) -> None:
    #     state = change['new']
    #     if state:
    #         self.new_name_ui.disabled = False
    #     else:
    #         self.new_name_ui.disabled = True

    # def select_json_type_you_want_to_create(self) -> None:
    #     """
    #     Display options for selecting the type of JSON configuration files to create.
        
    #     If only one slice range is selected, automatically chooses single JSON.
    #     If multiple slice ranges are selected, displays radio buttons to choose between:
    #     - Single JSON file for sequential reconstruction
    #     - Multiple JSON files for parallel reconstruction
        
    #     Updates self.json_type_requested based on selection.
    #     """

    #     _, _, nbr = self.display_plot_images.result

    #     if nbr == 1:
    #         display(HTML("1 single json (config file) will be created for the whole reconstruction"))
    #         self.json_type_requested = JsonTypeRequested.single

    #     else:

    #         self.json_type = widgets.RadioButtons(
    #             options=[JsonTypeRequested.single,
    #                     JsonTypeRequested.multi],
    #             value=JsonTypeRequested.single,
    #             description='Json types:',
    #             disabled=False,
    #             style={'description_width': 'initial'},
    #             layout=widgets.Layout(width='50%')
    #         )
    #         display(self.json_type)
    
    def create_hdf5_file(self) -> None:
        hdf5_input_file = self.hdf5_input_file
        output_folder = os.path.dirname(hdf5_input_file)
        hdf5_file_name = os.path.basename(hdf5_input_file)
        new_hdf5_file_name = hdf5_file_name.replace("_step2.hdf5", f"_updated_{get_current_time_in_special_file_name_format()}_step3.hdf5")
        output_hdf5_file = os.path.join(output_folder, new_hdf5_file_name)

        logging.info(f"Creating new HDF5 file with updated configuration for step 3: {new_hdf5_file_name}")
        logging.info(f"\t- input hdf5 file: {hdf5_input_file}")
        logging.info(f"\t- output hdf5 file: {output_hdf5_file}")
    
        top_slice: int
        bottom_slice: int
        nbr: int
        top_slice, bottom_slice, nbr = self.display_plot_images.result
    
        logging.info(f"\ttop_slice: {top_slice}")
        logging.info(f"\tbottom_slice: {bottom_slice}")
        logging.info(f"\tnbr_of_ranges: {nbr}")
    
        range_size: int = int((np.abs(top_slice - bottom_slice)) / nbr)

        list_slices: List[Tuple[int, int]] = []
        for _range_index in np.arange(nbr):
            _top_slice: int = top_slice + _range_index * range_size
            if _top_slice > NUMBER_OF_SLICES_TO_OVERAP:
                _top_slice -= (NUMBER_OF_SLICES_TO_OVERAP-1)  # to make sure we have an overlap between ranges of slices

            _bottom_slice: int = top_slice + _range_index * range_size + range_size
            if _bottom_slice < (self.data.shape[1] - NUMBER_OF_SLICES_TO_OVERAP):
                _bottom_slice += (NUMBER_OF_SLICES_TO_OVERAP - 1) # to make sure we have an overlap between ranges of slices

            list_slices.append((_top_slice, _bottom_slice))
            self.configuration.list_of_slices_to_reconstruct = list_slices

        logging.info(f"list_of_slices_to_reconstruct:")
        for _range_index, (_top_slice, _bottom_slice) in enumerate(list_slices):
            logging.info(f"\tRange {_range_index}: top_slice: {_top_slice}, bottom_slice: {_bottom_slice}")

        left: int
        right: int
        top: int
        bottom: int
        left, right, top, bottom = self.display_roi.result
        self.configuration.crop_region = CropRegion(left=left, right=right, top=top, bottom=bottom)

        # we need to recalculate the center of rotation because of the cropping
        original_center_of_rotation = self.configuration.center_of_rotation
        new_center_of_rotation = original_center_of_rotation - left
        self.configuration.center_of_rotation = new_center_of_rotation
        
        # calculate the center offset
        

        # working_dir: str = self.output_config_file
        # current_time: str = get_current_time_in_special_file_name_format()
        # config_file_name: str = f"{self.BASENAME_FILENAME}_{current_time}.json"
        # full_config_file_name: str = os.path.join(working_dir, config_file_name)
        # config_json: str = self.configuration.model_dump_json()
        # save_json(full_config_file_name, json_dictionary=config_json)
        # logging.info(f"config file saved: {full_config_file_name}")

        # sh_file_name: str = create_sh_file(json_file_name=full_config_file_name,
        #                             output_folder=working_dir)
        # display(HTML(f"Next and final step. Launch the following script from the command line:"))
        # display(HTML(f"config file name: <font color='blue'>{config_file_name}</font>"))
        # display(HTML(f"<font color='green'>{sh_file_name}</font>"))

    
    
    
    
    
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
            if _top_slice > NUMBER_OF_SLICES_TO_OVERAP:
                _top_slice -= (NUMBER_OF_SLICES_TO_OVERAP-1)  # to make sure we have an overlap between ranges of slices

            _bottom_slice: int = top_slice + _range_index * range_size + range_size
            if _bottom_slice < (self.data.shape[1] - NUMBER_OF_SLICES_TO_OVERAP):
                _bottom_slice += (NUMBER_OF_SLICES_TO_OVERAP - 1) # to make sure we have an overlap between ranges of slices

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
