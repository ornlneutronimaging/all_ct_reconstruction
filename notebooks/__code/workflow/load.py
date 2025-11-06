"""
Data Loading Utilities for CT Reconstruction Pipeline.

This module provides comprehensive data loading functionality for computed tomography
reconstruction workflows. It handles loading sample, open beam, and dark current data
from various file formats with multi-threading support for performance optimization.

Key Classes:
    - Load: Main class for CT data loading and organization

Key Features:
    - Multi-threaded TIFF file loading for performance
    - Support for sample, open beam, and dark current data types
    - Interactive folder selection with file browser widgets
    - Automatic file format detection and validation
    - Run-based data organization and filtering
    - Progress tracking with visual feedback
    - Memory-efficient data handling for large datasets

Dependencies:
    - PIL: Image file reading and processing
    - tqdm: Progress bar visualization
    - matplotlib: Data visualization and plotting
    - IPython: Jupyter notebook widget integration

Author: CT Reconstruction Pipeline Team
Created: Part of CT reconstruction development workflow
"""

from typing import Optional, List, Dict, Any, Union, Tuple
import glob
import os
import numpy as np
from numpy.typing import NDArray
import logging
from loguru import logger as loguru_logging
from tqdm import tqdm
from IPython.display import display, HTML
import ipywidgets as widgets
from PIL import Image
import random
import matplotlib.pyplot as plt
from ipywidgets import interactive

from __code import DataType, Run, OperatingMode
from __code.parent import Parent
from __code.utilities.file_folder_browser import FileFolderBrowser
from __code.utilities.load import load_data_using_multithreading, load_list_of_tif
from __code.utilities.files import retrieve_list_of_tif
from __code.utilities.math import farthest_point_sampling
from __code.config import DEBUG, DEFAULT_NAMING_CONVENTION_INDICES, PERCENTAGE_OF_DATA_TO_USE_FOR_RECONSTRUCTION, debug_folder, default_detector_type
from __code.utilities.exceptions import MetadataError


class HowToRetrieveAngleValue:
    define_naming_convention = "Define naming convention"
    use_angle_value_from_metadata_file = "Use angle value from metadata file"
    import_list_from_ascii_file = "Import list from ASCII file"


class Load(Parent):
    """
    Data loading and organization for CT reconstruction pipeline.
    
    This class provides comprehensive functionality for loading and organizing
    computed tomography data including sample projections, open beam images,
    and dark current measurements. It supports multi-threaded loading for
    performance and provides interactive interfaces for data selection.
    
    Inherits from Parent class which provides access to reconstruction pipeline
    state, working directories, and configuration parameters.
    
    Key Features:
        - Multi-threaded TIFF file loading with progress tracking
        - Interactive folder selection with file browser widgets
        - Run-based data organization and filtering
        - Automatic file format detection and validation
        - Memory-efficient handling of large datasets
        - Support for various CT data types (sample, OB, DC)
    
    Attributes
    ----------
    list_of_runs_to_use : Dict[DataType, List]
        Dictionary mapping data types to lists of run numbers to process
    
    Examples
    --------
    >>> loader = Load(parent=parent_instance)
    >>> loader.select_folder(DataType.sample)
    >>> loader.load_data()
    >>> sample_data = loader.get_sample_data()
    """

    list_of_runs_to_use: Dict[DataType, List[int]] = {DataType.sample: [],
                                                      DataType.ob: [],
    }

    def select_folder(self, data_type: DataType = DataType.sample, 
                     multiple_flag: bool = False, 
                     output_flag: bool = False) -> None:
        """
        Interactive folder selection for CT data loading.
        
        Provides a file browser widget for selecting directories containing
        CT data files. Supports selection of sample, open beam, or dark current
        data directories with validation and preview capabilities.
        
        Parameters
        ----------
        data_type : DataType, default=DataType.sample
            Type of CT data to load (sample, ob, dc)
        multiple_flag : bool, default=False
            Whether to allow multiple folder selection
        output_flag : bool, default=False
            Whether this is for output directory selection
            
        Notes
        -----
        - Creates interactive file browser widget
        - Validates selected directories for CT data files
        - Updates parent working directory configuration
        - Provides preview of selected data files
        """
        self.parent.current_data_type = data_type
        self.data_type = data_type
        if data_type in [DataType.reconstructed, DataType.extra]:
            working_dir = self.parent.working_dir[DataType.processed]
        else:
            working_dir = self.parent.working_dir[data_type]

        logging.info(f"Selecting folder for {data_type} ...")

        if DEBUG:
            self.data_selected(debug_folder[default_detector_type][self.parent.MODE][data_type])
            logging.info(f"{default_detector_type = }")
            self.parent.working_dir[DataType.nexus] = debug_folder[default_detector_type][self.parent.MODE].get(DataType.nexus, None)
            logging.info(f"DEBUG MODE: {data_type} folder selected: {debug_folder[default_detector_type][self.parent.MODE][data_type]}")
            _list_sep = self.parent.working_dir[DataType.sample].split(os.sep)
            # facility = _list_sep[0]
            instrument = _list_sep[2]
            ipts_number = _list_sep[3]
            _, ipts = ipts_number.split("-")
            self.parent.instrument = instrument
            self.parent.ipts_number = int(ipts)
            return

        logging.info(f"\t{working_dir = }")
        if not os.path.exists(working_dir):
            logging.warning(f"Working directory {working_dir} does not exist!")
            while (not os.path.exists(working_dir)):
                print(f"Working directory {working_dir} does not exist, trying to go up one level ...")
                working_dir = os.path.dirname(working_dir)
        else:
            logging.info(f"\tWorking directory exists.")
                
            # working_dir = os.path.abspath(os.path.expanduser("~"))

        logging.info(f"ipts_folder: {self.parent.working_dir[DataType.ipts]}")

        display(HTML(f"<u>REMINDER:</u>"))
        display(HTML(f"- Sample folder: <b>{os.path.basename(self.parent.working_dir[DataType.sample])}</b>"))

        try:

            if output_flag:
                o_file_browser = FileFolderBrowser(working_dir=working_dir,
                                                ipts_folder=self.parent.working_dir[DataType.ipts],
                                                next_function=self.data_selected)
                o_file_browser.select_output_folder_with_new(instruction=f"Select Top Folder of {data_type}")
            else:
                o_file_browser = FileFolderBrowser(working_dir=working_dir,
                                                next_function=self.data_selected)
                o_file_browser.select_input_folder(instruction=f"Select Top Folder of {data_type}",
                                                multiple_flag=multiple_flag)

        except NotADirectoryError as e:
            logging.error(f"Error selecting folder: {e}. You probably forgot to select your IPTS in the first cell!")
            display(HTML(f"<font color='red'><b>ERROR</b>: You probably forgot to select your IPTS in the first cell!</font>"))
            return

    def select_images(self, data_type=DataType.ob):
        self.parent.current_data_type = data_type
        self.data_type = data_type
        if data_type in [DataType.reconstructed, DataType.extra]:
            working_dir = self.parent.working_dir[DataType.processed]
        else:
            working_dir = self.parent.working_dir[data_type]

        if DEBUG:
            working_dir = debug_folder[default_detector_type][self.parent.MODE][data_type]
            if not os.path.exists(working_dir):
                return
            #list_images = glob.glob(os.path.join(working_dir, "*_0045_*.tif*"))
            list_images = glob.glob(os.path.join(working_dir, "*.tif*"))
            list_images.sort()
            self.images_selected(list_images=list_images)
            return

        if not os.path.exists(working_dir):
            while (not os.path.exists(working_dir)):
                print(f"Working directory {working_dir} does not exist, trying to go up one level ...")
                working_dir = os.path.dirname(working_dir)
                
            # working_dir = os.path.abspath(os.path.expanduser("~"))

        o_file_browser = FileFolderBrowser(working_dir=working_dir,
                                           next_function=self.images_selected)
        o_file_browser.select_images_with_search(instruction="Select all images ...",
                                                 filters={"TIFF": "*.tif*"})
    
    def images_selected(self, list_images):
        if list_images:
            print(f"top {self.data_type} folder is: {os.path.dirname(list_images[0])}")
        else:
            print(f"no {self.data_type} selected !")
            return
        
        list_images.sort()
        logging.info(f"{len(list_images)} {self.data_type} images have been selected!")
        display(HTML(f"<span style='color:green'>{len(list_images)} images have been selected as {self.data_type}!</span>"))
        self.parent.list_of_images[self.data_type] = list_images

    def use_all_or_fraction(self):
       
        self.select_widget = widgets.RadioButtons(options=['Use all data for reconstruction', 
                                                      'Use a fraction of the data for reconstruction'],
                                              description='',
                                              disabled=False,
                                              layout=widgets.Layout(width='80%'),
                                              value='Use all data for reconstruction',
                                              style={'description_width': 'initial'}
                                              )
        
        display(self.select_widget)

    def select_percentage_of_data_to_use(self):

        if self.select_widget.value == 'Use all data for reconstruction':
            self.parent.percentage_of_data_to_use_for_reconstruction = 100
            self.determine_projections_angles_to_use(percentage_to_use=100)
            display(HTML(f"All images will be used for the reconstruction: {self.parent.temp_nbr_of_images_will_be_used} images"))
            return
        
        def display_data_to_use(slider_index, preview=False):
        
            logging.info(f"Percentage of data to use for reconstruction: {slider_index}%")
            display(HTML(f"Processing how many images will be used for the reconstruction ..."))
            self.determine_projections_angles_to_use(percentage_to_use=slider_index)
            display(HTML(f"{self.parent.temp_nbr_of_images_will_be_used} images will be used for the reconstruction"))
    
            logging.info("Visualizing angles selected for the projections.")
            full_list_of_angles_rad = self.parent.full_list_of_angles_rad
            logging.info(f"\tFull list of angles (radians): {full_list_of_angles_rad}")
            logging.info(f"\tSelected angles (radians): {self.parent.temp_final_list_of_angles_rad}")
            
            if preview:
                fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={'projection': 'polar'})
                ax.plot(full_list_of_angles_rad, np.ones(len(full_list_of_angles_rad))*2, marker='o', linestyle='None', color='blue', markersize=5, label='All Angles')
                ax.set_rmax(2)
                ax.set_rticks([0.5, 1, 1.5])
                selected_list_of_angles_rad = self.parent.temp_final_list_of_angles_rad
                ax.plot(selected_list_of_angles_rad, np.ones(len(selected_list_of_angles_rad))*2, marker='o', linestyle='None', color='green', markersize=10, label='Selected Angles')
                ax.legend()
            
            return slider_index
        
        self.percentage_of_data_to_use_widget = interactive(display_data_to_use,
                                                            slider_index=widgets.IntSlider(value=PERCENTAGE_OF_DATA_TO_USE_FOR_RECONSTRUCTION,
                                                                                            min=5,
                                                                                            max=100,
                                                                                            step=1,
                                                                                            continuous_update=False,
                                                                                            layout=widgets.Layout(width='100%')),
                                                               preview=widgets.Checkbox(value=False,),
                                                               )

        display(self.percentage_of_data_to_use_widget)

    # def on_percentage_to_use_change(self, change):
    #     new_value = change['new']
    #     list_tiff = self.parent.list_of_images[DataType.sample]
    #     nbr_images = int(new_value / 100 * len(list_tiff))
    #     self.number_of_images_to_use.value = f"{nbr_images} images will be used for the reconstruction"
    #     self.determine_projections_angles_to_use()
    #     self.visualize_angles_selected()

    def how_to_retrieve_angle_value(self):
        self.how_to_retrieve_angle_value_widget = widgets.RadioButtons(
            options=[HowToRetrieveAngleValue.define_naming_convention, 
                     HowToRetrieveAngleValue.use_angle_value_from_metadata_file,
                     HowToRetrieveAngleValue.import_list_from_ascii_file],
            description='How to retrieve angle value:',
            disabled=False,
            style={'description_width': 'initial'}
        )
        display(self.how_to_retrieve_angle_value_widget)

    def retrieve_angle_value(self):
        """Retrieve angle value from the images file name or metadata file"""
        if self.how_to_retrieve_angle_value_widget.value == HowToRetrieveAngleValue.define_naming_convention:
            self.parent.retrieve_angle_value_from_metadata = False
            self.define_naming_convention()
        elif self.how_to_retrieve_angle_value_widget.value == HowToRetrieveAngleValue.use_angle_value_from_metadata_file:
            self.parent.retrieve_angle_value_from_metadata = True
            display(widgets.HTML("<font color='green'><b>INFO</b>: Angle value will be retrieved from the metadata file.</font>"))
        elif self.how_to_retrieve_angle_value_widget.value == HowToRetrieveAngleValue.import_list_from_ascii_file:
            logging.info(f"Importing list of angles from ASCII file ...")
            self.parent.retrieve_angle_value_from_metadata = False
            self.import_list_from_ascii_file()
        else:
            raise ValueError("Invalid option selected for angle value retrieval.")

    def testing_angle_values(self):
        list_angle_file = self.list_angle_file

        if not list_angle_file:
            logging.error("No file selected!")
            print("No file selected!")
            return

        if not os.path.exists(list_angle_file):
            logging.error(f"File {list_angle_file} does not exist!")
            print(f"File {list_angle_file} does not exist!")
            return
        
        try:
            list_of_angles = np.loadtxt(list_angle_file, dtype=float)
        except Exception as e:
            logging.error(f"Error loading ASCII file {list_angle_file}: {e}")
            display(widgets.HTML(f"<font color='red'><b>ERROR</b>: Could not load ASCII file {list_angle_file}. Ensure the file is formatted correctly.</font>"))
            return
        
        if len(list_of_angles) != len(self.parent.list_of_images[DataType.sample]):
            logging.error(f"Number of angles in {list_angle_file} does not match number of images.")
            display(widgets.HTML(f"<font color='red'><b>ERROR</b>: Number of angles in {list_angle_file} does not match number of images.</font>"))
            return

        display(widgets.HTML(f"<font color='green'><b>Number of angles from ASCII file and number of images match!</b></font>"))

        # make sure the angles values are within 0-360 degrees
        list_of_angles = np.array(list_of_angles) % 360

        self.parent.final_list_of_angles = list_of_angles.tolist()
        self.parent.final_list_of_angles_rad = np.deg2rad(list_of_angles).tolist()
        
        logging.info(f"Angle values imported from ASCII file: {self.parent.final_list_of_angles}")
        display(widgets.HTML(f"<font color='green'><b>INFO</b>: Angle values imported from ASCII file successfully.</font>"))

    def import_list_from_ascii_file(self):
        filters = {"Text files": "*.txt",
                   "All files": "*.*"}
        o_file_browser = FileFolderBrowser(working_dir=os.path.dirname(self.parent.working_dir[DataType.sample]),
                                           next_function=self.ascii_file_selected)
        o_file_browser.select_file(instruction="Select ASCII file containing list of angles ...",
                                   filters=filters,
                                   default_filter="Text files")
        
    def ascii_file_selected(self, list_angle_file):
        logging.info(f"ASCII file selected: {list_angle_file}")
        self.list_angle_file = list_angle_file

    def define_naming_convention(self):
        number_of_images = len(self.parent.list_of_images[DataType.sample])

        if number_of_images == 0:
            display(widgets.HTML("<font color='red'><b>ERROR</b>: Check the sample folder you selected! Folder does not contain any images. </font>"))
            return

        # random number between 0 and number_of_images
        random_index = np.random.randint(0, number_of_images)

        # random file name
        first_file = self.parent.list_of_images[DataType.sample][random_index]

        # remove extension
        first_file = os.path.splitext(first_file)[0]

        self.selected_file_label = widgets.Label(os.path.basename(first_file))

        first_hori_box = widgets.HBox([widgets.HTML("<b>File name:</b>"),
                                       self.selected_file_label])
        display(first_hori_box)

        list_splits = os.path.basename(first_file).split("_")
        self.list_checkboxes = []
        global_list_verti_box = []

        display(widgets.HTML("<b>Check the 2 fields to use to determine the angle value (degree.minutes)!</b>"))

        for _index, _split in enumerate(list_splits):

            if self.parent.list_states_checkbox is None:
                _state = False
                if DEBUG:
                    if _index in DEFAULT_NAMING_CONVENTION_INDICES:
                        _state = True
            else:
                _state = self.parent.list_states_checkbox[_index]

            _check = widgets.Checkbox(value=_state,
                                      description=f"{_split}")
            self.list_checkboxes.append(_check)
            global_list_verti_box.append(_check)
            _check.observe(self.on_check_change, names='value')

        verti_box = widgets.VBox(global_list_verti_box)
        display(verti_box)

        if self._are_2_checkboxes_selected():
            self.error_label = widgets.HTML("")
        else:    
            self.error_label = widgets.HTML("<font color='red'><b>ERROR</b>: Select 2 and only 2 checkboxes!</font>")
        
        display(self.error_label)
        display(widgets.HTML("<hr>"))
        self.widget_angle = widgets.Label("")
        display(widgets.HBox([widgets.Label("Angle value:"), self.widget_angle]))

    def get_list_index_of_checkboxes(self):
        self.parent.list_states_checkbox = [x.value for x in self.list_checkboxes]
        list_index = []
        for _index, _value in enumerate(self.parent.list_states_checkbox):
            if _value:
                list_index.append(_index)

        return list_index

    def _are_2_checkboxes_selected(self):
        list_index = self.get_list_index_of_checkboxes()
        if len(list_index) != 2:
            return False
        
        return True
    
    def on_check_change(self, change):
        if not self._are_2_checkboxes_selected():
            self.error_label.value = "<font color='red'><b>ERROR</b>: Select 2 and only 2 checkboxes!</font>"
        else:
            self.error_label.value = ""
            current_file_name = self.selected_file_label.value
            list_index = self.get_list_index_of_checkboxes()
            first_index, second_index = list_index
            file_name_split = current_file_name.split("_")
            self.widget_angle.value = f"{file_name_split[first_index]}.{file_name_split[second_index]}"

    def data_selected(self, top_folder):
        logging.info(f"{self.parent.current_data_type} top folder selected: {top_folder}")
        self.parent.working_dir[self.data_type] = top_folder
        # print(f"{self.data_type} top folder is: {top_folder}")
        display(HTML(f"<font color='green'><b>{self.data_type} folder selected</b>: {top_folder}</font>"))
        # loguru_logging.info(f"{self.data_type} folder selected: {top_folder} via LOGURU"  )
        logging.info(f"{self.data_type} folder selected: {top_folder}")

        if self.parent.MODE == OperatingMode.white_beam:
            list_tiff = glob.glob(os.path.join(top_folder, "*.tif*"))
            list_tiff.sort()
            self.parent.list_of_images[self.data_type] = list_tiff
            logging.info(f"{len(list_tiff)} {self.data_type} files!")

        if self.data_type == DataType.sample:
            self.parent.configuration.top_folder.sample = top_folder
        elif self.data_type == DataType.ob:
            self.parent.configuration.top_folder.ob = top_folder

    def save_list_of_angles(self, list_of_images):
        if self.how_to_retrieve_angle_value_widget.value == HowToRetrieveAngleValue.use_angle_value_from_metadata_file:
            self.retrieve_angle_value_from_metadata_file(list_of_images)
            
            self.parent.final_list_of_angles = [float(angle) for angle in self.parent.final_list_of_angles]
            # we need to sort the angles and then sort the list of images the same way
            sorted_indices = sorted(range(len(self.parent.final_list_of_angles)), key=lambda i: self.parent.final_list_of_angles[i])
            self.parent.final_list_of_angles = [self.parent.final_list_of_angles[i] for i in sorted_indices]
            self.parent.final_list_of_angles_rad = np.deg2rad(self.parent.final_list_of_angles)
            self.parent.list_of_images[DataType.sample] = [list_of_images[i] for i in sorted_indices]

            logging.info(f"Angle values retrieved from metadata file and sorted: {self.parent.final_list_of_angles}")
            logging.info(f"list of files sorted the same way: {[os.path.basename(file) for file in self.parent.list_of_images[DataType.sample]]}")

        elif self.how_to_retrieve_angle_value_widget.value == HowToRetrieveAngleValue.define_naming_convention:
            self.retrieve_angle_value_from_file_name(list_of_images)

    def retrieve_list_of_files_and_angles(self):
        """Retrieve angle values from the selected images."""
        self.save_list_of_angles(self.parent.list_of_images[DataType.sample])
        return {'list_of_images': self.parent.list_of_images[DataType.sample],
                'list_of_angles_deg': self.parent.final_list_of_angles,
                'list_of_angles_rad': self.parent.final_list_of_angles_rad}

    @staticmethod
    def retrieve_angle_value_from_tiff(file_name):
        try:
            _image = Image.open(file_name)
            _metadata = dict(_image.tag_v2)
            _full_string = _metadata[65039]
            _angle_value = _full_string.split(":")[1]  # Extract the angle value before the space
            return _angle_value
        except Exception as e:
            logging.error(f"Error retrieving angle value from TIFF file {file_name}: {e}")
            display(widgets.HTML(f"<font color='red'><b>ERROR</b>: Could not retrieve angle value from TIFF file {file_name}. Ensure the file has the correct metadata.</font>"))
            raise MetadataError(f"Could not retrieve angle value from TIFF file {file_name}. Ensure the file has the correct metadata.") from e

    def retrieve_angle_value_from_metadata_file(self, list_of_images):
        logging.info("Retrieving angle value from metadata file.")
        list_of_angles = []
        for _file in list_of_images:
            try:
                angle = Load.retrieve_angle_value_from_tiff(_file)
            except ValueError as e:
                logging.error(f"Error retrieving angle value from file {_file} using metadata: {e}")
                return
            
            list_of_angles.append(angle)

        self.parent.final_list_of_angles = np.array(list_of_angles)
        list_of_angles_rad = np.array([np.deg2rad(float(_angle)) for _angle in list_of_angles])
        self.parent.final_list_of_angles_rad = list_of_angles_rad

    def retrieve_angle_value_from_file_name(self, list_of_images):
        """Retrieve angle value from the file name based on the selected checkboxes."""

        logging.info("Retrieving angle value from file name.")
        list_checkboxes = self.list_checkboxes
        list_indices = []
        for _index, _checkbox in enumerate(list_checkboxes):
            if _checkbox.value:
                list_indices.append(_index)

        if len(list_indices) != 2:
            raise ValueError("You need to select 2 fields to determine the angle value (degree.minutes)")

        base_list_of_images = [os.path.basename(_file) for _file in list_of_images]
        list_of_angles = []
        for _file in base_list_of_images:
            _file = os.path.splitext(_file)[0]  # remove extension
            _splitted_named = _file.split("_")
            angle_degree = _splitted_named[list_indices[0]]
            angle_minute = _splitted_named[list_indices[1]]
            angle_value = float(f"{angle_degree}.{angle_minute}")
            list_of_angles.append(angle_value)

        self.parent.final_list_of_angles = np.array(list_of_angles)
        list_of_angles_rad = np.array([np.deg2rad(float(_angle)) for _angle in list_of_angles])
        self.parent.final_list_of_angles_rad = list_of_angles_rad
        for _file_name, _angle, _angle_rad in zip(base_list_of_images, list_of_angles, list_of_angles_rad):
            logging.info(f"\t{_file_name} : {_angle} degrees, {_angle_rad} radians")

    def determine_projections_angles_to_use(self, percentage_to_use=50):
        """Retrieve list of TIFF files from the selected folder."""
        logging.info(f"Determine list of projections to use according to percentage selected ...")
        
        # if retrieve_list_of_files_and_angles:
        logging.info(f"\tRetrieving list of files and angles ...")
        list_of_tiff = self.parent.list_of_images[DataType.sample]
        list_of_tiff.sort()
        _dict = self.retrieve_list_of_files_and_angles()
        list_of_tiff = _dict['list_of_images']
        list_of_angles_deg = _dict['list_of_angles_deg']
        # list_of_angles_rad = _dict['list_of_angles_rad']
        logging.info(f"Done retrieving list of files and angles.")
        
        nbr_images_to_use = int(percentage_to_use / 100 * len(list_of_tiff))
        if nbr_images_to_use < 5:
            nbr_images_to_use = 5
        logging.info(f"\tNumber of projections to use: {nbr_images_to_use}")
        list_of_angles_deg = [float(_value) for _value in self.parent.final_list_of_angles]
        self.parent.full_list_of_angles_rad = self.parent.final_list_of_angles_rad

        if percentage_to_use == 100:
            logging.info(f"\tUsing all angles for the reconstruction.")
            self.parent.temp_final_list_of_angles = list_of_angles_deg
            self.parent.temp_final_list_of_angles_rad = self.parent.final_list_of_angles_rad
            self.parent.temp_list_of_images = list_of_tiff
            self.parent.temp_nbr_of_images_will_be_used = len(list_of_tiff)
            return

        logging.info(f"\tSelecting angles using farthest point sampling ...")
        selected_angles = farthest_point_sampling(list_of_angles_deg, nbr_images_to_use)
        logging.info(f"\tDone selecting angles.")

        logging.info(f"Making sure 0 and 180 degrees are included in the selected angles.")
        if not list_of_angles_deg[0] in selected_angles:
            logging.info(f"\tAdding 0 degrees to the selected angles.")
            selected_angles.insert(0, np.float64(list_of_angles_deg[0]))

        # if not list_of_angles_deg[-1] in selected_angles:
        #     logging.info(f"\tAdding 180 degrees to the selected angles.")
        #     selected_angles.append(np.float64(list_of_angles_deg[-1]))

        logging.info(f"\tSelected angles: {selected_angles}")
        selected_indices = [np.where(list_of_angles_deg == angle)[0][0] for angle in selected_angles]
        logging.info(f"\tSelected indices: {selected_indices}")
        self.parent.temp_final_list_of_angles = selected_angles
        self.parent.temp_final_list_of_angles_rad = np.deg2rad(selected_angles)
        self.parent.temp_list_of_images = [list_of_tiff[i] for i in selected_indices]
        self.parent.temp_nbr_of_images_will_be_used = len(selected_angles)
        logging.info(f"\tList of images to use: {[os.path.basename(file) for file in self.parent.temp_list_of_images]}")

    def visualize_angles_selected(self):
        """Visualize the angles selected for the projections."""
        logging.info("Visualizing angles selected for the projections.")
        full_list_of_angles_rad = self.parent.full_list_of_angles_rad
        logging.info(f"\tFull list of angles (radians): {full_list_of_angles_rad}")
        logging.info(f"\tSelected angles (radians): {self.parent.temp_final_list_of_angles_rad}")
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
        ax.plot(full_list_of_angles_rad, np.ones(len(full_list_of_angles_rad))*2, marker='o', linestyle='None', color='blue', markersize=5, label='All Angles')
        ax.set_rmax(2)
        ax.set_rticks([0.5, 1, 1.5])
        selected_list_of_angles_rad = self.parent.temp_final_list_of_angles_rad
        ax.plot(selected_list_of_angles_rad, np.ones(len(selected_list_of_angles_rad))*2, marker='o', linestyle='None', color='red', markersize=10, label='Selected Angles')

    def load_white_beam_data(self):
        """ from white beam notebook """

        # move data from temp to final
        self.parent.final_list_of_angles = self.parent.temp_final_list_of_angles
        self.parent.final_list_of_angles_rad = self.parent.temp_final_list_of_angles_rad
        self.parent.list_of_images[DataType.sample] = self.parent.temp_list_of_images

        list_of_images = self.parent.list_of_images
        logging.info(f"loading the data:")

        for _data_type in list_of_images.keys():
            logging.info(f"\tworking with {_data_type} ... ")

            if not list_of_images[_data_type]:
                logging.info(f" nothing to load for {_data_type}, no files have been selected!")
                continue

            # list_of_images[_data_type].sort()
            list_tiff = list_of_images[_data_type]
            # nbr_images_to_use = int(self.percentage_to_use.value / 100 * len(list_of_images[_data_type]))
            # if nbr_images_to_use == 0:
            #     nbr_images_to_use = 1

            # logging.info(f"\t{nbr_images_to_use} images will be used for the reconstruction")    
            # list_tiff = random.sample(list_tiff, nbr_images_to_use)
            # logging.info(f"\t{len(set(list_tiff))} unique images will be used for the reconstruction")

            # list_tiff.sort()

            # list_tiff_index_to_use = np.random.randint(0, len(list_of_images[_data_type]), nbr_images_to_use)
            # list_tiff_index_to_use.sort()
            # list_tiff = [list_tiff[_index] for _index in list_tiff_index_to_use]
            
            self.parent.list_of_images[_data_type] = list_tiff
            list_of_images[_data_type] = list_tiff

            if _data_type == DataType.sample:
                self.save_list_of_angles(list_tiff)

            # self.parent.master_3d_data_array[_data_type] = load_data_using_multithreading(list_of_images[_data_type])

            if list_of_images[DataType.ob] is None:
                # we are dealing with normalized data
                dtype = np.float32
            else:
                dtype = np.uint16

            self.parent.master_3d_data_array[_data_type] = load_list_of_tif(list_of_images[_data_type], dtype=dtype)
            logging.info(f"{np.shape(self.parent.master_3d_data_array[_data_type]) = }")
            logging.info(f"\tloading {_data_type} ... done !")

        [height, width] = np.shape(self.parent.master_3d_data_array[DataType.sample][0])
        self.parent.image_size['height'] = height
        self.parent.image_size['width'] = width

    def load_data(self, combine=False):
        """combine is True when working with white beam (from tof notebook)"""
        
        logging.info(f"importing the data:")
       
        final_list_of_angles = []
        final_dict_of_pc = {}
        final_dict_of_frame_number = {}

        if combine:
            logging.info(f"\t combine mode is ON")
        else:
            logging.info(f"\t not combining the TOF images")

        list_of_runs_sorted = self.parent.list_of_runs_to_use
        self.parent.configuration.list_of_sample_runs = list_of_runs_sorted[DataType.sample]
        self.parent.configuration.list_of_ob_runs = list_of_runs_sorted[DataType.ob]

        for _data_type in self.parent.list_of_runs.keys():
            _master_data = []
            logging.info(f"\tworking with {_data_type}:")

            _final_list_of_pc = []
            _final_list_of_frame_number = []

            for _run in tqdm(list_of_runs_sorted[_data_type]):
                _full_path_run = self.parent.list_of_runs[_data_type][_run][Run.full_path]
                if _data_type == DataType.sample:
                    final_list_of_angles.append(self.parent.list_of_runs[_data_type][_run][Run.angle])

                logging.info(f"\t\tloading {os.path.basename(_full_path_run)} ...")
                list_tif = retrieve_list_of_tif(_full_path_run)
                _master_data.append(load_data_using_multithreading(list_tif,
                                                                   combine_tof=combine))
                _final_list_of_pc.append(self.parent.list_of_runs[_data_type][_run][Run.proton_charge_c])
                if self.parent.list_of_runs[_data_type][_run][Run.frame_number]:
                    _final_list_of_frame_number.append(self.parent.list_of_runs[_data_type][_run][Run.frame_number])
                logging.info(f"\t\t loading done!")
            self.parent.master_3d_data_array[_data_type] = np.array(_master_data)
            final_dict_of_pc[_data_type] = _final_list_of_pc
            final_dict_of_frame_number[_data_type] = _final_list_of_frame_number
        
        self.parent.final_list_of_angles = final_list_of_angles
        self.parent.configuration.list_of_angles = final_list_of_angles
        self.parent.final_dict_of_pc = final_dict_of_pc
        self.parent.final_dict_of_frame_number = final_dict_of_frame_number

        if combine:
            height, width = np.shape(self.parent.master_3d_data_array[DataType.sample][0])
            nbr_tof = 1
        else:
            nbr_tof, height, width = np.shape(self.parent.master_3d_data_array[DataType.sample][0])
        self.parent.image_size = {'height': height,
                                  'width': width,
                                  'nbr_tof': nbr_tof}
        
        logging.info(f"{self.parent.image_size} = ")

    def load_spectra_file(self):
        list_runs_to_use = self.parent.list_of_runs_to_use[DataType.sample]
        first_run = list_runs_to_use[0]
        full_path_to_run = self.parent.list_of_runs[DataType.sample][first_run][Run.full_path]
        list_files = glob.glob(os.path.join(full_path_to_run, "*_Spectra.txt"))
        if list_files and os.path.exists(list_files[0]):
            time_spectra_file = list_files[0]
        else:
            time_spectra_file = ""
        self.parent.spectra_file_full_path = time_spectra_file
        