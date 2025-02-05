import glob
import os
import numpy as np
import logging
from tqdm import tqdm
from IPython.display import display
import ipywidgets as widgets

from __code import DataType, Run, OperatingMode
from __code import DEBUG, debug_folder
from __code.parent import Parent
from __code.utilities.file_folder_browser import FileFolderBrowser
from __code.utilities.load import load_data_using_multithreading, load_list_of_tif, load_tiff
from __code.utilities.files import retrieve_list_of_tif
from __code.config import DEFAULT_NAMING_CONVENTION_INDICES, PERCENTAGE_OF_DATA_TO_USE_FOR_RECONSTRUCTION


class Load(Parent):

    list_of_runs_to_use = {DataType.sample: [],
                           DataType.ob: [],
    }

    def select_folder(self, data_type=DataType.sample, multiple_flag=False):

        self.parent.current_data_type = data_type
        self.data_type = data_type
        if data_type in [DataType.reconstructed, DataType.extra]:
            working_dir = self.parent.working_dir[DataType.processed]
        else:
            working_dir = self.parent.working_dir[DataType.top]

        print(f"{working_dir = }")

        if DEBUG:
            self.data_selected(debug_folder[self.parent.MODE][data_type])
            return

        if not os.path.exists(working_dir):
            working_dir = os.path.abspath(os.path.expanduser("~"))

        o_file_browser = FileFolderBrowser(working_dir=working_dir,
                                           next_function=self.data_selected)
        o_file_browser.select_input_folder(instruction=f"Select Top Folder of {data_type}",
                                           multiple_flag=multiple_flag)

    def select_images(self, data_type=DataType.ob):
        self.parent.current_data_type = data_type
        self.data_type = data_type
        if data_type in [DataType.reconstructed, DataType.extra]:
            working_dir = self.parent.working_dir[DataType.processed]
        else:
            working_dir = self.parent.working_dir[DataType.top]

        if DEBUG:
            working_dir = debug_folder[self.parent.MODE][data_type]
            if not os.path.exists(working_dir):
                return
            #list_images = glob.glob(os.path.join(working_dir, "*_0045_*.tif*"))
            list_images = glob.glob(os.path.join(working_dir, "*.tif*"))
            list_images.sort()
            self.images_selected(list_images=list_images)
            return

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
        self.parent.list_of_images[self.data_type] = list_images

    def select_percentage_of_data_to_use(self):
        self.percentage_to_use = widgets.IntSlider(value=PERCENTAGE_OF_DATA_TO_USE_FOR_RECONSTRUCTION,
                                    min=1,
                                    max=100,
                                    step=1,
                                    layout=widgets.Layout(width='100%'))
        display(self.percentage_to_use)

        list_of_tiff = self.parent.list_of_images[DataType.sample]
        percentage = self.percentage_to_use.value
        nbr_images = int(percentage / 100 * len(list_of_tiff))
        self.number_of_images_to_use = widgets.Label(f"{nbr_images} images will be used for the reconstruction")
        display(self.number_of_images_to_use)
        self.percentage_to_use.observe(self.on_percentage_to_use_change, names='value') 

    def on_percentage_to_use_change(self, change):
        new_value = change['new']
        list_tiff = self.parent.list_of_images[DataType.sample]
        nbr_images = int(new_value / 100 * len(list_tiff))
        self.number_of_images_to_use.value = f"{nbr_images} images will be used for the reconstruction"

    def define_naming_convention(self):
        number_of_images = len(self.parent.list_of_images[DataType.sample])
        # random number between 0 and number_of_images
        random_index = np.random.randint(0, number_of_images)

        # random file name
        first_file = self.parent.list_of_images[DataType.sample][random_index]
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
        print(f"Top {self.data_type} folder selected: {top_folder}")

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
            _splitted_named = _file.split("_")
            angle_degree = _splitted_named[list_indices[0]]
            angle_minute = _splitted_named[list_indices[1]]
            angle_value = float(f"{angle_degree}.{angle_minute}")
            list_of_angles.append(angle_value)

        self.parent.final_list_of_angles = np.array(list_of_angles)
        list_of_angles_rad = np.array([np.deg2rad(float(_angle)) for _angle in list_of_angles])
        self.parent.final_list_of_angles_rad = list_of_angles_rad

    def load_white_beam_data(self):
        """ from white beam notebook """
        list_of_images = self.parent.list_of_images
        logging.info(f"loading the data:")

        for _data_type in list_of_images.keys():
            logging.info(f"\tworking with {_data_type} ... ")

            if not list_of_images[_data_type]:
                logging.info(f" nothing to load for {_data_type}, no files have been selected!")
                continue

            list_of_images[_data_type].sort()
            list_tiff = list_of_images[_data_type]
            nbr_images_to_use = int(self.percentage_to_use.value / 100 * len(list_of_images[_data_type]))
            if nbr_images_to_use == 0:
                nbr_images_to_use = 1
                
            list_tiff_index_to_use = np.random.randint(0, len(list_of_images[_data_type]), nbr_images_to_use)
            list_tiff_index_to_use.sort()
            list_tiff = [list_tiff[_index] for _index in list_tiff_index_to_use]
            self.parent.list_of_images[_data_type] = list_tiff
            list_of_images[_data_type] = list_tiff

            if _data_type == DataType.sample:
                self.save_list_of_angles(list_tiff)

            # self.parent.master_3d_data_array[_data_type] = load_data_using_multithreading(list_of_images[_data_type])
            self.parent.master_3d_data_array[_data_type] = load_list_of_tif(list_of_images[_data_type])
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
        