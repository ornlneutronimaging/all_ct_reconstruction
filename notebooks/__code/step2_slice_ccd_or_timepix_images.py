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

from __code import OperatingMode, DataType, STEP3_SCRIPTS
from __code.config import DEBUG, debug_folder
from __code.utilities.configuration_file import CropRegion
from __code.utilities.configuration_file import select_file, loading_config_file_into_model
from __code.utilities.logging import setup_logging
from __code.utilities.files import retrieve_list_of_tif, make_or_reset_folder
from __code.utilities.create_scripts import create_sh_file, create_sh_hsnt_file
from __code.utilities.load import load_data_using_multithreading, load_list_of_tif
from __code.utilities.time import get_current_time_in_special_file_name_format
from __code.utilities.json import save_json

BASENAME_FILENAME, _ = os.path.splitext(os.path.basename(__file__))


class JsonTypeRequested:
    single = '1 json (reconstruction will run in sequence)'
    multi = 'multi jsons (to run reconstruction in parallel)'
    undefined = 'undefined'


class Step2SliceCcdOrTimePixImages:

    json_type_requested = JsonTypeRequested.undefined

    def __init__(self, system=None):

        # self.configuration = Configuration()
        self.working_dir = system.System.get_working_dir()
        if DEBUG:
            self.working_dir = debug_folder[OperatingMode.white_beam][DataType.extra]

        self.instrument = system.System.get_instrument_selected()

        setup_logging(BASENAME_FILENAME)      
        logging.info(f"working_dir: {self.working_dir}")
        logging.info(f"instrument: {self.instrument}")
        if DEBUG:
            logging.info(f"WARNING!!!! we are running using DEBUG mode!")

    def select_config_file(self):
        select_file(top_folder=self.working_dir,
                    next_function=self.load_config_file)

    def load_config_file(self, config_file_path):
        self.output_config_file = os.path.dirname(config_file_path)
        logging.info(f"configuration file loaded: {config_file_path}")
        self.configuration = loading_config_file_into_model(config_file_path)
        self.images_path = self.configuration.projections_pre_processing_folder
        print(f"Configuration file {os.path.basename(config_file_path)} loaded!")

    def load_and_crop(self):
        self.load_images()
        self.crop_settings()
        
    def load_images(self):
        logging.info(f"images_path: {self.images_path}")
        list_tiff = retrieve_list_of_tif(self.images_path)
        logging.info(f"list_tiff: {list_tiff}")

        #DEBUG
        list_tiff = list_tiff[0:30]

        self.data = load_list_of_tif(list_tiff, dtype=np.float32)
        # self.data = load_data_using_multithreading(list_tiff)
        # self.data = np.moveaxis(self.data, 1, 2)
        logging.info(f"loading images done!")
        logging.info(f"self.data.shape: {self.data.shape}")

    def select_range_of_slices(self):
        
        left, right, top, bottom = self.display_roi.result
        data = self.data[:, top:bottom, left:right]

        nbr_images, height, width = data.shape

        def plot_images(image_index, top_slice, bottom_slice, nbr):
            fig, ax = plt.subplots()
            im = ax.imshow(data[image_index], cmap='jet')
            plt.colorbar(im, ax=ax, shrink=0.5)
            
            range_size = int((np.abs(top_slice - bottom_slice)) / nbr)

            for _range_index in np.arange(nbr):
                _top_slice = top_slice + _range_index * range_size

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

    def crop_settings(self):

        nbr_images, height, width = self.data.shape

        master_vmin = np.min(self.data)
        master_vmax = np.max(self.data)

        def plot_crop(image_index, left, right, top, bottom, vmin, vmax, use_local):

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
  
    def select_json_type_you_want_to_create(self):

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
    
    def export_config_file(self):
         if self.json_type_requested == JsonTypeRequested.undefined:
            export_options = self.json_type.value
            if export_options == JsonTypeRequested.single:
                self.export_single_config_file()
            else:
                self.export_multi_config_file()

    def export_multi_config_file(self):
        top_slice, bottom_slice, nbr = self.display_plot_images.result
        logging.info(f"Exporting config file:")
        logging.info(f"\ttop_slice: {top_slice}")
        logging.info(f"\tbottom_slice: {bottom_slice}")
        logging.info(f"\tnbr_of_ranges: {nbr}")

        range_size = int((np.abs(top_slice - bottom_slice)) / nbr)

        left, right, top, bottom = self.display_roi.result
        self.configuration.crop_region = CropRegion(left=left, right=right, top=top, bottom=bottom)

        current_time = get_current_time_in_special_file_name_format()
        working_dir = self.output_config_file
        sub_folder_for_all_config_files = os.path.join(working_dir, f"all_config_files_{current_time}")
        make_or_reset_folder(sub_folder_for_all_config_files)

        ipts_number = self.configuration.ipts_number
        instrument = self.configuration.instrument
        hsnt_output_json_folder = os.path.join("/data", instrument, f"IPTS-{ipts_number}", "all_config_files")
        hsnt_output_folder = os.path.join("/data", instrument, f"IPTS-{ipts_number}")

        list_of_sh_hsnt_script_files = []
        list_of_json_files = []
        current_time = get_current_time_in_special_file_name_format()

        for _range_index in np.arange(nbr):

            list_slices = []

            _top_slice = top_slice + _range_index * range_size
            if _top_slice > 0:
                _top_slice -= 1  # to make sure we have a 2 pixels overlap between ranges of slices

            _bottom_slice = top_slice + _range_index * range_size + range_size
            if _bottom_slice < (self.data.shape[1] - 1):
                _bottom_slice += 1 # to make sure we have a 2 pixels overlap between ranges of slices

            list_slices.append((_top_slice, _bottom_slice))
            self.configuration.list_of_slices_to_reconstruct = list_slices
            
            config_file_name = f"{BASENAME_FILENAME}_{current_time}_from_{_top_slice}_to_{_bottom_slice}.json"
            full_config_file_name = os.path.join(sub_folder_for_all_config_files, config_file_name)
            list_of_json_files.append(full_config_file_name)
            config_json = self.configuration.model_dump_json()
            save_json(full_config_file_name, json_dictionary=config_json)
            logging.info(f"config file saved: {full_config_file_name} for slice range {_top_slice} to {_bottom_slice}")

            sh_hsnt_script_name = create_sh_hsnt_file(configuration=self.configuration,
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

    def export_single_config_file(self):
 
        top_slice, bottom_slice, nbr = self.display_plot_images.result
        logging.info(f"Exporting config file:")
        logging.info(f"\ttop_slice: {top_slice}")
        logging.info(f"\tbottom_slice: {bottom_slice}")
        logging.info(f"\tnbr_of_ranges: {nbr}")

        range_size = int((np.abs(top_slice - bottom_slice)) / nbr)

        list_slices = []
        for _range_index in np.arange(nbr):
            _top_slice = top_slice + _range_index * range_size
            if _top_slice > 0:
                _top_slice -= 1  # to make sure we have a 2 pixels overlap between ranges of slices

            _bottom_slice = top_slice + _range_index * range_size + range_size
            if _bottom_slice < (self.data.shape[1] - 1):
                _bottom_slice += 1 # to make sure we have a 2 pixels overlap between ranges of slices

            list_slices.append((_top_slice, _bottom_slice))
            self.configuration.list_of_slices_to_reconstruct = list_slices

        left, right, top, bottom = self.display_roi.result
        self.configuration.crop_region = CropRegion(left=left, right=right, top=top, bottom=bottom)

        working_dir = self.output_config_file
        current_time = get_current_time_in_special_file_name_format()
        config_file_name = f"{BASENAME_FILENAME}_{current_time}.json"
        full_config_file_name = os.path.join(working_dir, config_file_name)
        config_json = self.configuration.model_dump_json()
        save_json(full_config_file_name, json_dictionary=config_json)
        logging.info(f"config file saved: {full_config_file_name}")

        sh_file_name = create_sh_file(json_file_name=full_config_file_name,
                                      output_folder=working_dir)
        display(HTML(f"Next and final step. Launch the following script from the command line:"))
        display(HTML(f"config file name: <font color='blue'>{config_file_name}</font>"))
        display(HTML(f"<font color='green'>{sh_file_name}</font>"))
