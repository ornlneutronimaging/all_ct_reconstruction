from tqdm import tqdm
import os
import logging
import shutil
from IPython.display import display
from IPython.display import HTML
import numpy as np
import subprocess
import ipywidgets as widgets

from __code.utilities.system import get_user_name
from __code.utilities.save import make_tiff
from __code.utilities.json import save_json
from __code.utilities.configuration_file import SvmbirConfig
from __code.parent import Parent
from __code.utilities.create_scripts import create_sh_file, create_sh_hsnt_file
from __code import DataType, STEP2_NOTEBOOK
from __code.utilities.time import get_current_time_in_special_file_name_format
from __code.config import imaging_team
from __code.utilities.system import get_instrument_generic_name


class Export:

    base_image_name = "image"

    def __init__(self, image_3d=None, output_folder=None):
        self.image_3d = image_3d
        self.output_folder = output_folder

    def run(self):
       
        for _index, _data in tqdm(enumerate(self.image_3d)):
            short_file_name = f"{self.base_image_name}_{_index:04d}.tiff"
            full_file_name = os.path.join(self.output_folder, short_file_name)
            logging.info(f"\texporting {full_file_name}")
            make_tiff(data=_data, filename=full_file_name)
            # print(f"{type(_data) = }")
            # print(f"{_data.dtype = }")
            # print(f"{np.min(_data) = }")
            # print(f"{np.max(_data) = }")


class ExportExtra(Parent):
  
    def update_configuration(self):
        # especially for all svmbir settings
        if self.parent.o_svmbir is None:
            return

        instrument = self.parent.instrument
        ipts_number = self.parent.ipts_number
        self.parent.configuration.instrument = instrument
        self.parent.configuration.ipts_number = int(ipts_number)

        sharpness = self.parent.o_svmbir.sharpness_ui.value
        snr_db = self.parent.o_svmbir.snr_db_ui.value
        positivity = self.parent.o_svmbir.positivity_ui.value
        max_iterations = self.parent.o_svmbir.max_iterations_ui.value
        max_resolutions = self.parent.o_svmbir.max_resolutions_ui.value
        verbose = 1 if self.parent.o_svmbir.verbose_ui.value else 0

        svmbir_config = SvmbirConfig()
        svmbir_config.sharpness = sharpness
        svmbir_config.snr_db = snr_db
        svmbir_config.positivity = positivity
        svmbir_config.max_iterations = max_iterations
        svmbir_config.verbose = verbose
        # svmbir_config.top_slice = top_slice
        # svmbir_config.bottom_slice = bottom_slice
        self.parent.configuration.svmbir_config = svmbir_config

        logging.info(f"Updating svmbir configuration using ui data:")
        logging.info(f"\t{sharpness = }")
        logging.info(f"\t{snr_db = }")
        logging.info(f"\t{positivity = }")
        logging.info(f"\t{max_iterations = }")
        logging.info(f"\t{max_resolutions = }")
        logging.info(f"\t{verbose = }")

    def run(self, base_log_file_name=None, prefix=""):
        log_file_name = f"/SNS/VENUS/shared/log/{base_log_file_name}.log"
        output_folder = self.parent.working_dir[DataType.extra]
        try:
            shutil.copy(log_file_name, output_folder)
        except PermissionError:
            logging.error(f"PermissionError: cannot copy {log_file_name} to {output_folder}")
            
        # display(HTML(f"\tlog file from {log_file_name} to {output_folder}!"))

        configuration = self.parent.configuration

        # update configuration
        configuration.output_folder = output_folder

        # center of rotation if manual mode used
        if self.parent.o_center_and_tilt is not None:
            if self.parent.o_center_and_tilt.is_manual_mode():
                configuration.center_of_rotation = self.parent.o_center_and_tilt.get_center_of_rotation()

        base_sample_folder = os.path.basename(os.path.abspath(self.parent.working_dir[DataType.sample]))

        _time_ext = get_current_time_in_special_file_name_format()
        # config_file_name = f"/SNS/VENUS/shared/log/{base_sample_folder}_{_time_ext}.json"
        if prefix:
            config_file_name = os.path.join(output_folder, f"{prefix}_{base_sample_folder}_{_time_ext}.json")   
        else:
            config_file_name = os.path.join(output_folder, f"{base_sample_folder}_{_time_ext}.json")
        
        self.config_file_name = config_file_name

        config_json = configuration.model_dump_json()
        save_json(config_file_name, json_dictionary=config_json)
        self.config_json = config_json

        self.sh_file_name = create_sh_file(json_file_name=config_file_name,
                                           output_folder=output_folder)

        # ipts_number = configuration.ipts_number
        # instrument = configuration.instrument
        # instrument = get_instrument_generic_name(instrument)
        # self.hsnt_output_json_folder = os.path.join("/data", instrument, f"IPTS-{ipts_number}", "all_config_files")
        # self.hsnt_output_folder = os.path.join("/data", instrument, f"IPTS-{ipts_number}")
        # self.sh_hsnt_script_name = create_sh_hsnt_file(configuration=configuration,
        #                                                json_file_name=config_file_name, 
        #                                                hstn_output_json_folder=self.hsnt_output_json_folder)

        display(HTML(f"<font color='blue'><b>Next step</b></font>"))

        list_options = [
                f"Divide reconstruction into several jobs and run them in parallel",
                f"Manually launch script outside notebook",
                f"Launch the script directly from the notebook",
        ]
        # ucams = get_user_name()
        # if ucams in imaging_team:
        #     list_options.append(f"Create script to run from hsnt")

        # 3 options are offered to the user
        choices = widgets.RadioButtons(
            options=list_options,
            value="Launch the script directly from the notebook",
            description='',
            layout=widgets.Layout(width='100%'),
            disabled=False
        )
        display(choices)

        self.instructions = widgets.Textarea(value=f"Reload the configuration file {os.path.basename(self.config_file_name)} found in {os.path.dirname(self.config_file_name)} in the notebook {STEP2_NOTEBOOK}",
                                             layout=widgets.Layout(width='100%', height='160px'),
                                             disabled=True)
        display(self.instructions) 

        self.run_script = widgets.Button(
            description='Run script',
            disabled=False,
            button_style='success',
            tooltip='Run the script directly from the notebook',
            icon='play'
        )
        display(self.run_script)

        choices.observe(self.on_choice_change, names='value')
        self.run_script.on_click(self.on_run_script_click)

    def on_choice_change(self, change):
        if change['new'] == 'Launch the script directly from the notebook':
            self.run_script.disabled = False
        else:
            self.run_script.disabled = True

        if change['new'] == 'Divide reconstruction into several jobs and run them in the background':
            self.instructions.value = f"Reload the configuration file ({self.config_file_name}) in the notebook {STEP2_NOTEBOOK}"
        elif change['new'] == 'Manually launch script outside notebook':
            self.instructions.value = f"Launch the following script from the command line: {self.sh_file_name}"
        elif change['new'] == 'Create script to run from hsnt':
            self.instructions.value = f"1. Connect to hsnt\n" + \
                f"2. Copy the pre-processed data: > 'cp {self.parent.configuration.projections_pre_processing_folder} {self.hsnt_output_folder}'\n" + \
                f"3. Copy the config json file: > 'cp {self.config_file_name} {self.hsnt_output_json_folder}'\n" + \
                f"4. Copy the script to run: > 'cp {self.sh_hsnt_script_name} {self.hsnt_output_folder}'\n" + \
                f"5. Run the following script: > '{os.path.join(self.hsnt_output_folder,os.path.basename(self.sh_hsnt_script_name))}'"
        else:
            self.instructions.value = f"click the button below to run the script directly from the notebook"

    def on_run_script_click(self, b):
        print("Running the script directly from the notebook...")
        subprocess.run(["xterm", "-e", f"{self.sh_file_name}", "exec bash"], check=True)

        # display(HTML(f"<font color='blue'>From this point you have 3 options:</font>"))
        # display(HTML(f"<font color='blue'> 1. reload the configuration file </font>(<font color='green'>{config_file_name}</font>) in the notebook <font color='green'> {STEP2_NOTEBOOK}</font>"))
        # display(HTML(f"<br>"))
        # display(HTML(f"<font color='blue'> 2. launch the following script from the command line"))
        # display(HTML(f"<font color='green'>{sh_file_name}</font>"))
        # display(HTML(f"<br>"))
   
