from tqdm import tqdm
import os
import logging
import shutil
from IPython.display import display
from IPython.core.display import HTML
import numpy as np
import subprocess
import ipywidgets as widgets

from __code.utilities.save import make_tiff
from __code.utilities.json import save_json
from __code.parent import Parent
from __code.utilities.create_scripts import create_sh_file
from __code import DataType, STEP2_NOTEBOOK
from __code.utilities.time import get_current_time_in_special_file_name_format


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

        display(HTML(f"<font color='blue'><b>Next step</b></font>"))

        # 3 options are offered to the user
        choices = widgets.RadioButtons(
            options=[
                f"Divide reconstruction into several jobs and run them in parallel",
                f"Manually launch script outside notebook",
                f"Launch the script directly from the notebook",
            ],
            value="Launch the script directly from the notebook",
            description='',
            layout=widgets.Layout(width='100%'),
            disabled=False
        )
        display(choices)

        self.instructions = widgets.Textarea(value=f"Reload the configuration file {os.path.basename(self.config_file_name)} found in {os.path.dirname(self.config_file_name)} in the notebook {STEP2_NOTEBOOK}",
                                             layout=widgets.Layout(width='100%', height='80px'),
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
   
