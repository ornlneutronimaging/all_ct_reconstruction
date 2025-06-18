from tqdm import tqdm
import os
import logging
import shutil
from IPython.display import display
from IPython.core.display import HTML
import numpy as np

from __code.utilities.save import make_tiff
from __code.utilities.json import save_json
from __code.parent import Parent
from __code import DataType, STEP3_SCRIPTS, STEP2_NOTEBOOK
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

    def create_sh_file(self, json_file_name, output_folder):
        """
        Create a shell script to run the reconstruction with the given configuration file.
        """
        time_stamp = get_current_time_in_special_file_name_format()
        sh_file_name = os.path.join(output_folder, f"run_reconstruction_{time_stamp}.sh")

        json_file_name_on_linux = json_file_name.replace(" ", "\ ")

        with open(sh_file_name, 'w') as sh_file:
            sh_file.write("#!/bin/bash\n")
            sh_file.write(f"source /opt/anaconda/etc/profile.d/conda.sh\n")
            sh_file.write(f"conda activate /SNS/users/j35/micromamba/envs/svmbir_py310_micromamba\n")
            sh_file.write(f"python {STEP3_SCRIPTS} {json_file_name_on_linux}\n")
        
        os.chmod(sh_file_name, 0o755)
        return sh_file_name

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
        
        config_json = configuration.model_dump_json()
        save_json(config_file_name, json_dictionary=config_json)

        sh_file_name = self.create_sh_file(json_file_name=config_file_name, 
                                      output_folder=output_folder)

        display(HTML(f"<font color='blue'>From this point you have two options:</font>"))
        display(HTML(f"<font color='blue'> - reload the configuration file </font>(<font color='green'>{config_file_name}</font>) in the notebook <font color='green'> {STEP2_NOTEBOOK}</font>"))
        display(HTML(f"<br>"))
        display(HTML(f"<font color='blue'> - launch the following script from the command line"))
        display(HTML(f"<font color='green'>{sh_file_name}</font>"))
