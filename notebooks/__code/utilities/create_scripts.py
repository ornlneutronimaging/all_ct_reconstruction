import os

from __code import STEP3_SCRIPTS
from __code.utilities.time import get_current_time_in_special_file_name_format


def create_sh_file(json_file_name, output_folder):
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
