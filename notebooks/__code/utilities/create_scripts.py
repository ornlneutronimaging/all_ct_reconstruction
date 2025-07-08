import os
import shutil

from __code import STEP3_SCRIPTS
from __code.utilities.time import get_current_time_in_special_file_name_format
from __code.config import HSNT_SCRIPTS_FOLDER, HSNT_FOLDER


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


def create_sh_hsnt_file(configuration=None, json_file_name=None, hstn_output_json_folder=None):
    """
    create the script to run on hsnt
    """
    output_folder = configuration.output_folder

    time_stamp = get_current_time_in_special_file_name_format()
    sh_file_name = os.path.join(output_folder, f"run_reconstruction_on_hsnt_{time_stamp}.sh")

    # copy the files to the local folder on hsnt
    projections_pre_processing_folder = configuration.projections_pre_processing_folder

    hsnt_json_full_name = os.path.join(hstn_output_json_folder, os.path.basename(json_file_name))

    sbatch_commands = ["#SBATCH --job-name=recon1",
                       "#SBATCH --nodes=1 --exclusive",
                       "#SBATCH --mem=118G",
                       "#SBATCH --partition=cpu",
                       "#SBATCH --tmp=50G",
                       "#SBATCH --output=/data/MARS/IPTS-27829/logs/%x_%j.out",
                       "#SBATCH --error=/data/MARS/IPTS-27829/logs/%x_%j.err",
                ]

    with open(sh_file_name, 'w') as sh_file:
        sh_file.write("#!/bin/bash\n")
        sh_file.write(f"cp -rf {projections_pre_processing_folder} {HSNT_FOLDER}\n")
        sh_file.write("\n")
        for _com in sbatch_commands:
            sh_file.write(f"{_com}\n")
        sh_file.write("\n")
        
        sh_file.write('echo "Job start at $(date)"\n')
        sh_file.write("start=$(date +%s)\n")
        sh_file.write("\n")

        sh_file.write(f"source /homehxt/miniconda3/bin/activate svmbir_py310_micromamba\n")
        sh_file.write(f"python -u /data/scripts/all_ct_reconstruction/notebooks/step3_reconstruction_CCD_or_TimePix_images.py {hsnt_json_full_name}\n")
        sh_file.write("\n")

        sh_file.write('echo "Job end at $(date)"\n')
        sh_file.write("end=$(date +%s)\n")
        sh_file.write("runtime=$((end-start))\n")
        sh_file.write('echo "Job runtime: $((runtime / 3600)) hours $((runtime % 3600 / 60)) minutes $((runtime % 60)) seconds"\n')
       
    os.chmod(sh_file_name, 0o755)
    return sh_file_name