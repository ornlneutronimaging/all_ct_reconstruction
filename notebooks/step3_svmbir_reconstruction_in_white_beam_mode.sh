#!/bin/bash
source /opt/anaconda/etc/profile.d/conda.sh
conda activate hsnt

config_file=$1

python step3_svmbir_reconstruction_in_white_beam_mode.py $config_file
                                                            