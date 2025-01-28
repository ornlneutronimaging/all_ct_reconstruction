import os

from __code.utilities.time import get_current_time_in_special_file_name_format
from __code.utilities.files import make_or_reset_folder
from __code.workflow.export import Export


def export_slices(config_model, data_array):
    print(f"exporting ...", end="")
    output_folder = config_model.output_folder
    top_folder = os.path.basename(config_model.top_folder.sample)

    full_output_folder_name = os.path.join(output_folder, top_folder + 
                                          f"_reconstructed_on_{get_current_time_in_special_file_name_format()}")
    
    make_or_reset_folder(full_output_folder_name)
    print(f" -> {full_output_folder_name}")

    o_export = Export(image_3d=data_array,
                          output_folder=full_output_folder_name)
    o_export.run()

    print(" done!")
    print(f"Slices can be found in {full_output_folder_name}!")
    