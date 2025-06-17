import numpy as np
import os
import logging
import glob


def merge_reconstructed_slices(output_data_folder=None, top_slice=0, list_of_output_folders=None, list_of_slices_to_reconstruct=None):
    
    final_output_folder = output_data_folder

    logging.info(f"merge reconstructed slices ...")
    logging.info(f"\t{output_data_folder = }")
    logging.info(f"\t{top_slice = }")
    logging.info(f"\t{list_of_output_folders = }")
    logging.info(f"\t{list_of_slices_to_reconstruct = }")

    list_folder_tiff = {}
    for _index, _folder in enumerate(list_of_output_folders):
        _list_tiff = glob.glob(os.path.join(_folder, '*.tiff'))
        if len(_list_tiff) == 0:
            raise ValueError(f"no tiff files found in {_folder}")
        
        _list_tiff.sort()
        list_folder_tiff[_index] = _list_tiff

    list_slices_already_processed = []
    for _index, [top_slice_index, bottom_slice_index] in enumerate(list_of_slices_to_reconstruct):
        logging.info(f"working with folder: {os.path.dirname(list_folder_tiff[_index][0])}")
        logging.info(f"\tfrom slice #{top_slice_index} to slice #{bottom_slice_index}")

        list_slices = np.arange(top_slice_index, bottom_slice_index)
        for _tiff_index, _slice_index in enumerate(list_slices):
            if _slice_index in list_slices_already_processed:
                os.remove(list_folder_tiff[_index][_tiff_index]) # no need to move that slice, already processed

            else:
                list_slices_already_processed.append(_slice_index)
                logging.info(f"moving slice #{_slice_index} ({os.path.basename(list_folder_tiff[_index][_tiff_index])}) -> #image_{_slice_index + top_slice:03d}.tiff ... ")
                _new_tiff_file = os.path.join(final_output_folder, f"image_{_slice_index + top_slice:03d}.tiff")
                os.rename(list_folder_tiff[_index][_tiff_index], _new_tiff_file)
                
        # remove the input folder
        try:
            logging.info(f"removing folder {list_of_output_folders[_index]}!")
            os.rmdir(list_of_output_folders[_index])
        except OSError as e:
            logging.error(f"Error removing folder {list_of_output_folders[_index]}: {e}")
            continue
        
    logging.info(f"all slices merged into {final_output_folder}!")
