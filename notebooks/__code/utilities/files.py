import glob
import os
import json
import shutil


def retrieve_list_of_files_from_folders(list_folders):
    list_files = []
    for _folder in list_folders:
        _tiff_files = glob.glob(os.path.join(_folder, "*.tif*"))
        list_files = [*list_files, *_tiff_files]

    list_files.sort()
    return list_files


def retrieve_list_of_runs(top_folder):
    list_runs = glob.glob(os.path.join(top_folder, "Run_*"))
    list_runs.sort()
    return list_runs


def retrieve_list_of_tif(folder):
    list_tif = glob.glob(os.path.join(folder, "*.tif*"))
    list_tif.sort()
    return list_tif


def get_number_of_tif(folder):
    return len(retrieve_list_of_tif(folder))


def load_json(json_file_name):
    if not os.path.exists(json_file_name):
        return None

    with open(json_file_name) as json_file:
        data = json.load(json_file)

    return data


def save_json(json_file_name, json_dictionary=None):
    with open(json_file_name, 'w') as outfile:
        json.dump(json_dictionary, outfile)


def make_or_reset_folder(folder_name):
    if os.path.exists(folder_name):
         shutil.rmtree(folder_name)
    os.makedirs(folder_name)


def make_folder(folder_name):
    if os.path.exists(folder_name):
        return
    os.makedirs(folder_name)
    

def get_angle_value(run_full_path=None):
    """ extract the rotation angle value from a string name looking like 
    Run_####_20240927_date_..._148_443_######_<file_index>.tif
    """
    list_tiff = retrieve_list_of_tif(run_full_path)
    if len(list_tiff) == 0:
        return None
    
    first_tiff = list_tiff[0]
    list_part = first_tiff.split("_")
    return f"{list_part[-4]}.{list_part[-3]}"
