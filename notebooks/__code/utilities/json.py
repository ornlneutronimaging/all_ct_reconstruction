import json
import os


def load_json(json_file_name):
    if not os.path.exists(json_file_name):
        raise FileNotFoundError(f"JSON file {json_file_name} does not exist or you don't have permission to read it.")

    with open(json_file_name) as json_file:
        data = json.load(json_file)

    return data


def load_json_string(json_file_name):

    if not os.path.exists(json_file_name):
        raise FileNotFoundError(f"JSON file {json_file_name} does not exist or you don't have permission to read it.")

    json_string = load_json(json_file_name)  
    dict = json.loads(json_string)

    # dict = load_json(json_file_name)  

    return dict


def save_json(json_file_name, json_dictionary=None):
    with open(json_file_name, 'w') as outfile:
        json.dump(json_dictionary, outfile)
