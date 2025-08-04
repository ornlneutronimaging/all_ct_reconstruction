"""
JSON file handling utilities for CT reconstruction pipeline.

This module provides functions for loading and saving JSON configuration files
used throughout the CT reconstruction workflow. Includes error handling for
missing files and proper type checking.
"""

import json
import os
from typing import Any, Dict, Union, Optional


def load_json(json_file_name: str) -> Any:
    """
    Load and parse a JSON file.
    
    Args:
        json_file_name: Path to the JSON file to load
        
    Returns:
        Parsed JSON data (typically a dictionary)
        
    Raises:
        FileNotFoundError: If the JSON file doesn't exist or is not accessible
        json.JSONDecodeError: If the file contains invalid JSON
    """
    if not os.path.exists(json_file_name):
        raise FileNotFoundError(f"JSON file {json_file_name} does not exist or you don't have permission to read it.")

    with open(json_file_name) as json_file:
        data = json.load(json_file)

    return data


def load_json_string(json_file_name: str) -> Dict[str, Any]:
    """
    Load a JSON file that contains a JSON string and parse it to a dictionary.
    
    This function handles cases where the JSON file contains a JSON string
    that needs to be parsed twice (once to get the string, once to parse the string).
    
    Args:
        json_file_name: Path to the JSON file containing a JSON string
        
    Returns:
        Parsed dictionary from the JSON string
        
    Raises:
        FileNotFoundError: If the JSON file doesn't exist or is not accessible
        json.JSONDecodeError: If the file or string contains invalid JSON
    """
    if not os.path.exists(json_file_name):
        raise FileNotFoundError(f"JSON file {json_file_name} does not exist or you don't have permission to read it.")

    json_string: str = load_json(json_file_name)  
    dict_data: Dict[str, Any] = json.loads(json_string)

    return dict_data


def save_json(json_file_name: str, json_dictionary: Optional[Union[Dict[str, Any], str]] = None) -> None:
    """
    Save data to a JSON file.
    
    Args:
        json_file_name: Path where the JSON file should be saved
        json_dictionary: Data to save as JSON (dictionary or JSON string)
        
    Raises:
        TypeError: If json_dictionary is not JSON serializable
        PermissionError: If unable to write to the specified path
    """
    with open(json_file_name, 'w') as outfile:
        json.dump(json_dictionary, outfile)
