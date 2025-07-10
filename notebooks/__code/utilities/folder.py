from pathlib import Path
import os


def find_first_real_dir(start_dir="./"):
    """return the first existing folder from the tree up"""

    if type(start_dir) is str:
        start_dir = Path(start_dir)

    if start_dir.exists():
        return start_dir

    dir = start_dir
    while not Path(dir).exists():
        dir = Path(dir).parent

    return dir


def check_folder_write_permission(folder_path):
    """
    Checks if the current user has write permission to the specified folder.

    Args:
        folder_path (str): The path to the folder.

    Returns:
        bool: True if write permission is granted, False otherwise.
    """
    return os.access(folder_path, os.W_OK)
