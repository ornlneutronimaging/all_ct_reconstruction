from pathlib import Path


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
