import os
import psutil
import gc


def get_memory_usage():
    """
    Calculate the total memory usage of the current process and its children.

    Returns
    -------
    float
        Total memory usage in MB.
    """
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    mem_usage_mb = mem_info.rss / (1024 ** 2)  # Convert bytes to MB

    # Include memory usage of child processes
    for child in process.children(recursive=True):
        try:
            mem_info = child.memory_info()
            mem_usage_mb += mem_info.rss / (1024 ** 2)
        except psutil.NoSuchProcess:
            continue

    return mem_usage_mb


def print_memory_usage(message="", end="\n"):
    """
    Print the total memory usage.
    """
    mem_usage = get_memory_usage()

    if mem_usage < 999:
        units = 'MB'
    else:
        mem_usage /= 1000
        units = 'GB'
    print(f"{message}: total memory usage = {mem_usage:.2f}{units}", end=end)


def retrieve_memory_usage():
    mem_usage = get_memory_usage()

    if mem_usage < 999:
        units = 'MB'
    else:
        mem_usage /= 1000
        units = 'GB'

    return f"{mem_usage:.2f}{units}"


def delete_array(object_array=None):
    '''delete array from memory '''
    object_array = None
    gc.collect()
    

def get_user_name():
    return os.getlogin()  # add user name to the log file name


def get_instrument_generic_name(instrument):
    """
    Get the generic name of the instrument.
    
    Parameters
    ----------
    instrument : str
        The instrument name.
        
    Returns
    -------
    str
        The generic name of the instrument.
    """
    instrument = instrument.lower()  
    
    if instrument.startswith("cg1d"):
        return "MARS"
    elif instrument.startswith("bl3"):
        return "SNAP"
    elif instrument.startswith("bl10"):
        return "VENUS"
    else:
        return instrument  # Return as is if not recognized
    