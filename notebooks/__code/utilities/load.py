from skimage.io import imread
import numpy as np
import multiprocessing as mp 

from NeuNorm.normalization import Normalization

from __code.utilities.files import retrieve_list_of_tif
from __code import LOAD_DTYPE


def _worker(fl):
    return (imread(fl).astype(LOAD_DTYPE)).swapaxes(0,1)


def load_data_using_multithreading(list_tif, combine_tof=False):
    with mp.Pool(processes=40) as pool:
        data = pool.map(_worker, list_tif)

    if combine_tof:
        return np.array(data).sum(axis=0)
    else:
        return np.array(data)


def load_data(folder):
    list_tif = retrieve_list_of_tif(folder)
    o_norm = Normalization()
    o_norm.load(list_tif)
    return o_norm.data['sample']['data']


def load_list_of_tif(list_of_tiff):
    o_norm = Normalization()
    o_norm.load(list_of_tiff)
    return o_norm.data['sample']['data']

def load_tiff(tif_file_name):
    o_norm = Normalization()
    o_norm.load(tif_file_name)
    return np.squeeze(o_norm.data['sample']['data'])


def load_data_using_imread(folder):
    list_tif = retrieve_list_of_tif(folder)
    data = []
    for _file in list_tif:
        data.append(_hype_loader_sum)
        data.append((imread(_file).astype(np.float32)))
    return data
