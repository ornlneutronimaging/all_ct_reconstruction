from skimage.io import imread
import numpy as np
import multiprocessing as mp 
import dxchange

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


def load_list_of_tif(list_of_tiff, dtype=None):
    if dtype is None:
        dtype = np.uint16

    # init array
    first_image = dxchange.read_tiff(list_of_tiff[0])
    size_3d = [len(list_of_tiff), np.shape(first_image)[0], np.shape(first_image)[1]]
    data_3d_array = np.empty(size_3d, dtype=dtype)

    # load stack of tiff
    for _index, _file in enumerate(list_of_tiff):
        _array = dxchange.read_tiff(_file)
        data_3d_array[_index] = _array
    return data_3d_array


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
