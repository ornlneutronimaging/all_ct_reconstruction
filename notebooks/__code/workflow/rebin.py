from ipywidgets import interactive
from IPython.display import display
from IPython.core.display import HTML
import ipywidgets as widgets
import logging
from skimage.measure import block_reduce
import numpy as np
from tqdm import tqdm

from __code.parent import Parent


class Rebin(Parent):

    rebinning_method = np.mean

    def set_rebinning(self):
        display(widgets.Label('Rebinning factor:'))
        self.rebin_value = widgets.Dropdown(options=np.arange(2, 10), value=2)
        display(self.rebin_value)
     
    def execute_binning(self):
        logging.info(f"Rebinning using factor {self.rebin_value.value} ...")
        rebin_value = self.rebin_value.value
        block_size = (1, rebin_value, rebin_value)
        logging.info(f"\t{block_size = }")

        for _data_type in tqdm(master_3d_data.keys()):
            logging.info(f"\rebinning data type: {_data_type} ...")
            data_raw = self.parent.master_3d_data_array[_data_type]
            dtype = _data_raw.dtype
            print(f"\t{_data_raw.shape = }")
            print(f"\t(type(_data_raw) = {type(data_raw)}")
            print(f"\t{block_size = }")
            print(f"\t{dtype = }")
            _data_rebinned = block_reduce(data_raw, 
                                        block_size=block_size, 
                                        func=self.rebinning_method,
                                        func_kwargs={'dtype': dtype})
            # master_3d_data[_data_type] = _data_rebinned[:]
            logging.info(f"\rebinning data type: {_data_type} ... Done!")
      
        logging.info(f"Rebinning done!")
