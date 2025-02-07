from ipywidgets import interactive
from IPython.display import display
from IPython.core.display import HTML
import ipywidgets as widgets
import logging
from skimage.measure import block_reduce
import numpy as np

from __code.parent import Parent


class Rebin(Parent):

    # rebinning_method = np.sum

    def set_rebinning(self):
        display(widgets.Label('Rebinning factor:'))
        self.rebin_value = widgets.Dropdown(options=np.arange(2, 10), value=2)
        display(self.rebin_value)
     
    def execute_binning(self):
        logging.info(f"Rebinning using factor {self.rebin_value.value} ...")
        rebin_value = self.rebin_value.value
        block_size = (1, rebin_value, rebin_value)
        logging.info(f"\t{block_size = }")

        normalized_images = self.parent.normalized_images
        self.parent.before_rebinning = normalized_images[:]

        logging.info(f"\rebinning normalized data ...")
        dtype = normalized_images.dtype
        # print(f"\t{data_raw.shape = }")
        # print(f"\t(type(_data_raw) = {type(data_raw)}")
        # print(f"\t{block_size = }")
        # print(f"\t{dtype = }")
        _data_rebinned = block_reduce(normalized_images, 
                                    block_size=block_size, 
                                    func=np.mean,
                                    func_kwargs={'dtype': dtype})
        self.parent.normalized_images = _data_rebinned[:]

        logging.info(f"\rebinning normalized data ... Done!")
      