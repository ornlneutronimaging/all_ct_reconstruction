from ipywidgets import interactive
from IPython.display import display
from IPython.display import HTML
import ipywidgets as widgets
import logging
from skimage.measure import block_reduce
import numpy as np

from __code.parent import Parent
from __code import DataType


class Rebin(Parent):

    # rebinning_method = np.sum

    def set_rebinning(self):
        display(widgets.Label('Rebinning factor:'))
        self.rebin_value = widgets.Dropdown(options=np.arange(1, 10), value=2)
        display(self.rebin_value)
     
    def execute_binning_before_normalization(self):
        logging.info(f"Rebinning using factor {self.rebin_value.value} ...")
        rebin_value = self.rebin_value.value
        if rebin_value == 1:
            logging.info(f"Rebinning factor is 1, skipping rebinning ...")
            return

        block_size = (1, rebin_value, rebin_value)
        logging.info(f"\t{block_size = }")

        master_3d_data_array = self.parent.master_3d_data_array
        self.parent.before_rebinning = master_3d_data_array[DataType.sample][:]
        
        sample_raw_images = master_3d_data_array[DataType.sample]
        ob_raw_images = master_3d_data_array[DataType.ob]
        dc_raw_images = master_3d_data_array[DataType.dc]
                
        logging.info(f"\rebinning raw data ...")
        dtype = sample_raw_images.dtype
        _sample_data_rebinned = block_reduce(sample_raw_images, 
                                    block_size=block_size, 
                                    func=np.sum,
                                    func_kwargs={'dtype': dtype})
        self.parent.master_3d_data_array[DataType.sample] = _sample_data_rebinned[:]
        logging.info(f"\rebinning raw data ... Done!")

        if ob_raw_images is not None:
            logging.info(f"\rebinning ob data ...")
            _ob_data_rebinned = block_reduce(ob_raw_images, 
                                        block_size=block_size, 
                                        func=np.sum,
                                        func_kwargs={'dtype': dtype})
            self.parent.master_3d_data_array[DataType.ob] = _ob_data_rebinned[:]
            logging.info(f"\rebinning ob data ... Done!")

        if dc_raw_images is not None:
            logging.info(f"\rebinning dc data ...")
            _dc_data_rebinned = block_reduce(dc_raw_images, 
                                        block_size=block_size, 
                                        func=np.sum,
                                        func_kwargs={'dtype': dtype})
            self.parent.master_3d_data_array[DataType.dc] = _dc_data_rebinned[:]
            logging.info(f"\rebinning dc data ... Done!")


    def execute_binning_after_normalization(self):
        logging.info(f"Rebinning using factor {self.rebin_value.value} ...")
        rebin_value = self.rebin_value.value
        if rebin_value == 1:
            logging.info(f"Rebinning factor is 1, skipping rebinning ...")
            return
        
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
      