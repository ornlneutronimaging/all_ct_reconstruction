import numpy as np
import logging

from __code.parent import Parent
from __code import DataType


class CombineObDc(Parent):

    def run(self, ignore_dc=False):

        if self.parent.master_3d_data_array[DataType.ob] is None:
            logging.warning(f"Combine obs: No ob data found, skipping combination.")
            return

        if ignore_dc:
            logging.info(f"Combine obs:")
            list_to_combine = [DataType.ob]
        else:
            logging.info(f"Combine obs and dcs:")
            list_to_combine = [DataType.ob, DataType.dc]

        master_3d_data_array = self.parent.master_3d_data_array
        self.parent.master_3d_data_array[DataType.sample] = np.array(self.parent.master_3d_data_array[DataType.sample])

        for _data_type in list_to_combine:
#           if self.parent.list_of_images[_data_type] is not None:
            if master_3d_data_array[_data_type] is not None:
                logging.info(f"\t{_data_type} -> {np.shape(master_3d_data_array[_data_type])}")
                logging.info(f"\t             -> data type: {master_3d_data_array[_data_type].dtype}")
                # if len(self.parent.list_of_images[_data_type]) == 1: # only 1 image
                if len(master_3d_data_array[_data_type]) == 1: # only 1 image
                    continue
                else:
                    _combined_array = np.median(np.array(master_3d_data_array[_data_type]), axis=0).astype(np.ushort)
                    master_3d_data_array[_data_type] = _combined_array[:]
                    logging.info(f"\t{_data_type} -> {np.shape(master_3d_data_array[_data_type])}")
            else:
                logging.info(f"\t{_data_type} skipped!")

        self.parent.master_3d_data_array = master_3d_data_array

        if ignore_dc:
            logging.info(f"Combined obs!")    
        else:
            logging.info(f"Combined obs and dcs done !")    
