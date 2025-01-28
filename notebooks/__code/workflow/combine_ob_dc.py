import numpy as np
import logging

from __code.parent import Parent
from __code import DataType


class CombineObDc(Parent):

    def run(self):

        logging.info(f"Combine ob and dc:")
        master_3d_data_array = self.parent.master_3d_data_array
        self.parent.master_3d_data_array[DataType.sample] = np.array(self.parent.master_3d_data_array[DataType.sample])

        list_to_combine = [DataType.ob, DataType.dc]
        for _data_type in list_to_combine:
            if self.parent.list_of_images[_data_type] is not None:

                logging.info(f"\t{_data_type} -> {np.shape(master_3d_data_array[_data_type])}")
                logging.info(f"\t             -> data type: {master_3d_data_array[_data_type].dtype}")
                if len(self.parent.list_of_images[_data_type]) == 1: # only 1 image
                    continue
                else:
                    master_3d_data_array[_data_type] = np.median(np.array(master_3d_data_array[_data_type]), axis=0).astype(np.ushort)
                    logging.info(f"\t{_data_type} -> {np.shape(master_3d_data_array[_data_type])}")
            else:
                logging.info(f"\t{_data_type} skipped!")

        self.parent.master_3d_data_array = master_3d_data_array
        logging.info(f"Combined ob and dc done !")    
