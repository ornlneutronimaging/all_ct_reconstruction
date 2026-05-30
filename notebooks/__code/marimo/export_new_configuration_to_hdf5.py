import logging


class ExportNewConfigurationToHDF5:
    def __init__(self, new_configuration: dict = None, hdf5_file_path: str = "new_configuration.hdf5"):
        self.new_configuration = new_configuration
        self.hdf5_file_path = hdf5_file_path

        logging.addLevelName(logging.INFO, "Export New Configuration To HDF5")
        logging.info("Initialized ExportNewConfigurationToHDF5 with parameters:")
        logging.info(f"new_configuration keys: {list(self.new_configuration.keys()) if self.new_configuration else 'None'}, hdf5_file_path: {self.hdf5_file_path}")









    def export(self):
        import h5py
        import numpy as np

        with h5py.File(self.hdf5_file_path, 'w') as hdf5_file:
            for key, value in self.new_configuration.items():
                if isinstance(value, np.ndarray):
                    hdf5_file.create_dataset(key, data=value)
                else:
                    hdf5_file.attrs[key] = value