import os
import h5py
import numpy as np
import logging

from __code.parent import Parent
from __code import DataType
from __code.utilities.time import get_current_time_in_special_file_name_format


class ExportHdf5(Parent):
    
    def export(self):
        
        logging.info("Exporting pre-reconstruction data to HDF5 format...")
        
        top_folder = os.path.abspath(self.parent.working_dir[DataType.sample])
        logging.info(f"\tinput folder: {top_folder}")
        
        output_folder = os.path.abspath(self.parent.working_dir[DataType.extra])
        logging.info(f"\toutput folder: {output_folder}")
        
        _time_ext: str = get_current_time_in_special_file_name_format()
        full_output_file_name = os.path.join(output_folder, f"{os.path.basename(top_folder)}_projections_pre_data_{_time_ext}.hdf5")
        logging.info(f"\tfull output file name: {full_output_file_name}")
        
        list_of_angles = np.array(self.parent.final_list_of_angles)
        logging.info(f"\tlist of angles (degrees): {list_of_angles}")
        
        list_of_angles_rad = np.array([np.deg2rad(float(_angle)) for _angle in list_of_angles])
        logging.info(f"\tlist of angles (radians): {list_of_angles_rad}")
        
        normalized_images_log = self.parent.normalized_images_log
        logging.info(f"\tshape of normalized images (log): {normalized_images_log.shape}")
        logging.info(f"\tmin of normalized images (log): {np.min(normalized_images_log)}")
        logging.info(f"\tmax of normalized images (log): {np.max(normalized_images_log)}")
        logging.info(f"\tmean of normalized images (log): {np.mean(normalized_images_log)}")
        
        with h5py.File(full_output_file_name, 'w') as hdf5_file:
            hdf5_file.create_group('tomo/pro')
            hdf5_file.create_dataset('tomo/pro/proj_mlog_to_recon', data=normalized_images_log, dtype=np.float32)
            
            hdf5_file.create_group('tomo/info')
            hdf5_file.create_dataset('tomo/info/ang_deg', data=list_of_angles, dtype=np.float32)
            hdf5_file.create_dataset('tomo/info/ang_rad', data=list_of_angles_rad, dtype=np.float32)
        
        logging.info("Done saving data to HDF5 file...")
        