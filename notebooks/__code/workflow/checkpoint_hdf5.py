import os
import h5py
import numpy as np
import logging
import ipywidgets as widgets
from IPython.display import display
from numpy.typing import NDArray
import json

from __code.parent import Parent
from __code import DataType
from __code.utilities.time import get_current_time_in_special_file_name_format
from __code.utilities.file_folder_browser import FileFolderBrowser
from __code.utilities.configuration_file import SvmbirConfig, MbirjaxConfig
from __code.utilities.json import NumpyEncoder

CHECKPOINT_HDF5_FILTERS = {"HDF5 (.hdf5)": "*.hdf5", "HDF5 (.h5)": "*.h5"}
CHECKPOINT_FILENAME_SUFFIX = "_raw_checkpoint"


class CheckpointHdf5(Parent):
    """Save and load a raw-data checkpoint HDF5 file.

    The checkpoint is written after the raw data have been loaded and
    integrated (master_3d_data_array + final_list_of_angles are populated)
    but BEFORE any pre-processing (crop, outlier removal, normalisation …).
    Loading the checkpoint in step 2 restores exactly that state so the user
    can iterate on pre-processing without re-running the expensive step 1.

    HDF5 layout
    -----------
    raw/sample          : float32 array [N_angles, H, W]
    raw/ob              : float32 array [N_ob, H, W]   (optional – may be absent)
    angles/deg          : float32 array [N_angles]
    metadata/detector   : bytes  (UTF-8 string, e.g. "TimePix" or "CCD")
    """

    hdf5_output_folder: str = ""
    hdf5_input_file: str = ""

    # ------------------------------------------------------------------ save

    def select_output_folder(self) -> None:
        """Let the user browse to a folder where the HDF5 will be saved."""
        from __code.utilities.file_folder_browser import FileFolderBrowser

        start_dir = os.path.abspath(self.parent.working_dir[DataType.processed])
        logging.info(f"Selecting HDF5 output folder (start: {start_dir}) ...")

        self.output = widgets.Output()
        display(self.output)
    
        self.o_browser = FileFolderBrowser(working_dir=start_dir,
                                      ipts_folder=self.parent.working_dir[DataType.ipts],
                                      next_function=self.export)
        self.o_browser.select_output_folder_with_new(instruction="Select folder to save HDF5 checkpoint")
    
    def export(self, folder) -> None:
        """Save master_3d_data_array and final_list_of_angles to an HDF5 file."""
        logging.info("Exporting raw-data checkpoint to HDF5 ...")
        self.o_browser.list_output_folders_ui.shortcut_buttons.close() # close the jump to shared and home buttons 
        with self.output:
            self.output.clear_output()
            display(widgets.HTML(f"<b>Exporting checkpoint to HDF5...</b><br/>"))  

        # output_folder = getattr(self.parent, "hdf5_output_folder", "")
        output_folder = os.path.abspath(folder)
        if not output_folder:
            if type(self.parent.working_dir[DataType.sample]) == str:
                output_folder = os.path.dirname(self.parent.working_dir[DataType.sample])
            else:
                output_folder = os.path.dirname(self.parent.working_dir[DataType.sample][0])
            logging.warning(f"No output folder set – using {output_folder}")

        base_name = os.path.basename(self.parent.working_dir[DataType.sample][0]) if self.parent.working_dir[DataType.sample] else "unknown"

        _time_ext = get_current_time_in_special_file_name_format()
        filename = f"{base_name}_{_time_ext}_step1.hdf5"
        full_path = os.path.join(output_folder, filename)
        logging.info(f"\tOutput file: {full_path}")

        sample_array = self.parent.master_3d_data_array[DataType.sample]
        ob_array = self.parent.master_3d_data_array.get(DataType.ob, None)
        dc_array = self.parent.master_3d_data_array.get(DataType.dc, None)
        list_of_angles_deg = np.array(self.parent.final_list_of_angles, dtype=np.float32)
        detector_name = getattr(self.parent, "detector_name", "unknown")

        CheckpointHdf5._create_hdf5(
            sample_paths=self.parent.working_dir[DataType.sample],
            full_path=full_path,
            sample_array=sample_array,
            ob_array=ob_array,
            dc_array=dc_array,
            list_of_angles_deg=list_of_angles_deg,
            detector_name=detector_name,
            config=self.parent.configuration,
            working_dir=self.parent.working_dir,
        )
        
        logging.info("Done saving raw-data checkpoint.")
        with self.output:
            self.output.clear_output()
            display(widgets.HTML(
                f"<b>Checkpoint saved to:</b><br/><code>{full_path}</code>"
            ))

    # ------------------------------------------------------------------ load

    @staticmethod
    def _create_hdf5(full_path: str = "", 
                     sample_array: NDArray[np.floating] = None, 
                     ob_array: NDArray[np.floating] = None, 
                     dc_array: NDArray[np.floating] = None, 
                     list_of_angles_deg: NDArray[np.floating] = None, 
                     detector_name: str = "unknown",
                     config: dict = None,
                     working_dir: dict = None) -> None:
        
        if full_path == "":
            logging.error("No full_path provided for HDF5 export.")
            raise ValueError("full_path must be provided to create HDF5 checkpoint.")
        
        logging.info("Data to export to HDF5:")
        logging.info(f"{sample_array.shape =}")
        logging.info(f"{ob_array.shape if ob_array is not None else 'N/A'}")
        logging.info(f"{dc_array.shape if dc_array is not None else 'N/A'}")
        logging.info(f"Detector: {detector_name}")
        logging.info(f"Config: {config}")
        logging.info(f"Working dir: {working_dir}")
        logging.info(f"{len(list_of_angles_deg)} angles (deg): {list_of_angles_deg[:5]} ...")

        with h5py.File(full_path, "w") as f:
            f.create_dataset("raw/sample", data=np.array(sample_array, dtype=np.float32))
            if ob_array is not None:
                f.create_dataset("raw/ob", data=np.array(ob_array, dtype=np.float32))
            if dc_array is not None:
                f.create_dataset("raw/dc", data=np.array(dc_array, dtype=np.float32))
            if list_of_angles_deg is not None:
                f.create_dataset("angles/deg", data=list_of_angles_deg)
            f.create_group("metadata")
            f.create_dataset("metadata/config", data=json.dumps(config.model_dump(), cls=NumpyEncoder))
            f["metadata"].attrs["detector"] = detector_name
            f["metadata"].attrs["working_dir"] = json.dumps(working_dir)
        
    def select_input_file(self) -> None:
        """Let the user browse to an existing HDF5 checkpoint file."""

        self.data_type = DataType.hdf5
        start_dir = self.parent.working_dir[DataType.processed]
           
        # fall back to home if sample dir is not set yet
        if not start_dir or not os.path.exists(start_dir):
            start_dir = os.path.expanduser("~")

        logging.info(f"Selecting HDF5 checkpoint file (start: {start_dir}) ...")

        # def file_selected(file_path):
        #     self.parent.hdf5_input_file = file_path
        #     logging.info(f"HDF5 input file set to: {file_path}")
        #     display(widgets.HTML(f"<b>Selected file:</b> {file_path}"))

        self.output = widgets.Output()
        display(self.output)

        o_browser = FileFolderBrowser(working_dir=start_dir,
                                      next_function=self.load)
        o_browser.select_file(
            instruction="Select HDF5 checkpoint file saved from Step 1",
            filters=CHECKPOINT_HDF5_FILTERS,
            default_filter="HDF5 (.hdf5)",
        )

    def load(self, file_path: str) -> None:
        """Restore master_3d_data_array and final_list_of_angles from an HDF5 checkpoint."""
        if not file_path or not os.path.exists(file_path):
            raise FileNotFoundError(
                f"HDF5 checkpoint file not found: '{file_path}'. "
                "Please run select_hdf5_input_file() first."
            )

        with self.output:
            self.output.clear_output()
            display(widgets.HTML(f"<b>Loading checkpoint from HDF5...</b><br/>"))
        logging.info(f"Loading raw-data checkpoint from: {file_path}")

        with h5py.File(file_path, "r") as f:
            sample_array = f["raw/sample"][:]
            ob_array = f["raw/ob"][:] if "raw/ob" in f else None
            dc_array = f["raw/dc"][:] if "raw/dc" in f else None
            list_of_angles_deg = list(f["angles/deg"][:])
            config_json = f["metadata/config"][()] if "metadata/config" in f else None
            working_dir = f["metadata"].attrs.get("working_dir", None)

        self.parent.master_3d_data_array[DataType.sample] = sample_array
        if ob_array is not None:
            self.parent.master_3d_data_array[DataType.ob] = ob_array
        if dc_array is not None:
            self.parent.master_3d_data_array[DataType.dc] = dc_array
        self.parent.final_list_of_angles = list_of_angles_deg
        self.parent.final_list_of_angles_rad = [np.deg2rad(float(a)) for a in list_of_angles_deg]

        # derive working_dir so that downstream methods that rely on it work
        self.parent.working_dir = json.loads(working_dir) if working_dir is not None else {}

        logging.info(
            f"\tLoaded sample array shape : {sample_array.shape}\n"
            f"\tLoaded OB array shape     : {ob_array.shape if ob_array is not None else 'N/A'}\n"
            f"\tLoaded DC array shape     : {dc_array.shape if dc_array is not None else 'N/A'}\n"
            f"\tLoaded {len(list_of_angles_deg)} angles (deg): {list_of_angles_deg[:5]} ...\n"
            f"\tconfig: {config_json =}\n"
            f"\tworking_dir: {working_dir =}\n"
        )
        with self.output:
            display(widgets.HTML(
                f"<b>Checkpoint loaded:</b><br/>"
                f"<ul>"
                f"<li>Sample array shape: {sample_array.shape}</li>"
                f"<li>OB array shape: {ob_array.shape if ob_array is not None else 'N/A'}</li>"
                f"<li>DC array shape: {dc_array.shape if dc_array is not None else 'N/A'}</li>"
                f"<li>Number of angles: {len(list_of_angles_deg)}</li>"
                f"</ul>"
            ))
            
    def update_config_for_export(self, data_type: DataType) -> None:
        """Update config dictionary to export to HDF5"""
        
        list_of_angles: NDArray[np.floating] = np.array(self.parent.final_list_of_angles)
        list_of_angles_rad: NDArray[np.floating] = np.array([np.deg2rad(float(_angle)) for _angle in list_of_angles])
        self.parent.configuration.list_of_angles = list(list_of_angles_rad)
        
        output_folder = self.parent.working_dir[data_type]
        self.parent.configuration.output_folder = output_folder
        
        instrument: str = self.parent.instrument
        ipts_number: str = self.parent.ipts_number
        self.parent.configuration.instrument = instrument
        self.parent.configuration.ipts_number = int(ipts_number)

        # svmbir parameters
        if self.parent.o_svmbir is not None:
            sharpness: float = self.parent.o_svmbir.sharpness_ui.value
            snr_db: float = self.parent.o_svmbir.snr_db_ui.value
            positivity: float = self.parent.o_svmbir.positivity_ui.value
            max_iterations: int = self.parent.o_svmbir.max_iterations_ui.value
            max_resolutions: int = self.parent.o_svmbir.max_resolutions_ui.value
            verbose: int = 1 if self.parent.o_svmbir.verbose_ui.value else 0

            svmbir_config: SvmbirConfig = SvmbirConfig()
            svmbir_config.sharpness = sharpness
            svmbir_config.snr_db = snr_db
            svmbir_config.positivity = positivity
            svmbir_config.max_iterations = max_iterations
            svmbir_config.verbose = verbose
            self.parent.configuration.svmbir_config = svmbir_config

        # mbirjax parameters
        if self.parent.o_mbirjax is not None:
            mbirjax_config: MbirjaxConfig = MbirjaxConfig()
            mbirjax_config.sharpness = self.parent.o_mbirjax.sharpness_ui.value
            mbirjax_config.snr_db = self.parent.o_mbirjax.snr_db_ui.value
            mbirjax_config.positivity = self.parent.o_mbirjax.positivity_ui.value
            mbirjax_config.max_iterations = self.parent.o_mbirjax.max_iterations_ui.value
            mbirjax_config.verbose = self.parent.o_mbirjax.verbose_ui.value
            self.parent.configuration.mbirjax_config = mbirjax_config

    def create_hdf5_with_config_and_preprocessed_data(self) -> None:
        """Convenience method to export HDF5 checkpoint at the end of step 2."""
        logging.info("Exporting HDF5 checkpoint at the end of step 2 ...")
        self.data_type = DataType.extra
        
        normalized_images_log: NDArray[np.floating] = self.parent.normalized_images_log
        list_of_angles_deg: NDArray[np.floating] = np.array(self.parent.final_list_of_angles)

        output_folder = self.parent.working_dir[self.data_type]
        _time_ext = get_current_time_in_special_file_name_format()
        sample_basename = os.path.basename(self.parent.working_dir[DataType.sample][0]) if self.parent.working_dir[DataType.sample] else ["unknown"]

        config_dict = self.parent.configuration

        filename = f"{sample_basename}_step2_{_time_ext}.hdf5"
        full_path = os.path.join(output_folder, filename)
        logging.info(f"\tOutput file: {full_path}")

        CheckpointHdf5._create_hdf5(
            full_path=full_path,
            sample_array=normalized_images_log,
            list_of_angles_deg=list_of_angles_deg,
            # detector_name=self.parent.detector_name,
            config=config_dict,
        )
        logging.info("Done exporting HDF5 checkpoint at the end of step 2.")
        