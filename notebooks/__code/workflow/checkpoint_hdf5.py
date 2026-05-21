import os
import h5py
import numpy as np
import logging
import ipywidgets as widgets
from IPython.display import display

from __code.parent import Parent
from __code import DataType
from __code.utilities.time import get_current_time_in_special_file_name_format


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

        start_dir = self.parent.working_dir[DataType.processed]
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
        output_folder = folder
        if not output_folder:
            if type(self.parent.working_dir[DataType.sample]) == str:
                output_folder = os.path.dirname(self.parent.working_dir[DataType.sample])
            else:
                output_folder = os.path.dirname(self.parent.working_dir[DataType.sample][0])
            logging.warning(f"No output folder set – using {output_folder}")

        _time_ext = get_current_time_in_special_file_name_format()
        filename = f"raw_checkpoint_{_time_ext}.hdf5"
        full_path = os.path.join(output_folder, filename)
        logging.info(f"\tOutput file: {full_path}")

        sample_array = self.parent.master_3d_data_array[DataType.sample]
        ob_array = self.parent.master_3d_data_array.get(DataType.ob, None)
        list_of_angles = np.array(self.parent.final_list_of_angles, dtype=np.float32)

        detector_name = getattr(self.parent, "detector_name", "unknown")

        with h5py.File(full_path, "w") as f:
            f.create_dataset("raw/sample", data=np.array(sample_array, dtype=np.float32))
            if ob_array is not None:
                f.create_dataset("raw/ob", data=np.array(ob_array, dtype=np.float32))
            f.create_dataset("angles/deg", data=list_of_angles)
            f.create_group("metadata")
            f["metadata"].attrs["detector"] = detector_name

        logging.info("Done saving raw-data checkpoint.")
        with self.output:
            self.output.clear_output()
            display(widgets.HTML(
                f"<b>Checkpoint saved to:</b><br/><code>{full_path}</code>"
            ))

    # ------------------------------------------------------------------ load

    def select_input_file(self) -> None:
        """Let the user browse to an existing HDF5 checkpoint file."""
        from __code.utilities.file_folder_browser import FileFolderBrowser

        if type(self.parent.working_dir[DataType.sample]) == str:
            start_dir = self.parent.working_dir[DataType.sample]
        else:
            start_dir = self.parent.working_dir[DataType.sample][0]

        # fall back to home if sample dir is not set yet
        if not start_dir or not os.path.exists(start_dir):
            start_dir = os.path.expanduser("~")

        logging.info(f"Selecting HDF5 checkpoint file (start: {start_dir}) ...")

        def file_selected(file_path):
            self.parent.hdf5_input_file = file_path
            logging.info(f"HDF5 input file set to: {file_path}")
            display(widgets.HTML(f"<b>Selected file:</b> {file_path}"))

        o_browser = FileFolderBrowser(working_dir=start_dir,
                                      next_function=file_selected)
        o_browser.select_file(
            instruction="Select HDF5 checkpoint file saved from Step 1",
            filters=CHECKPOINT_HDF5_FILTERS,
            default_filter="HDF5 (.hdf5)",
        )

    def load(self) -> None:
        """Restore master_3d_data_array and final_list_of_angles from an HDF5 checkpoint."""
        file_path = getattr(self.parent, "hdf5_input_file", "")
        if not file_path or not os.path.exists(file_path):
            raise FileNotFoundError(
                f"HDF5 checkpoint file not found: '{file_path}'. "
                "Please run select_hdf5_input_file() first."
            )

        logging.info(f"Loading raw-data checkpoint from: {file_path}")

        with h5py.File(file_path, "r") as f:
            sample_array = f["raw/sample"][:]
            ob_array = f["raw/ob"][:] if "raw/ob" in f else None
            list_of_angles = list(f["angles/deg"][:])

        self.parent.master_3d_data_array[DataType.sample] = sample_array
        if ob_array is not None:
            self.parent.master_3d_data_array[DataType.ob] = ob_array
        self.parent.final_list_of_angles = list_of_angles
        self.parent.final_list_of_angles_rad = [np.deg2rad(float(a)) for a in list_of_angles]

        # derive working_dir so that downstream methods that rely on it work
        self.parent.working_dir[DataType.sample] = os.path.dirname(file_path)

        logging.info(
            f"\tLoaded sample array shape : {sample_array.shape}\n"
            f"\tLoaded {len(list_of_angles)} angles (deg): {list_of_angles[:5]} ..."
        )
        display(widgets.HTML(
            f"<b>Checkpoint loaded:</b><br/>"
            f"<ul>"
            f"<li>Sample array shape: {sample_array.shape}</li>"
            f"<li>Number of angles: {len(list_of_angles)}</li>"
            f"</ul>"
        ))
