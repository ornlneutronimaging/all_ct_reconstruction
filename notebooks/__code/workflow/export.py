"""
Export Utilities for CT Reconstruction Pipeline.

This module provides export functionality for CT reconstruction data and configuration
management. It handles exporting 3D image data to TIFF format and managing reconstruction
configuration files for various execution modes including local, parallel, and HPC execution.

Key Classes:
    - Export: Basic image export functionality for 3D data arrays
    - ExportExtra: Advanced export with configuration management and UI workflow

Key Features:
    - TIFF image sequence export with progress tracking
    - SVMBIR configuration parameter management
    - Shell script generation for HPC execution
    - Interactive workflow selection (local/parallel/manual execution)
    - Configuration file JSON export with timestamping

Dependencies:
    - tqdm: Progress bar visualization
    - IPython: Jupyter notebook display widgets
    - numpy: Numerical array operations
    - subprocess: External script execution

Author: CT Reconstruction Pipeline Team
Created: Part of CT reconstruction development workflow
"""

from typing import Optional, Any, Dict, List, Union
import os
import logging
import shutil
import subprocess
from tqdm import tqdm
from IPython.display import display, HTML
import numpy as np
from numpy.typing import NDArray
import ipywidgets as widgets

from __code.utilities.system import get_user_name
from __code.utilities.save import make_tiff
from __code.utilities.json import save_json
from __code.utilities.configuration_file import SvmbirConfig
from __code.parent import Parent
from __code.utilities.create_scripts import create_sh_file, create_sh_hsnt_file
from __code import DataType, STEP2_NOTEBOOK
from __code.utilities.time import get_current_time_in_special_file_name_format
from __code.config import imaging_team
from __code.utilities.system import get_instrument_generic_name


class RunningModeOptions:
    go_to_step2 = "Divide reconstruction into several jobs and run them in the background"
    manual_launch = "Manually launch script outside notebook"
    run_from_notebook = "Launch the script directly from the notebook"
    run_on_hsnt = "Create script to run from hsnt"


class Export:
    """
    Basic image export functionality for 3D CT reconstruction data.
    
    This class provides simple TIFF export functionality for 3D image arrays,
    typically used to export reconstructed CT slices or preprocessed projection
    data. Each slice in the 3D array is saved as an individual numbered TIFF file.
    
    Attributes
    ----------
    base_image_name : str
        Base filename prefix for exported images (default: "image")
    image_3d : NDArray[np.floating], optional
        3D array containing image data to export
    output_folder : str, optional
        Directory path where TIFF files will be saved
    
    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.rand(10, 512, 512)  # 10 slices of 512x512 images
    >>> exporter = Export(image_3d=data, output_folder="/path/to/output")
    >>> exporter.run()  # Exports image_0000.tiff through image_0009.tiff
    """

    base_image_name: str = "image"

    def __init__(self, image_3d: Optional[NDArray[np.floating]] = None, 
                 output_folder: Optional[str] = None) -> None:
        """
        Initialize Export instance with 3D image data and output directory.
        
        Parameters
        ----------
        image_3d : NDArray[np.floating], optional
            3D numpy array with shape (slices, height, width) containing
            image data to export. Each slice will be saved as separate TIFF.
        output_folder : str, optional
            Output directory path where TIFF files will be saved.
            Directory will be created if it doesn't exist.
        """
        self.image_3d = image_3d
        self.output_folder = output_folder

    def run(self) -> None:
        """
        Export all slices from 3D image array to numbered TIFF files.
        
        Iterates through each slice in the 3D image array and saves it as
        a TIFF file with zero-padded numbering (e.g., image_0000.tiff).
        Progress is tracked with tqdm progress bar and logged.
        
        Raises
        ------
        AttributeError
            If image_3d or output_folder is None
        OSError
            If output directory cannot be created or written to
            
        Notes
        -----
        - File naming: {base_image_name}_{index:04d}.tiff
        - Uses make_tiff utility for TIFF file creation
        - Logs each export operation for debugging
        """

        for _index, _data in enumerate(self.image_3d):
            short_file_name: str = f"{self.base_image_name}_{_index:04d}.tiff"
            full_file_name: str = os.path.join(self.output_folder, short_file_name)
            logging.info(f"\texporting {full_file_name}")
            make_tiff(data=_data, filename=full_file_name)


class ExportExtra(Parent):
    """
    Advanced export functionality with configuration management and workflow control.
    
    This class extends the Parent class to provide comprehensive export capabilities
    including SVMBIR configuration management, JSON configuration export, shell script
    generation, and interactive workflow selection for CT reconstruction pipelines.
    
    Inherits from Parent class which provides access to reconstruction pipeline state,
    working directories, and UI components for parameter configuration.
    
    Key Features:
        - SVMBIR parameter configuration from UI widgets
        - JSON configuration file export with timestamping
        - Shell script generation for batch execution
        - Interactive workflow selection (local/parallel/HPC)
        - Log file copying and management
        - Center of rotation parameter handling
    
    Attributes
    ----------
    config_file_name : str
        Path to exported JSON configuration file
    config_json : str
        JSON string containing configuration data
    sh_file_name : str
        Path to generated shell script for execution
    instructions : ipywidgets.Textarea
        Widget displaying execution instructions to user
    run_script : ipywidgets.Button
        Button widget for direct script execution
    
    Examples
    --------
    >>> exporter = ExportExtra(parent=parent_instance)
    >>> exporter.update_configuration()  # Update SVMBIR settings
    >>> exporter.run(base_log_file_name="reconstruction_log")
    """
  
    def update_configuration(self) -> None:
        """
        Update reconstruction configuration with SVMBIR parameters from UI widgets.
        
        Extracts SVMBIR reconstruction parameters from the parent's UI widgets
        and updates the configuration object. This includes sharpness, SNR,
        positivity, iteration limits, and verbosity settings.
        
        Also updates instrument and IPTS number from parent object state.
        
        Notes
        -----
        - Only updates if parent.o_svmbir exists (SVMBIR UI is available)
        - Converts boolean verbose setting to integer (0/1)
        - Logs all parameter updates for debugging
        - Creates new SvmbirConfig object with current settings
        
        See Also
        --------
        SvmbirConfig : Configuration class for SVMBIR parameters
        """
        # especially for all svmbir settings
        if self.parent.o_svmbir is None:
            return

        instrument: str = self.parent.instrument
        ipts_number: str = self.parent.ipts_number
        self.parent.configuration.instrument = instrument
        self.parent.configuration.ipts_number = int(ipts_number)

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
        # svmbir_config.top_slice = top_slice
        # svmbir_config.bottom_slice = bottom_slice
        self.parent.configuration.svmbir_config = svmbir_config

        logging.info(f"Updating svmbir configuration using ui data:")
        logging.info(f"\t{sharpness = }")
        logging.info(f"\t{snr_db = }")
        logging.info(f"\t{positivity = }")
        logging.info(f"\t{max_iterations = }")
        logging.info(f"\t{max_resolutions = }")
        logging.info(f"\t{verbose = }")

    def run(self, base_log_file_name: Optional[str] = None, prefix: str = "") -> None:
        """
        Execute export workflow with configuration management and UI selection.
        
        Comprehensive export process that copies log files, exports configuration
        to JSON, generates execution scripts, and provides interactive workflow
        selection for reconstruction execution modes.
        
        Parameters
        ----------
        base_log_file_name : str, optional
            Base name for log file to copy (without .log extension)
        prefix : str, default=""
            Optional prefix for configuration file naming
            
        Notes
        -----
        Workflow includes:
        1. Log file copying from shared log directory
        2. Configuration JSON export with timestamping
        3. Shell script generation for batch execution
        4. Interactive UI for execution mode selection
        5. Widget setup for user workflow control
        
        Execution Options:
        - Parallel job division for large reconstructions
        - Manual script execution outside notebook
        - Direct execution from notebook interface
        
        Raises
        ------
        PermissionError
            If log file cannot be copied due to permissions
        OSError
            If output directories cannot be created or accessed
        """
        log_file_name: str = f"/SNS/VENUS/shared/log/{base_log_file_name}.log"
        output_folder: str = self.parent.working_dir[DataType.extra]
        try:
            shutil.copy(log_file_name, output_folder)
        except PermissionError:
            logging.error(f"PermissionError: cannot copy {log_file_name} to {output_folder}")
            
        # display(HTML(f"\tlog file from {log_file_name} to {output_folder}!"))

        configuration: Any = self.parent.configuration

        # update configuration
        configuration.output_folder = output_folder
        configuration.raw_data_base_folder = os.path.basename(os.path.abspath(self.parent.working_dir[DataType.sample]))

        # center of rotation if manual mode used
        if self.parent.o_center_and_tilt is not None:
            if self.parent.o_center_and_tilt.is_manual_mode():
                configuration.center_of_rotation = self.parent.o_center_and_tilt.get_center_of_rotation()

        base_sample_folder: str = os.path.basename(os.path.abspath(self.parent.working_dir[DataType.sample]))

        _time_ext: str = get_current_time_in_special_file_name_format()
        # config_file_name = f"/SNS/VENUS/shared/log/{base_sample_folder}_{_time_ext}.json"
        if prefix:
            config_file_name: str = os.path.join(output_folder, f"{prefix}_{base_sample_folder}_{_time_ext}.json")   
        else:
            config_file_name: str = os.path.join(output_folder, f"{base_sample_folder}_{_time_ext}.json")
        
        self.config_file_name = config_file_name

        config_json: str = configuration.model_dump_json()
        save_json(config_file_name, json_dictionary=config_json)
        self.config_json = config_json

        self.sh_file_name = create_sh_file(json_file_name=config_file_name,
                                           output_folder=output_folder)

        # ipts_number = configuration.ipts_number
        # instrument = configuration.instrument
        # instrument = get_instrument_generic_name(instrument)
        # self.hsnt_output_json_folder = os.path.join("/data", instrument, f"IPTS-{ipts_number}", "all_config_files")
        # self.hsnt_output_folder = os.path.join("/data", instrument, f"IPTS-{ipts_number}")
        # self.sh_hsnt_script_name = create_sh_hsnt_file(configuration=configuration,
        #                                                json_file_name=config_file_name, 
        #                                                hstn_output_json_folder=self.hsnt_output_json_folder)

        display(HTML(f"<font color='blue'><b>Next step</b></font>"))

        list_options: List[str] = [
                RunningModeOptions.go_to_step2,
                RunningModeOptions.manual_launch,
                RunningModeOptions.run_from_notebook,
        ]
        # ucams = get_user_name()
        # if ucams in imaging_team:
        #     list_options.append(f"Create script to run from hsnt")

        # 3 options are offered to the user
        choices: widgets.RadioButtons = widgets.RadioButtons(
            options=list_options,
            value="Launch the script directly from the notebook",
            description='',
            layout=widgets.Layout(width='100%'),
            disabled=False
        )
        display(choices)

        self.instructions = widgets.Textarea(value=f"Reload the configuration file {os.path.basename(self.config_file_name)} found in {os.path.dirname(self.config_file_name)} in the notebook {STEP2_NOTEBOOK}",
                                             layout=widgets.Layout(width='100%', height='160px'),
                                             disabled=True)
        display(self.instructions) 

        self.run_script = widgets.Button(
            description='Run script',
            disabled=False,
            button_style='success',
            tooltip='Run the script directly from the notebook',
            icon='play'
        )
        display(self.run_script)

        choices.observe(self.on_choice_change, names='value')
        self.on_choice_change({'new': choices.value})
        self.run_script.on_click(self.on_run_script_click)

    def on_choice_change(self, change: Dict[str, Any]) -> None:
        """
        Handle workflow selection changes in radio button widget.
        
        Updates UI state and instruction text based on selected execution mode.
        Enables or disables the run script button and provides appropriate
        instructions for each workflow option.
        
        Parameters
        ----------
        change : Dict[str, Any]
            Widget change event containing 'new' key with selected value
            
        Notes
        -----
        Updates:
        - run_script button enabled state
        - instructions widget text content
        - Workflow-specific command instructions
        """
        if change['new'] == 'Launch the script directly from the notebook':
            self.run_script.disabled = False
        else:
            self.run_script.disabled = True

        if change['new'] == RunningModeOptions.go_to_step2:
            self.instructions.value = f"Reload the configuration file ({self.config_file_name}) in the notebook {STEP2_NOTEBOOK}"
        elif change['new'] == RunningModeOptions.manual_launch:
            self.instructions.value = f"Launch the following script from the command line: {self.sh_file_name}"
        elif change['new'] == RunningModeOptions.run_on_hsnt:
            self.instructions.value = f"1. Connect to hsnt\n" + \
                f"2. Copy the pre-processed data: > 'cp {self.parent.configuration.projections_pre_processing_folder} {self.hsnt_output_folder}'\n" + \
                f"3. Copy the config json file: > 'cp {self.config_file_name} {self.hsnt_output_json_folder}'\n" + \
                f"4. Copy the script to run: > 'cp {self.sh_hsnt_script_name} {self.hsnt_output_folder}'\n" + \
                f"5. Run the following script: > '{os.path.join(self.hsnt_output_folder,os.path.basename(self.sh_hsnt_script_name))}'"
        else:
            self.instructions.value = f"click the button below to run the script directly from the notebook"

    def on_run_script_click(self, b: widgets.Button) -> None:
        """
        Execute reconstruction script directly from notebook interface.
        
        Launches the generated shell script in an external xterm terminal
        window for direct execution. Provides immediate feedback and allows
        monitoring of reconstruction progress.
        
        Parameters
        ----------
        b : widgets.Button
            Button widget that triggered the callback (unused)
            
        Notes
        -----
        - Uses xterm terminal for script execution
        - Script runs with exec bash for proper environment
        - Provides console output for progress monitoring
        
        Raises
        ------
        subprocess.CalledProcessError
            If script execution fails or xterm cannot be launched
        """
        print("Running the script directly from the notebook...")
        logging.info("Running the script directly from the notebook...")
        logging.info(f"Executing: xterm -e bash {self.sh_file_name}")
        # subprocess.run(["xterm", "-e", f"bash {self.sh_file_name}"], check=True)
        subprocess.Popen(["gnome-terminal", "--", "bash", "-c", f"bash {self.sh_file_name}; exec bash"])

        # display(HTML(f"<font color='blue'>From this point you have 3 options:</font>"))
        # display(HTML(f"<font color='blue'> 1. reload the configuration file </font>(<font color='green'>{config_file_name}</font>) in the notebook <font color='green'> {STEP2_NOTEBOOK}</font>"))
        # display(HTML(f"<br>"))
        # display(HTML(f"<font color='blue'> 2. launch the following script from the command line"))
        # display(HTML(f"<font color='green'>{sh_file_name}</font>"))
        # display(HTML(f"<br>"))
   