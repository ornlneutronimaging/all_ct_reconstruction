"""
System configuration and working directory management for CT reconstruction pipeline.

This module provides the System class which handles:
- Working directory selection and validation
- Instrument and facility configuration
- IPTS (Integrated Proposal Tracking System) number management
- File system path resolution
- User interface for directory browsing

The System class maintains global state for the current working environment
and provides methods for configuring the reconstruction pipeline.
"""

from typing import Optional, List, Dict, ClassVar, Any

from matplotlib.pylab import f
from __code import config
from __code.utilities.logging import setup_logging
import logging
import getpass
import glob
import os
import platform
from ipywidgets import widgets
from IPython.display import display
from IPython.display import HTML
from IPython.display import clear_output
import subprocess

from __code.utilities.folder import find_first_real_dir

# Instrument configuration by facility
list_instrument_per_facility: Dict[str, List[str]] = {
    'HFIR': ['CG1D'],
    'SNS': ['SNAP', 'VENUS']
}


class System:
    """
    System configuration manager for CT reconstruction pipeline.
    
    This class provides centralized management of system configuration including
    working directories, instrument settings, and IPTS numbers. It maintains
    global state across the reconstruction workflow and provides methods for
    directory selection and validation.
    
    Class Attributes:
        working_dir: Current working directory path
        start_path: Starting path for directory navigation
        ipts_number: Current IPTS number
    """

    working_dir: ClassVar[str] = ''
    start_path: ClassVar[str] = ''
    ipts_number: ClassVar[str] = ''

    @classmethod
    def select_working_dir(cls, 
                          debugger_folder: str = '', 
                          system_folder: str = '',
                          facility: str = 'HFIR',
                          instrument: str = 'CG1D',
                          ipts: Optional[str] = None,
                          instrument_to_exclude: Optional[List[str]] = None,
                          notebook: str = "N/A") -> None:
        """
        Display interface for selecting working directory and configuring system.
        
        This method provides an interactive interface for users to select their
        working directory, instrument, and IPTS number. It handles both debugging
        mode and production mode configurations.
        
        Args:
            debugger_folder: Folder to use in debugging mode
            system_folder: System folder path
            facility: Facility name ('SNS' or 'HFIR') 
            instrument: Instrument name ('VENUS', 'SNAP', 'CG1D')
            ipts: IPTS number string
            instrument_to_exclude: List of instruments to exclude from selection
            notebook: Name of the calling notebook for reference
            
        Note:
            In debugging mode, predefined paths from config are used.
            In production mode, an interactive file browser is displayed.
        """

        setup_logging(basename_of_log_file="system")
        logging.info(f"*** Starting system ***")

        try:

            debugging = config.debugging
            logging.info(f"debugging: {debugging}")
            
            if debugging:
                print("** Using Debugging Mode! **")

            display(HTML("""
                       <style>
                       .result_label {
                          font-style: bold;
                          color: red;
                          font-size: 18px;
                       }
                       </style>
                       """))

            logging.info(f"facility: {facility}")
            logging.info(f"instrument: {instrument}")
            logging.info(f"ipts: {ipts}")
            if ipts:
                if ipts.startswith('IPTS-'):
                    _, ipts_number = ipts.split('-')
                    cls.ipts_number = ipts_number
                    logging.info(f"{cls.ipts_number = }")

            full_list_instruments = cls.get_full_list_instrument(instrument_to_exclude=instrument_to_exclude)
            logging.info(f"full_list_instruments: {full_list_instruments}")
            
            full_list_instruments.sort()
            if instrument in full_list_instruments:
                default_instrument = instrument
            else:
                default_instrument = full_list_instruments[0]

            start_path = cls.get_start_path(debugger_folder=debugger_folder,
                                            system_folder=system_folder,
                                            instrument=default_instrument)
            logging.info(f"start_path: {start_path}")
            
            cls.start_path = start_path

            select_instrument_ui = widgets.HBox([widgets.Label("Select Instrument",
                                                      layout=widgets.Layout(width='20%')),
                                        widgets.Select(options=full_list_instruments,
                                                       value=default_instrument,
                                                       layout=widgets.Layout(width='20%'))])
            cls.instrument_ui = select_instrument_ui.children[1]
            cls.instrument_ui.observe(cls.check_instrument_input, names='value')

            help_ui = widgets.Button(description="HELP",
                                     button_style='info')
            help_ui.on_click(cls.select_ipts_help)

            top_hbox = widgets.HBox([widgets.Label("IPTS-"),
                                     widgets.Text(value="",
                                                  layout=widgets.Layout(width='10%')),
                                     widgets.Label("DOES NOT EXIST!",
                                                   layout=widgets.Layout(width='20%'))])
            cls.result_label = top_hbox.children[2]
            cls.ipts_number = top_hbox.children[1]
            cls.result_label.add_class("result_label")
            or_label = widgets.Label("OR")

            list_and_default_folders = cls.get_list_folders(start_path=start_path)
            user_list_folders = list_and_default_folders['user_list_folders']
            default_value = list_and_default_folders['default_value']

            bottom_hbox = widgets.HBox([widgets.Label("Select Folder",
                                               layout=widgets.Layout(width="20%")),
                                 widgets.Select(options=user_list_folders,
                                                value=default_value,
                                                layout=widgets.Layout(height='300px')),
                                 ])
            cls.user_list_folders = user_list_folders
            box = widgets.VBox([select_instrument_ui, top_hbox, or_label, bottom_hbox, help_ui])
            display(box)

            cls.working_dir_ui = bottom_hbox.children[1]
            cls.manual_ipts_entry_ui = top_hbox.children[1]
            cls.manual_ipts_entry_ui.observe(cls.check_ipts_input, names='value')

            cls.result_label.value = ""

            if ipts is not None:
                cls.working_dir_ui.value = ipts
                _, ipts_number = ipts.split('-')
                cls.ipts_number.value = ipts_number

        except:
            cls.working_dir = os.path.expanduser("~")
            display(HTML('<span style="font-size: 15px; color:blue">working dir set to -> ' + cls.working_dir +
                         '</span>'))

        # cls.log_use(notebook=notebook)

    # @classmethod
    # def log_use(cls, notebook="N/A"):
    #     """
    #     Log notebook usage for tracking and analytics purposes.
    #     
    #     This method logs when a user starts using a specific notebook,
    #     recording the timestamp, username, and notebook name for usage
    #     analytics and system monitoring.
    #     
    #     Args:
    #         notebook: Name of the notebook being used
    #         
    #     Note:
    #         Currently commented out. When enabled, logs to a central log file
    #         if the directory exists and the notebook is not run locally.
    #     """
    #     if os.path.exists(os.path.dirname(LOGGER_FILE)):
    #         # no dot log usage if notebooks are run locally
    #         username = getpass.getuser()
    #         date = get_current_time_in_special_file_name_format()
    #         data = [f"{date}: {username} started using {notebook}"]
    #         append_to_file(data=data, output_file_name=LOGGER_FILE)

    @classmethod
    def get_full_list_instrument(cls, instrument_to_exclude: Optional[List[str]] = None) -> List[str]:
        """
        Get complete list of available instruments across all facilities.
        
        This method compiles a comprehensive list of all instruments available
        across all supported facilities, with optional exclusion of specific
        instruments.
        
        Args:
            instrument_to_exclude: List of instrument names to exclude from the result
            
        Returns:
            List of available instrument names
            
        Note:
            Instruments are collected from all facilities defined in
            list_instrument_per_facility configuration.
        """

        list_instrument: List[str] = []
        for _key in list_instrument_per_facility.keys():
            _facility_list_instrument: List[str] = list_instrument_per_facility[_key]
            for _instr in _facility_list_instrument:
                if instrument_to_exclude is not None:
                    if _instr in instrument_to_exclude:
                        continue
                list_instrument.append(_instr)
        return list_instrument

    @classmethod
    def get_list_folders(cls, start_path: str = '') -> Dict[str, Any]:
        """
        Get list of accessible folders in the specified start path.
        
        This method scans the start path directory and returns a list of folders
        that the user can access, along with an appropriate default selection.
        It handles both debugging and production modes differently.
        
        Args:
            start_path: Base directory path to scan for folders
            
        Returns:
            Dictionary containing:
            - 'user_list_folders': List of folder names user can access
            - 'default_value': Recommended default folder selection
            
        Note:
            In debugging mode, uses predefined paths from config.
            In production mode, only shows folders with read access permissions.
        """
        debugging: bool = config.debugging

        if debugging:
            instrument: str = cls.get_instrument_selected()
            computer_name: str = cls.get_computer_name()
            start_path = config.debugger_instrument_folder[computer_name][instrument]
            cls.start_path = start_path

        list_folders: List[str] = sorted(glob.glob(os.path.join(start_path, '*')), reverse=True)
        short_list_folders: List[str] = [os.path.basename(_folder) for _folder in list_folders if os.path.isdir(_folder)]
        # short_list_folders = sorted(short_list_folders)

        # if user mode, only display folder user can access
        default_value: str = ''
        if not debugging:
            user_list_folders: List[str] = [os.path.basename(_folder) for _folder in list_folders if os.access(_folder, os.R_OK)]
            if len(user_list_folders) > 0:
                default_value = user_list_folders[0]
        else:  # debugging
            user_list_folders = short_list_folders
            default_value = config.project_folder
            if not (default_value in user_list_folders):
                if len(user_list_folders) > 0:
                    default_value = user_list_folders[0]

        return {'user_list_folders': user_list_folders,
                'default_value': default_value}

    @classmethod
    def get_facility_from_instrument(cls, instrument: str = 'CG1D') -> str:
        """
        Determine facility name from instrument name.
        
        This method looks up which facility hosts a given instrument by
        searching through the instrument-facility mapping configuration.
        
        Args:
            instrument: Name of the instrument to look up
            
        Returns:
            Facility name hosting the instrument (defaults to 'HFIR' if not found)
            
        Example:
            >>> System.get_facility_from_instrument('VENUS')
            'SNS'
            >>> System.get_facility_from_instrument('CG1D')
            'HFIR'
        """

        for _facility in list_instrument_per_facility:
            list_instrument: List[str] = list_instrument_per_facility[_facility]
            if instrument in list_instrument:
                return _facility

        return 'HFIR'

    @classmethod
    def get_instrument_selected(cls) -> str:
        """
        Get currently selected instrument from UI.
        
        Returns:
            Name of the currently selected instrument
            
        Note:
            Accesses the instrument selection widget value.
        """
        return cls.instrument_ui.value

    @classmethod
    def get_ipts_number(cls) -> str:
        """
        Get current IPTS number from UI or class variable.
        
        This method retrieves the IPTS number either from the UI input
        field or from the class variable if the UI field is empty.
        
        Returns:
            Current IPTS number as string
            
        Note:
            Prioritizes UI input over class variable for current selection.
        """
        ipts_number: str = cls.ipts_number.value
        if ipts_number == '':
            ipts_number = cls.ipts_number
        
        return ipts_number

    @classmethod
    def get_computer_name(cls) -> str:
        """
        Get current computer/hostname.
        
        Returns:
            Computer hostname for system identification
            
        Note:
            Uses platform.node() to get the network name of the machine.
        """
        return platform.node()

    @classmethod
    def get_facility_selected(cls) -> str:
        """
        Get facility corresponding to currently selected instrument.
        
        Returns:
            Name of facility hosting the currently selected instrument
            
        Note:
            Combines get_instrument_selected() with get_facility_from_instrument().
        """
        return cls.get_facility_from_instrument(instrument=cls.get_instrument_selected())

    @classmethod
    def get_start_path(cls, debugger_folder: str = '', system_folder: str = '', instrument: str = '') -> str:
        """
        Determine appropriate start path for directory navigation.
        
        This method calculates the starting directory path based on the current
        mode (debugging vs production), user credentials, and instrument selection.
        It handles different scenarios for local development and production systems.
        
        Args:
            debugger_folder: Folder to use in debugging mode
            system_folder: System folder override path
            instrument: Instrument name for path construction
            
        Returns:
            Appropriate start path for directory navigation
            
        Note:
            In debugging mode, uses predefined paths from config.
            In production mode, constructs paths based on facility/instrument structure.
        """

        logging.info(f"Calculating start path with parameters: debugger_folder={debugger_folder}, system_folder={system_folder}, instrument={instrument}")
        facility: str = cls.get_facility_from_instrument(instrument=instrument)
        logging.info(f"\t{facility = }")

        username: str = getpass.getuser()
        logging.info(f"\tusername: {username}")

        debugging: bool = config.debugging
        debugger_username: str = config.debugger_username
        logging.info(f"\tdebugging: {debugging}")
        logging.info(f"\tdebugger_username: {debugger_username}")

        found_a_folder: bool = False
        if debugger_folder == '':
            for _folder in config.debugger_folder:
                if os.path.exists(_folder):
                    debugger_folder = _folder
                    found_a_folder = True
                    break

        if not found_a_folder:
            logging.info(f"\tinside not found a debugger folder, using home folder")
            debugger_folder = './'

        if debugging and (username == debugger_username):
            logging.info("\t** Using Debugging Mode! **")

            # check that in debugging mode, on analysis machine, default folder exists
            import socket

            if socket.gethostname() == config.analysis_machine:
                if not os.path.exists(debugger_folder):
                    debugging = False

            start_path = debugger_folder
            logging.info(f"\tdebugger_folder: start_path={start_path}")
            
        else:
            if system_folder == '':
                logging.info(f"\t{system_folder} is empty, constructing start path based on facility and instrument")
                start_path = "/{}/{}/".format(facility, instrument)
                logging.info(f"\tconstructed start_path: {start_path}")
            else:
                start_path = system_folder
                logging.info(f"\tsystem_folder provided: start_path={start_path}")
            import warnings
            warnings.filterwarnings('ignore')

        logging.info(f"Final start_path: {start_path}")
        return start_path

    @classmethod
    def select_ipts_help(cls, value: Any) -> None:
        """
        Display help documentation for IPTS selection.
        
        This method opens a web browser to display help documentation
        for selecting IPTS numbers in the system interface.
        
        Args:
            value: Button click event value (unused)
            
        Note:
            Opens the ORNL neutron imaging tutorial page for IPTS selection.
        """
        import webbrowser
        webbrowser.open("https://neutronimaging.pages.ornl.gov/tutorial/notebooks/select_ipts/")

    @classmethod
    def check_instrument_input(cls, value_dict: Dict[str, str]) -> None:
        """
        Handle instrument selection change event.
        
        This method responds to changes in the instrument selection UI by
        updating the available folder list and resetting related UI elements
        to reflect the new instrument context.
        
        Args:
            value_dict: Widget change event dictionary containing 'new' value
            
        Side Effects:
            - Updates working directory UI options
            - Resets IPTS number and result label
            - Updates start path for new instrument
        """
        instrument: str = value_dict['new']

        start_path: str = cls.get_start_path(instrument=instrument)
        cls.start_path = start_path
        list_and_default_folders: Dict[str, Any] = cls.get_list_folders(start_path=start_path)

        user_list_folders: List[str] = list_and_default_folders['user_list_folders']
        default_value: str = list_and_default_folders['default_value']

        cls.working_dir_ui.options = user_list_folders
        cls.working_dir_ui.value = default_value

        cls.ipts_number.value = ''
        cls.result_label.value = ''

    @classmethod
    def check_ipts_input(cls, value: Dict[str, str]) -> None:
        """
        Validate IPTS number input and update UI accordingly.
        
        This method validates user input for IPTS numbers by checking if the
        corresponding IPTS folder exists in the file system. It provides
        immediate feedback and updates the folder selection if valid.
        
        Args:
            value: Widget change event dictionary containing 'new' IPTS number
            
        Side Effects:
            - Updates result label with validation status
            - Sets working directory UI to IPTS folder if valid
            - Provides visual feedback (OK/DOES NOT EXIST)
        """
        ipts: str = value['new']
        full_ipts: str = 'IPTS-{}'.format(ipts)
        if os.path.exists(os.path.join(cls.start_path, full_ipts)):
            # display(HTML("""
            #            <style>
            #            .result_label {
            #               font-style: bold;
            #               color: green;
            #               font-size: 18px;
            #            }
            #            </style>
            #            """))
            cls.result_label.value = "OK"
            #select IPTS folder defined
            cls.working_dir_ui.value = full_ipts

        else:
            # display(HTML("""
            #            <style>
            #            .result_label {
            #               font-style: bold;
            #               color: red;
            #               font-size: 18px;
            #            }
            #            </style>
            #            """))
            cls.result_label.value = "DOES NOT EXIST!"

    @classmethod
    def get_working_dir(cls) -> str:
        """
        Get the current working directory path.
        
        This method returns the currently configured working directory,
        either from the class variable or by combining the start path
        with the selected folder from the UI.
        
        Returns:
            Complete path to the current working directory
            
        Note:
            If working_dir is already set, returns that value.
            Otherwise, constructs path from start_path and UI selection.
        """
        logging.info(f"Getting working directory. Current working_dir: {cls.working_dir}")
        if cls.working_dir:
            logging.info(f"\tabout to return {cls.working_dir = }")
            return cls.working_dir
        else:
            logging.info(f"\tabout to start_path: {os.path.join(cls.start_path, cls.working_dir_ui.value) = }")
            return os.path.join(cls.start_path, cls.working_dir_ui.value)
