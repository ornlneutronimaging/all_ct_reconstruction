"""
File and Folder Browser Utility Module

This module provides a comprehensive file and folder browser interface for 
CT reconstruction workflows. It utilizes custom file selector widgets to 
enable interactive file selection, folder browsing, and directory management 
within Jupyter notebook environments.

The module supports:
- Single and multiple file selection
- Directory browsing and selection  
- File filtering by type and extension
- Search-enabled file selection
- Output folder creation and management
- IPTS (Integrated Program Tracking System) folder navigation

Classes:
    FileFolderBrowser: Main browser interface with comprehensive file/folder selection capabilities

Dependencies:
    - __code.ipywe.myfileselector: Custom file selector widgets
    - __code.ipywe.fileselector: Enhanced file selector with search capabilities

Author: CT Reconstruction Development Team
"""

from typing import Optional, Dict, Any, Callable
from __code.ipywe import myfileselector as myfileselector
from __code.ipywe import fileselector as fileselector


class FileFolderBrowser:
    """
    Interactive file and folder browser for CT reconstruction workflows.
    
    Provides a comprehensive interface for file selection, folder browsing,
    and directory management within Jupyter notebook environments. Supports
    multiple selection modes, file filtering, and custom callback functions
    for workflow integration.
    
    Attributes:
        working_dir (str): Current working directory for file operations
        next_function (Optional[Callable]): Callback function executed after selection
        ipts_folder (Optional[str]): IPTS folder path for specialized navigation
        list_images_ui: UI widget for image file selection
        list_files_ui: UI widget for general file selection  
        list_input_folders_ui: UI widget for input folder selection
        list_output_folders_ui: UI widget for output folder selection
    """

    def __init__(self, 
                 working_dir: str = '',
                 next_function: Optional[Callable] = None,
                 ipts_folder: Optional[str] = None) -> None:
        """
        Initialize the file/folder browser.
        
        Args:
            working_dir: Starting directory for file operations
            next_function: Callback function to execute after file selection
            ipts_folder: IPTS folder path for specialized navigation
        """
        self.working_dir = working_dir
        self.next_function = next_function
        self.ipts_folder = ipts_folder

    def select_images(self, 
                      instruction: str = 'Select Images ...',
                      multiple_flag: bool = True,
                      filters: Dict[str, str] = None,
                      default_filter: str = "All") -> None:
        """
        Launch interactive image file selector.
        
        Creates and displays a file selector widget specifically configured
        for image file selection with customizable filters and instructions.
        
        Args:
            instruction: Display instruction for the user
            multiple_flag: Whether to allow multiple file selection
            filters: Dictionary of file type filters {"name": "pattern"}
            default_filter: Default filter to apply
        """
        if filters is None:
            filters = {"All": "*.*"}
            
        self.list_images_ui = myfileselector.MyFileSelectorPanel(
            instruction=instruction,
            start_dir=self.working_dir,
            multiple=multiple_flag,
            filters=filters,
            default_filter=default_filter,
            next=self.next_function
        )
        self.list_images_ui.show()

    def select_file(self, 
                    instruction: str = "Select file ...",
                    filters: Dict[str, str] = None,
                    default_filter: str = "All") -> None:
        """
        Launch interactive single file selector.
        
        Creates and displays a file selector widget configured for single
        file selection with customizable filters and instructions.
        
        Args:
            instruction: Display instruction for the user
            filters: Dictionary of file type filters {"name": "pattern"}
            default_filter: Default filter to apply
        """
        if filters is None:
            filters = {"All": "*.*"}
            
        self.list_files_ui = myfileselector.MyFileSelectorPanel(
            instruction=instruction,
            start_dir=self.working_dir,
            multiple=False,
            filters=filters,
            default_filter=default_filter,
            next=self.next_function
        )
        self.list_files_ui.show()

    def select_images_with_search(self, 
                                  instruction: str = 'Select Images ...',
                                  multiple_flag: bool = True,
                                  filters: Dict[str, str] = None,
                                  default_filter: str = "All") -> None:
        """
        Launch enhanced image selector with search capabilities.
        
        Creates and displays an enhanced file selector widget with built-in
        search functionality for efficient image file location and selection.
        
        Args:
            instruction: Display instruction for the user
            multiple_flag: Whether to allow multiple file selection
            filters: Dictionary of file type filters {"name": "pattern"}
            default_filter: Default filter to apply
        """
        if filters is None:
            filters = {"All": "*.*"}
            
        self.list_images_ui = fileselector.FileSelectorPanel(
            instruction=instruction,
            start_dir=self.working_dir,
            multiple=multiple_flag,
            filters=filters,
            default_filter=default_filter,
            next=self.next_function
        )
        self.list_images_ui.show()

    def select_input_folder(self, 
                           instruction: str = 'Select Input Folder ...', 
                           multiple_flag: bool = False) -> None:
        """
        Launch input folder selector.
        
        Creates and displays a directory selector widget specifically
        configured for input folder selection.
        
        Args:
            instruction: Display instruction for the user
            multiple_flag: Whether to allow multiple folder selection
        """
        self.list_input_folders_ui = myfileselector.MyFileSelectorPanel(
            instruction=instruction,
            start_dir=self.working_dir,
            type='directory',
            multiple=multiple_flag,
            next=self.next_function
        )
        self.list_input_folders_ui.show()

    def select_output_folder(self, 
                            instruction: str = 'Select Output Folder ...', 
                            multiple_flag: bool = False) -> None:
        """
        Launch output folder selector.
        
        Creates and displays a directory selector widget specifically
        configured for output folder selection.
        
        Args:
            instruction: Display instruction for the user
            multiple_flag: Whether to allow multiple folder selection
        """
        self.list_output_folders_ui = myfileselector.MyFileSelectorPanel(
            instruction=instruction,
            start_dir=self.working_dir,
            type='directory',
            multiple=multiple_flag,
            next=self.next_function
        )
        self.list_output_folders_ui.show()

    def select_output_folder_with_new(self, 
                                     instruction: str = 'Select Output Folder ...') -> None:
        """
        Launch enhanced output folder selector with directory creation.
        
        Creates and displays an advanced directory selector widget that
        includes capabilities for creating new directories and navigating
        IPTS folder structures.
        
        Args:
            instruction: Display instruction for the user
        """
        self.list_output_folders_ui = myfileselector.FileSelectorPanelWithJumpFolders(
            instruction=instruction,
            start_dir=self.working_dir,
            type='directory',
            ipts_folder=self.ipts_folder,
            next=self.next_function,
            newdir_toolbar_button=True
        )
