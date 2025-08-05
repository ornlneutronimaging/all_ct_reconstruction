"""
Custom Exception Classes for CT Reconstruction Pipeline

This module defines custom exception classes used throughout the CT reconstruction
pipeline to provide specific error handling and meaningful error messages for
various failure scenarios.

Classes:
    MetadataError: Exception for metadata retrieval failures

Author: CT Reconstruction Development Team
"""


class MetadataError(Exception):
    """
    Exception raised for errors in metadata retrieval processes.
    
    This exception is raised when the system fails to extract or process
    metadata from files, particularly during image or data file analysis
    in the CT reconstruction pipeline.
    
    Attributes:
        message (str): Explanation of the error
    """
    
    def __init__(self, message: str = "Error retrieving metadata from the file.") -> None:
        """
        Initialize the MetadataError exception.
        
        Args:
            message: Detailed explanation of the metadata error
        """
        self.message = message
        super().__init__(self.message)

