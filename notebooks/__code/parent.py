"""
Parent base class for CT reconstruction workflow components.

This module provides a base class that establishes the parent-child relationship
pattern used throughout the CT reconstruction pipeline, allowing components to
access shared data and configuration from their parent objects.
"""

from typing import Optional, Any


class Parent:
    """
    Base class for components in the CT reconstruction pipeline.
    
    This class establishes a parent-child relationship pattern that allows
    components to access shared data, configuration, and state from their
    parent objects. The MODE attribute is inherited from the parent to
    maintain consistency across the workflow.
    
    Attributes:
        parent: Reference to the parent object that created this instance
        MODE: Operating mode inherited from parent (e.g., ToF, White Beam)
    """

    def __init__(self, parent: Optional[Any] = None) -> None:
        """
        Initialize the Parent class.
        
        Args:
            parent: The parent object that created this instance. Should have
                   a MODE attribute that will be inherited by this instance.
                   
        Raises:
            AttributeError: If parent is provided but doesn't have a MODE attribute
        """
        self.parent: Optional[Any] = parent
        if parent is not None:
            self.MODE = parent.MODE
        else:
            self.MODE = None
        