"""
General Utility Functions for CT Reconstruction Pipeline

This module provides general-purpose utility functions for introspection and 
parameter handling within the CT reconstruction pipeline. These functions enable
dynamic analysis of class instances and retrieval of class attributes.

Functions:
    retrieve_parameters: Extract all non-private parameters from class instances
    retrieve_list_class_attributes_name: Get names of all non-private class attributes

Author: CT Reconstruction Development Team
"""

from typing import Dict, List, Any


def retrieve_parameters(instance: Any) -> Dict[str, Any]:
    """
    Retrieve all non-private parameters from a class instance.
    
    Extracts all attributes from a class instance that do not start with
    double underscores, providing a dictionary of parameter names and values.
    
    Args:
        instance: Class instance to extract parameters from
        
    Returns:
        Dictionary mapping parameter names to their values
        
    Example:
        >>> class MyClass:
        ...     def __init__(self):
        ...         self.param1 = "value1"
        ...         self.param2 = 42
        >>> obj = MyClass()
        >>> retrieve_parameters(obj)
        {'param1': 'value1', 'param2': 42}
    """
    
    list_all_variables = dict(instance)
    list_variables = [var for var in list_all_variables if not var.startswith('__')]
    my_dict = {_variable: getattr(instance, _variable) for _variable in list_variables}
    return my_dict


def retrieve_list_class_attributes_name(my_class: type) -> List[str]:
    """
    Retrieve the names of all non-private class attributes.
    
    Analyzes a class to extract the names of all attributes that do not
    start with double underscores, providing insight into the class structure.
    
    Args:
        my_class: Class type to analyze for attribute names
        
    Returns:
        List of non-private attribute names
        
    Example:
        >>> class MyClass:
        ...     class_var = "shared"
        ...     def method1(self): pass
        ...     def method2(self): pass
        >>> retrieve_list_class_attributes_name(MyClass)
        ['class_var', 'method1', 'method2']
    """
    
    list_all_variables = dir(my_class)
    list_variables = [var for var in list_all_variables if not var.startswith('__')]

    return list_variables
