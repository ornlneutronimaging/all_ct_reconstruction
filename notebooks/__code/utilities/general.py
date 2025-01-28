def retrieve_parameters(instance):
    """Retrieve all parameters from an instance of a class."""
    
    list_all_variables = dict(instance)
    list_variables = [var for var in list_all_variables if not var.startswith('__')]
    my_dict = {_variable: getattr(instance, _variable) for _variable in list_variables}
    return my_dict


def retrieve_list_class_attributes_name(my_class):
    """Retrieve the name of all the class attribute of a class"""
    
    list_all_variables = dir(my_class)
    list_variables = [var for var in list_all_variables if not var.startswith('__')]

    return list_variables
