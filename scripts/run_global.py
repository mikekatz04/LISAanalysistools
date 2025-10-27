import importlib.util
import sys
import argparse

import numpy as np
from mpi4py import MPI
import os
import warnings
from copy import deepcopy

from lisatools.globalfit.run import CurrentInfoGlobalFit, GlobalFit


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="Run the LISA Global Fit with LISA Analysis Tools.")

    parser.add_argument("-sfp", "--settings_file_path", required=True, help="The settings file.") # Positional
    parser.add_argument("-sff", "--settings_function", default="get_global_fit_settings", help="The function in the settings file that will import the settings information.") # Optional flag
    
    args = parser.parse_args()

    # Define the module name and the full path to the Python file
    file_path = args.settings_file_path
    if file_path[-3:] != ".py":
        raise ValueError("Imported settings file must be a python file (.py).")

    module_name = file_path.split("/")[-1].split(".py")[0]
    '/path/to/my_module.py' # Replace with the actual path to your .py file

    # Create a module specification from the file location
    spec = importlib.util.spec_from_file_location(module_name, file_path)

    # Create a new module object from the specification
    my_module = importlib.util.module_from_spec(spec)

    # Add the module to sys.modules (optional, but good practice for caching)
    sys.modules[module_name] = my_module

    # Execute the module's code
    spec.loader.exec_module(my_module)

    # Now you can access functions, classes, or variables from the imported module
    # For example, if my_module.py contains a function called 'my_function':
    settings_function = getattr(my_module, args.settings_function)
    
    curr_info = settings_function()

    gf = GlobalFit(curr_info, MPI.COMM_WORLD)
    gf.run_global_fit()
    breakpoint()