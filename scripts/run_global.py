import importlib.util
import sys
import argparse

import numpy as np
from mpi4py import MPI
import os
import warnings
from copy import deepcopy

import ast
import ctypes
import sys


def _pre_init_cuda() -> None:
    """Set the CUDA device before any cupy/GPU import.

    Parses the settings file path from sys.argv via AST (no code execution)
    to extract the ``gpus`` list, then calls ``cudaSetDevice`` through ctypes
    so that the CUDA runtime initialises on the correct device before cupy
    is imported anywhere in the module-level import chain.
    """
    sfp = next(
        (sys.argv[i + 1] for i, a in enumerate(sys.argv[:-1])
         if a in ("-sfp", "--settings_file_path")),
        None,
    )
    if sfp is None:
        return
    try:
        with open(sfp) as f:
            tree = ast.parse(f.read())
    except (OSError, SyntaxError):
        return
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for stmt in ast.walk(node):
                if isinstance(stmt, ast.Assign):
                    for target in stmt.targets:
                        if isinstance(target, ast.Name) and target.id == "gpus":
                            try:
                                gpus = ast.literal_eval(stmt.value)
                                ctypes.CDLL("libcudart.so").cudaSetDevice(gpus[0])
                                return
                            except Exception:
                                pass


_pre_init_cuda() # avoid allocating GPU memory on unrequested devices.

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
    
    # Setup checker when running out of memory on GPU. This will dump all live GPU arrays and their names to stdout.
    import gc
    import cupy as cp

    def find_names_for_array(arr, max_depth=3):
        """Best-effort: find variable names referencing this array."""
        names = []
        seen = set()

        def search(obj, depth, path):
            if depth > max_depth or id(obj) in seen:
                return
            seen.add(id(obj))
            for ref in gc.get_referrers(obj):
                if isinstance(ref, dict):
                    for k, v in ref.items():
                        if v is obj:
                            # is this dict a frame's locals/globals, or an instance __dict__?
                            for ref2 in gc.get_referrers(ref):
                                if hasattr(ref2, 'f_locals') and ref2.f_locals is ref:
                                    names.append(f"local '{k}' in {ref2.f_code.co_name}() line {ref2.f_lineno}")
                                elif hasattr(ref2, '__dict__') and ref2.__dict__ is ref:
                                    names.append(f"attribute '{k}' of {type(ref2).__name__} instance")
                elif isinstance(ref, (list, tuple)):
                    search(ref, depth + 1, path)

        search(arr, 0, [])
        return names

    def dump_gpu_arrays_with_names(min_size_mb=10):
        arrays = []
        for obj in gc.get_objects():
            if isinstance(obj, cp.ndarray) and obj.nbytes / 1024**2 >= min_size_mb:
                names = find_names_for_array(obj)
                arrays.append((obj.nbytes / 1024**2, obj.shape, obj.dtype, names))
        arrays.sort(reverse=True, key=lambda x: x[0])
        for size_mb, shape, dtype, names in arrays:
            print(f"{size_mb:>10.1f} MB  {str(shape):<25} {dtype}  -> {names or 'no named ref found'}")

    try:
        gf.run_global_fit()
    except cp.cuda.memory.OutOfMemoryError:
        print("=== OOM — dumping live GPU arrays ===")
        dump_gpu_arrays_with_names(min_size_mb=5)
        pool = cp.get_default_memory_pool()
        print(f"Pool used:  {pool.used_bytes()/1024**2:.1f} MB")
        print(f"Pool total: {pool.total_bytes()/1024**2:.1f} MB")
        raise
        
    #breakpoint()