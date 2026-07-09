import importlib.util
import multiprocessing
import sys
import argparse

import numpy as np
import os
import warnings
from copy import deepcopy

import ast
import ctypes
import sys

# NOTE (spawn safety): multiprocessing 'spawn' children (e.g. the eryn flow
# trainer) re-import this module at module level WITH the parent's sys.argv.
# Everything heavy or CUDA-touching must therefore live behind the
# parent-process guard in _pre_init_cuda or inside the __main__ block below —
# otherwise every spawned child creates a ~400 MB CUDA context on gpus[0].


def _pre_init_cuda() -> None:
    """Set the CUDA device before any cupy/GPU import.

    Parses the settings file path from sys.argv via AST (no code execution)
    to extract the ``gpus`` list, then calls ``cudaSetDevice`` through ctypes
    so that the CUDA runtime initialises on the correct device before cupy
    is imported anywhere in the module-level import chain.

    No-op in multiprocessing children: spawn propagates the parent's argv, so
    without the guard a spawned trainer would set its device to gpus[0] and
    end up with a stray CUDA context on a sampling GPU.
    """
    if multiprocessing.parent_process() is not None:
        return
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


if __name__ == "__main__":
    # Heavy imports stay inside the __main__ block: spawned children re-import
    # this module but must not pull in mpi4py/lisatools/cupy (each would add
    # startup cost and a CUDA context to every trainer process).
    from mpi4py import MPI
    from lisatools.globalfit.run import CurrentInfoGlobalFit, GlobalFit

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
    try:
        gf.run_global_fit()
    except KeyboardInterrupt:
        # Attempt a normal (clean) shutdown, but with a deadline: if any
        # thread/finalizer wedges during teardown (e.g. a GIL-released CUDA
        # call, or an mp.Queue feeder flushing to a dead trainer child), a
        # daemon watchdog hard-exits so Ctrl+C always terminates the process.
        # State is safe either way (backups every backup_iter; the flow
        # checkpoint is written atomically).
        import threading

        print(
            "\nKeyboardInterrupt — shutting down (hard exit in 10 s if teardown hangs).",
            file=sys.stderr,
            flush=True,
        )
        watchdog = threading.Timer(10.0, lambda: os._exit(130))
        watchdog.daemon = True
        watchdog.start()
        sys.exit(130)
    #breakpoint()