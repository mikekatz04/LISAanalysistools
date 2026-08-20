"""Run the LISA Global Fit with LISA Analysis Tools.

Memory instrumentation notes
----------------------------
There are two entirely separate failure modes, and they need separate handling:

1. HOST memory exhaustion. The Slurm cgroup OOM killer sends SIGKILL, which
   Python cannot catch -- no exception, no traceback, no atexit, no flush. The
   only defence is to observe the footprint *before* the limit is hit, which is
   what the background watchdog thread does.

2. DEVICE memory exhaustion. This raises cupy.cuda.memory.OutOfMemoryError and
   is catchable, so the handler at the bottom is reachable. It emits at WARNING
   level -- at DEBUG with no logging configuration, the records are discarded
   and the handler appears to do nothing.

``host_mem_watchdog.py`` must sit next to this file (Python puts the script's
own directory on sys.path).
"""

import argparse
import ast
import ctypes
import importlib.util
import os
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


_pre_init_cuda()  # avoid allocating GPU memory on unrequested devices.

# Everything below may pull in cupy transitively, so it must come after the
# cudaSetDevice call above. memory_monitoring imports cupy lazily inside its
# functions, so it is safe to import here either way.
import logging  # noqa: E402

from mpi4py import MPI  # noqa: E402

from memory_monitoring import (  # noqa: E402
    setup_logging,
    start_watchdog,
    log_mem,
    dump_gpu_arrays,
    mem_summary,
)

from lisatools.globalfit.run import CurrentInfoGlobalFit, GlobalFit  # noqa: E402


def load_settings(file_path: str, function_name: str):
    """Import the settings module by path and call the settings function."""
    if os.path.splitext(file_path)[1] != ".py":
        raise ValueError("Imported settings file must be a python file (.py).")

    module_name = os.path.splitext(os.path.basename(file_path))[0]

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load a module spec from {file_path!r}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    try:
        settings_function = getattr(module, function_name)
    except AttributeError:
        raise AttributeError(
            f"{file_path!r} has no function named {function_name!r}"
        ) from None

    return settings_function()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the LISA Global Fit with LISA Analysis Tools."
    )
    parser.add_argument(
        "-sfp", "--settings_file_path", required=True,
        help="The settings file.",
    )
    parser.add_argument(
        "-sff", "--settings_function", default="get_global_fit_settings",
        help="The function in the settings file that supplies the settings.",
    )
    parser.add_argument(
        "--mem-interval", type=float, default=60.0,
        help="Seconds between host-memory samples. 0 disables the watchdog.",
    )
    parser.add_argument(
        "--mem-trace", action="store_true",
        help="Enable tracemalloc. Only for leak hunting -- it costs CPU and "
             "memory, and cannot see cupy pinned or C-extension allocations.",
    )
    parser.add_argument(
        "--mem-abort-frac", type=float, default=None,
        help="Fraction of the cgroup limit at which to SIGINT ourselves to "
             "obtain a traceback. Leave unset unless you are debugging a host "
             "OOM: an interrupt mid-write can corrupt an HDF5 checkpoint.",
    )
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    setup_logging(level=logging.INFO, rank=comm.Get_rank())
    logger = logging.getLogger("run_global")

    logger.info("startup: %s", mem_summary("startup"))

    if args.mem_interval > 0:
        start_watchdog(
            interval=args.mem_interval,
            warn_frac=0.75,
            dump_frac=0.85,
            abort_frac=args.mem_abort_frac,
            trace=args.mem_trace,
        )

    curr_info = load_settings(args.settings_file_path, args.settings_function)
    log_mem("after settings load")

    gf = GlobalFit(curr_info, comm)
    log_mem("after GlobalFit construction")

    # Imported late so that a missing cupy does not break a CPU-only run.
    import cupy as cp

    try:
        gf.run_global_fit()
    except cp.cuda.memory.OutOfMemoryError:
        # Reachable only for DEVICE OOM. A host OOM arrives as SIGKILL and
        # never gets here -- that is what the watchdog above is for.
        logger.warning("=== device OOM -- dumping live GPU arrays ===")
        logger.warning(mem_summary("device OOM"))
        dump_gpu_arrays(min_size_mb=5.0, names=True)
        pool = cp.get_default_memory_pool()
        logger.warning("pool used:  %.1f MB", pool.used_bytes() / 1024 ** 2)
        logger.warning("pool total: %.1f MB", pool.total_bytes() / 1024 ** 2)
        raise
    except KeyboardInterrupt:
        # Also the landing point for the watchdog's SIGINT, if enabled.
        logger.warning("interrupted: %s", mem_summary("interrupt"))
        raise

    logger.info("run_global_fit completed: %s", mem_summary("done"))
    


if __name__ == "__main__":
    main()