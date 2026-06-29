"""Re-run LISA global-fit POSTPROCESSING on an already-finished run.

This is a standalone, single-process driver that re-generates the submission
files (L3C-compliant HDF5 + JSON manifests, residuals, posteriors) from an
existing MCMC backend ``.h5`` **without re-running the MCMC**.

Why this script exists / key design point
------------------------------------------
``GlobalFit.run_global_fit`` is written for a multi-rank MPI launch: the
"main" rank drives the sampler while a "results" rank and any auxiliary ranks
sit in ``comm.recv`` loops (see ``run.py`` and
``hdfbackend.save_to_backend_asynchronously_and_plot``). Calling
``run_global_fit`` with a single process would still execute the main branch,
but it also re-runs the full MCMC and contains rank choreography that is only
correct when ``comm.Get_size() >= 1`` with the expected helper ranks present.

Instead this script calls the *building blocks* directly:

    curr  = get_global_fit_settings()          # re-creates input data + sensitivity
    gf    = GlobalFit(curr, MPI.COMM_WORLD)     # size-1 communicator
    state = gf.load_info(priors)                # last MCMC sample from the .h5
    acs   = gf.setup_acs(state)                 # AnalysisContainerArray
    SubmissionWriter(backend=backend, curr=curr, ess=ess).write_submission(acs)

None of ``load_info`` / ``setup_acs`` / ``acs.likelihood`` / ``SubmissionWriter``
issue any ``comm.send`` / ``comm.recv`` — those calls live *only* inside
``run_global_fit``'s rank branches and the async-save helper, neither of which
is invoked here. The backend is built without a communicator (``comm=None``), so
nothing in the write path touches MPI. Hence a single process cannot deadlock.

Safety: the main MCMC ``.h5`` is treated **read-only** on this path. The only
mutating calls on a ``GFHDFBackend`` (``reset`` / ``grow`` / ``save_step*``) are
reachable solely from ``run_global_fit``, which is never called here.
Constructing ``CurrentInfoGlobalFit`` / ``GFHDFBackend`` over the file is
provably non-destructive: ``run.py`` itself constructs them and then reads the
last sample from the same file, so construction does not truncate.

Side effects (non-destructive, into the *existing run's* directories):
  * ``get_global_fit_settings()`` re-runs data preprocessing and re-writes the
    input-data plots into ``general_info.artifacts_file_dir``.
  * ``GlobalFit.__init__`` writes ``global_fit.log`` and dumps the settings
    into the artifacts dir.
  * In the full (non --dry-run) path, ``write_submission`` writes into
    ``general_info.submission_parent_folder``.

Usage
-----
    # Smoke test: build everything up to setup_acs, print shapes, then stop
    # *before* SubmissionWriter (no full-chain read, no clustering):
    uv run python LISAanalysistools/scripts/postprocess_run.py \
        -sfp LISAanalysistools/mojito_input/run_1_global_fit_settings.py --dry-run

    # Full submission write (WARNING: GB clustering can take hours):
    uv run python LISAanalysistools/scripts/postprocess_run.py \
        -sfp LISAanalysistools/mojito_input/run_1_global_fit_settings.py
"""

import argparse
import ast
import ctypes
import importlib.util
import os
import sys


def _pre_init_cuda() -> None:
    """Set the CUDA device before any cupy/GPU import.

    Parses the settings file path from ``sys.argv`` via AST (no code execution)
    to extract the ``gpus`` list, then calls ``cudaSetDevice`` through ctypes so
    the CUDA runtime initialises on the correct device before cupy is imported
    anywhere in the module-level import chain. Mirrors ``run_global.py`` so this
    script honours the same GPU assignment and does not allocate memory on
    unrequested devices.
    """
    sfp = next(
        (
            sys.argv[i + 1]
            for i, a in enumerate(sys.argv[:-1])
            if a in ("-sfp", "--settings_file_path")
        ),
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


_pre_init_cuda()  # must run before importing run.py (which imports cupy)

import numpy as np
from mpi4py import MPI

from eryn.state import BranchSupplemental

from lisatools.globalfit.hdfbackend import GFHDFBackend
from lisatools.globalfit.run import GlobalFit
from lisatools.globalfit.stock.erebor import Setup


def load_settings_function(file_path: str, function_name: str):
    """Dynamically import the settings module and return its settings function.

    Mirrors the dynamic-import logic in ``scripts/run_global.py``.
    """
    if file_path[-3:] != ".py":
        raise ValueError("Imported settings file must be a python file (.py).")

    module_name = file_path.split("/")[-1].split(".py")[0]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    my_module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = my_module
    spec.loader.exec_module(my_module)

    return getattr(my_module, function_name)


def build_priors_and_periodic(gf: GlobalFit):
    """Build the combined ``priors`` / ``periodic`` dicts across all branches.

    Replicates the assembly loop in ``GlobalFit.run_global_fit`` (run.py, the
    block immediately before ``state = self.load_info(priors)``).
    """
    priors = {}
    periodic = {}
    for name in gf.engine_info.branch_names:
        if name not in gf.curr.source_info:
            continue

        source = gf.curr.source_info[name]

        if isinstance(source, dict):
            for key, value in source["priors"].items():
                priors[key] = value
            if "periodic" in source and source["periodic"] is not None:
                for key, value in source["periodic"].items():
                    periodic[key] = value

        if isinstance(source, Setup):
            for key, value in source.priors.items():
                priors[key] = value
            if hasattr(source, "periodic") and source.periodic is not None:
                for key, value in source.periodic.items():
                    periodic[key] = value

    return priors, periodic


def _print_shapes(gf: GlobalFit, state, acs) -> None:
    """Print a lightweight, no-GPU summary of the loaded state and containers."""
    print("\n================ postprocess_run: state / acs summary ================")
    print(f"  branch_names : {gf.engine_info.branch_names}")
    print(f"  ntemps       : {gf.ntemps}")
    print(f"  nwalkers     : {gf.nwalkers}")

    coords = getattr(state, "branches_coords", {}) or {}
    inds = getattr(state, "branches_inds", {}) or {}
    for name in coords:
        c_shape = getattr(coords[name], "shape", None)
        i_shape = getattr(inds.get(name, None), "shape", None)
        print(f"  branch '{name}': coords={c_shape} inds={i_shape}")

    ll = getattr(state, "log_like", None)
    if ll is not None:
        print(f"  state.log_like shape : {getattr(ll, 'shape', None)}")
        try:
            print(f"  state.log_like[0]    : {np.asarray(ll)[0]}")
        except Exception:
            pass

    try:
        print(f"  acs.gpus            : {getattr(acs, 'gpus', None)}")
        f_arr = getattr(acs, "f_arr", None)
        if f_arr is not None:
            print(f"  acs.f_arr shape     : {getattr(f_arr, 'shape', None)}")
    except Exception as exc:  # pragma: no cover - purely informational
        print(f"  (could not introspect acs: {exc})")
    print("======================================================================\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Re-run the LISA Global Fit postprocessing / submission writing on an "
            "already-finished run (uses the existing MCMC .h5; does NOT re-run MCMC)."
        )
    )
    parser.add_argument(
        "-sfp",
        "--settings_file_path",
        required=True,
        help="The settings file (same as run_global.py).",
    )
    parser.add_argument(
        "-sff",
        "--settings_function",
        default="get_global_fit_settings",
        help="Function in the settings file that returns the settings/current-info object.",
    )
    parser.add_argument(
        "--ess",
        type=int,
        default=20000,
        help="Target effective sample size passed to SubmissionWriter.",
    )
    parser.add_argument(
        "--dry-run",
        "--skip-clustering",
        dest="dry_run",
        action="store_true",
        help=(
            "Build everything up to and including setup_acs (+ one likelihood eval), "
            "print shapes, and exit BEFORE constructing SubmissionWriter / "
            "write_submission. Useful as a smoke test and while the GB-clustering "
            "save path is being fixed."
        ),
    )
    parser.add_argument(
        "--force-recluster",
        dest="force_recluster",
        action="store_true",
        help=(
            "Ignore any cached GB clustering and re-run the (slow) 'Comparing samples' "
            "step from scratch. By default a cache next to the run .h5 is reused."
        ),
    )
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    if comm.Get_size() > 1:
        print(
            "WARNING: postprocess_run.py is a single-process script; launch it "
            f"without mpirun. Detected {comm.Get_size()} ranks - only rank 0 will "
            "do work, the others exit immediately."
        )
        if comm.Get_rank() != 0:
            return

    # --- Load settings (re-creates input_data_residual_array + sensitivity_backend) ---
    settings_function = load_settings_function(
        args.settings_file_path, args.settings_function
    )
    curr = settings_function()

    # postprocess_run.py expects an ALREADY-FINISHED run. Unlike run.py, a missing
    # backend here is user error (wrong -sfp), not a legitimate cold start, so we
    # halt rather than let load_info silently fall back to prior sampling.
    main_file_path = curr.general_info.main_file_path
    if not os.path.exists(main_file_path):
        raise FileNotFoundError(
            f"MCMC backend not found: {main_file_path}\n"
            "postprocess_run.py re-processes a finished run; it does not cold-start "
            "from priors. Check the -sfp settings file points at the intended run."
        )

    # --- Instantiate GlobalFit on a size-1 communicator (no MCMC is run) ---
    gf = GlobalFit(curr, comm)

    # --- Priors / periodic, exactly as run_global_fit assembles them ---
    priors, periodic = build_priors_and_periodic(gf)

    # --- Load the last MCMC sample from the backend .h5 (read-only) ---
    state = gf.load_info(priors)

    # --- Attach the walker-index supplemental, as run.py does after load_info ---
    supps_base_shape = (gf.ntemps, gf.nwalkers)
    walker_vals = np.tile(np.arange(gf.nwalkers), (gf.ntemps, 1))
    supps = BranchSupplemental(
        {"walker_inds": walker_vals}, base_shape=supps_base_shape, copy=True
    )
    state.supplemental = supps

    # --- Build the analysis containers + evaluate the likelihood once ---
    acs = gf.setup_acs(state)
    state.log_like[:] = acs.likelihood(complex=False)

    _print_shapes(gf, state, acs)

    if args.dry_run:
        print(
            "[dry-run] Built settings, state (load_info), and acs (setup_acs) "
            "successfully. Stopping before SubmissionWriter / write_submission."
        )
        return

    # --- Full submission-writing path ---
    if curr.general_info.submission_parent_folder is None:
        print(
            "general_info.submission_parent_folder is None - there is no submission "
            "target configured in this settings file. Nothing to write; exiting. "
            "(Re-run with --dry-run to validate the load path.)"
        )
        return

    # Import here so the --dry-run smoke test does not depend on postprocessing.py
    # (which may be under concurrent edit) being importable beyond what run.py needs.
    from lisatools.globalfit.postprocessing import SubmissionWriter

    # Build the backend the same way run.py does (the read-only form: main file
    # path + per-branch sub-backends/states, no comm/compression). Pass the
    # GFHDFBackend INSTANCE (not the path string) so SubmissionWriter keeps the
    # sub_backend/sub_state_bases that GB clustering relies on.
    backend = GFHDFBackend(
        curr.general_info.main_file_path,
        sub_backend=gf.engine_info.branch_backends,
        sub_state_bases=gf.engine_info.branch_states,
    )

    print(
        f"Writing submission to {curr.general_info.submission_parent_folder} "
        f"(ess={args.ess}). NOTE: GB clustering may take a long time."
    )
    submission_writer = SubmissionWriter(backend=backend, curr=curr, ess=args.ess)
    submission_writer.force_recluster = args.force_recluster
    submission_writer.write_submission(acs)
    print("write_submission complete.")


if __name__ == "__main__":
    main()
