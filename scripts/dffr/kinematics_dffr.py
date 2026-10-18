"""Kinematics tables of standalone dffr, for the step 1 exchange of the DFFR integration.

Runs in the environment of the dffr repository, not in the lisatools one:

    DFFR_REPOSITORY/.venv/bin/python kinematics_dffr.py --orbit-file FILE.h5 --output DFFR.npz

It builds two tables on the same segment midpoints as ``kinematics_regression.py esa``: dffr's
C builder with the stencil of lisatools (five points, 2000 s, on a 50 s table), and its
lisaorbits builder with analytic derivatives. Times are counted from the start of the orbit
file on the lisatools side; here the file's ``t0`` is added, which is exact for whole seconds.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from importlib import metadata as importlib_metadata
from pathlib import Path

import dffr
import numpy as np
from dffr.detector import TABLE_FIELDS

DAY = 86400.0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--orbit-file", required=True, help="orbit file read by lisatools")
    parser.add_argument(
        "--first-start",
        type=float,
        default=5 * DAY,
        help="first segment start after the start of the orbit file [s]",
    )
    parser.add_argument("--num-segments", type=int, default=365)
    parser.add_argument(
        "--table-step",
        type=float,
        default=50.0,
        help="step of the lisatools table the C builder reads [s]",
    )
    parser.add_argument("--derivative-step", type=float, default=2000.0)
    parser.add_argument("--stencil-points", type=int, default=5)
    parser.add_argument("--output", required=True, metavar="OUT_NPZ")
    args = parser.parse_args(argv)

    wrapper = dffr.OrbitsWrapper(path=args.orbit_file)
    # ! Whole-second offsets keep t0 + offset - t0 exact, so both sides see the same times.
    if args.first_start != round(args.first_start):
        raise ValueError("--first-start must be a whole number of seconds")
    grid = dffr.STFTGrid(
        t0=wrapper.time_origin + args.first_start,
        segment_duration=DAY,
        num_time_segments=args.num_segments,
        first_bin=1,
        num_bins=1,
    )
    builders = {
        "c_stencil": dffr.OrbitKinematics(
            wrapper,
            grid,
            use_c_implementation=True,
            derivative_step=args.derivative_step,
            stencil_points=args.stencil_points,
            table_step=args.table_step,
            force_backend="cpu",
        ),
        "lisaorbits": dffr.OrbitKinematics(
            wrapper, grid, use_c_implementation=False, force_backend="cpu"
        ),
    }
    fields = {
        f"{builder}__{name}": table.get(name)
        for builder, table in builders.items()
        for name, _ in TABLE_FIELDS
    }

    repository = Path(dffr.__file__).resolve().parents[2]
    commit = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    versions = {}
    for package in ("lisaanalysistools", "lisaorbits", "numpy", "scipy"):
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            versions[package] = None
    metadata = dict(
        dffr_commit=commit,
        versions=versions,
        orbit_file=args.orbit_file,
        time_origin=wrapper.time_origin,
        c_builder=dict(
            table_step=args.table_step,
            derivative_step=args.derivative_step,
            stencil_points=args.stencil_points,
        ),
        grid=dict(first_start=args.first_start, segment_time=DAY, num_segments=args.num_segments),
    )
    np.savez(args.output, metadata=json.dumps(metadata), **fields)
    print(f"[kinematics] wrote {args.output}")


if __name__ == "__main__":
    main()
