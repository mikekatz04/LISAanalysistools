"""Kinematics table of the lisatools orbits, for the step 1 regressions of the DFFR integration.

What each mode records, and the results, are in docs/response/dffr/tests.md.

    python kinematics_regression.py esa --device cpu --output LAT.npz
    python kinematics_regression.py l1 --device cpu --mojito-l1 FILE.h5 --output L1.npz
    python kinematics_regression.py compare LAT.npz DFFR.npz --output REPORT.json

``esa`` builds the table on the ESA orbits that lisatools ships, at times counted from the start
of the file (``t0 = 0``, as the base orbit classes count them). ``l1`` builds it on a Mojito L1
file over a year of one-day segments that starts 10 days after the first sample both grids hold,
and compares the receiver velocities with the velocities of the file. ``compare`` reads the
table of ``esa`` and the two tables of ``kinematics_dffr.py`` (dffr's C stencil builder and its
lisaorbits builder) and reports the largest difference per field.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stft_fixtures as fixtures  # noqa: E402

import lisatools  # noqa: E402
from lisatools.domaincomputation import KINEMATICS_TABLE_FIELDS  # noqa: E402

DAY = 86400.0
#: Grid of the ESA exchange, counted from the start of the orbit file: one-day segments from
#: 5 days in, as in the step 0 truth run, over a year.
ESA_FIRST_START = 5 * DAY
#: Production-like grid on the Mojito orbits: data from 10 days after the first common sample.
L1_START_OFFSET = fixtures.DATA_START_OFFSET
NUM_SEGMENTS = 365


def build_table(orbits, device, first_start, segment_time, num_segments) -> dict:
    """Fill a kinematics table from ``orbits``; host arrays of shape (..., num_segments)."""
    backend = lisatools.get_backend(device)
    xp = backend.xp
    arrays = {
        name: xp.zeros(int(np.prod(shape)) * num_segments)
        for name, shape in KINEMATICS_TABLE_FIELDS
    }
    table = backend.KinematicsTableWrap(
        first_start + 0.5 * segment_time, segment_time, num_segments, *arrays.values()
    )
    cpp_orbits = orbits.pycppdetector
    table.build(cpp_orbits)
    fixtures.synchronize(device)
    return {
        name: fixtures.to_host(arrays[name]).reshape(shape + (num_segments,))
        for name, shape in KINEMATICS_TABLE_FIELDS
    }


def grid_metadata(orbits, first_start, segment_time, num_segments) -> dict:
    return dict(
        first_start=first_start,
        segment_time=segment_time,
        num_segments=num_segments,
        position_grid=dict(
            t0=float(orbits.pycppdetector_args[0]),
            dt=float(orbits.pycppdetector_args[1]),
            num=int(orbits.pycppdetector_args[2]),
        ),
        ltt_grid=dict(
            t0=float(orbits.pycppdetector_args[3]),
            dt=float(orbits.pycppdetector_args[4]),
            num=int(orbits.pycppdetector_args[5]),
        ),
    )


def save(path, fields, metadata):
    np.savez(path, metadata=json.dumps(metadata), **fields)
    print(f"[kinematics] wrote {path}")


def run_esa(args):
    orbits, _ = fixtures.load_orbits("esa", args.device)
    fields = build_table(orbits, args.device, args.first_start, DAY, args.num_segments)
    with h5py.File(orbits.filename, "r") as orbit_file:
        file_t0 = float(orbit_file.attrs["t0"])
    metadata = dict(
        run=fixtures.run_metadata(args.device),
        orbits=dict(kind="esa", file=orbits.filename, file_t0=file_t0, frame=orbits.frame),
        grid=grid_metadata(orbits, args.first_start, DAY, args.num_segments),
    )
    save(args.output, fields, metadata)


def run_l1(args):
    orbits, common_start = fixtures.load_orbits("l1", args.device, args.mojito_l1)
    first_start = common_start + L1_START_OFFSET
    fields = build_table(orbits, args.device, first_start, DAY, args.num_segments)
    midpoints = first_start + DAY * (np.arange(args.num_segments) + 0.5)

    # * The positions come from get_pos, the same interpolation the table reads: equal.
    # * The velocities are five-point derivatives of positions splined through the file samples,
    # * against a separate spline of the file velocities.
    checks = {}
    xp = lisatools.get_backend(args.device).xp
    for index in range(3):
        times = xp.asarray(midpoints)
        position = fixtures.to_host(orbits.get_pos(times, index + 1)).T
        velocity = fixtures.to_host(orbits.get_vel(times, index + 1)).T
        checks[f"sc{index + 1}"] = dict(
            position_max_abs_diff=float(
                np.abs(fields["position_receiver"][index] - position).max()
            ),
            velocity_max_rel_diff=float(
                np.abs(fields["velocity_receiver"][index] - velocity).max() / np.abs(velocity).max()
            ),
        )
    print(json.dumps(checks, indent=2))
    metadata = dict(
        run=fixtures.run_metadata(args.device, args.mojito_l1),
        orbits=dict(kind="l1", frame=orbits.frame, common_start=common_start),
        grid=grid_metadata(orbits, first_start, DAY, args.num_segments),
        checks=checks,
    )
    save(args.output, fields, metadata)


def run_compare(args):
    lat = np.load(args.lat_npz)
    dffr = np.load(args.dffr_npz)
    lat_grid = json.loads(str(lat["metadata"]))["grid"]
    dffr_grid = json.loads(str(dffr["metadata"]))["grid"]
    for key in ("first_start", "segment_time", "num_segments"):
        if lat_grid[key] != dffr_grid[key]:
            raise ValueError(f"the two tables are on different grids: {key}")

    report = {}
    for builder in ("c_stencil", "lisaorbits"):
        report[builder] = {}
        print(f"\n[compare] LAT table against dffr's {builder} builder")
        print(f"  {'field':32s} {'max |diff|':>12s} {'max |ref|':>12s} {'relative':>10s}")
        for name, _ in KINEMATICS_TABLE_FIELDS:
            reference = dffr[f"{builder}__{name}"]
            diff = np.abs(lat[name] - reference).max()
            scale = np.abs(reference).max()
            report[builder][name] = dict(
                max_abs_diff=float(diff),
                max_abs_reference=float(scale),
                max_rel_diff=float(diff / scale),
            )
            print(f"  {name:32s} {diff:12.3e} {scale:12.3e} {diff / scale:10.2e}")
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(f"\n[compare] wrote {args.output}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    modes = parser.add_subparsers(dest="mode", required=True)

    esa = modes.add_parser("esa", help="table on the ESA orbits that lisatools ships")
    esa.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    esa.add_argument(
        "--first-start",
        type=float,
        default=ESA_FIRST_START,
        help="first segment start after the start of the orbit file [s]",
    )
    esa.add_argument("--num-segments", type=int, default=NUM_SEGMENTS)
    esa.add_argument("--output", required=True, metavar="OUT_NPZ")

    l1 = modes.add_parser("l1", help="table on a Mojito L1 file, and its velocity check")
    l1.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    l1.add_argument("--mojito-l1", required=True, help="Mojito L1 .h5 file")
    l1.add_argument("--num-segments", type=int, default=NUM_SEGMENTS)
    l1.add_argument("--output", required=True, metavar="OUT_NPZ")

    compare = modes.add_parser("compare", help="LAT table against the two dffr tables")
    compare.add_argument("lat_npz")
    compare.add_argument("dffr_npz")
    compare.add_argument("--output", required=True, metavar="OUT_JSON")

    args = parser.parse_args(argv)
    {"esa": run_esa, "l1": run_l1, "compare": run_compare}[args.mode](args)


if __name__ == "__main__":
    main()
