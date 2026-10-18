"""Measured errors behind tests/test_orbit_kinematics.py, next to the bounds the tests assert.

What the numbers mean is in docs/response/dffr/tests.md. Run from the root of
lisa-analysis-tools:

    python scripts/dffr/kinematics_unit_numbers.py --device cpu
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import lisatools  # noqa: E402
from tests import test_orbit_kinematics as unit  # noqa: E402


def table_on(backend, tables):
    wrap, arrays = unit.build_orbits_wrap(backend, tables)
    return unit.build_table(backend, wrap, **unit.STFT_GRID)


def print_rows(title, rows):
    print(f"\n{title}")
    print(f"  {'field':32s} {'measured':>11s} {'bound':>11s} {'ratio':>7s}")
    for name, measured, bound in rows:
        print(f"  {name:32s} {measured:11.3e} {bound:11.3e} {measured / bound:7.3f}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    args = parser.parse_args(argv)
    backend = lisatools.get_backend(args.device)
    midpoints = unit._midpoints(**unit.STFT_GRID)
    reference_time = unit.POSITION_GRID["t0"] + 6.0e5
    print(f"[unit numbers] backend {backend.name}")

    quadratic = unit.QuadraticModel(reference_time)
    tables = unit.sample_tables(quadratic)
    table = table_on(backend, tables)
    expected = unit.kinematics_table(quadratic, midpoints)
    bounds = unit.rounding_bounds(tables, link_distance=2.0e9)
    rows = [
        (name, np.abs(table[name] - expected[name]).max(), bounds[name])
        for name in (
            "velocity_receiver",
            "acceleration_receiver",
            "light_travel_time_rate",
            "light_travel_time_acceleration",
        )
    ]
    print_rows("quadratic orbits: absolute error against the exact derivatives", rows)

    circular = unit.CircularModel(reference_time)
    tables = unit.sample_tables(circular)
    table = table_on(backend, tables)
    expected = unit.kinematics_table(circular, midpoints)
    rows = [
        (name, np.abs(table[name] - expected[name]).max() / np.abs(expected[name]).max(), budget)
        for name, budget in unit.CIRCLE_RELATIVE_BUDGET.items()
    ]
    print_rows("circular orbits: relative error against the analytic derivatives", rows)

    mirror = unit.kinematics_table(unit.NumpyOrbits(tables), midpoints)
    bounds = unit.rounding_bounds(tables, link_distance=2.0e9)
    rows = [
        (name, np.abs(table[name] - mirror[name]).max(), bounds[name])
        for name, _ in unit.KINEMATICS_TABLE_FIELDS
    ]
    print_rows("circular orbits: absolute difference from the NumPy mirror", rows)
    identical = [
        name
        for name, _ in unit.KINEMATICS_TABLE_FIELDS
        if np.array_equal(table[name], mirror[name])
    ]
    print(
        f"  bit-identical to the mirror: {len(identical)} of "
        f"{len(unit.KINEMATICS_TABLE_FIELDS)} fields {identical}"
    )

    if args.device == "gpu":
        cpu_table = table_on(lisatools.get_backend("cpu"), tables)
        rows = [
            (name, np.abs(table[name] - cpu_table[name]).max(), bounds[name])
            for name, _ in unit.KINEMATICS_TABLE_FIELDS
        ]
        print_rows("circular orbits: CUDA against CPU, absolute difference", rows)

    from lisatools.detector import ESAOrbits

    orbits = ESAOrbits(force_backend=args.device)
    first_start, segment_time, num_segments = 10.0 * 86400.0, 86400.0, 30
    esa_table = unit.build_table(
        backend, orbits.pycppdetector, first_start, segment_time, num_segments
    )
    esa_midpoints = backend.xp.asarray(unit._midpoints(first_start, segment_time, num_segments))
    rows = []
    for index in range(3):
        file_velocity = unit._to_host(orbits.get_vel(esa_midpoints, index + 1)).T
        error = np.abs(esa_table["velocity_receiver"][index] - file_velocity).max()
        relative_error = error / np.abs(file_velocity).max()
        rows.append(
            (f"velocity_receiver sc{index + 1}", relative_error, unit.ESA_VELOCITY_RELATIVE_BOUND)
        )
    print_rows("ESA orbits: receiver velocity against the file velocity, relative", rows)


if __name__ == "__main__":
    main()
