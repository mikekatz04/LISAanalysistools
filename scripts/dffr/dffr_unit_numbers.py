"""Measured errors behind tests/test_dffr.py, next to the bounds the tests assert.

What the numbers mean is in docs/response/dffr/tests.md. Run from the root of
lisa-analysis-tools:

    python scripts/dffr/dffr_unit_numbers.py --device cpu
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import lisatools  # noqa: E402
from tests import test_dffr as unit  # noqa: E402


def print_rows(title, rows):
    print(f"\n{title}")
    print(f"  {'quantity':52s} {'measured':>11s} {'bound':>11s} {'ratio':>7s}")
    for name, measured, bound in rows:
        print(f"  {name:52s} {measured:11.3e} {bound:11.3e} {measured / bound:7.3f}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    args = parser.parse_args(argv)
    backend = lisatools.get_backend(args.device)
    print(f"[unit numbers] backend {backend.name}")

    rows = []
    for chirp_rate, detuning, kernel_error, moment_error in unit.sub_interval_errors(backend):
        label = f"alpha {chirp_rate:+.0e}, {detuning / unit.FREQUENCY_BIN:+6.2f} bins"
        rows += [
            (f"kernel, {label}", kernel_error, unit.KERNEL_BOUND),
            (f"moment, {label}", moment_error, unit.MOMENT_BOUND),
        ]
    print_rows("sub-interval integral and moment against quadrature, DFFR's normalisation", rows)

    num_cases = len(unit.WINDOWED_CHIRP_RATES) * len(unit.WINDOWED_BINS)
    rows = []
    for window_taper, (kernel_error, moment_error) in unit.windowed_kernel_errors(backend).items():
        label = f"taper {window_taper:.4f}, largest of {num_cases} cases"
        rows += [
            (f"kernel, {label}", kernel_error, unit.WINDOWED_KERNEL_BOUND),
            (f"moment, {label}", moment_error, unit.WINDOWED_MOMENT_BOUND),
        ]
    print_rows("windowed transform and moment against quadrature, same normalisation", rows)

    rows = [
        (
            f"generation {generation}, largest absolute difference",
            error,
            unit.LINK_COEFFICIENT_BOUND,
        )
        for generation, error in unit.link_coefficient_errors(backend).items()
    ]
    print_rows("closed-form TDI coefficients against the units of TDIConfig", rows)

    if args.device == "gpu":
        rows = [
            (name, difference, unit.GPU_AGAINST_CPU_BOUND)
            for name, difference in unit.gpu_against_cpu_differences(backend).items()
        ]
        print_rows("CUDA against CPU", rows)


if __name__ == "__main__":
    main()
