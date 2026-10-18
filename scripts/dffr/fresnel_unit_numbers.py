"""Measured errors behind tests/test_fresnel.py, next to the bounds the tests assert.

What the numbers mean is in docs/response/dffr/tests.md. Run from the root of
lisa-analysis-tools:

    python scripts/dffr/fresnel_unit_numbers.py --device cpu
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import lisatools  # noqa: E402
from tests import test_fresnel as unit  # noqa: E402


def print_rows(title, rows):
    print(f"\n{title}")
    print(f"  {'quantity':44s} {'measured':>11s} {'bound':>11s} {'ratio':>7s}")
    for name, measured, bound in rows:
        print(f"  {name:44s} {measured:11.3e} {bound:11.3e} {measured / bound:7.3f}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    args = parser.parse_args(argv)
    backend = lisatools.get_backend(args.device)
    print(f"[unit numbers] backend {backend.name}")

    rows = []
    for branch, (cosine_error, sine_error) in unit.integral_errors(backend).items():
        cosine_bound, sine_bound = unit.INTEGRAL_BOUNDS[branch]
        rows += [
            (f"C, {branch}", cosine_error, cosine_bound),
            (f"S, {branch}", sine_error, sine_bound),
        ]
    print_rows("C and S against scipy.special.fresnel: absolute error", rows)

    errors = unit.auxiliary_errors(backend)
    rows = [
        (f"{name}, {branch} ({kind})", errors[(branch, name)], bound)
        for (branch, name), (bound, kind) in unit.AUXILIARY_BOUNDS.items()
    ]
    print_rows("f, g and their derivatives against quadrature", rows)

    rows = []
    for chirp_rate, detuning, kernel_error, moment_error in unit.phase_kernel_errors(backend):
        label = f"alpha {chirp_rate:+.0e}, {detuning / unit.FREQUENCY_BIN:+6.2f} bins"
        rows += [
            (f"kernel, {label}", kernel_error, unit.KERNEL_BOUND),
            (f"moment, {label}", moment_error, unit.MOMENT_BOUND),
        ]
    print_rows("phase kernel and moment against quadrature, DFFR's normalisation", rows)

    if args.device == "gpu":
        differences = unit.gpu_against_cpu_differences(backend)
        rows = [
            (f"{name} ({kind})", differences[name], bound)
            for name, (bound, kind) in unit.GPU_FUNCTION_BOUNDS.items()
        ]
        rows += [
            (name, differences[name], unit.GPU_PHASE_KERNEL_BOUND)
            for name in ("phase_kernel", "phase_moment")
        ]
        print_rows("CUDA against CPU", rows)


if __name__ == "__main__":
    main()
