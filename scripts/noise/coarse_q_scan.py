#!/usr/bin/env python
"""Scan candidate time-coarsening factors for the real-WDM noise likelihood.

The script builds the same fine CPU data/model configuration as
``run_noise_only.py``, freezes channelwise Welch--Satterthwaite weights at the
injected/reference noise parameters, and compares each candidate ``Q`` with the
ordinary fine likelihood.  It also perturbs the physical model and reports the
error in ``Delta logL``; that difference, rather than the additive likelihood
offset, is the quantity that can move a posterior.

Example::

    python scripts/noise/coarse_q_scan.py --mode foreground --full \
        --q-list 1 4 16 64 169 338
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import run_noise_only


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--q-list", type=int, nargs="+", default=[1, 4, 16, 64, 169, 338]
    )
    parser.add_argument(
        "--mode", choices=("instrument", "foreground"), default="foreground"
    )
    parser.add_argument("--noise-file", default=run_noise_only.NOISE_FILE)
    parser.add_argument("--galfor-file", default=run_noise_only.GALFOR_FILE)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--two-years", action="store_true")
    parser.add_argument(
        "--modulation",
        nargs="?",
        const=run_noise_only.MODULATION_FILE,
        default=None,
    )
    parser.add_argument("--unequal-arm", action="store_true")
    parser.add_argument(
        "--no-coarse-ws",
        action="store_true",
        help="scan plain Bartlett weights instead of the default channelwise WS weights",
    )
    parser.add_argument(
        "--comparison-psd-scale",
        type=float,
        default=1.05,
        help="multiply both physical instrument amplitude parameters at the comparison point",
    )
    parser.add_argument(
        "--comparison-galfor-amp-scale",
        type=float,
        default=1.10,
        help="multiply the physical foreground amplitude at the comparison point",
    )
    parser.add_argument(
        "--scratch-dir", default="./gf_output_noise/coarse_q_scan/"
    )
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def _run_args(args):
    run_args = run_noise_only.parse_args([])
    run_args.mode = args.mode
    run_args.noise_file = args.noise_file
    run_args.galfor_file = args.galfor_file
    run_args.full = args.full
    run_args.two_years = args.two_years
    run_args.modulation = args.modulation
    run_args.unequal_arm = args.unequal_arm
    run_args.gpus = None
    run_args.out_dir = os.path.join(args.scratch_dir, "")
    run_args.tag = "q_scan"
    run_args.progress = False
    run_args.verbose = args.verbose
    # The scan itself constructs coarse views. Keep the run configuration fine.
    run_args.coarse_Q = 1
    return run_args


def _json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def main(argv=None):
    args = parse_args(argv)
    if any(q < 1 for q in args.q_list):
        raise SystemExit("every --q-list value must be >= 1")
    if args.comparison_psd_scale <= 0.0 or args.comparison_galfor_amp_scale <= 0.0:
        raise SystemExit("comparison scales must be positive")

    os.makedirs(args.scratch_dir, exist_ok=True)
    fit = run_noise_only.build_fit(args.mode, _run_args(args))
    general = fit.build_general()

    psd_fiducial = np.asarray(general.psd_injection, dtype=float)
    galfor_fiducial = (
        np.asarray(general.galfor_injection, dtype=float)
        if args.mode == "foreground"
        else None
    )
    fiducial = general.sensitivity_backend(
        "q_scan_fiducial",
        psd_fiducial,
        galfor_params=galfor_fiducial,
    )

    psd_comparison = psd_fiducial * args.comparison_psd_scale
    galfor_comparison = None if galfor_fiducial is None else galfor_fiducial.copy()
    if galfor_comparison is not None:
        galfor_comparison[0] *= args.comparison_galfor_amp_scale
    comparison = general.sensitivity_backend(
        "q_scan_comparison",
        psd_comparison,
        galfor_params=galfor_comparison,
    )

    from lisatools.coarsewdm import coarse_q_scan

    results = coarse_q_scan(
        general.domain_settings,
        fiducial,
        sorted(set(args.q_list)),
        general.input_data_residual_array,
        comparison_sens_mat=comparison,
        use_ws=not args.no_coarse_ws,
    )
    metadata = {
        "mode": args.mode,
        "weighting": "Bartlett" if args.no_coarse_ws else "channelwise WS",
        "fiducial": {
            "psd": psd_fiducial,
            "galfor": galfor_fiducial,
        },
        "comparison": {
            "psd": psd_comparison,
            "galfor": galfor_comparison,
        },
    }

    if args.json:
        print(json.dumps({"metadata": metadata, "results": results}, default=_json_ready, indent=2))
        return

    print(f"mode={args.mode}; weighting={metadata['weighting']}")
    print(f"fiducial psd={psd_fiducial.tolist()}; galfor={None if galfor_fiducial is None else galfor_fiducial.tolist()}")
    print(f"comparison psd={psd_comparison.tolist()}; galfor={None if galfor_comparison is None else galfor_comparison.tolist()}")
    print(
        "\n Q  Ncell  speedup  min(Qeff/n)  median(Qeff/n)  "
        "coarse-fine logL  Delta-logL error  worst diag variation [X,Y,Z]"
    )
    for row in results:
        variation = np.array2string(
            row["worst_diagonal_fractional_variation"], precision=3, separator=","
        )
        print(
            f"{row['Q']:3d} {row['Ncoarse']:6d} {row['nominal_speedup']:8.2f} "
            f"{row['qeff_ratio_min']:13.5f} {row['qeff_ratio_median']:16.5f} "
            f"{row['fiducial_logl_gap']:17.6g} {row['delta_logl_gap']:17.6g}  {variation}"
        )


if __name__ == "__main__":
    main()
