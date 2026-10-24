"""Timings of two sets of benchmark runs side by side, for the records in results.md.

    python compare_benchmarks.py --reference step0/cpu_esa_run*.json \\
        --candidate step1/cpu_esa_run*.json

Each set holds several runs of ``stft_benchmarks.py`` in separate processes. For every method
and batch size the table gives the median over the runs of each run's median time, the ratio
candidate / reference, and the spread of each set, (largest - smallest) / median of the run
medians, which is the run-to-run noise a ratio is read against. A point lies within that noise
when |ratio - 1| does not exceed the larger of the two spreads; this is the timing criterion of
a step that must not change the speed.
"""

from __future__ import annotations

import argparse
import json

import numpy as np


def run_medians(paths):
    """{(method, batch_size): [median of each run in seconds]}."""
    medians = {}
    for path in paths:
        with open(path) as handle:
            for record in json.load(handle)["results"]:
                key = (record["method"], record["batch_size"])
                medians.setdefault(key, []).append(record["median_s"])
    return medians


def spread(values):
    values = np.asarray(values)
    return (values.max() - values.min()) / np.median(values)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--reference", nargs="+", required=True, metavar="JSON")
    parser.add_argument("--candidate", nargs="+", required=True, metavar="JSON")
    args = parser.parse_args(argv)

    reference = run_medians(args.reference)
    candidate = run_medians(args.candidate)
    print(
        f"{'method':24s} {'batch':>6s} {'reference [ms]':>15s} {'candidate [ms]':>15s} "
        f"{'ratio':>7s} {'ref spread':>11s} {'cand spread':>12s} {'within':>7s}"
    )
    ratios = []
    num_within = 0
    for key in sorted(reference, key=lambda item: (item[0], item[1])):
        if key not in candidate:
            continue
        reference_ms = 1e3 * np.median(reference[key])
        candidate_ms = 1e3 * np.median(candidate[key])
        ratio = candidate_ms / reference_ms
        ratios.append(ratio)
        noise = max(spread(reference[key]), spread(candidate[key]))
        within = abs(ratio - 1.0) <= noise
        num_within += int(within)
        print(
            f"{key[0]:24s} {key[1]:6d} {reference_ms:15.2f} {candidate_ms:15.2f} {ratio:7.3f} "
            f"{100 * spread(reference[key]):10.2f}% {100 * spread(candidate[key]):11.2f}% "
            f"{'yes' if within else 'no':>7s}"
        )
    print(f"\nratio range {min(ratios):.3f} to {max(ratios):.3f}, median {np.median(ratios):.3f}")
    print(
        f"{num_within} of {len(ratios)} points within the larger run-to-run spread of the two sets"
    )


if __name__ == "__main__":
    main()
