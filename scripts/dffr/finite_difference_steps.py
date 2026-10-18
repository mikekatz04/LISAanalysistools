"""Gradients and Fisher matrices of the golden fixtures at several finite-difference steps.

    python finite_difference_steps.py --capture OUT.npz --device cpu [--configs ...]
    python finite_difference_steps.py --compare REF.npz NEW.npz

The gradients and Fisher matrices of STFTGBComputations are central differences of templates. A
change of the code that moves only rounding enters them as noise divided by the step, so its
effect falls with the step; a change of the modelled signal does not depend on the step.
--capture evaluates ``get_ll_grad_stft`` and ``information_matrix`` (defaults and the TDI track)
with each method's default steps multiplied by 1, 10 and 100. --compare prints two tables:

* per array and factor, the largest change relative to the array's peak, and the ratio between
  factors;
* the criterion for these outputs: per gradient component (largest over sources) and per Fisher
  entry, the change between the two codes at the default step against each code's own noise,
  the difference between its values at the default and the tenfold step. All are divided by the
  reference at the tenfold step: the component's largest value, or sqrt(|F_ii F_jj|). Every
  component or entry whose change exceeds the larger of the two noises is listed.

Recorded in docs/response/dffr/tests.md.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import stft_fixtures as fixtures
from stft_golden_outputs import NUM_FISHER_SOURCES, first_sources, load_blob

STEP_FACTORS = (1, 10, 100)
DEFAULT_CONFIGS = (
    "rect",
    "tukey_mid",
    "astro",
    "esa_tdi2_envelope",
    "esa_tdi1_const",
    "esa_aet_diag",
)
NUM_PARAMETERS = 9
#: Parameter order of get_ll_grad_stft and information_matrix.
PARAMETER_NAMES = ("amp", "f0", "fdot", "fddot", "phi0", "iota", "psi", "lam", "beta")


def evaluate(fixture):
    """{array@factor: host array} for the gradient and two Fisher variants at every factor."""
    comp = fixture.comp
    xp = comp.xp
    params = fixture.params
    cells = xp.asarray(fixture.cells)
    starts = fixture.window_starts
    fisher_sources = first_sources(fixture.cells, NUM_FISHER_SOURCES)
    fisher_params = params[fisher_sources]
    fisher_cells = xp.asarray(fixture.cells[fisher_sources])
    # * Each method's own default steps, so factor 1 reproduces the golden outputs.
    gradient_steps = comp._default_param_eps(NUM_PARAMETERS)
    fisher_steps, _ = comp._information_matrix_steps(NUM_PARAMETERS, None, None, 1.0e-6)
    outputs = {}
    for factor in STEP_FACTORS:
        outputs[f"grad@{factor}"] = comp.get_ll_grad_stft(
            params,
            param_eps=gradient_steps * factor,
            data_index=cells,
            noise_index=cells,
            start_freq_inds=starts,
        )
        for name, options in (("fisher", dict()), ("fisher_tdi_track", dict(astro_track=False))):
            outputs[f"{name}@{factor}"] = comp.information_matrix(
                fisher_params,
                param_eps=fisher_steps * factor,
                noise_index=fisher_cells,
                start_freq_inds=starts,
                **options,
            )
    fixtures.synchronize(fixture.description.get("backend"))
    return {key: fixtures.to_host(value) for key, value in outputs.items()}


def capture(path, backend, config_names, mojito_l1):
    orbits_cache = {}
    blob = {}
    for name in config_names:
        fixture = fixtures.build_fixture(name, backend, mojito_l1, orbits_cache)
        fixture.description["backend"] = backend
        for key, value in evaluate(fixture).items():
            blob[f"{name}/{key}"] = value
        print(f"[capture] {name}", flush=True)
    np.savez(path, **blob)
    print(f"[capture] wrote {path} ({len(blob)} arrays)")


def peak_changes(reference, candidate, stems):
    header = " ".join(f"{'x' + str(factor):>10s}" for factor in STEP_FACTORS)
    print(f"{'array':38s} {header}   ratios between successive factors")
    for stem in stems:
        changes = []
        for factor in STEP_FACTORS:
            key = f"{stem}@{factor}"
            peak = np.abs(reference[key]).max()
            changes.append(np.abs(candidate[key] - reference[key]).max() / peak)
        ratios = [
            changes[index] / changes[index + 1] if changes[index + 1] > 0 else np.inf
            for index in range(len(changes) - 1)
        ]
        values = " ".join(f"{change:10.2e}" for change in changes)
        print(f"{stem:38s} {values}   " + " ".join(f"{ratio:6.1f}" for ratio in ratios))


def scaled_measure(stem, reference_tenfold):
    """Function mapping a difference array to its size per gradient component or Fisher entry."""
    if stem.rsplit("/", 1)[-1].startswith("grad"):
        # * Gradients (sources, parameters): per parameter, largest over sources.
        scale = np.abs(reference_tenfold).max(axis=0)
        return lambda difference: np.abs(difference).max(axis=0) / scale
    diagonal = np.abs(np.diagonal(reference_tenfold, axis1=-2, axis2=-1))
    scale = np.sqrt(diagonal[..., :, None] * diagonal[..., None, :])
    return lambda difference: np.divide(
        np.abs(difference), scale, out=np.zeros_like(scale), where=scale > 0
    )


def noise_criterion(reference, candidate, stems):
    print(
        f"\n{'array':38s} {'within':>11s} {'largest change':>15s} {'largest noise':>14s} "
        f"{'largest change/noise':>21s}"
    )
    outliers = []
    for stem in stems:
        reference_default, reference_tenfold = reference[f"{stem}@1"], reference[f"{stem}@10"]
        candidate_default, candidate_tenfold = candidate[f"{stem}@1"], candidate[f"{stem}@10"]
        measure = scaled_measure(stem, reference_tenfold)
        change = measure(candidate_default - reference_default)
        noise = np.maximum(
            measure(reference_default - reference_tenfold),
            measure(candidate_default - candidate_tenfold),
        )
        ratio = np.divide(change, noise, out=np.zeros_like(change), where=noise > 0)
        ratio[(noise == 0) & (change > 0)] = np.inf
        within = int(np.count_nonzero(change <= noise))
        print(
            f"{stem:38s} {within:5d} of {change.size:<4d} {change.max():15.2e} {noise.max():14.2e} "
            f"{ratio.max():21.2f}"
        )
        for index in zip(*np.nonzero(change > noise)):
            outliers.append((stem, index, change[index], noise[index]))
    for stem, index, change_value, noise_value in outliers:
        print(
            f"  above the noise: {stem}, {entry_label(index)}: change {change_value:.2e}, "
            f"noise {noise_value:.2e}"
        )


def entry_label(index):
    """Parameter name of a gradient component, or source and parameter pair of a Fisher entry."""
    if len(index) == 1:
        return PARAMETER_NAMES[index[0]]
    source, row, column = index
    return f"source {source} ({PARAMETER_NAMES[row]}, {PARAMETER_NAMES[column]})"


def compare(reference_path, candidate_path):
    reference, candidate = load_blob(reference_path), load_blob(candidate_path)
    stems = sorted({key.rsplit("@", 1)[0] for key in reference if key in candidate})
    peak_changes(reference, candidate, stems)
    noise_criterion(reference, candidate, stems)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--capture", metavar="OUT_NPZ")
    mode.add_argument("--compare", nargs=2, metavar=("REF_NPZ", "NEW_NPZ"))
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--configs", nargs="+", default=list(DEFAULT_CONFIGS))
    parser.add_argument("--mojito-l1", default=None, help="Mojito L1 .h5 file (l1 fixtures)")
    args = parser.parse_args(argv)
    if args.capture:
        capture(args.capture, args.device, args.configs, args.mojito_l1)
    else:
        compare(*args.compare)
    return 0


if __name__ == "__main__":
    sys.exit(main())
