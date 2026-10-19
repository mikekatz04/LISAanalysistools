"""Golden outputs of the seven STFT GB kernels behind ``gbgpu.gbcomps.STFTGBComputations``.

Evaluates ``get_ll_stft``, ``get_swap_ll_stft``, ``get_fstat_ll_stft``, ``fill_global_stft``,
``get_ll_grad_stft``, ``get_swap_ll_grad_stft`` and ``information_matrix`` on the fixtures of
``stft_fixtures.py`` and stores every output, or compares a build against a stored capture.
The oracle configurations keep the array names of
``scripts/validation/stft_column_policy_oracle.py``, so the two captures can be compared.

Usage:
    python stft_golden_outputs.py --capture OUT.npz --device cpu --mojito-l1 FILE.h5
    python stft_golden_outputs.py --compare REF.npz --device cpu --mojito-l1 FILE.h5
    python stft_golden_outputs.py --compare-files REF.npz NEW.npz [--common-only]

Comparisons ask for bit identity unless ``--rtol`` is given; differences are then normalised by
the largest absolute value of each reference array. The exit status is 1 when any array fails.
``--without-swap`` leaves out the two swap kernels, in captures and comparisons; the CPU runs use
it from step 2 on. ``--response dffr`` evaluates the DFFR response on its configurations
(``stft_fixtures.DFFR_CONFIGS``), without the Fisher matrices, which it does not have yet.
Fixtures, tolerances and results are recorded in ``docs/response/dffr/tests.md``.
"""

import argparse
import json
import sys

import numpy as np

import stft_fixtures as fixtures

METADATA_KEY = "__metadata__"
#: Output names of get_swap_ll_stft (sw_*) and get_swap_ll_grad_stft (gswap_*).
SWAP_OUTPUT_PREFIXES = ("sw_", "gswap_")

# * Fisher evaluations: four sources, two per store window in the production fixtures.
NUM_FISHER_SOURCES = 4
# * F-statistic: 4 log-likelihood and 6 swap block evaluations per source, so a lighter batch.
NUM_FSTAT_SOURCES = 8
# * Subset of (amp, f0, fdot, fddot, phi0, iota, psi, lam, beta): f0, fdot, lam, beta.
FISHER_SUBSET = [1, 2, 7, 8]


def swap_partners(cells):
    """Index of each source's swap partner: three places back within its cell.

    With one cell this is ``np.roll(params, 3, axis=0)``, the partner set of the oracle script.
    """
    partners = np.empty(cells.size, dtype=np.int64)
    for cell in np.unique(cells):
        members = np.flatnonzero(cells == cell)
        partners[members] = np.roll(members, 3)
    return partners


def first_sources(cells, count):
    """The first ``count`` sources, taken evenly from every cell."""
    selected = []
    unique_cells = np.unique(cells)
    per_cell = count // unique_cells.size
    for cell in unique_cells:
        selected.extend(np.flatnonzero(cells == cell)[:per_cell].tolist())
    return np.sort(np.asarray(selected))


def is_swap_output(key):
    """True for the outputs of get_swap_ll_stft and get_swap_ll_grad_stft (any fixture prefix)."""
    return key.rsplit("/", 1)[-1].startswith(SWAP_OUTPUT_PREFIXES)


def evaluate(fixture, with_swap=True):
    """Every kernel output of one fixture, as host arrays keyed like the oracle script.

    ``with_swap=False`` leaves out the two swap kernels, as the CPU runs do from step 2 on.
    """
    comp = fixture.comp
    xp = comp.xp
    backend = fixture.description.get("backend")
    params = fixture.params
    cells = xp.asarray(fixture.cells)
    starts = fixture.window_starts
    partners = swap_partners(fixture.cells)
    params_partner = params[partners]
    is_production = fixture.description["kind"] == "production"
    outputs = {}

    log_like = comp.get_ll_stft(params, data_index=cells, noise_index=cells, start_freq_inds=starts)
    outputs["ll"] = log_like
    outputs["d_h"] = comp.d_h_out
    outputs["h_h"] = comp.h_h_out

    if with_swap:
        swap_names = ["sw_like_a", "sw_like_r", "sw_dha", "sw_dhr", "sw_aa", "sw_rr", "sw_ar"]
        swap_terms = comp.get_swap_ll_stft(
            params, params_partner, data_index=cells, noise_index=cells, start_freq_inds=starts
        )
        outputs.update(zip(swap_names, swap_terms))

    templates = xp.zeros((fixture.num_templates,) + fixture.template_shape, dtype=xp.complex128)
    # * Opposite signs on alternate sources, so the scatter both adds and subtracts.
    factors = np.where(np.arange(params.shape[0]) % 2 == 0, 1.0, -0.5)
    fill_rows = cells if is_production else xp.asarray(np.arange(params.shape[0]) % 2)
    comp.fill_global_stft(
        params,
        templates,
        data_index=xp.ascontiguousarray(fill_rows.astype(xp.int32)),
        factors=xp.asarray(factors),
        start_freq_inds=starts,
    )
    outputs["fill"] = templates

    outputs["grad"] = comp.get_ll_grad_stft(
        params, data_index=cells, noise_index=cells, start_freq_inds=starts
    )
    if with_swap:
        grad_add, grad_remove = comp.get_swap_ll_grad_stft(
            params, params_partner, data_index=cells, noise_index=cells, start_freq_inds=starts
        )
        outputs["gswap_a"] = grad_add
        outputs["gswap_r"] = grad_remove
    if is_production:
        # * Envelope on: production templates use it, derivative paths default to off.
        outputs["grad_envelope"] = comp.get_ll_grad_stft(
            params,
            data_index=cells,
            noise_index=cells,
            start_freq_inds=starts,
            linear_envelope=True,
        )
        if with_swap:
            grad_add, grad_remove = comp.get_swap_ll_grad_stft(
                params,
                params_partner,
                data_index=cells,
                noise_index=cells,
                start_freq_inds=starts,
                linear_envelope=True,
            )
            outputs["gswap_a_envelope"] = grad_add
            outputs["gswap_r_envelope"] = grad_remove

    fstat_sources = first_sources(fixture.cells, NUM_FSTAT_SOURCES)
    fstat_cells = xp.asarray(fixture.cells[fstat_sources])
    fstat_n, fstat_m = comp.get_fstat_ll_stft(
        params[fstat_sources],
        data_index=fstat_cells,
        noise_index=fstat_cells,
        start_freq_inds=starts,
    )
    outputs["fstat_N"] = fstat_n
    outputs["fstat_M"] = fstat_m
    outputs["fstat_N_cmplx"] = comp.N_arr_cmplx
    outputs["fstat_M_cmplx"] = comp.M_mat_cmplx

    if comp.response == "dffr":
        fixtures.synchronize(backend)
        return {key: fixtures.to_host(value) for key, value in outputs.items()}
    fisher_sources = first_sources(fixture.cells, NUM_FISHER_SOURCES)
    fisher_params = params[fisher_sources]
    fisher_cells = xp.asarray(fixture.cells[fisher_sources])
    fisher_variants = {
        "fisher": dict(),
        "fisher_easy_cd": dict(easy_central_difference=True),
        "fisher_tdi_track": dict(astro_track=False),
        "fisher_subset": dict(inds=FISHER_SUBSET),
    }
    for key, options in fisher_variants.items():
        outputs[key] = comp.information_matrix(
            fisher_params, noise_index=fisher_cells, start_freq_inds=starts, **options
        )
    if is_production:
        outputs["fisher_envelope"] = comp.information_matrix(
            fisher_params,
            noise_index=fisher_cells,
            start_freq_inds=starts,
            linear_envelope=True,
        )

    fixtures.synchronize(backend)
    return {key: fixtures.to_host(value) for key, value in outputs.items()}


def capture(path, backend, config_names, mojito_l1, with_swap=True, response="tdi_on_the_fly"):
    orbits_cache = {}
    blob = {}
    descriptions = {}
    for name in config_names:
        fixture = fixtures.build_fixture(name, backend, mojito_l1, orbits_cache, response)
        fixture.description["backend"] = backend
        outputs = evaluate(fixture, with_swap)
        for key, value in outputs.items():
            blob[f"{name}/{key}"] = value
        descriptions[name] = fixture.description
        print(
            f"[capture] {name}: {len(outputs)} arrays, |d_h[0]| = {abs(outputs['d_h'][0]):.6e}",
            flush=True,
        )
    metadata = fixtures.run_metadata(backend, mojito_l1)
    metadata["configs"] = descriptions
    metadata["with_swap"] = with_swap
    metadata["response"] = response
    blob[METADATA_KEY] = np.array(json.dumps(metadata))
    np.savez(path, **blob)
    print(f"[capture] wrote {path} ({len(blob) - 1} arrays)")


def compare_arrays(reference, candidate, rtol):
    """Comparison record of one array; ``passed`` is bit identity or the normalised tolerance."""
    record = dict(shape=list(reference.shape), identical=bool(np.array_equal(reference, candidate)))
    if reference.shape != candidate.shape:
        record.update(passed=False, reason="shape differs")
        return record
    difference = np.abs(candidate - reference)
    scale = float(np.max(np.abs(reference))) if reference.size else 0.0
    max_abs = float(difference.max()) if difference.size else 0.0
    record.update(
        max_abs_diff=max_abs,
        max_rel_diff=max_abs / scale if scale > 0 else (0.0 if max_abs == 0 else np.inf),
        num_differing=int(np.count_nonzero(difference)),
        size=int(reference.size),
        max_ulp=_max_ulp(reference, candidate),
    )
    record["passed"] = record["identical"] if rtol == 0 else record["max_rel_diff"] <= rtol
    return record


def _max_ulp(reference, candidate):
    """Largest difference in units of the last place, over real and imaginary parts."""
    parts = [(reference.real, candidate.real)]
    if np.iscomplexobj(reference):
        parts.append((reference.imag, candidate.imag))
    largest = 0.0
    for reference_part, candidate_part in parts:
        reference_part = np.asarray(reference_part, dtype=np.float64)
        candidate_part = np.asarray(candidate_part, dtype=np.float64)
        differs = reference_part != candidate_part
        if not differs.any():
            continue
        spacing = np.spacing(np.maximum(np.abs(reference_part), np.abs(candidate_part))[differs])
        ulps = np.abs(candidate_part[differs] - reference_part[differs]) / spacing
        largest = max(largest, float(ulps.max()))
    return largest


def report(reference_blob, candidate_blob, rtol, common_only=False):
    reference_keys = {key for key in reference_blob if key != METADATA_KEY}
    candidate_keys = {key for key in candidate_blob if key != METADATA_KEY}
    keys = sorted(reference_keys & candidate_keys)
    missing = sorted(reference_keys - candidate_keys)
    if missing and not common_only:
        print(f"[compare] {len(missing)} reference arrays missing from the candidate: {missing}")
    num_failed = 0 if (common_only or not missing) else len(missing)
    for key in keys:
        record = compare_arrays(
            np.asarray(reference_blob[key]), np.asarray(candidate_blob[key]), rtol
        )
        if record["identical"]:
            continue
        status = "ok  " if record["passed"] else "FAIL"
        num_failed += 0 if record["passed"] else 1
        print(
            f"[{status}] {key}: max|diff| {record.get('max_abs_diff', np.nan):.3e}, "
            f"max rel {record.get('max_rel_diff', np.nan):.3e}, "
            f"{record.get('num_differing', 0)}/{record.get('size', 0)} differ, "
            f"max {record.get('max_ulp', np.nan):.1f} ulp"
        )
    num_identical = sum(
        np.array_equal(np.asarray(reference_blob[key]), np.asarray(candidate_blob[key]))
        for key in keys
    )
    print(
        f"[compare] {len(keys)} arrays compared, {num_identical} bit-identical, "
        f"{num_failed} failed (rtol = {rtol:g})"
    )
    return 1 if num_failed else 0


def load_blob(path):
    with np.load(path, allow_pickle=False) as stored:
        return {key: stored[key] for key in stored.files}


def stored_configs(blob):
    if METADATA_KEY in blob:
        return list(json.loads(str(blob[METADATA_KEY]))["configs"])
    return sorted({key.split("/", 1)[0] for key in blob})


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--capture", metavar="OUT_NPZ")
    mode.add_argument("--compare", metavar="REF_NPZ")
    mode.add_argument("--compare-files", nargs=2, metavar=("REF_NPZ", "NEW_NPZ"))
    parser.add_argument(
        "--device",
        "--backend",
        dest="backend",
        choices=["cpu", "gpu"],
        default="cpu",
        help="cpu or gpu (--backend is the older name)",
    )
    parser.add_argument("--mojito-l1", default=None, help="Mojito L1 .h5 file (l1 fixtures)")
    parser.add_argument("--configs", nargs="+", choices=fixtures.ALL_CONFIGS, default=None)
    parser.add_argument(
        "--rtol", type=float, default=0.0, help="normalised tolerance; 0 asks for bit identity"
    )
    parser.add_argument(
        "--common-only",
        action="store_true",
        help="with --compare-files, compare only the arrays both files hold",
    )
    parser.add_argument(
        "--without-swap",
        action="store_true",
        help="leave out get_swap_ll_stft and get_swap_ll_grad_stft, in captures and comparisons",
    )
    parser.add_argument("--response", choices=fixtures.RESPONSES, default=fixtures.RESPONSES[0])
    args = parser.parse_args(argv)
    with_swap = not args.without_swap
    default_configs = (
        fixtures.ALL_CONFIGS if args.response == fixtures.RESPONSES[0] else fixtures.DFFR_CONFIGS
    )

    def without_swap(blob):
        # * A reference that holds swap arrays still serves a run without them.
        return {key: value for key, value in blob.items() if with_swap or not is_swap_output(key)}

    if args.capture:
        config_names = args.configs or list(default_configs)
        capture(args.capture, args.backend, config_names, args.mojito_l1, with_swap, args.response)
        return 0
    if args.compare_files:
        reference_blob, candidate_blob = (load_blob(path) for path in args.compare_files)
        return report(
            without_swap(reference_blob), without_swap(candidate_blob), args.rtol, args.common_only
        )

    reference_blob = load_blob(args.compare)
    config_names = args.configs or stored_configs(reference_blob)
    candidate_blob = {}
    orbits_cache = {}
    for name in config_names:
        fixture = fixtures.build_fixture(
            name, args.backend, args.mojito_l1, orbits_cache, args.response
        )
        fixture.description["backend"] = args.backend
        for key, value in evaluate(fixture, with_swap).items():
            candidate_blob[f"{name}/{key}"] = value
    reference_subset = {
        key: value
        for key, value in without_swap(reference_blob).items()
        if key.split("/", 1)[0] in config_names
    }
    return report(reference_subset, candidate_blob, args.rtol)


if __name__ == "__main__":
    sys.exit(main())
