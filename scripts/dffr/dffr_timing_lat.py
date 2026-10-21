"""Timing of DFFR in STFTGBComputations against standalone dffr, on the same inputs.

    python dffr_timing_lat.py --write-inputs INPUTS.npz --setup benchmark --side-bins 20
    python dffr_timing_lat.py --time INPUTS.npz --output OURS.npz --device gpu
    DFFR_PYTHON dffr_timing_dffr.py --time INPUTS.npz --output THEIRS.npz --device gpu
    python dffr_timing_lat.py --report --ours OURS_run*.npz --theirs THEIRS_run*.npz

``--write-inputs`` stores one setup on the ESA orbits that LAT ships, in the layout of
``dffr_exchange_lat.py``: LAT's kinematics table of the grid, the grid and the settings, the
sources, and seeded data and inverse covariance over one domain. The setups:

- ``benchmark``: the setup of standalone dffr's ``scripts/benchmark.py``: one-day segments, the
  band from 1.5 to 28.5 mHz, Tukey taper 0.116, ``--side-bins`` side bins (20 there), and sources
  drawn as there (2 to 28 mHz, seed 7). The ESA orbits replace its Keplerian ones.
- ``production``: the settings of the exchange (the production band from 0.1 to 29 mHz as one
  domain, Tukey taper 0.1157, 10 side bins) and the sources of the production fixtures.

Both use TDI 2nd generation, the linear envelope, first-order delays and a chirp-rate floor of
1e-22 Hz/s.

``--time`` runs ``fill_global_stft`` and ``get_ll_stft`` of the comp on the first B sources, for
every batch size B: one warm-up call, then ``--repeats`` calls, each closed by a device
synchronisation (``stft_benchmarks.time_call``). It keeps every call time, and (d|h) and (h|h) of
the largest batch; with ``--keep-fill`` also its fill_global grid. ``dffr_timing_dffr.py`` does
the same with the standalone package. Its ``get_ll`` computes (d|d) on every call, which the comp
reads from its group, so that step is timed on its own as well.

``--report`` prints, per setup, method and batch size, the median call time per source of each
code (the median over the runs of each run's median), the ratio of the comp to the standalone and
the spread of the run medians of each, (largest - smallest) / median. Then the agreement of the
outputs: the largest difference relative to the peak of the standalone's array.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dffr_exchange_lat import (  # noqa: E402
    CHIRP_RATE_FLOOR,
    FIRST_ORDER_DELAYS,
    LINEAR_ENVELOPE,
    TDI_GENERATION,
    build_dffr_comp,
)
from stft_fixtures import (  # noqa: E402
    DATA_START_OFFSET,
    PRODUCTION_N_SIDE_BINS,
    PRODUCTION_SEED,
    PRODUCTION_WINDOW_ALPHA,
    SAMPLING_INTERVAL,
    SEGMENT_DURATION,
    array_module,
    draw_cell_arrays,
    load_orbits,
    production_settings,
    production_sources,
    run_metadata,
    to_host,
)

SETUPS = ("benchmark", "production")
METHODS = ("fill_global", "get_ll")
BATCH_SIZES = (1, 10, 100, 1000, 10000)
#: Band, taper and seed of standalone dffr's scripts/benchmark.py at 01530a5.
BENCHMARK_MIN_FREQ = 1.5e-3
BENCHMARK_MAX_FREQ = 2.85e-2
BENCHMARK_WINDOW_TAPER = 0.116
BENCHMARK_SEED = 7
#: Grid attributes that build_dffr_comp reads, stored in the description.
GRID_KEYS = ("NT", "NF_active", "t0", "min_freq", "max_freq", "dt", "df")


def benchmark_sources(count, rng):
    """Sources of standalone dffr's ``random_sources`` (scripts/benchmark.py), same draw order."""
    return np.stack(
        [
            np.full(count, 1.0e-22),
            rng.uniform(0.002, 0.028, count),
            10.0 ** rng.uniform(-18.0, -15.0, count),
            np.zeros(count),
            rng.uniform(0, 2 * np.pi, count),
            np.arccos(rng.uniform(-1, 1, count)),
            rng.uniform(0, np.pi, count),
            rng.uniform(0, 2 * np.pi, count),
            np.arcsin(rng.uniform(-1, 1, count)),
        ],
        axis=-1,
    )


def setup_inputs(setup, side_bins, num_segments, num_sources, device):
    """Band, settings, sources, data and inverse covariance of one setup."""
    from lisatools.domains import STFTSettings

    orbits, orbits_start = load_orbits("esa", device)
    data_start = orbits_start + DATA_START_OFFSET
    if setup == "production":
        band, _ = production_settings(data_start, num_segments, device)
        params, _, rng = production_sources(band, num_sources)
        window_taper, side_bins = PRODUCTION_WINDOW_ALPHA, PRODUCTION_N_SIDE_BINS
    else:
        band = STFTSettings(
            t0=data_start,
            dt=SEGMENT_DURATION,
            df=1.0 / SEGMENT_DURATION,
            NT=num_segments,
            NF=int(SEGMENT_DURATION / SAMPLING_INTERVAL) // 2 + 1,
            min_freq=BENCHMARK_MIN_FREQ,
            max_freq=BENCHMARK_MAX_FREQ,
            force_backend=device,
        )
        params = benchmark_sources(num_sources, np.random.default_rng(BENCHMARK_SEED))
        rng = np.random.default_rng(PRODUCTION_SEED)
        window_taper = BENCHMARK_WINDOW_TAPER
    data, inverse_covariance = draw_cell_arrays(rng, num_segments, band.NF_active, False)
    settings = dict(
        setup=setup,
        orbits="esa",
        orbits_start=orbits_start,
        data_start=data_start,
        t_ref=orbits_start,
        window_taper=window_taper,
        num_side_bins=side_bins,
        tdi_generation=TDI_GENERATION,
        linear_envelope=LINEAR_ENVELOPE,
        first_order_delays=FIRST_ORDER_DELAYS,
        chirp_rate_floor=CHIRP_RATE_FLOOR,
        num_sources=num_sources,
    )
    return orbits, band, settings, params, data, inverse_covariance


def write_inputs(path, setup, side_bins, num_segments, num_sources, device):
    """Store one setup with LAT's kinematics table of its grid."""
    from lisatools.domaincomputation import KINEMATICS_TABLE_FIELDS

    orbits, band, settings, params, data, inverse_covariance = setup_inputs(
        setup, side_bins, num_segments, num_sources, device
    )
    _, _, table_arrays = build_dffr_comp(
        orbits,
        settings["t_ref"],
        band,
        data,
        inverse_covariance,
        device,
        n_side_bins=settings["num_side_bins"],
        window_alpha=settings["window_taper"],
    )
    grid = {key: getattr(band, key) for key in GRID_KEYS}
    grid = {
        key: (int(value) if key in ("NT", "NF_active") else float(value))
        for key, value in grid.items()
    }
    # * The keys of dffr_exchange_lat.py, which dffr_exchange_dffr.build_response reads.
    description = dict(
        settings,
        **grid,
        segment_duration=grid["dt"],
        num_segments=grid["NT"],
        first_bin=int(band.ind_min),
        num_bins=grid["NF_active"],
        f_min=grid["min_freq"],
        table_device=device,
    )
    kinematics = {
        f"kinematics__{name}": to_host(table_arrays[name]).reshape(shape + (band.NT,))
        for name, shape in KINEMATICS_TABLE_FIELDS
    }
    np.savez(
        path,
        description=json.dumps(description),
        params=params,
        data=data[None],
        inverse_covariance=inverse_covariance[None],
        **kinematics,
    )
    print(
        f"[timing] wrote {path}: {setup}, band {2 * settings['num_side_bins'] + 1}, "
        f"{band.NT} segments, {band.NF_active} bins, {num_sources} sources"
    )


def time_ours(inputs_path, output_path, device, batch_sizes, repeats, keep_fill):
    """Time fill_global_stft and get_ll_stft of the comp on the stored inputs."""
    from stft_benchmarks import time_call

    from lisatools.domaincomputation import KINEMATICS_TABLE_FIELDS

    inputs = np.load(inputs_path)
    description = json.loads(str(inputs["description"]))
    band = SimpleNamespace(**{key: description[key] for key in GRID_KEYS})
    orbits, _ = load_orbits("esa", device)
    table_fields = {name: inputs[f"kinematics__{name}"] for name, _ in KINEMATICS_TABLE_FIELDS}
    comp, _, _ = build_dffr_comp(
        orbits,
        description["t_ref"],
        band,
        inputs["data"][0],
        inputs["inverse_covariance"][0],
        device,
        n_side_bins=description["num_side_bins"],
        window_alpha=description["window_taper"],
        table_fields=table_fields,
    )
    xp = array_module(device)
    params_all = xp.asarray(np.ascontiguousarray(inputs["params"]))
    template = xp.zeros((1, 3, band.NT, band.NF_active), dtype=np.complex128)

    times = {}
    for batch in batch_sizes:
        params = xp.ascontiguousarray(params_all[:batch])
        indices = xp.zeros(batch, dtype=np.int32)
        factors = xp.ones(batch)
        calls = {
            "fill_global": lambda: comp.fill_global_stft(
                params, template, data_index=indices, factors=factors
            ),
            "get_ll": lambda: comp.get_ll_stft(params, data_index=indices, noise_index=indices),
        }
        for method, call in calls.items():
            times[f"{method}/{batch}"] = np.asarray(time_call(call, device, repeats))
            print(
                f"[timing] ours {method:12s} batch {batch:6d}: "
                f"{1e6 * np.median(times[f'{method}/{batch}']) / batch:10.3f} us per source",
                flush=True,
            )

    # * Outputs of the largest batch, for the agreement with the standalone.
    outputs = {}
    if keep_fill:
        template[...] = 0.0
        comp.fill_global_stft(params, template, data_index=indices, factors=factors)
        outputs["fill_global"] = to_host(template)
    comp.get_ll_stft(params, data_index=indices, noise_index=indices)
    outputs["d_h"] = to_host(comp.d_h_out)
    outputs["h_h"] = to_host(comp.h_h_out)
    metadata = dict(run_metadata(device), description=description, repeats=repeats)
    np.savez(
        output_path,
        metadata=json.dumps(metadata),
        **{f"time__{key}": value for key, value in times.items()},
        **{f"output__{key}": value for key, value in outputs.items()},
    )
    print(f"[timing] wrote {output_path}")


def setup_label(description):
    return f"{description['setup']}, band {2 * description['num_side_bins'] + 1}"


def load_runs(paths):
    """Per setup label: the times of every run, and the outputs of the runs that hold them."""
    runs = {}
    for path in paths:
        stored = np.load(path)
        description = json.loads(str(stored["metadata"]))["description"]
        entry = runs.setdefault(setup_label(description), dict(times=[], outputs={}))
        entry["times"].append(
            {key[len("time__") :]: stored[key] for key in stored.files if key.startswith("time__")}
        )
        for key in stored.files:
            if key.startswith("output__"):
                entry["outputs"].setdefault(key[len("output__") :], stored[key])
    return runs


def spread(values):
    values = np.asarray(values)
    return (values.max() - values.min()) / np.median(values)


def report(ours_paths, theirs_paths):
    """Time per source of each code, their ratio and spreads, then the output agreement."""
    ours, theirs = load_runs(ours_paths), load_runs(theirs_paths)
    for label in sorted(set(ours) & set(theirs)):
        print(
            f"\n## {label}: {len(ours[label]['times'])} runs of ours, "
            f"{len(theirs[label]['times'])} of the standalone"
        )
        print(
            f"{'method':12s} {'batch':>6s} {'ours [us]':>11s} {'dffr [us]':>11s} "
            f"{'dffr - (d|d)':>13s} {'ratio':>7s} {'ratio - (d|d)':>14s} "
            f"{'spread ours':>12s} {'spread dffr':>12s}"
        )
        keys = sorted(
            ours[label]["times"][0], key=lambda key: (key.split("/")[0], int(key.split("/")[1]))
        )
        for key in keys:
            method, batch = key.split("/")
            batch = int(batch)
            ours_medians = [np.median(run[key]) for run in ours[label]["times"]]
            theirs_medians = [np.median(run[key]) for run in theirs[label]["times"]]
            ours_time = np.median(ours_medians) / batch
            theirs_time = np.median(theirs_medians) / batch
            # ? Only the standalone's get_ll pays (d|d) per call; the comp reads it from the group.
            if method == "get_ll":
                d_d_medians = [np.median(run[f"d_d/{batch}"]) for run in theirs[label]["times"]]
                theirs_without = (np.median(theirs_medians) - np.median(d_d_medians)) / batch
            else:
                theirs_without = theirs_time
            print(
                f"{method:12s} {batch:6d} {1e6 * ours_time:11.3f} {1e6 * theirs_time:11.3f} "
                f"{1e6 * theirs_without:13.3f} {ours_time / theirs_time:7.3f} "
                f"{ours_time / theirs_without:14.3f} {spread(ours_medians):12.2%} "
                f"{spread(theirs_medians):12.2%}"
            )
        print(f"\n{'output':14s} {'dffr peak':>12s} {'largest diff':>14s} {'relative':>10s}")
        for name in sorted(set(ours[label]["outputs"]) & set(theirs[label]["outputs"])):
            mine, other = ours[label]["outputs"][name], theirs[label]["outputs"][name]
            peak = float(np.abs(other).max())
            difference = float(np.abs(mine - other).max())
            print(f"{name:14s} {peak:12.4e} {difference:14.4e} {difference / peak:10.2e}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write-inputs", metavar="INPUTS_NPZ")
    action.add_argument("--time", metavar="INPUTS_NPZ")
    action.add_argument("--report", action="store_true")
    parser.add_argument("--setup", choices=SETUPS, default="benchmark")
    parser.add_argument("--side-bins", type=int, default=20)
    parser.add_argument("--num-segments", type=int, default=365)
    parser.add_argument("--num-sources", type=int, default=max(BATCH_SIZES))
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=list(BATCH_SIZES))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--keep-fill", action="store_true", help="store the fill_global grid")
    parser.add_argument("--output", metavar="OUT_NPZ")
    parser.add_argument("--ours", nargs="+", default=[])
    parser.add_argument("--theirs", nargs="+", default=[])
    args = parser.parse_args(argv)
    if args.write_inputs:
        write_inputs(
            args.write_inputs,
            args.setup,
            args.side_bins,
            args.num_segments,
            args.num_sources,
            args.device,
        )
        return 0
    if args.time:
        if args.output is None:
            parser.error("--time needs --output")
        time_ours(
            args.time, args.output, args.device, args.batch_sizes, args.repeats, args.keep_fill
        )
        return 0
    return report(args.ours, args.theirs)


if __name__ == "__main__":
    sys.exit(main())
