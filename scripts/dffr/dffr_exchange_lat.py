"""LAT side of the exchange with standalone dffr: DFFR outputs of both on the same inputs.

    python dffr_exchange_lat.py --capture LAT.npz --device cpu
    DFFR_REPOSITORY/.venv/bin/python dffr_exchange_dffr.py --inputs LAT.npz --output DFFR.npz
    python dffr_exchange_lat.py --compare LAT.npz DFFR.npz

``--capture`` stores the inputs of a DFFR evaluation (the kinematics table of the grid, the grid,
the settings, the sources, the data and the inverse covariance) and LAT's outputs on them: the
segment table, the compact templates and their band starts (diagnostic entry points), and the
fill_global grid, (d|h) and (h|h) (``STFTGBComputations(response="dffr")``, so the comparison
covers the comp's wiring). ``dffr_exchange_dffr.py`` recomputes the same outputs with the
standalone package from the same inputs, and ``--compare`` prints, per output, the largest
difference relative to the peak of the dffr array, against the gate of 1e-13; the band starts must
agree exactly.

Fixture: the ESA orbits that LAT ships, 32 one-day segments from 10 days after their start, the
production band (0.1 to 29 mHz) as one domain, the Tukey taper 0.1157, n_side_bins = 10, TDI 2nd
generation, linear envelope and first-order delays on, the 24 sources of the production fixtures,
and seeded data and inverse covariance over the band. Standalone dffr has no store windows, so the
domain is the whole band, and both sides then read the same f_min. Times stay in LAT's frame, the
start of the orbit file; DFFR uses differences of times only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from stft_fixtures import (  # noqa: E402
    DATA_START_OFFSET,
    PRODUCTION_N_SIDE_BINS,
    PRODUCTION_WINDOW_ALPHA,
    ShimGroup,
    array_module,
    draw_cell_arrays,
    load_orbits,
    production_settings,
    production_sources,
    to_host,
)

NUM_SEGMENTS = 32
NUM_SOURCES = 24
CHIRP_RATE_FLOOR = 1.0e-22
TDI_GENERATION = 2
LINEAR_ENVELOPE = True
FIRST_ORDER_DELAYS = True
#: Largest difference relative to the peak of each dffr array that the exchange accepts.
GATE = 1.0e-13

#: Fields of the segment table, as dffr's segment_table names them, with their row length.
SEGMENT_TABLE_FIELDS = (
    ("effective_frequency", 3, np.float64),
    ("chirp_rate", 3, np.float64),
    ("reference_phasor", 3, np.complex128),
    ("link_amplitude", 6, np.complex128),
    ("envelope_slope", 6, np.complex128),
    ("emission_phasor", 6, np.complex128),
    ("emission_frequency", 6, np.float64),
    ("tdi_weights", 18, np.complex128),
)


def capture(path: str, device: str) -> None:
    """Build the fixture, run LAT's DFFR on it and store inputs and outputs."""
    from gbgpu.gbcomps import STFTGBComputations

    from lisatools.domaincomputation import KINEMATICS_TABLE_FIELDS

    xp = array_module(device)
    orbits, orbits_start = load_orbits("esa", device)
    data_start = orbits_start + DATA_START_OFFSET
    band, _ = production_settings(data_start, NUM_SEGMENTS, device)
    params, _, rng = production_sources(band, NUM_SOURCES)
    data, inverse_covariance = draw_cell_arrays(rng, NUM_SEGMENTS, band.NF_active, False)

    comp = STFTGBComputations(
        stft_comps=None,
        T=NUM_SEGMENTS * band.dt,
        t_ref=orbits_start,
        orbits=orbits,
        tdi_config="2nd generation",
        force_backend=device,
        n_side_bins=PRODUCTION_N_SIDE_BINS,
        response="dffr",
        first_order_delays=FIRST_ORDER_DELAYS,
        chirp_rate_floor=CHIRP_RATE_FLOOR,
    )
    backend = comp.backend
    data_device = xp.asarray(data[None])
    inverse_covariance_device = xp.asarray(inverse_covariance[None])
    grid = (band.NT, band.NF_active, 3, band.t0, band.min_freq, band.max_freq, band.dt, band.df)
    domain = backend.STFTDomainWrap(
        *grid,
        data_device.reshape(-1),
        inverse_covariance_device.reshape(-1),
        1,
        1,
        backend.TDITypeDict["XYZ"],
    )
    fresnel = backend.STFTFresnelWrap(
        *grid,
        window_alpha=PRODUCTION_WINDOW_ALPHA,
        use_midpoint=True,
        linear_envelope=LINEAR_ENVELOPE,
    )
    table_arrays = {
        name: xp.zeros(int(np.prod(shape)) * band.NT) for name, shape in KINEMATICS_TABLE_FIELDS
    }
    table = backend.KinematicsTableWrap(
        band.t0 + 0.5 * band.dt, band.dt, band.NT, *table_arrays.values()
    )
    table.build(comp.cpp_orbits)
    # * fill_global and get_ll run through the comp; the group stand-in hands it this table.
    comp.stft_comps = ShimGroup(
        fresnel,
        domain,
        num_noise=1,
        buffers=(data_device, inverse_covariance_device),
        linear_envelope=LINEAR_ENVELOPE,
        kinematics=table,
    )

    wrap = backend.GBComputationGroupWrap()
    knobs = (
        orbits_start,
        PRODUCTION_N_SIDE_BINS,
        TDI_GENERATION,
        LINEAR_ENVELOPE,
        FIRST_ORDER_DELAYS,
        CHIRP_RATE_FLOOR,
    )
    params_device = xp.asarray(np.ascontiguousarray(params))
    num_sources = params.shape[0]
    indices = xp.zeros(num_sources, dtype=np.int32)

    outputs = {}
    table_outputs = {
        name: xp.zeros(num_sources * band.NT * length, dtype=dtype)
        for name, length, dtype in SEGMENT_TABLE_FIELDS
    }
    starts = xp.zeros(num_sources * band.NT, dtype=np.int32)
    wrap.gb_dffr_segment_table(
        *table_outputs.values(),
        starts,
        table,
        fresnel,
        domain,
        params_device.reshape(-1),
        num_sources,
        9,
        *knobs,
    )
    # * The kernel writes [source][segment][k]; dffr returns [source][k][segment].
    for name, length, _ in SEGMENT_TABLE_FIELDS:
        outputs[f"segment_table__{name}"] = np.moveaxis(
            to_host(table_outputs[name]).reshape(num_sources, band.NT, length), -1, 1
        )
    outputs["segment_table__start_freq_ind"] = to_host(starts).reshape(num_sources, band.NT)

    band_width = 2 * PRODUCTION_N_SIDE_BINS + 1
    compact = xp.zeros(num_sources * 3 * band.NT * band_width, dtype=np.complex128)
    compact_starts = xp.zeros(num_sources * band.NT, dtype=np.int32)
    wrap.gb_dffr_fill_template(
        compact,
        compact_starts,
        table,
        fresnel,
        domain,
        params_device.reshape(-1),
        num_sources,
        9,
        *knobs,
    )
    outputs["compact_template"] = to_host(compact).reshape(num_sources, 3, band.NT, band_width)
    outputs["compact_start"] = to_host(compact_starts).reshape(num_sources, band.NT)

    grid_template = xp.zeros((1, 3, band.NT, band.NF_active), dtype=np.complex128)
    comp.fill_global_stft(
        params_device, grid_template, data_index=indices, factors=xp.ones(num_sources)
    )
    outputs["fill_global"] = to_host(grid_template)

    comp.get_ll_stft(params_device, data_index=indices, noise_index=indices)
    outputs["d_h"] = to_host(comp.d_h_out)
    outputs["h_h"] = to_host(comp.h_h_out)

    num_times = band.NT
    kinematics = {
        f"kinematics__{name}": to_host(table_arrays[name]).reshape(shape + (num_times,))
        for name, shape in KINEMATICS_TABLE_FIELDS
    }
    description = dict(
        device=device,
        orbits="esa",
        orbits_start=orbits_start,
        data_start=data_start,
        t0=band.t0,
        segment_duration=band.dt,
        num_segments=int(band.NT),
        first_bin=int(band.ind_min),
        num_bins=int(band.NF_active),
        df=band.df,
        f_min=band.min_freq,
        t_ref=orbits_start,
        window_taper=PRODUCTION_WINDOW_ALPHA,
        num_side_bins=PRODUCTION_N_SIDE_BINS,
        tdi_generation=TDI_GENERATION,
        linear_envelope=LINEAR_ENVELOPE,
        first_order_delays=FIRST_ORDER_DELAYS,
        chirp_rate_floor=CHIRP_RATE_FLOOR,
        num_sources=num_sources,
    )
    np.savez(
        path,
        description=json.dumps(description),
        params=params,
        data=data[None],
        inverse_covariance=inverse_covariance[None],
        **kinematics,
        **{f"lat__{name}": value for name, value in outputs.items()},
    )
    print(f"[exchange] wrote {path}")


def compare(lat_path: str, dffr_path: str) -> int:
    """Per output, the largest difference relative to the peak of the dffr array."""
    lat = np.load(lat_path)
    dffr = np.load(dffr_path)
    names = sorted(key[len("lat__") :] for key in lat.files if key.startswith("lat__"))
    failures = 0
    print(f"{'output':40s} {'dffr peak':>12s} {'largest diff':>14s} {'relative':>10s}  result")
    for name in names:
        ours, theirs = lat[f"lat__{name}"], dffr[f"dffr__{name}"]
        if ours.shape != theirs.shape:
            print(f"{name:40s} shapes differ: {ours.shape} against {theirs.shape}  FAIL")
            failures += 1
            continue
        if np.issubdtype(ours.dtype, np.integer):
            equal = np.array_equal(ours, theirs)
            print(
                f"{name:40s} {'':>12s} {'':>14s} {'':>10s}  "
                f"{'exact' if equal else 'DIFFER'} {'pass' if equal else 'FAIL'}"
            )
            failures += 0 if equal else 1
            continue
        peak = float(np.abs(theirs).max())
        difference = float(np.abs(ours - theirs).max())
        relative = difference / peak if peak > 0.0 else difference
        passed = relative <= GATE
        failures += 0 if passed else 1
        print(
            f"{name:40s} {peak:12.4e} {difference:14.4e} {relative:10.2e}  "
            f"{'pass' if passed else 'FAIL'}"
        )
    print(f"[exchange] {len(names)} outputs compared, {failures} above the gate of {GATE:g}")
    return 1 if failures else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--capture", metavar="LAT_NPZ")
    action.add_argument("--compare", nargs=2, metavar=("LAT_NPZ", "DFFR_NPZ"))
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    args = parser.parse_args(argv)
    if args.capture:
        capture(args.capture, args.device)
        return 0
    return compare(*args.compare)


if __name__ == "__main__":
    sys.exit(main())
