"""DFFR against the existing STFT response on the production sources: overlap and norm ratio.

    python dffr_against_existing.py --device cpu [--output RESULTS.npz]

Both responses evaluate the same sources on the same grid. Per source, the script fills the
template of each over the whole band (fill_global, factor one) and prints, with a the DFFR
template and b the existing one, summed over the three channels with unit weights:

* the mismatch 1 - Re(a|b) / sqrt((a|a) (b|b));
* the phase of (a|b), which a constant phase offset between the two would set;
* the norm ratio (a|a) / (b|b).

There is no threshold. The numbers describe how far apart the two models are; how far each lies
from a time-domain truth is measured in step 6.

Fixture: that of dffr_exchange_lat.py (ESA orbits, 32 one-day segments from 10 days after their
start, the production band as one domain, the Tukey taper 0.1157 with the segment midpoints,
n_side_bins = 10, the 24 sources of the production fixtures). The settings are the two production
settings of the existing response on ESA orbits, TDI 2nd generation with the linear envelope and
TDI 1st generation without it, and the two crossed ones, which tell the TDI generation and the
envelope apart. DFFR takes the same generation and envelope, with its first-order delay terms on.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from stft_fixtures import (  # noqa: E402
    DATA_START_OFFSET,
    PRODUCTION_N_SIDE_BINS,
    PRODUCTION_WINDOW_ALPHA,
    SEGMENT_DURATION,
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
FIRST_ORDER_DELAYS = True
#: (name, TDI generation, linear envelope): the production settings on ESA orbits, then the
#: crossed ones.
SETTINGS = (
    ("esa_tdi2_envelope", 2, True),
    ("esa_tdi1_const", 1, False),
    ("esa_tdi2_const", 2, False),
    ("esa_tdi1_envelope", 1, True),
)


class _ShimGroup:
    """Stand-in for STFTComputationGroup with what fill_global_stft reads."""

    def __init__(self, cpp_fresnel, cpp_domain):
        self.cpp_fresnel = cpp_fresnel
        self.cpp_domain = cpp_domain
        self.d_d = None


def templates_of_both(device, orbits, band, params, data, generation, linear_envelope):
    """(DFFR, existing) templates (num_sources, 3, NT, NF_active), one per source, host arrays."""
    from gbgpu.gbcomps import STFTGBComputations

    from lisatools.domaincomputation import KINEMATICS_TABLE_FIELDS

    xp = array_module(device)
    orbits_object, orbits_start = orbits
    comp = STFTGBComputations(
        stft_comps=None,
        T=NUM_SEGMENTS * SEGMENT_DURATION,
        t_ref=orbits_start,
        orbits=orbits_object,
        tdi_config=f"{'1st' if generation == 1 else '2nd'} generation",
        force_backend=device,
        n_side_bins=PRODUCTION_N_SIDE_BINS,
        window_factor=1.0,
        freq_from_tdi_phase=True,
        window_alpha=PRODUCTION_WINDOW_ALPHA,
        use_midpoint=True,
        linear_envelope=linear_envelope,
    )
    backend = comp.backend
    # ! The domain wrap holds raw pointers into these arrays; fill_global does not read them.
    data_device = xp.asarray(data[0][None])
    inverse_covariance_device = xp.asarray(data[1][None])
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
        linear_envelope=linear_envelope,
    )
    comp.stft_comps = _ShimGroup(fresnel, domain)
    table_arrays = {
        name: xp.zeros(int(np.prod(shape)) * band.NT) for name, shape in KINEMATICS_TABLE_FIELDS
    }
    table = backend.KinematicsTableWrap(
        band.t0 + 0.5 * band.dt, band.dt, band.NT, *table_arrays.values()
    )
    table.build(comp.cpp_orbits)

    num_sources = params.shape[0]
    shape = (num_sources, 3, band.NT, band.NF_active)
    # * One template per source: the fill kernels add source i into template data_index[i].
    indices = xp.arange(num_sources, dtype=np.int32)
    factors = xp.ones(num_sources)
    dffr = xp.zeros(shape, dtype=np.complex128)
    backend.GBComputationGroupWrap().gb_dffr_fill_global(
        dffr.reshape(-1),
        table,
        fresnel,
        domain,
        xp.asarray(np.ascontiguousarray(params)).reshape(-1),
        indices,
        factors,
        num_sources,
        9,
        orbits_start,
        PRODUCTION_N_SIDE_BINS,
        generation,
        linear_envelope,
        FIRST_ORDER_DELAYS,
        CHIRP_RATE_FLOOR,
        xp.zeros(0, dtype=np.int32),
    )
    existing = xp.zeros(shape, dtype=np.complex128)
    comp.fill_global_stft(params, existing, data_index=indices, factors=factors)
    return to_host(dffr), to_host(existing)


def distances(dffr, existing):
    """Per source: mismatch, phase of (a|b) and norm ratio, with a = DFFR and b = existing."""
    first = dffr.reshape(len(dffr), -1)
    second = existing.reshape(len(existing), -1)
    cross = np.einsum("ij,ij->i", first.conj(), second)
    first_norm = np.einsum("ij,ij->i", first.conj(), first).real
    second_norm = np.einsum("ij,ij->i", second.conj(), second).real
    mismatch = 1.0 - cross.real / np.sqrt(first_norm * second_norm)
    return mismatch, np.angle(cross), first_norm / second_norm


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--output", default=None, metavar="RESULTS_NPZ")
    args = parser.parse_args(argv)

    orbits = load_orbits("esa", args.device)
    band, _ = production_settings(orbits[1] + DATA_START_OFFSET, NUM_SEGMENTS, args.device)
    params, _, rng = production_sources(band, NUM_SOURCES)
    data = draw_cell_arrays(rng, NUM_SEGMENTS, band.NF_active, False)

    results = {"params": params}
    for name, generation, linear_envelope in SETTINGS:
        dffr, existing = templates_of_both(
            args.device, orbits, band, params, data, generation, linear_envelope
        )
        mismatch, phase, norm_ratio = distances(dffr, existing)
        results.update(
            {
                f"{name}__mismatch": mismatch,
                f"{name}__phase": phase,
                f"{name}__norm_ratio": norm_ratio,
            }
        )
        print(f"\n{name}: TDI generation {generation}, linear envelope {linear_envelope}")
        headings = ("source", "f0 [mHz]", "mismatch", "phase [rad]", "norm ratio")
        widths = (6, 10, 11, 12, 11)
        print("  " + " ".join(f"{text:>{width}s}" for text, width in zip(headings, widths)))
        for index in range(len(params)):
            print(
                f"  {index:6d} {1e3 * params[index, 1]:10.5f} {mismatch[index]:11.3e} "
                f"{phase[index]:+12.3e} {norm_ratio[index]:11.6f}"
            )
        print(
            f"  mismatch: median {np.median(mismatch):.3e}, largest {mismatch.max():.3e}; "
            f"largest |phase| {np.abs(phase).max():.3e} rad; norm ratio from "
            f"{norm_ratio.min():.6f} to {norm_ratio.max():.6f}"
        )
    if args.output:
        np.savez(args.output, **results)
        print(f"\n[against existing] wrote {args.output}")


if __name__ == "__main__":
    main()
