#!/usr/bin/env python
"""Foreground model posterior draws over the GALFOR brick ALONE.

    python scripts/noise/galfor_only_asd.py noise-galfor-pe2/noise_foreground_try7_testing.h5 \
        --unequal-arm --wdm-psd-method layer_calibrated \
        --modulation scripts/noise/modulation_unequal.dat \
        --discard 200 --ndraws 40

Unlike ``asd_channels.py``, the data here is the GALFOR brick on its own -- the
NOISE brick is still opened (it carries the orbits, the time grid and the
domain ``t0``) but its Doppler series is NOT summed in. So the black curve is
the confusion foreground realization by itself, with no instrument noise
underneath it, and the only model curve that belongs on top of it is the
``galfor`` branch's additive covariance.

That component is pulled out exactly, not by subtraction:
``backend.component_covariance("galfor", params)`` builds the one additive
branch, so a draw costs a foreground build instead of a full unequal-arm
instrument build. The ``psd`` parameters do not enter at all.

ASDs are ``sqrt`` of the one-sided PSD, i.e. of ``2 x`` the WDM pixel variance,
time-averaged per layer -- the same convention as ``asd_channels.py``.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import corner_noise  # noqa: E402
import ppc_noise  # noqa: E402
import run_noise_only  # noqa: E402

PROJECTIONS = [
    ("X", np.array([1.0, 0.0, 0.0])),
    ("Y", np.array([0.0, 1.0, 0.0])),
    ("Z", np.array([0.0, 0.0, 1.0])),
    ("A", np.array([-1.0, 0.0, 1.0]) / np.sqrt(2.0)),
    ("E", np.array([1.0, -2.0, 1.0]) / np.sqrt(6.0)),
    ("T", np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)),
]

C_DATA = "#0b0b0b"
C_DRAW = "#1a9e77"
C_MED = "#eb6834"


def project(C, v):
    """``v^T C v`` over the trailing (layer, time) axes."""
    return np.real(np.einsum("i,ij...,j->...", v, C, v))


# ---------------------------------------------------------------------------
# the one substantive change to the run's pipeline: drop the NOISE series
# ---------------------------------------------------------------------------


def patch_galfor_only():
    """Make ``NoiseBrickStep`` pour the GALFOR brick and nothing else.

    Everything else about the step is left alone -- times, sampling rate and
    orbits still come from the NOISE file, so the domain, the ``t0`` used to
    look up the link delays and the tabulated modulation, and every
    preprocessing stage (highpass, 200 h edge trim, downsample to 5 s, WDM
    pour) are bit-for-bit what the run used. Only the array being conditioned
    changes.
    """
    from lisatools.globalfit.preprocessing import BaseProcessingStep

    def __init__(self, noise_file, galfor_file=None, orbits_kwargs=None, verbose=True):
        from lisatools.detector import L1Orbits

        if galfor_file is None:
            raise SystemExit("galfor-only data needs a --galfor-file.")
        xyz, times, fs = run_noise_only._read_xyz(noise_file)
        fg, _, _ = run_noise_only._read_xyz(galfor_file)
        if fg.shape != xyz.shape:
            raise ValueError(f"GALFOR {fg.shape} and NOISE {xyz.shape} differ")
        del xyz  # the instrument realization is deliberately discarded

        BaseProcessingStep.__init__(self, times, fg, fs, verbose=verbose)
        self.orbits_class = L1Orbits
        self.orbits = L1Orbits(noise_file, **(orbits_kwargs or {}))
        self.orbits._ensure_configured()

    run_noise_only.NoiseBrickStep.__init__ = __init__


def main(argv=None):
    args = parse_args(argv)
    out_path = args.out or f"{os.path.splitext(args.file)[0]}_galfor_only_asd.png"
    if args.title is None:
        run = os.path.basename(args.file).removesuffix(".h5")
        args.title = (f"{run}: galactic-foreground model over the GALFOR brick alone "
                      "(no instrument noise in the data) — top: TDI X/Y/Z, "
                      "bottom: A/E/T (shaded: the foreground band)")

    cached = load_cache(args.cache)
    if cached is not None:
        print(f"restyling from {args.cache} (no rebuild)")
        plot(*cached, out_path, (args.band_lo, args.band_hi), args.title)
        print(f"wrote {out_path}")
        return

    branches = ppc_noise.chain_branches(args.file)
    if "galfor" not in branches:
        raise SystemExit(f"{args.file} has no galfor branch.")
    branches = ["psd", "galfor"]

    inferred_full, inferred_two_years = ppc_noise.resolve_grid(
        args.file, args.grid, args.noise_file
    )
    args.full = inferred_full if args.grid == "auto" else args.grid == "full"
    args.two_years = args.two_years or inferred_two_years

    flats = {b: corner_noise.load_samples(args.file, b, mask=False,
                                          discard=args.discard, thin=args.thin)
             for b in branches}
    log_sampling = {b: corner_noise.resolve_basis(flats[b]) == "log" for b in branches}

    rng = np.random.default_rng(args.seed)
    nsamples = flats["galfor"].shape[0]
    n = min(args.ndraws, nsamples)
    picks = rng.choice(nsamples, size=n, replace=False)

    print(f"{args.file}")
    print(f"  grid            {'two-year brick' if args.two_years else 'preset'} "
          f"(nt={ppc_noise.stored_nt(args.file)})")
    print(f"  galfor mod.     {args.modulation or 'stationary'}")
    print(f"  draws           {n} of {nsamples} "
          f"(discard={args.discard}, thin={args.thin}, seed={args.seed})")
    print("\nbuilding the run's model on GALFOR-ONLY data...", flush=True)

    patch_galfor_only()
    general_info, source_info = ppc_noise.build_general_and_sources(
        args, "foreground", log_sampling
    )
    settings = general_info.domain_settings
    backend = general_info.sensitivity_backend
    transform = getattr(source_info["galfor"], "transform", None)

    from lisatools.utils.utility import asnumpy

    gal_rows = np.asarray([ppc_noise.to_physical(transform, r)
                           for r in flats["galfor"][picks]])
    gal_med = ppc_noise.to_physical(transform, np.median(flats["galfor"], axis=0))

    import whitening_test
    print("  posterior median galfor  "
          + "  ".join(f"{k}={v:.6g}"
                      for k, v in zip(whitening_test.GALFOR_BASIS, gal_med)))

    w = np.asarray(asnumpy(general_info.input_data_residual_array.data_res_arr.arr))

    import time as _time
    t0 = _time.time()
    med_C = np.asarray(asnumpy(backend.component_covariance("galfor", gal_med)))
    print(f"  one foreground build: {_time.time() - t0:.1f} s, shape {med_C.shape}")

    keep = np.isfinite(med_C).all(axis=(0, 1)).all(axis=1) & (
        np.real(med_C[0, 0]) > 0).any(axis=1)
    f = np.asarray(asnumpy(settings.f_arr))[keep]

    curves = {label: dict(
        data=(np.einsum("i,i...->...", v, w)[keep] ** 2).mean(axis=1),
        median=project(med_C, v)[keep].mean(axis=1),
        draws=np.empty((n, keep.sum())),
    ) for label, v in PROJECTIONS}
    del med_C

    for k, row in enumerate(gal_rows):
        C = np.asarray(asnumpy(backend.component_covariance("galfor", row)))
        for label, v in PROJECTIONS:
            curves[label]["draws"][k] = project(C, v)[keep].mean(axis=1)
        del C
        print(f"\r  draw {k + 1}/{n}", end="", flush=True)
    print()

    band = (f >= args.band_lo) & (f <= args.band_hi)
    print(f"\n  data/model over {args.band_lo:.0e}-{args.band_hi:.0e} Hz "
          "(ratio of band-mean PSD, median params):")
    for label, _ in PROJECTIONS:
        c = curves[label]
        r = np.mean(c["data"][band]) / np.mean(c["median"][band])
        lo, hi = np.percentile(
            [np.mean(d[band]) for d in c["draws"]], [5, 95])
        print(f"    {label:>2s}  {r:8.4f}   model band-mean PSD "
              f"5-95%: {lo:.3e} .. {hi:.3e}")

    save_cache(args.cache, curves, f, gal_rows)
    plot(curves, f, out_path, (args.band_lo, args.band_hi), args.title)
    print(f"\nwrote {out_path}")


def plot(curves, f, path, band, title):
    """Six channels, each an ASD panel over a data/model ratio strip.

    The strip is not decoration. The foreground posterior is ~0.5% wide, so on
    a log ASD axis spanning four decades every draw lands under the median line
    and the figure would claim a spread it cannot show. The ratio strip is
    scaled to the in-band draws, which is where that half-percent is legible.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter

    def asd(p):
        return np.sqrt(np.clip(2.0 * np.asarray(p), 1e-300, None))

    ndraws = len(next(iter(curves.values()))["draws"])
    fmhz = f * 1e3

    fig = plt.figure(figsize=(16.5, 10.0))
    outer = GridSpec(2, 3, figure=fig, hspace=0.30, wspace=0.22,
                     left=0.075, right=0.985, top=0.90, bottom=0.07)

    for k, (label, _) in enumerate(PROJECTIONS):
        c = curves[label]
        cell = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[k // 3, k % 3],
                                       height_ratios=(3.0, 1.15), hspace=0.06)
        ax = fig.add_subplot(cell[0])
        rx = fig.add_subplot(cell[1], sharex=ax)

        for a in (ax, rx):
            a.axvspan(band[0] * 1e3, band[1] * 1e3, color="0.86", alpha=0.5,
                      lw=0, zorder=0)

        for d in c["draws"]:
            ax.plot(fmhz, asd(d), color=C_DRAW, lw=1.0, alpha=0.16, zorder=2)
            rx.plot(fmhz, c["data"] / d, color=C_DRAW, lw=1.0, alpha=0.16, zorder=2)
        ax.plot([], [], color=C_DRAW, lw=1.3, alpha=0.55,
                label=f"posterior draws ($n$={ndraws})")
        ax.plot(fmhz, asd(c["median"]), color=C_MED, lw=1.5, zorder=4,
                label="posterior median")
        rx.plot(fmhz, c["data"] / c["median"], color=C_MED, lw=1.3, zorder=4)
        ax.plot(fmhz, asd(c["data"]), color=C_DATA, lw=1.5, zorder=5,
                label="GALFOR brick")
        rx.axhline(1.0, color=C_DATA, lw=1.0, ls=":", zorder=3)

        dat = asd(c["data"])
        # Let the model's roll-off dive visibly out of the panel rather than
        # spending four decades of axis on it.
        ax.set_ylim(dat.min() / 40.0, dat.max() * 3.0)
        ax.set(xscale="log", yscale="log", title=f"channel {label}")
        ax.tick_params(labelbottom=False)
        ax.grid(alpha=0.25, which="both")
        ax.legend(fontsize=8, loc="lower left", framealpha=0.9)

        # Scale the strip on the shaded foreground band only. Past its tanh knee
        # the model dives through the data's residual floor and the ratio runs
        # to ~150; scaling on that would flatten the half-percent the strip
        # exists to show, so the roll-off is allowed to leave the strip instead.
        in_band = (f >= band[0]) & (f <= band[1])
        r = np.concatenate([c["data"][in_band] / d[in_band] for d in c["draws"]])
        lo, hi = np.percentile(r, [0.5, 99.5])
        pad = max(0.004, 0.30 * (hi - lo))
        rx.set_ylim(lo - pad, hi + pad)
        rx.set(xscale="log")
        rx.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
        rx.xaxis.set_major_formatter(ScalarFormatter())
        rx.xaxis.set_minor_formatter(NullFormatter())
        rx.grid(alpha=0.25, which="both")
        rx.set_ylabel("data/model", fontsize=8)
        rx.tick_params(labelsize=8)
        if k // 3 == 1:
            rx.set_xlabel("frequency [mHz]")
        if k % 3 == 0:
            ax.set_ylabel(r"ASD [Hz$^{-1/2}$]")

    fig.suptitle(title, fontsize=11)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def load_cache(path):
    if not path or not os.path.exists(path):
        return None
    z = np.load(path)
    curves = {label: {k: z[f"{label}_{k}"] for k in ("data", "median", "draws")}
              for label, _ in PROJECTIONS}
    return curves, z["f"]


def save_cache(path, curves, f, gal_rows):
    if not path:
        return
    flat = {"f": f, "gal_rows": gal_rows}
    for label, c in curves.items():
        for k, v in c.items():
            flat[f"{label}_{k}"] = v
    np.savez_compressed(path, **flat)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("file", help="foreground run HDF5 -- supplies the grid and the chain")
    p.add_argument("--ndraws", type=int, default=40)
    p.add_argument("--seed", type=int, default=20260818)
    p.add_argument("--discard", type=int, default=0)
    p.add_argument("--thin", type=int, default=1)
    p.add_argument("--band-lo", type=float, default=5e-4)
    p.add_argument("--band-hi", type=float, default=3e-3)
    p.add_argument("--noise-file", default=run_noise_only.NOISE_FILE)
    p.add_argument("--galfor-file", default=run_noise_only.GALFOR_FILE)
    p.add_argument("--modulation", nargs="?", const=run_noise_only.MODULATION_FILE,
                   default=None, metavar="PATH")
    p.add_argument("--unequal-arm", action="store_true")
    p.add_argument("--wdm-psd-method", choices=("fold", "layer_constant", "layer_calibrated"),
                   default="fold")
    p.add_argument("--two-years", action="store_true")
    p.add_argument("--grid", default="auto", choices=("auto", "lite", "full"))
    p.add_argument("--gpus", type=int, nargs="+")
    p.add_argument("--cache", metavar="NPZ",
                   help="read/write the six channels' curves here; a restyle then "
                   "skips the brick re-pour")
    p.add_argument("--scratch-dir", default="./gf_output_ppc/")
    p.add_argument("--title", default=None)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("-o", "--out", help="output png")
    return p.parse_args(argv)


if __name__ == "__main__":
    main()
