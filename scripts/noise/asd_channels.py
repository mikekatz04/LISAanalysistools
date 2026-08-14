#!/usr/bin/env python
"""Measured ASD vs model, decomposed, in X/Y/Z (top) and A/E/T (bottom).

    python scripts/noise/asd_channels.py noise-galfor-pe2/noise_foreground_try4_testing.h5 \
        --unequal-arm --wdm-psd-method layer_calibrated \
        --modulation scripts/noise/modulation_multi.dat \
        --params "notebook:amp=1.18955159143e-44,fk=2.10304500452e-3,alpha=3.39975538551,f_1=2.46392506051e-3,f_2=0.989314627841e-3"

Six panels, one per channel, each with four curves: the data's own ASD, the
total model, and the model split into its instrument and foreground parts (the
split is exact -- the components are additive, so foreground = total minus the
same model with the galfor amplitude at its prior floor).

The point of the bottom row is T. The stationary isotropic foreground
correlation (diag 1, off-diag -1/2) is an exactly singular matrix whose null
direction is (1,1,1)/sqrt(3) = T, and ``modulation_multi.dat`` preserves that
null to machine precision (``sum_ij M_ij`` = 9e-16). So the foreground curve is
identically zero in the T panel -- it has no line to draw -- while the data
sits visibly above the instrument-only model across the foreground band. X, Y
and Z cannot show this: each of them is dominated by the large eigenvalues,
where the same parameters fit well.

ASDs are ``sqrt`` of the one-sided PSD, i.e. of ``2 x`` the WDM pixel variance
(the folded WDM covariance is half the one-sided Fourier PSD -- see
``ppc_noise.py``), time-averaged per layer.

Pass the same data/model flags the run used, as for ``ppc_noise.py``.
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
import whitening_test  # noqa: E402

# Row 1 is the raw TDI channels, row 2 the noise-orthogonal combinations.
PROJECTIONS = [
    ("X", np.array([1.0, 0.0, 0.0])),
    ("Y", np.array([0.0, 1.0, 0.0])),
    ("Z", np.array([0.0, 0.0, 1.0])),
    ("A", np.array([-1.0, 0.0, 1.0]) / np.sqrt(2.0)),
    ("E", np.array([1.0, -2.0, 1.0]) / np.sqrt(6.0)),
    ("T", np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)),
]

C_DATA = "#0b0b0b"
C_TOTAL = "#eb6834"
C_INST = "#2a78d6"
C_FG = "#1a9e77"


def project(C, v):
    """``v^T C v`` over the trailing (layer, time) axes."""
    return np.real(np.einsum("i,ij...,j->...", v, C, v))


def main(argv=None):
    args = parse_args(argv)

    cached = load_cache(args.cache)
    if cached is not None:
        curves, f = cached
        out_path = args.out or f"{os.path.splitext(args.file)[0]}_asd.png"
        print(f"restyling from {args.cache} (no rebuild)")
        plot(curves, f, args.params[0].split(":")[0] if args.params else "posterior median",
             out_path, (args.band_lo, args.band_hi))
        print(f"wrote {out_path}")
        return

    branches = ppc_noise.chain_branches(args.file)
    if "galfor" not in branches:
        raise SystemExit(
            f"{args.file} has no galfor branch; this figure is about the foreground's "
            "contribution, so it needs a foreground run."
        )
    mode = "foreground"
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

    print(f"{args.file}")
    print(f"  grid            {'two-year brick' if args.two_years else 'preset'} "
          f"(nt={ppc_noise.stored_nt(args.file)})")
    print(f"  instrument      {'unequal-arm' if args.unequal_arm else 'equal-arm'}"
          + (f", wdm_psd_method={args.wdm_psd_method}" if args.unequal_arm else ""))
    print(f"  galfor mod.     {args.modulation or 'stationary'}")
    print("\nbuilding the run's data + noise model...", flush=True)

    general_info, source_info = ppc_noise.build_general_and_sources(args, mode, log_sampling)
    settings = general_info.domain_settings
    backend = general_info.sensitivity_backend
    transforms = {b: getattr(s, "transform", None) for b, s in source_info.items()}

    if args.params:
        name, values = whitening_test.parse_param_set(args.params[0])
        psd_p, gal_p = whitening_test.physical_vectors(values, mode)
    else:
        name = "run posterior median"
        psd_p = ppc_noise.to_physical(transforms["psd"], np.median(flats["psd"], axis=0))
        gal_p = ppc_noise.to_physical(transforms["galfor"], np.median(flats["galfor"], axis=0))

    print(f"  parameter set   {name}")
    print("    psd    " + "  ".join(f"{k}={v:.6g}"
                                    for k, v in zip(whitening_test.PSD_BASIS, psd_p)))
    print("    galfor " + "  ".join(f"{k}={v:.6g}"
                                    for k, v in zip(whitening_test.GALFOR_BASIS, gal_p)))

    from lisatools.utils.utility import asnumpy

    total = np.asarray(asnumpy(backend("asd_total", psd_p, galfor_params=gal_p).sens_mat))
    # The components are additive, so dropping the foreground amplitude to its
    # prior floor isolates the instrument exactly -- no separate code path.
    floor = np.array(gal_p, dtype=float)
    floor[0] = 1e-47
    inst = np.asarray(asnumpy(backend("asd_inst", psd_p, galfor_params=floor).sens_mat))
    w = np.asarray(asnumpy(general_info.input_data_residual_array.data_res_arr.arr))

    keep = np.isfinite(total).all(axis=(0, 1)).all(axis=1) & (np.real(total[0, 0]) > 0).all(axis=1)
    f = np.asarray(settings.f_arr)[keep]

    curves = {}
    for label, v in PROJECTIONS:
        tot = project(total, v)[keep].mean(axis=1)
        ins = project(inst, v)[keep].mean(axis=1)
        dat = (np.einsum("i,i...->...", v, w)[keep] ** 2).mean(axis=1)
        curves[label] = dict(data=dat, total=tot, inst=ins, fg=tot - ins)
        frac = (tot - ins) / tot
        print(f"  {label:>2s}  foreground is {100 * np.max(frac):7.3f}% of the model at its "
              f"peak layer;  data/model over {args.band_lo:.0e}-{args.band_hi:.0e} Hz = "
              f"{np.mean((dat / tot)[(f >= args.band_lo) & (f <= args.band_hi)]):.4f}")

    save_cache(args.cache, curves, f)
    plot(curves, f, name, args.out or f"{os.path.splitext(args.file)[0]}_asd.png",
         (args.band_lo, args.band_hi))
    print(f"\nwrote {args.out or f'{os.path.splitext(args.file)[0]}_asd.png'}")


def plot(curves, f, set_name, path, band):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter

    # one-sided PSD = 2 x the WDM pixel variance; ASD = its sqrt
    def asd(p):
        return np.sqrt(np.clip(2.0 * p, 1e-300, None))

    def visible(p, ref):
        """``asd(p)`` with the numerically-dead tail masked off.

        Above ~6 mHz the foreground rolls off past double precision, and
        sqrt of a clipped 1e-300 is 1e-150 -- a real point as far as the
        autoscaler is concerned, which flattens every curve in the panel into
        a line at the top of a 140-decade axis. Masking rather than clipping
        lets the line simply stop where the component stops mattering.
        """
        out = asd(p)
        return np.where(np.asarray(p) > 1e-8 * float(np.max(ref)), out, np.nan)

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.5), sharex=True)
    fmhz = f * 1e3

    for ax, (label, _) in zip(axes.ravel(), PROJECTIONS):
        c = curves[label]
        ax.axvspan(band[0] * 1e3, band[1] * 1e3, color="0.85", alpha=0.35, lw=0, zorder=0)
        ax.plot(fmhz, asd(c["data"]), color=C_DATA, lw=1.6, label="data", zorder=5)
        ax.plot(fmhz, asd(c["total"]), color=C_TOTAL, lw=1.4, label="model total")
        ax.plot(fmhz, visible(c["inst"], c["total"]), color=C_INST, lw=1.1, ls="--",
                label="instrument")
        lo = min(np.nanmin(asd(c["data"])), np.nanmin(asd(c["total"])))
        hi = max(np.nanmax(asd(c["data"])), np.nanmax(asd(c["total"])))
        ax.set_ylim(0.35 * lo, 2.5 * hi)
        fg = c["fg"]
        # In T the foreground is identically zero (the isotropic null), so there
        # is no line to draw -- say so rather than leaving a blank legend entry.
        if np.max(fg) > 1e-12 * np.max(c["total"]):
            ax.plot(fmhz, visible(fg, c["total"]), color=C_FG, lw=1.6, ls="--",
                    label="foreground")
        else:
            ax.plot([], [], color=C_FG, lw=1.1, ls="--", label="foreground $\\equiv 0$")
            ax.annotate("foreground contributes\nNOTHING to T\n(isotropic null)",
                        xy=(0.97, 0.04), xycoords="axes fraction", fontsize=9.5,
                        color=C_FG, va="bottom", ha="right", weight="bold")
        ax.set(xscale="log", yscale="log", title=f"channel {label}")
        ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.grid(alpha=0.25, which="both")
        ax.legend(fontsize=8, loc="upper left")

    for ax in axes[1]:
        ax.set_xlabel("frequency [mHz]")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"ASD [Hz$^{-1/2}$]")

    fig.suptitle(f"foreground ASD over the data at «{set_name}» — "
                 "top: TDI X/Y/Z, bottom: noise-orthogonal A/E/T "
                 "(shaded: the foreground band)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def load_cache(path):
    """``(curves, f)`` from a previous run, or ``None``."""
    if not path or not os.path.exists(path):
        return None
    z = np.load(path)
    curves = {
        label: {k: z[f"{label}_{k}"] for k in ("data", "total", "inst", "fg")}
        for label, _ in PROJECTIONS
    }
    return curves, z["f"]


def save_cache(path, curves, f):
    if not path:
        return
    flat = {"f": f}
    for label, c in curves.items():
        for k, v in c.items():
            flat[f"{label}_{k}"] = v
    np.savez_compressed(path, **flat)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("file", help="foreground run HDF5 -- supplies the grid and the data")
    p.add_argument("--params", action="append", default=[], metavar="NAME:k=v,...",
                   help="the parameter set to draw (default: the run's posterior median)")
    p.add_argument("--discard", type=int, default=0)
    p.add_argument("--thin", type=int, default=1)
    p.add_argument("--band-lo", type=float, default=5e-4)
    p.add_argument("--band-hi", type=float, default=3e-3)
    p.add_argument("--noise-file", default=run_noise_only.NOISE_FILE)
    p.add_argument("--galfor-file", default=run_noise_only.GALFOR_FILE)
    p.add_argument("--modulation", nargs="?", const=run_noise_only.MODULATION_FILE, default=None,
                   metavar="PATH")
    p.add_argument("--unequal-arm", action="store_true")
    p.add_argument("--wdm-psd-method", choices=("fold", "layer_constant", "layer_calibrated"),
                   default="fold")
    p.add_argument("--two-years", action="store_true")
    p.add_argument("--grid", default="auto", choices=("auto", "lite", "full"))
    p.add_argument("--gpus", type=int, nargs="+")
    p.add_argument("--cache", metavar="NPZ",
                   help="read the six channels' curves from this npz if it exists, else "
                   "compute and write it -- a restyle then skips the brick re-pour")
    p.add_argument("--scratch-dir", default="./gf_output_ppc/")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("-o", "--out", help="output png (default: <file>_asd.png)")
    return p.parse_args(argv)


if __name__ == "__main__":
    main()
