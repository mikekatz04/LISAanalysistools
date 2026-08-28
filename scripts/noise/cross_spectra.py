#!/usr/bin/env python
"""Measured vs modelled channel CORRELATIONS, per WDM layer and per time column.

    python scripts/noise/cross_spectra.py noise-galfor-pe2/noise_foreground_try4_testing.h5 \
        --discard 100 --unequal-arm --wdm-psd-method layer_calibrated \
        --modulation scripts/noise/modulation_multi.dat

``ppc_noise.py`` and ``whitening_test.py``'s ratio panel both read only
``C[i, i]``. The likelihood does not: it inverts the whole 3x3 pixel
covariance, so the XY / XZ / YZ cross-spectra carry real weight, and on the
two-year foreground fit they are what the posterior is actually being steered
by -- the Fourier-domain least-squares parameters whiten every diagonal
(chi2/pix 1.0002) and still lose 4205 in full-covariance log-likelihood.

This plots the quantity responsible, as a dimensionless correlation so the
amplitude drops out:

    rho_ij(m) = sum_n w_i w_j / sqrt(sum_n w_i^2 * sum_n w_j^2)      (data)
    rho_ij(m) = sum_n C_ij   / sqrt(sum_n C_ii   * sum_n C_jj  )      (model)

Same construction on both sides, so they are comparable term by term. Two
reductions, mirroring ppc_noise's:

    frequency  average over time columns -> rho vs f. The foreground's
               stationary isotropic limit fixes rho = -1/2 for its own
               contribution at every frequency; the instrument's unequal-arm
               cross-spectrum supplies the rest. A galaxy is not isotropic, so
               this is the panel where that assumption is tested directly.
    time       average over the foreground-dominated layers -> rho vs t. This
               one tests the off-diagonal COLUMNS of the modulation table
               (XY, XZ, YZ), which nothing else in the noise tooling looks at.

The estimator band is closed form: a sample correlation over ``N`` independent
pixels has sd ``(1 - rho^2) / sqrt(N)``, which at Nt = 16060 is ~0.006 -- so
this comparison is sharp, and a 0.02 offset is already many sigma.

Pass the same data/model flags the run used, exactly as for ``ppc_noise.py``.
``--params`` adds named parameter sets alongside the run's posterior median
(same syntax as ``whitening_test.py``); ``--no-foreground`` adds a set with the
galfor amplitude at its prior floor, which shows how much of rho the
instrument alone accounts for.
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

PAIRS = ((0, 1, "XY"), (0, 2, "XZ"), (1, 2, "YZ"))

# A, E, T as row vectors over (X, Y, Z). T = (1,1,1)/sqrt(3) is the direction
# the stationary isotropic foreground correlation (diag 1, off-diag -1/2)
# annihilates exactly -- that matrix has eigenvalues (0, 3/2, 3/2). So the
# foreground puts NO power in T, the model's C_TT there is the instrument
# alone, and the likelihood's T term ``w_T^2 / C_TT`` is divided by a small
# number. That is why a parameter set can whiten X, Y and Z and still be
# rejected by the full 3x3: the diagonals and the pairwise correlations are
# both blind to the near-null direction that dominates C^-1.
AET = {
    "A": np.array([-1.0, 0.0, 1.0]) / np.sqrt(2.0),
    "E": np.array([1.0, -2.0, 1.0]) / np.sqrt(6.0),
    "T": np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0),
}
C_SETS = ["#2a78d6", "#eb6834", "#4a3aa7", "#1a9e77"]
C_DATA = "#0b0b0b"
C_INK = "0.35"


def layer_rho(a, b, aa, bb):
    """``sum_n a*b / sqrt(sum_n aa * sum_n bb)`` per layer, for data or model alike."""
    return np.sum(a * b, axis=1) / np.sqrt(np.sum(aa, axis=1) * np.sum(bb, axis=1))


def column_rho(a, b, aa, bb, smooth):
    """The same correlation reduced over LAYERS instead, one value per column."""
    num = ppc_noise.boxcar(np.sum(a * b, axis=0), smooth)
    den = np.sqrt(
        ppc_noise.boxcar(np.sum(aa, axis=0), smooth)
        * ppc_noise.boxcar(np.sum(bb, axis=0), smooth)
    )
    return num / den


def rho_sigma(rho, n):
    """1-sigma of a sample correlation over ``n`` independent pixels."""
    return (1.0 - rho**2) / np.sqrt(max(n, 1))


def collect(general_info, backend, sets, band, smooth):
    """Measured and modelled correlations for every pair and parameter set."""
    from lisatools.utils.utility import asnumpy

    w = np.asarray(asnumpy(general_info.input_data_residual_array.data_res_arr.arr))

    models = {}
    for name, psd_p, gal_p in sets:
        kwargs = {} if gal_p is None else dict(galfor_params=gal_p)
        sens = backend(f"xspec_{name.replace(' ', '_')}", psd_p, **kwargs)
        models[name] = np.real(np.asarray(asnumpy(sens.sens_mat)))

    ref = next(iter(models.values()))
    # Per-LAYER, not per-pixel: both reductions sum a whole layer (or a whole
    # column over layers), so a layer is either usable throughout or dropped.
    # ``ref`` is the full (nch, nch, Nf, Nt) covariance here, so the channel
    # axes collapse first and the time axis second.
    keep = np.isfinite(ref).all(axis=(0, 1)).all(axis=1) & (ref[0, 0] > 0).all(axis=1)
    # The time panel must not average a layer the frequency panel drops as
    # degenerate, or the two rows describe different pixel sets.
    band = band & keep
    nt = w.shape[2]

    aet = {}
    for name, C in models.items():
        for label, v in AET.items():
            proj = np.einsum("i,ijmn,j->mn", v, C, v)[keep]
            wv = np.einsum("i,imn->mn", v, w)[keep]
            entry = aet.setdefault(label, dict(model={}, data=None, sigma={}))
            entry["data"] = np.sum(wv**2, axis=1)
            entry["model"][name] = np.sum(proj, axis=1)
            # w_v^2 is C_vv times a chi^2_1 per pixel, so the time-average of
            # the ratio has the same closed-form scatter as ppc_noise's panel B.
            entry["sigma"][name] = ppc_noise.freq_sigma(proj)

    out = {}
    for i, j, label in PAIRS:
        entry = dict(
            data_freq=layer_rho(w[i][keep], w[j][keep], w[i][keep] ** 2, w[j][keep] ** 2),
            data_time=column_rho(
                w[i][band], w[j][band], w[i][band] ** 2, w[j][band] ** 2, smooth
            ),
            model_freq={},
            model_time={},
            sigma_freq=None,
            sigma_time=None,
        )
        for name, C in models.items():
            entry["model_freq"][name] = layer_rho(
                C[i, j][keep], np.ones_like(C[i, j][keep]), C[i, i][keep], C[j, j][keep]
            )
            entry["model_time"][name] = column_rho(
                C[i, j][band], np.ones_like(C[i, j][band]), C[i, i][band], C[j, j][band], smooth
            )
        first = next(iter(models))
        entry["sigma_freq"] = rho_sigma(entry["model_freq"][first], nt)
        entry["sigma_time"] = rho_sigma(entry["model_time"][first], int(band.sum()) * smooth)
        out[label] = entry
    return out, aet, keep, band


def plot(out, aet, f_arr, t_arr, keep, title, path):
    """rho vs frequency / rho vs time per pair, then A/E/T variance ratios."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter

    labels = list(out)
    fig, axes = plt.subplots(3, len(labels), figsize=(5.6 * len(labels), 12), squeeze=False)
    f = np.asarray(f_arr)[keep] * 1e3
    t = np.asarray(t_arr) / 86400.0

    for j, label in enumerate(labels):
        e = out[label]

        ax = axes[0][j]
        ax.plot(f, e["data_freq"], color=C_DATA, lw=1.4, label="data", zorder=5)
        for k, (name, model) in enumerate(e["model_freq"].items()):
            color = C_SETS[k % len(C_SETS)]
            ax.plot(f, model, color=color, lw=1.2, label=name)
            ax.fill_between(f, model - 1.645 * e["sigma_freq"], model + 1.645 * e["sigma_freq"],
                            color=color, alpha=0.15, lw=0)
        ax.axhline(-0.5, color=C_INK, ls=":", lw=0.9)
        ax.annotate("isotropic -1/2", (f[0], -0.5), fontsize=7, color=C_INK,
                    va="bottom", ha="left")
        ax.set(xscale="log", xlabel="frequency [mHz]",
               ylabel=rf"$\rho_{{{label}}}$, time-averaged", title=f"pair {label}")
        ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)

        ax = axes[1][j]
        ax.plot(t, e["data_time"], color=C_DATA, lw=0.8, label="data", zorder=5)
        for k, (name, model) in enumerate(e["model_time"].items()):
            color = C_SETS[k % len(C_SETS)]
            ax.plot(t, model, color=color, lw=1.2, label=name)
            ax.fill_between(t, model - 1.645 * e["sigma_time"], model + 1.645 * e["sigma_time"],
                            color=color, alpha=0.15, lw=0)
        ax.set(xlabel="time from data start [days]",
               ylabel=rf"$\rho_{{{label}}}$ over the foreground band")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)

        # Row 3 re-uses the column slot for the A/E/T channel of the same
        # index -- pairs and AET channels are different things, so each axis
        # says which it is.
        chan = ("A", "E", "T")[j]
        e = aet[chan]
        ax = axes[2][j]
        for k, (name, model) in enumerate(e["model"].items()):
            color = C_SETS[k % len(C_SETS)]
            ax.plot(f, e["data"] / model, color=color, lw=1.2, label=name)
            ax.fill_between(f, 1 - 1.645 * e["sigma"][name], 1 + 1.645 * e["sigma"][name],
                            color=color, alpha=0.15, lw=0)
        ax.axhline(1.0, color=C_INK, ls="--", lw=0.9)
        ax.set(xscale="log", xlabel="frequency [mHz]",
               ylabel=f"data / model variance, {chan}",
               title=f"channel {chan}" + (" — foreground null" if chan == "T" else ""))
        ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("file", help="run HDF5 -- supplies the grid, the data and the mode")
    p.add_argument("--params", action="append", default=[], metavar="NAME:k=v,...",
                   help="a named parameter set (repeatable), as in whitening_test.py")
    p.add_argument("--no-foreground", action="store_true",
                   help="add a set with the galfor amplitude at its prior floor, i.e. the "
                   "instrument's own correlation with no foreground at all")
    p.add_argument("--discard", type=int, default=0)
    p.add_argument("--thin", type=int, default=1)
    p.add_argument("--band", default="5e-4,3e-3", metavar="LO,HI",
                   help="frequency band [Hz] the TIME panel averages over -- the layers "
                   "the foreground dominates (default 5e-4,3e-3)")
    p.add_argument("--time-smooth", type=int, default=24,
                   help="running-mean width in wavelet columns for the time panel")
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
    p.add_argument("--scratch-dir", default="./gf_output_ppc/")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("-o", "--out", help="output png (default: <file>_xspec.png)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    branches = ppc_noise.chain_branches(args.file)
    if "psd" not in branches:
        raise SystemExit(f"no psd branch in {args.file}; found {branches}")
    mode = "foreground" if "galfor" in branches else "instrument"
    branches = ["psd"] + (["galfor"] if mode == "foreground" else [])

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
    print(f"  mode            {mode}")
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

    med = {b: np.median(flats[b], axis=0) for b in branches}
    psd_med = ppc_noise.to_physical(transforms["psd"], med["psd"])
    gal_med = (None if mode == "instrument"
               else ppc_noise.to_physical(transforms["galfor"], med["galfor"]))
    sets = [("run posterior median", psd_med, gal_med)]
    for spec in args.params:
        name, values = whitening_test.parse_param_set(spec)
        sets.append((name,) + whitening_test.physical_vectors(values, mode))
    if args.no_foreground and gal_med is not None:
        floor = np.array(gal_med, dtype=float)
        floor[0] = 1e-47  # amp at the prior floor: the foreground switched off
        sets.append(("instrument only", psd_med, floor))

    lo, hi = (float(v) for v in args.band.split(","))
    f_arr = np.asarray(settings.f_arr)
    band = (f_arr >= lo) & (f_arr <= hi)
    if not band.any():
        raise SystemExit(f"--band {args.band} contains no active layer of this grid")

    print(f"grid {settings.Nf_active} layers x {settings.Nt_active} columns; time panel "
          f"averages the {int(band.sum())} layers in {lo:.1e}-{hi:.1e} Hz")
    print(f"reducing {len(sets)} parameter set(s)...", flush=True)

    out, aet, keep, band = collect(general_info, backend, sets, band, args.time_smooth)

    fband = band[keep]
    print(f"\nmean (data - model) / sigma, over the {int(fband.sum())} "
          f"foreground-band layers and over all {int(keep.sum())} layers")
    print(f"  {'set':24s} {'pair':>5s} {'fg band':>10s} {'all layers':>12s} {'max |z|':>9s}")
    for name in out["XY"]["model_freq"]:
        for label in out:
            e = out[label]
            z = (e["data_freq"] - e["model_freq"][name]) / e["sigma_freq"]
            print(f"  {name[:22]:22s} {label:>7s} {np.mean(z[fband]):10.1f} "
                  f"{np.mean(z):12.1f} {np.max(np.abs(z)):9.1f}")

    fband_a = band[keep]
    print("\nA/E/T variance, data vs model: mean (data/model - 1) / sigma per layer")
    print(f"  {'set':24s} {'chan':>5s} {'fg band':>10s} {'all layers':>12s} {'ratio@fg':>10s}")
    for name in aet["T"]["model"]:
        for label in ("A", "E", "T"):
            e = aet[label]
            ratio = e["data"] / e["model"][name]
            z = (ratio - 1.0) / e["sigma"][name]
            print(f"  {name[:22]:22s} {label:>7s} {np.mean(z[fband_a]):10.1f} "
                  f"{np.mean(z):12.1f} {np.mean(ratio[fband_a]):10.4f}")
    print("  T is the direction the isotropic foreground annihilates -- the "
          "likelihood weights it hardest")

    out_path = args.out or f"{os.path.splitext(args.file)[0]}_xspec.png"
    plot(out, aet, settings.f_arr, settings.t_arr, keep,
         f"{os.path.basename(args.file)} — channel correlations, data vs model", out_path)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
