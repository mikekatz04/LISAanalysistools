#!/usr/bin/env python
"""Data scalogram + posterior evolutionary PSD on a few time slices.

    python scripts/noise/ppc_scalogram.py noise-galfor-pe/noise_foreground_full5_testing.h5 \
        --unequal-arm --modulation scripts/noise/modulation_multi.dat

The WDM likelihood is exactly ``w ~ N(0, C)``: the wavelet coefficient of the
data in layer ``m``, time column ``n`` is a zero-mean Gaussian whose variance
``C[m, n]`` is what the noise run samples. So the posterior predictive check is
the direct one -- draw ``C`` from the posterior and look at it over the data's
own ``w**2``.

This is the *un-reduced* version of that check (``ppc_noise.py`` averages the
time axis away). Two things are drawn:

    top     the scalogram of the total data (NOISE + GALFOR) -- ``w**2`` over
            the full (frequency, time) plane, log color. ``--time-bin`` averages
            adjacent columns for DISPLAY only (default 8, ~8.5 h): at one
            column per pixel every cell is an independent chi^2_1 draw and the
            image is pure speckle, with the frequency trend the only thing
            legible through it. Nothing else in the figure is binned.

    bottom  four equally spaced time slices. Each is a single column of that
            plane -- the raw wavelet coefficients at one instant, in the
            background -- with ``--ndraws`` posterior draws of ``C`` for that
            same column laid over it at low opacity.

A single pixel is a 1-dof variance estimate: ``w**2 / C`` is chi^2_1, whose
median is 0.45 and whose lower tail runs decades below the mean. The slice
panels therefore show a broad cloud that sits mostly *under* the model bundle
even when the model is perfect -- that is the shape of chi^2_1, not a bias. The
median line drawn per panel is the statistic to read: it should track
``0.4549 * C``, and that reference is drawn too.

``--slice-width`` averages both the data and the model over a window of columns
centred on each slice, trading time resolution for a tighter cloud (width ``k``
makes each point chi^2_k / k). The window is applied identically to both, so
the comparison stays exact.

``--cache`` writes (or reads back) everything the figure needs as an npz, so
restyling it does not re-pour the bricks.

The model has to be built the way the RUN built it, so this shares
``ppc_noise``'s builder and needs the same data/model flags the run used
(``--modulation``, ``--unequal-arm``, ``--wdm-psd-method``, the brick paths).
Mode, sampling basis and the wavelet grid -- lite, full, or ``--two-years``,
the last read off the run's stored ``Nt`` -- are inferred from the chain, as
there.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import corner_noise  # noqa: E402  (sibling script; chain loading + basis detection)
import ppc_noise  # noqa: E402  (sibling script; the run rebuild + draw loading)
import run_noise_only  # noqa: E402  (sibling script; brick paths)

# E[w_mn^2] == S_wdm[m] == S_n(f_m) / 2 -- see ppc_noise.py's docstring.
WDM_TO_PSD = ppc_noise.WDM_TO_PSD

C_DATA = ppc_noise.C_DATA  # blue   -- the measured coefficients, everywhere
C_MODEL = ppc_noise.C_MODEL  # orange -- posterior draws of the model
C_INK = ppc_noise.C_INK  # neutral ink: reference lines, never a series

# Sequential = ONE hue, light -> dark (the blue ramp, steps 100..700). The
# scalogram is the data, so it wears the data's hue; the model bundle over it
# is the only other thing on the figure and stays orange.
BLUE_RAMP = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]


def slice_columns(nt, nslices, width):
    """Centres of ``nslices`` equal chunks of ``nt`` columns -> list of slices.

    Centres, not endpoints: the first and last wavelet columns carry the
    transform's edge effect, and a slice sitting on one would show a defect of
    the WDM basis rather than of the noise model.
    """
    edges = np.linspace(0, nt, nslices + 1)
    half = width // 2
    out = []
    for i in range(nslices):
        c = int(0.5 * (edges[i] + edges[i + 1]))
        lo = max(0, c - half)
        hi = min(nt, lo + width)
        lo = max(0, hi - width)  # keep the width if the centre ran into an edge
        out.append((c, slice(lo, hi)))
    return out


def collect(general_info, source_info, rows, ch, ref_params, cols, verbose):
    """Data plane + per-slice posterior draws of the model.

    Only the slice columns of each draw are kept: one ``C`` is (Nf, Nt) and
    ``ndraws`` of them held at once would be pointless when four columns of
    each is all the figure uses.
    """
    from lisatools.utils.utility import asnumpy

    backend = general_info.sensitivity_backend
    transforms = {b: getattr(setup, "transform", None) for b, setup in source_info.items()}
    branches = list(rows)

    def build(name, row_of):
        kwargs = {}
        if "galfor" in branches:
            kwargs["galfor_params"] = ppc_noise.to_physical(
                transforms["galfor"], row_of("galfor")
            )
        return backend(name, ppc_noise.to_physical(transforms["psd"], row_of("psd")), **kwargs)

    ref = ppc_noise.covariance_channel(build("scal_ref", lambda b: ref_params[b]), ch)
    w = asnumpy(general_info.input_data_residual_array.data_res_arr.arr[ch])

    # A zeroed / non-finite layer is a fold artifact (``instrument_fill_nans``
    # zeroes the f=0 divergence), not data. Drop it from the data and the model
    # alike so every panel is the same pixels.
    keep = np.isfinite(ref).all(axis=1) & (ref > 0).all(axis=1)
    if not keep.any():
        raise SystemExit(f"channel {ch}: reference model is degenerate everywhere.")
    w, ref = w[keep], ref[keep]

    ndraws = len(next(iter(rows.values())))
    draws = np.empty((ndraws, len(cols), int(keep.sum())))
    for i in range(ndraws):
        if verbose and (i % 25 == 0):
            print(f"  draw {i}/{ndraws}", flush=True)
        C = ppc_noise.covariance_channel(build(f"scal_{i}", lambda b, i=i: rows[b][i]), ch)[keep]
        for j, (_, sl) in enumerate(cols):
            draws[i, j] = C[:, sl].mean(axis=1)

    data = np.stack([(w[:, sl] ** 2).mean(axis=1) for _, sl in cols])
    return dict(keep=keep, w2=w**2, ref=ref, data_slices=data, model_slices=draws)


def bin_time(plane, t_days, n):
    """Average ``n`` adjacent time columns of ``(Nf, Nt)`` -> ``(plane, centres)``.

    Display only, and only for the scalogram: at one column per screen pixel
    every cell is an independent chi^2_1 draw, so the image is speckle and the
    frequency trend is the only structure that survives it. A trailing partial
    block is dropped rather than averaged over fewer columns -- it would be
    noisier than its neighbours and read as a real edge feature.
    """
    if n <= 1:
        return plane, t_days
    nb = plane.shape[1] // n
    if nb == 0:
        return plane, t_days
    cut = nb * n
    return (
        plane[:, :cut].reshape(plane.shape[0], nb, n).mean(axis=2),
        t_days[:cut].reshape(nb, n).mean(axis=1),
    )


# chi^2_k / k median, k = the number of columns averaged into one slice point.
def chi2_median(k):
    from scipy.stats import chi2

    return chi2.ppf(0.5, k) / k


def plot(out, f_mhz, t_days, cols, width, time_bin, scale, unit_label, title, path):
    """Scalogram across the top, the four slice panels in a row beneath it."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    from matplotlib.lines import Line2D
    from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter

    cmap = LinearSegmentedColormap.from_list("lat_blue", BLUE_RAMP)
    ndraws = out["model_slices"].shape[0]
    nsl = len(cols)
    # Overlaid curves should read as a density, not as one legible line -- but
    # this posterior is tight enough that all ndraws land on top of each other,
    # so a 1/ndraws opacity makes the whole bundle invisible rather than faint.
    # The floor is what keeps the superposition visible.
    alpha = float(np.clip(18.0 / max(ndraws, 1), 0.18, 0.4))
    halo = [pe.withStroke(linewidth=2.2, foreground="white")]

    fig = plt.figure(figsize=(4.0 * nsl, 9.4))
    grid = fig.add_gridspec(2, nsl, height_ratios=[1.25, 1.0], hspace=0.26, wspace=0.08, top=0.93)

    # -- the scalogram ------------------------------------------------------
    ax = fig.add_subplot(grid[0, :])
    plane, t_plane = bin_time(out["w2"] * scale, t_days, time_bin)
    # Log color over a chi^2_k/k quantity: the low tail still runs well under
    # the mean, so the ends of the ramp are percentiles of the plane rather
    # than its extrema, which would spend most of the ramp on a few pixels.
    vmin = np.percentile(plane[plane > 0], 1.0)
    vmax = np.percentile(plane, 99.0)
    mesh = ax.pcolormesh(
        t_plane,
        f_mhz,
        np.clip(plane, vmin, vmax),
        cmap=cmap,
        norm=LogNorm(vmin=vmin, vmax=vmax),
        shading="nearest",
        rasterized=True,
    )
    cb = fig.colorbar(mesh, ax=ax, pad=0.012, aspect=28)
    cb.set_label(unit_label, fontsize=9)
    cb.ax.tick_params(labelsize=8)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylabel("frequency [mHz]", fontsize=9)
    ax.set_xlabel("time from data start [days]", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_title(
        "data scalogram — NOISE + GALFOR, WDM pixel power"
        + (f"  ({time_bin} columns averaged per pixel)" if time_bin > 1 else ""),
        fontsize=10,
    )

    # Inside the axes, not above them: the slice markers would otherwise sit in
    # the title's line.
    for j, (c, _) in enumerate(cols):
        ax.axvline(t_days[c], color="white", lw=1.6, alpha=0.9)
        ax.axvline(t_days[c], color=C_INK, lw=1.0, ls="--", alpha=0.95)
        ax.annotate(
            f"{j + 1}",
            xy=(t_days[c], 0.965),
            xycoords=("data", "axes fraction"),
            ha="center",
            va="top",
            fontsize=8.5,
            color=C_INK,
            path_effects=halo,
        )

    # -- the slice panels ---------------------------------------------------
    med_ratio = chi2_median(width)
    # In the PLOTTED units: the panels draw ``* scale``, so limits taken off the
    # raw arrays would sit a factor ``scale`` low and clip the top of the data.
    hi = scale * max(out["model_slices"].max(), out["data_slices"].max())
    # A decade and a half under the model floor holds the bulk of the chi^2
    # lower tail. Going to the data minimum instead would let a single
    # near-null pixel -- which chi^2_1 produces in every slice -- add three
    # empty decades and flatten everything the panel is for. Whatever falls
    # below is counted and reported in the panel rather than silently cropped.
    floor = scale * out["model_slices"].min() * 10.0**-1.25

    axes = []
    for j, (c, sl) in enumerate(cols):
        axj = fig.add_subplot(grid[1, j], sharey=axes[0] if axes else None)
        axes.append(axj)
        d = out["data_slices"][j] * scale
        m = out["model_slices"][j] * scale

        axj.plot(f_mhz, m.T, color=C_MODEL, lw=1.1, alpha=alpha, zorder=3)
        axj.plot(f_mhz, d, color=C_DATA, lw=0.0, marker="o", ms=3.4, alpha=0.8, zorder=2)
        # The chi^2 median of the median-posterior model: where the cloud
        # should sit, as opposed to where E[w^2] = C sits.
        axj.plot(
            f_mhz,
            np.median(m, axis=0) * med_ratio,
            color=C_INK,
            lw=1.1,
            ls="--",
            zorder=4,
            path_effects=halo,
        )

        below = int(np.sum(d < floor))
        axj.set_xscale("log")
        axj.set_yscale("log")
        axj.set_ylim(floor * 0.8, hi * 1.8)
        axj.set_xlabel("frequency [mHz]", fontsize=9)
        axj.set_title(
            f"{j + 1}   t = {t_days[c]:.1f} d" + (f"   ({width} cols)" if width > 1 else ""),
            fontsize=9,
        )
        axj.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
        axj.xaxis.set_major_formatter(ScalarFormatter())
        axj.xaxis.set_minor_formatter(NullFormatter())
        axj.grid(alpha=0.25, lw=0.5, which="both")
        axj.spines["top"].set_visible(False)
        axj.spines["right"].set_visible(False)
        axj.tick_params(labelsize=8)
        if j:
            axj.tick_params(labelleft=False)
        else:
            axj.set_ylabel(unit_label, fontsize=9)
        if below:
            axj.annotate(
                f"{below} below axis",
                xy=(0.03, 0.03),
                xycoords="axes fraction",
                fontsize=7,
                color=C_INK,
            )

    # Identity is never opacity-alone: the proxies are opaque.
    axes[0].legend(
        handles=[
            Line2D([], [], color=C_DATA, lw=0, marker="o", ms=4, label="data $w^2$"),
            Line2D([], [], color=C_MODEL, lw=1.6, label=f"posterior $C$ ({ndraws} draws)"),
            Line2D(
                [], [], color=C_INK, lw=1.2, ls="--",
                label=rf"median model $\times\,{med_ratio:.3f}$",
            ),
        ],
        fontsize=7.5,
        frameon=False,
        loc="upper left",
    )

    fig.suptitle(title, fontsize=10, y=0.985)
    fig.savefig(path, bbox_inches="tight", dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# cache: everything the figure needs, minus the five-minute rebuild
# ---------------------------------------------------------------------------

# Collection settings a cached npz must agree with before it may be reused. The
# grid and model flags are in here too: a cache built with the equal-arm model,
# or off the lite grid, holds a DIFFERENT C for the same chain and would be
# silently wrong rather than merely stale.
CACHE_KEYS = (
    "file", "channel", "ndraws", "seed", "discard", "thin", "tempered",
    "nslices", "slice_width", "full", "unequal_arm", "modulation",
)


def cache_signature(args, ndraws):
    sig = {k: getattr(args, k) for k in CACHE_KEYS if k != "file" and k != "ndraws"}
    sig["file"] = os.path.abspath(args.file)
    sig["ndraws"] = ndraws  # the realized count, which --ndraws only bounds
    return {k: str(v) for k, v in sig.items()}


def cache_load(path, sig):
    """The cached collection, or ``None`` if absent or built for other settings."""
    if not path or not os.path.exists(path):
        return None
    with np.load(path, allow_pickle=False) as z:
        stored = {k: str(z[f"sig_{k}"]) for k in CACHE_KEYS if f"sig_{k}" in z}
        if stored != sig:
            diff = [f"    {k}: cached {stored.get(k)!r} != now {sig[k]!r}"
                    for k in sig if stored.get(k) != sig[k]]
            print(f"ignoring {path}: built for other settings\n" + "\n".join(diff))
            return None
        out = {k: z[k] for k in ("keep", "w2", "ref", "data_slices", "model_slices")}
        extra = (z["f_arr"], z["t_arr"], [int(c) for c in z["col_centres"]])
    print(f"loaded {path} (no rebuild)")
    return out, extra


def cache_store(path, sig, out, f_arr, t_arr, cols):
    if not path:
        return
    np.savez_compressed(
        path,
        f_arr=f_arr,
        t_arr=t_arr,
        col_centres=np.array([c for c, _ in cols]),
        **out,
        **{f"sig_{k}": np.array(v) for k, v in sig.items()},
    )
    print(f"cached the collection to {path}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("file", help="run HDF5, e.g. noise-galfor-pe/noise_foreground_full5_testing.h5")
    p.add_argument("--discard", type=int, default=0, help="burn-in iterations to drop")
    p.add_argument("--thin", type=int, default=1)
    p.add_argument("--tempered", action="store_true", help="read from the sub_backend ladder")
    p.add_argument("--ndraws", type=int, default=100, help="posterior draws of C (default 100)")
    p.add_argument("--seed", type=int, default=0, help="seed for picking which samples to draw")
    p.add_argument("--channel", type=int, default=0, help="TDI channel index (default 0 = X)")
    p.add_argument("--nslices", type=int, default=4, help="equally spaced time slices (default 4)")
    p.add_argument(
        "--slice-width",
        type=int,
        default=1,
        help="wavelet time columns averaged into each slice (default 1, i.e. a "
        "single column). Applied to the data and the model alike; a width of k "
        "turns each data point from chi^2_1 into chi^2_k / k",
    )
    p.add_argument(
        "--time-bin",
        type=int,
        default=8,
        help="time columns averaged into one scalogram pixel, for DISPLAY only "
        "(default 8, ~8.5 h). 1 shows every column, which is pure chi^2_1 "
        "speckle. Does not touch the slice panels -- that is --slice-width",
    )
    p.add_argument(
        "--cache",
        metavar="NPZ",
        help="read the collected planes and draws from this npz if it exists, "
        "else compute them and write it. Skips the brick re-pour entirely on a "
        "restyle; the cached collection settings are checked against the CLI",
    )
    p.add_argument(
        "--units",
        default="psd",
        choices=("psd", "wdm"),
        help="'psd' (default) scales by 2 to the one-sided Fourier PSD; 'wdm' "
        "leaves the raw wavelet pixel power the likelihood sees",
    )
    p.add_argument(
        "--grid",
        default="auto",
        choices=("auto", "lite", "full"),
        help="the run's wavelet grid; 'auto' reads it off the backend file name",
    )
    p.add_argument("--noise-file", default=run_noise_only.NOISE_FILE)
    p.add_argument("--galfor-file", default=run_noise_only.GALFOR_FILE)
    p.add_argument(
        "--modulation",
        nargs="?",
        const=run_noise_only.MODULATION_FILE,
        default=None,
        metavar="PATH",
        help="tabulated foreground time modulation -- pass exactly what the run was given",
    )
    p.add_argument(
        "--unequal-arm",
        action="store_true",
        help="orbit-informed instrument covariance -- pass it iff the run did",
    )
    p.add_argument(
        "--wdm-psd-method",
        choices=("fold", "layer_constant", "layer_calibrated"),
        default="fold",
        help="unequal-arm WDM PSD construction -- pass what the run used. On a "
        "two-year grid the exact 'fold' default streams all Nt columns (~1 h); "
        "'layer_calibrated' is seconds for a ~1e-6 residual (see ppc_noise.py)",
    )
    p.add_argument(
        "--two-years",
        action="store_true",
        help="force the full-brick grid + conditioning; normally inferred from "
        "the run's stored Nt",
    )
    p.add_argument("--gpus", type=int, nargs="+", help="GPU device ids (omit for CPU)")
    p.add_argument(
        "--scratch-dir",
        default="./gf_output_ppc/",
        help="where the rebuild drops its artifacts (never the run's own dir)",
    )
    p.add_argument("--verbose", action="store_true", help="stream the build's DEBUG logs")
    p.add_argument("-o", "--out", help="output png (default: <file>_scalogram.png)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    branches = ppc_noise.chain_branches(args.file)
    if "psd" not in branches:
        raise SystemExit(f"no psd branch in {args.file}; found {branches}")
    if "sgwb" in branches:
        raise SystemExit("this chain carries an sgwb branch; run_noise_only builds noise_only.")
    mode = "foreground" if "galfor" in branches else "instrument"
    branches = ["psd"] + (["galfor"] if mode == "foreground" else [])
    inferred_full, inferred_two_years = ppc_noise.resolve_grid(
        args.file, args.grid, args.noise_file
    )
    args.full = inferred_full if args.grid == "auto" else args.grid == "full"
    args.two_years = args.two_years or inferred_two_years

    read_kwargs = dict(discard=args.discard, thin=args.thin, tempered=args.tempered)
    rows, flats = ppc_noise.load_aligned(args.file, branches, args.ndraws, args.seed, **read_kwargs)
    ndraws = len(rows["psd"])
    log_sampling = {b: corner_noise.resolve_basis(flats[b]) == "log" for b in branches}
    ref_params = {b: np.median(flats[b], axis=0) for b in branches}

    print(f"{args.file}")
    print(f"  mode            {mode}  (from the chain's branches)")
    grid_nt = ppc_noise.stored_nt(args.file)
    grid_name = "two-year brick" if args.two_years else ("full" if args.full else "lite")
    print(
        f"  grid            {grid_name}"
        + (f" (nt={grid_nt})  [from the run\'s stored Nt]" if grid_nt is not None else "")
    )
    print(
        "  basis           "
        + ", ".join(f"{b}={'log' if log_sampling[b] else 'linear'}" for b in branches)
    )
    print(
        f"  instrument      {'unequal-arm' if args.unequal_arm else 'equal-arm'}"
        + (f", wdm_psd_method={args.wdm_psd_method}" if args.unequal_arm else "")
    )
    print(f"  galfor mod.     {args.modulation or 'stationary'}")
    print(f"  samples         {flats['psd'].shape[0]} -> {ndraws} draws (seed {args.seed})")

    cached = cache_load(args.cache, cache_signature(args, ndraws))
    if cached is not None:
        out, (f_arr, t_arr, centres) = cached
        cols = slice_columns(len(t_arr), args.nslices, args.slice_width)
        # The signature pins nslices and slice_width, so slice_columns has to
        # reproduce the cached centres exactly. If it does not, the cache and
        # this build disagree about the grid itself -- refuse rather than plot
        # the model of one column over the data of another.
        if [c for c, _ in cols] != list(centres):
            raise SystemExit(
                f"{args.cache}: cached slice centres {list(centres)} != recomputed "
                f"{[c for c, _ in cols]}; delete it and rebuild."
            )
    else:
        print(
            "\nbuilding the run's data + noise model (loads and re-pours the bricks)...",
            flush=True,
        )
        general_info, source_info = ppc_noise.build_general_and_sources(args, mode, log_sampling)
        settings = general_info.domain_settings
        f_arr = np.asarray(settings.f_arr)
        t_arr = np.asarray(settings.t_arr)

        nch = general_info.input_data_residual_array.data_res_arr.arr.shape[0]
        if not 0 <= args.channel < nch:
            raise SystemExit(f"channel {args.channel} out of range; the data has {nch}")

        nt = len(t_arr)
        if not 1 <= args.slice_width <= nt:
            raise SystemExit(f"--slice-width must be in 1..{nt}")
        cols = slice_columns(nt, args.nslices, args.slice_width)

        print(
            f"grid {settings.Nf_active} layers x {settings.Nt_active} columns, "
            f"{f_arr[0]:.2e}-{f_arr[-1]:.2e} Hz, {settings.Tobs / 86400:.2f} d\n"
            f"slices at columns {[c for c, _ in cols]} "
            f"(t = {', '.join(f'{t_arr[c] / 86400:.1f}' for c, _ in cols)} d)\n"
            f"collecting {ndraws} draws...",
            flush=True,
        )
        out = collect(
            general_info, source_info, rows, args.channel, ref_params, cols, args.verbose
        )
        cache_store(args.cache, cache_signature(args, ndraws), out, f_arr, t_arr, cols)

    # Whitened chi^2 per pixel: E[w^2 / C] = 1 exactly under the model. One
    # number for the whole plane, then one per slice column.
    chi2 = out["w2"] / out["ref"]
    npix = chi2.size
    print(f"\n<w^2/C> over the whole plane: {chi2.mean():.4f}  "
          f"({npix} pixels, 1 +/- {np.sqrt(2 / npix):.4f})")
    print(f"  {'slice':>6s} {'t [d]':>8s} {'<w^2/C>':>9s} {'z':>7s}")
    for j, (c, sl) in enumerate(cols):
        block = chi2[:, sl]
        z = (block.mean() - 1.0) / np.sqrt(2.0 / block.size)
        print(f"  {j + 1:6d} {t_arr[c] / 86400:8.1f} {block.mean():9.4f} {z:+7.1f}")

    scale = WDM_TO_PSD if args.units == "psd" else 1.0
    unit_label = (
        r"one-sided PSD $S(f)$ [1/Hz]" if args.units == "psd" else "WDM pixel power $w^2$"
    )
    title = (
        f"{os.path.basename(args.file)} — channel {'XYZ'[args.channel]}, "
        f"data scalogram and {ndraws} posterior draws of the evolutionary PSD"
    )
    out_path = args.out or f"{os.path.splitext(args.file)[0]}_scalogram.png"
    plot(
        out,
        f_arr[out["keep"].astype(bool)] * 1e3,
        t_arr / 86400.0,
        cols,
        args.slice_width,
        args.time_bin,
        scale,
        unit_label,
        title,
        out_path,
    )
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
