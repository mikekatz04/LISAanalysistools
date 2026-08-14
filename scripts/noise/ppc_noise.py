#!/usr/bin/env python
"""Posterior draws of the fitted evolutionary PSD, over the data that produced it.

    python scripts/noise/ppc_noise.py noise-galfor-pe/noise_foreground_full5_testing.h5
    python scripts/noise/ppc_noise.py <file> --discard 300 --ndraws 400 --channel all

The noise model IS an evolutionary PSD: ``C[i, i, m, n]`` is the predicted
variance of the data's wavelet coefficient in layer ``m``, time column ``n``
(the WDM likelihood is exactly ``w ~ N(0, C)`` -- ``inner_product`` sums
``4 * 0.25 * w C^-1 w`` and ``logdet_factor`` is 0.5). The data's own
evolutionary PSD is ``w**2``. So the plot is the direct one: measure the data's
variance, overlay several hundred posterior draws of ``C`` at low opacity, and
read the fit off the superposition.

The measurement is reduced two ways, because a single wavelet pixel is a
1-dof variance estimate and shows nothing on its own:

    A  frequency   average over time columns -> per-layer PSD vs f, log-log.
    B  ratio       the same over the median-posterior model -- a 2% miss is
                   invisible in A across four decades and obvious here.
    C  time        average over layers -> broadband power vs time. Each layer
                   is divided by its own reference level first, so every layer
                   counts equally instead of the loudest one setting the curve.
                   Flat under a stationary model; this is the panel that shows
                   an annual foreground modulation or a drifting instrument.

Averaging down does not make the measurement exact, so B and C carry the
estimator's own 1-sigma scatter as a grey reference band. It is closed form, not
simulated: ``w**2 / C`` is chi^2_1 per pixel, so the frequency panel's fractional
scatter is ``sqrt(2 sum_n C^2) / sum_n C`` per layer and the time panel's is the
same construction across layers, divided by the smoothing width. (Checked
against 4000 Monte-Carlo replicate datasets: ratio 1.000, within 4% at the
1st/99th percentiles -- drawing fake noise measures this quantity and nothing else.) Both
formulas assume the pixel independence the likelihood itself assumes, so data
excursions beyond the band are a real model failure, correlation included.

Units. The folded WDM covariance is half the one-sided Fourier PSD -- verified
against ``get_sensitivity(f_arr, ...)``: ``2 * C / S_n(f) = 1 + <0.5%`` across
the band. Panel A is therefore plotted as ``2 x`` the pixel variance and
labelled as the one-sided PSD, directly comparable to ``fit_galfor.py``'s Welch
estimate. ``--units wdm`` leaves it as the raw pixel variance.

The model has to be built the way the RUN built it, so this script re-uses
``run_noise_only.build_fit`` and needs the same data/model flags the run used
(``--modulation``, ``--unequal-arm``, ``--wdm-psd-method``, the brick paths).
Three are inferred instead of asked for: the run mode comes from which branches
the chain carries, the psd/galfor sampling basis from the chain itself (so the
parameter transforms can never disagree with the samples), and the wavelet grid
-- lite, full, or ``--two-years`` -- from the run's own stored ``Nt``.
Everything resolved is printed before the build -- check it against the run.

Cost, on a two-year grid: the pour is a few minutes, and an exact-fold
unequal-arm basis is ~1 h on top (16060 columns, once -- every draw after it is
a linear combination of the cached bases). ``--wdm-psd-method layer_calibrated``
turns that hour into seconds for a residual ~1e-6, far below anything this plot
resolves.

Building loads and re-pours the bricks (a few minutes, several GB). It writes
nothing into the run's directory: the build is pointed at ``--scratch-dir``.
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
import run_noise_only  # noqa: E402  (sibling script; the run configuration itself)

# E[w_mn^2] == S_wdm[m] == S_n(f_m) / 2 -- see the module docstring.
WDM_TO_PSD = 2.0

# Two series, categorical identity, fixed order (never cycled): slots 1 and 2 of
# the validated default palette, the strongest pair in it (all-pairs CVD dE 24.7,
# normal-vision 33.6, both above 3:1 on white).
C_DATA = "#2a78d6"  # slot 1, blue   -- the measured evolutionary PSD
C_MODEL = "#eb6834"  # slot 2, orange -- posterior draws of the model
C_INK = "0.25"  # neutral ink: reference lines and the estimator band, never a series


# ---------------------------------------------------------------------------
# reductions of an evolutionary PSD (data ``w**2`` or model ``C`` alike)
# ---------------------------------------------------------------------------


def freq_stat(evo: np.ndarray) -> np.ndarray:
    """``(Nf, Nt)`` evolutionary PSD -> per-layer time average, ``(Nf,)``."""
    return evo.mean(axis=1)


def time_stat(evo: np.ndarray, ref_layer: np.ndarray, smooth: int) -> np.ndarray:
    """``(Nf, Nt)`` -> layer-normalized broadband power per time column, ``(Nt,)``.

    Dividing by ``ref_layer`` before the layer average is what makes this a
    *broadband* statistic: the raw mean over layers is ~entirely the lowest
    (loudest) layer, whose level is four decades above the top of the band.
    Normalized, every layer contributes equally and the result is ~1 under a
    correct stationary model.
    """
    return boxcar(np.mean(evo / ref_layer[:, None], axis=0), smooth)


def _running(y: np.ndarray, n: int):
    """``(window sum, window occupancy)`` of width ``n``, edges truncated.

    ``mode="same"`` alone would divide the short end windows by the full width
    and pull both ends of the time panel toward zero -- indistinguishable from
    the WDM edge effect the panel is meant to reveal. Carrying the actual
    occupancy keeps the ends unbiased, and the variance below needs it too.
    """
    k = np.ones(int(n))
    return np.convolve(y, k, mode="same"), np.convolve(np.ones_like(y), k, mode="same")


def boxcar(y: np.ndarray, n: int) -> np.ndarray:
    """Running mean of width ``n``, with edge windows truncated (not zero-padded)."""
    if n <= 1:
        return y
    total, count = _running(y, n)
    return total / count


# ---------------------------------------------------------------------------
# how well the reductions measure -- closed form, no simulated datasets
# ---------------------------------------------------------------------------
#
# Every pixel of the measured evolutionary PSD is ``w**2`` with ``w ~ N(0, C)``,
# i.e. ``C`` times a chi^2_1: mean ``C``, variance ``2 C**2``. Both panels
# average those pixels, so both estimator scatters follow from that one fact and
# the model's own ``C`` -- there is nothing a replicate dataset could add.


def freq_sigma(C: np.ndarray) -> np.ndarray:
    """Fractional 1-sigma of ``freq_stat(w**2)`` about ``freq_stat(C)``, per layer.

    ``Var(mean_n w^2) = (1/Nt^2) sum_n 2 C^2``, over a mean of ``(1/Nt) sum_n C``.
    Reduces to ``sqrt(2/Nt)`` for a time-constant layer and stays exact when the
    layer is not (a modulated foreground, per-time-slice unequal arms).
    """
    return np.sqrt(2.0 * np.sum(C**2, axis=1)) / np.sum(C, axis=1)


def time_sigma(C: np.ndarray, ref_layer: np.ndarray, smooth: int) -> np.ndarray:
    """Absolute 1-sigma of ``time_stat(w**2, ...)`` about ``time_stat(C, ...)``.

    Per column the statistic is ``(1/Nf) sum_m w^2/ref``, so its variance is
    ``(2/Nf^2) sum_m (C/ref)^2``. The boxcar then averages ``count`` independent
    columns, dividing that variance by ``count`` -- which is why the edge
    occupancy has to be carried rather than assumed equal to the width.
    """
    r = C / ref_layer[:, None]
    var = 2.0 * np.sum(r**2, axis=0) / r.shape[0] ** 2
    if smooth <= 1:
        return np.sqrt(var)
    total, count = _running(var, smooth)
    return np.sqrt(total) / count


def within(data: np.ndarray, model: np.ndarray, sigma: np.ndarray, nsig: float = 1.645) -> float:
    """Fraction of points inside the model's ``+/- nsig * sigma`` interval (90% by default)."""
    return float(np.mean(np.abs(data - model) <= nsig * sigma))


def to_physical(transform_fn, row):
    """One branch's sampling-basis row -> the physical basis the noise model takes.

    Mirrors ``PSDMove._to_physical``: ``both_transforms`` can hand back a leading
    axis for a single row, so squeeze it. A ``None`` transform means the branch
    was sampled linearly and the row is already physical.
    """
    if transform_fn is None or row is None:
        return row
    out = transform_fn.both_transforms(np.array(row, dtype=float), copy=True)
    return np.atleast_1d(np.asarray(out).squeeze())


# ---------------------------------------------------------------------------
# chain + fit loading
# ---------------------------------------------------------------------------


def chain_branches(path, group="global_fit"):
    """Branch names stored in the run's cold chain."""
    import h5py

    with h5py.File(path, "r") as f:
        if group not in f:  # files written before the "mcmc" -> "global_fit" rename
            group = "mcmc"
        return sorted(f[group]["chain"])


def load_aligned(path, branches, ndraws, seed, **read_kwargs):
    """``(rows, flats)`` -- ``ndraws`` posterior draws, aligned across branches.

    Each branch's cold sub-state syncs its beta=1 row into the shared main state
    (``ModuleSubState.sync_cold_row``), so at a given iteration walker ``w``
    holds every branch's cold value and row ``i`` of each flattened block is one
    point in the JOINT noise space. ``mask=False`` is what enforces that: it
    refuses rather than silently dropping a different set of rows per branch.
    """
    flats = {b: corner_noise.load_samples(path, b, mask=False, **read_kwargs) for b in branches}
    counts = {a.shape[0] for a in flats.values()}
    if len(counts) != 1:
        raise SystemExit(f"branches have different sample counts {counts}; cannot pair them.")
    nsamples = counts.pop()
    if nsamples == 0:
        raise SystemExit("no samples left after --discard/--thin.")

    rng = np.random.default_rng(seed)
    n = min(ndraws, nsamples)
    # Without replacement while the chain can afford it: duplicated draws would
    # pile identical curves at the same opacity and misrepresent the density.
    picks = rng.choice(nsamples, size=n, replace=n > nsamples)
    return {b: a[picks] for b, a in flats.items()}, flats


def stored_nt(path, group="global_fit"):
    """The run's WDM ``Nt``, or ``None`` for a backend that did not store it.

    ``domain_settings/kwargs/omega`` is one entry per WDM time column, so its
    length IS the grid the run sampled -- the same handle ``run_noise_only.
    check_resume`` uses to refuse an incompatible resume.
    """
    import h5py

    with h5py.File(path, "r") as f:
        if group not in f:  # files written before the "mcmc" -> "global_fit" rename
            group = "mcmc"
        grp = f.get(group)
        omega = None if grp is None else grp.get("domain_settings/kwargs/omega")
        return None if omega is None else int(omega.shape[0])


def resolve_grid(path, choice, noise_file):
    """``(full, two_years)`` -- the wavelet grid the run actually sampled.

    Read off the run's own stored ``Nt`` (:func:`stored_nt`) rather than
    guessed from the file name: 256 is the lite preset, 1024 full, and the
    ``--two-years`` grid is whatever the brick leaves after the edge trim
    (16060 for the bundled one), which no name convention encodes. Getting
    this wrong changes Nt, and the build would then pour a different dataset
    than the run sampled.

    ``choice`` other than ``auto`` forces the preset for backends too old to
    carry ``omega``; the name is the last resort. Under ``--two-years`` the
    preset picks only the ladder sizes -- the grid comes from the brick -- so
    it does not matter which one an inferred two-year run reports.
    """
    nt = stored_nt(path)
    if nt is not None:
        if nt == 256:
            return False, False
        if nt == 1024:
            return True, False
        two_year_nt = run_noise_only._two_year_grid(noise_file)[1]
        if nt == two_year_nt:
            return True, True
        raise SystemExit(
            f"{os.path.basename(path)} was sampled on an Nt={nt} grid, which is "
            f"neither preset (lite 256, full 1024) nor the --two-years grid for "
            f"{os.path.basename(noise_file)} (Nt={two_year_nt}). Point --noise-file "
            "at the brick the run used; the model has to be built on the run's grid."
        )
    if choice != "auto":
        return choice == "full", "2yr" in os.path.basename(path)
    base = os.path.basename(path)
    has_full, has_lite = "full" in base, "lite" in base
    if has_full == has_lite:  # both or neither: nothing to infer from
        raise SystemExit(
            f"{base!r} stores no WDM grid and its name does not carry the preset; "
            "pass --grid full or --grid lite (full: nt=1024, lite: nt=256 -- it "
            "must match the run)."
        )
    return has_full, "2yr" in base


def build_general_and_sources(args, mode, log_sampling):
    """Build the run's data + model, isolated from the run's own directory.

    ``build_general`` pours the bricks onto the run's WDM grid and constructs the
    sensitivity backend; ``build_sources`` fills the per-branch priors and, more
    to the point here, the sampling->physical ``TransformContainer``. Neither
    creates an HDF backend -- that only happens in the full ``fit.build()`` --
    but ``GeneralSetup`` does make an artifacts directory and drop a
    ``wdm_data.png`` in it, so the fit is pointed at the scratch dir.
    """
    run_args = run_noise_only.parse_args([])  # every field at its documented default
    run_args.noise_file = args.noise_file
    run_args.galfor_file = args.galfor_file
    run_args.full = args.full
    # The run's conditioning AND its grid: build_fit re-derives Nf/Nt from the
    # brick and re-installs the highpass + 200 h edge trim the run poured with.
    run_args.two_years = args.two_years
    run_args.modulation = args.modulation
    run_args.unequal_arm = args.unequal_arm
    run_args.wdm_psd_method = args.wdm_psd_method
    run_args.gpus = args.gpus
    run_args.out_dir = os.path.join(args.scratch_dir, "")
    run_args.tag = "ppc"
    run_args.progress = False
    run_args.verbose = args.verbose
    # Inferred from the chain, not asked for: this is what guarantees the
    # transform applied to a sample matches the basis the sample is in.
    run_args.linear_psd = not log_sampling.get("psd", True)
    run_args.linear_galfor = not log_sampling.get("galfor", True)

    os.makedirs(run_args.out_dir, exist_ok=True)
    fit = run_noise_only.build_fit(mode, run_args)
    general_info = fit.build_general()
    source_info = fit.build_sources()
    return general_info, source_info


# ---------------------------------------------------------------------------
# the comparison itself
# ---------------------------------------------------------------------------


def covariance_channel(sens, channel):
    """``(Nf_active, Nt_active)`` real evolutionary PSD for one TDI channel."""
    from lisatools.utils.utility import asnumpy

    return np.real(asnumpy(sens.sens_mat)[channel, channel])


def reduce_all(general_info, source_info, rows, channels, ref_params, smooth, verbose):
    """Reduce the measured and the drawn evolutionary PSDs onto the two statistics.

    Returns a dict keyed by channel. Only the reductions are kept: a single
    draw's ``C`` is ``(Nf, Nt)`` and several hundred of them would be no use
    held at once.
    """
    from lisatools.utils.utility import asnumpy

    backend = general_info.sensitivity_backend
    transforms = {b: getattr(setup, "transform", None) for b, setup in source_info.items()}
    branches = list(rows)

    def build(name, row_of):
        kwargs = {}
        if "galfor" in branches:
            kwargs["galfor_params"] = to_physical(transforms["galfor"], row_of("galfor"))
        return backend(name, to_physical(transforms["psd"], row_of("psd")), **kwargs)

    ref_sens = build("ppc_ref", lambda b: ref_params[b])
    data_arr = asnumpy(general_info.input_data_residual_array.data_res_arr.arr)

    ndraws = len(next(iter(rows.values())))
    out = {}

    for ch in channels:
        w = data_arr[ch]
        ref = covariance_channel(ref_sens, ch)
        # A zeroed / non-finite pixel is a fold artifact (``instrument_fill_nans``
        # zeroes the f=0 divergence), not data: it would divide the time
        # statistic by zero. Drop those layers from BOTH families so every curve
        # is the same reduction over the same pixels.
        keep = np.isfinite(ref).all(axis=1) & (ref > 0).all(axis=1)
        if not keep.any():
            raise SystemExit(f"channel {ch}: reference model is degenerate everywhere.")
        w, ref = w[keep], ref[keep]
        ref_layer = freq_stat(ref)

        # Whitened per-pixel chi^2. E[w^2 / C_ref] = 1 exactly under the model,
        # with variance 2/Npix, so the z-score is a single sharp number for "is
        # the level right" -- the one statistic that uses every pixel rather
        # than a reduction of them.
        chi2 = float(np.mean(w**2 / ref))
        npix = w.size

        out[ch] = dict(
            keep=keep,
            npix=npix,
            chi2=chi2,
            chi2_z=(chi2 - 1.0) / np.sqrt(2.0 / npix),
            ref_freq=ref_layer,
            ref_time=time_stat(ref, ref_layer, smooth),
            sigma_freq=freq_sigma(ref),  # fractional, panel B
            sigma_time=time_sigma(ref, ref_layer, smooth),  # absolute, panel C
            data_freq=freq_stat(w**2),
            data_time=time_stat(w**2, ref_layer, smooth),
            model_freq=np.empty((ndraws, int(keep.sum()))),
            model_time=np.empty((ndraws, ref.shape[1])),
        )

    for i in range(ndraws):
        if verbose and (i % 50 == 0):
            print(f"  draw {i}/{ndraws}", flush=True)
        sens = build(f"ppc_{i}", lambda b, i=i: rows[b][i])
        for ch in channels:
            o = out[ch]
            C = covariance_channel(sens, ch)[o["keep"]]
            o["model_freq"][i] = freq_stat(C)
            o["model_time"][i] = time_stat(C, o["ref_freq"], smooth)

    return out


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------


def plot(out, channels, f_arr, t_arr, scale, unit_label, ndraws, title, path):
    """Three rows (frequency / ratio / time) per channel column."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter

    # A few hundred overlaid lines only read as a density if no single one is
    # legible on its own; floor it so a 20-draw smoke still shows something.
    alpha = float(np.clip(4.0 / max(ndraws, 1), 0.02, 0.35))
    # The data line has to stay readable where it crosses the densest part of
    # the posterior bundle -- a surface-colored ring is the standard relief.
    halo = [pe.withStroke(linewidth=2.4, foreground="white")]
    t_days = t_arr / 86400.0
    # mHz, not Hz: the band spans 0.4-8 mHz, i.e. barely a decade, so the log
    # axis has one major tick in it and matplotlib falls back to labelling
    # every minor tick in scientific notation -- an unreadable smear.
    f_mhz = f_arr * 1e3

    ncol = len(channels)
    fig = plt.figure(figsize=(6.2 * ncol, 9.6))
    grid = fig.add_gridspec(
        3, ncol, height_ratios=[3.0, 1.1, 2.0], hspace=0.32, wspace=0.24, top=0.93
    )

    for col, ch in enumerate(channels):
        o = out[ch]
        f = f_mhz[o["keep"]]
        ax_f = fig.add_subplot(grid[0, col])
        ax_r = fig.add_subplot(grid[1, col], sharex=ax_f)
        ax_t = fig.add_subplot(grid[2, col])

        # -- A: per-layer time-averaged PSD ----------------------------------
        ax_f.plot(f, (o["model_freq"] * scale).T, color=C_MODEL, lw=0.8, alpha=alpha)
        ax_f.plot(f, o["data_freq"] * scale, color=C_DATA, lw=1.2, path_effects=halo)
        ax_f.set_xscale("log")
        ax_f.set_yscale("log")
        ax_f.set_ylabel(unit_label, fontsize=9)
        ax_f.set_title(f"channel {'XYZ'[ch]}", fontsize=10)

        # -- B: the same, over the median-posterior model ---------------------
        # The estimator band goes down first and in neutral ink: it is what the
        # measurement can resolve, not another model.
        ax_r.fill_between(
            f,
            1.0 - o["sigma_freq"],
            1.0 + o["sigma_freq"],
            color=C_INK,
            alpha=0.16,
            lw=0,
            zorder=0,
        )
        ax_r.axhline(1.0, color=C_INK, lw=1.0, ls="--", zorder=1)
        ax_r.plot(
            f, (o["model_freq"] / o["ref_freq"]).T, color=C_MODEL, lw=0.8, alpha=alpha, zorder=2
        )
        ax_r.plot(
            f, o["data_freq"] / o["ref_freq"], color=C_DATA, lw=1.2, path_effects=halo, zorder=3
        )
        ax_r.set_ylabel("ratio to median model", fontsize=9)
        ax_r.set_xlabel("frequency [mHz]", fontsize=9)
        ax_r.set_ylim(*symmetric_limits(o["data_freq"] / o["ref_freq"], o["sigma_freq"], 1.0))

        # -- C: layer-normalized broadband power per time column -------------
        ax_t.fill_between(
            t_days,
            o["ref_time"] - o["sigma_time"],
            o["ref_time"] + o["sigma_time"],
            color=C_INK,
            alpha=0.16,
            lw=0,
            zorder=0,
        )
        ax_t.plot(t_days, o["ref_time"], color=C_INK, lw=1.0, ls="--", zorder=1)
        ax_t.plot(t_days, o["model_time"].T, color=C_MODEL, lw=0.8, alpha=alpha, zorder=2)
        ax_t.plot(t_days, o["data_time"], color=C_DATA, lw=1.2, path_effects=halo, zorder=3)
        ax_t.set_xlabel("time from data start [days]", fontsize=9)
        ax_t.set_ylabel("broadband power / model", fontsize=9)
        ax_t.set_ylim(*symmetric_limits(o["data_time"], o["sigma_time"], o["ref_time"]))

        for ax in (ax_f, ax_r):
            # Decade-major ticks subdivided at 2 and 5, labelled as plain
            # numbers; every other minor tick unlabelled.
            ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.xaxis.set_minor_formatter(NullFormatter())
        for ax in (ax_f, ax_r, ax_t):
            ax.grid(alpha=0.25, lw=0.5, which="both")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(labelsize=8)
        # ``which="both"``: the shared-axis label suppression has to reach the
        # minor tick labels too, or the log axis keeps drawing them under A.
        ax_f.tick_params(which="both", labelbottom=False)

        if col == 0:
            # Identity is never opacity-alone: the legend proxies are opaque.
            ax_f.legend(
                handles=[
                    Line2D([], [], color=C_DATA, lw=1.6, label="data"),
                    Line2D(
                        [], [], color=C_MODEL, lw=1.6, label=f"posterior model draws ({ndraws})"
                    ),
                    Patch(facecolor=C_INK, alpha=0.16, label=r"$\pm1\sigma$ estimator scatter"),
                ],
                fontsize=8,
                frameon=False,
                loc="upper left",
            )

    fig.suptitle(title, fontsize=10, y=0.995)
    fig.savefig(path, bbox_inches="tight", dpi=130)
    plt.close(fig)


def symmetric_limits(data, sigma, center, pad=0.35):
    """y-limits holding the data and at least +/-2.5 sigma of the estimator band.

    Keyed off the band rather than the data alone so a single wild layer edge
    cannot flatten the few-percent structure these panels exist to show, and off
    the data as well so an excursion is never silently clipped out of frame.
    """
    lo = min(np.min(data), np.min(center - 2.5 * sigma))
    hi = max(np.max(data), np.max(center + 2.5 * sigma))
    return lo - pad * (hi - lo), hi + pad * (hi - lo)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("file", help="run HDF5, e.g. noise-galfor-pe/noise_foreground_full5_testing.h5")
    p.add_argument("--discard", type=int, default=0, help="burn-in iterations to drop")
    p.add_argument("--thin", type=int, default=1)
    p.add_argument("--tempered", action="store_true", help="read from the sub_backend ladder")
    p.add_argument(
        "--ndraws",
        type=int,
        default=300,
        help="posterior draws overlaid (default 300). Each costs one sensitivity "
        "rebuild, ~3 ms on the stock 59x1024 active grid",
    )
    p.add_argument("--seed", type=int, default=0, help="seed for picking which samples to draw")
    p.add_argument(
        "--channel",
        default="0",
        help="TDI channel index, a comma-separated list, or 'all' (one column each). Default 0 (X)",
    )
    p.add_argument(
        "--time-smooth",
        type=int,
        default=24,
        help="running-mean width, in wavelet time columns, for the time panel. "
        "A column is Nf*dt = 3840 s on both grid presets, so the default 24 is "
        "~1 day. The estimator band narrows with it by the same sqrt(count), so "
        "the panel stays honest at any width. 1 disables it",
    )
    p.add_argument(
        "--units",
        default="psd",
        choices=("psd", "wdm"),
        help="'psd' (default) scales the frequency panel by 2 to the one-sided "
        "Fourier PSD, matching fit_galfor.py; 'wdm' leaves it as the raw "
        "wavelet pixel variance the likelihood sees",
    )
    p.add_argument(
        "--grid",
        default="auto",
        choices=("auto", "lite", "full"),
        help="the run's wavelet grid: full (nt=1024) or lite (nt=256). 'auto' "
        "(default) reads it off the backend file name",
    )
    p.add_argument("--noise-file", default=run_noise_only.NOISE_FILE)
    p.add_argument("--galfor-file", default=run_noise_only.GALFOR_FILE)
    p.add_argument(
        "--modulation",
        nargs="?",
        const=run_noise_only.MODULATION_FILE,
        default=None,
        metavar="PATH",
        help="tabulated galactic-foreground time modulation -- pass exactly what "
        "the run was given, or the model here will not be the model that was fit",
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
        help="unequal-arm WDM PSD construction (run_noise_only.py --wdm-psd-method): "
        "pass what the run used. On a two-year grid the default exact 'fold' "
        "streams all Nt columns -- ~1 h before the first draw. 'layer_calibrated' "
        "is one exact fold plus a per-layer correction (seconds, residual ~1e-6 in "
        "the basis, ~500x below the posterior width): a sound stand-in here even "
        "for a run that folded, since this plot resolves nothing near that level",
    )
    p.add_argument(
        "--two-years",
        action="store_true",
        help="force the full-brick grid + conditioning (run_noise_only.py "
        "--two-years). Inferred from the run's stored Nt, so it is only needed "
        "for a backend too old to carry one",
    )
    p.add_argument("--gpus", type=int, nargs="+", help="GPU device ids (omit for CPU)")
    p.add_argument(
        "--scratch-dir",
        default="./gf_output_ppc/",
        help="where the rebuild drops its artifacts (never the run's own dir)",
    )
    p.add_argument("--verbose", action="store_true", help="stream the build's DEBUG logs")
    p.add_argument("-o", "--out", help="output png (default: <file>_ppc.png)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    branches = chain_branches(args.file)
    if "sgwb" in branches:
        raise SystemExit(
            "this chain carries an sgwb branch; run_noise_only.py builds the "
            "noise_only variant, which has none. Extend build_general_and_sources "
            "to the noise_sgwb variant before plotting it."
        )
    if "psd" not in branches:
        raise SystemExit(f"no psd branch in {args.file}; found {branches}")
    mode = "foreground" if "galfor" in branches else "instrument"
    branches = ["psd"] + (["galfor"] if mode == "foreground" else [])
    inferred_full, inferred_two_years = resolve_grid(args.file, args.grid, args.noise_file)
    args.full = inferred_full if args.grid == "auto" else args.grid == "full"
    args.two_years = args.two_years or inferred_two_years

    read_kwargs = dict(discard=args.discard, thin=args.thin, tempered=args.tempered)
    rows, flats = load_aligned(args.file, branches, args.ndraws, args.seed, **read_kwargs)
    ndraws = len(rows["psd"])
    log_sampling = {b: corner_noise.resolve_basis(flats[b]) == "log" for b in branches}
    # The reference model: posterior median in the SAMPLING basis, transformed
    # like any other row. Only ever a normalizer, so the median-of-marginals
    # (rather than a joint point estimate) is all this needs to be.
    ref_params = {b: np.median(flats[b], axis=0) for b in branches}

    print(f"{args.file}")
    print(f"  mode            {mode}  (from the chain's branches)")
    grid_nt = stored_nt(args.file)
    grid_name = (
        "two-year brick" if args.two_years else ("full" if args.full else "lite")
    )
    print(
        f"  grid            {grid_name}"
        + (f" (nt={grid_nt})" if grid_nt is not None else "")
        + ("  [from the run\'s stored Nt]" if grid_nt is not None else "  [from the file name]")
    )
    print(
        "  basis           "
        + ", ".join(f"{b}={'log' if log_sampling[b] else 'linear'}" for b in branches)
        + "  (from the chain)"
    )
    print(
        f"  instrument      {'unequal-arm' if args.unequal_arm else 'equal-arm'}"
        + (f", wdm_psd_method={args.wdm_psd_method}" if args.unequal_arm else "")
    )
    print(f"  galfor mod.     {args.modulation or 'stationary'}")
    print(f"  samples         {flats['psd'].shape[0]} -> {ndraws} draws (seed {args.seed})")
    print("\nbuilding the run's data + noise model (loads and re-pours the bricks)...", flush=True)

    general_info, source_info = build_general_and_sources(args, mode, log_sampling)
    settings = general_info.domain_settings
    f_arr = np.asarray(settings.f_arr)
    t_arr = np.asarray(settings.t_arr)

    nch = general_info.input_data_residual_array.data_res_arr.arr.shape[0]
    if args.channel == "all":
        channels = list(range(nch))
    else:
        channels = [int(c) for c in args.channel.split(",") if c.strip() != ""]
    for ch in channels:
        if not 0 <= ch < nch:
            raise SystemExit(f"channel {ch} out of range; the data has {nch}")

    print(
        f"grid {settings.Nf_active} layers x {settings.Nt_active} columns, "
        f"{settings.f_arr[0]:.2e}-{settings.f_arr[-1]:.2e} Hz, "
        f"{settings.Tobs / 86400:.2f} d\nreducing {ndraws} draws...",
        flush=True,
    )
    out = reduce_all(
        general_info, source_info, rows, channels, ref_params, args.time_smooth, args.verbose
    )

    print("\ndata inside the median model's 90% estimator interval")
    print(f"  {'':8s} {'frequency':>11s} {'time':>9s} {'chi2/pix':>10s} {'z':>8s}")
    for ch in channels:
        o = out[ch]
        cov_f = within(o["data_freq"] / o["ref_freq"], 1.0, o["sigma_freq"])
        cov_t = within(o["data_time"], o["ref_time"], o["sigma_time"])
        print(
            f"  {'XYZ'[ch]:8s} {100 * cov_f:10.1f}% {100 * cov_t:8.1f}% "
            f"{o['chi2']:10.4f} {o['chi2_z']:+8.1f}"
        )
    npix = out[channels[0]]["npix"]
    print(
        f"  expected ~90% in each; chi2/pix is <w^2/C> over {npix} active pixels "
        f"(1 +/- {np.sqrt(2 / npix):.4f}), z its deviation in sigma"
    )

    scale = WDM_TO_PSD if args.units == "psd" else 1.0
    unit_label = r"one-sided PSD $S(f)$ [1/Hz]" if args.units == "psd" else "WDM pixel variance"
    title = (
        f"{os.path.basename(args.file)} — evolutionary PSD posterior vs data, {mode}, "
        f"{ndraws} draws (discard {args.discard}, thin {args.thin})"
    )
    out_path = args.out or f"{os.path.splitext(args.file)[0]}_ppc.png"
    plot(out, channels, f_arr, t_arr, scale, unit_label, ndraws, title, out_path)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
