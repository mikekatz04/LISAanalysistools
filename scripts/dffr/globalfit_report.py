"""Report of DFFR validation item 5: two global-fit runs of the high-frequency GBs, one per response.

Reads two run folders written by gb_high_freq_globalfit_settings.py and writes into OUTPUT:
report.txt and summary.json (run health, speed and GPU memory, offsets of the posterior means from
the injected values, and the two posteriors of every source compared), and the figures
health.png, speed_memory.png, convergence.png, truth_offsets.png,
comparison_{bias,sd_ratio,ks}.png, traces/gbNN.png and corners/gbNN.png (one per injected source).

    python scripts/dffr/globalfit_report.py --runs RUN_A RUN_B --output OUTPUT

Only the cold chain is used, after discarding the first DISCARD fraction of the iterations. Run A
is the reference of the comparison: differences are in units of its posterior standard deviation.
The runs stop on a wall clock, so they reach different iteration counts; the two posteriors are
compared over the first iterations that both reached.
The GB move stores the leaves in band order, so a slot is not a source: in every iteration and
walker each live leaf is matched to the injected source nearest in f0.
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime

import h5py
import numpy as np
from scipy.stats import ks_2samp

PARAMETERS = ("lnA", "f0", "fdot", "phi0", "cos_iota", "psi", "alpha", "sin_delta")
LABELS = (
    r"$\ln A$",
    r"$f_0$",
    r"$\dot f$",
    r"$\phi_0$",
    r"$\cos\iota$",
    r"$\psi$",
    r"$\alpha$",
    r"$\sin\delta$",
)
UNITS = ("", "mHz", "Hz/s", "rad", "", "rad", "rad", "")
# * Validated categorical slots 1-2 of the dataviz reference palette, then ink, muted and grid.
RUN_COLORS = ("#2a78d6", "#eb6834")
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
# ? 95 % two-sample KS level: D = 1.36 sqrt((n_a + n_b) / (n_a n_b)) for independent samples.
KS_LEVEL_COEFFICIENT = 1.36
LOG_TIME_FORMAT = "%Y-%m-%d %H:%M:%S,%f"
# * Leaf-to-source match window in f0: injected sources lie 8e-4 mHz apart or more, posterior
# * widths of f0 are 3e-7 mHz or less.
MATCH_TOLERANCE_MHZ = 1e-5


@dataclass
class RunData:
    """Cold chain, sampler counters and log records of one run."""

    name: str
    folder: str
    iterations: int
    coords: np.ndarray  # (iterations, walkers, leaves, 8) sampling basis
    alive: np.ndarray  # (iterations, walkers, leaves)
    log_like: np.ndarray  # (iterations, temperatures, walkers)
    band_edges_hz: np.ndarray
    counters: dict = field(default_factory=dict)
    iteration_starts: list = field(default_factory=list)
    proposal_s: list = field(default_factory=list)
    drift_lnl: list = field(default_factory=list)
    rows_outside: list = field(default_factory=list)
    alive_per_temperature: list = field(default_factory=list)
    # * One dict per iteration from the move's GB_TIMING line: stage name -> seconds.
    stage_s: list = field(default_factory=list)
    gpu_memory_mib: np.ndarray = field(default_factory=lambda: np.zeros(0))
    gpu_utilisation: np.ndarray = field(default_factory=lambda: np.zeros(0))
    gpu_time_s: np.ndarray = field(default_factory=lambda: np.zeros(0))
    # * Filled by match_sources: (iterations, walkers, sources, 8), its mask, and unmatched leaves.
    matched: np.ndarray = field(default_factory=lambda: np.zeros(0))
    found: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    unmatched: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))


def match_sources(run: RunData, truth_f0_mhz: np.ndarray) -> None:
    """Assign every live cold-chain leaf to the injected source nearest in f0, within the window."""
    iterations, walkers, _, ndim = run.coords.shape
    run.matched = np.full((iterations, walkers, truth_f0_mhz.size, ndim), np.nan)
    run.found = np.zeros((iterations, walkers, truth_f0_mhz.size), dtype=bool)
    run.unmatched = np.zeros((iterations, walkers), dtype=int)
    for iteration in range(iterations):
        for walker in range(walkers):
            leaves = np.flatnonzero(run.alive[iteration, walker])
            distance = np.abs(run.coords[iteration, walker, leaves, 1][:, None] - truth_f0_mhz)
            nearest = distance.argmin(axis=1)
            close = distance[np.arange(leaves.size), nearest] < MATCH_TOLERANCE_MHZ
            # ? A source claimed by two leaves keeps neither; it then counts as missing.
            claims = np.bincount(nearest[close], minlength=truth_f0_mhz.size)
            good = close & (claims[nearest] == 1)
            run.matched[iteration, walker, nearest[good]] = run.coords[
                iteration, walker, leaves[good]
            ]
            run.found[iteration, walker, nearest[good]] = True
            run.unmatched[iteration, walker] = int((~good).sum())


def load_run(folder: str) -> RunData:
    backend_paths = glob.glob(os.path.join(folder, "*_parameter_estimation_main.h5"))
    if len(backend_paths) != 1:
        raise FileNotFoundError(f"expected one backend file in {folder}, found {backend_paths}")
    with h5py.File(backend_paths[0], "r") as handle:
        group = handle["global_fit"]
        iterations = int(group.attrs["iteration"])
        # * Sampler axis 0 and temperature 0: the cold chain.
        coords = group["chain/gb"][:iterations, 0, 0]
        alive = group["inds/gb"][:iterations, 0, 0]
        log_like = group["log_like"][:iterations, 0]
        sub_group = group["sub_backend/gb"]
        band_edges_hz = sub_group["band_edges"][:]
        counters = {
            name: sub_group[name][:iterations]
            for name in (
                "band_num_accepted",
                "band_num_proposed",
                "band_num_accepted_rj",
                "band_num_proposed_rj",
                "band_swaps_accepted",
                "band_swaps_proposed",
            )
            if name in sub_group
        }
    run = RunData(
        name=os.path.basename(os.path.normpath(folder)),
        folder=folder,
        iterations=iterations,
        coords=coords,
        alive=alive,
        log_like=log_like,
        band_edges_hz=band_edges_hz,
        counters=counters,
    )
    parse_log(run, os.path.join(folder, "run.log"))
    parse_gpu_memory(run, os.path.join(folder, "gpu_memory.csv"))
    return run


def parse_log(run: RunData, path: str) -> None:
    if not os.path.exists(path):
        return
    stamp = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3}) - ")
    with open(path, errors="replace") as handle:
        for line in handle:
            match = stamp.match(line)
            if "Number of active leaves before proposal" in line and match:
                run.iteration_starts.append(datetime.strptime(match.group(1), LOG_TIME_FORMAT))
            elif "Runtime of rj_prior proposal is" in line:
                run.proposal_s.append(float(re.search(r"is ([\d.]+) seconds", line).group(1)))
            elif "incremental ll drift" in line:
                run.drift_lnl.append(float(re.search(r"drift ([\d.eE+-]+)", line).group(1)))
                outside = re.search(r"(\d+) of (\d+) rows outside", line)
                if outside:
                    run.rows_outside.append((int(outside.group(1)), int(outside.group(2))))
            elif "[GB_TIMING" in line:
                run.stage_s.append(
                    {name: float(value) for name, value in re.findall(r"(\w+)=([\d.]+)s\b", line)}
                )
            elif "Alive sources per temp after run_proposal" in line:
                run.alive_per_temperature.append(
                    [int(value) for value in re.findall(r"\d+", line.split(":", 3)[-1])]
                )


def parse_gpu_memory(run: RunData, path: str) -> None:
    if not os.path.exists(path):
        return
    times, memory, utilisation = [], [], []
    with open(path) as handle:
        for line in handle:
            fields = [value.strip() for value in line.split(",")]
            if len(fields) < 3 or not fields[1].endswith("MiB"):
                continue
            times.append(datetime.strptime(fields[0], "%Y/%m/%d %H:%M:%S.%f").timestamp())
            memory.append(float(fields[1].split()[0]))
            utilisation.append(float(fields[2].split()[0]))
    if times:
        run.gpu_time_s = np.asarray(times) - times[0]
        run.gpu_memory_mib = np.asarray(memory)
        run.gpu_utilisation = np.asarray(utilisation)


def injected_sources(
    band_edges_hz: np.ndarray, mojito_folder: str
) -> tuple[np.ndarray, np.ndarray]:
    """Catalogue rows and sampling-basis values of the injected sources, as the settings file picks them."""
    from lisatools.globalfit.recipe import gb_catalogue_to_sampling_basis

    path = os.path.join(mojito_folder, "catalogues", "wdwd_cat_mojito_lite_processed.hdf5")
    with h5py.File(path, "r") as handle:
        binaries = handle["Binaries"]
        frequencies_hz = binaries["GW22FrequencySSBFrame"][:]
        rows = np.flatnonzero(
            (frequencies_hz >= band_edges_hz[1]) & (frequencies_hz < band_edges_hz[-2])
        )
        keys = (
            "Amplitude",
            "GW22FrequencySSBFrame",
            "GW22FrequencyDerivativeSourceFrame",
            "TrueAnomaly",
            "InclinationAngle",
            "RightAscension",
            "Declination",
            "PolarisationAngle",
        )
        entry = {key: binaries[key][:][rows] for key in keys}
    return rows, np.atleast_2d(gb_catalogue_to_sampling_basis(entry))


def wrap(values, period):
    """Wrap into [-period / 2, period / 2)."""
    return (values + 0.5 * period) % period - 0.5 * period


def offsets_from_truth(samples: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, float]:
    """Sample minus truth per parameter, with (phi0, psi) moved to the mode of the truth.

    (phi0 + pi, psi + pi/2) gives the same waveform, so samples nearer the other mode are moved.
    Returns the offsets and the fraction of samples moved.
    """
    offsets = samples - truth
    psi_offset = wrap(offsets[:, 5], np.pi)
    moved = np.abs(psi_offset) > 0.25 * np.pi
    psi_offset[moved] -= np.sign(psi_offset[moved]) * 0.5 * np.pi
    offsets[:, 5] = psi_offset
    offsets[:, 3] = wrap(offsets[:, 3] + np.where(moved, np.pi, 0.0), 2.0 * np.pi)
    offsets[:, 6] = wrap(offsets[:, 6], 2.0 * np.pi)
    return offsets, float(moved.mean()) if moved.size else 0.0


def integrated_time(series: np.ndarray, window_c: float = 5.0) -> float:
    """Integrated autocorrelation time (in iterations) of (iterations, walkers), walker-averaged ACF."""
    steps = series.shape[0]
    if steps < 4:
        return float("nan")
    centred = series - series.mean(axis=0)
    size = 2 ** int(np.ceil(np.log2(2 * steps)))
    spectrum = np.fft.rfft(centred, n=size, axis=0)
    acf = np.fft.irfft(spectrum * np.conjugate(spectrum), n=size, axis=0)[:steps].real
    acf = acf.mean(axis=1)
    if acf[0] <= 0.0:
        return float("nan")
    acf /= acf[0]
    # * Sokal's automatic window: the first lag m with m >= c * tau(m).
    taus = 2.0 * np.cumsum(acf) - 1.0
    lags = np.arange(steps)
    inside = lags < window_c * taus
    window = int(np.argmin(inside)) if not inside.all() else steps - 1
    return float(max(taus[window], 1.0))


@dataclass
class SourceStats:
    mean: np.ndarray
    sd: np.ndarray
    tau: np.ndarray
    n_eff: np.ndarray
    alive_fraction: float
    moved_fraction: float
    samples: np.ndarray  # offsets from the truth, alive cold-chain samples after the discard
    thinned: np.ndarray


def source_stats(run: RunData, source: int, truth: np.ndarray, first: int) -> SourceStats:
    coords = run.matched[first:, :, source]
    alive = run.found[first:, :, source]
    steps, walkers = alive.shape
    offsets, _ = offsets_from_truth(np.nan_to_num(coords.reshape(-1, coords.shape[-1])), truth)
    offsets = offsets.reshape(steps, walkers, -1)
    moved = float(np.mean(np.abs(wrap(coords[..., 5][alive] - truth[5], np.pi)) > 0.25 * np.pi))
    # ? The ACF needs whole walkers: walkers whose source died somewhere are left out of tau.
    whole = alive.all(axis=0)
    taus = np.array(
        [
            integrated_time(offsets[:, whole, index]) if whole.any() else np.nan
            for index in range(offsets.shape[-1])
        ]
    )
    samples = offsets[alive]
    n_samples = samples.shape[0]
    thin = int(np.ceil(np.nanmax(taus))) if np.isfinite(taus).any() else 1
    thinned = offsets[::thin][alive[::thin]]
    return SourceStats(
        mean=samples.mean(axis=0),
        sd=samples.std(axis=0, ddof=1),
        tau=taus,
        n_eff=n_samples / np.where(np.isfinite(taus), taus, 1.0),
        alive_fraction=float(alive.mean()),
        moved_fraction=moved,
        samples=samples,
        thinned=thinned,
    )


def compare(stats_a: SourceStats, stats_b: SourceStats) -> dict:
    """Run B against run A, per parameter, in units of run A's standard deviation."""
    bias = (stats_b.mean - stats_a.mean) / stats_a.sd
    ratio = stats_b.sd / stats_a.sd
    bias_noise = np.sqrt(1.0 / stats_a.n_eff + ratio**2 / stats_b.n_eff)
    ratio_noise = ratio * np.sqrt(0.5 / stats_a.n_eff + 0.5 / stats_b.n_eff)
    ks = np.array(
        [
            ks_2samp(stats_a.thinned[:, index], stats_b.thinned[:, index]).statistic
            for index in range(len(PARAMETERS))
        ]
    )
    n_a, n_b = stats_a.thinned.shape[0], stats_b.thinned.shape[0]
    ks_level = KS_LEVEL_COEFFICIENT * np.sqrt((n_a + n_b) / (n_a * n_b))
    return dict(
        bias=bias,
        bias_noise=bias_noise,
        sd_ratio=ratio,
        sd_ratio_noise=ratio_noise,
        ks=ks,
        ks_level=float(ks_level),
    )


def iteration_seconds(run: RunData) -> np.ndarray:
    starts = np.array([stamp.timestamp() for stamp in run.iteration_starts])
    return np.diff(starts)


def acceptance(run: RunData, first: int, bands: np.ndarray) -> dict:
    """In-model and RJ acceptance of the cold chain and swap acceptance per temperature pair."""

    def ratio(accepted, proposed):
        accepted_sum, proposed_sum = float(np.sum(accepted)), float(np.sum(proposed))
        return accepted_sum / proposed_sum if proposed_sum > 0 else float("nan")

    counters = run.counters
    result = {}
    if "band_num_accepted" in counters:
        result["in_model_cold"] = ratio(
            counters["band_num_accepted"][first:, bands, 0],
            counters["band_num_proposed"][first:, bands, 0],
        )
    if "band_num_accepted_rj" in counters:
        result["rj_cold"] = ratio(
            counters["band_num_accepted_rj"][first:, :, 0],
            counters["band_num_proposed_rj"][first:, :, 0],
        )
    if "band_swaps_accepted" in counters:
        accepted = counters["band_swaps_accepted"][first:, bands].sum(axis=(0, 1))
        proposed = counters["band_swaps_proposed"][first:, bands].sum(axis=(0, 1))
        result["swaps_per_pair"] = np.where(
            proposed > 0, accepted / np.maximum(proposed, 1), np.nan
        )
    return result


def style_axes(axes):
    axes.set_facecolor(SURFACE)
    axes.grid(True, color=GRID, linewidth=0.6)
    axes.tick_params(colors=INK2, labelsize=8)
    axes.xaxis.label.set_size(9)
    axes.yaxis.label.set_size(9)
    for spine in axes.spines.values():
        spine.set_color(GRID)


def figure_health(runs, alive_injected, swap_rates, out):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), facecolor=SURFACE)
    for run, color, alive_count, swaps in zip(runs, RUN_COLORS, alive_injected, swap_rates):
        axes[0].plot(alive_count, "o-", color=color, ms=3, lw=1.5, label=run.name)
        axes[1].plot(
            run.log_like[:, 0].mean(axis=1) - run.log_like[0, 0].mean(),
            "o-",
            color=color,
            ms=3,
            lw=1.5,
        )
        if swaps is not None:
            axes[2].plot(np.arange(1, len(swaps) + 1), swaps, "o-", color=color, ms=4, lw=1.5)
    axes[0].set_title("injected sources alive (fewest over cold walkers)", fontsize=10, color=INK)
    axes[1].set_title(
        r"cold-chain $\ln L$ (walker mean) minus its first value", fontsize=10, color=INK
    )
    axes[2].set_title(
        "swap acceptance per temperature pair (bands with sources)", fontsize=10, color=INK
    )
    axes[0].set_xlabel("iteration")
    axes[1].set_xlabel("iteration")
    axes[2].set_xlabel("temperature pair (from the cold end)")
    for axis in axes:
        style_axes(axis)
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def figure_speed_memory(runs, out):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), facecolor=SURFACE)
    for run, color in zip(runs, RUN_COLORS):
        seconds = iteration_seconds(run)
        if seconds.size:
            axes[0].plot(
                np.arange(1, seconds.size + 1),
                seconds / 60.0,
                "o-",
                color=color,
                ms=3,
                lw=1.5,
                label=run.name,
            )
        if run.gpu_memory_mib.size:
            axes[1].plot(run.gpu_time_s / 3600.0, run.gpu_memory_mib / 1024.0, color=color, lw=1.5)
    axes[0].set_title("wall time per iteration (min)", fontsize=10, color=INK)
    axes[0].set_xlabel("iteration")
    axes[1].set_title("GPU memory in use (GiB; the A100 holds 40)", fontsize=10, color=INK)
    axes[1].set_xlabel("hours since the run started")
    for axis in axes:
        style_axes(axis)
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def figure_per_parameter(
    values, noises, reference, ylabel, title, runs_labels, frequencies_mhz, out
):
    """2 x 4 panels, one per parameter: one dot per source (by f0), noise as error bars."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True, facecolor=SURFACE)
    for index, axis in enumerate(axes.flat):
        for rank, (series, color, label) in enumerate(
            zip(values, RUN_COLORS[: len(values)], runs_labels)
        ):
            noise = noises.get(label) if noises else None
            # * Runs side by side (0.06 mHz apart) so equal values stay visible.
            dodge_mhz = 0.06 * (rank - 0.5 * (len(values) - 1))
            axis.errorbar(
                frequencies_mhz + dodge_mhz,
                series[:, index],
                yerr=None if noise is None else noise[:, index],
                fmt="o",
                ms=4,
                color=color,
                ecolor=color,
                elinewidth=0.8,
                alpha=0.9,
                label=label,
            )
        for level in reference:
            axis.axhline(level, color=MUTED, lw=0.8, ls="--" if level else "-")
        axis.set_title(LABELS[index], fontsize=11, color=INK)
        style_axes(axis)
    for axis in axes[1]:
        axis.set_xlabel("f0 (mHz)")
    for axis in axes[:, 0]:
        axis.set_ylabel(ylabel)
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle(title, fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def figure_traces(runs, source, truth_sd, out):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(16, 6.5), sharex=True, facecolor=SURFACE)
    for run, color in zip(runs, RUN_COLORS):
        coords = run.matched[:, :, source]
        alive = run.found[:, :, source]
        offsets, _ = offsets_from_truth(
            np.nan_to_num(coords.reshape(-1, coords.shape[-1])), run.truth[source]
        )
        offsets = offsets.reshape(coords.shape) / truth_sd
        offsets[~alive] = np.nan
        for index, axis in enumerate(axes.flat):
            axis.plot(offsets[:, :, index], color=color, lw=0.4, alpha=0.35)
    for index, axis in enumerate(axes.flat):
        axis.axhline(0.0, color=INK, lw=0.8)
        axis.set_title(
            LABELS[index] + r" offset from the injection / $\sigma_A$", fontsize=9, color=INK
        )
        style_axes(axis)
    for axis in axes[1]:
        axis.set_xlabel("iteration")
    handles = [plt.Line2D([], [], color=color, lw=2) for color in RUN_COLORS[: len(runs)]]
    axes[0, 0].legend(handles, [run.name for run in runs], fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=90)
    plt.close(fig)


def figure_corner(stats_pair, names, slot_label, scale, out):
    import cantuccio
    import matplotlib.pyplot as plt

    columns = [label.replace("$", "") for label in LABELS]
    columns = [rf"$\Delta {column}/\sigma_A$" for column in columns]
    data = [dict(zip(columns, (stats.samples / scale).T)) for stats in stats_pair]
    fig, axes = cantuccio.cornerplot(
        data,
        columns=columns,
        truths=dict(zip(columns, np.zeros(len(columns)))),
        labels=list(names),
        colors=list(RUN_COLORS[: len(data)]),
        contour_levels=(0.68, 0.90),
        diag_mode="kde",
        offdiag_mode="kde",
        kde_kwargs={"fast": True},
        truth_kwargs={"color": INK, "ls": "-", "lw": 0.9, "alpha": 0.8, "label": "injected value"},
        legend_kwargs={"fontsize": 15},
        n_ticks=3,
        diagonal_ticks=True,
    )
    with plt.style.context(cantuccio.visuals.get_stylefile()):
        fig.text(0.97, 0.985, slot_label, ha="right", va="top", fontsize=20)
        fig.text(
            0.97,
            0.955,
            r"offsets from the injected values in units of run A's posterior sd; contours 68\% and 90\%",
            ha="right",
            va="top",
            fontsize=14,
            color=MUTED,
        )
        fig.savefig(out, dpi=80)
    plt.close(fig)


def truncated(run: RunData, iterations: int) -> RunData:
    """The run cut to its first ``iterations``, so both runs can be compared at equal length."""
    cut = copy.copy(run)
    for name in ("coords", "alive", "log_like", "matched", "found", "unmatched"):
        setattr(cut, name, getattr(run, name)[:iterations])
    cut.iterations = iterations
    return cut


def walker_spread(run: RunData, reference_iteration: int) -> np.ndarray:
    """Spread of the walkers per iteration and parameter, over its value at the reference iteration.

    Median over the injected binaries; (iterations, parameters). A run that has left its start
    point and mixed has a flat spread.
    """
    offsets = run.matched - run.truth[None, None]
    # * Angles about the truth, so a walker across the wrap does not inflate the spread.
    offsets[..., 3] = wrap(offsets[..., 3], 2.0 * np.pi)
    offsets[..., 5] = wrap(offsets[..., 5], np.pi)
    offsets[..., 6] = wrap(offsets[..., 6], 2.0 * np.pi)
    spread = np.nanstd(offsets, axis=1)
    return np.nanmedian(spread / spread[reference_iteration], axis=1)


def figure_convergence(growth, names, common, out):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(16, 6.5), sharex=True, facecolor=SURFACE)
    for index, axis in enumerate(axes.flat):
        for series, color, label in zip(growth, RUN_COLORS, names):
            axis.plot(
                np.arange(1, series.shape[0] + 1),
                series[:, index],
                color=color,
                lw=1.8,
                label=label,
            )
        axis.axvline(common, color=MUTED, lw=0.8, ls="--")
        axis.axhline(1.0, color=MUTED, lw=0.8)
        axis.set_title(LABELS[index], fontsize=11, color=INK)
        style_axes(axis)
    for axis in axes[1]:
        axis.set_xlabel("iteration")
    for axis in axes[:, 0]:
        axis.set_ylabel(f"walker spread / spread at iteration {common}")
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle(
        "spread of the cold walkers about the injected values, median over the binaries; "
        "dashed: the common iteration count",
        fontsize=12,
        color=INK,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", nargs=2, required=True, help="run folders A (reference) and B")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--discard", type=float, default=0.25, help="fraction of iterations dropped"
    )
    parser.add_argument(
        "--mojito",
        default=os.environ.get("MOJITO_FOLDER", "/project/lisa_nlddpc/Data/mojito_light"),
    )
    parser.add_argument("--no-corners", action="store_true", help="skip the per-source corners")
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    os.makedirs(os.path.join(args.output, "traces"), exist_ok=True)
    os.makedirs(os.path.join(args.output, "corners"), exist_ok=True)

    runs = [load_run(folder) for folder in args.runs]
    if not np.array_equal(runs[0].band_edges_hz, runs[1].band_edges_hz):
        raise ValueError("the two runs have different band edges")
    rows, truths = injected_sources(runs[0].band_edges_hz, args.mojito)
    n_sources = rows.size
    for run in runs:
        run.truth = truths
        match_sources(run, truths[:, 1])
    names = [run.name for run in runs]
    order = np.argsort(truths[:, 1])
    frequencies_mhz = truths[:, 1]
    band_of_source = (
        np.searchsorted(runs[0].band_edges_hz, frequencies_mhz * 1e-3, side="right") - 1
    )
    bands_with_sources = np.unique(band_of_source)

    firsts = [int(np.floor(args.discard * run.iterations)) for run in runs]
    stats = [
        [source_stats(run, slot, truths[slot], first) for slot in range(n_sources)]
        for run, first in zip(runs, firsts)
    ]
    # ! The runs share a wall clock, not an iteration count, and their walkers still spread from
    # ! the common start: the two posteriors are compared at the iteration count both reached.
    common = min(run.iterations for run in runs)
    first_common = int(np.floor(args.discard * common))
    stats_common = [
        [
            source_stats(truncated(run, common), slot, truths[slot], first_common)
            for slot in range(n_sources)
        ]
        for run in runs
    ]
    comparisons = [
        compare(stats_common[0][slot], stats_common[1][slot]) for slot in range(n_sources)
    ]
    growth = [walker_spread(run, common - 1) for run in runs]
    figure_convergence(growth, names, common, os.path.join(args.output, "convergence.png"))

    # * Health: injected slots alive, births into the free slots, ln L rebuilds, store windows.
    alive_injected = [run.found.sum(axis=2).min(axis=1) for run in runs]
    births = [run.unmatched for run in runs]
    accept = [acceptance(run, first, bands_with_sources) for run, first in zip(runs, firsts)]
    swap_rates = [entry.get("swaps_per_pair") for entry in accept]
    figure_health(runs, alive_injected, swap_rates, os.path.join(args.output, "health.png"))
    figure_speed_memory(runs, os.path.join(args.output, "speed_memory.png"))

    means = [np.array([entry.mean for entry in run_stats]) for run_stats in stats]
    sds = [np.array([entry.sd for entry in run_stats]) for run_stats in stats]
    n_effs = [np.array([entry.n_eff for entry in run_stats]) for run_stats in stats]
    z_truth = [mean / sd for mean, sd in zip(means, sds)]
    figure_per_parameter(
        z_truth,
        {name: 1.0 / np.sqrt(n_eff) for name, n_eff in zip(names, n_effs)},
        (0.0, -2.0, 2.0),
        r"(mean - injected) / $\sigma$",
        "posterior mean offsets from the injected values (each run in its own sd)",
        names,
        frequencies_mhz,
        os.path.join(args.output, "truth_offsets.png"),
    )
    bias = np.array([entry["bias"] for entry in comparisons])
    bias_noise = np.array([entry["bias_noise"] for entry in comparisons])
    ratio = np.array([entry["sd_ratio"] for entry in comparisons])
    ratio_noise = np.array([entry["sd_ratio_noise"] for entry in comparisons])
    ks = np.array([entry["ks"] for entry in comparisons])
    ks_level = np.array([entry["ks_level"] for entry in comparisons])
    label_ab = f"{names[1]} - {names[0]}"
    figure_per_parameter(
        [bias],
        {label_ab: bias_noise},
        (0.0,),
        r"$(\mu_B - \mu_A)/\sigma_A$",
        f"difference of the posterior means, {label_ab}; bars = MCMC noise of the difference",
        [label_ab],
        frequencies_mhz,
        os.path.join(args.output, "comparison_bias.png"),
    )
    figure_per_parameter(
        [ratio],
        {label_ab: ratio_noise},
        (1.0,),
        r"$\sigma_B/\sigma_A$",
        f"ratio of the posterior sd, {names[1]} / {names[0]}; bars = MCMC noise of the ratio",
        [label_ab],
        frequencies_mhz,
        os.path.join(args.output, "comparison_sd_ratio.png"),
    )
    figure_per_parameter(
        [ks / ks_level[:, None]],
        None,
        (1.0,),
        "KS D / 95 % level",
        f"two-sample KS statistic of {names[1]} against {names[0]}, over its 95 % level for "
        "thinned samples",
        [label_ab],
        frequencies_mhz,
        os.path.join(args.output, "comparison_ks.png"),
    )

    for rank, slot in enumerate(order):
        scale = sds[0][slot]
        tag = f"gb{rank:02d}_{frequencies_mhz[slot]:.4f}mHz"
        figure_traces(runs, slot, scale, os.path.join(args.output, "traces", tag + ".png"))
        if not args.no_corners:
            figure_corner(
                [stats_common[0][slot], stats_common[1][slot]],
                names,
                f"{tag}: catalogue row {rows[slot]}, first {common} iterations of each run",
                stats_common[0][slot].sd,
                os.path.join(args.output, "corners", tag + ".png"),
            )

    convergence = {"common_iterations": common, "first_common": first_common}
    for run, run_growth, z in zip(runs, growth, z_truth):
        # * Growth of the spread over the last quarter of the common span; 1 when it has settled.
        quarter = int(np.floor(0.75 * common)) - 1
        convergence[run.name] = {
            "spread_growth_last_quarter": (run_growth[common - 1] / run_growth[quarter]).tolist(),
            "spread_at_end_over_common": run_growth[-1].tolist(),
            "truth_offset_dispersion": np.nanstd(z, axis=0).tolist(),
        }
    write_report(
        args,
        runs,
        firsts,
        rows,
        truths,
        order,
        stats,
        comparisons,
        accept,
        births,
        alive_injected,
        convergence,
    )


def write_report(
    args,
    runs,
    firsts,
    rows,
    truths,
    order,
    stats,
    comparisons,
    accept,
    births,
    alive_injected,
    convergence,
):
    names = [run.name for run in runs]
    common = convergence["common_iterations"]
    lines = [
        f"DFFR validation item 5: {names[0]} (A, reference) against {names[1]} (B). Cold chain, "
        f"first {args.discard:.0%} of the iterations dropped. Offsets of angles use the mode "
        "(phi0, psi) of the injection. Truth offsets use each run's iterations; the comparison of "
        f"the two runs and the corners use the first {common} iterations of each, dropping "
        f"{convergence['first_common']}.",
        "",
        "Convergence, per parameter (" + ", ".join(PARAMETERS) + ")",
    ]
    for name in names:
        entry = convergence[name]
        lines += [
            f"  {name}: walker spread growth over the last quarter of the first {common} "
            "iterations "
            + " ".join(f"{value:.3f}" for value in entry["spread_growth_last_quarter"]),
            "    spread at the last iteration over the spread at iteration "
            f"{common}: "
            + " ".join(f"{value:.3f}" for value in entry["spread_at_end_over_common"]),
            "    dispersion of (mean - injected)/sd over the binaries (1 when the posteriors are "
            "converged): " + " ".join(f"{value:.2f}" for value in entry["truth_offset_dispersion"]),
        ]
    lines += ["", "Run health and speed"]
    summary = {"runs": {}, "sources": [], "convergence": convergence}
    for run, first, entry, birth, alive_count in zip(runs, firsts, accept, births, alive_injected):
        seconds = iteration_seconds(run)
        drift = np.asarray(run.drift_lnl)
        outside = np.asarray(run.rows_outside) if run.rows_outside else np.zeros((0, 2))
        record = {
            "iterations": run.iterations,
            "discarded": first,
            "iteration_minutes_median": float(np.median(seconds) / 60.0) if seconds.size else None,
            "proposal_minutes_median": (
                float(np.median(run.proposal_s) / 60.0) if run.proposal_s else None
            ),
            "gpu_memory_peak_gib": (
                float(run.gpu_memory_mib.max() / 1024.0) if run.gpu_memory_mib.size else None
            ),
            # * Median seconds per iteration of the move's stages, largest first.
            "stage_seconds_median": {
                name: float(np.median([entry.get(name, 0.0) for entry in run.stage_s]))
                for name in sorted(
                    {name for entry in run.stage_s for name in entry},
                    key=lambda name: -np.median([entry.get(name, 0.0) for entry in run.stage_s]),
                )
            },
            "ll_rebuilds": int(drift.size),
            "ll_drift_max": float(drift.max()) if drift.size else 0.0,
            "rows_outside_store_window": [int(outside[:, 0].sum()), int(outside[:, 1].sum())],
            "fewest_injected_alive": int(alive_count.min()),
            "births_cold_max": int(birth.max()),
            "in_model_acceptance_cold": entry.get("in_model_cold"),
            "rj_acceptance_cold": entry.get("rj_cold"),
            "swap_acceptance_per_pair": (
                None if entry.get("swaps_per_pair") is None else entry["swaps_per_pair"].tolist()
            ),
        }
        summary["runs"][run.name] = record
        lines += [
            f"  {run.name}: {run.iterations} iterations ({first} dropped); median "
            f"{record['iteration_minutes_median']} min per iteration (proposal "
            f"{record['proposal_minutes_median']} min); GPU memory peak {record['gpu_memory_peak_gib']} GiB",
            "    stages (median s per iteration): "
            + ", ".join(
                f"{name} {seconds:.1f}"
                for name, seconds in list(record["stage_seconds_median"].items())[:10]
            ),
            f"    ln L rebuilt after drift {record['ll_rebuilds']} times, largest drift "
            f"{record['ll_drift_max']:.3g}; rows outside their store window "
            f"{record['rows_outside_store_window'][0]} of {record['rows_outside_store_window'][1]}",
            f"    injected sources alive: at least {record['fewest_injected_alive']} of {len(rows)} in every "
            f"cold walker; live leaves matched to no injected source (most in one walker) {record['births_cold_max']}",
            f"    acceptance (cold, bands with sources): in-model {record['in_model_acceptance_cold']}, "
            f"RJ {record['rj_acceptance_cold']}",
        ]
    lines += [
        "",
        "Per source (by f0): z = (mean - injected)/sd per run over all its iterations; at the "
        "common iteration count: B - A in sd of A (+- MCMC noise), sd ratio B/A, KS D over its "
        "95 % level",
    ]
    for slot in order:
        stats_a, stats_b, comparison = stats[0][slot], stats[1][slot], comparisons[slot]
        lines.append(
            f"  row {rows[slot]}  f0 {truths[slot, 1]:.5f} mHz  alive A {stats_a.alive_fraction:.3f} "
            f"B {stats_b.alive_fraction:.3f}  (phi0, psi) moved A {stats_a.moved_fraction:.3f} B "
            f"{stats_b.moved_fraction:.3f}  N_eff (median over parameters) A "
            f"{np.median(stats_a.n_eff):.0f} B {np.median(stats_b.n_eff):.0f}"
        )
        for index, parameter in enumerate(PARAMETERS):
            lines.append(
                f"    {parameter:9s} z A {stats_a.mean[index] / stats_a.sd[index]:+7.3f}  z B "
                f"{stats_b.mean[index] / stats_b.sd[index]:+7.3f}  B-A {comparison['bias'][index]:+7.3f} "
                f"+- {comparison['bias_noise'][index]:.3f}  sd B/A {comparison['sd_ratio'][index]:.4f} "
                f"+- {comparison['sd_ratio_noise'][index]:.4f}  KS {comparison['ks'][index] / comparison['ks_level']:.2f}"
            )
        summary["sources"].append(
            {
                "catalogue_row": int(rows[slot]),
                "source_index": int(slot),
                "injected": truths[slot].tolist(),
                "runs": {
                    run.name: {
                        "mean_offset": entry.mean.tolist(),
                        "sd": entry.sd.tolist(),
                        "tau": entry.tau.tolist(),
                        "n_eff": entry.n_eff.tolist(),
                        "alive_fraction": entry.alive_fraction,
                    }
                    for run, entry in zip(runs, (stats_a, stats_b))
                },
                "comparison": {
                    key: np.asarray(value).tolist() for key, value in comparison.items()
                },
            }
        )
    with open(os.path.join(args.output, "report.txt"), "w") as handle:
        handle.write("\n".join(lines) + "\n")
    with open(os.path.join(args.output, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=1)
    print("\n".join(lines[:20]))


if __name__ == "__main__":
    main()
