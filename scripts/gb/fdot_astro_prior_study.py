"""Catalogue study of fdot_astro = fdot_catalogue - fdot_gr(f0, Mc) for GBs.

The mojito wdwd catalogue is an INTERACTING DWD population: its fdot
(``GW22FrequencyDerivativeSourceFrame``) includes tides/mass transfer on
top of the GW-driven chirp, so ``fdot != get_fdot(f0, Mc)`` and ~half the
sources have fdot <= 0 (which the pure chirp-mass sampling basis cannot
represent). This script supports the (f0, Mc, fdot_astro) sampling-basis
extension:

1. computes fdot_gr from the catalogue's mass-based chirp mass
   (``ChirpMassSSBFrame``) via :func:`gbgpu.utils.utility.get_fdot` and the
   residual fdot_astro = fdot - fdot_gr;
2. renders the visual analysis (fdot / fdot_gr / fdot_astro vs f0,
   distributions, frequency scaling, sign fractions);
3. evaluates candidate sampled variables (raw fdot_astro vs frequency-scaled
   u = fdot_astro / envelope(f0)) and candidate 1-D priors (uniform box,
   Laplace, Student-t, asinh-normal), with per-frequency-bin catalogue
   coverage and "unphysicality" metrics (prior mass that drives the TOTAL
   fdot outside the empirical get_fdot_mojito envelopes, or that dominates
   fdot_gr);
4. prototypes + validates the 9-column Eryn ``TransformContainer``
   (sampling (..., Mc, ..., fdot_astro) -> physical (..., fdot, fddot=0))
   against catalogue truth, including every fdot <= 0 row, plus the
   deepcopy/pickle rule.

Usage::

    python fdot_astro_prior_study.py

Environment: MOJITO_DATA_PATH [~/.mojito_cache/brickmarket/mojito_light_v1_0_0/],
N_SUBSAMPLE [2000000; 0 = full 15.5M], OUT_DIR [scripts/gb/fdot_astro_figs/],
SEED [42].
"""

from __future__ import annotations

import copy
import os
import pickle

import h5py
import numpy as np
import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["text.usetex"] = False
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy import stats

from gbgpu.utils.utility import get_chirp_mass_from_f_fdot, get_fdot
from lisatools.globalfit.priors.gbpriors import get_fdot_mojito
from lisatools.globalfit.stock.erebor.transforms import make_gb_transform_container

CAT_PATH = os.path.join(
    os.environ.get(
        "MOJITO_DATA_PATH",
        os.path.expanduser("~/.mojito_cache/brickmarket/mojito_light_v1_0_0/"),
    ),
    "catalogues", "wdwd_cat_mojito_lite_processed.hdf5",
)
N_SUBSAMPLE = int(os.environ.get("N_SUBSAMPLE", 2_000_000))
OUT_DIR = os.environ.get(
    "OUT_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "fdot_astro_figs")
)
SEED = int(os.environ.get("SEED", 42))

MC_LIMS = (0.001, 1.0)  # stock GBNoFgGBSettings.m_chirp_lims
F0_LIMS_HZ = (1e-4, 1e-2)  # catalogue span == stock full band
N_F0_BINS = 24
MIN_BIN_COUNT = 200  # bins with fewer sources are excluded from quantile stats
REL_EPS = 1e-6  # |fdot_astro|/fdot_gr below this = "detached" (fdot == fdot_gr)
# fdot indistinguishable from 0 below ~1/(2*pi*Tobs^2) (order-of-magnitude,
# SNR~O(10) Fisher scale) -- context lines for how much of fdot_astro is
# actually measurable
FDOT_RES_90D = 1.0 / (2 * np.pi * (90 * 86400.0) ** 2)
FDOT_RES_4YR = 1.0 / (2 * np.pi * (4 * 31558149.76) ** 2)

# fixed categorical assignment (CVD-safe, linestyles as secondary encoding)
C_BLUE, C_ORANGE, C_PURPLE, C_GRAY = "#1f77b4", "#ff7f0e", "#9467bd", "#555555"


# ---------------------------------------------------------------------------
# 9-column transform: the REAL library factory (ratio basis, user-selected
# prior design: r = fdot_astro / fdot_gr ~ U[-M, M]; fdot = fdot_gr*(1+r))
# ---------------------------------------------------------------------------

RATIO_M_DEFAULT = 5.0  # the "something like [-5, 5]" box; knob in Stage 2


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def signed_log_edges(linthresh, vmax, per_decade=6):
    """Symmetric symlog bin edges: +/- decades plus one linear bin at zero."""
    n_dec = int(np.ceil(np.log10(vmax / linthresh)))
    pos = np.logspace(np.log10(linthresh), np.log10(linthresh) + n_dec,
                      n_dec * per_decade + 1)
    return np.concatenate([-pos[::-1], pos])


def symlog_hist2d(ax, x, y, x_edges, linthresh, vmax, per_decade=6):
    y_edges = signed_log_edges(linthresh, vmax, per_decade)
    H, _, _ = np.histogram2d(x, np.clip(y, y_edges[0], y_edges[-1]),
                             bins=[x_edges, y_edges])
    pcm = ax.pcolormesh(x_edges, y_edges, H.T, norm=LogNorm(vmin=1),
                        cmap="viridis", rasterized=True)
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=linthresh)
    return pcm


def binned_stat(x, y, edges, fn, min_count=MIN_BIN_COUNT):
    idx = np.digitize(x, edges) - 1
    out = np.full(len(edges) - 1, np.nan)
    for i in range(len(edges) - 1):
        m = idx == i
        if np.count_nonzero(m) >= min_count:
            out[i] = fn(y[m])
    return out


def powerlaw_fit(f_centers, values):
    """log10(values) = log10(C) + p*log10(f); returns (C, p)."""
    good = np.isfinite(values) & (values > 0)
    p, logC = np.polyfit(np.log10(f_centers[good]), np.log10(values[good]), 1)
    return 10.0 ** logC, p


# ---------------------------------------------------------------------------
# catalogue load + core computation
# ---------------------------------------------------------------------------

def load_catalogue(rng):
    with h5py.File(CAT_PATH, "r") as f:
        B = f["Binaries"]
        n_total = B["GW22FrequencySSBFrame"].shape[0]
        z_max = B["Redshift"][:].max()
        assert z_max == 0.0, f"nonzero Redshift ({z_max}); SSB==source frame assumed"
        if 0 < N_SUBSAMPLE < n_total:
            keep = np.sort(rng.choice(n_total, size=N_SUBSAMPLE, replace=False))
        else:
            keep = slice(None)
        cat = {}
        for key in ["GW22FrequencySSBFrame", "ChirpMassSSBFrame",
                    "GW22FrequencyDerivativeSourceFrame"]:
            cat[key] = B[key][:][keep]
    cat["n_total"] = n_total
    cat["keep"] = keep
    return cat


def load_validation_columns(keep_idx):
    """Extra columns for the transform validation, subset only."""
    cols = {}
    with h5py.File(CAT_PATH, "r") as f:
        B = f["Binaries"]
        for key in ["Amplitude", "TrueAnomaly", "InclinationAngle",
                    "PolarisationAngle", "RightAscension", "Declination",
                    "GW22FrequencySSBFrame", "ChirpMassSSBFrame",
                    "GW22FrequencyDerivativeSourceFrame"]:
            cols[key] = B[key][:][keep_idx]
    return cols


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def fig1_fdot_vs_f0(f0, fdot, fdot_gr, f_edges, out):
    fig, ax = plt.subplots(figsize=(9, 6))
    pcm = symlog_hist2d(ax, f0, fdot, f_edges, linthresh=1e-24, vmax=1e-13)
    fig.colorbar(pcm, ax=ax, label="sources / bin")
    ff = np.logspace(-4, -2, 200)
    ax.plot(ff, get_fdot_mojito(ff, "+"), color=C_ORANGE, lw=2,
            label=r"get_fdot_mojito '+' env")
    ax.plot(ff, get_fdot_mojito(ff, "-"), color=C_ORANGE, lw=2, ls="--",
            label=r"get_fdot_mojito '-' env")
    ax.plot(ff, get_fdot(f=ff, Mc=np.full_like(ff, MC_LIMS[1])), color=C_BLUE,
            lw=2, label=rf"fdot_gr, Mc={MC_LIMS[1]}")
    ax.plot(ff, get_fdot(f=ff, Mc=np.full_like(ff, MC_LIMS[0])), color=C_BLUE,
            lw=2, ls="--", label=rf"fdot_gr, Mc={MC_LIMS[0]}")
    ax.set_xlabel("f0 [Hz]")
    ax.set_ylabel("fdot (catalogue) [Hz/s]")
    ax.set_title("Catalogue fdot vs f0 with empirical + GR envelopes")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def fig2_fdot_astro_vs_f0(f0, fdot_astro, f_edges, env_fits, out):
    fig, ax = plt.subplots(figsize=(9, 6))
    pcm = symlog_hist2d(ax, f0, fdot_astro, f_edges, linthresh=1e-24, vmax=1e-13)
    fig.colorbar(pcm, ax=ax, label="sources / bin")
    ff = np.logspace(-4, -2, 200)
    ax.plot(ff, -np.abs(get_fdot_mojito(ff, "-")), color=C_ORANGE, lw=2, ls="--",
            label="-|mojito '-' env|")
    ax.plot(ff, np.abs(get_fdot_mojito(ff, "-")), color=C_ORANGE, lw=2,
            label="+|mojito '-' env|")
    C_fit, p_fit = env_fits["powerlaw"]
    ax.plot(ff, C_fit * ff ** p_fit, color=C_PURPLE, lw=2,
            label=rf"q90 |fdot_astro| fit: {C_fit:.2e} f^{{{p_fit:.2f}}}")
    ax.plot(ff, -C_fit * ff ** p_fit, color=C_PURPLE, lw=2, ls="--")
    for res, lab in [(FDOT_RES_90D, "fdot resolution ~90 d"),
                     (FDOT_RES_4YR, "fdot resolution ~4 yr")]:
        ax.axhline(res, color=C_GRAY, lw=1, ls=":")
        ax.axhline(-res, color=C_GRAY, lw=1, ls=":")
        ax.text(1.1e-4, res * 1.5, lab, color=C_GRAY, fontsize=8)
    ax.set_xlabel("f0 [Hz]")
    ax.set_ylabel("fdot_astro = fdot - fdot_gr [Hz/s]")
    ax.set_title("fdot_astro vs f0 (interacting residual)")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def fig3_distributions(fdot_astro, ratio, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    edges = signed_log_edges(1e-26, 1e-13, per_decade=4)
    axes[0].hist(np.clip(fdot_astro, edges[0], edges[-1]), bins=edges,
                 color=C_BLUE, histtype="stepfilled", alpha=0.8)
    for res in [FDOT_RES_90D, FDOT_RES_4YR]:
        axes[0].axvline(res, color=C_GRAY, lw=1, ls=":")
        axes[0].axvline(-res, color=C_GRAY, lw=1, ls=":")
    axes[0].set_xscale("symlog", linthresh=1e-26)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("fdot_astro [Hz/s]  (dotted: 90d / 4yr fdot resolution)")
    axes[0].set_ylabel("sources / bin")
    axes[0].set_title("fdot_astro (all f0)")

    r_edges = signed_log_edges(1e-4, 1e6, per_decade=4)
    axes[1].hist(np.clip(ratio, r_edges[0], r_edges[-1]), bins=r_edges,
                 color=C_PURPLE, histtype="stepfilled", alpha=0.8)
    axes[1].axvline(1.0, color=C_ORANGE, lw=1.5, ls="--")
    axes[1].axvline(-1.0, color=C_ORANGE, lw=1.5, ls="--")
    axes[1].set_xscale("symlog", linthresh=1e-4)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("fdot_astro / fdot_gr")
    axes[1].set_title("astro-to-GR ratio (|ratio|>1 = astro dominates)")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def fig4_binned(f0, fdot, fdot_astro, ratio, f_edges, out):
    f_cent = np.sqrt(f_edges[:-1] * f_edges[1:])
    neg = fdot_astro < 0
    frac_fdot_neg = binned_stat(f0, (fdot < 0).astype(float), f_edges, np.mean)
    frac_astro_neg = binned_stat(f0, neg.astype(float), f_edges, np.mean)
    frac_dom = binned_stat(f0, (np.abs(ratio) > 1).astype(float), f_edges, np.mean)
    # magnitude quantiles over the NEGATIVE (interaction-dominated) branch;
    # the positive branch is a tiny GW-dominated correction on another scale
    fi, ai = f0[neg], np.abs(fdot_astro[neg])
    q50 = binned_stat(fi, ai, f_edges, lambda v: np.quantile(v, 0.50))
    q90 = binned_stat(fi, ai, f_edges, lambda v: np.quantile(v, 0.90))
    q99 = binned_stat(fi, ai, f_edges, lambda v: np.quantile(v, 0.99))
    C_fit, p_fit = powerlaw_fit(f_cent, q90)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(f_cent, frac_fdot_neg, color=C_BLUE, marker="o", ms=4,
                 label="frac(fdot < 0)")
    axes[0].plot(f_cent, frac_astro_neg, color=C_ORANGE, marker="s", ms=4,
                 label="frac(fdot_astro < 0)")
    axes[0].plot(f_cent, frac_dom, color=C_PURPLE, marker="^", ms=4,
                 label="frac(|fdot_astro| > fdot_gr)")
    axes[0].set_xscale("log")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_xlabel("f0 [Hz]")
    axes[0].set_ylabel("fraction")
    axes[0].set_title("Sign / dominance fractions per f0 bin")
    axes[0].legend(fontsize=9)

    for q, lab, ls in [(q50, "q50", "-"), (q90, "q90", "--"), (q99, "q99", ":")]:
        axes[1].plot(f_cent, q, color=C_BLUE, ls=ls, marker="o", ms=3, label=lab)
    axes[1].plot(f_cent, C_fit * f_cent ** p_fit, color=C_PURPLE, lw=2,
                 label=rf"q90 fit {C_fit:.2e} f^{{{p_fit:.2f}}}")
    axes[1].plot(f_cent, np.abs(get_fdot_mojito(f_cent, "-")), color=C_ORANGE,
                 lw=2, ls="--", label="|mojito '-' env|")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].axhline(FDOT_RES_90D, color=C_GRAY, lw=1, ls=":")
    axes[1].axhline(FDOT_RES_4YR, color=C_GRAY, lw=1, ls=":")
    axes[1].set_xlabel("f0 [Hz]")
    axes[1].set_ylabel("|fdot_astro| quantiles [Hz/s]")
    axes[1].set_title("neg-branch |fdot_astro| scaling")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return (C_fit, p_fit)


def fig5_scaled_u(f0, abs_fdot_astro, envelopes, f_edges, out):
    """log10|u| distributions per f0 half-decade for each candidate envelope
    (NEGATIVE branch: fdot_astro < 0, the branch that dominates fdot_gr).

    Winner = the envelope whose |u|-distributions are frequency-invariant.
    Returns {name: flatness} where flatness = max/min of per-bin q90(|u|).
    """
    bands = [(1e-4, 3.16e-4), (3.16e-4, 1e-3), (1e-3, 3.16e-3), (3.16e-3, 1e-2)]
    flatness = {}
    fig, axes = plt.subplots(1, len(envelopes), figsize=(5.5 * len(envelopes), 4.5),
                             sharey=True)
    band_colors = plt.get_cmap("viridis")(np.linspace(0.1, 0.85, len(bands)))
    for ax, (name, env_fn) in zip(np.atleast_1d(axes), envelopes.items()):
        log_u = np.log10(abs_fdot_astro / env_fn(f0))
        for (lo, hi), col in zip(bands, band_colors):
            m = (f0 >= lo) & (f0 < hi)
            n_band = int(np.count_nonzero(m))
            if n_band < 50:
                # the negative branch does not exist below ~0.3 mHz --
                # record the (near-)empty band instead of silently omitting
                ax.plot([], [], color=col,
                        label=f"[{lo * 1e3:.2f}, {hi * 1e3:.2f}] mHz "
                              f"(n={n_band})")
                continue
            ax.hist(log_u[m], bins=120, range=(-8, 4), density=True,
                    histtype="step", lw=1.6, color=col,
                    label=f"[{lo * 1e3:.2f}, {hi * 1e3:.2f}] mHz "
                          f"(n={n_band})")
        q90_u = binned_stat(f0, 10.0 ** log_u, f_edges,
                            lambda v: np.quantile(v, 0.90))
        good = np.isfinite(q90_u) & (q90_u > 0)
        flatness[name] = float(np.nanmax(q90_u[good]) / np.nanmin(q90_u[good]))
        ax.set_yscale("log")
        ax.set_xlabel("log10 |u| = log10(|fdot_astro| / env(f0))")
        ax.set_title(f"{name}\nq90 flatness max/min = {flatness[name]:.2f}")
        ax.legend(fontsize=7)
    np.atleast_1d(axes)[0].set_ylabel("density")
    fig.suptitle("Scaled-variable study (fdot_astro < 0 branch): "
                 "is |u| frequency-invariant?")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return flatness


def fig6_candidates(log_u, candidates, out):
    """Overlay candidate pdfs on the log10|u| histogram (negative branch).

    Candidates are parameterized directly in x = log10|u|, so pdf_x
    overlays are exact in this space.
    """
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.hist(log_u, bins=200, density=True, color="#cccccc",
            histtype="stepfilled", label="catalogue (fdot_astro < 0)")
    xx = np.linspace(np.quantile(log_u, 1e-4), np.quantile(log_u, 1 - 1e-4), 800)
    for (name, dist), col, ls in zip(
            candidates.items(), [C_BLUE, C_ORANGE, C_PURPLE, C_GRAY],
            ["-", "--", "-.", ":"]):
        ax.plot(xx, dist["pdf_x"](xx), color=col, ls=ls, lw=2, label=name)
    ax.set_yscale("log")
    ax.set_ylim(1e-6, None)
    ax.set_xlabel("x = log10 |u|")
    ax.set_ylabel("density")
    ax.set_title("Candidate priors on the negative-branch magnitude log10|u|")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# candidate priors + metrics
# ---------------------------------------------------------------------------

def fit_candidates(log_u, rng):
    """Fit candidate 1-D priors on x = log10|u| of the NEGATIVE branch.

    (The positive branch is a tiny sub-resolution correction handled as a
    near-zero mixture component; the negative branch is the astrophysically
    dominant magnitude these candidates describe.)

    Each entry: pdf_x (density in x), rvs_u (draws of u < 0), interval_x(q)
    central interval in x, params description.
    """
    fit_sub = rng.choice(log_u, size=min(300_000, log_u.size), replace=False)
    cands = {}

    lo, hi = np.quantile(log_u, [5e-4, 1 - 5e-4])
    width = hi - lo
    cands["loguniform"] = dict(
        params=f"log10|u| box [{lo:.3g}, {hi:.3g}] (99.9% quantile range)",
        pdf_x=lambda x, lo=lo, hi=hi, w=width: np.where(
            (x >= lo) & (x <= hi), 1.0 / w, 0.0),
        rvs_u=lambda size, lo=lo, hi=hi: -10.0 ** rng.uniform(lo, hi, size),
        interval_x=lambda q, lo=lo, hi=hi: (
            lo + (1 - q) / 2 * (hi - lo), hi - (1 - q) / 2 * (hi - lo)),
    )

    mu_n, sig_n = stats.norm.fit(fit_sub)
    cands["lognormal"] = dict(
        params=f"log10|u| ~ N(mu={mu_n:.3g}, sigma={sig_n:.3g})",
        pdf_x=lambda x: stats.norm.pdf(x, mu_n, sig_n),
        rvs_u=lambda size: -10.0 ** stats.norm.rvs(mu_n, sig_n, size=size,
                                                   random_state=rng),
        interval_x=lambda q: stats.norm.interval(q, mu_n, sig_n),
    )

    df_t, loc_t, scale_t = stats.t.fit(fit_sub)
    cands["log_student_t"] = dict(
        params=f"log10|u| ~ t(df={df_t:.3g}, loc={loc_t:.3g}, "
               f"scale={scale_t:.3g})",
        pdf_x=lambda x: stats.t.pdf(x, df_t, loc_t, scale_t),
        rvs_u=lambda size: -10.0 ** stats.t.rvs(df_t, loc_t, scale_t,
                                                size=size, random_state=rng),
        interval_x=lambda q: stats.t.interval(q, df_t, loc_t, scale_t),
    )

    a_s, loc_s, scale_s = stats.skewnorm.fit(fit_sub)
    cands["log_skewnormal"] = dict(
        params=f"log10|u| ~ skewnorm(a={a_s:.3g}, loc={loc_s:.3g}, "
               f"scale={scale_s:.3g})",
        pdf_x=lambda x: stats.skewnorm.pdf(x, a_s, loc_s, scale_s),
        rvs_u=lambda size: -10.0 ** stats.skewnorm.rvs(
            a_s, loc_s, scale_s, size=size, random_state=rng),
        interval_x=lambda q: stats.skewnorm.interval(q, a_s, loc_s, scale_s),
    )
    return cands


def coverage_table(log_u, f0, cands):
    """Per-candidate catalogue coverage of central 99% / 99.9% x-intervals."""
    rows = {}
    decades = [(1e-4, 1e-3), (1e-3, 1e-2)]
    for name, c in cands.items():
        lo99, hi99 = c["interval_x"](0.99)
        lo999, hi999 = c["interval_x"](0.999)
        row = dict(
            cov99_all=float(np.mean((log_u >= lo99) & (log_u <= hi99))),
            cov999_all=float(np.mean((log_u >= lo999) & (log_u <= hi999))),
        )
        for (dlo, dhi) in decades:
            m = (f0 >= dlo) & (f0 < dhi)
            row[f"cov99_{dlo * 1e3:g}mHz"] = float(
                np.mean((log_u[m] >= lo99) & (log_u[m] <= hi99)))
        rows[name] = row
    return rows


def fit_wneg_logistic(f0, neg, f_edges):
    """Logistic fit of the negative-branch weight vs log10(f0)."""
    f_cent = np.sqrt(f_edges[:-1] * f_edges[1:])
    w = binned_stat(f0, neg.astype(float), f_edges, np.mean)
    good = np.isfinite(w)
    x, y = np.log10(f_cent[good]), np.clip(w[good], 1e-4, 1 - 1e-4)
    b, a = np.polyfit(x, np.log(y / (1 - y)), 1)
    return (a, b), lambda f: 1.0 / (1.0 + np.exp(-(a + b * np.log10(f)))), w


def fig7_total_fdot_check(f0, mc, fdot, best_cand, env_fn, wneg_fn,
                          w_neg_const, rng, out, n_draw=400_000):
    """THE physicality check: prior-implied TOTAL fdot vs catalogue, per band.

    Draws (f0, Mc) from the astrophysical F0-Mc GMM prior and fdot_astro
    from the branch mixture (near-zero positive branch vs negative-branch
    candidate), once with a constant w_neg and once with the logistic
    w_neg(f0); overlays the implied total-fdot distribution on the
    catalogue's per f0 half-decade.
    """
    from lisatools.sampling.f0_mchirp_prior import F0McGMMSampling

    gmm = F0McGMMSampling.from_heatmap(
        f0_lims_mHz=(F0_LIMS_HZ[0] * 1e3, F0_LIMS_HZ[1] * 1e3), mc_lims=MC_LIMS,
        seed=SEED,
    )
    draws = gmm.rvs(size=n_draw)
    u_mag = -best_cand["rvs_u"](n_draw)  # |u| draws
    # three (f0, Mc) sources: the deployed GMM prior (const and logistic
    # w_neg), and a catalogue bootstrap that isolates the fdot_astro model
    # from any F0-Mc prior mismatch
    boot = rng.integers(0, f0.size, size=n_draw)
    variants = {
        "GMM (f0,Mc), const w_neg": (
            draws[:, 0] * 1e-3, draws[:, 1],
            lambda f: np.full_like(f, w_neg_const)),
        "GMM (f0,Mc), logistic w_neg(f0)": (
            draws[:, 0] * 1e-3, draws[:, 1], wneg_fn),
        "catalogue (f0,Mc) bootstrap, logistic w_neg(f0)": (
            f0[boot], mc[boot], wneg_fn),
    }
    totals, f0_of = {}, {}
    for lab, (f0_d, mc_d, w_fn) in variants.items():
        fdot_gr_d = get_fdot(f=f0_d, Mc=mc_d)
        is_neg = rng.random(n_draw) < w_fn(f0_d)
        totals[lab] = fdot_gr_d + np.where(is_neg, -u_mag * env_fn(f0_d), 0.0)
        f0_of[lab] = f0_d

    bands = [(1e-4, 3.16e-4), (3.16e-4, 1e-3), (1e-3, 3.16e-3), (3.16e-3, 1e-2)]
    edges = signed_log_edges(1e-24, 1e-13, per_decade=4)
    ticks = [-1e-15, -1e-18, -1e-21, 0, 1e-21, 1e-18, 1e-15]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for ax, (lo, hi) in zip(axes.ravel(), bands):
        m_cat = (f0 >= lo) & (f0 < hi)
        w_cat = np.full(int(np.count_nonzero(m_cat)),
                        1.0 / max(np.count_nonzero(m_cat), 1))
        ax.hist(np.clip(fdot[m_cat], edges[0], edges[-1]), bins=edges,
                weights=w_cat, color="#bbbbbb", histtype="stepfilled",
                label="catalogue fdot")
        for (lab, tot), col, ls in zip(totals.items(),
                                       [C_BLUE, C_ORANGE, C_PURPLE],
                                       ["-", "--", "-."]):
            m_d = (f0_of[lab] >= lo) & (f0_of[lab] < hi)
            n_d = int(np.count_nonzero(m_d))
            if n_d < MIN_BIN_COUNT:
                continue
            ax.hist(np.clip(tot[m_d], edges[0], edges[-1]), bins=edges,
                    weights=np.full(n_d, 1.0 / n_d), histtype="step",
                    lw=1.8, color=col, ls=ls, label=lab)
        ax.set_xscale("symlog", linthresh=1e-24)
        ax.set_yscale("log")
        ax.set_xticks(ticks)
        ax.set_ylim(1e-5, 1.0)
        ax.set_title(f"[{lo * 1e3:.2f}, {hi * 1e3:.2f}] mHz")
        ax.legend(fontsize=7)
        ax.set_ylabel("fraction of sources / bin")
    for ax in axes[1]:
        ax.set_xlabel("total fdot [Hz/s]")
    fig.suptitle("Prior-implied TOTAL fdot vs catalogue (physicality check)")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def fig8_branch_vs_mc(f0, mc, neg, f_edges, out):
    """Branch membership in the (f0, Mc) plane + w_neg(Mc) marginal.

    The negative (interaction-dominated) branch lives at low Mc -- this is
    the correlation an independent 1-D fdot_astro prior cannot represent,
    and what a joint (f0, Mc, fdot_astro) prior would condition on.
    """
    mc_edges = np.linspace(0.0, 1.0, 51)
    H_all, _, _ = np.histogram2d(f0, mc, bins=[f_edges, mc_edges])
    H_neg, _, _ = np.histogram2d(f0[neg], mc[neg], bins=[f_edges, mc_edges])
    with np.errstate(invalid="ignore"):
        frac = np.where(H_all >= 20, H_neg / H_all, np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    pcm = axes[0].pcolormesh(f_edges, mc_edges, frac.T, cmap="viridis",
                             vmin=0, vmax=1, rasterized=True)
    fig.colorbar(pcm, ax=axes[0], label="frac(fdot_astro < 0)")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("f0 [Hz]")
    axes[0].set_ylabel(r"Mc [$M_\odot$] (mass-based)")
    axes[0].set_title("Negative-branch fraction in the (f0, Mc) plane")

    mc_cent = 0.5 * (mc_edges[:-1] + mc_edges[1:])
    w_mc = binned_stat(mc, neg.astype(float), mc_edges, np.mean)
    axes[1].plot(mc_cent, w_mc, color=C_BLUE, marker="o", ms=4)
    axes[1].set_xlabel(r"Mc [$M_\odot$]")
    axes[1].set_ylabel("frac(fdot_astro < 0)")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].set_title("Branch membership vs Mc (marginal)")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def unphysicality_metrics(cands, env_fn, rng, w_pos=0.0, n_draw=200_000):
    """Prior-implied TOTAL fdot vs the empirical envelopes.

    Draws (f0, Mc) from the run's actual astrophysical F0-Mc GMM prior over
    the full stock box, fdot_astro = u * env(f0) with u from the mixture
    (u = 0 with probability ``w_pos`` -- the tiny sub-resolution positive
    branch -- else a negative-branch candidate draw), and reports the prior
    mass that (a) has |fdot_astro| > fdot_gr (astro dominates), (b) pushes
    total fdot below the empirical '-' envelope, (c) above the '+' envelope.
    """
    from lisatools.sampling.f0_mchirp_prior import F0McGMMSampling

    gmm = F0McGMMSampling.from_heatmap(
        f0_lims_mHz=(F0_LIMS_HZ[0] * 1e3, F0_LIMS_HZ[1] * 1e3), mc_lims=MC_LIMS,
        seed=SEED,
    )
    draws = gmm.rvs(size=n_draw)
    f0_Hz = draws[:, 0] * 1e-3
    mc = draws[:, 1]
    fdot_gr = get_fdot(f=f0_Hz, Mc=mc)
    env = env_fn(f0_Hz)
    lo_env = get_fdot_mojito(f0_Hz, "-")
    hi_env = get_fdot_mojito(f0_Hz, "+")
    out = {}
    for name, c in cands.items():
        u_draw = c["rvs_u"](n_draw)
        u_draw[rng.random(n_draw) < w_pos] = 0.0
        fdot_astro = u_draw * env
        total = fdot_gr + fdot_astro
        out[name] = dict(
            astro_dominates=float(np.mean(np.abs(fdot_astro) > fdot_gr)),
            below_minus_env=float(np.mean(total < lo_env)),
            above_plus_env=float(np.mean(total > hi_env)),
        )
    return out


def ratio_prior_analysis(f0, mc, fdot, fdot_gr, ratio, neg, rng, out):
    """The SELECTED prior: r = fdot_astro/fdot_gr ~ U[-M, M] (M ~ 5).

    Reports (a) catalogue coverage of the [-M, M] box at TRUTH Mc for a
    range of M -- note the sampler does not need truth-Mc coverage (the
    mirror seeding convention lands at r = 0 / r = -2, and (Mc, r) trade
    off along the fdot ridge), so this is interpretive; (b) the induced
    physical measure (Jacobian) facts; (c) an F9 figure with the induced
    total-fdot distribution vs the catalogue.
    """
    r_cat = ratio  # fdot_astro / fdot_gr at truth Mc
    print("\n=== SELECTED prior: r = fdot_astro/fdot_gr ~ U[-M, M] ===")
    print("catalogue r quantiles (truth Mc):")
    print("  pos branch: " + ", ".join(
        f"q{q:g}={np.quantile(r_cat[~neg], q):.3g}" for q in [0.5, 0.999]))
    print("  neg branch: " + ", ".join(
        f"q{q:g}={np.quantile(r_cat[neg], q):.3g}"
        for q in [0.001, 0.01, 0.5, 0.99, 0.999]))
    print("coverage of |r| <= M at truth Mc (all / neg branch):")
    for M in [2.0, 3.0, 5.0, 10.0, 30.0, 100.0]:
        inside = np.abs(r_cat) <= M
        print(f"  M = {M:6g}: {np.mean(inside):.4f} / "
              f"{np.mean(inside[neg]):.4f}")
    print("Jacobian / induced-measure facts (r sampled, prior defined in the")
    print("sampling basis -- no in-sampler Jacobian needed; these describe")
    print("the induced PHYSICAL prior):")
    print("  |dfdot/dr| = fdot_gr(f0, Mc) -> p(fdot | f0, Mc) = 1/(2 M fdot_gr),")
    print("  uniform over [(1-M) fdot_gr, (1+M) fdot_gr] (support scales with Mc);")
    print("  along a measured-fdot ridge the Mc marginal gains a 1/fdot_gr(Mc)")
    print("  ~ Mc^(-5/3) tilt vs the f0-Mc prior: e.g. Mc=0.2 vs 1.0 -> "
          f"{0.2 ** (-5 / 3):.1f}x.")

    # induced total-fdot under the deployed GMM x U[-M, M] vs the catalogue
    from lisatools.sampling.f0_mchirp_prior import F0McGMMSampling

    n_draw = 400_000
    gmm = F0McGMMSampling.from_heatmap(
        f0_lims_mHz=(F0_LIMS_HZ[0] * 1e3, F0_LIMS_HZ[1] * 1e3), mc_lims=MC_LIMS,
        seed=SEED,
    )
    draws = gmm.rvs(size=n_draw)
    f0_d = draws[:, 0] * 1e-3
    fdot_gr_d = get_fdot(f=f0_d, Mc=draws[:, 1])
    M = RATIO_M_DEFAULT
    total_d = fdot_gr_d * (1.0 + rng.uniform(-M, M, n_draw))
    lo_env_d = get_fdot_mojito(f0_d, "-")
    hi_env_d = get_fdot_mojito(f0_d, "+")
    print(f"induced total-fdot (GMM (f0,Mc) x U[-{M:g},{M:g}]): "
          f"frac fdot<0 = {np.mean(total_d < 0):.4f}, "
          f"frac < '-' env = {np.mean(total_d < lo_env_d):.4f}, "
          f"frac > '+' env = {np.mean(total_d > hi_env_d):.4f}")

    edges = signed_log_edges(1e-24, 1e-13, per_decade=4)
    ticks = [-1e-15, -1e-18, -1e-21, 0, 1e-21, 1e-18, 1e-15]
    bands = [(3.16e-4, 1e-3), (3.16e-3, 1e-2)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    r_edges = signed_log_edges(1e-4, 1e3, per_decade=6)
    axes[0].hist(np.clip(r_cat, r_edges[0], r_edges[-1]), bins=r_edges,
                 weights=np.full(r_cat.size, 1.0 / r_cat.size),
                 color="#bbbbbb", histtype="stepfilled",
                 label="catalogue r (truth Mc)")
    for Mv, col, ls in [(2, C_PURPLE, ":"), (5, C_BLUE, "-"), (10, C_ORANGE, "--")]:
        for sgn in (-1, 1):
            axes[0].axvline(sgn * Mv, color=col, ls=ls, lw=1.6,
                            label=f"M = {Mv}" if sgn < 0 else None)
    axes[0].set_xscale("symlog", linthresh=1e-4)
    axes[0].set_xticks([-1e2, -1e0, -1e-2, 0, 1e-2, 1e0, 1e2])
    axes[0].set_yscale("log")
    axes[0].set_xlabel("r = fdot_astro / fdot_gr")
    axes[0].set_ylabel("fraction of sources / bin")
    axes[0].set_title("catalogue ratio vs the U[-M, M] box")
    axes[0].legend(fontsize=8)
    for ax, (lo, hi) in zip(axes[1:], bands):
        m_cat = (f0 >= lo) & (f0 < hi)
        n_cat = int(np.count_nonzero(m_cat))
        ax.hist(np.clip(fdot[m_cat], edges[0], edges[-1]), bins=edges,
                weights=np.full(n_cat, 1.0 / max(n_cat, 1)), color="#bbbbbb",
                histtype="stepfilled", label="catalogue fdot")
        m_d = (f0_d >= lo) & (f0_d < hi)
        n_d = int(np.count_nonzero(m_d))
        ax.hist(np.clip(total_d[m_d], edges[0], edges[-1]), bins=edges,
                weights=np.full(n_d, 1.0 / max(n_d, 1)), histtype="step",
                lw=1.8, color=C_BLUE,
                label=f"GMM x U[-{M:g},{M:g}] induced total fdot")
        ax.set_xscale("symlog", linthresh=1e-24)
        ax.set_yscale("log")
        ax.set_xticks(ticks)
        ax.set_ylim(1e-5, 1.0)
        ax.set_xlabel("total fdot [Hz/s]")
        ax.set_title(f"[{lo * 1e3:.2f}, {hi * 1e3:.2f}] mHz")
        ax.legend(fontsize=8)
    fig.suptitle("F9 -- the selected ratio prior")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# transform validation
# ---------------------------------------------------------------------------

def validate_transform(rng, n_val=200_000):
    print("\n=== 9-column ratio-basis TransformContainer validation "
          "(the LIBRARY factory) ===")
    with h5py.File(CAT_PATH, "r") as f:
        n_total = f["Binaries"]["GW22FrequencySSBFrame"].shape[0]
        fdot_full = f["Binaries"]["GW22FrequencyDerivativeSourceFrame"][:]
    # random subset PLUS the most negative-fdot rows (worst cases)
    keep = np.union1d(
        rng.choice(n_total, size=min(n_val, n_total), replace=False),
        np.argsort(fdot_full)[:1000],
    )
    del fdot_full
    cols = load_validation_columns(keep)

    f0 = cols["GW22FrequencySSBFrame"]
    mc_cat = cols["ChirpMassSSBFrame"]
    fdot_cat = cols["GW22FrequencyDerivativeSourceFrame"]
    fdot_gr_cat = get_fdot(f=f0, Mc=mc_cat)
    ratio_cat = fdot_cat / fdot_gr_cat - 1.0  # r at the truth Mc

    # sampled-basis rows at catalogue truth (validated GB<->mojito convention:
    # sampled phi0 stores -TrueAnomaly; physical phi0 = +TrueAnomaly)
    sampled = np.column_stack([
        np.log(cols["Amplitude"]),
        f0 * 1e3,
        mc_cat,
        (-cols["TrueAnomaly"]) % (2 * np.pi),
        np.cos(cols["InclinationAngle"]),
        cols["PolarisationAngle"],
        cols["RightAscension"],
        np.sin(cols["Declination"]),
        ratio_cat,
    ])

    tc = make_gb_transform_container(
        use_chirp_mass=True, use_fdot_astro=True, mc_lims=MC_LIMS
    )
    assert tc.input_basis[-1] == "fdot_astro_ratio" and tc.ndim == 9
    physical = tc.both_transforms(sampled)

    assert physical.shape == (sampled.shape[0], 9)
    rel_fdot = np.abs(physical[:, 2] - fdot_cat) / np.maximum(np.abs(fdot_cat), 1e-30)
    dphi = (physical[:, 4] - cols["TrueAnomaly"]) % (2 * np.pi)
    dphi = np.minimum(dphi, 2 * np.pi - dphi)
    checks = {
        "f0 [Hz] exact": float(np.max(np.abs(physical[:, 1] - f0))),
        "fdot rel err (max)": float(np.max(rel_fdot)),
        "fddot == 0 (max |.|)": float(np.max(np.abs(physical[:, 3]))),
        "phi0 == +TrueAnomaly (max |d|)": float(np.max(dphi)),
    }

    # physical-side round trip through the mirror-convention inverse
    # (incl. every fdot <= 0 row)
    sampled_back = tc.both_inverse_transforms(physical)
    physical_2 = tc.both_transforms(sampled_back)
    checks["round-trip fdot rel err (max)"] = float(np.max(
        np.abs(physical_2[:, 2] - physical[:, 2])
        / np.maximum(np.abs(physical[:, 2]), 1e-30)))
    checks["round-trip other cols (max |d|)"] = float(np.max(
        np.abs(np.delete(physical_2, 2, axis=1) - np.delete(physical, 2, axis=1))))
    mc_back = sampled_back[:, 2]
    r_back = sampled_back[:, 8]
    checks["inverse Mc in box"] = float(np.mean(
        (mc_back >= MC_LIMS[0]) & (mc_back <= MC_LIMS[1])))
    inbox = (mc_back > MC_LIMS[0]) & (mc_back < MC_LIMS[1])
    pos_inbox = (fdot_cat > 0) & inbox
    neg_inbox = (fdot_cat < 0) & inbox
    checks["fdot>0 in-box: |r_back| max"] = float(
        np.max(np.abs(r_back[pos_inbox])))
    checks["fdot<0 in-box: |r_back+2| max"] = float(
        np.max(np.abs(r_back[neg_inbox] + 2.0)))
    info = {
        f"frac |r_back| <= M={RATIO_M_DEFAULT:g} (INFO)": float(
            np.mean(np.abs(r_back) <= RATIO_M_DEFAULT)),
        "r_back min / max (INFO)": (float(r_back.min()), float(r_back.max())),
    }

    # sprint deepcopy/pickle rule
    tc2 = pickle.loads(pickle.dumps(copy.deepcopy(tc)))
    checks["pickle/deepcopy identical output"] = float(np.max(np.abs(
        tc2.both_transforms(sampled[:1000]) - physical[:1000])))

    n_neg = int(np.count_nonzero(fdot_cat <= 0))
    print(f"validation rows: {sampled.shape[0]} ({n_neg} with fdot <= 0)")
    ok = True
    tol = {
        "f0 [Hz] exact": 1e-17,  # a few ulp at f0 = 1e-2 Hz (mHz->Hz->mHz)
        "fdot rel err (max)": 1e-10,
        "fddot == 0 (max |.|)": 0.0,
        "phi0 == +TrueAnomaly (max |d|)": 1e-12,
        "round-trip fdot rel err (max)": 1e-10,
        "round-trip other cols (max |d|)": 1e-10,
        "fdot>0 in-box: |r_back| max": 1e-10,
        "fdot<0 in-box: |r_back+2| max": 1e-10,
        "pickle/deepcopy identical output": 0.0,
    }
    for name, val in checks.items():
        if name == "inverse Mc in box":
            passed = val == 1.0
        else:
            passed = val <= tol.get(name, np.inf)
        print(f"  {name:42s} {val:12.6g}   {'PASS' if passed else 'FAIL'}")
        ok &= passed
    for name, val in info.items():
        print(f"  {name:42s} {val}")
    print(f"TransformContainer validation: {'ALL PASS' if ok else 'FAILURES ABOVE'}")
    return ok


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)
    print(f"catalogue: {CAT_PATH}")
    cat = load_catalogue(rng)
    f0 = cat["GW22FrequencySSBFrame"]
    mc = cat["ChirpMassSSBFrame"]
    fdot = cat["GW22FrequencyDerivativeSourceFrame"]
    n = f0.size
    print(f"rows analyzed: {n} of {cat['n_total']} (N_SUBSAMPLE={N_SUBSAMPLE})")

    fdot_gr = get_fdot(f=f0, Mc=mc)
    fdot_astro = fdot - fdot_gr
    ratio = fdot_astro / fdot_gr
    neg = fdot_astro < 0  # interaction-dominated branch (|astro| > gr)
    pos = ~neg            # GW-dominated branch (tiny positive correction)
    w_pos = float(np.mean(pos))

    print("\n=== global stats (two-branch structure) ===")
    print(f"frac(fdot < 0)              = {np.mean(fdot < 0):.4f}")
    print(f"frac(fdot_astro < 0)        = {np.mean(neg):.4f}")
    print(f"frac(|fdot_astro|>fdot_gr)  = {np.mean(np.abs(ratio) > 1):.4f}")
    print(f"frac(exactly-detached, |ratio|<{REL_EPS:g}) = "
          f"{np.mean(np.abs(ratio) < REL_EPS):.5f}")
    print(f"branch crossover check: neg-branch min |ratio| = "
          f"{np.min(np.abs(ratio[neg])):.3e}, "
          f"pos-branch max ratio = {np.max(ratio[pos]):.3e}")
    print("positive branch (GW-dominated): ratio quantiles "
          + ", ".join(f"q{q:g}={np.quantile(ratio[pos], q):.2e}"
                      for q in [0.5, 0.95, 0.999]))
    print("negative branch (interaction-dominated): fdot_astro quantiles")
    for q in [0.005, 0.05, 0.5, 0.95, 0.995]:
        print(f"  q{q:<6g} = {np.quantile(fdot_astro[neg], q): .4e}")
    print(f"fdot resolution scale 1/(2 pi Tobs^2): 90d = {FDOT_RES_90D:.2e}, "
          f"4yr = {FDOT_RES_4YR:.2e} Hz/s")
    print(f"frac of ALL sources with |fdot_astro| > 4yr resolution = "
          f"{np.mean(np.abs(fdot_astro) > FDOT_RES_4YR):.5f}")
    print(f"frac of ALL sources with |fdot_astro| > 90d resolution = "
          f"{np.mean(np.abs(fdot_astro) > FDOT_RES_90D):.6f}")
    print("branch-vs-Mc correlation: Mc quantiles [q05, q50, q95]")
    for lab, m in [("pos (GW-dom)", pos), ("neg (int-dom)", neg)]:
        qs = np.quantile(mc[m], [0.05, 0.5, 0.95])
        print(f"  {lab}: [{qs[0]:.3f}, {qs[1]:.3f}, {qs[2]:.3f}]")

    f_edges = np.logspace(np.log10(F0_LIMS_HZ[0]), np.log10(F0_LIMS_HZ[1]),
                          N_F0_BINS + 1)
    f_cent = np.sqrt(f_edges[:-1] * f_edges[1:])

    fig1_fdot_vs_f0(f0, fdot, fdot_gr, f_edges,
                    os.path.join(OUT_DIR, "f1_fdot_vs_f0.png"))
    C_fit, p_fit = fig4_binned(f0, fdot, fdot_astro, ratio, f_edges,
                               os.path.join(OUT_DIR, "f4_binned_stats.png"))
    fig2_fdot_astro_vs_f0(f0, fdot_astro, f_edges,
                          {"powerlaw": (C_fit, p_fit)},
                          os.path.join(OUT_DIR, "f2_fdot_astro_vs_f0.png"))
    fig3_distributions(fdot_astro, ratio,
                       os.path.join(OUT_DIR, "f3_distributions.png"))
    print(f"\nq90 |fdot_astro| (neg branch) power law: "
          f"{C_fit:.4e} * f^{p_fit:.4f}")

    med_mc = float(np.median(mc))
    envelopes = {
        "|mojito '-' env|": lambda f: np.abs(get_fdot_mojito(f, "-")),
        f"fdot_gr(f, Mc={med_mc:.2f})": lambda f: get_fdot(
            f=f, Mc=np.full_like(f, med_mc)),
        f"powerlaw fit f^{p_fit:.2f}": lambda f: C_fit * f ** p_fit,
    }
    # envelope + candidate study on the NEGATIVE branch (the positive branch
    # is a tiny sub-resolution correction -> near-zero mixture component)
    f0_n, abs_astro_n = f0[neg], np.abs(fdot_astro[neg])
    flatness = fig5_scaled_u(f0_n, abs_astro_n, envelopes, f_edges,
                             os.path.join(OUT_DIR, "f5_scaled_u.png"))
    print("\n=== envelope flatness (q90(|u|) max/min across f0 bins; 1 = perfect) ===")
    for name, fl in flatness.items():
        print(f"  {name:32s} {fl:8.2f}")
    win_name = min(flatness, key=flatness.get)
    win_env = envelopes[win_name]
    print(f"winner envelope: {win_name}")

    log_u = np.log10(abs_astro_n / win_env(f0_n))
    cands = fit_candidates(log_u, rng)
    fig6_candidates(log_u, cands, os.path.join(OUT_DIR, "f6_candidate_priors.png"))

    cov = coverage_table(log_u, f0_n, cands)
    unphys = unphysicality_metrics(cands, win_env, rng, w_pos=w_pos)

    print("\n=== per-decade branch weights (pos: tiny GW-dominated / "
          "neg: interaction-dominated) ===")
    for dlo, dhi in [(1e-4, 1e-3), (1e-3, 1e-2)]:
        m = (f0 >= dlo) & (f0 < dhi)
        print(f"  [{dlo * 1e3:g}, {dhi * 1e3:g}] mHz: pos "
              f"{np.mean(pos[m]):.3f} / neg {np.mean(neg[m]):.3f}")

    print("\n=== candidate summary (neg-branch x = log10(|fdot_astro|/env); "
          f"mixture w_pos(~0) = {w_pos:.3f}) ===")
    header = (f"{'candidate':34s} {'cov99':>7s} {'cov99.9':>8s} "
              f"{'astro>gr':>9s} {'<-env':>7s} {'>+env':>7s}")
    print(header)
    print("-" * len(header))
    cat_baseline = dict(
        astro_dominates=float(np.mean(np.abs(ratio) > 1)),
        below_minus_env=float(np.mean(fdot < get_fdot_mojito(f0, "-"))),
        above_plus_env=float(np.mean(fdot > get_fdot_mojito(f0, "+"))),
    )
    for name in cands:
        c, up = cov[name], unphys[name]
        print(f"{name:34s} {c['cov99_all']:7.4f} {c['cov999_all']:8.4f} "
              f"{up['astro_dominates']:9.4f} {up['below_minus_env']:7.4f} "
              f"{up['above_plus_env']:7.4f}")
        print(f"    params: {cands[name]['params']}")
    print(f"{'CATALOGUE baseline':34s} {'-':>7s} {'-':>8s} "
          f"{cat_baseline['astro_dominates']:9.4f} "
          f"{cat_baseline['below_minus_env']:7.4f} "
          f"{cat_baseline['above_plus_env']:7.4f}")

    (a_w, b_w), wneg_fn, w_bins = fit_wneg_logistic(f0, neg, f_edges)
    print(f"\nlogistic w_neg(f0) fit: 1/(1+exp(-({a_w:.3f} + {b_w:.3f}"
          f"*log10 f0)))")
    print("NOTE: w_neg(f0) is NOT monotonic (sharp onset ~0.4 mHz then "
          "decline) -- the logistic is illustrative only; the binned "
          "empirical curve is:")
    f_cent = np.sqrt(f_edges[:-1] * f_edges[1:])
    for fc, wv in zip(f_cent, w_bins):
        if np.isfinite(wv):
            print(f"  f0 = {fc * 1e3:7.3f} mHz  w_neg = {wv:.3f}")
    fig7_total_fdot_check(f0, mc, fdot, cands["lognormal"], win_env, wneg_fn,
                          float(np.mean(neg)), rng,
                          os.path.join(OUT_DIR, "f7_total_fdot_check.png"))
    fig8_branch_vs_mc(f0, mc, neg, f_edges,
                      os.path.join(OUT_DIR, "f8_branch_vs_mc.png"))
    for mc_cut, side in [(0.1, "<"), (0.2, ">")]:
        m = mc < mc_cut if side == "<" else mc > mc_cut
        print(f"w_neg | Mc {side} {mc_cut}: {np.mean(neg[m]):.4f} "
              f"(covers {np.mean(m):.3f} of catalogue)")

    print("\n=== structure summary for the prior decision ===")
    print(f"p(fdot_astro | f0) = [1 - w_neg]*(near-zero positive branch: "
          f"fdot_astro = {np.median(ratio[pos]):.2e} * fdot_gr, "
          f"sub-resolution -> ~0)")
    print(f"                   + w_neg*(negative branch: fdot_astro = "
          f"-|u|*env(f0), env = {win_name})")
    print(f"w_neg: global {np.mean(neg):.3f}; f0-dependent (logistic above); "
          f"neg-branch |u| tightly log-distributed (see candidates)")

    ratio_prior_analysis(f0, mc, fdot, fdot_gr, ratio, neg, rng,
                         os.path.join(OUT_DIR, "f9_ratio_prior.png"))
    validate_transform(rng)
    print(f"\nfigures in: {OUT_DIR}")


if __name__ == "__main__":
    main()
