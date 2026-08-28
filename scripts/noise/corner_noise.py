#!/usr/bin/env python
"""Corner plot of a noise run's sampled parameters.

Reads the global-fit HDF backend with h5py directly — no lisatools import — so
it runs anywhere ``corner`` + ``matplotlib`` are installed::

    python scripts/noise/corner_noise.py noise-pe/noise_instrument_testing.h5
    python scripts/noise/corner_noise.py <file> --branch galfor --discard 500

``global_fit/chain/<branch>`` is the **cold** chain — the 2026-07 cold-chain
flip stores beta=1 only at the top level (ntemps=1) — shaped
``(iterations, nsamplers, ntemps, nwalkers, nleaves, ndim)``. The per-branch
tempered ladder lives under ``global_fit/sub_backend/<branch>/chain`` shaped
``(iterations, ntemps, nwalkers, nleaves, ndim)``; ``--tempered`` reads the
cold slice from there instead (useful when comparing against the ladder).

Every run also prints a chain-diagnostics table -- integrated autocorrelation
time ``tau`` per parameter and the resulting effective sample size -- unless
``--no-diagnostics`` is passed.

Truth lines default to ``--reference lsq``: the least-squares fit of the noise
model to the actual brick, which is what a mojito run should be judged against.
``--reference injection`` restores the stock LDC parameters, which are a truth
only for a synthetic run that injected them. See :data:`REFERENCES`.

This script stays on the h5py + numpy floor deliberately, so it only ever looks
at the CHAIN. To see the fit against the DATA -- posterior draws of the fitted
evolutionary PSD over the data's own measured variance -- see ``ppc_noise.py``,
which needs the full lisatools stack and the mojito bricks to rebuild both.
"""

from __future__ import annotations

import argparse
import os

import h5py
import numpy as np

# Sampled parameters per branch, in chain order. psd and galfor are sampled
# logged whenever the run set <branch>.log_sampling -- run_noise_only.py's
# default, off elsewhere. The BASE differs per branch: psd in ln, galfor in
# log10 (see LOG_FN). galfor keeps alpha linear (it is an O(1) index); sgwb
# has no transform at all. See LOG_LABELS / LOG_PARAMS / resolve_basis below.
LABELS = {
    # f_1 / f_2 are the roll-off and tanh-transition frequency SCALES of
    # HyperbolicTangentGalacticForeground in Hz, not the "Slope1"/"Slope2"
    # numbers -- those are a different parameterization and get converted
    # (f_1 = slope1**(-1/alpha), f_2 = 1/slope2) before reaching the model.
    "psd": [r"$S_{\rm oms}$", r"$S_{\rm tm}$"],
    "galfor": [r"amp", r"$f_k$", r"$\alpha$", r"$f_1$", r"$f_2$"],
    "sgwb": [r"$\log_{10}A$", r"$\alpha$"],
}
LOG_LABELS = {
    "psd": [r"$\ln S_{\rm oms}$", r"$\ln S_{\rm tm}$"],
    "galfor": [
        r"$\log_{10}$ amp",
        r"$\log_{10} f_k$",
        r"$\alpha$",
        r"$\log_{10} f_1$",
        r"$\log_{10} f_2$",
    ],
}
# Which columns a log-basis chain actually carries logged (the rest stay
# linear): mirrors PSD/GALFOR_LOG_PARAMS in stock/erebor/noise.py.
LOG_PARAMS = {"psd": (0, 1), "galfor": (0, 1, 3, 4)}
# ...and in WHICH base. The two branches differ (2026-08): psd stayed in ln,
# galfor moved to log10 because its four logged parameters span 4-12 decades
# and are quoted in decades everywhere else. Keyed the same way so a joint
# psd+galfor corner converts each block with its own function.
LOG_FN = {"psd": np.log, "galfor": np.log10}

# Reference values, in LINEAR units -- a log-basis chain compares against
# LOG_FN[branch] of them. Two sets are available; --reference picks one.
#
# psd: the stock injection (lisatools.globalfit.stock.erebor.variants.noise).
# These ARE the levels the NOISE brick was generated at, so they are a truth in
# the strict sense. In mojito mode the run re-fits them from the brick's
# tabulated noise_estimates, landing within ~0.5%. The same values are used
# under both references -- the LSQ fit below held them fixed at exactly these
# while fitting the foreground, so there is no separate LSQ psd number.
PSD_INJECTION = [15e-12, 3e-15]

# galfor "injection": the stock 4-yr LDC parameters. NOT a truth for a mojito
# run -- the GALFOR brick is a synthesized galaxy, not a draw from this
# 5-parameter model, so nothing guarantees the brick's spectrum is reproduced
# at these parameters (it is not: they sit ~0.4 decades high in amp and put the
# roll-off in the wrong place). Useful only for synthetic runs that literally
# injected them.
#
# f_1 / f_2 are the model-basis frequencies in Hz. They held the raw
# "Slope1"/"Slope2" values (3014.3 / 2957.7) until 2026-08; those belong to a
# different parameterization and are converted (f_1 = slope1**(-1/alpha),
# f_2 = 1/slope2) before reaching the model. A chain plotted against the
# unconverted numbers shows truth lines ~6.4 decades off in f_1 and ~6.9 in f_2.
GALFOR_INJECTION = [
    3.26651613e-44,  # amp
    2.09278117e-03,  # fk
    1.18300266e00,  # alpha
    1.14556409e-03,  # f_1
    3.38095297e-04,  # f_2
]

# galfor "lsq" (the DEFAULT): the Fourier-domain least-squares fit of this same
# 5-parameter model to the GALFOR_731d_2.5s_L1 brick. This is the reference a
# mojito foreground run should be compared against -- it is the best this model
# can do on this data, which is the question a corner plot is actually asking.
#
# Provenance: cd1l-validation/galactic_foreground_estimation/
# foreground_estimation.ipynb, the `least_squares` cell that prints "Fit
# parameters" (the FIRST one; a later cell repeats the fit over a wider band
# and lands elsewhere -- alpha 7.14, amp 8.29e-45 -- because the shape is
# degenerate, see below). Its f_2 = 9.89e-4 is the number quoted in the
# GALFOR_PRIOR_RANGE ceiling rationale (stock/erebor/noise.py).
#
# Fit conditions, which differ from a run's and bound how literally to read the
# lines: log-residual least_squares with loss="soft_l1" over 2e-4 .. 5e-3 Hz,
# against the brick's X-channel periodogram, using an EQUAL-ARM analytic TDI-2
# X response and instrument noise held at PSD_INJECTION. A run fits
# 3e-4 .. 8e-3 Hz with the unequal-arm model, so agreement to a few 0.01 dex in
# amp / f_1 / alpha is the most that should be expected.
#
# Caveat that matters more than the numbers: fk and f_2 are NOT identified by
# this data. The exp(-(f/f_1)^alpha) roll-off and the 0.5(1+tanh(-(f-fk)/f_2))
# cutoff suppress the same high-frequency side, and above ~3.9 mHz the
# foreground is already under the instrument noise, so the likelihood is flat
# along a ridge in (amp, fk, f_2). Chains and least-squares fits alike settle at
# whichever point on that ridge they wandered to. Read the fk / f_2 truth lines
# as "one valid point on the ridge", not as a target the posterior must cover.
GALFOR_LSQ = [
    1.18955159143e-44,  # amp    (log10 -43.925)
    2.10304500452e-03,  # fk     (log10  -2.677)
    3.39975538551e00,  # alpha
    2.46392506051e-03,  # f_1    (log10  -2.609)
    9.89314627841e-04,  # f_2    (log10  -3.005)
]

REFERENCES = {
    "lsq": {"psd": PSD_INJECTION, "galfor": GALFOR_LSQ},
    "injection": {"psd": PSD_INJECTION, "galfor": GALFOR_INJECTION},
}
DEFAULT_REFERENCE = "lsq"


# galfor chains written before 2026-08 carry ln, not log10. The two are
# unmistakable on the amplitude column: the log10 prior spans -46 .. -42,
# while ln of that same support spans -106 .. -97. Anything below this cut is
# an ln chain, so convert it (ln -> log10 is a divide by ln 10) rather than
# refuse — an archived run still plots with correct labels and truth overlays.
LEGACY_LN_CUT = -60.0


def to_log10_if_legacy_ln(branch, samples, cols=None):
    """``(samples, converted)`` -- rescale a legacy ln galfor block to log10.

    Only ever fires for galfor (psd is still ln by design) and only on the
    logged columns; alpha is linear in both bases and must not be touched.
    A linear-basis block never trips the cut, since its amp is ~1e-44 > 0.
    """
    if branch != "galfor" or not samples.size:
        return samples, False
    if samples[..., 0].min() >= LEGACY_LN_CUT:
        return samples, False
    samples = samples.copy()
    cols = list(LOG_PARAMS[branch]) if cols is None else list(cols)
    samples[..., cols] /= np.log(10.0)
    return samples, True


def resolve_basis(samples, basis="auto"):
    """``"log"`` / ``"linear"`` for a psd or galfor sample block.

    ``"auto"`` reads it off the sign of the FIRST column -- Soms_d (6e-12 ..
    2e-10) and the foreground amplitude (1e-46 .. 1e-42) are both strictly
    positive in the linear basis and strictly negative once logged (psd, ln:
    -25.8 .. -22.3; galfor, log10: -46 .. -42), so one negative sample settles
    it. Base-agnostic, since ln and log10 share a sign. The later columns
    cannot serve: the f_1/f_2 ranges straddle 1, so their logs take either
    sign. Falls back to ``"linear"`` on an empty block.
    """
    if basis != "auto":
        return basis
    return "log" if samples.size and np.min(samples[:, 0]) < 0.0 else "linear"


def labels_and_truths(branch, samples, basis="auto", reference=DEFAULT_REFERENCE):
    """``(labels, truths, basis)`` for a branch block in whichever basis it is in.

    ``reference`` selects the truth set (see :data:`REFERENCES`): ``"lsq"`` (the
    default) compares galfor against the least-squares fit of the model to the
    GALFOR brick, ``"injection"`` against the stock LDC parameters. psd is the
    injection under both.

    Branches with no log form (sgwb, unknown) come back linear untouched.
    """
    truths_for = REFERENCES[reference]
    if branch not in LOG_PARAMS:
        return list(LABELS[branch]), truths_for.get(branch), "linear"
    basis = resolve_basis(samples, basis)
    if basis != "log":
        return list(LABELS[branch]), list(truths_for[branch]), basis
    truths = np.asarray(truths_for[branch], dtype=float)
    cols = list(LOG_PARAMS[branch])
    truths[cols] = LOG_FN[branch](truths[cols])  # ln for psd, log10 for galfor
    return list(LOG_LABELS[branch]), list(truths), basis


def load_samples(path, branch, discard=0, thin=1, tempered=False, group="global_fit", mask=True):
    """Cold-chain samples for one branch -> ``(nsamples, ndim)``."""
    with h5py.File(path, "r") as f:
        if group not in f:  # files written before the "mcmc" -> "global_fit" rename
            group = "mcmc"
        g = f[group]
        iteration = int(g.attrs["iteration"])
        if iteration <= 0:
            raise SystemExit(f"{path}: no stored iterations yet.")
        sl = slice(discard, iteration, thin)

        if tempered:
            grp = g["sub_backend"][branch]
            chain, inds = grp["chain"][sl], grp["inds"][sl]
            chain, inds = chain[:, 0], inds[:, 0]  # cold slice of the ladder
        else:
            if branch not in g["chain"]:
                raise SystemExit(f"{path}: no branch {branch!r}; present: {list(g['chain'])}")
            chain, inds = g["chain"][branch][sl], g["inds"][branch][sl]

    ndim = chain.shape[-1]
    # Everything left of (nleaves, ndim) is iterations/walkers -> flatten.
    # inds masks leaves that are "off" (always on for the 1-leaf noise branches).
    flat, converted = to_log10_if_legacy_ln(branch, chain.reshape(-1, ndim))
    if converted:
        print(f"note: {branch} chain is in the legacy ln basis; converted to log10")
    if not mask:
        # Joint mode: rows must stay aligned across branches, so masking is
        # refused rather than silently dropping different rows per branch.
        if not inds.all():
            raise SystemExit(
                f"branch {branch!r} has switched-off leaves; a joint plot needs "
                "every row present. Plot it on its own instead."
            )
        return flat
    return flat[inds.reshape(-1)]


def load_trace(path, branch, discard=0, thin=1, tempered=False, group="global_fit"):
    """Cold chain with the iteration and walker axes kept -> ``(iters, (niter, nwalkers, ndim))``."""
    with h5py.File(path, "r") as f:
        if group not in f:
            group = "mcmc"
        g = f[group]
        iteration = int(g.attrs["iteration"])
        if iteration <= 0:
            raise SystemExit(f"{path}: no stored iterations yet.")
        sl = slice(discard, iteration, thin)
        if tempered:
            chain = g["sub_backend"][branch]["chain"][sl]
        else:
            if branch not in g["chain"]:
                raise SystemExit(f"{path}: no branch {branch!r}; present: {list(g['chain'])}")
            chain = g["chain"][branch][sl]
        iters = np.arange(iteration)[sl]

    ndim, nleaves, nwalkers = chain.shape[-1], chain.shape[-2], chain.shape[-3]
    # Collapse whatever sits between iterations and walkers -- (nsamplers,
    # ntemps) at top level, both 1; ntemps in the sub-backend -- then take
    # rung 0, which is the cold chain in either layout.
    chain = chain.reshape(chain.shape[0], -1, nwalkers, nleaves, ndim)[:, 0]
    trace, _ = to_log10_if_legacy_ln(branch, chain[:, :, 0, :])
    return iters, trace  # leaf 0 (the noise branches carry one)


# Integrated autocorrelation time via the standard FFT estimator with Sokal's
# automated windowing -- the same estimator emcee/eryn report, reimplemented
# here rather than imported so the script keeps its "h5py + numpy only" floor.
AUTOCORR_WINDOW_C = 5.0  # window closes at the first M with M >= c * tau(M)
AUTOCORR_SAFE_N = 50  # ...and tau is only believable once the chain runs 50 tau


def _next_pow_two(n):
    i = 1
    while i < n:
        i <<= 1
    return i


def autocorr_func_1d(x):
    """Normalized autocorrelation function of a 1-D series, or ``None`` if constant.

    FFT estimator: zero-pad to twice the next power of two, so the circular
    wrap-around of the transform lands outside the lags that are kept.
    """
    x = np.asarray(x, dtype=float)
    n = _next_pow_two(len(x))
    f = np.fft.rfft(x - x.mean(), n=2 * n)
    acf = np.fft.irfft(f * np.conjugate(f), n=2 * n)[: len(x)]
    if acf[0] <= 0.0:  # zero-variance walker: a stuck or genuinely fixed parameter
        return None
    return acf / acf[0]


def _auto_window(tau_of_m, c=AUTOCORR_WINDOW_C):
    """Sokal's window: the smallest lag ``M`` with ``M >= c * tau(M)``.

    An all-True mask means the window never closes -- the chain is far shorter
    than its own correlation time -- so the last lag is taken. (emcee returns
    lag 0 there, i.e. tau = 1, which reads as a perfectly mixed chain; the last
    lag at least errs toward "badly correlated", which is the truth.)
    """
    below = np.arange(len(tau_of_m)) < c * tau_of_m
    if below.all():
        return len(tau_of_m) - 1
    return int(np.argmin(below))


def integrated_time(trace, c=AUTOCORR_WINDOW_C):
    """``(tau, ess, acf)`` for a ``(niter, nwalkers, ndim)`` cold-chain trace.

    ``tau`` is the integrated autocorrelation time per parameter in STORED
    steps (multiply by ``--thin`` for sampler iterations), ``ess = nwalkers *
    niter / tau`` the ensemble's effective sample size, and ``acf`` the
    ``(ndim, niter)`` walker-averaged autocorrelation function the estimate was
    built from. Parameters that never moved come back ``nan``.
    """
    niter, nwalkers, ndim = trace.shape
    taus = np.full(ndim, np.nan)
    acfs = np.full((ndim, niter), np.nan)
    for d in range(ndim):
        per_walker = [autocorr_func_1d(trace[:, w, d]) for w in range(nwalkers)]
        per_walker = [a for a in per_walker if a is not None]
        if not per_walker:
            continue
        # Ensemble estimator: average the walkers' ACFs, NOT the ACF of the
        # walker-mean -- the latter averages away the between-walker scatter
        # and reports a chain far better mixed than it is.
        f = np.mean(per_walker, axis=0)
        acfs[d] = f
        tau_of_m = 2.0 * np.cumsum(f) - 1.0
        # Floored at 1: the estimator's finite-N noise pushes tau slightly below
        # 1 on a short, well-mixed chain, and an ESS above the sample count is
        # not a thing worth printing.
        taus[d] = max(tau_of_m[_auto_window(tau_of_m, c)], 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ess = nwalkers * niter / taus
    return taus, ess, acfs


def print_diagnostics(trace, labels, thin=1):
    """Per-parameter tau / ESS / ACF table for a ``(niter, nwalkers, ndim)`` trace."""
    niter, nwalkers, ndim = trace.shape
    taus, ess, acfs = integrated_time(trace)
    step = "1 stored step = 1 iteration" if thin == 1 else f"1 stored step = {thin} iterations"
    w = max(max(len(x) for x in labels), 8)
    print(f"\nautocorrelation: {nwalkers} walkers x {niter} stored steps ({step})")
    print(f"  {'':{w}s} {'tau':>8s} {'N/tau':>7s} {'ESS':>9s} {'rho(1)':>7s} {'rho(10)':>8s}")
    for i, lab in enumerate(labels):
        if not np.isfinite(taus[i]):
            print(f"  {lab:{w}s} {'--':>8s}   (constant in every walker)")
            continue
        # Lags past the chain length simply do not exist yet on a short run.
        r1 = acfs[i, 1] if niter > 1 else np.nan
        r10 = acfs[i, 10] if niter > 10 else np.nan
        short = niter < AUTOCORR_SAFE_N * taus[i]
        print(
            f"  {lab:{w}s} {taus[i]:8.1f} {niter / taus[i]:7.1f} {ess[i]:9.1f} "
            f"{r1:7.3f} {r10:8.3f}{'  !' if short else ''}"
        )
    if np.isfinite(ess).any():
        i = int(np.nanargmin(ess))
        total = nwalkers * niter
        print(
            f"  worst: {labels[i]} -- ESS {ess[i]:.0f} of {total} samples "
            f"({100.0 * ess[i] / total:.1f}% independent)"
        )
    if np.any(niter < AUTOCORR_SAFE_N * taus[np.isfinite(taus)]):
        print(
            f"  ! chain is shorter than {AUTOCORR_SAFE_N} tau there, so that tau is a "
            "LOWER bound and its ESS an upper one -- run longer before trusting either"
        )


def plot_trace(iters, trace, labels, truths, out):
    """One row per parameter, one scatter point per walker per stored step.

    Points, not connected lines: a walker crossing between modes would draw a
    full-height segment joining them, and a handful of those blanket the panel
    that is supposed to show WHERE the modes are and how the walkers divide
    between them. Unjoined points also make the occupancy of each mode read by
    density, which is the thing worth seeing in a multimodal trace.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    _, nwalkers, ndim = trace.shape
    # Walker index is ORDINAL, not categorical, so it takes a perceptual ramp
    # rather than a categorical palette: no validated categorical set carries
    # more than 3 mutually-overlapping series, and a ramp never has to cycle.
    # Truncated at 0.85 to keep the pale end off a white background.
    colors = plt.cm.viridis(np.linspace(0.0, 0.85, nwalkers))

    # A trace panel is a density, not a set of individually readable marks, so
    # the marker shrinks and fades as the panel fills: ~4 pt / opaque on a few
    # hundred points, ~1 pt / faint once a long chain x many walkers puts tens
    # of thousands in the same strip. Both clipped so neither extreme degenerates.
    npts = max(trace.shape[0] * nwalkers, 1)
    ms = float(np.clip(200.0 / np.sqrt(npts), 3.0, 6.0))
    alpha = float(np.clip(6000.0 / npts, 0.15, 0.9))

    fig, axes = plt.subplots(ndim, 1, sharex=True, figsize=(10, 1.9 * ndim), squeeze=False)
    for i, ax in enumerate(axes[:, 0]):
        for w in range(nwalkers):
            ax.plot(
                iters,
                trace[:, w, i],
                ls="none",
                marker=".",
                ms=ms,
                mew=0,
                alpha=alpha,
                color=colors[w],
            )
        if truths is not None and truths[i] is not None:
            # reference line wears neutral ink, never a series color
            ax.axhline(truths[i], color="0.25", lw=1.2, ls="--", zorder=5)
        ax.set_ylabel(labels[i], fontsize=9)
        ax.grid(alpha=0.25, lw=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[-1, 0].set_xlabel("iteration")
    if nwalkers <= 8:
        # Explicit proxies: the plotted markers are deliberately small and
        # semi-transparent, and a legend key at that size identifies nothing.
        axes[0, 0].legend(
            handles=[
                Line2D(
                    [], [], ls="none", marker="o", ms=4, mew=0, color=colors[w], label=f"walker {w}"
                )
                for w in range(nwalkers)
            ],
            ncol=min(nwalkers, 4),
            fontsize=7,
            frameon=False,
            loc="best",
        )
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", dpi=130)
    plt.close(fig)


def load_joint(path, branches, basis="auto", reference=DEFAULT_REFERENCE, **kwargs):
    """Cold-chain samples for several branches, column-stacked -> ``(nsamples, sum(ndim))``.

    Valid because each branch's sub-state syncs its cold row into the shared
    main state (``ModuleSubState.sync_cold_row``): at a given iteration, walker
    ``w`` holds every branch's cold value, so the row is one point in the joint
    space. Branches are stored with independent temperature ladders, but only
    the beta=1 rung lands here.
    """
    cols, labels, truths = [], [], []
    for branch in branches:
        arr = load_samples(path, branch, mask=False, **kwargs)
        cols.append(arr)
        lab, tr, _ = labels_and_truths(branch, arr, basis, reference=reference)
        labels += [
            f"{branch}: {x}"
            for x in (lab if len(lab) == arr.shape[1] else [f"p{i}" for i in range(arr.shape[1])])
        ]
        truths += list(tr) if tr and len(tr) == arr.shape[1] else [None] * arr.shape[1]
    n = {c.shape[0] for c in cols}
    if len(n) != 1:
        raise SystemExit(f"branches have different sample counts {n}; cannot pair them.")
    return np.hstack(cols), labels, truths


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("file", help="run HDF5, e.g. noise-pe/noise_instrument_testing.h5")
    p.add_argument(
        "--branch",
        default="psd",
        help="branch to plot, or a comma-separated list for a joint corner "
        f"(e.g. psd,galfor). Known: {', '.join(sorted(LABELS))}",
    )
    p.add_argument("--discard", type=int, default=0, help="burn-in iterations to drop")
    p.add_argument("--thin", type=int, default=1)
    p.add_argument("--tempered", action="store_true", help="read from the sub_backend ladder")
    p.add_argument(
        "--plot",
        default="corner",
        choices=("corner", "trace", "both"),
        help="corner (default), per-parameter walker traces, or both",
    )
    p.add_argument(
        "--basis",
        default="auto",
        choices=("auto", "log", "linear"),
        help="sampling basis of the psd/galfor chains: log (run_noise_only.py's "
        "default; ln for psd, log10 for galfor), linear, or auto -- detected "
        "per branch from the sign of its amplitude column, which the two "
        "priors' supports separate cleanly",
    )
    p.add_argument(
        "--no-diagnostics",
        action="store_true",
        help="skip the tau / effective-sample-size table (it re-reads the chain "
        "with the iteration and walker axes kept)",
    )
    p.add_argument(
        "--reference",
        default=DEFAULT_REFERENCE,
        choices=tuple(REFERENCES),
        help="which reference values the truth lines show. 'lsq' (default) "
        "uses the least-squares fit of the 5-param model to the GALFOR brick "
        "-- the best this model can do on this data, and the right comparison "
        "for a mojito run. 'injection' uses the stock LDC parameters, which "
        "are only a truth for a synthetic run that injected them. psd is the "
        "injection either way. NOTE: galfor's fk and f_2 are not identified by "
        "this data (flat ridge above ~3.9 mHz, where the foreground is already "
        "under the instrument noise), so their lines mark one point on that "
        "ridge rather than a target the posterior has to cover",
    )
    p.add_argument(
        "--truths",
        help="comma-separated truths, overriding --reference entirely",
    )
    p.add_argument("--no-truths", action="store_true")
    p.add_argument("-o", "--out", help="output png (default: <file>_<branch>_corner.png)")
    args = p.parse_args(argv)

    branches = [b.strip() for b in args.branch.split(",") if b.strip()]
    for b in branches:
        if b not in LABELS:
            raise SystemExit(f"unknown branch {b!r}; known: {', '.join(sorted(LABELS))}")
    read_kwargs = dict(discard=args.discard, thin=args.thin, tempered=args.tempered)

    if len(branches) > 1:
        samples, labels, truths = load_joint(
            args.file, branches, basis=args.basis, reference=args.reference, **read_kwargs
        )
    else:
        samples = load_samples(args.file, branches[0], **read_kwargs)
        labels, truths, basis = labels_and_truths(
            branches[0], samples, args.basis, reference=args.reference
        )
        if branches[0] in LOG_PARAMS:
            print(f"{branches[0]} chain read as the {basis} basis")
        if samples.shape[1] != len(labels):
            labels = [f"p{i}" for i in range(samples.shape[1])]

    if args.truths:
        truths = [float(v) for v in args.truths.split(",")]
        print("truth lines: --truths override")
    elif args.no_truths:
        pass
    else:
        print(f"truth lines: --reference {args.reference}")
        if args.reference == "lsq" and "galfor" in branches:
            # The plot cannot show this and it changes how the lines read.
            print("  galfor fk / f_2 are unidentified by the data -- their")
            print("  lines mark one point on a flat ridge, not a target.")
    if args.no_truths:
        truths = None
    if truths is not None and len(truths) != samples.shape[1]:
        raise SystemExit(f"{len(truths)} truths for {samples.shape[1]} parameters")

    stem = f"{os.path.splitext(args.file)[0]}_{'_'.join(branches)}"

    print(f"{'+'.join(branches)}: {samples.shape[0]} samples x {samples.shape[1]} params")
    for i, lab in enumerate(labels):
        lo, med, hi = np.percentile(samples[:, i], [16, 50, 84])
        print(f"  {lab:16s} {med:.4e}  (-{med - lo:.2e} +{hi - med:.2e})")

    want_trace = args.plot in ("trace", "both")
    if want_trace or not args.no_diagnostics:
        # Same read for both consumers: the diagnostics need the iteration and
        # walker axes that load_samples flattens away, which is what load_trace
        # keeps. Branches share an iteration axis, so they concatenate on ndim.
        traces, iters = [], None
        for b in branches:
            iters, tr = load_trace(args.file, b, **read_kwargs)
            traces.append(tr)
        trace = np.concatenate(traces, axis=-1)
        if not args.no_diagnostics:
            print_diagnostics(trace, labels, thin=args.thin)

    if want_trace:
        out = args.out if (args.out and args.plot == "trace") else f"{stem}_trace.png"
        plot_trace(iters, trace, labels, truths, out)
        print(f"\n{trace.shape[1]} walkers x {trace.shape[0]} iterations -> {out}")
        if args.plot == "trace":
            return

    if samples.shape[0] < 10 * samples.shape[1]:
        print(
            f"\nWARNING: only {samples.shape[0]} samples — too few for a meaningful "
            "corner plot. It will render, but it shows the initial draw, not a posterior."
        )

    import matplotlib

    matplotlib.use("Agg")
    import corner
    import matplotlib.pyplot as plt

    # Linear levels are ~1e-11 / ~1e-44, so corner's default ".2f" titles would
    # all read 0.00 -> ".2e". A log-sampled chain is O(10), where ".2e" reads
    # "-2.58e+01" and a fixed-point title is the legible one. One fmt covers
    # the whole figure, so a joint log-psd + galfor corner keeps ".2e".
    scale = np.median(np.abs(samples), axis=0)
    title_fmt = ".3f" if np.all((scale > 1e-3) & (scale < 1e4)) else ".2e"

    fig = corner.corner(
        samples,
        labels=labels,
        truths=truths,
        show_titles=True,
        quantiles=[0.16, 0.5, 0.84],
        title_fmt=title_fmt,
        title_kwargs={"fontsize": 8},
    )
    out = args.out or f"{stem}_corner.png"
    fig.savefig(out, bbox_inches="tight", dpi=130)
    plt.close(fig)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
