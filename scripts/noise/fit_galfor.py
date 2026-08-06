#!/usr/bin/env python
"""Least-squares fit of the 5-param foreground model to a GALFOR brick.

Measures the brick's own X-channel PSD (Welch) and fits

    HyperbolicTangentGalacticForeground.specific_Sh_function(f, amp, fk, alpha, f_1, f_2)

through the SAME path the likelihood uses -- ``get_sensitivity(..., sens_fn=
"X2TDISens", include_instrument=False)`` -- so the fitted parameters are
directly the ones the galfor branch samples. Prints the fit plus a suggested
``GALFOR_PRIOR_RANGE`` centred on it::

    python scripts/noise/fit_galfor.py ../GALFOR_731d_2.5s_L1.h5
    python scripts/noise/fit_galfor.py <file> --tobs-days 45.51 --plot fit.png

The brick is the confusion foreground (resolvable binaries already
subtracted) and carries no instrument noise, so its raw periodogram IS the
foreground PSD -- nothing to subtract before fitting.

``--tobs-days`` fits only the leading window of the brick instead of all of
it. The galaxy is anisotropic and the LISA antenna pattern sweeps it
annually, so a 45 d window (the noise_only lite preset) sees a different
mean level than the 731 d average; run both to size the prior.
"""

from __future__ import annotations

import argparse

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import welch

# Model-basis parameter names, in the order specific_Sh_function takes them.
BASIS = ("amp", "fk", "alpha", "f_1", "f_2")
# Fit in log10 for everything but alpha -- the same basis the branch samples
# (GALFOR_LOG_PARAMS in stock/erebor/noise.py), which is also what makes the
# 12-decade f_1 range tractable for a Levenberg-Marquardt step.
LOG_COLS = (0, 1, 3, 4)


def read_xyz(path):
    """``(xyz (3, N), dt)`` from a mojito L1 file."""
    from mojito import MojitoL1File

    with MojitoL1File(path) as f:
        xyz = np.asarray(f.tdis.xyz_doppler[:]).T
        dt = 1.0 / f.tdis.time_sampling.fs
    return xyz, dt


def measure_psd(xyz, dt, nperseg, fmin, fmax, nbins):
    """Log-rebinned Welch PSD of the X channel -> ``(f, psd, counts, navg)``.

    X alone, not an XYZ average: the foreground component builds the
    covariance as ``C[i,i] = mag``, ``C[i,j] = -mag/2`` off a SINGLE magnitude
    (``GalacticForeground.base_covariance``), and that magnitude is defined by
    the X-channel sensitivity ``_XYZ_ELEMENT_SENS[2][0]``. Y and Z carry the
    same magnitude by construction, so averaging them in would only add the
    correlated off-diagonal structure back into a quantity that does not model
    it.

    The rebin is not cosmetic. Above ~3 mHz the confusion foreground is dead
    (resolvable binaries are already subtracted out of this brick), so raw
    Welch bins there scatter across ten decades around a near-zero mean. Their
    LOG has an unbounded left tail, and unbinned log-residuals let a handful of
    those bins dominate the cost and drag alpha / f_1 to the rails. Averaging
    the POWER (not the log) within each band restores an unbiased,
    well-conditioned estimate. Log spacing also stops the dense high-frequency
    end from outvoting the low end by sheer bin count.
    """
    f, psd = welch(xyz[0], fs=1.0 / dt, nperseg=nperseg, detrend=False)
    keep = (f >= fmin) & (f <= fmax)
    f, psd = f[keep], psd[keep]

    edges = np.geomspace(fmin, fmax, nbins + 1)
    idx = np.clip(np.digitize(f, edges) - 1, 0, nbins - 1)
    counts = np.bincount(idx, minlength=nbins)
    ok = counts > 0
    f_b = np.bincount(idx, weights=f, minlength=nbins)[ok] / counts[ok]
    psd_b = np.bincount(idx, weights=psd, minlength=nbins)[ok] / counts[ok]
    return f_b, psd_b, counts[ok], xyz.shape[1] // nperseg


def model_psd(f, params):
    """Foreground X-channel PSD for ``params`` in the model (linear) basis."""
    from lisatools.sensitivity import get_sensitivity
    from lisatools.stochastic import HyperbolicTangentGalacticForeground

    return get_sensitivity(
        f,
        sens_fn="X2TDISens",
        stochastic_params=tuple(float(p) for p in params),
        stochastic_function=HyperbolicTangentGalacticForeground,
        include_instrument=False,
        fill_nans=0.0,
    )


def to_fit_basis(p):
    """Linear model basis -> the fit's log10 basis (alpha stays linear)."""
    q = np.asarray(p, dtype=float).copy()
    q[list(LOG_COLS)] = np.log10(q[list(LOG_COLS)])
    return q


def from_fit_basis(q):
    """Fit basis -> linear model basis."""
    p = np.asarray(q, dtype=float).copy()
    p[list(LOG_COLS)] = 10.0 ** p[list(LOG_COLS)]
    return p


def expand(q_free, free, start_fit):
    """Free-parameter vector -> the full 5-vector in the fit basis."""
    q = np.asarray(start_fit, dtype=float).copy()
    q[free] = q_free
    return q


def residuals(q, f, psd, weights):
    """Weighted log-residual vector.

    Each rebinned point averages ``counts * navg`` independent chi^2_2 draws,
    so log(psd) has variance ~1/(counts*navg) independent of the level:
    fitting the log with sqrt(counts) weights is the correctly weighted fit,
    and it spans the band's 6 decades of dynamic range without the bright
    low-frequency end swamping the cutoff that pins fk and f_2.
    """
    model = model_psd(f, from_fit_basis(q))
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (np.log(model) - np.log(psd)) * weights
    return np.nan_to_num(r, nan=0.0, posinf=50.0, neginf=-50.0)


def fit(f, psd, weights, start, bounds, fixed=()):
    """Least-squares fit -> ``(params_linear, result, free_indices)``.

    ``fixed`` names parameters held at their ``start`` value. Holding
    ``alpha, f_1`` at the LDC values is the standard way to break this model's
    exp-vs-tanh cutoff degeneracy: with 5 free parameters the fit wanders to
    whichever corner it starts nearest and the amplitude inherits that
    ambiguity, which is precisely the number we need pinned.
    """
    start_fit = to_fit_basis(start)
    free = [i for i, n in enumerate(BASIS) if n not in fixed]
    lo, hi = to_fit_basis(bounds[0]), to_fit_basis(bounds[1])

    def fn(q_free):
        return residuals(expand(q_free, free, start_fit), f, psd, weights)

    res = least_squares(
        fn,
        start_fit[free],
        bounds=(lo[free], hi[free]),
        xtol=1e-14,
        ftol=1e-14,
        max_nfev=20000,
    )
    return from_fit_basis(expand(res.x, free, start_fit)), res, free


def identifiability(res, npts):
    """Per-parameter 1-sigma in the FIT basis, from the Jacobian at the optimum.

    ``J^T J`` inverted with a pseudo-inverse: the exp roll-off
    ``exp(-(f/f_1)**alpha)`` and the tanh cutoff ``1 + tanh(-(f-fk)/f_2)`` both
    suppress the same high-frequency side, so (alpha, f_1) is close to
    degenerate with (fk, f_2) and J is near-singular by construction. A huge
    sigma here is the signal that the data does not pin that parameter, and
    its prior has to come from physics rather than from this fit.
    """
    dof = max(npts - len(res.x), 1)
    s2 = 2.0 * res.cost / dof  # residuals are already weighted
    JTJ = res.jac.T @ res.jac
    return np.sqrt(np.clip(np.diag(np.linalg.pinv(JTJ) * s2), 0.0, None))


def report(name, params, f, psd, sigma, free):
    """Print the fit, its per-parameter sigma, and its scatter."""
    model = model_psd(f, params)
    ratio = np.log10(model / psd)
    sig = dict(zip(free, sigma))
    print(f"\n--- {name} ---")
    for i, (n, v) in enumerate(zip(BASIS, params)):
        extra = f"   (log10 = {np.log10(v):+8.3f})" if n != "alpha" else "            "
        if i not in sig:
            print(f"  {n:6s} {v: .6e}{extra}       (held fixed)")
            continue
        # sigma is in the FIT basis: decades for the log parameters, additive
        # for alpha. >1 decade means the data does not constrain it.
        unit = "" if n == "alpha" else " dec"
        pin = "  <-- unconstrained" if sig[i] > 1.0 else ""
        print(f"  {n:6s} {v: .6e}{extra}  +/- {sig[i]:8.3f}{unit}{pin}")
    print(
        f"  residual log10(model/data): median {np.median(ratio):+.3f}  "
        f"rms {np.sqrt(np.mean(ratio ** 2)):.3f}"
    )
    return model


def suggest_prior(fits, pad_decades, alpha_pad):
    """Prior ranges spanning every fit, padded.

    Padding is applied in the sampling basis (decades for the four log
    parameters, additive for alpha) so the prior is symmetric the way the
    proposal sees it.
    """
    arr = np.array([to_fit_basis(p) for p in fits])
    lo_fit, hi_fit = arr.min(axis=0), arr.max(axis=0)
    lo, hi = lo_fit - pad_decades, hi_fit + pad_decades
    lo[2], hi[2] = lo_fit[2] - alpha_pad, hi_fit[2] + alpha_pad
    lo[2] = max(lo[2], 1e-3)  # alpha > 0: it is a power-law index in exp(-(f/f_1)^alpha)
    return from_fit_basis(lo), from_fit_basis(hi)


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("file", help="GALFOR brick, e.g. ../GALFOR_731d_2.5s_L1.h5")
    p.add_argument("--fmin", type=float, default=3e-4, help="fit band low edge (Hz)")
    p.add_argument("--fmax", type=float, default=8e-3, help="fit band high edge (Hz)")
    p.add_argument(
        "--nperseg",
        type=int,
        default=1 << 19,
        help="Welch segment length in samples (default 524288 ~ 15 d at 2.5 s, "
        "df ~ 7.6e-7 Hz)",
    )
    p.add_argument(
        "--tobs-days",
        type=float,
        action="append",
        help="fit only the leading N days of the brick; repeatable, and the "
        "suggested prior spans every window fitted (default: the whole brick "
        "plus the 45.51 d noise_only lite window)",
    )
    p.add_argument(
        "--nbins",
        type=int,
        default=120,
        help="log-spaced bins the Welch PSD is averaged into before fitting "
        "(default 120 ~ 25 per decade)",
    )
    p.add_argument(
        "--fix",
        help="comma-separated parameters to hold at the stock LDC value "
        "(e.g. alpha,f_1) -- breaks the exp-vs-tanh cutoff degeneracy so the "
        "amplitude is actually determined",
    )
    p.add_argument("--pad-decades", type=float, default=1.5, help="prior padding, decades")
    p.add_argument("--alpha-pad", type=float, default=1.0, help="prior padding on alpha")
    p.add_argument("--plot", help="write a data-vs-fit PSD plot here")
    args = p.parse_args(argv)

    fixed = tuple(n.strip() for n in args.fix.split(",") if n.strip()) if args.fix else ()
    for n in fixed:
        if n not in BASIS:
            raise SystemExit(f"unknown parameter {n!r}; known: {', '.join(BASIS)}")

    xyz, dt = read_xyz(args.file)
    total_days = xyz.shape[1] * dt / 86400.0
    print(f"{args.file}: {xyz.shape[1]} samples, dt = {dt} s, {total_days:.1f} d")

    windows = args.tobs_days or [None, 45.51]

    # Stock 4-yr injection, converted to the model basis, as the starting guess.
    alpha0 = 1.18300266e00
    start = np.array(
        [3.26651613e-44, 2.09278117e-03, alpha0,
         3.01430978e03 ** (-1.0 / alpha0), 1.0 / 2.95774596e03]
    )
    # Deliberately loose: the fit should be free to leave the current prior.
    bounds = (
        np.array([1e-50, 1e-6, 1e-3, 1e-8, 1e-8]),
        np.array([1e-38, 1e0, 10.0, 1e2, 1e2]),
    )

    fits, curves = [], []
    for w in windows:
        if w is None:
            seg, name = xyz, f"full brick ({total_days:.1f} d)"
        else:
            n = int(round(w * 86400.0 / dt))
            if n > xyz.shape[1]:
                raise SystemExit(f"--tobs-days {w} exceeds the brick's {total_days:.1f} d")
            seg, name = xyz[:, :n], f"first {w:g} d"
        nperseg = min(args.nperseg, seg.shape[1] // 4)
        f, psd, counts, navg = measure_psd(
            seg, dt, nperseg, args.fmin, args.fmax, args.nbins
        )
        weights = np.sqrt(counts)
        params, res, free = fit(f, psd, weights, start, bounds, fixed=fixed)
        print(
            f"\n[{name}] {len(f)} log-bins ({counts.min()}-{counts.max()} raw "
            f"bins each), {navg} Welch segments, cost {res.cost:.4f}"
        )
        model = report(name, params, f, psd, identifiability(res, len(f)), free)
        fits.append(params)
        curves.append((name, f, psd, model))

    lo, hi = suggest_prior(fits, args.pad_decades, args.alpha_pad)
    print(
        f"\n=== suggested GALFOR_PRIOR_RANGE "
        f"(spans every window, +/-{args.pad_decades} decades; "
        f"alpha +/-{args.alpha_pad}) ==="
    )
    print("GALFOR_PRIOR_RANGE = (")
    for n, a, b in zip(BASIS, lo, hi):
        print(f"    ({a:.6g}, {b:.6g}),  # {n}")
    print(")")

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        # Ordinal windows -> a perceptual ramp, not a categorical palette.
        colors = plt.cm.viridis(np.linspace(0.0, 0.75, len(curves)))
        for (name, f, psd, model), c in zip(curves, colors):
            ax.loglog(f, psd, lw=0.6, alpha=0.35, color=c)
            ax.loglog(f, model, lw=1.8, color=c, label=f"{name} fit")
        ax.set_xlabel("frequency [Hz]")
        ax.set_ylabel(r"X-channel PSD [1/Hz]")
        ax.grid(alpha=0.25, lw=0.5, which="both")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(args.plot, bbox_inches="tight", dpi=130)
        print(f"\nwrote {args.plot}")


if __name__ == "__main__":
    main()
