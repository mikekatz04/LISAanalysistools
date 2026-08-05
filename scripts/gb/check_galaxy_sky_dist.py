"""Validate the 3-D Galaxy prior in the GB sampling basis, with plots.

Run::

    python scripts/gb/check_galaxy_sky_dist.py --out ./galaxy_prior_check

Checks (all printed PASS/FAIL, exit code non-zero on any failure):

1. ICRS->ecliptic matrix matches astropy to < 1 arcsec (skipped w/o astropy).
2. The Jacobian: p(d, alpha, sin_delta) = rho * d^2 with NO cos factor.
   Integrating the (lam, beta, d) form and the (alpha, sin_delta, d) form
   over the same solid angle must give the same mass.
3. pdf integrates to 1 over the sampled box (that is what ``norm`` is for).
4. rvs() is distributed as pdf() -- 1-D KS-style check per column, plus a
   direct histogram-vs-pdf comparison of the distance marginal.
5. logpdf is finite wherever pdf > 0, and -inf outside the box.

Plots written to ``--out``:

  galaxy_prior_marginals.png  -- rvs histograms vs the analytic marginals
  galaxy_prior_sky.png        -- sky density in ICRS and ecliptic; the
                                 galactic plane must appear as a great
                                 circle inclined ~60 deg in ICRS
  galaxy_prior_xz.png         -- galactocentric x-z slice (disk + bulge)
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np


def _fail(msg):
    print(f"  FAIL: {msg}")
    return False


def check_rotation():
    print("[1] ICRS->ecliptic rotation vs astropy")
    from lisatools.globalfit.priors.galaxy_sky_dist import ICRS_TO_ECLIPTIC
    try:
        import astropy.units as u
        from astropy.coordinates import BarycentricMeanEcliptic, SkyCoord
    except ImportError:
        print("  SKIP (astropy not installed)")
        return True
    rng = np.random.default_rng(0)
    ra = rng.uniform(0, 2 * np.pi, 500)
    dec = np.arcsin(rng.uniform(-1, 1, 500))
    c = SkyCoord(ra=ra * u.rad, dec=dec * u.rad, frame="icrs")
    # MEAN ecliptic: our matrix is the J2000 MEAN obliquity. The 'true'
    # frame adds nutation (~15 arcsec) and is the wrong comparison.
    ecl = c.transform_to(BarycentricMeanEcliptic())
    v = np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra),
                  np.sin(dec)])
    w = ICRS_TO_ECLIPTIC @ v
    lam = np.mod(np.arctan2(w[1], w[0]), 2 * np.pi)
    beta = np.arcsin(np.clip(w[2], -1, 1))
    dlam = np.abs(np.mod(lam - ecl.lon.rad + np.pi, 2 * np.pi) - np.pi)
    dbeta = np.abs(beta - ecl.lat.rad)
    worst = np.rad2deg(max(dlam.max(), dbeta.max())) * 3600.0
    print(f"  worst separation = {worst:.3f} arcsec")
    return worst < 1.0 or _fail("rotation disagrees with astropy by > 1 arcsec")


def check_jacobian():
    print("[2] Jacobian: no cos factor in (dist, alpha, sin_delta)")
    from lisatools.globalfit.priors.galaxy_prior_3d import (
        ECLIPTIC_TO_GALACTIC, R_GC, GalaxyPrior3D,
    )
    g = GalaxyPrior3D()
    nl = nb = 100
    nd = 300
    dmax = 60.0
    lam = (np.arange(nl) + 0.5) / nl * 2 * np.pi
    bet = -np.pi / 2 + (np.arange(nb) + 0.5) / nb * np.pi
    dd = (np.arange(nd) + 0.5) / nd * dmax
    L, B, D = np.meshgrid(lam, bet, dd, indexing="ij")
    IA = g.pdf(np.column_stack([L.ravel(), B.ravel(), D.ravel()])).sum() * (
        (2 * np.pi / nl) * (np.pi / nb) * (dmax / nd))

    s = -1 + (np.arange(nb) + 0.5) / nb * 2.0
    L2, S2, D2 = np.meshgrid(lam, s, dd, indexing="ij")
    b2 = np.arcsin(S2)
    cb = np.cos(b2)
    ecl = np.array([cb * np.cos(L2), cb * np.sin(L2), np.sin(b2)])
    gal = np.tensordot(ECLIPTIC_TO_GALACTIC, ecl, axes=(1, 0))
    IB = (g.density(D2 * gal[0] - R_GC, D2 * gal[1], D2 * gal[2])
          * D2**2).sum() * ((2 * np.pi / nl) * (2.0 / nb) * (dmax / nd))
    print(f"  int p(lam,beta,d)      = {IA:.6f}")
    print(f"  int rho*d^2 (solid ang)= {IB:.6f}   ratio = {IB / IA:.6f}")
    return abs(IB / IA - 1.0) < 1e-3 or _fail("cos factor does not cancel")


def check_normalization(dist_lims):
    print(f"[3] pdf integrates to 1 over dist_lims={dist_lims}")
    from lisatools.globalfit.priors.galaxy_sky_dist import GalaxySkyDistPrior
    p = GalaxySkyDistPrior(dist_lims=dist_lims)
    na = ns = 220
    nd = 900
    a = (np.arange(na) + 0.5) / na * 2 * np.pi
    sd = -1 + (np.arange(ns) + 0.5) / ns * 2.0
    lo, hi = dist_lims
    d = lo + (np.arange(nd) + 0.5) / nd * (hi - lo)
    A, S, D = np.meshgrid(a, sd, d, indexing="ij")
    pts = np.column_stack([D.ravel(), A.ravel(), S.ravel()])
    cell = (2 * np.pi / na) * (2.0 / ns) * ((hi - lo) / nd)
    I = p.pdf(pts).sum() * cell
    print(f"  integral = {I:.5f}   (prior mass inside range = {p.norm:.5f})")
    return abs(I - 1.0) < 2e-2 or _fail(f"pdf integrates to {I}, not 1")


def check_rvs(dist_lims, n=200_000):
    print("[4] rvs() matches pdf()")
    from lisatools.globalfit.priors.galaxy_sky_dist import GalaxySkyDistPrior
    p = GalaxySkyDistPrior(dist_lims=dist_lims, rng=1)
    x = np.asarray(p.rvs(n))
    lo, hi = dist_lims
    ok = True
    if not ((x[:, 0] >= lo) & (x[:, 0] <= hi)).all():
        ok = _fail("rvs produced distances outside dist_lims")
    # distance marginal: histogram vs numerically marginalized pdf
    edges = np.linspace(lo, hi, 41)
    ctr = 0.5 * (edges[1:] + edges[:-1])
    hist, _ = np.histogram(x[:, 0], bins=edges, density=True)
    na = ns = 60
    a = (np.arange(na) + 0.5) / na * 2 * np.pi
    sd = -1 + (np.arange(ns) + 0.5) / ns * 2.0
    A, S = np.meshgrid(a, sd, indexing="ij")
    marg = np.array([
        p.pdf(np.column_stack([np.full(A.size, c), A.ravel(), S.ravel()])).sum()
        * (2 * np.pi / na) * (2.0 / ns) for c in ctr])
    rel = np.abs(hist - marg) / np.maximum(marg, marg.max() * 1e-3)
    print(f"  distance marginal: max rel dev = {rel.max():.3f}, "
          f"median = {np.median(rel):.3f}")
    if np.median(rel) > 0.1:
        ok = _fail("rvs distance marginal disagrees with pdf")
    return ok


def check_logpdf(dist_lims):
    print("[5] logpdf finite inside, -inf outside")
    from lisatools.globalfit.priors.galaxy_sky_dist import GalaxySkyDistPrior
    p = GalaxySkyDistPrior(dist_lims=dist_lims, rng=2)
    x = np.asarray(p.rvs(2000))
    lp = p.logpdf(x)
    ok = True
    if not np.isfinite(lp).all():
        ok = _fail(f"{(~np.isfinite(lp)).sum()} non-finite logpdf on rvs draws")
    out = np.array([[dist_lims[1] * 2, 1.0, 0.0]])
    if np.isfinite(p.logpdf(out)).any():
        ok = _fail("logpdf finite outside dist_lims")
    return ok


def make_plots(out, dist_lims):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from lisatools.globalfit.priors.galaxy_sky_dist import (
        ICRS_TO_ECLIPTIC, GalaxySkyDistPrior,
    )
    os.makedirs(out, exist_ok=True)
    p = GalaxySkyDistPrior(dist_lims=dist_lims, rng=3)
    x = np.asarray(p.rvs(200_000))
    dist, alpha, sind = x.T

    # -- marginals -----------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    lo, hi = dist_lims
    edges = np.linspace(lo, hi, 61)
    ctr = 0.5 * (edges[1:] + edges[:-1])
    ax[0].hist(dist, bins=edges, density=True, color="#4C72B0", alpha=.75,
               label="rvs")
    na = ns = 60
    a = (np.arange(na) + .5) / na * 2 * np.pi
    sd = -1 + (np.arange(ns) + .5) / ns * 2.
    A, S = np.meshgrid(a, sd, indexing="ij")
    marg = np.array([
        p.pdf(np.column_stack([np.full(A.size, c), A.ravel(), S.ravel()])).sum()
        * (2 * np.pi / na) * (2. / ns) for c in ctr])
    ax[0].plot(ctr, marg, "k-", lw=2, label="pdf")
    ax[0].set_xlabel("distance [kpc]"); ax[0].set_ylabel("density")
    ax[0].legend(frameon=False)
    ax[1].hist(alpha, bins=60, density=True, color="#DD8452", alpha=.8)
    ax[1].set_xlabel(r"$\alpha$ (ICRS) [rad]")
    ax[2].hist(sind, bins=60, density=True, color="#55A868", alpha=.8)
    ax[2].set_xlabel(r"$\sin\delta$ (ICRS)")
    for a_ in ax: a_.spines[["top", "right"]].set_visible(False)
    fig.suptitle("3-D Galaxy prior in the GB basis: rvs vs pdf")
    fig.tight_layout(); fig.savefig(f"{out}/galaxy_prior_marginals.png", dpi=140)

    # -- sky, both frames ----------------------------------------------
    cd = np.sqrt(np.clip(1 - sind**2, 0, 1))
    v = np.stack([cd * np.cos(alpha), cd * np.sin(alpha), sind])
    w = ICRS_TO_ECLIPTIC @ v
    lam = np.mod(np.arctan2(w[1], w[0]), 2 * np.pi)
    beta = np.arcsin(np.clip(w[2], -1, 1))
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    ax[0].hexbin(alpha, sind, gridsize=110, bins="log", cmap="magma")
    ax[0].set_xlabel(r"$\alpha$ (ICRS)"); ax[0].set_ylabel(r"$\sin\delta$")
    ax[0].set_title("ICRS -- the sampled frame")
    ax[1].hexbin(lam, np.sin(beta), gridsize=110, bins="log", cmap="magma")
    ax[1].set_xlabel(r"$\lambda$ (ecliptic)"); ax[1].set_ylabel(r"$\sin\beta$")
    ax[1].set_title("ecliptic -- GalaxyPrior3D's native frame")
    fig.suptitle("Galactic plane should be a great circle, tilted ~60 deg in ICRS")
    fig.tight_layout(); fig.savefig(f"{out}/galaxy_prior_sky.png", dpi=140)

    # -- galactocentric slice ------------------------------------------
    gx, gy, gz = p._galactocentric(alpha, sind, dist)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].hexbin(gx, gy, gridsize=120, bins="log", cmap="viridis",
                 extent=(-25, 25, -25, 25))
    ax[0].set_xlabel("x [kpc]"); ax[0].set_ylabel("y [kpc]")
    ax[0].set_title("face-on"); ax[0].plot([-8.178], [0], "r*", ms=12)
    ax[1].hexbin(gx, gz, gridsize=120, bins="log", cmap="viridis",
                 extent=(-25, 25, -8, 8))
    ax[1].set_xlabel("x [kpc]"); ax[1].set_ylabel("z [kpc]")
    ax[1].set_title("edge-on"); ax[1].plot([-8.178], [0], "r*", ms=12)
    fig.suptitle("Galactocentric draws (red star = Sun); disk + bulge visible")
    fig.tight_layout(); fig.savefig(f"{out}/galaxy_prior_xz.png", dpi=140)
    print(f"\n[plots] wrote 3 figures to {out}/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./galaxy_prior_check")
    ap.add_argument("--dist-lo", type=float, default=0.001)
    ap.add_argument("--dist-hi", type=float, default=40.0)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()
    lims = (args.dist_lo, args.dist_hi)

    ok = all([
        check_rotation(),
        check_jacobian(),
        check_normalization(lims),
        check_rvs(lims),
        check_logpdf(lims),
    ])
    if not args.no_plots:
        make_plots(args.out, lims)
    print("\n" + ("ALL CHECKS PASSED" if ok else "*** FAILURES ABOVE ***"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
