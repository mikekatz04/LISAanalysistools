#!/usr/bin/env python
"""PROOF OF CONCEPT — does the O(f^2) response break the T-channel null?

Background. The galactic-foreground covariance model is
``C_ij(f,t) = Sgal_XX(f) * M_ij(t)`` with ``M`` from a modulation table, and
that table's T projection ``sum_ij M_ij`` is zero to machine precision (9e-16),
so the foreground contributes NOTHING to T = (X+Y+Z)/sqrt(3). Measured on the
two-year brick, the data has ~21% more T power than the model can supply.

Mathur & Cornish (LISA_Higher_Order_Response.pdf) give the next non-vanishing
order of the response in ``fL = 2 pi f / f_*`` -- the O(f) term averages to
zero, so the correction is O(f^2) -- tabulated as spherical-harmonic modes
``l in [0,6]`` in closed form in the orbital phase alpha and the constellation
phase lambda, for the self (X^n X^n) and cross (X^n X^m) responses.

The question this script answers: can that O(f^2) term give the foreground a
non-zero T response?

Method. The three channels are the SAME kernel at lambda, lambda + 2pi/3,
lambda + 4pi/3 (glass_galaxy.c: "putting beta = 2Pi/3 into XY yields YZ"), so

    sum_ij M_ij = sum_n [ XX(alpha, lambda + 2 pi n / 3)
                          + 2 XY(alpha, lambda + 2 pi n / 3) ]

That threefold sum is a comb filter on the lambda harmonics: it annihilates
every e^{i m lambda} whose m is not a multiple of 3, and multiplies the rest by
3. So the null can only ever be broken by the m = 0, +-3, +-6 harmonics of a
mode -- which is checkable mode by mode, and is checked here.

Part 1 uses the EXISTING leading-order kernels (the user's galaxy_modulation.py,
ported from glass_galaxy.c) to establish *why* the O(f^0) null holds: trivially,
by harmonic selection, or by a real cancellation between the self and cross
kernels. That distinction decides whether O(f^2) has any reason to null too.

Part 2 runs the same test on the O(f^2) modes transcribed from the note's
appendix. Only the SHORT expressions are transcribed -- see MODES_OF2 -- so the
coverage is partial and stated as such; the long ones should come from the
authors' source rather than from OCR of a PDF, since one sign error in a
40-term expression would silently decide the answer.
"""

from __future__ import annotations

import os
import sys

import numpy as np

GALMOD_DIR = os.path.expanduser("~/code/cd1l-validation/galactic_foreground_estimation")
if GALMOD_DIR not in sys.path:
    sys.path.insert(0, GALMOD_DIR)

# galaxy_modulation imports astropy_healpix at module scope for the sky-map
# step, which this test never reaches -- it only wants the response kernels.
# Stub it rather than pull the dependency into this venv.
if "astropy_healpix" not in sys.modules:
    import types

    sys.modules["astropy_healpix"] = types.ModuleType("astropy_healpix")

import galaxy_modulation as gm  # noqa: E402

TWO_PI_3 = 2.0 * np.pi / 3.0


# ---------------------------------------------------------------------------
# the threefold sum, and what it does to a lambda harmonic
# ---------------------------------------------------------------------------


def threefold(fn, alpha, lam):
    """``sum_n fn(alpha, lam + 2 pi n / 3)`` -- the T projection's lambda sum."""
    return sum(fn(alpha, lam + n * TWO_PI_3) for n in range(3))


def harmonic_content(fn, alpha, n_lambda=720, tol=1e-10):
    """Which ``e^{i m lambda}`` harmonics ``fn(alpha, .)`` contains, at fixed alpha."""
    lam = np.arange(n_lambda) * 2.0 * np.pi / n_lambda
    coef = np.fft.rfft(np.array([fn(alpha, l) for l in lam])) / n_lambda
    return [m for m, c in enumerate(coef) if abs(c) > tol * max(1.0, abs(coef[0]))]


# ---------------------------------------------------------------------------
# Part 1 -- leading order, from the existing kernels
# ---------------------------------------------------------------------------


def leading_mode(kernel, l, m, part):
    """One (l, m) mode of a leading-order kernel as ``fn(alpha, lambda)``."""

    def fn(alpha, lam):
        ca, sa = gm._trig_powers(alpha)
        cb, sb = gm._trig_powers(lam)
        R, I = kernel(ca, sa, cb, sb)
        return (R if part == "R" else I)[l][m]

    return fn


def part1(alpha=0.7, lam=0.3):
    print("=" * 78)
    print("PART 1 -- leading order O(f^0), from galaxy_modulation.py's own kernels")
    print("=" * 78)
    print("For each mode: is sum_n [XX + 2 XY] zero, and if so, why?\n")
    print(f"  {'mode':>10s} {'sum_n XX':>13s} {'sum_n 2XY':>13s} {'total':>13s} "
          f"{'lambda harmonics of XX':>26s}")

    total_nonzero = []
    for l in range(gm.LMAX + 1):
        for m in range(l + 1):
            for part in ("R", "I"):
                if m == 0 and part == "I":
                    continue  # m=0 has no sine part
                xx = leading_mode(gm.kernel_XX, l, m, part)
                xy = leading_mode(gm.kernel_XY, l, m, part)
                s_xx = threefold(xx, alpha, lam)
                s_xy = 2.0 * threefold(xy, alpha, lam)
                tot = s_xx + s_xy
                harm = harmonic_content(xx, alpha)
                label = f"{l}{m}{'c' if part == 'R' else 's'}"
                if abs(s_xx) > 1e-12 or abs(s_xy) > 1e-12:
                    print(f"  {label:>10s} {s_xx:13.6f} {s_xy:13.6f} {tot:13.2e} "
                          f"{str(harm):>26s}")
                if abs(tot) > 1e-10:
                    total_nonzero.append(label)

    print()
    if total_nonzero:
        print(f"  !! modes with a NON-zero T projection: {total_nonzero}")
    else:
        print("  every mode nulls in T.")
    print("\n  Modes printed above are the ones that SURVIVE the threefold sum "
          "(harmonics\n  0, 3, 6 ...). Where both columns are non-zero and the "
          "total is not, the null is\n  a real cancellation between the self and "
          "cross kernels -- NOT harmonic selection.")

    # The one-point table above could be a coincidence. Verify the identity
    #     sum_n XY_lm(a, l + 2 pi n/3) == -1/2 sum_n XX_lm(a, l + 2 pi n/3)
    # over a grid, and separately check that it does NOT hold pointwise -- if it
    # did, the modulation's off-diagonals would be a flat -1/2 and the measured
    # rho_XY could not swing -0.70 .. -0.48 as it does.
    grid_a = np.linspace(0.0, 2 * np.pi, 23, endpoint=False)
    grid_l = np.linspace(0.0, 2 * np.pi, 19, endpoint=False)
    worst_summed, worst_pointwise, scale = 0.0, 0.0, 0.0
    for l_ in range(gm.LMAX + 1):
        for m_ in range(l_ + 1):
            for part_ in ("R", "I"):
                if m_ == 0 and part_ == "I":
                    continue
                xx = leading_mode(gm.kernel_XX, l_, m_, part_)
                xy = leading_mode(gm.kernel_XY, l_, m_, part_)
                for aa in grid_a:
                    for ll in grid_l:
                        sxx, sxy = threefold(xx, aa, ll), threefold(xy, aa, ll)
                        scale = max(scale, abs(sxx))
                        worst_summed = max(worst_summed, abs(sxy + 0.5 * sxx))
                        worst_pointwise = max(
                            worst_pointwise, abs(xy(aa, ll) + 0.5 * xx(aa, ll))
                        )
    print(f"\n  identity check over {len(grid_a)}x{len(grid_l)} phase points, all modes:")
    print(f"    max |sum_n XY + 1/2 sum_n XX|  = {worst_summed:.3e}   "
          f"(vs typical |sum_n XX| ~ {scale:.3g})")
    print(f"    max |XY + 1/2 XX| POINTWISE    = {worst_pointwise:.3e}   "
          "(non-zero => the -1/2 holds only AFTER the threefold sum)")
    return total_nonzero


# ---------------------------------------------------------------------------
# Part 2 -- O(f^2), transcribed from the note's appendix
# ---------------------------------------------------------------------------
#
# Transcribed by hand from LISA_Higher_Order_Response.pdf. Each returns the
# bracket only; the common f^2/f_*^2 prefactor is irrelevant to a null test and
# is omitted. ONLY the short expressions are here:
#
#   self  l=0 (p11), l=1 (p11), l=2 m=0 (p11)
#   cross l=1 (p25)
#
# The long ones (cross l=0 p25, cross l=2 p26-30, everything at l>=3) are NOT
# transcribed. They are 30-40 term expressions and OCR of them cannot be
# trusted for a sign-sensitive cancellation test.
SQ3 = np.sqrt(3.0)


def self_00(alpha, lam):
    return -21649.0 * np.sqrt(np.pi) / 26880.0


def self_10(alpha, lam):
    c = 27.0 * np.sqrt(3.0 * np.pi) / 4480.0
    return c * (np.cos(alpha) * np.cos(lam) + np.sin(alpha) * np.sin(lam))


def self_11c(alpha, lam):
    c = 27.0 * np.sqrt(2.0 * np.pi) / 8960.0
    return c * (-np.sin(alpha) * np.cos(alpha) * np.sin(lam) + 2.0 * np.cos(lam)
                - np.cos(alpha) ** 2 * np.cos(lam))


def self_11s(alpha, lam):
    c = 27.0 * np.sqrt(2.0 * np.pi) / 8960.0
    return c * (-np.cos(lam) * np.sin(alpha) * np.cos(alpha) + np.sin(lam)
                + np.sin(lam) * np.cos(alpha) ** 2)


def self_20(alpha, lam):
    c = 67.0 * np.sqrt(5.0 * np.pi) / 3584.0
    return c * (-2.0 * np.cos(alpha) ** 2 * np.cos(lam) ** 2 + np.cos(alpha) ** 2
                + np.cos(lam) ** 2
                - 2.0 * np.sin(lam) * np.cos(alpha) * np.sin(alpha) * np.cos(lam)
                + 439.0 / 402.0)


def cross_10(alpha, lam):
    c = 27.0 * np.sqrt(np.pi) / 17920.0
    return c * (-2.0 * SQ3 * np.cos(lam) * np.cos(alpha)
                - 6.0 * np.sin(alpha) * np.cos(lam)
                - 2.0 * SQ3 * np.sin(lam) * np.sin(alpha)
                + 6.0 * np.cos(alpha) * np.sin(lam))


def cross_11c(alpha, lam):
    c = 27.0 * np.sqrt(2.0 * np.pi) / 17920.0
    return c * (np.sin(alpha) * np.cos(alpha) * np.cos(lam) * SQ3
                - np.cos(alpha) ** 2 * np.sin(lam) * SQ3
                + 2.0 * np.sin(lam) * SQ3
                + np.sin(lam) * np.sin(alpha) * np.cos(alpha)
                + np.cos(lam) * np.cos(alpha) ** 2
                - 2.0 * np.cos(lam))


def cross_11s(alpha, lam):
    c = 27.0 * np.sqrt(2.0 * np.pi) / 17920.0
    return c * (-np.sin(alpha) * np.cos(alpha) * np.sin(lam) * SQ3
                - np.cos(alpha) ** 2 * np.cos(lam) * SQ3
                - np.cos(lam) * SQ3
                + np.sin(alpha) * np.cos(alpha) * np.cos(lam)
                - np.cos(alpha) ** 2 * np.sin(lam)
                - np.sin(lam))


MODES_OF2 = {
    "00": (self_00, None),
    "10": (self_10, cross_10),
    "11c": (self_11c, cross_11c),
    "11s": (self_11s, cross_11s),
    "20": (self_20, None),
}


def part2(alpha=0.7, lam=0.3):
    print()
    print("=" * 78)
    print("PART 2 -- O(f^2), transcribed appendix modes (PARTIAL coverage)")
    print("=" * 78)
    print(f"  {'mode':>6s} {'sum_n self':>14s} {'sum_n 2*cross':>15s} {'total':>13s} "
          f"{'lambda harmonics':>20s}   verdict")
    for label, (self_fn, cross_fn) in MODES_OF2.items():
        s_self = threefold(self_fn, alpha, lam)
        harm = harmonic_content(self_fn, alpha)
        if cross_fn is None:
            print(f"  {label:>6s} {s_self:14.6f} {'(not transcribed)':>15s} "
                  f"{'?':>13s} {str(harm):>20s}   "
                  + ("survives the comb -> DECIDED BY THE CROSS TERM"
                     if abs(s_self) > 1e-12 else "self part vanishes"))
            continue
        s_cross = 2.0 * threefold(cross_fn, alpha, lam)
        tot = s_self + s_cross
        verdict = "nulls" if abs(tot) < 1e-10 else "BREAKS THE NULL"
        if abs(s_self) < 1e-12 and abs(s_cross) < 1e-12:
            verdict = "nulls trivially (comb kills every harmonic)"
        print(f"  {label:>6s} {s_self:14.6f} {s_cross:15.6f} {tot:13.2e} "
              f"{str(harm):>20s}   {verdict}")


def main():
    rng = np.random.default_rng(0)
    a, l = float(rng.uniform(0, 2 * np.pi)), float(rng.uniform(0, 2 * np.pi))
    print(f"evaluated at a random phase point alpha={a:.4f}, lambda={l:.4f}\n")
    part1(a, l)
    part2(a, l)


if __name__ == "__main__":
    main()
