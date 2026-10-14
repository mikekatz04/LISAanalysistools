"""Reproduce the phentax t(f)-bisection failure that killed the 1yr flagship job 760.

phentax's ``compute_phase_coeffs_22`` root-finds the time at which the (2,2)
frequency equals ``f_min`` / ``f_ref`` (``get_time_of_frequency``: an optimistix
``Bisection`` with ``atol = rtol = 1e-12`` and ``max_steps = 1000``). lisatools
always passes ``t_min`` and ``t_ref``, so the solve's result is discarded -- but
the selection is a ``lax.cond`` under ``jax.vmap``, which evaluates BOTH branches,
so the solve runs (and can raise) for every row anyway.

The function being bisected, ``2 pi M f - omega(t)``, has a JUMP at ``t = tEarly``
(phentax evaluates ``omega(t - tt0)`` below the PN / pseudo-PN junction). When the
target ``2 pi M f_min`` falls inside that jump, the bisection interval collapses
onto the discontinuity and ``|f|`` never gets below ``atol``: after 1000 steps
equinox raises ``The maximum number of steps was reached in the nonlinear
solver`` and the whole vmapped batch -- then the rank, then the run -- dies.

This script computes the junction geometry for one ``(q, s1z, s2z)`` and scans
the total mass across the predicted sliver. Expected output (phentax HEAD
2026-08-28, f_min = 7e-5 Hz, MBHB 10's q = 3.621, spins 0.732 / 0.600)::

    eta=0.16957 tEarly=-2.096534e+05 M  tt0=6.862995e+02 M
    omega just above tEarly = 9.1580956e-03; just below = 9.1467959e-03 (+1.23e-03 rel)
    predicted failing sliver in total mass: [4.222227e+06, 4.227443e+06] Msun
    inside the sliver : {'MAXSTEPS': 38, 'ok': 2}
    outside the sliver: {'ok': 40}

Every leaf has such a sliver at its own ``(eta, s1z, s2z)``; a hot-rung walker
of a loosely constrained leaf (MBHB 10: inspiral-only in the 1yr data) walks
into it eventually. The global fit now rejects such a row
(``MBHBatchedLikeMove._score_rows_isolating_failures``); the upstream fix is
``optx.root_find(..., throw=False)`` or a continuous ``time_of_freq``.

Usage (CPU, ~1 min)::

    python scripts/mbh/phentax_tf_bisection_sliver.py --q 3.621 --s1 0.732 --s2 0.600
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

jax.config.update("jax_enable_x64", True)

from phentax.core.phase import imr_omega  # noqa: E402
from phentax.waveform import IMRPhenomTHM  # noqa: E402

from lisatools.utils.constants import MTSUN_SI  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--f-min", type=float, default=7e-5, help="start frequency [Hz] (MBH_START_FREQ)")
    ap.add_argument("--q", type=float, default=3.621, help="mass ratio m1/m2 >= 1")
    ap.add_argument("--s1", type=float, default=0.732)
    ap.add_argument("--s2", type=float, default=0.600)
    ap.add_argument("--T-days", type=float, default=90.0, help="waveform duration (t_min = -T)")
    ap.add_argument("--tol", type=float, default=1e-12, help="phentax atol = rtol (MBH phenom_tol)")
    ap.add_argument("--M-ref", type=float, default=2.83e6, help="mass at which the coefficients are read")
    args = ap.parse_args()

    T = args.T_days * 86400.0
    wf = IMRPhenomTHM(higher_modes=[21, 33, 44], include_negative_modes=True, t_low_fit=True,
                      coarse_grain=False, atol=args.tol, rtol=args.tol, T=T)

    def coeffs(M):
        m1 = M * args.q / (1.0 + args.q)
        m2 = M / (1.0 + args.q)
        p = wf._process_parameters(
            jnp.atleast_1d(m1), jnp.atleast_1d(m2), jnp.atleast_1d(args.s1), jnp.atleast_1d(args.s2),
            jnp.atleast_1d(1000.0), jnp.atleast_1d(0.3), jnp.atleast_1d(1.0), jnp.atleast_1d(0.5),
            delta_t=5.0, t_min=jnp.atleast_1d(-T), t_ref=jnp.atleast_1d(0.0),
            f_min=args.f_min, f_ref=args.f_min,
        )
        return jax.vmap(wf._compute_coeffs_22)(p)   # the production call (gridaligned.py)

    wfp, _, pc = coeffs(args.M_ref)
    eta = float(wfp.eta[0])
    tEarly = float(pc.tEarly[0])
    tt0 = float(pc.tt0[0])
    pc1 = jax.tree_util.tree_map(lambda a: a[0], pc)
    w_hi = float(imr_omega(jnp.asarray(tEarly), eta, pc1))
    w_lo = float(imr_omega(jnp.asarray(tEarly - tt0), eta, pc1))
    print(f"eta={eta:.5f} tEarly={tEarly:.6e} M  tt0={tt0:.6e} M")
    print(f"omega just above tEarly = {w_hi:.7e}; just below = {w_lo:.7e} ({(w_hi - w_lo) / w_hi:+.2e} rel)")
    lo, hi = sorted([w_lo, w_hi])
    M_lo = lo / (2 * np.pi * args.f_min * MTSUN_SI)
    M_hi = hi / (2 * np.pi * args.f_min * MTSUN_SI)
    print(f"predicted failing sliver in total mass: [{M_lo:.6e}, {M_hi:.6e}] Msun")

    Ms = np.concatenate([
        np.linspace(M_lo * 0.995, M_lo, 20, endpoint=False),
        np.linspace(M_lo, M_hi, 40),
        np.linspace(M_hi, M_hi * 1.005, 21)[1:],
    ])
    res = []
    for M in Ms:
        try:
            jax.block_until_ready(coeffs(float(M)))
            res.append("ok")
        except Exception as exc:  # noqa: BLE001 -- the equinox runtime error, wrapped or bare
            s = str(exc)
            res.append("MAXSTEPS" if "maximum number of steps" in s
                       else ("NOROOT" if "not contained" in s else "OTHER"))
    res = np.array(res)
    inside = (Ms >= M_lo) & (Ms <= M_hi)
    print("inside the sliver :", dict(zip(*np.unique(res[inside], return_counts=True))))
    print("outside the sliver:", dict(zip(*np.unique(res[~inside], return_counts=True))))
    bad = np.where(res != "ok")[0]
    if bad.size:
        print(f"failing M range: [{Ms[bad].min():.6e}, {Ms[bad].max():.6e}]")


if __name__ == "__main__":
    main()
