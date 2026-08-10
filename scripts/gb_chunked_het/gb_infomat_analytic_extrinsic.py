#!/usr/bin/env python
"""Are the GB extrinsic derivatives EXACT linear combinations of 4 basis waveforms?

Motivation
----------
``info_matrix_ll.information_matrix_from_ll`` costs ``1 + 2*nd + 4*C(nd,2)``
likelihood rows -- 163 at nd=9. It already batches all of them into ONE
``call_ll`` and already exploits the ``i<j`` symmetry, so the only way to make
it cheaper is to need fewer DIMENSIONS numerically.

The GB waveform makes that possible. From ``SharedMemoryGBGPU.cu``::

    Aplus  = amp * (1 + cos^2 iota)
    Across = -2 * amp * cos iota
    DP = { Aplus*cos2psi, -Across*sin2psi }
    DC = { -Aplus*sin2psi, -Across*cos2psi }

and ``phi0`` enters the phase purely additively (``argS = phi0 + ...``). So the
four extrinsic parameters ``(amp, iota, psi, phi0)`` enter ONLY through four
amplitude coefficients multiplying waveforms that depend on the INTRINSIC
parameters alone -- the Jaranowski-Krol / F-statistic decomposition::

    h(theta_int, theta_ext) = sum_k a_k(theta_ext) h_k(theta_int)

If that holds, then for every extrinsic theta::

    dh/dtheta = sum_k (da_k/dtheta) h_k     -> EXACT, zero waveform evaluations

and the extrinsic block of the Fisher matrix is a pure contraction of
``da_k/dtheta`` with the 4x4 Gram matrix ``M_kl = <h_k|h_l>`` -- which the
F-stat path (``get_fstat_ll_wdm``) ALREADY computes.

What this script tests
----------------------
Deliberately CONVENTION-FREE: it never hand-derives ``a_k``. It only asks
whether the relevant vectors lie in a 4-dimensional span, which is invariant
under any linear change of basis (so a result in TD carries to FD and WDM).

1. SPAN       Build 4 basis waveforms at 4 extrinsic configurations with the
              intrinsic parameters held fixed. Project a 5th, random extrinsic
              waveform onto that span and report the residual. ~1e-16 => the
              decomposition is exact and 4-dimensional.
2. DERIVS     Finite-difference dh/dtheta for each extrinsic theta and for an
              intrinsic CONTROL (f0). Project each onto the span. Extrinsic
              derivatives must be ~0 outside it; the control must not be.
3. FISHER     Build the extrinsic 4x4 noise-weighted Fisher block two ways --
              finite differences vs the analytic contraction through the Gram
              matrix -- and compare entrywise.

Run (CPU, thread pools pinned per the laptop budget):

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    python gb_infomat_analytic_extrinsic.py
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

import gbgpu  # noqa: F401  (registers the gbgpu_* backend namespace)

from lisatools.detector import ESAOrbits
from lisatools.utils.constants import YRSID_SI
from lisatools.response.tdiconfig import TDIConfig
from lisatools.response.tdionfly import GBTDIonTheFly
from lisatools.datacontainer import DataResidualArray
from lisatools.diagnostic import inner_product
from lisatools.sensitivity import XYZ2SensitivityMatrix
from lisatools.domains import TDSettings, TDSignal, FDSettings

# GBGPU parameter order: [amp, f0, fdot, fddot, phi0, iota, psi, lam, beta]
IAMP, IF0, IFDOT, IFDDOT, IPHI0, IIOTA, IPSI, ILAM, IBETA = range(9)
EXTRINSIC = {"ln_amp": IAMP, "phi0": IPHI0, "iota": IIOTA, "psi": IPSI}

BASE = np.array([
    3.0e-23,        # amp
    4.0e-3,         # f0
    1.0e-17,        # fdot
    0.0,            # fddot
    2.09802430298,  # phi0
    0.7853981634,   # iota
    1.234019814,    # psi
    4.09808143,     # lam
    0.3,            # beta
])


def _f(x):
    if hasattr(x, "get"):
        x = x.get()
    a = np.asarray(x).reshape(-1)
    assert a.size == 1
    return float(a[0])


class Gen:
    """TD-accurate GB generator + FD projection, on the installed classes."""

    def __init__(self, Tobs_yr=0.25, dt=15.0, n_sparse=None):
        if n_sparse is None:
            n_sparse = int(os.environ.get("N_SPARSE", 4096))
        self.dt = dt
        N = int(Tobs_yr * YRSID_SI / dt)
        self.N = N - (N % 2)
        self.Tobs = self.N * dt
        self.t_arr = np.arange(self.N) * dt
        self.td_set = TDSettings(self.N, dt, force_backend="cpu")
        freqs = np.fft.rfftfreq(self.N, dt)
        self.fd_set = FDSettings(len(freqs), freqs[1] - freqs[0],
                                 min_freq=1e-4, max_freq=2.0e-2,
                                 force_backend="cpu")
        self.sens = XYZ2SensitivityMatrix(self.fd_set, model="scirdv1")
        self.gen = GBTDIonTheFly(
            np.linspace(0.0, self.t_arr[-1], n_sparse), self.Tobs, 0.0,
            1.0 / dt, 1, tdi_config=TDIConfig("2nd generation"),
            orbits=ESAOrbits(force_backend="cpu"), tdi_chan="XYZ",
            force_backend="cpu")

    def td(self, p) -> np.ndarray:
        """(3, N) time-domain XYZ, flattened to a single real vector."""
        out = self.gen(*np.asarray(p, dtype=float).reshape(1, 9).T,
                       convert_to_ra_dec=False, return_spline=True)
        a = np.asarray(out.eval_tdi(self.t_arr))
        if a.ndim == 3 and a.shape[0] == 1:
            a = a[0]
        return np.ascontiguousarray(a, dtype=float)

    def vec(self, p) -> np.ndarray:
        return self.td(p).reshape(-1)

    def ip(self, a_td, b_td) -> float:
        """Noise-weighted <a|b> through the installed diagnostic path."""
        fa = TDSignal(a_td, settings=self.td_set).transform(self.fd_set)
        fb = TDSignal(b_td, settings=self.td_set).transform(self.fd_set)
        return _f(inner_product(DataResidualArray(fa), DataResidualArray(fb),
                                psd=self.sens))


def basis_params(base):
    """4 extrinsic configurations spanning the amplitude space.

    The classic F-stat choice: ``iota = pi/2`` kills ``Across``, and
    ``psi in {0, pi/4}`` x ``phi0 in {0, pi/2}`` gives four independent
    amplitude combinations. Any 4 spanning configurations would do -- the
    span test does not depend on which.
    """
    out = []
    for psi in (0.0, 0.25 * np.pi):
        for phi0 in (0.0, 0.5 * np.pi):
            p = base.copy()
            p[IIOTA] = 0.5 * np.pi
            p[IPSI] = psi
            p[IPHI0] = phi0
            out.append(p)
    return out


def project(B_pinv, B, v):
    """Least-squares projection of v onto span(B). Returns (coeffs, resid_frac)."""
    c = B_pinv @ v
    r = v - B @ c
    return c, float(np.linalg.norm(r) / max(np.linalg.norm(v), 1e-300))


def displace(p, idx, h):
    q = p.copy()
    if idx == IAMP:          # log-amplitude step
        q[IAMP] *= np.exp(h)
    else:
        q[idx] += h
    return q


def main():
    g = Gen()
    print(f"[setup] N={g.N}  dt={g.dt}  Tobs={g.Tobs / YRSID_SI:.4f} yr")

    # ---------------------------------------------------------------- 1. SPAN
    print("\n== 1. SPAN: is the extrinsic family exactly 4-dimensional? ==")
    B = np.stack([g.vec(p) for p in basis_params(BASE)], axis=1)   # (3N, 4)
    B_pinv = np.linalg.pinv(B)
    print(f"   basis {B.shape},  cond = {np.linalg.cond(B):.3e}")

    rng = np.random.default_rng(7)
    worst = 0.0
    for trial in range(5):
        p = BASE.copy()
        p[IAMP] = BASE[IAMP] * float(np.exp(rng.uniform(-1, 1)))
        p[IPHI0] = float(rng.uniform(0, 2 * np.pi))
        p[IIOTA] = float(np.arccos(rng.uniform(-1, 1)))
        p[IPSI] = float(rng.uniform(0, np.pi))
        _, res = project(B_pinv, B, g.vec(p))
        worst = max(worst, res)
        print(f"   random extrinsic #{trial + 1}: residual outside span = {res:.3e}")
    print(f"   -> worst {worst:.3e}  "
          f"({'EXACT (4-dimensional)' if worst < 1e-10 else 'NOT exact'})")

    # -------------------------------------------------------------- 2. DERIVS
    print("\n== 2. DERIVS: does dh/dtheta lie in that span? ==")
    print("   (extrinsic must; the f0 control must NOT)")
    steps = {"ln_amp": 1e-4, "phi0": 1e-5, "iota": 1e-5, "psi": 1e-5}
    dh = {}
    for name, idx in EXTRINSIC.items():
        h = steps[name]
        d = (g.vec(displace(BASE, idx, h)) - g.vec(displace(BASE, idx, -h))) / (2 * h)
        dh[name] = d
        _, res = project(B_pinv, B, d)
        print(f"   d/d {name:7s}: residual outside span = {res:.3e}  "
              f"{'IN SPAN' if res < 1e-8 else 'outside'}")
    h0 = 1e-4 / g.Tobs
    d_f0 = (g.vec(displace(BASE, IF0, h0)) - g.vec(displace(BASE, IF0, -h0))) / (2 * h0)
    _, res_f0 = project(B_pinv, B, d_f0)
    print(f"   d/d f0     : residual outside span = {res_f0:.3e}  "
          f"{'(control behaved: intrinsic is NOT in the span)' if res_f0 > 1e-3 else '(control FAILED)'}")

    # -------------------------------------------------------------- 3. FISHER
    print("\n== 3. FISHER: extrinsic 4x4 block, numerical vs analytic ==")
    names = list(EXTRINSIC)
    # Numerical: noise-weighted Gram of the finite-difference derivatives.
    F_num = np.zeros((4, 4))
    for i, a in enumerate(names):
        for j in range(i, 4):
            v = g.ip(dh[a].reshape(3, -1), dh[names[j]].reshape(3, -1))
            F_num[i, j] = F_num[j, i] = v

    # Analytic: each derivative is B @ c, so <d_i|d_j> = c_i^T M c_j with
    # M_kl = <h_k|h_l> -- the F-stat Gram matrix, already computed in-fit.
    C = np.stack([B_pinv @ dh[a] for a in names], axis=1)           # (4, 4)
    M = np.zeros((4, 4))
    Bcols = [B[:, k].reshape(3, -1) for k in range(4)]
    for k in range(4):
        for l in range(k, 4):
            M[k, l] = M[l, k] = g.ip(Bcols[k], Bcols[l])
    F_ana = C.T @ M @ C

    den = np.maximum(np.abs(F_num), 1e-300)
    rel = np.abs(F_ana - F_num) / den
    print(f"   {'':9s}" + "".join(f"{n:>13s}" for n in names))
    for i, a in enumerate(names):
        print(f"   {a:9s}" + "".join(f"{F_num[i, j]:13.4e}" for j in range(4)))
    print(f"\n   max relative difference (analytic vs numerical) = {rel.max():.3e}")
    print(f"   -> {'MATCH' if rel.max() < 1e-6 else 'MISMATCH'}")

    # ------------------------------------------------------------- cost model
    print("\n== cost ==")
    for nd_int in (4, 5):
        rows_full = 1 + 2 * (nd_int + 4) + 4 * ((nd_int + 4) * (nd_int + 3) // 2)
        rows_int = 1 + 2 * nd_int + 4 * (nd_int * (nd_int - 1) // 2)
        print(f"   {nd_int} intrinsic + 4 extrinsic: {rows_full} rows now -> "
              f"{rows_int} if the extrinsic block is analytic "
              f"({rows_full / rows_int:.1f}x fewer)")


if __name__ == "__main__":
    main()
