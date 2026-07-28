#!/usr/bin/env python
"""Minimal three-regime parity test for the TD-spline TDI-on-the-fly kernel.

Isolates the kernel from all MBH/phentax/orbit-trim machinery: synthetic smooth
amp/phase splines + an analytic EqualArmlengthOrbits, evaluated at N in
{2000 (<=48 KB legacy), 4000 (opt-in shared), 20000 (global scratch)}.

  BACKEND=cpu     python spline_tdionfly_parity.py
  BACKEND=cuda12x python spline_tdionfly_parity.py

If N=2000 works but N=4000 crashes on GPU -> the bug is in the kernel body at
N>2340 (the size that never ran before the launch fix). If all N work -> the
crash is specific to the MBH data path, not the kernel.
"""
import os
import numpy as np

BACKEND = os.environ.get("BACKEND", "cpu")

from lisatools import get_backend
from lisatools.detector import EqualArmlengthOrbits
from lisatools.response.tdiconfig import TDIConfig
from lisatools.response.tdionfly import TDTDIonTheFly

YRSID = 31558149.7635456
DT = 10.0
NMODES = 2


def build(N):
    """Synthetic on-the-fly TDI at eval length N. Returns the output TDI array
    brought to host (numpy)."""
    xp = get_backend(BACKEND).xp

    t0 = 0.4 * YRSID                 # well inside one year
    T = N * DT
    tc = t0 + 0.6 * T               # "merger"-like bump centre

    # knot grid: pad beyond the eval window so retarded reads stay in-domain
    pad = 3000.0
    n_knots = N + int(2 * pad / DT) + 4
    tk = t0 - pad + DT * xp.arange(n_knots, dtype=float)
    amp1 = 1e-21 * xp.exp(-((tk - tc) / (0.2 * T + 1.0)) ** 2)
    ph1 = 2.0 * np.pi * (1e-3 * (tk - t0) + 0.5e-9 * (tk - t0) ** 2)

    nta = xp.repeat(tk[None, :], NMODES, axis=0)
    amp = xp.repeat(amp1[None, :], NMODES, axis=0)
    phase = xp.repeat(ph1[None, :], NMODES, axis=0)

    eval_t = t0 + DT * xp.arange(N, dtype=float)
    eval_t = xp.repeat(eval_t[None, :], NMODES, axis=0)

    orbit = EqualArmlengthOrbits(force_backend=BACKEND)
    # Build the linear-interp position/ltt tables + device OrbitsWrap explicitly
    # (the on-the-fly kernel reads them); do not rely on lazy config, which the
    # device path may not trigger before the kernel dereferences the orbit.
    orbit.configure(linear_interp_setup=True, dt=DT)
    tdi_config = TDIConfig("2nd generation", force_backend=BACKEND)

    g = TDTDIonTheFly(
        eval_t, amp, phase,
        sampling_frequency=1.0 / DT, num_sub=NMODES, t_input=nta,
        tdi_config=tdi_config, orbits=orbit, force_backend=BACKEND,
    )
    out = g(np.zeros(NMODES), np.zeros(NMODES),
            np.full(NMODES, 1.0), np.full(NMODES, 0.3), return_spline=False)
    return out


def to_host(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


if __name__ == "__main__":
    print(f"BACKEND={BACKEND}", flush=True)
    for N in (2000, 4000, 20000):
        tier = "legacy<=48KB" if 21 * N <= 48 * 1024 else "opt-in/global>48KB"
        print(f"\n=== N={N} ({tier}) ===", flush=True)
        out = build(N)
        amp = to_host(out.tdi_amp)
        ph = to_host(out.tdi_phase)
        print(f"  tdi_amp shape {amp.shape} finite={np.isfinite(amp).all()} "
              f"max|amp|={np.nanmax(np.abs(amp)):.3e} "
              f"max|phase|={np.nanmax(np.abs(ph)):.3e}", flush=True)
        print(f"  N={N} OK", flush=True)
    print("\nALL N COMPLETED", flush=True)
