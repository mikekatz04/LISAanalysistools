#!/usr/bin/env python
"""Prototype: a MOVING time-frequency window for the SOBBH signal-heterodyne.

A SOBBH chirps across many WDM frequency layers over the observation (tens to
hundreds; see the printout), so the GB-style FIXED 5-layer band around f_low
misses most of the power for long windows. This prototype validates the fix the
user proposed: heterodyne only the fraction of time-pixels of each layer where
the carrier sits there -- i.e. a DIAGONAL band in (m, n) that follows the
carrier track f(t).

Concept proof (pure Python, dense WDM, no polyphase/bin-fold needed -- those are
already validated separately):

  1. Build a year-long chirping SOBBH injection; dense complex WDM c0 (=ref).
  2. Carrier track m_track[n] = argmax_m sum_c |c0[c,m,n]|^2  (data-driven; no
     waveform-frequency call). Report layers crossed.
  3. For a few f_low perturbations, compute the WDM logL three ways and compare
     to the FULL dense logL:
       - FULL        : sum over all (m,n)            [reference truth]
       - MOVING WIN  : sum over (n, m in m_track[n] +/- M_local)
       - FIXED band  : sum over (m_floor(f0) +/- M, all n)   [the GB way]
     The moving window should recover FULL; the fixed band should not.

Run::  python sobbh_moving_window_proto.py
Env:   M_LOCAL (2), TARGET_LOGL list fixed, NF/NT
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy.signal.windows import tukey as _tukey

from lisatools.detector import ESAOrbits
from lisatools.domains import TDSettings, TDSignal, WDMSettings
from lisatools.sensitivity import XYZ2SensitivityMatrix
from lisatools.utils.constants import YRSID_SI

from lisatools.response.tdiconfig import TDIConfig
from lisatools.response.tdionfly import SOBBHTDIonTheFly

import bbhx  # noqa: F401


def main():
    M_LOCAL = int(os.environ.get("M_LOCAL", "2"))
    backend = "cpu"
    dt = 10.0
    Nf = int(os.environ.get("NF", "1460"))
    Nt = int(os.environ.get("NT", "2560"))
    Nobs = Nf * Nt
    EC = 20
    t_start = int(0.5 * YRSID_SI / dt) * dt
    t_arr = np.arange(Nobs) * dt + t_start
    Tobs = Nobs * dt

    orbits = ESAOrbits(force_backend=backend)
    tdi_config = TDIConfig("2nd generation", force_backend=backend)
    t_tdi = np.linspace(t_arr[0], t_arr[-1], 16384)
    gen = SOBBHTDIonTheFly(t_tdi, Tobs, t_start, 1.0 / dt, 1,
                           tdi_config=tdi_config, orbits=orbits, tdi_chan="XYZ",
                           force_backend=backend)

    def td(p):
        sp = gen(*np.asarray(p, float).reshape(11, 1), convert_to_ra_dec=False, return_spline=True)
        return np.asarray(sp.eval_tdi(t_arr))[0]

    td_set = TDSettings(Nobs, dt, force_backend=backend)
    window = _tukey(Nobs, alpha=0.05).astype(float)
    wdm = WDMSettings(Nf, Nt, dt, t0=t_start, min_freq=1e-4, max_freq=35e-3,
                      min_time=EC * Nf * dt, max_time=(Nt - EC) * Nf * dt,
                      is_complex=False, force_backend=backend)
    layer_df = wdm.layer_df
    ind_min_f = int(wdm.ind_min_f)

    # year-long STRONGLY-chirping SOBBH (high f + massive -> big sweep over Tobs)
    f_low = float(os.environ.get("F_LOW_HZ", "0.018"))
    ref = np.array([60.0, 55.0, 0.1, 0.2, 4.0e8, f_low, 1.1,
                    np.arccos(0.3), 0.7, 3.1, np.arcsin(0.2)])
    print(f"[inj] Tobs={Tobs/YRSID_SI:.2f}yr  f_low={f_low*1e3:.3f}mHz  m1=60 m2=55", flush=True)

    inv = np.asarray(XYZ2SensitivityMatrix(wdm, model="scirdv1").invC)
    inv = np.where(np.isfinite(inv), inv, 0.0)             # (3,3,Nf_a,Nt_a)

    d = np.asarray(TDSignal(td(ref), settings=td_set).transform(wdm, window=window).arr)  # (3,Nf_a,Nt_a)
    nch, Nf_a, Nt_a = d.shape

    # REFERENCE-GUIDED pixel mask (per user direction 2026-06-19): use the
    # reference signal to pick which (m,n) pixels to evaluate, with a WIDTH
    # around it in BOTH time and frequency. Mask = (|c0| significant) dilated by
    # +/- W_M layers and +/- W_N time pixels. This follows the chirp trajectory
    # (a diagonal tube) and captures the WDM spread automatically -- robust to a
    # fast sweep that a single-carrier argmax track would under-cover.
    from scipy.ndimage import binary_dilation
    W_M = int(os.environ.get("W_M", "3"))
    W_N = int(os.environ.get("W_N", "3"))
    power_mn = (d**2).sum(axis=0)                          # (Nf_a, Nt_a)
    thresh = float(os.environ.get("THRESH_FRAC", "1e-6")) * power_mn.max()
    sig = power_mn > thresh
    struct = np.ones((2 * W_M + 1, 2 * W_N + 1), dtype=bool)
    mm_grid = binary_dilation(sig, structure=struct)      # reference-guided tube
    occ = np.where(mm_grid.any(axis=1))[0]
    print(f"[track] reference power spans local m in [{occ.min()},{occ.max()}] "
          f"({occ.max() - occ.min()} layers) of {Nf_a}; tube width W_M={W_M} W_N={W_N}",
          flush=True)

    def ip_full(a, b):
        return float(np.einsum("cmn,cdmn,dmn->", a, inv, b))

    def ip_masked(a, b, mask_mn):
        am = a * mask_mn[None]; bm = b * mask_mn[None]
        return float(np.einsum("cmn,cdmn,dmn->", am, inv, bm))

    d_d = ip_full(d, d)

    # fixed-band mask (the GB way): |m - m_floor(f0)| <= M_LOCAL, all n
    m0 = int(round(f_low / layer_df)) - ind_min_f
    fb = (np.abs(np.arange(Nf_a) - m0) <= M_LOCAL)[:, None] & np.ones((1, Nt_a), bool)

    frac_mov = mm_grid.mean()
    frac_fix = fb.mean()
    print(f"[cost] reference-guided tube = {frac_mov*100:.2f}% of grid; "
          f"fixed-band = {frac_fix*100:.2f}%", flush=True)

    print(f"\n   {'pert':>14s} {'logL_full':>13s} {'logL_moving':>13s} {'logL_fixed':>13s} "
          f"{'|mov-full|':>11s} {'|fix-full|':>11s}", flush=True)
    for label, delta in [("zero", 0.0), ("df0+0.1lf", 0.1 * layer_df),
                         ("df0+1lf", 1.0 * layer_df), ("df0+5lf", 5.0 * layer_df)]:
        p = ref.copy(); p[5] = f_low + delta
        h = np.asarray(TDSignal(td(p), settings=td_set).transform(wdm, window=window).arr)
        # full
        dh_f = ip_full(d, h); hh_f = ip_full(h, h)
        ll_full = dh_f - 0.5 * hh_f - 0.5 * d_d
        # moving window (mask both d and h to the track band)
        dh_m = ip_masked(d, h, mm_grid); hh_m = ip_masked(h, h, mm_grid)
        ll_mov = dh_m - 0.5 * hh_m - 0.5 * d_d
        # fixed band
        dh_x = ip_masked(d, h, fb); hh_x = ip_masked(h, h, fb)
        ll_fix = dh_x - 0.5 * hh_x - 0.5 * d_d
        print(f"   {label:>14s} {ll_full:+13.4e} {ll_mov:+13.4e} {ll_fix:+13.4e} "
              f"{abs(ll_mov - ll_full):11.3e} {abs(ll_fix - ll_full):11.3e}", flush=True)

    print("\n[interpretation] moving window tracks FULL logL; fixed band misses the "
          "chirped power -> large error. This is why SOBBH get_ll needs the moving "
          "time-frequency window (a per-time-pixel layer band following the carrier).",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
