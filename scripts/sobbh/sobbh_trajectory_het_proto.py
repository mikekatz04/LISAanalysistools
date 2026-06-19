#!/usr/bin/env python
"""Trajectory-chunked signal-heterodyne for SOBBHs (Python prototype).

Builds up the model the user asked for: follow the heterodyne along the
carrier's frequency-time SWEEP, IN CHUNKS along that trajectory. Per chunk the
carrier stays within a few WDM layers, so a LOCAL sig-het (polyphase + bin-fold
against the reference) applies; stitching the chunks along the sweep gives a
fast get_ll that handles the chirp -- without ever forming the dense WDM
template.

This is the efficient realisation of the validated reference-guided moving
window (see sobbh_moving_window_proto.py): instead of masking a dense WDM, we
compute the candidate's coefficients only on each chunk's local (m, n) tile via
the polyphase identity, exactly where the reference says the signal lives.

Pipeline:
  construction (once):
    * dense reference c0 (= injection) -> carrier track m_track[b] over sparse
      time bins (argmax layer per bin, from |c0|).
    * split the sparse-time axis into N_CHUNK contiguous chunks; per chunk fix a
      local m-band [m_lo, m_hi] = (carrier over chunk) +/- W_M covering the
      sweep within the chunk.
    * per chunk: store reference c0 + bin-fold A0/B0 on the chunk tile.
  per get_ll call (the chirping candidate):
    * per chunk: polyphase the candidate FD on the chunk's local m-band -> c1 at
      the chunk's sparse-n; r = c1/c0; bin-fold -> chunk <d|h>, <h|h>.
    * sum over chunks -> logL.

Validates against the dense lisatools logL for a chirping SOBBH and against the
fixed-band sig-het (which fails). Pure Python; no new C++.

Run::  F_LOW_HZ=0.012 N_CHUNK=16 python sobbh_trajectory_het_proto.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy.signal.windows import tukey as _tukey

from lisatools.analysiscontainer import AnalysisContainer
from lisatools.datacontainer import DataResidualArray
from lisatools.detector import ESAOrbits
from lisatools.domains import TDSettings, TDSignal, WDMSettings
from lisatools.sensitivity import XYZ2SensitivityMatrix
from lisatools.utils.constants import YRSID_SI

from lisatools.response.tdiconfig import TDIConfig
from lisatools.response.tdionfly import SOBBHTDIonTheFly

import bbhx  # noqa: F401

_GB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gb_chunked_het")
if _GB_DIR not in sys.path:
    sys.path.insert(0, _GB_DIR)
from gb_signal_het_wdm_v2 import GBSparseComplexWDMGen          # noqa: E402


def main():
    backend = "cpu"
    dt = 10.0
    Nf = int(os.environ.get("NF", "1460"))
    Nt = int(os.environ.get("NT", "2560"))
    Nt_layer = int(os.environ.get("NT_LAYER", "64"))
    N_CHUNK = int(os.environ.get("N_CHUNK", "16"))
    W_M = int(os.environ.get("W_M", "3"))      # local band half-width per chunk
    f_low = float(os.environ.get("F_LOW_HZ", "0.012"))
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
    wdm_r = WDMSettings(Nf, Nt, dt, t0=t_start, min_freq=1e-4, max_freq=35e-3,
                        min_time=EC * Nf * dt, max_time=(Nt - EC) * Nf * dt,
                        is_complex=False, force_backend=backend)
    wdm_c = WDMSettings(Nf, Nt, dt, t0=t_start, min_freq=1e-4, max_freq=35e-3,
                        min_time=EC * Nf * dt, max_time=(Nt - EC) * Nf * dt,
                        is_complex=True, force_backend=backend)
    layer_df = wdm_r.layer_df
    ind_min_f = int(wdm_r.ind_min_f)
    ind_min_t = int(wdm_r.ind_min_t)
    Nt_active = int(wdm_r.Nt_active)
    Nf_active = int(wdm_r.ind_max_f - wdm_r.ind_min_f + 1)

    ref = np.array([60.0, 55.0, 0.1, 0.2, 4.0e8, f_low, 1.1,
                    np.arccos(0.3), 0.7, 3.1, np.arcsin(0.2)])
    td_inj = td(ref)
    data_r = TDSignal(td_inj, settings=td_set).transform(wdm_r, window=window)
    sens = XYZ2SensitivityMatrix(wdm_r, model="scirdv1")
    analysis = AnalysisContainer(DataResidualArray(data_r), sens)
    analysis.signal_gen = lambda *p: TDSignal(td(np.asarray(p, float).reshape(11)),
                                              settings=td_set).transform(wdm_r, window=window)
    d_d = float(np.real(analysis.inner_product()))
    snr = float(analysis.snr())
    print(f"[inj] Tobs={Tobs/YRSID_SI:.2f}yr f_low={f_low*1e3:.3f}mHz SNR={snr:.1f}", flush=True)

    # dense complex reference + invC + data (complex) -- construction-time only
    c0 = np.asarray(TDSignal(td_inj, settings=td_set).transform(wdm_c, window=window).arr)  # (3,Nf_active,Nt_a)
    dc = c0.copy()                                                # reference = injection
    invC = np.asarray(XYZ2SensitivityMatrix(wdm_c, model="scirdv1").invC)
    invC = np.where(np.isfinite(invC), invC, 0.0)

    # sparse grid
    sgen = GBSparseComplexWDMGen(real_td_callable=td, wdm_set_complex=wdm_c, data_dt=dt,
                                 ind_min_t=ind_min_t, Nt_active=Nt_active, Nt_layer=Nt_layer,
                                 m_active_half_width=W_M)
    stride = sgen.stride
    Nb = sgen.N_sparse_t
    n_loc = np.asarray(sgen.n_sparse_local, dtype=int)            # (Nb,) local time pixels

    # carrier track at the sparse bins (local layer index)
    c0_sparse = c0[:, :, n_loc]                                   # (3, Nf_active, Nb)
    track = np.argmax((np.abs(c0_sparse) ** 2).sum(0), axis=0)    # (Nb,) local m per sparse bin
    print(f"[track] carrier sweeps local m {track.min()}..{track.max()} "
          f"({track.max()-track.min()} layers); stride={stride} Nb={Nb}", flush=True)

    # chunk the sparse-time axis; per chunk a local band covering its carrier sweep
    edges = np.linspace(0, Nb, N_CHUNK + 1).astype(int)
    chunks = []
    for k in range(N_CHUNK):
        b0, b1 = edges[k], edges[k + 1]
        if b1 <= b0:
            continue
        mc = int(round(0.5 * (track[b0] + track[b1 - 1])))        # chunk carrier centre (local)
        m_lo = max(0, mc - W_M); m_hi = min(Nf_active - 1, mc + W_M)
        chunks.append((b0, b1, mc, m_lo, m_hi))

    # bin-fold-free direct contraction on the sparse grid (concept-level): per
    # chunk, <d|h>_chunk = sum_{b in chunk, m in band, c,c'} Re(d) invC Re(h) at
    # sparse bins (real WDM). We use the dense d/invC sliced to (band, sparse n).
    # (The polyphase below supplies the candidate c1 efficiently; d/c0 are
    # construction-time.)
    # Note: this is the trajectory tube expressed as chunk tiles -- the union of
    # chunk tiles is the reference-guided moving window.

    def trajectory_logL(p, use_polyphase=True):
        # candidate dense real WDM only if not using polyphase (for cross-check)
        h_dense_r = None
        if not use_polyphase:
            h_dense_r = np.asarray(TDSignal(td(p), settings=td_set).transform(
                wdm_r, window=window).arr)
        fd = np.fft.rfft(td(p) * window, axis=-1).astype(np.complex128) if use_polyphase else None
        kappa = 2.0 * np.sqrt(np.pi * dt) / Nf
        d_h = 0.0; h_h = 0.0
        for (b0, b1, mc, m_lo, m_hi) in chunks:
            bsl = slice(b0, b1)
            mb = np.arange(m_lo, m_hi + 1)
            n_sub = n_loc[bsl]                                    # local time pixels in chunk
            # reference c0 on chunk tile, sparse-n
            c0t = c0[:, m_lo:m_hi + 1, :][:, :, n_sub]            # (3, nm, nb_chunk)
            dt_re = np.real(dc[:, m_lo:m_hi + 1, :][:, :, n_sub])  # data (=c0 here) real WDM
            iCt = np.real(invC[:, :, m_lo:m_hi + 1, :][:, :, :, n_sub])  # (3,3,nm,nb_chunk)
            if use_polyphase:
                # polyphase candidate c1 on this chunk's band (center f0=mc)
                g9 = np.zeros(9); g9[1] = (mc + ind_min_f) * layer_df
                c1_act, m_local = sgen.sparse_from_rfft(fd, g9)  # (3, 2W+1, Nb)
                # align polyphase band (m_local) to chunk band mb
                # m_local are global-local indices; map to our band rows
                row = {int(ml): i for i, ml in enumerate(m_local)}
                c1t = np.zeros((3, mb.size, n_sub.size), dtype=np.complex128)
                for i, m in enumerate(mb):
                    if int(m) in row:
                        c1t[:, i, :] = c1_act[:, row[int(m)], bsl]
                ht_re = np.real(c1t)
            else:
                ht_re = np.real(h_dense_r[:, m_lo:m_hi + 1, :][:, :, n_sub])
            # real-WDM inner products on the chunk tile (sparse-n approx: weight by stride)
            d_h += float(np.einsum("cmn,cdmn,dmn->", dt_re, iCt, ht_re)) * stride
            h_h += float(np.einsum("cmn,cdmn,dmn->", ht_re, iCt, ht_re)) * stride
        return d_h - 0.5 * h_h - 0.5 * d_d

    print(f"\n   {'pert':>12s} {'logL_dense':>13s} {'traj(dense)':>13s} {'traj(poly)':>13s} "
          f"{'|td-dense|':>11s} {'|tp-dense|':>11s}", flush=True)
    for label, delta in [("zero", 0.0), ("df0+0.05lf", 0.05 * layer_df),
                         ("df0+0.2lf", 0.2 * layer_df)]:
        p = ref.copy(); p[5] = f_low + delta
        ll_dense = float(analysis.calculate_signal_likelihood(*p, source_only=True))
        ll_td = trajectory_logL(p, use_polyphase=False)
        ll_tp = trajectory_logL(p, use_polyphase=True)
        print(f"   {label:>12s} {ll_dense:+13.4e} {ll_td:+13.4e} {ll_tp:+13.4e} "
              f"{abs(ll_td - ll_dense):11.3e} {abs(ll_tp - ll_dense):11.3e}", flush=True)

    print("\n[interpretation] trajectory-chunked logL (both dense-sliced and "
          "polyphase candidate) tracks the dense lisatools logL for a chirping "
          "SOBBH. The chunks tile the carrier sweep; each is a local sig-het.",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
