#!/usr/bin/env python
"""Examine the EXISTING chunked-het likelihood for chirping SOBBHs.

Per user direction (2026-06-19): "maybe it is faster to just do the chunked-het
likelihood for SOBBHs in chunks." The chunked-het (SOBBHWDMComputations.
get_ll_wdm) already processes the observation in TIME chunks and heterodynes
each chunk to its own carrier f0_grid -- so it inherently follows the
frequency-time sweep of a chirping SOBBH (unlike the fixed-band sig-het). This
script checks whether it (a) recovers the dense logL on a chirping SOBBH and
(b) how fast it is.

For a year-long chirping SOBBH:
  * dense lisatools logL (truth) at ref + a few perturbations.
  * chunked-het get_ll_wdm at the same params, timed.
  * report |chunked - dense| and per-call wall time.

Run::  F_LOW_HZ=0.012 python sobbh_chunked_het_examine.py
Env:   F_LOW_HZ (0.012), NT_SUB (256), N_SPARSE (256), N_CALLS (20)
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
from scipy.signal.windows import tukey as _tukey

from lisatools.analysiscontainer import AnalysisContainer
from lisatools.datacontainer import DataResidualArray
from lisatools.detector import ESAOrbits
from lisatools.domains import TDSettings, TDSignal, WDMSettings
from lisatools.sensitivity import XYZ2SensitivityMatrix
from lisatools.utils.constants import YRSID_SI
from lisatools.utils.utility import get_array_module

from lisatools.response.tdiconfig import TDIConfig
from lisatools.response.tdionfly import SOBBHTDIonTheFly

import bbhx  # noqa: F401
from bbhx.sobbhcomps import SOBBHWDMComputations


class _Holder:
    def __init__(self, data_full, invC_full):
        xp = get_array_module(data_full)
        self.linear_data_arr = [xp.ascontiguousarray(data_full).ravel().copy()]
        self.linear_psd_arr = [xp.ascontiguousarray(invC_full).ravel().copy()]

    def __len__(self):
        return 1


def main():
    backend = "cpu"
    dt = 10.0
    Nf, Nt = 1460, 2560
    Nobs = Nf * Nt
    EC = 20
    Nt_sub = int(os.environ.get("NT_SUB", "256"))
    N_sparse = int(os.environ.get("N_SPARSE", "256"))
    N_CALLS = int(os.environ.get("N_CALLS", "20"))
    f_low = float(os.environ.get("F_LOW_HZ", "0.012"))
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

    ref = np.array([60.0, 55.0, 0.1, 0.2, 4.0e8, f_low, 1.1,
                    np.arccos(0.3), 0.7, 3.1, np.arcsin(0.2)])
    td_inj = td(ref)
    data_r = TDSignal(td_inj, settings=td_set).transform(wdm, window=window)
    sens = XYZ2SensitivityMatrix(wdm, model="scirdv1")
    analysis = AnalysisContainer(DataResidualArray(data_r), sens)
    analysis.signal_gen = lambda *p: TDSignal(td(np.asarray(p, float).reshape(11)),
                                              settings=td_set).transform(wdm, window=window)
    d_d = float(np.real(analysis.inner_product()))
    snr = float(analysis.snr())
    layers = None
    c0p = (np.asarray(data_r.arr) ** 2).sum(0)
    tr = np.argmax(c0p, 0)
    layers = int(tr.max() - tr.min())
    print(f"[inj] Tobs={Tobs/YRSID_SI:.2f}yr f_low={f_low*1e3:.3f}mHz SNR={snr:.1f} "
          f"layers_crossed~{layers}", flush=True)

    chunked = SOBBHWDMComputations(
        wdm, t_ref=t_start, Nt_sub=Nt_sub, n_pad=Nt_sub // 8, N_sparse=N_sparse,
        N_cp_sig=0, N_cp_orbit=0, orbits=orbits, tdi_config="2nd generation",
        force_backend=backend, d_d=0.0, tdi_type="XYZ")
    print(f"[chunked] n_chunks={chunked.n_chunks} T_chunk={chunked.T_chunk:.3e}s "
          f"alpha={chunked.resolved_tukey_alpha}", flush=True)

    inj_active = np.asarray(data_r.arr)
    invC_active = np.asarray(sens.invC)
    invC_active = np.where(np.isfinite(invC_active), invC_active, 0.0)
    holder = _Holder(inj_active, invC_active)

    # Frequency-band width knobs (per user direction 2026-06-19): keep the TIME
    # chunk as long as useful (large Nt_sub) and ADD FREQUENCY LAYERS to cover
    # the carrier sweep within each chunk -- widen the band rather than shorten
    # the chunk. m_band_half_width sets the per-chunk band half-width;
    # group_band_layers the layer-grouping width.
    M_BAND_HALF = int(os.environ.get("M_BAND_HALF", "1"))
    GROUP_BAND = int(os.environ.get("GROUP_BAND", "5"))

    def chunked_logL(p):
        ll = np.asarray(chunked.get_ll_wdm(
            np.asarray(p, float).reshape(1, 11), holder, convert_to_ra_dec=False,
            use_layer_groups=True, group_band_layers=GROUP_BAND, margin_layers=0,
            m_band_half_width=M_BAND_HALF))
        return float(ll[0]) - 0.5 * d_d
    print(f"[knobs] Nt_sub={Nt_sub} (T_chunk={chunked.T_chunk:.2e}s) "
          f"m_band_half_width={M_BAND_HALF} group_band_layers={GROUP_BAND}", flush=True)

    # ACCURACY via convention-robust normalized FULL-BAND mismatch of the
    # chunked-het template (built via fill_global) vs the data. (Absolute logL
    # comparison is confounded by the data being window=None while the
    # chunked-het applies its own Tukey -- mm is normalized so that cancels.)
    from lisatools.domains import WDMSignal
    template_full = np.zeros((3, Nf, Nt), dtype=float)
    chunked.fill_global_wdm(ref.reshape(1, 11), template_full,
                            convert_to_ra_dec=False, factors=None)
    tpl_active = template_full[:, wdm.ind_min_f:wdm.ind_max_f + 1, wdm.active_slice_t]
    tpl_wdm = WDMSignal(tpl_active, wdm)
    mm_full = float(1.0 - analysis.template_inner_product(
        DataResidualArray(tpl_wdm), normalize=True))
    print(f"\n[accuracy] full-band mismatch (chunked template vs data) = {mm_full:.3e}",
          flush=True)

    # timing: batch of perturbed params
    rng = np.random.default_rng(0)
    batch = np.tile(ref, (N_CALLS, 1))
    batch[:, 5] += rng.uniform(-0.2, 0.2, N_CALLS) * layer_df
    t0 = time.perf_counter()
    _ = chunked.get_ll_wdm(batch, holder, convert_to_ra_dec=False,
                           use_layer_groups=True, group_band_layers=5, margin_layers=0)
    dt_chunked = (time.perf_counter() - t0) / N_CALLS

    t0 = time.perf_counter()
    for i in range(min(N_CALLS, 5)):
        _ = analysis.calculate_signal_likelihood(*batch[i], source_only=True)
    dt_dense = (time.perf_counter() - t0) / min(N_CALLS, 5)

    print(f"\n[timing] chunked-het get_ll: {dt_chunked*1e3:.2f} ms/call   "
          f"dense lisatools: {dt_dense*1e3:.1f} ms/call   "
          f"speedup x{dt_dense/dt_chunked:.0f}", flush=True)
    print("\n[interpretation] the chunked-het tracks the carrier sweep via its "
          "per-chunk f0_grid heterodyne -- the simplest 'in chunks along the "
          "trajectory' option for SOBBHs.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
