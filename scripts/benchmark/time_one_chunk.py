"""Sweep chunk geometry from many small chunks down to 1 chunk.

For each (Nt_sub, N_sparse, n_pad) config, builds the chunked-het
generator, computes mm5 vs a high-fidelity reference (the existing
13-chunk default), and times ``chunked.get_ll`` and
``chunked.fill_global``.

The CPU build now uses ``FAST_WDM_N_SPARSE_MAX = 4096`` (gated by
``#ifdef __CUDACC__``); GPU is unaffected.

Env knobs:
    NF, NT, DT   : WDM grid (default 1460, 2048, 10.0)
    REPEATS       : timed reps per config (default 3)
    CONFIGS       : comma-sep ``Nt_sub:N_sparse:n_pad`` triples.
                    Default: standard 13-chunk + a few 1-chunk variants.
"""
from __future__ import annotations

import os
import time

import numpy as np

from lisatools.detector import ESAOrbits
from lisatools.utils.constants import YRSID_SI
from lisatools.response.tdiconfig import TDIConfig
from lisatools.response.tdionfly import GBTDIonTheFly
from lisatools.datacontainer import DataResidualArray
from lisatools.analysiscontainer import AnalysisContainer
from lisatools.sensitivity import XYZ2SensitivityMatrix
from lisatools.domains import TDSettings, TDSignal, WDMSettings, WDMSignal

from gb_wdm_het import GBWDMHeterodyne


NF       = int(os.environ.get("NF", "1460"))
NT       = int(os.environ.get("NT", "2048"))   # power of 2 -> permits Nt_sub == Nt
DT       = float(os.environ.get("DT", "10.0"))
REPEATS  = int(os.environ.get("REPEATS", "3"))
NCH      = 3

# Default sweep. Config format: ``Nt_sub:N_sparse:n_pad[:N_cp_sig]``.
# N_cp_sig defaults to 48 (the half-day-wavelet-validated value); the
# 1-chunk-per-obs configs need it bumped to keep the slow-signal
# control-point spacing at ~6 hours across the now-much-longer
# T_chunk. Cap is ``FAST_WDM_N_CP_SIG_MAX`` (2048 on CPU, 48 on GPU).
_DEFAULT_CONFIGS = ",".join([
    "256:256:32:48",        # 11 chunks at Nt=2048   -- baseline
    "512:512:32:48",        # 5 chunks
    "1024:1024:32:48",      # 3 chunks
    "2048:1024:0:48",       # 1 chunk, low N_sparse, low N_cp_sig  (sanity)
    "2048:4096:0:48",       # 1 chunk, dense N_sparse, low N_cp_sig
    "2048:4096:0:512",      # 1 chunk, dense N_sparse, N_cp_sig=512  -- ~14h spacing
    "2048:4096:0:1024",     # 1 chunk, dense N_sparse, N_cp_sig=1024 -- ~7h spacing
    "2048:4096:0:2048",     # 1 chunk, dense N_sparse, N_cp_sig=2048 -- ~3.5h spacing
])
CONFIGS = os.environ.get("CONFIGS", _DEFAULT_CONFIGS)


def main() -> int:
    print(f"[setup] NF={NF} NT={NT} dt={DT} REPEATS={REPEATS}", flush=True)
    Tobs = NF * NT * DT
    t_start = int(0.5 * YRSID_SI / DT) * DT
    Nobs = NF * NT
    t_arr = np.arange(Nobs) * DT + t_start

    orbits = ESAOrbits()
    tdi_config = TDIConfig("2nd generation")
    gb_gen_inj = GBTDIonTheFly(
        np.linspace(t_arr[0], t_arr[-1], 16384), Tobs, t_start, 1.0 / DT, 1,
        tdi_config=tdi_config, orbits=orbits, tdi_chan="XYZ",
        force_backend="cpu",
    )
    layer_df = 1.0 / (2.0 * NF * DT)
    m_ref = int(3e-3 / layer_df)
    f0_inj = (m_ref + 0.5) * layer_df
    params = np.array([[8.0e-22, f0_inj, 1.0e-17, 0.0,
                         2.1, 0.24, 1.23, 4.1, 0.04]])

    _data_inj = np.zeros((NCH, Nobs))
    inj = gb_gen_inj(
        params[0, 0:1], params[0, 1:2], params[0, 2:3], params[0, 3:4],
        params[0, 4:5], params[0, 5:6], params[0, 6:7], params[0, 7:8],
        params[0, 8:9],
        convert_to_ra_dec=False, return_spline=True,
    )
    _data_inj[:] = inj.eval_tdi(t_arr)

    td_set = TDSettings(Nobs, DT, force_backend="cpu")
    window = np.ones(Nobs)
    wavelet_duration = NF * DT
    edge_cut = 20
    wdm_set = WDMSettings(
        NF, NT, DT,
        min_freq=1e-4, max_freq=35e-3,
        min_time=edge_cut * wavelet_duration,
        max_time=(NT - edge_cut) * wavelet_duration,
    )
    inj_all = TDSignal(_data_inj, settings=td_set).transform(wdm_set, window=window)
    injection = DataResidualArray(inj_all)
    sens_mat = XYZ2SensitivityMatrix(injection.data_res_arr.settings, model="scirdv1")

    # invC zeroed outside source's narrow band -- per the realistic MCMC recipe.
    inj_active = np.asarray(injection.data_res_arr.arr)
    psd_active = np.asarray(sens_mat.sens_mat)
    psd_diag = (np.stack([psd_active[c, c] for c in range(NCH)], axis=0)
                if psd_active.ndim == 4 else psd_active)
    with np.errstate(divide="ignore", invalid="ignore"):
        invC_active = 1.0 / np.where(
            np.isfinite(psd_diag) & (psd_diag > 0), psd_diag, np.inf,
        )
    invC_active = np.where(np.isfinite(invC_active), invC_active, 0.0)
    data_d_full = np.zeros((NCH, NF, NT), dtype=float)
    invC_full   = np.zeros_like(data_d_full)
    ilo = wdm_set.ind_min_f
    ihi = wdm_set.ind_max_f + 1
    tslice = wdm_set.active_slice_t if wdm_set.Nt_active != wdm_set.Nt else slice(None)
    if wdm_set.Nt_active == wdm_set.Nt:
        data_d_full[:, ilo:ihi, :] = inj_active
        invC_full  [:, ilo:ihi, :] = invC_active
    else:
        data_d_full[:, ilo:ihi, tslice] = inj_active
        invC_full  [:, ilo:ihi, tslice] = invC_active

    # mm5 reference template (chunked-het standard config) for accuracy
    chunked_ref = GBWDMHeterodyne(
        Nf=NF, Nt=NT, dt=DT, T_full=Tobs, t_ref_full=t_start,
        Nt_sub=256, n_pad=32, N_sparse=256,
        backend="cpu", tdi_gen="2nd generation",
        orbits=orbits, t_obs_start=float(t_start),
        use_cpp=True, N_cp_sig=48, N_cp_orbit=32,
    )
    chunked_ref._ensure_cpp_setup()
    tpl_ref_full = np.zeros((NCH, NF, NT), dtype=float)
    chunked_ref.fill_global(tpl_ref_full, [params[0]])
    tpl_ref_active = tpl_ref_full[:, ilo:ihi, tslice]
    new_wdm_set = WDMSettings(
        wdm_set.Nf, wdm_set.Nt, wdm_set.data_dt,
        min_time=wdm_set.min_time, max_time=wdm_set.max_time,
        min_freq=float(params[0, 1] - 3 * wdm_set.layer_df),
        max_freq=float(params[0, 1] + 2 * wdm_set.layer_df),
    )
    m_lo_band = new_wdm_set.ind_min_f - wdm_set.ind_min_f
    m_hi_band = new_wdm_set.ind_max_f - wdm_set.ind_min_f + 1
    inj_arr = np.asarray(injection.data_res_arr.arr)
    inj_narrow = DataResidualArray(WDMSignal(inj_arr[:, m_lo_band:m_hi_band], new_wdm_set))
    sens_narrow = XYZ2SensitivityMatrix(new_wdm_set, model="scirdv1")
    analysis_narrow = AnalysisContainer(inj_narrow, sens_narrow)
    tpl_ref_narrow = DataResidualArray(WDMSignal(tpl_ref_active[:, m_lo_band:m_hi_band], new_wdm_set))
    mm5_ref = 1.0 - float(analysis_narrow.template_inner_product(tpl_ref_narrow, normalize=True))
    print(f"[ref] standard 13-chunk mm5 = {mm5_ref:.3e}  (reference for accuracy compare)",
          flush=True)
    print()

    def _bench(fn, repeats=REPEATS):
        _ = fn()  # warmup
        t0 = time.perf_counter()
        for _ in range(repeats):
            _ = fn()
        return (time.perf_counter() - t0) / repeats

    print(f"{'cfg':18s}  {'n_chunks':>9s}  {'fill_global (ms)':>17s}  "
          f"{'get_ll (ms)':>13s}  {'mm5':>10s}", flush=True)
    print("-" * 80, flush=True)

    for cfg_str in CONFIGS.split(","):
        cfg_str = cfg_str.strip()
        if not cfg_str:
            continue
        parts = [int(x) for x in cfg_str.split(":")]
        if len(parts) == 3:
            Nt_sub_v, N_sparse_v, n_pad_v = parts
            N_cp_sig_v = 48
        else:
            Nt_sub_v, N_sparse_v, n_pad_v, N_cp_sig_v = parts
        try:
            chunked = GBWDMHeterodyne(
                Nf=NF, Nt=NT, dt=DT, T_full=Tobs, t_ref_full=t_start,
                Nt_sub=Nt_sub_v, n_pad=n_pad_v, N_sparse=N_sparse_v,
                backend="cpu", tdi_gen="2nd generation",
                orbits=orbits, t_obs_start=float(t_start),
                use_cpp=True, N_cp_sig=N_cp_sig_v, N_cp_orbit=32,
            )
            chunked._ensure_cpp_setup()
        except Exception as e:
            print(f"{cfg_str:18s}  CONFIG FAILED: {type(e).__name__}: {e}", flush=True)
            continue
        n_chunks = chunked.n_chunks

        # mm5 check
        tpl_full = np.zeros((NCH, NF, NT), dtype=float)
        chunked.fill_global(tpl_full, [params[0]])
        tpl_active = tpl_full[:, ilo:ihi, tslice]
        tpl_narrow = DataResidualArray(WDMSignal(tpl_active[:, m_lo_band:m_hi_band], new_wdm_set))
        mm5 = 1.0 - float(analysis_narrow.template_inner_product(tpl_narrow, normalize=True))

        # timing -- preallocated template buf (no NumPy zero-fill in timed fn)
        template_buf = np.zeros((NCH, NF, NT), dtype=float)
        t_fg = _bench(lambda: chunked.fill_global(template_buf, [params[0]]))
        t_gl = _bench(lambda: chunked.get_ll(
            data_d_full, invC_full, [params[0]], use_layer_groups=True,
        ))
        print(f"{cfg_str:18s}  {n_chunks:9d}  {t_fg*1e3:17.2f}  {t_gl*1e3:13.2f}  "
              f"{mm5:10.3e}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
