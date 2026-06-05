"""Compare per-source wall time of chunked-het ``get_ll`` against
lisatools' ``AnalysisContainer.template_likelihood`` on the mm5
narrow band. Uses available pipeline code -- no kernel rewrites.

Same injection geometry as ``gb_chunked_test_script.py``: one GB at
f0 ~ 3 mHz, 1-year Tobs at dt=10. The mm5 narrow band is the
canonical ``[f0 - 3*layer_df, f0 + 2*layer_df]`` slice (see
``gb_chunked_prior_draws.py:283-340`` and the sprint root CLAUDE.md).

Reports:
    chunked.get_ll (default layer-groups ON)  -- per-source ms
    chunked.get_ll (layer-groups OFF, full-Nf)
    AnalysisContainer.template_likelihood     -- narrow-band, mm5 region
    AnalysisContainer.template_likelihood     -- full-band reference
    mm5 of the chunked template vs injection  -- accuracy

Env knobs:
    N        : batch size for chunked.get_ll  (default 1)
    REPEATS  : timed repetitions              (default 3)
"""
from __future__ import annotations

import os
import time

import numpy as np
from scipy import signal as sig

from lisatools.detector import ESAOrbits
from lisatools.utils.constants import YRSID_SI
from lisatools.response.tdiconfig import TDIConfig
from lisatools.response.tdionfly import GBTDIonTheFly
from lisatools.datacontainer import DataResidualArray
from lisatools.analysiscontainer import AnalysisContainer
from lisatools.sensitivity import XYZ2SensitivityMatrix
from lisatools.domains import TDSettings, TDSignal, WDMSettings, WDMSignal

from gb_wdm_het import GBWDMHeterodyne


N        = int(os.environ.get("N", "1"))
REPEATS  = int(os.environ.get("REPEATS", "3"))
NF       = 1460
NT       = 256 * 10
DT       = 10.0
NCH      = 3
NT_SUB   = 256
N_SPARSE = 256
N_PAD    = 32


def main() -> int:
    print(f"[setup] N={N} REPEATS={REPEATS} NF={NF} NT={NT} dt={DT}", flush=True)
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

    # Injection params -- matches gb_chunked_test_script.py defaults
    # adjusted for in-band f0 (3 mHz).
    layer_df = 1.0 / (2.0 * NF * DT)
    m_ref = int(3e-3 / layer_df)
    f_frac = 0.5
    f0_inj = (m_ref + f_frac) * layer_df
    amp_inj    = np.array([8.0e-22])
    f0         = np.array([f0_inj])
    fdot       = np.array([1.0e-17])
    fddot      = np.array([0.0])
    phi0       = np.array([2.09802430298])
    inc        = np.array([0.23984234])
    psi        = np.array([1.234019814])
    lam        = np.array([4.09808143])
    beta       = np.array([0.04])
    params = np.array([amp_inj, f0, fdot, fddot, phi0, inc, psi, lam, beta]).T

    _data_inj = np.zeros((NCH, Nobs))
    inj_spline = gb_gen_inj(
        amp_inj, f0, fdot, fddot, phi0, inc, psi, lam, beta,
        convert_to_ra_dec=False, return_spline=True,
    )
    _data_inj[:] = inj_spline.eval_tdi(t_arr)

    # Build the parent WDM grid and full-band AC ----------------------
    td_set = TDSettings(Nobs, DT, force_backend="cpu")
    window = np.ones(Nobs)
    min_freq = 0.0001
    max_freq = 35.0e-3
    edge_cut = 20
    wavelet_duration = NF * DT
    min_time = edge_cut * wavelet_duration
    max_time = (NT - edge_cut) * wavelet_duration
    wdm_set = WDMSettings(
        NF, NT, DT,
        min_freq=min_freq, max_freq=max_freq,
        min_time=min_time, max_time=max_time,
    )
    data_inj_all = TDSignal(_data_inj, settings=td_set).transform(wdm_set, window=window)
    injection_full = DataResidualArray(data_inj_all)
    sens_mat = XYZ2SensitivityMatrix(injection_full.data_res_arr.settings, model="scirdv1")
    analysis = AnalysisContainer(injection_full, sens_mat)

    # Chunked-het generator ------------------------------------------
    chunked = GBWDMHeterodyne(
        Nf=NF, Nt=NT, dt=DT, T_full=Tobs, t_ref_full=t_start,
        Nt_sub=NT_SUB, n_pad=N_PAD, N_sparse=N_SPARSE,
        backend="cpu", tdi_gen="2nd generation",
        orbits=orbits, t_obs_start=float(t_start),
        use_cpp=True,
        N_cp_sig=int(os.environ.get("N_CP_SIG", "48")),
        N_cp_orbit=int(os.environ.get("N_CP_ORBIT", "32")),
    )
    chunked._ensure_cpp_setup()

    # Build the chunked template once -- both timing paths need it.
    template_full = np.zeros((NCH, NF, NT), dtype=float)
    chunked.fill_global(template_full, [params[0]])

    # Active-band view for the lisatools-side WDMSignal ---------------
    ilo = wdm_set.ind_min_f
    ihi = wdm_set.ind_max_f + 1
    tslice = wdm_set.active_slice_t if wdm_set.Nt_active != wdm_set.Nt else slice(None)
    template_active = template_full[:, ilo:ihi, tslice]
    template_full_wdm = WDMSignal(template_active, wdm_set)
    overlap_full = float(analysis.template_inner_product(template_full_wdm, normalize=True))
    mm_full = 1.0 - overlap_full
    print(f"[acc]  full-band mismatch (chunked tpl vs inj)         = {mm_full:.3e}", flush=True)

    # mm5 narrow band -- canonical mm5 setup (gb_chunked_prior_draws.py:283-340)
    new_wdm_set = WDMSettings(
        wdm_set.Nf, wdm_set.Nt, wdm_set.data_dt,
        min_time=wdm_set.min_time, max_time=wdm_set.max_time,
        min_freq=float(f0[0] - 3 * wdm_set.layer_df),
        max_freq=float(f0[0] + 2 * wdm_set.layer_df),
    )
    m_lo_band = new_wdm_set.ind_min_f - wdm_set.ind_min_f
    m_hi_band = new_wdm_set.ind_max_f - wdm_set.ind_min_f + 1
    inj_arr = np.asarray(injection_full.data_res_arr.arr)
    inj_narrow = DataResidualArray(WDMSignal(inj_arr[:, m_lo_band:m_hi_band], new_wdm_set))
    sens_narrow = XYZ2SensitivityMatrix(new_wdm_set, model="scirdv1")
    analysis_narrow = AnalysisContainer(inj_narrow, sens_narrow)
    tpl_narrow = DataResidualArray(WDMSignal(template_active[:, m_lo_band:m_hi_band], new_wdm_set))
    mm5 = 1.0 - float(analysis_narrow.template_inner_product(tpl_narrow, normalize=True))
    print(f"[acc]  mm5  (mm5 narrow band)                          = {mm5:.3e}", flush=True)

    # ----------------------------------------------------------------
    # Build the data_d_full / invC_full once -- match the MCMC's recipe
    # (gb_chunked_test_script.py line ~389-406).
    inj_active = np.asarray(injection_full.data_res_arr.arr)
    psd_active = np.asarray(sens_mat.sens_mat)
    if psd_active.ndim == 4:
        psd_diag = np.stack([psd_active[c, c] for c in range(NCH)], axis=0)
    else:
        psd_diag = psd_active
    with np.errstate(divide="ignore", invalid="ignore"):
        invC_active = 1.0 / np.where(
            np.isfinite(psd_diag) & (psd_diag > 0),
            psd_diag, np.inf,
        )
    invC_active = np.where(np.isfinite(invC_active), invC_active, 0.0)
    data_d_full = np.zeros((NCH, NF, NT), dtype=float)
    invC_full   = np.zeros_like(data_d_full)
    if wdm_set.Nt_active == wdm_set.Nt:
        data_d_full[:, ilo:ihi, :] = inj_active
        invC_full  [:, ilo:ihi, :] = invC_active
    else:
        data_d_full[:, ilo:ihi, tslice] = inj_active
        invC_full  [:, ilo:ihi, tslice] = invC_active

    # ----------------------------------------------------------------
    # Build a batch of N source params clustered around the injection.
    rng = np.random.default_rng(0)
    spread = np.array([1e-3, 1e-8, 1e-3, 0.0, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3])
    params_batch = np.tile(params[0], (N, 1)) * (
        1.0 + rng.standard_normal((N, 9)) * spread
    )
    params_batch[:, 3] = 0.0
    params_list_batch = [params_batch[i] for i in range(N)]

    # ----------------------------------------------------------------
    # Timing  (everything is per-call wall time; per-source = / N)
    # ----------------------------------------------------------------
    def _bench(label, fn, repeats=REPEATS):
        # Warmup
        _ = fn()
        t0 = time.perf_counter()
        for _ in range(repeats):
            _ = fn()
        per_call = (time.perf_counter() - t0) / repeats
        print(f"[time] {label:60s} : {per_call*1e3:10.3f} ms / call",
              flush=True)
        return per_call

    print()
    print("=== Lisatools (template-likelihood already-built-template) ===", flush=True)
    t_lt_full = _bench(
        f"AnalysisContainer.template_likelihood  (full band, single source)",
        lambda: analysis.template_likelihood(template_full_wdm),
    )
    t_lt_mm5 = _bench(
        f"AnalysisContainer.template_likelihood  (mm5 narrow band, single)",
        lambda: analysis_narrow.template_likelihood(tpl_narrow),
    )

    print()
    print("=== Chunked-het waveform-only (``fill_global``) ===", flush=True)
    # fill_global builds the template into a global Nf x Nt buffer.
    # The new narrow-band per-source restriction (matching get_ll's
    # layer-groups) makes this strictly the "waveform build" cost --
    # subtract from get_ll to isolate the inner-product / likelihood
    # contribution.
    #
    # NOTE: a NumPy ``template_buf.fill(0.0)`` over (3, 1460, 2560) is
    # ~88 MB of memory bandwidth and dominates any per-call C++ work.
    # We zero it ONCE here, then time only ``chunked.fill_global`` calls
    # below -- the buffer just accumulates; the timing measures C++
    # kernel time, not Python-side bookkeeping.
    template_buf = np.zeros((NCH, NF, NT), dtype=float)
    def _fill_fn():
        chunked.fill_global(template_buf, params_list_batch)
        return None
    t_fg = _bench(
        f"chunked.fill_global  N={N}  (narrow-band per source, no zero-fill)",
        _fill_fn,
    )
    print(f"   -> per source: {t_fg/N*1e3:10.3f} ms", flush=True)

    print()
    print("=== Chunked-het ``get_ll`` (template + inner product fused) ===", flush=True)
    t_grp = _bench(
        f"chunked.get_ll  N={N}  (use_layer_groups=True, default)",
        lambda: chunked.get_ll(
            data_d_full, invC_full, params_list_batch, use_layer_groups=True,
        ),
    )
    print(f"   -> per source: {t_grp/N*1e3:10.3f} ms", flush=True)

    t_full = _bench(
        f"chunked.get_ll  N={N}  (use_layer_groups=False)",
        lambda: chunked.get_ll(
            data_d_full, invC_full, params_list_batch, use_layer_groups=False,
        ),
    )
    print(f"   -> per source: {t_full/N*1e3:10.3f} ms", flush=True)

    # Waveform vs likelihood breakdown.
    t_like_only = t_grp - t_fg
    print()
    print(f"[breakdown] waveform (fill_global)  : {t_fg/N*1e3:9.2f} ms / src "
          f"({t_fg/t_grp*100:5.1f}% of get_ll)", flush=True)
    print(f"[breakdown] likelihood (get_ll - fg) : {t_like_only/N*1e3:9.2f} ms / src "
          f"({t_like_only/t_grp*100:5.1f}% of get_ll)", flush=True)

    print()
    print("=== Summary (per single GB likelihood evaluation) ===")
    print(f"  lisatools full-band template_likelihood : {t_lt_full*1e3:9.2f} ms")
    print(f"  lisatools mm5     template_likelihood   : {t_lt_mm5 *1e3:9.2f} ms")
    print(f"  chunked-het get_ll (grouped, default)    : {t_grp/N*1e3:9.2f} ms / src "
          f"(N={N})")
    print(f"  chunked-het get_ll (ungrouped)           : {t_full/N*1e3:9.2f} ms / src "
          f"(N={N})")
    print(f"  mm5 accuracy of chunked template         : {mm5:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
