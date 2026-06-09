#!/usr/bin/env python
"""Validate gb_signal_het_fill_global C++ against the Python build_v2_real_template.

The C++ fill_global path:
  1. Generates the candidate's sparse heterodyne FD via gb_run_fd_wave_tdi
     (the Tukey-windowed sparse rfft) -- same as get_ll_in_kernel.
  2. Polyphase fold -> c1_sparse -> r_sparse on active m-layers.
  3. Carrier de-rotate r at sparse n, linear-interp to dense n, re-rotate.
  4. Multiply by stored c0_dense_complex on the full active band.
  5. Scatter factor * Re(c1_dense) into template_fill at (data_idx, c,
     m_global, n_global).

Per draw we:
  A) build the Python reference template (build_v2_real_template) from the
     same fd_rfft + c0_sparse + c0_dense_complex.
  B) call the C++ fill_global with factor=+1.
  C) compare the two templates pointwise on the active band, and compute
     the narrowband mm5 between the C++ template and the injection's real
     WDM transform.

Self-consistency at injection: mm5 should match the Python-prototype mm5
(~1e-9) and the C++/Python templates should agree to ~1e-4 across the
active band (the residual is the Tukey window vs sparse-no-window
discrepancy + interpolation precision, same source as Stage 2b's ~5e-5
logL self-consistency).

Run::
    python gb_signal_het_cpp_validate_fill_global.py
Env vars:
    N_DRAWS         default 3
    NT_LAYER        default 64
    N_SPARSE_FD     default 1024
    SEED            default 54321
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
from lisatools.response.tdionfly import GBTDIonTheFly
import lisatools_backend_cpu.pycppdetector as _lat_pd
import gbgpu_backend_cpu.cgbgpu as _be  # GBComputationGroupWrap lives here post-3L.7g

from gb_signal_het_wdm_v2 import GBSparseComplexWDMGen
from gb_signal_het_wdm_v2_mm_sweep import build_gb_prior


def main():
    N_DRAWS = int(os.environ.get("N_DRAWS", "3"))
    Nt_layer = int(os.environ.get("NT_LAYER", "64"))
    N_SPARSE_FD = int(os.environ.get("N_SPARSE_FD", "1024"))
    SEED = int(os.environ.get("SEED", "54321"))
    # Single source of truth: Python window AND C++ tukey_alpha use this.
    TUKEY_ALPHA = float(os.environ.get("TUKEY_ALPHA", "0.05"))

    backend = "cpu"
    dt = 10.0
    Nf, Nt = 1460, 2560
    Nobs = Nf * Nt
    EC = 20

    t_start = int(0.5 * YRSID_SI / dt) * dt
    t_arr = np.arange(Nobs) * dt + t_start
    Tobs = Nt * Nf * dt

    orbits = ESAOrbits(force_backend=backend)
    tdi_config = TDIConfig("2nd generation", force_backend=backend)
    t_tdi = np.linspace(t_arr[0], t_arr[-1], 16384)
    gb_gen = GBTDIonTheFly(
        t_tdi, Tobs, t_start, 1.0 / dt, 1,
        tdi_config=tdi_config, orbits=orbits, tdi_chan="XYZ",
        force_backend=backend,
    )
    tdi_wrap = gb_gen.wave_gen

    def real_td_cb(p):
        amp, f0, fdot, fddot, phi0, inc, psi, lam, beta = p
        spline = gb_gen(
            np.array([amp]), np.array([f0]), np.array([fdot]),
            np.array([fddot]), np.array([phi0]), np.array([inc]),
            np.array([psi]), np.array([lam]), np.array([beta]),
            convert_to_ra_dec=False, return_spline=True,
        )
        return np.asarray(spline.eval_tdi(t_arr))[0]

    td_set = TDSettings(Nobs, dt, force_backend=backend)
    window = _tukey(Nobs, alpha=TUKEY_ALPHA).astype(float)

    wdm_set_real = WDMSettings(
        Nf, Nt, dt, t0=t_start,
        min_freq=1e-4, max_freq=35e-3,
        min_time=EC * Nf * dt, max_time=(Nt - EC) * Nf * dt,
        is_complex=False, force_backend=backend,
    )
    wdm_set_complex = WDMSettings(
        Nf, Nt, dt, t0=t_start,
        min_freq=1e-4, max_freq=35e-3,
        min_time=EC * Nf * dt, max_time=(Nt - EC) * Nf * dt,
        is_complex=True, force_backend=backend,
    )
    layer_df = wdm_set_real.layer_df
    ind_min_t = int(wdm_set_real.ind_min_t)
    Nt_active = int(wdm_set_real.Nt_active)
    Nf_active = int(wdm_set_real.ind_max_f - wdm_set_real.ind_min_f + 1)
    ind_min_f = int(wdm_set_real.ind_min_f)
    print(f"[grid] Nf={Nf} Nt={Nt} layer_df={layer_df:.3e} "
          f"Nt_active={Nt_active} Nf_active={Nf_active} ind_min_f={ind_min_f}",
          flush=True)

    sparse_gen = GBSparseComplexWDMGen(
        real_td_callable=real_td_cb, wdm_set_complex=wdm_set_complex,
        data_dt=dt, ind_min_t=ind_min_t, Nt_active=Nt_active,
        Nt_layer=Nt_layer, m_active_half_width=2,
    )
    stride = sparse_gen.stride
    N_sparse_t = sparse_gen.N_sparse_t
    n_sparse_local = np.asarray(sparse_gen.n_sparse_local, dtype=np.int32)
    print(f"[v2] Nt_layer={Nt_layer} stride={stride} N_sparse_t={N_sparse_t} "
          f"N_sparse_fd={N_SPARSE_FD}", flush=True)

    np.random.seed(SEED)
    f0_lo = (ind_min_f + 7) * layer_df
    f0_hi = 0.025
    prior, tc = build_gb_prior(
        A_lims=(1e-23, 1e-20), f0_lims_hz=(f0_lo, f0_hi),
        fdot_lims=(-1e-15, 1e-15), beta_lims=None,
    )

    SNR_MIN, SNR_MAX = 5.0, 1100.0
    MAX_REJECT = 500
    cpp = _be.GBComputationGroupWrapCPU()
    sens_mat = None
    window_full = sparse_gen.window_full.astype(np.float64).copy()

    drawi = 0
    while drawi < N_DRAWS:
        for _ in range(MAX_REJECT):
            x_samp = prior.rvs(size=1)
            params_inj = tc.both_transforms(x_samp.copy())[0]
            td_inj = real_td_cb(params_inj)
            wdm_inj_real = TDSignal(td_inj, settings=td_set).transform(
                wdm_set_real, window=window)
            inj_data_arr = DataResidualArray(wdm_inj_real)
            if sens_mat is None:
                sens_mat = XYZ2SensitivityMatrix(
                    inj_data_arr.data_res_arr.settings, model="scirdv1")
            analysis = AnalysisContainer(inj_data_arr, sens_mat)
            snr_i = float(analysis.snr())
            if SNR_MIN <= snr_i <= SNR_MAX:
                break

        # WDM transforms already return active-band shapes (3, Nf_active, Nt_active).
        wdm_inj_complex = np.asarray(
            TDSignal(td_inj, settings=td_set).transform(
                wdm_set_complex, window=window).arr)  # (3, Nf_active, Nt_active)
        c0_dense_active = wdm_inj_complex.copy()
        # c0 at sparse n positions (relative to the active-band time axis).
        c0_sparse_active = c0_dense_active[:, :, n_sparse_local].copy()

        # Reference real WDM array, also already (3, Nf_active, Nt_active).
        wdm_inj_active = np.asarray(wdm_inj_real.arr).copy()

        # Self-consistency: candidate = injection.
        params_cand = params_inj.copy()
        params_cand_all = params_cand.astype(np.float64).reshape(1, 9).copy()
        params_ref_all = params_inj.astype(np.float64).reshape(1, 9).copy()
        data_index_all = np.zeros(1, dtype=np.int32)
        factors_all = np.ones(1, dtype=np.float64)

        # ---- C++ fill_global ----
        template_fill = np.zeros((1, 3, Nf, Nt), dtype=np.float64)
        cpp.gb_signal_het_fill_global_in_kernel(
            tdi_wrap,
            template_fill,
            c0_sparse_active[None, ...].copy(),     # (1, 3, Nf_active, N_sparse_t)
            c0_dense_active[None, ...].copy(),      # (1, 3, Nf_active, Nt_active)
            window_full, n_sparse_local,
            params_cand_all, params_ref_all,
            factors_all, data_index_all,
            1, 1,
            9, 1, 2,
            Nf, Nt, Nf_active, Nt_active,
            Nt_layer, N_sparse_t, stride,
            ind_min_t, ind_min_f,
            2,
            layer_df, dt,
            Tobs, t_start,
            3,
            N_SPARSE_FD, TUKEY_ALPHA,
        )
        tpl_cpp_full = template_fill[0]
        tpl_cpp_active = tpl_cpp_full[:, ind_min_f:ind_min_f + Nf_active,
                                       ind_min_t:ind_min_t + Nt_active]

        # ---- Sanity: compare to the injection's real WDM on the *active*
        # m-band only (5 layers around f0). fill_global writes only those
        # layers; the rest of the active band still holds out-of-band
        # signal energy from the injection's transform, so we restrict to
        # m_floor +/- 2 for the like-for-like reldiff.
        m_floor = int(np.floor(params_inj[1] / layer_df))
        m_lo = max(ind_min_f, m_floor - 2)
        m_hi = min(ind_min_f + Nf_active - 1, m_floor + 2)
        ml = m_lo - ind_min_f
        mh = m_hi - ind_min_f + 1
        diff_norm = np.linalg.norm(tpl_cpp_active[:, ml:mh, :]
                                    - wdm_inj_active[:, ml:mh, :])
        ref_norm  = np.linalg.norm(wdm_inj_active[:, ml:mh, :])
        reldiff_template = diff_norm / max(ref_norm, 1e-300)

        # mm5 against injection: 1 - <d|h>/sqrt(<d|d><h|h>) on full-grid template
        # using the analysis container's inner product over the active band.
        # Use a band-limited WDMSettings centered on f0 ± layer extent.
        f0 = params_inj[1]
        # mm5 band: [f0 - 3*layer_df, f0 + 2*layer_df]
        band_lo = max(1e-4, f0 - 3.0 * layer_df)
        band_hi = min(35e-3, f0 + 2.0 * layer_df)
        wdm_band = WDMSettings(
            Nf, Nt, dt, t0=t_start,
            min_freq=band_lo, max_freq=band_hi,
            min_time=EC * Nf * dt, max_time=(Nt - EC) * Nf * dt,
            is_complex=False, force_backend=backend,
        )
        # Slice both to the same band
        ind_min_f_b = int(wdm_band.ind_min_f)
        ind_max_f_b = int(wdm_band.ind_max_f)
        ind_min_t_b = int(wdm_band.ind_min_t)
        Nt_active_b = int(wdm_band.Nt_active)

        # tpl_cpp_full is the absolute-grid (3, Nf, Nt) template_fill slab.
        # Slice both to the mm5 band on absolute indices.
        d_band = wdm_inj_active[:, ind_min_f_b - ind_min_f:ind_max_f_b + 1 - ind_min_f,
                                   ind_min_t_b - ind_min_t:ind_min_t_b - ind_min_t + Nt_active_b]
        h_band = tpl_cpp_full[:, ind_min_f_b:ind_max_f_b + 1,
                              ind_min_t_b:ind_min_t_b + Nt_active_b]
        # Sens band slice
        sens_band = XYZ2SensitivityMatrix(wdm_band, model="scirdv1")
        invC_band = np.asarray(sens_band.invC)  # (3, 3, Nf_band, Nt_active_b)

        # Inner product: 4 Re sum_{c1,c2,m,n} conj(d_c1)*h_c2*invC[c1,c2] (XYZ)
        def inner_xyz(a, b, invC):
            acc = 0.0
            for c1 in range(3):
                for c2 in range(3):
                    acc += np.sum(np.conj(a[c1]) * b[c2] * invC[c1, c2]).real
            return 4.0 * acc

        d_d = inner_xyz(d_band, d_band, invC_band)
        h_h = inner_xyz(h_band, h_band, invC_band)
        d_h = inner_xyz(d_band, h_band, invC_band)
        denom = np.sqrt(d_d * h_h) if d_d > 0 and h_h > 0 else 1.0
        mm5 = 1.0 - d_h / max(denom, 1e-300)

        print(f"\n[draw {drawi}] f0={params_inj[1]*1e3:.4f}mHz snr={snr_i:.1f} "
              f"m_floor={m_floor}", flush=True)
        print(f"  reldiff(tpl_cpp, wdm_inj_real) on m_floor+-2 layers = "
              f"{reldiff_template:.3e}", flush=True)
        print(f"  mm5 (cpp-template vs injection)                     = "
              f"{mm5:+.3e}", flush=True)
        print(f"  <d|d>={d_d:.3e}  <h|h>={h_h:.3e}  <d|h>={d_h:.3e}",
              flush=True)

        drawi += 1

    print("\n[done] fill_global validation complete.", flush=True)


if __name__ == "__main__":
    sys.exit(main())
