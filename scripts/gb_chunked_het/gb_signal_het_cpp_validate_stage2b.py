#!/usr/bin/env python
"""Cross-validate Stage 2b in-kernel sparse-FD signal-het.

Stage 2b fuses gb_run_fd_wave_tdi (source-class heterodyned sparse rfft) with
the polyphase + bin-fold pipeline -- the candidate's X_het is built inside
the wrapper instead of externally. No per-source FD storage in global memory.

Per draw, three independent computations of <d|h>, <h|h>, logL:

  A) Stage 2a C++ (sparse X_het built from slicing dense rfft, k_f0 from
     Python rounding f0 -> bin)
  B) Stage 2b C++ (X_het built in-kernel via gb_run_fd_wave_tdi_wrap)
  C) Stage 1 C++ (dense rfft input, reference)

Stage 2a and Stage 2b should agree at heterodyne precision (~1e-4 within an
MCMC step; ~1e-7 at injection). The two paths source X_het differently:
  - Stage 2a: dense rfft(Tukey * td) sliced around k_f0 = round(f0 / df_abs)
  - Stage 2b: source-class direct sparse heterodyned FD (no full-N rfft)

Plus a parameter-direction sweep (Df{i}) per draw covering all 9 GB
directions, so the heterodyne accuracy is characterized end-to-end.

Run::
    python gb_signal_het_cpp_validate_stage2b.py
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
from gb_signal_het_cpp_validate import python_bin_fold


def main():
    N_DRAWS = int(os.environ.get("N_DRAWS", "3"))
    Nt_layer = int(os.environ.get("NT_LAYER", "64"))
    N_SPARSE_FD = int(os.environ.get("N_SPARSE_FD", "1024"))
    SEED = int(os.environ.get("SEED", "54321"))
    # Single source of truth: Python window AND C++ tukey_alpha use this.
    TUKEY_ALPHA = float(os.environ.get("TUKEY_ALPHA", "0.05"))

    # Df{i} sweep -- same set as gb_signal_het_cpp_validate.py
    PERTURBATIONS = [
        ("zero",            None, lambda x0, lf: 0.0),
        ("df0_p1e-3*lf",    1, lambda x0, lf: 1e-3 * lf),
        ("df0_p1e-2*lf",    1, lambda x0, lf: 1e-2 * lf),
        ("df0_p5e-2*lf",    1, lambda x0, lf: 5e-2 * lf),
        ("dfdot_p1e-18",    2, lambda x0, lf: 1e-18),
        ("dfdot_p1e-17",    2, lambda x0, lf: 1e-17),
        ("damp_p1e-3",      0, lambda x0, lf: x0[0] * 1e-3),
        ("damp_p1e-2",      0, lambda x0, lf: x0[0] * 1e-2),
        ("dphi0_p1e-4",     4, lambda x0, lf: 1e-4),
        ("dphi0_p1e-3",     4, lambda x0, lf: 1e-3),
        ("dbeta_p1e-4",     8, lambda x0, lf: 1e-4),
        ("dlam_p1e-4",      7, lambda x0, lf: 1e-4),
    ]

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
    tdi_wrap = gb_gen.wave_gen  # GBTDIonTheFlyWrapCPU

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
    print(f"[grid] Nf={Nf} Nt={Nt} layer_df={layer_df:.3e} Nt_active={Nt_active} "
          f"Nf_active={Nf_active}", flush=True)

    sparse_gen = GBSparseComplexWDMGen(
        real_td_callable=real_td_cb, wdm_set_complex=wdm_set_complex,
        data_dt=dt, ind_min_t=ind_min_t, Nt_active=Nt_active,
        Nt_layer=Nt_layer, m_active_half_width=2,
    )
    stride = sparse_gen.stride
    N_sparse_t = sparse_gen.N_sparse_t
    n_sparse_local = np.asarray(sparse_gen.n_sparse_local, dtype=np.int32)
    print(f"[v2] Nt_layer={Nt_layer} stride={stride} N_sparse_t={N_sparse_t}",
          flush=True)
    print(f"[stage2b] N_sparse_fd={N_SPARSE_FD} T_obs={Tobs:.3e} t_start={t_start:.3e}",
          flush=True)

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
    df_abs = 1.0 / Tobs

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

        wdm_inj_complex = np.asarray(
            TDSignal(td_inj, settings=td_set).transform(
                wdm_set_complex, window=window).arr)
        c0_dense_complex = wdm_inj_complex.copy()
        c0_sparse_complex = c0_dense_complex[:, :, n_sparse_local].copy()
        sens_complex = XYZ2SensitivityMatrix(wdm_set_complex, model="scirdv1")
        invC_complex = np.asarray(sens_complex.invC).copy()
        A0, A1, B0, B1 = python_bin_fold(
            wdm_inj_complex, c0_dense_complex, invC_complex,
            n_sparse_local, stride, Nt_active, tdi_type="XYZ",
        )
        # Common arrays
        c0_sparse_all = c0_sparse_complex[None, ...].copy()
        A0_all = A0[None, ...].copy()
        A1_all = A1[None, ...].copy()
        B0_all = B0[None, ...].copy()
        B1_all = B1[None, ...].copy()
        params_ref_all = params_inj.astype(np.float64).reshape(1, 9).copy()
        data_index_all = np.zeros(1, dtype=np.int32)

        print(f"\n[draw {drawi}] f0={params_inj[1]*1e3:.4f}mHz snr={snr_i:.1f}",
              flush=True)
        print(f"  {'perturbation':>18s} {'delta':>13s} "
              f"{'logL_s1':>14s} {'logL_s2a':>14s} {'logL_s2b':>14s} "
              f"{'rd(s2b,s2a)':>13s} {'rd(s2b,s1)':>13s}", flush=True)

        for label, idx, delta_fn in PERTURBATIONS:
            params_cand = params_inj.copy()
            delta = float(delta_fn(params_inj, layer_df))
            if idx is not None:
                params_cand[idx] = params_inj[idx] + delta

            params_cand_all = params_cand.astype(np.float64).reshape(1, 9).copy()
            td_cand = real_td_cb(params_cand)
            fd_rfft_cand = np.fft.rfft(td_cand * window, axis=-1).astype(np.complex128)
            n_rfft = fd_rfft_cand.shape[-1]

            f0_cand = float(params_cand[1])
            k_f0 = int(round(f0_cand / df_abs))
            half_NS = N_SPARSE_FD // 2

            # Build X_het by slicing dense rfft (Stage 2a path)
            X_het = np.zeros((3, N_SPARSE_FD), dtype=np.complex128)
            for c in range(3):
                for i in range(N_SPARSE_FD):
                    k_abs = k_f0 + (i - half_NS)
                    if 0 <= k_abs < n_rfft:
                        X_het[c, i] = fd_rfft_cand[c, k_abs]
            X_het_all = X_het[None, ...].copy()
            k_f0_all = np.array([k_f0], dtype=np.int32)

            # ---- A) Stage 1 C++ (dense FD) ----
            d_h_s1 = np.zeros(1, dtype=np.float64); h_h_s1 = np.zeros(1, dtype=np.float64)
            cpp.gb_signal_het_get_ll(
                d_h_s1, h_h_s1,
                fd_rfft_cand[None, ...].copy(), c0_sparse_all,
                A0_all, A1_all, B0_all, B1_all,
                window_full, n_sparse_local,
                params_cand_all, params_ref_all, data_index_all,
                1, 1,
                9, 1, 2,
                Nf, Nt, Nf_active, Nt_active,
                Nt_layer, N_sparse_t, stride,
                ind_min_t, ind_min_f,
                2,
                layer_df, dt,
                3, 0, n_rfft,
            )

            # ---- B) Stage 2a C++ (sparse, X_het from dense slice) ----
            d_h_s2a = np.zeros(1, dtype=np.float64); h_h_s2a = np.zeros(1, dtype=np.float64)
            cpp.gb_signal_het_get_ll_sparse(
                d_h_s2a, h_h_s2a,
                X_het_all, k_f0_all, c0_sparse_all,
                A0_all, A1_all, B0_all, B1_all,
                window_full, n_sparse_local,
                params_cand_all, params_ref_all, data_index_all,
                1, 1,
                9, 1, 2,
                Nf, Nt, Nf_active, Nt_active,
                Nt_layer, N_sparse_t, stride,
                ind_min_t, ind_min_f,
                2,
                layer_df, dt,
                3, 0, N_SPARSE_FD,
            )

            # ---- C) Stage 2b C++ (in-kernel X_het via gb_run_fd_wave_tdi) ----
            # Pass TUKEY_ALPHA to match the dense rfft(Tukey*td) used to build
            # the analysis-side reference. Python pushes the value through to
            # the sparse FFT inside gb_run_fd_wave_tdi -- single source of
            # truth tied to the constant at the top of main().
            d_h_s2b = np.zeros(1, dtype=np.float64); h_h_s2b = np.zeros(1, dtype=np.float64)
            cpp.gb_signal_het_get_ll_in_kernel(
                tdi_wrap,
                d_h_s2b, h_h_s2b,
                c0_sparse_all,
                A0_all, A1_all, B0_all, B1_all,
                window_full, n_sparse_local,
                params_cand_all, params_ref_all, data_index_all,
                1, 1,
                9, 1, 2,
                Nf, Nt, Nf_active, Nt_active,
                Nt_layer, N_sparse_t, stride,
                ind_min_t, ind_min_f,
                2,
                layer_df, dt,
                Tobs, t_start,
                3, 0, N_SPARSE_FD,
                TUKEY_ALPHA,
            )

            ll_s1 = float(d_h_s1[0]) - 0.5 * float(h_h_s1[0])
            ll_s2a = float(d_h_s2a[0]) - 0.5 * float(h_h_s2a[0])
            ll_s2b = float(d_h_s2b[0]) - 0.5 * float(h_h_s2b[0])

            def rd(a, b):
                return abs(a - b) / max(abs(b), 1.0)

            print(f"  {label:>18s} {delta:>+13.3e} "
                  f"{ll_s1:>+14.6e} {ll_s2a:>+14.6e} {ll_s2b:>+14.6e} "
                  f"{rd(ll_s2b, ll_s2a):>13.3e} {rd(ll_s2b, ll_s1):>13.3e}",
                  flush=True)

        drawi += 1

    print("\n[done] Stage 2b cross-backend validation complete.", flush=True)


if __name__ == "__main__":
    sys.exit(main())
