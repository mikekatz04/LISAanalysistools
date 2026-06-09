#!/usr/bin/env python
"""Cross-validate Stage 2a sparse-FD signal-het: C++ vs Python mirror vs Stage 1.

Per draw, three independent computations of <d|h>, <h|h>, logL:

  A) Stage 1 C++ (dense rfft input) -- the validated reference from previous run
  B) Stage 2a C++ (sparse X_het + k_f0 input)
  C) Stage 2a Python mirror (gb_signal_het_v2_sparse_mirror.signal_het_get_ll_sparse_py)

All three should agree:
  - At low N_sparse_fd: small window-leakage discrepancy from truncating the
    dense rfft to a window of N_sparse_fd bins around k_f0 (the parts outside
    contain small residual signal energy)
  - At large N_sparse_fd >> Nt: leakage goes to zero, Stage 2a -> Stage 1 (FP)

Run::
    python gb_signal_het_cpp_validate_stage2.py
Env vars:
    N_DRAWS         default 2
    DF0_FRAC        default 0 (self-consistency)
    NT_LAYER        default 64
    N_SPARSE_FD     default 1024 (1024 captures ~all GB intrinsic bandwidth)
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
from gb_signal_het_v2_sparse_mirror import signal_het_get_ll_sparse_py


def main():
    N_DRAWS = int(os.environ.get("N_DRAWS", "2"))
    DF0_FRAC = float(os.environ.get("DF0_FRAC", "0"))
    Nt_layer = int(os.environ.get("NT_LAYER", "64"))
    N_SPARSE_FD = int(os.environ.get("N_SPARSE_FD", "1024"))
    SEED = int(os.environ.get("SEED", "54321"))
    # Tukey alpha applied to the dense TD before rfft (Stage 2a consumes the
    # resulting carrier-intact rfft slice; no separate C++ tukey is needed
    # because Stage 2a takes X_het as input rather than generating FD).
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
    print(f"[stage2a] N_sparse_fd={N_SPARSE_FD}", flush=True)

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

    # Absolute FD bin spacing
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

        wdm_inj_arr = np.asarray(wdm_inj_real.arr)
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

        # Candidate
        params_cand = params_inj.copy()
        params_cand[1] = params_inj[1] + DF0_FRAC * layer_df
        td_cand = real_td_cb(params_cand)
        fd_rfft_cand = np.fft.rfft(td_cand * window, axis=-1).astype(np.complex128)
        n_rfft = fd_rfft_cand.shape[-1]

        # Build X_het via downselect of dense rfft. C++ Stage 2a expects
        # X_het[i] = dense_rfft[k_f0 + (i - N_sparse_fd/2)].
        f0_cand = float(params_cand[1])
        k_f0 = int(round(f0_cand / df_abs))
        half_NS = N_SPARSE_FD // 2
        # Build via slice (zero-pad ends if needed)
        X_het = np.zeros((3, N_SPARSE_FD), dtype=np.complex128)
        for c in range(3):
            for i in range(N_SPARSE_FD):
                k_abs = k_f0 + (i - half_NS)
                if 0 <= k_abs < n_rfft:
                    X_het[c, i] = fd_rfft_cand[c, k_abs]
        X_het_all = X_het[None, ...].copy()                       # (1, 3, N_sparse_fd)
        k_f0_all = np.array([k_f0], dtype=np.int32)

        # Common arrays
        c0_sparse_all = c0_sparse_complex[None, ...].copy()
        A0_all = A0[None, ...].copy()
        A1_all = A1[None, ...].copy()
        B0_all = B0[None, ...].copy()
        B1_all = B1[None, ...].copy()
        params_cand_all = params_cand.astype(np.float64).reshape(1, 9).copy()
        params_ref_all = params_inj.astype(np.float64).reshape(1, 9).copy()
        data_index_all = np.zeros(1, dtype=np.int32)

        # ---- A) Stage 1 C++ (dense FD input) ----
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

        # ---- B) Stage 2a C++ (sparse FD input) ----
        d_h_s2 = np.zeros(1, dtype=np.float64); h_h_s2 = np.zeros(1, dtype=np.float64)
        cpp.gb_signal_het_get_ll_sparse(
            d_h_s2, h_h_s2,
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

        # ---- C) Stage 2a Python mirror ----
        d_h_py, h_h_py = signal_het_get_ll_sparse_py(
            X_het_all, k_f0_all,
            c0_sparse_all,
            A0_all, A1_all, B0_all, B1_all,
            window_full, n_sparse_local,
            params_cand_all, data_index_all,
            9, 1,
            Nf, Nt, Nf_active, Nt_layer, N_sparse_t, stride,
            ind_min_t, ind_min_f,
            2,
            layer_df, dt,
            3, 0, N_SPARSE_FD,
        )

        # Compare
        def rd(a, b):
            return abs(a - b) / max(abs(b), 1.0)

        s1 = (float(d_h_s1[0]), float(h_h_s1[0]))
        s2 = (float(d_h_s2[0]), float(h_h_s2[0]))
        py = (float(d_h_py[0]), float(h_h_py[0]))
        ll_s1 = s1[0] - 0.5 * s1[1]
        ll_s2 = s2[0] - 0.5 * s2[1]
        ll_py = py[0] - 0.5 * py[1]

        print(f"\n[draw {drawi}] f0={params_inj[1]*1e3:.4f}mHz snr={snr_i:.1f} "
              f"k_f0={k_f0}", flush=True)
        print(f"  Stage 1 (dense)    d_h={s1[0]:+.6e} h_h={s1[1]:+.6e} logL={ll_s1:+.6e}",
              flush=True)
        print(f"  Stage 2a (sparse C) d_h={s2[0]:+.6e} h_h={s2[1]:+.6e} logL={ll_s2:+.6e}",
              flush=True)
        print(f"  Stage 2a (Py mirror)d_h={py[0]:+.6e} h_h={py[1]:+.6e} logL={ll_py:+.6e}",
              flush=True)
        print(f"  reldiff(S2a-C vs S1)        : d_h={rd(s2[0], s1[0]):.3e} "
              f"h_h={rd(s2[1], s1[1]):.3e} logL={rd(ll_s2, ll_s1):.3e}",
              flush=True)
        print(f"  reldiff(S2a-C vs S2a-Py)    : d_h={rd(s2[0], py[0]):.3e} "
              f"h_h={rd(s2[1], py[1]):.3e} logL={rd(ll_s2, ll_py):.3e}",
              flush=True)

        drawi += 1


if __name__ == "__main__":
    sys.exit(main())
