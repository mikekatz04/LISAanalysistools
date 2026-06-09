#!/usr/bin/env python
"""Compare get_ll vs (fill_global -> lisatools inner product) over prior draws.

End-to-end consistency check between the two C++ signal-het entry points:

  A) C++ get_ll_in_kernel: bin-folded inner-product accumulator
       d_h = sum (A0 r + A1 dr/dn)         per (c, m_active, b_sparse)
       h_h = sum (B0 r*r + B1 cross terms) per (c, c2, m_active, b_sparse)
     -> logL_get_ll = d_h - 0.5 * h_h

  B) C++ fill_global_in_kernel: same r_sparse pipeline, but
     reconstructs the dense template via linear-interp r_demod ->
     re-rotate -> multiply by stored c0_dense_complex -> scatter
     Re(c1_dense) into template_fill[data, c, m_global, n_global].
     Then logL via lisatools' canonical inner product:
       template = WDMSignal(template_fill[0], wdm_set_real)
       d_h = AnalysisContainer.template_inner_product(template, complex=True)
       h_h = inner_product(template, template, psd=sens_mat)
       logL_fg = Re(d_h) - 0.5 * h_h

Both paths share the same r_sparse + carrier de-rotation, so any
divergence is rooted in (a) the bin-fold vs full-template inner-product
sum order or (b) the linear-r interpolation in fill_global vs the
A1/B1 derivative correction in get_ll. Across a prior sweep, the two
logLs should track closely.

Run::
    python gb_signal_het_cpp_validate_get_ll_vs_fill_global.py
Env vars:
    N_DRAWS         default 5
    NT_LAYER        default 64
    N_SPARSE_FD     default 1024
    SEED            default 54321
    DF0_FRAC        default 1e-3  -- candidate offset from injection in
                                     units of layer_df.  0 = self-consistency.
"""

from __future__ import annotations

import os
import sys

import numpy as np
from scipy.signal.windows import tukey as _tukey

from lisatools.analysiscontainer import AnalysisContainer
from lisatools.datacontainer import DataResidualArray
from lisatools.detector import ESAOrbits
from lisatools.diagnostic import inner_product
from lisatools.domains import TDSettings, TDSignal, WDMSettings, WDMSignal
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
    N_DRAWS = int(os.environ.get("N_DRAWS", "5"))
    Nt_layer = int(os.environ.get("NT_LAYER", "64"))
    N_SPARSE_FD = int(os.environ.get("N_SPARSE_FD", "1024"))
    SEED = int(os.environ.get("SEED", "54321"))
    DF0_FRAC = float(os.environ.get("DF0_FRAC", "1e-3"))
    TUKEY_ALPHA = 0.05

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
          f"Nt_active={Nt_active} Nf_active={Nf_active}", flush=True)

    sparse_gen = GBSparseComplexWDMGen(
        real_td_callable=real_td_cb, wdm_set_complex=wdm_set_complex,
        data_dt=dt, ind_min_t=ind_min_t, Nt_active=Nt_active,
        Nt_layer=Nt_layer, m_active_half_width=2,
    )
    stride = sparse_gen.stride
    N_sparse_t = sparse_gen.N_sparse_t
    n_sparse_local = np.asarray(sparse_gen.n_sparse_local, dtype=np.int32)
    print(f"[v2] Nt_layer={Nt_layer} stride={stride} N_sparse_t={N_sparse_t} "
          f"N_sparse_fd={N_SPARSE_FD} tukey_alpha={TUKEY_ALPHA} "
          f"Df0_frac={DF0_FRAC}", flush=True)

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
    sens_mat_real = None
    sens_mat_complex = None
    window_full = sparse_gen.window_full.astype(np.float64).copy()

    print(f"\n{'draw':>4s} {'f0[mHz]':>9s} {'snr':>6s} "
          f"{'logL_get_ll':>14s} {'logL_fg_lat':>14s} "
          f"{'abs_diff':>11s} {'reldiff':>11s} "
          f"{'<d|h>_g':>12s} {'<d|h>_fg':>12s} "
          f"{'<h|h>_g':>12s} {'<h|h>_fg':>12s}", flush=True)

    drawi = 0
    while drawi < N_DRAWS:
        for _ in range(MAX_REJECT):
            x_samp = prior.rvs(size=1)
            params_inj = tc.both_transforms(x_samp.copy())[0]
            td_inj = real_td_cb(params_inj)
            wdm_inj_real = TDSignal(td_inj, settings=td_set).transform(
                wdm_set_real, window=window)
            inj_data_arr = DataResidualArray(wdm_inj_real)
            if sens_mat_real is None:
                sens_mat_real = XYZ2SensitivityMatrix(
                    inj_data_arr.data_res_arr.settings, model="scirdv1")
                sens_mat_complex = XYZ2SensitivityMatrix(wdm_set_complex,
                                                         model="scirdv1")
            analysis = AnalysisContainer(inj_data_arr, sens_mat_real)
            snr_i = float(analysis.snr())
            if SNR_MIN <= snr_i <= SNR_MAX:
                break

        # Reference c0 (complex) + bin-folded A/B coefficients.
        wdm_inj_complex = np.asarray(
            TDSignal(td_inj, settings=td_set).transform(
                wdm_set_complex, window=window).arr)
        c0_dense_active = wdm_inj_complex.copy()
        c0_sparse_active = c0_dense_active[:, :, n_sparse_local].copy()
        invC_complex = np.asarray(sens_mat_complex.invC).copy()
        A0, A1, B0, B1 = python_bin_fold(
            wdm_inj_complex, c0_dense_active, invC_complex,
            n_sparse_local, stride, Nt_active, tdi_type="XYZ",
        )

        c0_sparse_all = c0_sparse_active[None, ...].copy()
        c0_dense_all  = c0_dense_active[None, ...].copy()
        A0_all = A0[None, ...].copy()
        A1_all = A1[None, ...].copy()
        B0_all = B0[None, ...].copy()
        B1_all = B1[None, ...].copy()
        params_ref_all = params_inj.astype(np.float64).reshape(1, 9).copy()
        data_index_all = np.zeros(1, dtype=np.int32)
        factors_all = np.ones(1, dtype=np.float64)

        params_cand = params_inj.copy()
        params_cand[1] = params_inj[1] + DF0_FRAC * layer_df
        params_cand_all = params_cand.astype(np.float64).reshape(1, 9).copy()

        # ---- A) C++ get_ll ----
        d_h_g = np.zeros(1, dtype=np.float64)
        h_h_g = np.zeros(1, dtype=np.float64)
        cpp.gb_signal_het_get_ll_in_kernel(
            tdi_wrap,
            d_h_g, h_h_g,
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
        d_h_get_ll = float(d_h_g[0])
        h_h_get_ll = float(h_h_g[0])
        logL_get_ll = d_h_get_ll - 0.5 * h_h_get_ll

        # ---- B) C++ fill_global -> lisatools inner product ----
        template_fill = np.zeros((1, 3, Nf, Nt), dtype=np.float64)
        cpp.gb_signal_het_fill_global_in_kernel(
            tdi_wrap,
            template_fill,
            c0_sparse_all,
            c0_dense_all,
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
        # Wrap full (3, Nf, Nt) buffer as a WDMSignal on the real grid; the
        # constructor auto-slices to (3, Nf_active, Nt_active).
        tpl_signal = WDMSignal(template_fill[0], wdm_set_real)

        # lisatools inner product on the active band.
        d_h_fg_complex = analysis.template_inner_product(tpl_signal, complex=True)
        h_h_fg = inner_product(tpl_signal, tpl_signal, psd=sens_mat_real)
        d_h_fg = float(np.real(d_h_fg_complex))
        h_h_fg = float(np.real(h_h_fg))
        logL_fg_lat = d_h_fg - 0.5 * h_h_fg

        abs_diff = abs(logL_get_ll - logL_fg_lat)
        denom = max(abs(logL_get_ll), abs(logL_fg_lat), 1.0)
        reldiff = abs_diff / denom

        print(f"{drawi:>4d} {params_inj[1]*1e3:>9.4f} {snr_i:>6.1f} "
              f"{logL_get_ll:>+14.6e} {logL_fg_lat:>+14.6e} "
              f"{abs_diff:>11.3e} {reldiff:>11.3e} "
              f"{d_h_get_ll:>+12.4e} {d_h_fg:>+12.4e} "
              f"{h_h_get_ll:>+12.4e} {h_h_fg:>+12.4e}",
              flush=True)

        drawi += 1

    print("\n[done] get_ll vs fill_global+lisatools across prior draws.",
          flush=True)


if __name__ == "__main__":
    sys.exit(main())
