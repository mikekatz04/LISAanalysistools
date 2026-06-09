#!/usr/bin/env python
"""Sweep fill_global sparse density while holding get_ll's density fixed.

Hypothesis: the residual ~1e-4 reldiff between
  A) C++ get_ll_in_kernel       (bin-folded A0/A1/B0/B1 inner product)
  B) C++ fill_global_in_kernel + lisatools template_inner_product
is dominated by linear-r-interpolation truncation in fill_global. get_ll's
bin-fold has an A1*dr/dn derivative correction; fill_global only linearly
interpolates r_sparse to dense. As fill_global's sparse density grows,
its linear interp converges to the true r at all dense pixels and the
gap to get_ll should shrink ~quadratically (linear-interp error ~ stride^2
times the curvature of r).

Design: get_ll is locked at the canonical NT_LAYER_GET_LL=64 (the
already-validated baseline) so its logL is fixed per draw. fill_global
is the only thing that varies; sweeping its Nt_layer changes its sparse
density via stride = Nt / Nt_layer, N_sparse_t ~ Nt_active / stride.

Nt_layer in {64, 96, 128, 192} gives fill_global stride {40, 27, 20, 13}
and N_sparse_t {63, 94, 126, 194}.

Run::
    python gb_signal_het_density_sweep.py
Env vars:
    N_DRAWS              default 3
    N_SPARSE_FD          default 1024
    SEED                 default 54321
    DF0_FRAC             default 1e-3
    NT_LAYER_GET_LL      default 64  (held fixed for get_ll)
    NT_LAYERS_FG         default "64,96,128,192" (swept for fill_global)
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
    N_DRAWS = int(os.environ.get("N_DRAWS", "3"))
    N_SPARSE_FD = int(os.environ.get("N_SPARSE_FD", "1024"))
    SEED = int(os.environ.get("SEED", "54321"))
    DF0_FRAC = float(os.environ.get("DF0_FRAC", "1e-3"))
    # get_ll is held fixed at this Nt_layer for every draw -- the baseline.
    NT_LAYER_GET_LL = int(os.environ.get("NT_LAYER_GET_LL", "64"))
    # fill_global sweeps these values; reldiff is measured against the FIXED
    # get_ll baseline so the sweep isolates fill_global's interpolation
    # convergence.
    NT_LAYERS_FG = [int(x) for x in
                    os.environ.get("NT_LAYERS_FG", "64,96,128,192").split(",")]
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
          f"Nt_active={Nt_active} Nf_active={Nf_active}", flush=True)
    print(f"[setup] get_ll fixed at Nt_layer={NT_LAYER_GET_LL}; "
          f"fill_global sweeps Nt_layer in {NT_LAYERS_FG}; "
          f"Df0_frac={DF0_FRAC}  TUKEY_ALPHA={TUKEY_ALPHA}",
          flush=True)

    # Build sparse_gen for get_ll (fixed) and each fill_global Nt_layer.
    all_ntls = sorted(set([NT_LAYER_GET_LL] + NT_LAYERS_FG))
    sparse_gens = {}
    for ntl in all_ntls:
        sg = GBSparseComplexWDMGen(
            real_td_callable=real_td_cb, wdm_set_complex=wdm_set_complex,
            data_dt=dt, ind_min_t=ind_min_t, Nt_active=Nt_active,
            Nt_layer=ntl, m_active_half_width=2,
        )
        sparse_gens[ntl] = sg
        tag = " (get_ll)" if ntl == NT_LAYER_GET_LL else ""
        print(f"  Nt_layer={ntl:>4d}  stride={sg.stride:>3d}  "
              f"N_sparse_t={sg.N_sparse_t:>4d}{tag}", flush=True)

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

    # Storage for stats per fill_global Nt_layer across draws.
    stats = {ntl: {"reldiff": []} for ntl in NT_LAYERS_FG}

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

        wdm_inj_complex = np.asarray(
            TDSignal(td_inj, settings=td_set).transform(
                wdm_set_complex, window=window).arr)
        c0_dense_active = wdm_inj_complex.copy()
        invC_complex = np.asarray(sens_mat_complex.invC).copy()

        params_ref_all = params_inj.astype(np.float64).reshape(1, 9).copy()
        data_index_all = np.zeros(1, dtype=np.int32)
        factors_all = np.ones(1, dtype=np.float64)
        params_cand = params_inj.copy()
        params_cand[1] = params_inj[1] + DF0_FRAC * layer_df
        params_cand_all = params_cand.astype(np.float64).reshape(1, 9).copy()

        # ---- get_ll baseline (held fixed per draw) ----
        sg_b = sparse_gens[NT_LAYER_GET_LL]
        stride_b = sg_b.stride
        N_sparse_t_b = sg_b.N_sparse_t
        n_sparse_local_b = np.asarray(sg_b.n_sparse_local, dtype=np.int32)
        window_full_b = sg_b.window_full.astype(np.float64).copy()
        c0_sparse_b = c0_dense_active[:, :, n_sparse_local_b].copy()
        A0_b, A1_b, B0_b, B1_b = python_bin_fold(
            wdm_inj_complex, c0_dense_active, invC_complex,
            n_sparse_local_b, stride_b, Nt_active, tdi_type="XYZ",
        )
        d_h_g = np.zeros(1, dtype=np.float64)
        h_h_g = np.zeros(1, dtype=np.float64)
        cpp.gb_signal_het_get_ll_in_kernel(
            tdi_wrap,
            d_h_g, h_h_g,
            c0_sparse_b[None, ...].copy(),
            A0_b[None, ...].copy(), A1_b[None, ...].copy(),
            B0_b[None, ...].copy(), B1_b[None, ...].copy(),
            window_full_b, n_sparse_local_b,
            params_cand_all, params_ref_all, data_index_all,
            1, 1,
            9, 1, 2,
            Nf, Nt, Nf_active, Nt_active,
            NT_LAYER_GET_LL, N_sparse_t_b, stride_b,
            ind_min_t, ind_min_f,
            2,
            layer_df, dt,
            Tobs, t_start,
            3, 0, N_SPARSE_FD,
            TUKEY_ALPHA,
        )
        logL_get_ll = float(d_h_g[0]) - 0.5 * float(h_h_g[0])

        print(f"\n[draw {drawi}] f0={params_inj[1]*1e3:.4f}mHz "
              f"snr={snr_i:.1f}  logL_get_ll={logL_get_ll:+.6e} "
              f"(Nt_layer={NT_LAYER_GET_LL}, N_sparse_t={N_sparse_t_b})",
              flush=True)
        print(f"  {'Nt_layer_fg':>11s} {'N_sp_t':>6s} {'stride':>6s} "
              f"{'logL_fg_lat':>14s} "
              f"{'abs_diff':>10s} {'reldiff':>10s}", flush=True)

        # ---- fill_global density sweep, all compared to fixed get_ll ----
        c0_dense_all = c0_dense_active[None, ...].copy()
        for ntl in NT_LAYERS_FG:
            sg = sparse_gens[ntl]
            stride = sg.stride
            N_sparse_t = sg.N_sparse_t
            n_sparse_local = np.asarray(sg.n_sparse_local, dtype=np.int32)
            window_full = sg.window_full.astype(np.float64).copy()
            c0_sparse_active = c0_dense_active[:, :, n_sparse_local].copy()

            template_fill = np.zeros((1, 3, Nf, Nt), dtype=np.float64)
            cpp.gb_signal_het_fill_global_in_kernel(
                tdi_wrap,
                template_fill,
                c0_sparse_active[None, ...].copy(),
                c0_dense_all,
                window_full, n_sparse_local,
                params_cand_all, params_ref_all,
                factors_all, data_index_all,
                1, 1,
                9, 1, 2,
                Nf, Nt, Nf_active, Nt_active,
                ntl, N_sparse_t, stride,
                ind_min_t, ind_min_f,
                2,
                layer_df, dt,
                Tobs, t_start,
                3,
                N_SPARSE_FD, TUKEY_ALPHA,
            )
            tpl_signal = WDMSignal(template_fill[0], wdm_set_real)
            d_h_fg = float(np.real(
                analysis.template_inner_product(tpl_signal, complex=True)))
            h_h_fg = float(np.real(
                inner_product(tpl_signal, tpl_signal, psd=sens_mat_real)))
            logL_fg_lat = d_h_fg - 0.5 * h_h_fg

            abs_diff = abs(logL_get_ll - logL_fg_lat)
            denom = max(abs(logL_get_ll), abs(logL_fg_lat), 1.0)
            reldiff = abs_diff / denom
            stats[ntl]["reldiff"].append(reldiff)

            print(f"  {ntl:>11d} {N_sparse_t:>6d} {stride:>6d} "
                  f"{logL_fg_lat:>+14.6e} "
                  f"{abs_diff:>10.3e} {reldiff:>10.3e}", flush=True)

        drawi += 1

    # Summary across draws. get_ll is fixed at NT_LAYER_GET_LL; the table
    # shows how reldiff(get_ll, fill_global+lisatools) responds to upping
    # fill_global's density.
    print(f"\n[summary] reldiff(logL_get_ll@Nt_layer={NT_LAYER_GET_LL}, "
          f"logL_fg_lat@Nt_layer_fg) across {N_DRAWS} draws", flush=True)
    print(f"  {'Nt_layer_fg':>11s} {'N_sp_t':>6s} {'median':>11s} "
          f"{'p90':>11s} {'max':>11s}", flush=True)
    for ntl in NT_LAYERS_FG:
        r = np.asarray(stats[ntl]["reldiff"])
        n_sp_t = sparse_gens[ntl].N_sparse_t
        med = float(np.median(r)) if r.size else float("nan")
        p90 = float(np.quantile(r, 0.90)) if r.size else float("nan")
        mx = float(np.max(r)) if r.size else float("nan")
        print(f"  {ntl:>11d} {n_sp_t:>6d} {med:>11.3e} {p90:>11.3e} {mx:>11.3e}",
              flush=True)


if __name__ == "__main__":
    sys.exit(main())
