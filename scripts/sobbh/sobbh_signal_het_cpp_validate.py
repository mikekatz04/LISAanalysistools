#!/usr/bin/env python
"""Validate the C++ ``sobbh_signal_het_get_ll`` against a pure-Python prototype.

SOBBH duplicate of ``gb_chunked_het/gb_signal_het_cpp_validate.py`` (2026-06-18).
Stage-1 correctness gate for the SOBBH signal-het port. For each SOBBH prior
draw + a set of parameter perturbations:

  * Build reference c0_dense_complex (lisatools complex WDM at the injection).
  * Bin-fold A0/A1/B0/B1 from (c0, data, invC) -- the source-agnostic
    ``python_bin_fold`` from the GB scripts (reused verbatim).
  * Candidate FD = rfft(Tukey * td_cand).
  * C++ ``sobbh_signal_het_get_ll`` -> per-binary <d|h>, <h|h>.
  * Same via the Python v2 prototype.
  * reldiff(d_h), reldiff(h_h), reldiff(logL) + direct lisatools logL.

Assert all reldiffs <= RELDIFF_TARGET (default 1e-10): the C++ kernel must
reproduce the Python polyphase + bin-fold to machine precision.

Run::
    python sobbh_signal_het_cpp_validate.py
Env:
    N_DRAWS (3), NT_LAYER (64), SEED (54321), RELDIFF_TARGET (1e-10)
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

import bbhx  # noqa: F401  (registers bbhx_<flavor> backend family)
import bbhx_backend_cpu.cbbhx as _be

_GB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gb_chunked_het")
if _GB_DIR not in sys.path:
    sys.path.insert(0, _GB_DIR)
from gb_signal_het_wdm_v2 import GBSparseComplexWDMGen          # noqa: E402
from gb_signal_het_cpp_validate import python_bin_fold          # noqa: E402

NPARAMS = 11
F0_IDX = 5


def _sample_sobbh_params(rng, f_low_lo, f_low_hi, m1_lo, m1_hi, m2_lo, m2_hi):
    m1 = rng.uniform(m1_lo, m1_hi)
    m2 = rng.uniform(m2_lo, min(m2_hi, m1))
    s1 = rng.uniform(-0.5, 0.5); s2 = rng.uniform(-0.5, 0.5)
    distance = 10 ** rng.uniform(8.0, 9.5)
    f_low = rng.uniform(f_low_lo, f_low_hi)
    phi_c = rng.uniform(0.0, 2 * np.pi)
    cos_inc = rng.uniform(-1.0, 1.0)
    psi = rng.uniform(0.0, np.pi)
    lam = rng.uniform(0.0, 2 * np.pi)
    sin_beta = rng.uniform(-1.0, 1.0)
    return np.array([m1, m2, s1, s2, distance, f_low, phi_c,
                     np.arccos(cos_inc), psi, lam, np.arcsin(sin_beta)])


def main():
    N_DRAWS = int(os.environ.get("N_DRAWS", "3"))
    Nt_layer = int(os.environ.get("NT_LAYER", "64"))
    SEED = int(os.environ.get("SEED", "54321"))
    RELDIFF_TARGET = float(os.environ.get("RELDIFF_TARGET", "1e-10"))

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
    sobbh_gen = SOBBHTDIonTheFly(t_tdi, Tobs, t_start, 1.0 / dt, 1,
                                 tdi_config=tdi_config, orbits=orbits, tdi_chan="XYZ",
                                 force_backend=backend)

    def real_td_cb(p):
        m1, m2, s1, s2, distance, f_low, phi_c, inc, psi, lam, beta = p
        sp = sobbh_gen(np.array([m1]), np.array([m2]), np.array([s1]), np.array([s2]),
                       np.array([distance]), np.array([f_low]), np.array([phi_c]),
                       np.array([inc]), np.array([psi]), np.array([lam]), np.array([beta]),
                       convert_to_ra_dec=False, return_spline=True)
        return np.asarray(sp.eval_tdi(t_arr))[0]

    td_set = TDSettings(Nobs, dt, force_backend=backend)
    window = _tukey(Nobs, alpha=0.05).astype(float)

    wdm_kw = dict(t0=t_start, min_freq=1e-4, max_freq=35e-3,
                  min_time=EC * Nf * dt, max_time=(Nt - EC) * Nf * dt, force_backend=backend)
    wdm_set_real = WDMSettings(Nf, Nt, dt, is_complex=False, **wdm_kw)
    wdm_set_complex = WDMSettings(Nf, Nt, dt, is_complex=True, **wdm_kw)
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
        Nt_layer=Nt_layer, m_active_half_width=2)
    stride = sparse_gen.stride
    N_sparse_t = sparse_gen.N_sparse_t
    n_sparse_local = np.asarray(sparse_gen.n_sparse_local, dtype=np.int32)
    window_full = sparse_gen.window_full.astype(np.float64).copy()
    print(f"[v2] Nt_layer={Nt_layer} stride={stride} N_sparse_t={N_sparse_t}", flush=True)

    rng = np.random.default_rng(SEED)
    buf = 7
    f_low_lo = (ind_min_f + buf) * layer_df
    f_low_hi = (int(wdm_set_real.ind_max_f) - buf) * layer_df

    # f0 (idx5), m1 (idx0), phi_c (idx6), inc (idx7), lam (idx9), beta (idx10).
    PERTURBATIONS = [
        ("zero",          None, lambda x0, lf: 0.0),
        ("df0_p1e-3·lf",  5, lambda x0, lf: 1e-3 * lf),
        ("df0_p1e-2·lf",  5, lambda x0, lf: 1e-2 * lf),
        ("dphi_c_p1e-3",  6, lambda x0, lf: 1e-3),
        ("dinc_p1e-4",    7, lambda x0, lf: 1e-4),
        ("dlam_p1e-4",    9, lambda x0, lf: 1e-4),
        ("dbeta_p1e-4",  10, lambda x0, lf: 1e-4),
    ]

    cpp = _be.SOBBHComputationGroupWrapCPU()
    sens_mat = None
    all_reldiffs = []

    drawi = 0
    attempts = 0
    while drawi < N_DRAWS and attempts < 500:
        attempts += 1
        params_inj = _sample_sobbh_params(rng, f_low_lo, f_low_hi, 20.0, 50.0, 10.0, 40.0)
        td_inj = real_td_cb(params_inj)
        if not np.all(np.isfinite(td_inj)):
            continue
        wdm_inj_real = TDSignal(td_inj, settings=td_set).transform(wdm_set_real, window=window)
        inj_data_arr = DataResidualArray(wdm_inj_real)
        if sens_mat is None:
            sens_mat = XYZ2SensitivityMatrix(inj_data_arr.data_res_arr.settings, model="scirdv1")
        analysis = AnalysisContainer(inj_data_arr, sens_mat)
        snr_i = float(analysis.snr())
        if not (8.0 <= snr_i <= 2000.0):
            continue

        wdm_inj_complex = np.asarray(
            TDSignal(td_inj, settings=td_set).transform(wdm_set_complex, window=window).arr)
        c0_dense_complex = wdm_inj_complex.copy()
        c0_sparse_complex = c0_dense_complex[:, :, n_sparse_local].copy()
        sens_complex = XYZ2SensitivityMatrix(wdm_set_complex, model="scirdv1")
        invC_complex = np.asarray(sens_complex.invC).copy()
        A0, A1, B0, B1 = python_bin_fold(
            wdm_inj_complex, c0_dense_complex, invC_complex,
            n_sparse_local, stride, Nt_active, tdi_type="XYZ")
        d_d = float(np.real(analysis.inner_product()))

        def _lisa_signal_gen(*params11):
            td = real_td_cb(np.asarray(params11, dtype=float).reshape(NPARAMS))
            return TDSignal(td, settings=td_set).transform(wdm_set_real, window=window)
        analysis.signal_gen = _lisa_signal_gen

        print(f"\n[draw {drawi}] f_low={params_inj[5]*1e3:.4f}mHz snr={snr_i:.1f}  "
              f"<d|d>={d_d:.4e}", flush=True)
        print(f"   {'perturbation':>16s} {'delta':>12s} {'logL_cpp':>13s} "
              f"{'logL_lisa':>13s} {'|cpp-lisa|':>11s} {'|cpp-py|':>11s}", flush=True)

        c0_sparse_all = c0_sparse_complex[None, ...].copy()
        A0_all = A0[None, ...].copy(); A1_all = A1[None, ...].copy()
        B0_all = B0[None, ...].copy(); B1_all = B1[None, ...].copy()
        params_ref_all = params_inj.astype(np.float64).reshape(1, NPARAMS).copy()
        data_index_all = np.zeros(1, dtype=np.int32)

        for label, idx, delta_fn in PERTURBATIONS:
            params_cand = params_inj.copy()
            delta = float(delta_fn(params_inj, layer_df))
            if idx is not None:
                params_cand[idx] = params_inj[idx] + delta
            td_cand = real_td_cb(params_cand)
            fd_rfft_cand = np.fft.rfft(td_cand * window, axis=-1).astype(np.complex128)
            n_rfft = fd_rfft_cand.shape[-1]

            # Python prototype reference (dummy 9-vec carries f_low at index 1).
            g9 = np.zeros(9); g9[1] = params_cand[5]
            c1_active_py, m_local_active_py = sparse_gen.sparse_from_rfft(fd_rfft_cand, g9)
            c1_full_py = np.zeros((3, Nf_active, N_sparse_t), dtype=np.complex128)
            c1_full_py[:, m_local_active_py, :] = c1_active_py
            c0_mag = np.abs(c0_sparse_complex)
            floor = np.maximum(1e-12 * c0_mag.max(axis=-1, keepdims=True), 1e-300)
            c0_mask = c0_mag > floor
            denom = np.where(c0_mask, c0_sparse_complex, 1.0 + 0.0j)
            r_py = np.where(c0_mask, c1_full_py / denom, 0.0 + 0.0j)
            Dn = float(stride)
            dr_py = np.zeros_like(r_py)
            Nb = r_py.shape[-1]
            if Nb >= 3:
                dr_py[..., 1:-1] = (r_py[..., 2:] - r_py[..., :-2]) / (2.0 * Dn)
                dr_py[..., 0] = (r_py[..., 1] - r_py[..., 0]) / Dn
                dr_py[..., -1] = (r_py[..., -1] - r_py[..., -2]) / Dn
            d_h_py = 0.5 * float((A0 * r_py + A1 * dr_py).sum().real)
            r_outer = r_py.conj()[:, None, :, :] * r_py[None, :, :, :]
            cross_drr = (r_py.conj()[:, None, :, :] * dr_py[None, :, :, :]
                         + dr_py.conj()[:, None, :, :] * r_py[None, :, :, :])
            h_h_py = 0.5 * float((B0 * r_outer + B1 * cross_drr).sum().real)

            fd_rfft_all = fd_rfft_cand[None, ...].copy()
            params_cand_all = params_cand.astype(np.float64).reshape(1, NPARAMS).copy()
            d_h_out = np.zeros(1, dtype=np.float64); h_h_out = np.zeros(1, dtype=np.float64)
            cpp.sobbh_signal_het_get_ll(
                d_h_out, h_h_out, fd_rfft_all, c0_sparse_all,
                A0_all, A1_all, B0_all, B1_all,
                window_full, n_sparse_local,
                params_cand_all, params_ref_all, data_index_all,
                1, 1, NPARAMS, F0_IDX, F0_IDX,
                Nf, Nt, Nf_active, Nt_active,
                Nt_layer, N_sparse_t, stride,
                ind_min_t, ind_min_f, 2,
                layer_df, dt, 3, 0, n_rfft, -1.0)  # max_r<=0 -> no clip (match py)
            d_h_cpp = float(d_h_out[0]); h_h_cpp = float(h_h_out[0])

            rd_dh = abs(d_h_cpp - d_h_py) / max(abs(d_h_py), 1.0)
            rd_hh = abs(h_h_cpp - h_h_py) / max(abs(h_h_py), 1.0)
            logL_cpp = d_h_cpp - 0.5 * h_h_cpp - 0.5 * d_d
            logL_py = d_h_py - 0.5 * h_h_py - 0.5 * d_d
            rd_ll = abs(logL_cpp - logL_py) / max(abs(logL_py), 1.0)
            all_reldiffs.append((rd_dh, rd_hh, rd_ll))

            logL_lisa = float(analysis.calculate_signal_likelihood(*params_cand, source_only=True))
            print(f"   {label:>16s} {delta:+12.3e} {logL_cpp:+13.4e} "
                  f"{logL_lisa:+13.4e} {abs(logL_cpp - logL_lisa):11.3e} "
                  f"{abs(logL_cpp - logL_py):11.3e}", flush=True)
        drawi += 1

    if not all_reldiffs:
        print("[fail] no successful draws")
        return 1
    max_rd = max(max(a, b, c) for a, b, c in all_reldiffs)
    print(f"\n[summary] N_DRAWS={drawi} Nt_layer={Nt_layer}", flush=True)
    print(f"  max reldiff (d_h, h_h, logL) = ("
          f"{max(a for a, _, _ in all_reldiffs):.3e}, "
          f"{max(b for _, b, _ in all_reldiffs):.3e}, "
          f"{max(c for _, _, c in all_reldiffs):.3e})", flush=True)
    if max_rd <= RELDIFF_TARGET:
        print(f"\nPASS (max reldiff {max_rd:.3e} <= target {RELDIFF_TARGET:.3e})")
        return 0
    print(f"\nFAIL (max reldiff {max_rd:.3e} > target {RELDIFF_TARGET:.3e})")
    return 1


if __name__ == "__main__":
    sys.exit(main())
