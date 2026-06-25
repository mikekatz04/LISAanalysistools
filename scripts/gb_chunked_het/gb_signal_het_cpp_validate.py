#!/usr/bin/env python
"""Validate the new C++ ``gb_signal_het_get_ll`` against the Python v2 prototype.

Stage 1 of the GPU port plan. Tests:

  1. Build the same fixture as the v2 mm_sweep (SEED=54321, Nf=1460, Nt=2560,
     Tukey alpha=0.05, Nt_layer=64, m_active_half_width=2).
  2. For each of the first ``N_DRAWS`` prior draws:
       * Precompute reference c0_dense_complex (lisatools complex WDM at x_inj)
       * Precompute bin-folded A0/A1/B0/B1 from (c0, data, invC) -- the Python
         v1 ``_build_coefficients`` translated to a Python construction step
         that we hand to C++ as inputs.
       * Compute candidate FD = rfft(Tukey * td_cand) for x_cand = x_inj + DF0
       * Call C++ ``gb_signal_het_get_ll`` -- gets per-binary <d|h>, <h|h>.
       * Compute the same via the Python v1 prototype.
       * Print reldiff(d_h_cpp vs d_h_py), reldiff(h_h_cpp vs h_h_py),
         reldiff(logL_cpp vs logL_py).

  3. Assert all reldiffs <= ``RELDIFF_TARGET`` (default 1e-10).

Run::
    python gb_signal_het_cpp_validate.py
Env vars:
    N_DRAWS         default 3
    DF0_FRAC        default 0 (self-consistency: x_cand = x_inj)
    NT_LAYER        default 64
    SEED            default 54321
    RELDIFF_TARGET  default 1e-10
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


def python_bin_fold(data_complex, c0_complex, invC, n_b_idx_local, stride,
                    Nt_active, tdi_type):
    """Python port of ``_build_coefficients`` from gb_signal_het_wdm.py:325-356.

    Bins are uniform stride; last bin absorbs the leftover (matches v2 layout).
    Returns A0, A1, B0, B1 with shapes:
        A0/A1: (nch, Nf_active, N_sparse_t) complex
        B0/B1: (nch, nch, Nf_active, N_sparse_t) complex if XYZ
               (nch, Nf_active, N_sparse_t) complex if AE/AET
    """
    nch, Nf_act, _ = c0_complex.shape
    N_sparse_t = len(n_b_idx_local)
    # build _bin_idx (per-pixel bin index) + _n_off_per_pixel
    bin_edges = np.arange(N_sparse_t + 1) * stride
    bin_edges[-1] = Nt_active
    bin_sizes = np.diff(bin_edges).astype(int)
    bin_idx = np.repeat(np.arange(N_sparse_t), bin_sizes)
    assert bin_idx.shape[0] == Nt_active
    n_off = (np.arange(Nt_active) - n_b_idx_local[bin_idx]).astype(float)

    if tdi_type == "XYZ":
        D = np.einsum("cmn,cdmn->dmn", data_complex.conj(), invC)
    else:
        D = data_complex.conj() * invC

    w_A = D * c0_complex
    A0 = np.zeros((nch, Nf_act, N_sparse_t), dtype=np.complex128)
    A1 = np.zeros((nch, Nf_act, N_sparse_t), dtype=np.complex128)
    for b in range(N_sparse_t):
        mask = bin_idx == b
        A0[:, :, b] = w_A[:, :, mask].sum(axis=-1)
        A1[:, :, b] = (w_A[:, :, mask] * n_off[mask]).sum(axis=-1)

    if tdi_type == "XYZ":
        E = c0_complex.conj()[:, None, :, :] * c0_complex[None, :, :, :] * invC
        B0 = np.zeros((nch, nch, Nf_act, N_sparse_t), dtype=np.complex128)
        B1 = np.zeros((nch, nch, Nf_act, N_sparse_t), dtype=np.complex128)
        for b in range(N_sparse_t):
            mask = bin_idx == b
            B0[:, :, :, b] = E[:, :, :, mask].sum(axis=-1)
            B1[:, :, :, b] = (E[:, :, :, mask] * n_off[mask]).sum(axis=-1)
    else:
        E_diag = (c0_complex.conj() * c0_complex * invC).astype(np.complex128)
        B0 = np.zeros((nch, Nf_act, N_sparse_t), dtype=np.complex128)
        B1 = np.zeros((nch, Nf_act, N_sparse_t), dtype=np.complex128)
        for b in range(N_sparse_t):
            mask = bin_idx == b
            B0[:, :, b] = E_diag[:, :, mask].sum(axis=-1)
            B1[:, :, b] = (E_diag[:, :, mask] * n_off[mask]).sum(axis=-1)
    return A0, A1, B0, B1


def python_bin_fold_real(data_complex, c0_complex, invC, n_b_idx_local, stride,
                         Nt_active, tdi_type):
    """REAL-projection bin-fold coefficients (match the REAL WDM likelihood, not
    the Hermitian one). See gb_sighet_realproj_proto.py for the proof (1e-13).

    <d|h>: a SINGLE repacked complex coeff per channel,
        ``A0p = 2*(A0re + i*A0im)``,  A0re = Sum Re(d).ReInvC.Re(c0),
                                      A0im = Sum Re(d).ReInvC.Im(c0),
        so the EXISTING kernel's ``0.5*Re(A0p*r + A1p*dr)`` yields the exact real
        ``<d|h>`` at the SAME storage as today (no conjugate, no doubling).
    <h|h>: the conj blocks B0/B1 (as today) PLUS nonconj blocks B0nc/B1nc
        (``Sum c0_c.ReInvC.c0_c2`` with NO conjugate on c0_c). The kernel forms
        ``0.5*Re(B0.conj(rc)rc2 + B0nc.rc.rc2 + B1/B1nc dr terms)`` -> exact real
        ``<h|h>`` minus the dr^2 (noff^2) term (the linear-r budget; add a noff^2
        set later only if the off-reference check needs it).

    Returns A0p, A1p, B0, B1, B0nc, B1nc (same shapes as python_bin_fold's A0/B0).
    """
    nch, Nf_act, _ = c0_complex.shape
    N_sparse_t = len(n_b_idx_local)
    bin_edges = np.arange(N_sparse_t + 1) * stride; bin_edges[-1] = Nt_active
    bin_idx = np.repeat(np.arange(N_sparse_t), np.diff(bin_edges).astype(int))
    assert bin_idx.shape[0] == Nt_active
    n_off = (np.arange(Nt_active) - n_b_idx_local[bin_idx]).astype(float)

    Re_d = np.real(data_complex)
    iC = np.real(invC)                      # real WDM sensitivity
    u = np.real(c0_complex); w = np.imag(c0_complex)

    # ---- <d|h> repack: A0re/A0im integrands, packed into one complex ----
    if tdi_type == "XYZ":
        Dre = np.einsum("cmn,cdmn->dmn", Re_d, iC)   # Sum_c Re(d_c) ReInvC[c,d]
    else:
        Dre = Re_d * iC
    wA_re = Dre * u; wA_im = Dre * w
    A0p = np.zeros((nch, Nf_act, N_sparse_t), dtype=np.complex128)
    A1p = np.zeros((nch, Nf_act, N_sparse_t), dtype=np.complex128)
    for b in range(N_sparse_t):
        m = bin_idx == b; nf = n_off[m]
        A0p[:, :, b] = 2.0 * (wA_re[:, :, m].sum(-1) + 1j * wA_im[:, :, m].sum(-1))
        A1p[:, :, b] = 2.0 * ((wA_re[:, :, m] * nf).sum(-1) + 1j * (wA_im[:, :, m] * nf).sum(-1))

    # ---- <h|h>: conj + nonconj blocks (real invC) ----
    if tdi_type == "XYZ":
        Ec = c0_complex.conj()[:, None] * iC * c0_complex[None, :]   # conj(c0_c) iC c0_c2
        En = c0_complex[:, None] * iC * c0_complex[None, :]          # c0_c iC c0_c2 (no conj)
        shp = (nch, nch, Nf_act, N_sparse_t)
    else:
        Ec = c0_complex.conj() * iC * c0_complex
        En = c0_complex * iC * c0_complex
        shp = (nch, Nf_act, N_sparse_t)
    B0 = np.zeros(shp, dtype=np.complex128); B1 = np.zeros(shp, dtype=np.complex128)
    B0nc = np.zeros(shp, dtype=np.complex128); B1nc = np.zeros(shp, dtype=np.complex128)
    for b in range(N_sparse_t):
        m = bin_idx == b; nf = n_off[m]
        B0[..., b] = Ec[..., m].sum(-1); B1[..., b] = (Ec[..., m] * nf).sum(-1)
        B0nc[..., b] = En[..., m].sum(-1); B1nc[..., b] = (En[..., m] * nf).sum(-1)
    return A0p, A1p, B0, B1, B0nc, B1nc


def python_get_ll_v1(c1_sparse_active, m_local_active, c0_sparse_complex,
                      A0, A1, B0, B1, tdi_type, c0_mask):
    """Python v1 get_ll body translated for ONE binary, taking the
    polyphase output directly. Returns (d_h, h_h)."""
    nch, Nf_act, N_sparse_t = c0_sparse_complex.shape
    # Build full c1_sparse from active (matches v2 _r_and_dr zero-pad)
    c1_full = np.zeros_like(c0_sparse_complex)
    c1_full[:, m_local_active, :] = c1_sparse_active
    denom = np.where(c0_mask, c0_sparse_complex, 1.0 + 0.0j)
    r = np.where(c0_mask, c1_full / denom, 0.0 + 0.0j)
    Dn = float(N_sparse_t)   # not used directly; we use stride for dr
    # Actually use stride for Dn:
    # The Python v2 uses self._mean_Dn which is float(np.mean(bin_sizes)).
    # For our uniform-stride layout that's stride.
    # We'll pass stride explicitly later.
    Nb = r.shape[-1]
    dr = np.zeros_like(r)
    if Nb >= 3:
        dr[..., 1:-1] = (r[..., 2:] - r[..., :-2]) / 2.0
        dr[..., 0] = r[..., 1] - r[..., 0]
        dr[..., -1] = r[..., -1] - r[..., -2]
    # Note: we'll scale by 1/stride below
    return r, dr


def main():
    N_DRAWS = int(os.environ.get("N_DRAWS", "3"))
    Nt_layer = int(os.environ.get("NT_LAYER", "64"))
    SEED = int(os.environ.get("SEED", "54321"))
    RELDIFF_TARGET = float(os.environ.get("RELDIFF_TARGET", "1e-10"))

    # Sweep over multiple parameter directions Df{i} to characterize the
    # heterodyne approximation across all directions, not just f0. Each entry
    # is (label, param_idx_or_None, delta_callable(params_inj, layer_df)).
    PERTURBATIONS = [
        ("zero",            None, lambda x0, lf: 0.0),
        ("df0_p1e-3·lf",    1, lambda x0, lf: 1e-3 * lf),
        ("df0_p1e-2·lf",    1, lambda x0, lf: 1e-2 * lf),
        ("df0_p5e-2·lf",    1, lambda x0, lf: 5e-2 * lf),
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
    window = _tukey(Nobs, alpha=0.05).astype(float)

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

    np.random.seed(SEED)
    f0_lo = (ind_min_f + 7) * layer_df
    f0_hi = 0.025
    prior, tc = build_gb_prior(
        A_lims=(1e-23, 1e-20), f0_lims_hz=(f0_lo, f0_hi),
        fdot_lims=(-1e-15, 1e-15), beta_lims=None,
    )

    SNR_MIN, SNR_MAX = 5.0, 1100.0
    MAX_REJECT = 500

    # Backend handle
    cpp = _be.GBComputationGroupWrapCPU()
    sens_mat = None

    # Aggregate stats
    all_reldiffs = []

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

        wdm_inj_arr = np.asarray(wdm_inj_real.arr)                  # (3, Nf_act, Nt_act) real
        wdm_inj_complex = np.asarray(
            TDSignal(td_inj, settings=td_set).transform(
                wdm_set_complex, window=window).arr
        )                                                            # (3, Nf_act, Nt_act) complex
        # c0 dense = lisatools complex WDM at x_inj
        c0_dense_complex = wdm_inj_complex.copy()
        c0_sparse_complex = c0_dense_complex[:, :, n_sparse_local].copy()

        # invC (XYZ cross-channel) on complex WDM grid
        sens_complex = XYZ2SensitivityMatrix(wdm_set_complex, model="scirdv1")
        invC_complex = np.asarray(sens_complex.invC).copy()         # (3, 3, Nf_act, Nt_active)

        # Bin-fold A0/A1/B0/B1
        A0, A1, B0, B1 = python_bin_fold(
            wdm_inj_complex, c0_dense_complex, invC_complex,
            n_sparse_local, stride, Nt_active, tdi_type="XYZ",
        )
        # IMPORTANT: subsample to sparse positions for c0_sparse storage; A0/A1/B0/B1
        # already at sparse-bin grid via bin_fold (last axis size N_sparse_t).

        # Reference <d|d> for full logL (needs the data at x_inj inner-producted with itself)
        d_d = float(np.real(analysis.inner_product()))

        # Lisatools signal_gen for direct calculate_signal_likelihood path
        def _lisa_signal_gen(*params9):
            td = real_td_cb(np.asarray(params9, dtype=float).reshape(9))
            return TDSignal(td, settings=td_set).transform(
                wdm_set_real, window=window)
        analysis.signal_gen = _lisa_signal_gen

        print(f"\n[draw {drawi}] f0={params_inj[1]*1e3:.4f}mHz snr={snr_i:.1f}  "
              f"<d|d>={d_d:.4e}", flush=True)
        print(f"   {'perturbation':>18s} {'delta':>12s} {'logL_cpp':>13s} "
              f"{'logL_lisa':>13s} {'|cpp-lisa|':>11s} {'|cpp-py|':>11s}",
              flush=True)

        # Common per-draw arrays (computed once; reused for every perturbation)
        c0_sparse_all = c0_sparse_complex[None, ...].copy()
        A0_all = A0[None, ...].copy()
        A1_all = A1[None, ...].copy()
        B0_all = B0[None, ...].copy()
        B1_all = B1[None, ...].copy()
        params_ref_all = params_inj.astype(np.float64).reshape(1, 9).copy()
        data_index_all = np.zeros(1, dtype=np.int32)
        window_full = sparse_gen.window_full.astype(np.float64).copy()

        for label, idx, delta_fn in PERTURBATIONS:
            # Candidate
            params_cand = params_inj.copy()
            delta = float(delta_fn(params_inj, layer_df))
            if idx is not None:
                params_cand[idx] = params_inj[idx] + delta
            td_cand = real_td_cb(params_cand)
            fd_rfft_cand = np.fft.rfft(td_cand * window, axis=-1).astype(
                np.complex128)
            n_rfft = fd_rfft_cand.shape[-1]

            # Python prototype reference
            c1_active_py, m_local_active_py = sparse_gen.sparse_from_rfft(
                fd_rfft_cand, params_cand
            )
            c1_full_py = np.zeros((3, Nf_active, N_sparse_t), dtype=np.complex128)
            c1_full_py[:, m_local_active_py, :] = c1_active_py
            c0_mag = np.abs(c0_sparse_complex)
            floor = 1e-12 * c0_mag.max(axis=-1, keepdims=True)
            floor = np.maximum(floor, 1e-300)
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
            d_h_py_c = (A0 * r_py + A1 * dr_py).sum()
            r_outer = r_py.conj()[:, None, :, :] * r_py[None, :, :, :]
            cross_drr = (r_py.conj()[:, None, :, :] * dr_py[None, :, :, :]
                         + dr_py.conj()[:, None, :, :] * r_py[None, :, :, :])
            h_h_py_c = (B0 * r_outer + B1 * cross_drr).sum()
            d_h_py = 0.5 * float(d_h_py_c.real)
            h_h_py = 0.5 * float(h_h_py_c.real)

            # C++ call
            fd_rfft_all = fd_rfft_cand[None, ...].copy()
            params_cand_all = params_cand.astype(np.float64).reshape(1, 9).copy()
            d_h_out = np.zeros(1, dtype=np.float64)
            h_h_out = np.zeros(1, dtype=np.float64)
            cpp.gb_signal_het_get_ll(
                d_h_out, h_h_out,
                fd_rfft_all, c0_sparse_all,
                A0_all, A1_all, B0_all, B1_all,
                window_full, n_sparse_local,
                params_cand_all, params_ref_all,
                data_index_all,
                1, 1,
                9, 1, 2,
                Nf, Nt, Nf_active, Nt_active,
                Nt_layer, N_sparse_t, stride,
                ind_min_t, ind_min_f,
                2,
                layer_df, dt,
                3, 0,
                n_rfft,
            )
            d_h_cpp = float(d_h_out[0])
            h_h_cpp = float(h_h_out[0])

            rd_dh = abs(d_h_cpp - d_h_py) / max(abs(d_h_py), 1.0)
            rd_hh = abs(h_h_cpp - h_h_py) / max(abs(h_h_py), 1.0)
            # FULL logL = <d|h> - 0.5*<h|h> - 0.5*<d|d>
            logL_cpp = d_h_cpp - 0.5 * h_h_cpp - 0.5 * d_d
            logL_py  = d_h_py  - 0.5 * h_h_py  - 0.5 * d_d
            rd_ll = abs(logL_cpp - logL_py) / max(abs(logL_py), 1.0)
            all_reldiffs.append((rd_dh, rd_hh, rd_ll))

            # Direct lisatools likelihood at x_cand (closes the loop)
            logL_lisa = float(
                analysis.calculate_signal_likelihood(*params_cand, source_only=True)
            )
            err_lisa = abs(logL_cpp - logL_lisa)
            err_py = abs(logL_cpp - logL_py)
            print(f"   {label:>18s} {delta:+12.3e} {logL_cpp:+13.4e} "
                  f"{logL_lisa:+13.4e} {err_lisa:11.3e} {err_py:11.3e}", flush=True)

        drawi += 1

    max_rd = max(max(a, b, c) for a, b, c in all_reldiffs)
    print(f"\n[summary] N_DRAWS={N_DRAWS} Nt_layer={Nt_layer}",
          flush=True)
    print(f"  max reldiff (d_h, h_h, logL) = "
          f"({max(a for a, _, _ in all_reldiffs):.3e}, "
          f"{max(b for _, b, _ in all_reldiffs):.3e}, "
          f"{max(c for _, _, c in all_reldiffs):.3e})", flush=True)
    if max_rd <= RELDIFF_TARGET:
        print(f"\nPASS (max reldiff {max_rd:.3e} <= target {RELDIFF_TARGET:.3e})")
        return 0
    else:
        print(f"\nFAIL (max reldiff {max_rd:.3e} > target {RELDIFF_TARGET:.3e})")
        return 1


if __name__ == "__main__":
    sys.exit(main())
