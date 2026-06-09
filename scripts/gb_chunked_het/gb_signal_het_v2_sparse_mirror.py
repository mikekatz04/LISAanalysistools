"""Python mirror of the C++ Stage 2a sparse-FD signal-het polyphase + bin-fold.

Mirrors ``GBComputationGroup::gb_signal_het_get_ll_sparse_wrap`` algorithmically.
Used both for understanding and for the validation script
``gb_signal_het_cpp_validate_stage2.py``.

Key change vs Stage 1 (which used a dense rfft input):
  - Polyphase fold iterates only N_sparse_fd nonzero bins of the sparse FD
    (the source's intrinsic spectral support around f0 in absolute frame)
  - X_het[i] holds the absolute FD value at bin k = k_f0 + (i - N_sparse_fd/2)
"""

from __future__ import annotations

import numpy as np


def signal_het_get_ll_sparse_py(
    X_het_all,                  # (num_bin, nch, N_sparse_fd) complex128
    k_f0_all,                   # (num_bin,) int -- absolute-FD bin index of f0
    c0_sparse_all,              # (num_data, nch, Nf_active, N_sparse_t) complex128
    A0_all, A1_all,             # (num_data, nch, Nf_active, N_sparse_t) complex128
    B0_all, B1_all,             # (num_data, nch, nch, Nf_active, N_sparse_t) for XYZ
    wdm_window,                 # (Nt,) float64 -- phitilde at full-Nt grid
    n_sparse_local_arr,         # (N_sparse_t,) int -- sparse n positions in local active
    params_cand_all,            # (num_bin, nparams) float64
    data_index_all,             # (num_bin,) int
    nparams, f0_idx,
    Nf, Nt, Nf_active, Nt_layer, N_sparse_t, stride,
    ind_min_t, ind_min_f,
    m_active_half_width,
    layer_df, dt,
    nchannels, tdi_type,
    N_sparse_fd,
):
    """Return (d_h_out, h_h_out) arrays of length num_bin."""
    num_bin = X_het_all.shape[0]
    M = 2 * m_active_half_width + 1
    half_Nt = Nt // 2
    half_NS = N_sparse_fd // 2
    kappa = 2.0 * np.sqrt(np.pi * dt) / float(Nf)
    n_start = ind_min_t + int(n_sparse_local_arr[0])
    TWO_PI = 2.0 * np.pi

    d_h_out = np.zeros(num_bin, dtype=np.float64)
    h_h_out = np.zeros(num_bin, dtype=np.float64)

    for bin_i in range(num_bin):
        f0_cand = float(params_cand_all[bin_i, f0_idx])
        m_floor = int(np.floor(f0_cand / layer_df))
        m_active = np.clip(
            m_floor + np.arange(-m_active_half_width, m_active_half_width + 1),
            ind_min_f, ind_min_f + Nf_active - 1,
        )
        data_idx = int(data_index_all[bin_i])
        k_f0 = int(k_f0_all[bin_i])

        # ---- Polyphase fold over N_sparse_fd nonzero bins -----------
        fold = np.zeros((nchannels, M, Nt_layer), dtype=np.complex128)
        for i in range(N_sparse_fd):
            k_abs = k_f0 + (i - half_NS)
            # For each active m-layer, see if this absolute bin lies in its
            # Nt-wide polyphase window.
            for im in range(M):
                j = k_abs - m_active[im] * half_Nt + half_Nt
                if not (0 <= j < Nt):
                    continue
                j_off = j - half_Nt
                phase_arg = TWO_PI * j_off * n_start / Nt
                prephase = np.exp(1j * phase_arg)
                w_pp = wdm_window[j] * prephase
                r_slot = j % Nt_layer
                # All channels at once
                fold[:, im, r_slot] += X_het_all[bin_i, :, i] * w_pp

        # ---- iFFT length Nt_layer (keep first N_sparse_t outputs) ----
        c1_sparse = np.zeros((nchannels, M, N_sparse_t), dtype=np.complex128)
        # Use numpy's IFFT for the per (c, im) fold:
        ifft_full = np.fft.ifft(fold, n=Nt_layer, axis=-1)   # (nch, M, Nt_layer)
        # Apply per-pixel lisatools coefficient
        for n_layer in range(N_sparse_t):
            n_global = n_start + n_layer * stride
            sign_scale = ((-1.0) ** n_global) / float(stride)
            after_ifft_lt = ifft_full[:, :, n_layer] * sign_scale  # (nch, M)
            # Per m_active layer: coef depends on m_global
            for im in range(M):
                m_global = int(m_active[im])
                m_plus_n = (m_global + n_global) & 1
                conj_cmn = 1.0 + 0.0j if m_plus_n == 0 else 0.0 - 1.0j
                sign_mn = (-1.0) ** ((m_global + 1) * n_global)
                coef = kappa * sign_mn * conj_cmn
                c1_sparse[:, im, n_layer] = after_ifft_lt[:, im] * coef

        # ---- r and dr/dn at sparse bin centres ----------------------
        m_local = m_active - ind_min_f
        c0_active = np.empty((nchannels, M, N_sparse_t), dtype=np.complex128)
        for c in range(nchannels):
            for im in range(M):
                c0_active[c, im, :] = c0_sparse_all[data_idx, c, int(m_local[im]), :]
        c0_mag = np.abs(c0_active)
        floor = 1e-12 * c0_mag.max(axis=-1, keepdims=True)
        floor = np.maximum(floor, 1e-300)
        mask = c0_mag > floor
        denom = np.where(mask, c0_active, 1.0 + 0.0j)
        r = np.where(mask, c1_sparse / denom, 0.0 + 0.0j)
        # dr/dn via centred FD with mean bin width = stride
        Dn = float(stride)
        dr = np.zeros_like(r)
        Nb = N_sparse_t
        if Nb >= 3:
            dr[..., 1:-1] = (r[..., 2:] - r[..., :-2]) / (2.0 * Dn)
            dr[..., 0]    = (r[..., 1] - r[..., 0]) / Dn
            dr[..., -1]   = (r[..., -1] - r[..., -2]) / Dn

        # ---- Bin-folded inner products ----------------------------
        d_h_raw = 0.0 + 0.0j
        h_h_raw = 0.0 + 0.0j
        for c in range(nchannels):
            for im in range(M):
                ml = int(m_local[im])
                a0 = A0_all[data_idx, c, ml, :]
                a1 = A1_all[data_idx, c, ml, :]
                d_h_raw += (a0 * r[c, im] + a1 * dr[c, im]).sum()

        if tdi_type == 0:   # XYZ
            for c in range(nchannels):
                for c2 in range(nchannels):
                    for im in range(M):
                        ml = int(m_local[im])
                        b0 = B0_all[data_idx, c, c2, ml, :]
                        b1 = B1_all[data_idx, c, c2, ml, :]
                        r_outer = np.conj(r[c, im]) * r[c2, im]
                        cross_drr = (np.conj(r[c, im]) * dr[c2, im]
                                     + np.conj(dr[c, im]) * r[c2, im])
                        h_h_raw += (b0 * r_outer + b1 * cross_drr).sum()
        else:               # AE/AET diag
            for c in range(nchannels):
                for im in range(M):
                    ml = int(m_local[im])
                    b0 = B0_all[data_idx, c, ml, :]
                    b1 = B1_all[data_idx, c, ml, :]
                    rsq = (np.conj(r[c, im]) * r[c, im]).real
                    cross_drr = (np.conj(r[c, im]) * dr[c, im]
                                 + np.conj(dr[c, im]) * r[c, im])
                    h_h_raw += (b0 * rsq + b1 * cross_drr).sum()

        d_h_out[bin_i] = 0.5 * d_h_raw.real
        h_h_out[bin_i] = 0.5 * h_h_raw.real

    return d_h_out, h_h_out
