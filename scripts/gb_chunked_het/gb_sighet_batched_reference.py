"""GBSignalHetReferenceSet -- batched, sub-batched sig-het reference generator.

Builds the sig-het reference quantities (carrier ``c0`` + bin-fold coefficients
``A0/A1/B0/B1/B0nc/B1nc``) for a SET of ``num_data`` references in one object, and
evaluates ``get_ll`` for candidates that each pick their reference by a length-
``num_bin`` ``data_index`` (the kernel already supports this -- proven in
``test_sighet_vectorized_dataindex.py``).

Design (per user direction):
  * **Sub-batched** over references (``ref_batch_size``): the dense ``c0`` and the
    bin-fold ``Ec/En`` intermediates are ``(B, nch, nch, Nf_active, Nt_active)`` --
    memory-heavy, so references are processed ``ref_batch_size`` at a time.
  * **chunked-het c0**: each reference's dense ``c0`` is built with
    ``reference_impl="chunked"`` (the polyphase WDM over the active band at full Nt
    resolution -- the same machinery the kernel runs for candidate ``c1``), NOT the
    full dense transform. Toggleable via ``reference_impl``.
  * **Vectorized bin-fold**: ``_bin_fold_real_batched`` runs the bin-fold over the
    whole sub-batch's leading axis in one set of einsums (not B separate calls).

Validated against per-reference ``GBSignalHetComputations`` in
``test_sighet_batched_reference.py``.
"""
from __future__ import annotations

import numpy as np

from gbsignalhetcomputations import GBSignalHetComputations, reference_c0_dense


def _bin_fold_real_batched(data_complex, c0_all, invC, n_b_idx_local, stride,
                           Nt_active, tdi_type="XYZ"):
    """Vectorized ``python_bin_fold_real`` over a leading ``B`` axis on ``c0_all``.

    ``c0_all`` ``(B, nch, Nf_act, Nt_active)`` -- one reference per slot.
    ``data_complex``/``invC`` shared across the sub-batch (each reference can carry its
    own ``R_cond`` data later -- add a leading B to ``Re_d``/``Dre`` then). Returns
    ``A0p/A1p`` ``(B, nch, Nf_act, N_sparse_t)`` and ``B0/B1/B0nc/B1nc``
    ``(B, nch, nch, Nf_act, N_sparse_t)`` (XYZ). Mirrors ``python_bin_fold_real`` term
    for term; validated == per-reference.
    """
    B, nch, Nf_act, _ = c0_all.shape
    N_sparse_t = len(n_b_idx_local)
    bin_edges = np.arange(N_sparse_t + 1) * stride; bin_edges[-1] = Nt_active
    bin_idx = np.repeat(np.arange(N_sparse_t), np.diff(bin_edges).astype(int))
    assert bin_idx.shape[0] == Nt_active
    n_off = (np.arange(Nt_active) - n_b_idx_local[bin_idx]).astype(float)

    Re_d = np.real(data_complex); iC = np.real(invC)
    u = np.real(c0_all); w = np.imag(c0_all)                    # (B, nch, Nf, Nt_active)

    # ---- <d|h> repack (A0re/A0im integrands), shared data folded with each c0 ----
    if tdi_type == "XYZ":
        Dre = np.einsum("cmn,cdmn->dmn", Re_d, iC)              # (nch, Nf, Nt_active)
    else:
        Dre = Re_d * iC
    wA_re = Dre[None] * u; wA_im = Dre[None] * w                # (B, nch, Nf, Nt_active)
    A0p = np.zeros((B, nch, Nf_act, N_sparse_t), dtype=np.complex128)
    A1p = np.zeros_like(A0p)
    for b in range(N_sparse_t):
        m = bin_idx == b; nf = n_off[m]
        A0p[..., b] = 2.0 * (wA_re[..., m].sum(-1) + 1j * wA_im[..., m].sum(-1))
        A1p[..., b] = 2.0 * ((wA_re[..., m] * nf).sum(-1) + 1j * (wA_im[..., m] * nf).sum(-1))

    # ---- <h|h>: conj + nonconj blocks ----
    if tdi_type == "XYZ":
        Ec = c0_all.conj()[:, :, None] * iC[None] * c0_all[:, None]   # (B, c, c2, Nf, Nt_active)
        En = c0_all[:, :, None] * iC[None] * c0_all[:, None]
        shp = (B, nch, nch, Nf_act, N_sparse_t)
    else:
        Ec = c0_all.conj() * iC[None] * c0_all
        En = c0_all * iC[None] * c0_all
        shp = (B, nch, Nf_act, N_sparse_t)
    B0 = np.zeros(shp, dtype=np.complex128); B1 = np.zeros(shp, dtype=np.complex128)
    B0nc = np.zeros(shp, dtype=np.complex128); B1nc = np.zeros(shp, dtype=np.complex128)
    for b in range(N_sparse_t):
        m = bin_idx == b; nf = n_off[m]
        B0[..., b] = Ec[..., m].sum(-1); B1[..., b] = (Ec[..., m] * nf).sum(-1)
        B0nc[..., b] = En[..., m].sum(-1); B1nc[..., b] = (En[..., m] * nf).sum(-1)
    return A0p, A1p, B0, B1, B0nc, B1nc


class GBSignalHetReferenceSet:
    """Sig-het likelihood with a SET of ``num_data`` references built in sub-batches.

    Args mirror :class:`GBSignalHetComputations`, except ``ref_params_all`` is
    ``(num_data, 9)`` and ``ref_batch_size`` bounds the per-batch dense-``c0`` memory.
    ``reference_impl`` ("dense"|"chunked") selects the per-reference ``c0`` build.
    """

    def __init__(self, data_td, ref_params_all, *, ref_batch_size=4,
                 reference_impl="chunked", **comp_kwargs):
        ref_params_all = np.atleast_2d(np.asarray(ref_params_all, dtype=float))
        self.num_data = int(ref_params_all.shape[0])
        self.reference_impl = str(reference_impl).lower()

        # One template comp gives the SHARED setup (cpp / tdi_wrap / window / sparse_gen
        # / data_complex / invC_complex / geometry); it also builds reference 0.
        self._comp = GBSignalHetComputations(
            data_td, ref_params_all[0], reference_impl=self.reference_impl, **comp_kwargs)
        sh = self._comp._gen_shared
        self.d_d = self._comp.d_d
        g = self._comp._g
        stride, Nt_active = sh["stride"], sh["Nt_active"]
        n_loc = sh["n_sparse_local"]; Nf_active = sh["Nf_active"]
        ind_min_t, ind_min_f = g["ind_min_t"], g["ind_min_f"]

        c0s, A0s, A1s, B0s, B1s, B0ncs, B1ncs = [], [], [], [], [], [], []
        for lo in range(0, self.num_data, int(ref_batch_size)):
            hi = min(lo + int(ref_batch_size), self.num_data)
            # dense c0 per reference in this sub-batch (chunked or dense), stacked
            c0_batch = np.stack([
                reference_c0_dense(
                    self.reference_impl, ref_params_all[i], sh["real_td_cb"],
                    sh["td_set"], sh["wdm_set_complex"], sh["window"], sh["sparse_gen"],
                    ind_min_t, ind_min_f, Nt_active, Nf_active)
                for i in range(lo, hi)], axis=0)                  # (B, nch, Nf_act, Nt_active)
            A0, A1, B0, B1, B0nc, B1nc = _bin_fold_real_batched(
                sh["data_complex"], c0_batch, sh["invC_complex"], n_loc, stride,
                Nt_active, tdi_type="XYZ")
            c0s.append(c0_batch[:, :, :, n_loc]); A0s.append(A0); A1s.append(A1)
            B0s.append(B0); B1s.append(B1); B0ncs.append(B0nc); B1ncs.append(B1nc)

        cat = lambda L: np.ascontiguousarray(np.concatenate(L, axis=0))
        self.c0_sparse_all = cat(c0s)
        self.A0_all = cat(A0s); self.A1_all = cat(A1s)
        self.B0_all = cat(B0s); self.B1_all = cat(B1s)
        self.B0nc_all = cat(B0ncs); self.B1nc_all = cat(B1ncs)
        self.params_ref_all = np.ascontiguousarray(ref_params_all)
        self.window_full = self._comp.window_full
        self.n_sparse_local = self._comp.n_sparse_local

    def get_ll(self, params, data_index):
        """logL for candidates ``params`` ``(N,9)``; ``data_index`` ``(N,)`` picks each
        candidate's reference (0..num_data-1)."""
        x = np.ascontiguousarray(np.atleast_2d(np.asarray(params, dtype=float)))
        N = x.shape[0]
        di = np.asarray(data_index, dtype=np.int32)
        assert di.shape[0] == N and int(di.max()) < self.num_data and int(di.min()) >= 0
        d_h = np.zeros(N); h_h = np.zeros(N); g = self._comp._g
        self._comp.cpp.gb_signal_het_get_ll_in_kernel(
            self._comp.tdi_wrap, d_h, h_h, self.c0_sparse_all,
            self.A0_all, self.A1_all, self.B0_all, self.B1_all,
            self.B0nc_all, self.B1nc_all,
            self.window_full, self.n_sparse_local, x, self.params_ref_all, di,
            N, self.num_data, 9, 1, 2,
            g["Nf"], g["Nt"], g["Nf_active"], g["Nt_active"],
            g["nt_layer"], g["N_sparse_t"], g["stride"],
            g["ind_min_t"], g["ind_min_f"], g["m_half"],
            g["layer_df"], g["dt"], g["Tobs"], g["t0"],
            3, 0, g["n_sparse_fd"], g["tukey_alpha"], g["max_r"], 1,
        0)  # n_cp_sig=0: direct build (script baseline)
        self.last_d_h = np.asarray(d_h).copy(); self.last_h_h = np.asarray(h_h).copy()
        return -0.5 * self.d_d + np.asarray(d_h) - 0.5 * np.asarray(h_h)
