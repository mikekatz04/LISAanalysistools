"""Generic (source-agnostic) signal-heterodyne WDM primitives.

Per the sprint architecture rule (generic polyphase + bin-fold primitives live in
LISAanalysistools; source classes own only their FD-bin producer), these are the
Python-level sig-het helpers shared by the GB (``gbgpu``) and SOBBH (``bbhx``)
signal-het frontends:

* :func:`sparse_time_grid` -- the polyphase sparse-time grid (stride / count /
  bin-centre positions) for a WDM active region. Pure arithmetic, no WDM machinery.
* :func:`bin_fold_real` -- the REAL-projection bin-fold coefficients
  (``A0/A1/B0/B1/B0nc/B1nc``) that the sig-het kernel folds the heterodyne ratio
  ``r = c1/c0`` against to form ``<d|h>`` / ``<h|h>``.

The WDM analysis window itself is NOT re-implemented here -- it is taken directly
from :attr:`lisatools.domains.WDMSettings.window` (``phitilde`` sampled at the full
``Nt`` grid). The reference ``c0`` is produced by the source backend
(``gb_signal_het_make_reference``), not by any Python polyphase.
"""
from __future__ import annotations

import numpy as np


def sparse_time_grid(Nt: int, Nt_active: int, nt_layer: int):
    """Polyphase sparse-time grid for a WDM active region.

    Returns ``(stride, N_sparse_t, n_sparse_local)`` where ``stride = Nt // nt_layer``,
    ``N_sparse_t = Nt_active // stride`` and ``n_sparse_local`` are the bin-centre
    positions ``stride//2 + arange(N_sparse_t) * stride`` (LOCAL active-time indices,
    ``int32``). These are the polyphase iFFT output positions the sig-het kernel and
    the backend reference producer share.
    """
    stride = int(Nt) // int(nt_layer)
    N_sparse_t = int(Nt_active) // stride
    n_sparse_local = (stride // 2 + np.arange(N_sparse_t) * stride).astype(np.int32)
    return stride, N_sparse_t, n_sparse_local


def bin_fold_real(data_complex, c0_complex, invC, n_b_idx_local, stride,
                  Nt_active, tdi_type="XYZ"):
    """REAL-projection sig-het bin-fold coefficients (match the REAL WDM likelihood).

    ``data_complex`` / ``c0_complex`` ``(nch, Nf_active, Nt_active)`` complex WDM
    (data, reference carrier); ``invC`` the WDM inverse-sensitivity. Returns
    ``A0p, A1p`` ``(nch, Nf_active, N_sparse_t)`` (the repacked ``<d|h>`` coeffs the
    kernel forms ``0.5 Re(A0p*r + A1p*dr)`` from) and ``B0, B1, B0nc, B1nc``
    (``(nch, nch, Nf_active, N_sparse_t)`` for ``tdi_type="XYZ"``; ``(nch, ...)``
    otherwise) -- the conj + nonconj ``<h|h>`` blocks that give the real projection
    ``0.5 Re(B0 conj(rc)rc2 + B0nc rc rc2 + B1/B1nc dr terms)``.

    Moved verbatim (numerics-identical) from the sig-het dev prototype's
    ``python_bin_fold_real``; the proof of the real-projection identity (1e-13) lives
    in the dev ``gb_sighet_realproj_proto.py``.
    """
    nch, Nf_act, _ = c0_complex.shape
    N_sparse_t = len(n_b_idx_local)
    bin_edges = np.arange(N_sparse_t + 1) * stride
    bin_edges[-1] = Nt_active
    bin_idx = np.repeat(np.arange(N_sparse_t), np.diff(bin_edges).astype(int))
    assert bin_idx.shape[0] == Nt_active
    n_off = (np.arange(Nt_active) - np.asarray(n_b_idx_local)[bin_idx]).astype(float)

    Re_d = np.real(data_complex)
    iC = np.real(invC)
    u = np.real(c0_complex)
    w = np.imag(c0_complex)

    # ---- <d|h> repack: A0re/A0im integrands packed into one complex ----
    if tdi_type == "XYZ":
        Dre = np.einsum("cmn,cdmn->dmn", Re_d, iC)
    else:
        Dre = Re_d * iC
    wA_re = Dre * u
    wA_im = Dre * w
    A0p = np.zeros((nch, Nf_act, N_sparse_t), dtype=np.complex128)
    A1p = np.zeros((nch, Nf_act, N_sparse_t), dtype=np.complex128)
    for b in range(N_sparse_t):
        m = bin_idx == b
        nf = n_off[m]
        A0p[:, :, b] = 2.0 * (wA_re[:, :, m].sum(-1) + 1j * wA_im[:, :, m].sum(-1))
        A1p[:, :, b] = 2.0 * ((wA_re[:, :, m] * nf).sum(-1) + 1j * (wA_im[:, :, m] * nf).sum(-1))

    # ---- <h|h>: conj + nonconj blocks (real invC) ----
    if tdi_type == "XYZ":
        Ec = c0_complex.conj()[:, None] * iC * c0_complex[None, :]
        En = c0_complex[:, None] * iC * c0_complex[None, :]
        shp = (nch, nch, Nf_act, N_sparse_t)
    else:
        Ec = c0_complex.conj() * iC * c0_complex
        En = c0_complex * iC * c0_complex
        shp = (nch, Nf_act, N_sparse_t)
    B0 = np.zeros(shp, dtype=np.complex128)
    B1 = np.zeros(shp, dtype=np.complex128)
    B0nc = np.zeros(shp, dtype=np.complex128)
    B1nc = np.zeros(shp, dtype=np.complex128)
    for b in range(N_sparse_t):
        m = bin_idx == b
        nf = n_off[m]
        B0[..., b] = Ec[..., m].sum(-1)
        B1[..., b] = (Ec[..., m] * nf).sum(-1)
        B0nc[..., b] = En[..., m].sum(-1)
        B1nc[..., b] = (En[..., m] * nf).sum(-1)
    return A0p, A1p, B0, B1, B0nc, B1nc
