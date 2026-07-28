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

    ``data_complex`` / ``c0_complex`` ``(..., nch, Nf_active, Nt_active)`` complex WDM
    (data, reference carrier); ``invC`` the WDM inverse-sensitivity. Returns
    ``A0p, A1p`` ``(..., nch, Nf_active, N_sparse_t)`` (the repacked ``<d|h>`` coeffs the
    kernel forms ``0.5 Re(A0p*r + A1p*dr)`` from) and ``B0, B1, B0nc, B1nc``
    (``(..., nch, nch, Nf_active, N_sparse_t)`` for ``tdi_type="XYZ"``; ``(..., nch, ...)``
    otherwise) -- the conj + nonconj ``<h|h>`` blocks that give the real projection
    ``0.5 Re(B0 conj(rc)rc2 + B0nc rc rc2 + B1/B1nc dr terms)``.

    Any number of LEADING BATCH AXES is supported and broadcast between the
    three inputs, so a whole in-model block folds in ONE call (per-source
    residual + invC + reference) instead of one call per source, and a shared
    ``data_complex``/``invC`` may be folded against a stack of references.

    Ported from the sig-het dev prototype's ``python_bin_fold_real``; the proof of
    the real-projection identity (1e-13) lives in the dev
    ``gb_sighet_realproj_proto.py``. Term-for-term identical to that prototype
    EXCEPT for summation order -- see the segment-matrix note below.

    xp-generic: the array module follows ``data_complex`` (numpy or cupy) so
    the GPU sig-het setup can fold device-resident residual slabs without a
    host round-trip.
    """
    try:
        from .utils.utility import get_array_module
        xp = get_array_module(data_complex)
    except (ImportError, ValueError):
        xp = np

    N_sparse_t = len(n_b_idx_local)
    bin_edges = np.arange(N_sparse_t + 1) * stride
    bin_edges[-1] = Nt_active
    bin_idx = np.repeat(np.arange(N_sparse_t), np.diff(bin_edges).astype(int))
    assert bin_idx.shape[0] == Nt_active
    n_b_local_np = np.asarray(
        n_b_idx_local.get() if hasattr(n_b_idx_local, "get") else n_b_idx_local
    )
    n_off_np = (np.arange(Nt_active) - n_b_local_np[bin_idx]).astype(float)

    # Segment-sum matrices. Every per-bin reduction below is a sum over a
    # CONTIGUOUS run of the time axis (bin b spans [bin_edges[b],
    # bin_edges[b+1])), so the whole bin loop collapses into two
    # (Nt_active, N_sparse_t) matmuls: ``S`` for the plain sum and ``Sw`` for
    # the n_off-weighted sum. That replaces 2*N_sparse_t boolean-masked
    # reductions -- each a separate kernel launch allocating a gather -- with
    # 6 batched GEMMs, and it broadcasts over leading batch axes for free.
    # Built on the host and uploaded once (tiny; independent of the batch).
    #
    # NOTE: summation ORDER changes vs the old per-bin ``.sum(-1)``, so
    # results move at FP epsilon (~1e-16 relative). That is far inside the
    # 1e-10/1e-12 sig-het parity budgets and the ~1e-15 C++-mirror check; the
    # C++ kernel still folds bin-by-bin and is unchanged.
    S_np = np.zeros((Nt_active, N_sparse_t), dtype=np.float64)
    S_np[np.arange(Nt_active), bin_idx] = 1.0
    S = xp.asarray(S_np)
    Sw = xp.asarray(S_np * n_off_np[:, None])

    c0_complex = xp.asarray(c0_complex)
    invC = xp.asarray(invC)
    Re_d = xp.real(data_complex)
    iC = xp.real(invC)
    u = xp.real(c0_complex)
    w = xp.imag(c0_complex)

    # ---- <d|h> repack: A0re/A0im integrands packed into one complex ----
    if tdi_type == "XYZ":
        Dre = xp.einsum("...cmn,...cdmn->...dmn", Re_d, iC)
    else:
        Dre = Re_d * iC
    wA_re = Dre * u
    wA_im = Dre * w
    A0p = 2.0 * ((wA_re @ S) + 1j * (wA_im @ S))
    A1p = 2.0 * ((wA_re @ Sw) + 1j * (wA_im @ Sw))

    # ---- <h|h>: conj + nonconj blocks (real invC) ----
    if tdi_type == "XYZ":
        Ec = c0_complex.conj()[..., :, None, :, :] * iC * c0_complex[..., None, :, :, :]
        En = c0_complex[..., :, None, :, :] * iC * c0_complex[..., None, :, :, :]
    else:
        Ec = c0_complex.conj() * iC * c0_complex
        En = c0_complex * iC * c0_complex
    B0 = Ec @ S
    B1 = Ec @ Sw
    B0nc = En @ S
    B1nc = En @ Sw
    return A0p, A1p, B0, B1, B0nc, B1nc
