#!/usr/bin/env python
"""
Signal-based relative-binning heterodyne in the complex/quadrature
WDM domain for galactic-binary sources.

Mirrors BBHx's :class:`bbhx.likelihood.HeterodynedLikelihood`
(``BBHx/src/bbhx/likelihood.py:231-705``) but operates on the WDM
time-frequency plane instead of the dense FD bin grid.

Key idea
--------

Pick a reference ``x0`` close in parameter space to candidate ``x1``.
Build both signals' complex/quadrature WDM coefficients

.. math::

    c[c, m, n]  =  h_{c,\\mathrm{real\\,WDM}}[m,n]
                  + i\\, h_{c,\\mathrm{Hilbert\\,quadrature\\,WDM}}[m,n]

via ``TDSignal.transform(WDMSettings(is_complex=True))``. The
imaginary part is the Hilbert-pair companion of the standard real
WDM coefficient -- same information, but with the Wilson ``(m+n)``
parity sign-alternation rotated out. The per-pixel ratio

.. math::

    r_{c}[m, n]  =  c_{1,c}[m, n]  /  c_{0,c}[m, n]

is therefore smooth in BOTH ``m`` and ``n`` for parameter-close
``x0``, ``x1`` (where standard real-WDM ``w1[m,n]/w0[m,n]`` is
jagged from the parity flip).

Inner product
-------------

The lisatools complex-WDM inner product (``is_complex=True``,
differential_component = 0.125) recovers the standard real-WDM
inner product to FP precision:

.. math::

    <a|b>_\\mathrm{IP}  =  \\tfrac12 \\,\\mathrm{Re}
                            \\sum_{c,c',m,n}
                            a_c^\\ast[m,n]\\, \\Sigma^{-1}_{cc'}[m,n]\\, b_{c'}[m,n].

Linearise ``r`` within sparse time-bins ``b`` (per channel, per m-layer):

.. math::

    r_c[m, n]  \\approx  r_c[m, b]  +  (dr_c/dn)[m, b]\\, (n - n_b),

and the heterodyne identity gives, with bin-summed coefficients

.. math::

    A0_{c'}[m, b] &= \\sum_{n \\in b} D_{c'}[m,n]\\, c_{0,c'}[m,n], \\\\
    A1_{c'}[m, b] &= \\sum_{n \\in b} D_{c'}[m,n]\\, c_{0,c'}[m,n]\\, (n - n_b), \\\\
    D_{c'}[m, n]  &= \\sum_c d_c^\\ast[m,n]\\, \\Sigma^{-1}_{cc'}[m,n],

the candidate inner product

.. math::

    <d|h_1>  \\approx  \\tfrac12 \\,\\mathrm{Re}
                       \\sum_{m, b, c'}
                       \\Big(A0_{c'}[m,b]\\, r_{c'}[m,b]
                            + A1_{c'}[m,b]\\, (dr_{c'}/dn)[m,b]\\Big).

Analogous ``B0``, ``B1`` per ``(c, c')`` pair for ``<h_1|h_1>``.

Storage
-------

Coefficients are per ``(channel, m-layer, sparse-bin)`` -- NOT
summed over ``m``. For GBs the source band is narrow in ``m``, but
the per-m form costs only ``nch * Nf_active * N_sparse_t`` complex
per coefficient and lets ``r`` carry its m-dependence (which it
does: Delta f0 shifts the layer support, Delta fdot tilts it in
time, etc.). With ``nch=3``, ``Nf_active=1020``, ``N_sparse_t=160``
that is ~3.9 MB per A0 / A1, and ~12 MB for B0 / B1 with the
``(c, c')`` outer product. Negligible.

The v1 (correctness-first) flow regenerates the dense
``c1[c, m, n]`` complex WDM template at every ``get_ll`` call via
the caller-supplied ``gb_complex_wdm_gen`` callable, then sub-samples
``c1[:, :, n_b_idx]`` at the sparse-bin centres. The runtime cost is
dominated by that dense regeneration; the heterodyne speedup itself
appears only when we wire in a sparse-time-only complex-WDM generator,
which is the explicit v2 follow-up.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np

from lisatools.response.parallelbase import FastLISAResponseParallelModule


class GBSignalHetWDMGetLL(FastLISAResponseParallelModule):
    """Signal-based relative-binning heterodyne for GBs (complex WDM).

    Parameters
    ----------
    wdm_set_complex
        ``WDMSettings`` with ``is_complex=True``. Used to interpret
        the complex-WDM data, c0, c1, and the inner product (the
        0.125 differential component).
    data_wdm_complex
        ``(nch, Nf_act, Nt_active)`` complex array (or ``WDMSignal``
        with ``.arr`` of that shape). The data already transformed
        into the complex/quadrature WDM domain.
    invC
        Cross-channel inverse sensitivity:
        ``(nch, nch, Nf_act, Nt_active)`` real for XYZ, or
        ``(nch, Nf_act, Nt_active)`` real diagonal for AE / AET.
        Comes from
        ``XYZ2SensitivityMatrix(wdm_set_complex, ...).invC`` (XYZ)
        or ``1 / Sigma_diag`` (AE / AET).
    c0_dense_complex
        ``(nch, Nf_act, Nt_active)`` complex -- the reference signal
        already transformed into the complex/quadrature WDM domain
        via the same ``TDSignal.transform`` call.
    gb_complex_wdm_gen
        Callable ``gb_complex_wdm_gen(params_9) -> (nch, Nf_act,
        Nt_active) complex array``. Builds the candidate signal's
        complex/quadrature WDM template. v1 expects this to do the
        full dense TD -> complex-WDM transform per call.
    reference_params
        ``(9,)`` GB params ``x0``.
    N_sparse_t
        Number of sparse time bins. Must divide ``Nt_active`` evenly.
        Default ``Nt_active // 16``.
    tdi_type
        ``"XYZ"`` (cross-channel ``invC``) or ``"AE"`` / ``"AET"``
        (diagonal).
    force_backend
        ``"cpu"`` (v1 is CPU-only; cuda / jax come later).
    """

    @classmethod
    def supported_backends(cls):
        return ["cpu"]

    def __init__(
        self,
        wdm_set_complex,
        data_wdm_complex,
        invC,
        c0_dense_complex,
        gb_complex_wdm_gen: Callable,
        reference_params,
        N_sparse_t: Optional[int] = None,
        tdi_type: str = "XYZ",
        force_backend: Optional[str] = None,
    ):
        super().__init__(force_backend=force_backend)
        xp = self.backend.xp

        if not getattr(wdm_set_complex, "is_complex", False):
            raise ValueError(
                "wdm_set_complex must have is_complex=True; "
                "the signal-het machinery requires complex/quadrature WDM."
            )
        if tdi_type not in {"XYZ", "AE", "AET"}:
            raise ValueError("tdi_type must be 'XYZ', 'AE', or 'AET'.")

        self.wdm_set = wdm_set_complex
        self.tdi_type = tdi_type
        self.gb_complex_wdm_gen = gb_complex_wdm_gen
        self.reference_params = xp.asarray(
            reference_params, dtype=float
        ).reshape(9)

        # --- data (complex WDM) ------------------------------------------
        data_arr = getattr(data_wdm_complex, "arr", data_wdm_complex)
        self.data = xp.asarray(data_arr).astype(xp.complex128, copy=False)
        if self.data.ndim != 3:
            raise ValueError(
                f"data_wdm_complex must be (nch, Nf_act, Nt_active); "
                f"got shape {self.data.shape}"
            )
        nch, Nf_act, Nt_active = self.data.shape
        self.nch = nch
        self.Nf_act = Nf_act
        self.Nt_active = Nt_active

        # --- c0 (reference signal, complex WDM) --------------------------
        c0_arr = getattr(c0_dense_complex, "arr", c0_dense_complex)
        self.c0 = xp.asarray(c0_arr).astype(xp.complex128, copy=False)
        if self.c0.shape != (nch, Nf_act, Nt_active):
            raise ValueError(
                f"c0_dense_complex shape {self.c0.shape} != data shape "
                f"({nch}, {Nf_act}, {Nt_active})"
            )

        # --- invC ---------------------------------------------------------
        invC_arr = xp.asarray(invC)
        if tdi_type == "XYZ":
            if invC_arr.shape != (nch, nch, Nf_act, Nt_active):
                raise ValueError(
                    f"XYZ invC must be ({nch}, {nch}, {Nf_act}, {Nt_active}); "
                    f"got {invC_arr.shape}"
                )
        else:
            if invC_arr.shape != (nch, Nf_act, Nt_active):
                raise ValueError(
                    f"{tdi_type} invC must be ({nch}, {Nf_act}, {Nt_active}); "
                    f"got {invC_arr.shape}"
                )
        self.invC = invC_arr

        # --- sparse time bins --------------------------------------------
        # Allow ANY N_sparse_t up to Nt_active. We partition [0, Nt_active)
        # into N_sparse_t roughly-equal contiguous bins via np.linspace,
        # so individual bin widths differ by at most 1. Bin centres are
        # the integer middle pixel of each bin.
        if N_sparse_t is None:
            N_sparse_t = max(Nt_active // 16, 1)
        if N_sparse_t < 1 or N_sparse_t > Nt_active:
            raise ValueError(
                f"N_sparse_t={N_sparse_t} must satisfy 1 <= N <= Nt_active="
                f"{Nt_active}"
            )
        self.N_sparse_t = int(N_sparse_t)
        bin_edges = np.linspace(0, Nt_active, self.N_sparse_t + 1).astype(int)
        self._bin_edges = xp.asarray(bin_edges)            # (N_sparse_t + 1,)
        bin_sizes = np.diff(bin_edges)                     # (N_sparse_t,)
        self._bin_sizes = xp.asarray(bin_sizes, dtype=float)
        # Bin centre pixel (integer): floor((start + end - 1) / 2).
        n_b_idx_np = ((bin_edges[:-1] + bin_edges[1:] - 1) // 2).astype(int)
        self.n_b_idx_local = xp.asarray(n_b_idx_np)
        # Per-pixel offset (n - n_b[bin_idx[n]]) and per-pixel bin index.
        bin_idx_np = np.repeat(np.arange(self.N_sparse_t), bin_sizes)
        assert bin_idx_np.shape[0] == Nt_active
        self._bin_idx = xp.asarray(bin_idx_np)             # (Nt_active,)
        n_off_per_pixel = (
            np.arange(Nt_active) - n_b_idx_np[bin_idx_np]
        ).astype(float)
        self._n_off_per_pixel = xp.asarray(n_off_per_pixel)  # (Nt_active,)
        # Average bin width (for forming a "per-pixel" dr/dn at runtime).
        self._mean_Dn = float(np.mean(bin_sizes))

        # --- cache c0 at sparse bin centres (per channel AND per m) ------
        # Shape (nch, Nf_act, N_sparse_t) complex. This is the denominator
        # for r = c1_sparse / c0_sparse at eval time.
        self.c0_sparse = self.c0[:, :, self.n_b_idx_local]
        # |c0_sparse|-based mask: outside the source band c0 ~ 0, so r is
        # undefined there but A0 / B0 are also ~ 0; we just set r := 0
        # there at eval to avoid NaNs.
        c0_mag = xp.abs(self.c0_sparse)
        # Per (channel, m) floor relative to that layer's peak |c0|.
        floor = 1e-12 * c0_mag.max(axis=-1, keepdims=True)
        floor = xp.maximum(floor, 1e-300)
        self._c0_mask = c0_mag > floor

        # --- build coefficients (per-m, bin-summed) ----------------------
        self._build_coefficients()

        # --- reference scalars -------------------------------------------
        d_d_raw = self._ip_complex_sum(self.data, self.data)
        h0_h0_raw = self._ip_complex_sum(self.c0, self.c0)
        d_h0_raw = self._ip_complex_sum(self.data, self.c0)
        self.ref_d_d = 0.5 * float(xp.real(d_d_raw))
        self.ref_h0_h0 = 0.5 * float(xp.real(h0_h0_raw))
        self.ref_d_h0 = 0.5 * float(xp.real(d_h0_raw))
        self.ref_ll = (
            self.ref_d_h0 - 0.5 * self.ref_h0_h0 - 0.5 * self.ref_d_d
        )

        # diagnostics filled by get_ll
        self.d_h_out = None
        self.h_h_out = None
        self.r_out = None

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _ip_complex_sum(self, sig1, sig2):
        """Raw complex-WDM IP sum ``Sigma sig1.conj() * sig2 * invC``.

        Caller multiplies the real part by 0.5 to recover the scalar
        real-WDM inner product (matches lisatools'
        ``inner_product`` with differential_component=0.125 and the
        canonical ``4 * dc`` prefactor).
        """
        xp = self.backend.xp
        if self.tdi_type == "XYZ":
            tmp = xp.einsum("cmn,cdmn->dmn", sig1.conj(), self.invC)
            return (tmp * sig2).sum()
        return (sig1.conj() * sig2 * self.invC).sum()

    def _bin_fold(self, w):
        """Reduce a per-pixel weight along the last axis into N_sparse_t bins.

        Supports non-uniform bin widths via the cached ``_bin_idx`` and
        ``_n_off_per_pixel`` arrays. Returns the pair (S0, S1) =
        (sum, (n-n_b)-weighted sum) per bin.
        """
        xp = self.backend.xp
        # leading shape (without the time axis)
        lead = w.shape[:-1]
        Nb = self.N_sparse_t
        bin_idx = self._bin_idx
        n_off = self._n_off_per_pixel

        # Bring time axis last (already last), flatten leading axes to one.
        w_flat = w.reshape(-1, w.shape[-1])               # (L, Nt_active)
        L = w_flat.shape[0]
        S0 = xp.zeros((L, Nb), dtype=w_flat.dtype)
        S1 = xp.zeros((L, Nb), dtype=w_flat.dtype)
        # np.add.at handles repeated indices correctly.
        # For each row j of w_flat, accumulate w_flat[j, n] into S0[j, bin_idx[n]].
        # Implementing via expanded indexing:
        # We use the (row_index, bin_index) tuple via take/index_add.
        # Simplest portable form: loop over bins.
        for b in range(Nb):
            mask = bin_idx == b
            S0[:, b] = w_flat[:, mask].sum(axis=-1)
            S1[:, b] = (w_flat[:, mask] * n_off[mask]).sum(axis=-1)
        return S0.reshape(*lead, Nb), S1.reshape(*lead, Nb)

    def _build_coefficients(self):
        """Bin-fold the dense per-pixel weights into per-(m, b) coefficients.

        Shapes after this method:
          A0       (nch, Nf_act, N_sparse_t)        complex
          A1       (nch, Nf_act, N_sparse_t)        complex
          B0       (nch, nch, Nf_act, N_sparse_t)   complex   (XYZ)
                or (nch, Nf_act, N_sparse_t) complex          (diag)
          B1       same as B0
        """
        xp = self.backend.xp
        c0 = self.c0
        d = self.data

        # D[c', m, n] = (sum_c d_c.conj() * invC[c, c', m, n])  (XYZ)
        #             = d.conj() * invC                         (diag)
        if self.tdi_type == "XYZ":
            D = xp.einsum("cmn,cdmn->dmn", d.conj(), self.invC)
        else:
            D = d.conj() * self.invC

        # A side
        w_A = D * c0                                       # (nch, Nf_act, Nt_active)
        self.A0, self.A1 = self._bin_fold(w_A)

        # B side
        if self.tdi_type == "XYZ":
            E = c0.conj()[:, None, :, :] * c0[None, :, :, :] * self.invC
            self.B0, self.B1 = self._bin_fold(E)
        else:
            E_diag = (c0.conj() * c0 * self.invC).astype(xp.complex128, copy=False)
            self.B0, self.B1 = self._bin_fold(E_diag)

    # ------------------------------------------------------------------
    # ratio + derivative at sparse bin centres
    # ------------------------------------------------------------------
    def _r_and_dr(self, params_x1) -> Tuple[np.ndarray, np.ndarray]:
        """r[c, m, b] and dr/dn[c, m, b] at sparse-bin centres.

        v1: regenerate the full dense ``c1`` complex WDM template and
        sub-sample at the sparse bin centres. v2 will plug in a
        sparse-time-only generator that avoids the dense transform.
        """
        xp = self.backend.xp
        c1_dense = xp.asarray(
            self.gb_complex_wdm_gen(np.asarray(params_x1, dtype=float))
        ).astype(xp.complex128, copy=False)
        if c1_dense.shape != (self.nch, self.Nf_act, self.Nt_active):
            raise ValueError(
                f"gb_complex_wdm_gen returned shape {c1_dense.shape}; "
                f"expected ({self.nch}, {self.Nf_act}, {self.Nt_active})"
            )
        c1_sparse = c1_dense[:, :, self.n_b_idx_local]     # (nch, Nf_act, Nb)
        # safe divide
        denom = xp.where(self._c0_mask, self.c0_sparse, 1.0)
        r = xp.where(self._c0_mask, c1_sparse / denom, 0.0 + 0.0j)
        # centred FD across b in pixel units. Use the mean bin width to
        # convert "per-sparse-bin" slope into "per-dense-pixel" slope --
        # the (n - n_b) weighting in A1, B1 is in dense pixel units.
        Dn = self._mean_Dn
        dr = xp.zeros_like(r)
        Nb = r.shape[-1]
        if Nb >= 3:
            dr[..., 1:-1] = (r[..., 2:] - r[..., :-2]) / (2.0 * Dn)
            dr[..., 0] = (r[..., 1] - r[..., 0]) / Dn
            dr[..., -1] = (r[..., -1] - r[..., -2]) / Dn
        elif Nb == 2:
            dr[..., 0] = (r[..., 1] - r[..., 0]) / Dn
            dr[..., 1] = dr[..., 0]
        return r, dr

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def get_ll(self, params_x1) -> float:
        """Heterodyne log-likelihood for candidate parameters.

        Returns
        -------
        float
            ``ll = <d|h1> - 0.5 <h1|h1> - 0.5 <d|d>``, with all three
            terms matching the corresponding lisatools real-WDM inner
            products to within the relative-binning approximation
            error (small for x1 close to x0).
        """
        xp = self.backend.xp
        r, dr = self._r_and_dr(params_x1)

        # <d|h1>: 0.5 * Re sum_{m, b, c'} (A0 r + A1 dr)
        d_h_raw = (self.A0 * r + self.A1 * dr).sum()
        d_h = 0.5 * float(xp.real(d_h_raw))

        # <h1|h1>
        if self.tdi_type == "XYZ":
            r_outer = r.conj()[:, None, :, :] * r[None, :, :, :]
            cross_drr = (
                r.conj()[:, None, :, :] * dr[None, :, :, :]
                + dr.conj()[:, None, :, :] * r[None, :, :, :]
            )
            h_h_raw = (self.B0 * r_outer + self.B1 * cross_drr).sum()
        else:
            rsq = (r.conj() * r).real.astype(xp.complex128, copy=False)
            cross_drr = r.conj() * dr + dr.conj() * r
            h_h_raw = (self.B0 * rsq + self.B1 * cross_drr).sum()
        h_h = 0.5 * float(xp.real(h_h_raw))

        ll = d_h - 0.5 * h_h - 0.5 * self.ref_d_d

        self.d_h_out = d_h
        self.h_h_out = h_h
        self.r_out = (r, dr)
        return ll

    def get_d_h_and_h_h(self, params_x1) -> Tuple[float, float]:
        """Return ``(<d|h1>, <h1|h1>)`` -- handy for validation prints."""
        _ = self.get_ll(params_x1)
        return self.d_h_out, self.h_h_out
