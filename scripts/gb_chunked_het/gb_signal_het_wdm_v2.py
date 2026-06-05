#!/usr/bin/env python
"""GB signal-heterodyned waveform v2 -- minimal per-layer iFFT generator.

Same heterodyne machinery as :class:`gb_signal_het_wdm.GBSignalHetWDMGetLL`
(v1): complex/quadrature WDM, bin-folded coefficients A0/A1/B0/B1,
candidate-to-reference ratio ``r(t) = c1(t)/c0(t)`` linearised at sparse
bin centres, ``<d|h>``/``<h|h>`` from cheap einsums.

What changes
------------

v1 regenerated the FULL dense complex WDM template per ``get_ll`` call
via ``TDSignal.transform(WDMSettings(is_complex=True))`` and sub-sampled
at the sparse bin centres -- the ~Nt_active-long iFFT dominated wall
time. v2 computes the sparse complex coefficients DIRECTLY: per active
m-layer, gather a window of FD bins around the layer center, multiply
by the resampled :math:`\\tilde\\phi` window, run a minimal-length iFFT
of size ``Nt_layer``, and read off only the sparse pixels.

Active band defaults to ``m_floor +/- 2`` around the candidate's
carrier ``f0`` (5 layers); ``m_active_half_width`` adjusts the half-
width. The minimal ``Nt_layer`` is auto-derived once at construction by
sweeping ``[16, 32, 64, ...]`` until v2's sparse output matches the
lisatools dense complex WDM reference to a configurable per-pixel
reldiff target (default 1e-12).

The sparse n grid is forced to be aligned with the iFFT output: stride
``Nt // Nt_layer`` in the underlying ``Nt`` grid, with bin centres at
offset ``stride/2`` from each bin edge. v1's ``_build_coefficients`` /
``_bin_fold`` / ``_r_and_dr`` paths are reused with this explicit grid.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from scipy import special

from lisatools.response.parallelbase import FastLISAResponseParallelModule

from gb_signal_het_wdm import GBSignalHetWDMGetLL


# ---------------------------------------------------------------------------
# WDM analysis window (resampled phitilde -- matches lisatools).
# ---------------------------------------------------------------------------


def _phitilde(omega: np.ndarray, dOmega: float, A: Optional[float] = None,
              p: int = 4) -> np.ndarray:
    """WDM analysis window. Matches ``WDMSettings.phitilde`` in lisatools.

    See ``LISAanalysistools/src/lisatools/domains.py:1929-1948``.
    """
    if A is None:
        A = dOmega / 4.0
    B = dOmega - 2.0 * A
    abs_om = np.abs(omega)
    z = np.zeros_like(omega, dtype=float)
    inner = abs_om < A
    z[inner] = 1.0 / np.sqrt(dOmega)
    band = (abs_om >= A) & (abs_om < A + B)
    if np.any(band):
        x = np.clip((abs_om[band] - A) / B, 0.0, 1.0)
        y = special.betainc(p, p, x)
        z[band] = 1.0 / np.sqrt(dOmega) * np.cos(y * np.pi / 2.0)
    return z


def _build_full_window(Nf: int, Nt: int) -> np.ndarray:
    """Build the WDM analysis window at the full Nt-length FD-bin grid.

    The window phitilde has support ``|omega| < pi/Nf``, which corresponds to
    ``|j| < Nt/2`` in FD-bin units (one bin = ``2*pi/N`` in omega). So the
    natural window length is exactly ``Nt`` regardless of how many output
    pixels we want.

    Returns shape ``(Nt,)`` float -- ``phitilde`` sampled at FD-bin offsets
    ``j = -Nt/2, -Nt/2+1, ..., Nt/2-1`` around the layer centre.
    """
    N = Nf * Nt
    j_arr = np.arange(Nt) - Nt // 2
    omega = 2.0 * np.pi / float(N) * j_arr.astype(float)
    dOmega_s = np.pi / float(Nf)
    return _phitilde(omega, dOmega_s)


# ---------------------------------------------------------------------------
# Per-layer minimal-iFFT kernel.
# ---------------------------------------------------------------------------


def _compute_sparse_complex_wdm(
    fd_rfft: np.ndarray,        # (nch, N//2 + 1) complex -- rfft(td) (NO dt scaling)
    m_layers: np.ndarray,       # (Nm,) int -- global WDM layer indices
    Nf: int,
    Nt: int,
    Nt_layer: int,
    window_full: np.ndarray,    # (Nt,) float -- phitilde at full Nt grid
    n_global_grid: np.ndarray,  # (N_sparse,) int -- positions in Nt grid
    stride: int,                # Nt // Nt_layer (Nt_layer == Nt // stride)
) -> np.ndarray:
    """Return ``c[c, m_layer, n_sparse]`` complex WDM coefficients via polyphase.

    For each active m-layer:

      1. Gather Nt FD bins around layer centre (Hermitian-wrapped from rfft).
      2. Multiply by the full Nt-length window + per-bin pre-phase.
      3. Polyphase fold: reshape (Nt) -> (stride, Nt_layer), sum over stride.
      4. iFFT of length Nt_layer -- yields outputs at strided positions
         ``n_global = n_start + n_layer * stride`` exactly.
      5. Apply lisatools' WDM coefficient layout (kappa, conj(Cmn), sign).

    No truncation: the polyphase identity is exact for any stride dividing Nt.
    The output matches lisatools' ``TDSignal.transform(WDMSettings(is_complex=
    True))[c, m, n_global]`` to FP precision (modulo the missing ``sqrt(dt)``
    factor that the caller scales in).

    ``n_global_grid`` MUST be ``n_start + arange(N_sparse) * stride`` with
    ``N_sparse <= Nt_layer``.
    """
    N = Nf * Nt
    nch = fd_rfft.shape[0]
    Nm = int(len(m_layers))
    N_sparse = int(len(n_global_grid))
    if Nt_layer * stride != Nt:
        raise ValueError(
            f"Nt_layer * stride ({Nt_layer}*{stride}) must equal Nt ({Nt})"
        )
    if N_sparse > Nt_layer:
        raise ValueError(f"N_sparse={N_sparse} > Nt_layer={Nt_layer}")

    n_start = int(n_global_grid[0])
    half_Nt = Nt // 2
    j_arr = np.arange(Nt) - half_Nt            # (Nt,) centered offsets for FD gather
    # Prephase uses the NATURAL index i = 0..Nt-1 (not centered j) to match the
    # untransformed input to lisatools' iFFT (which is np.fft.ifft of an array
    # whose element 0 is the leftmost windowed FD bin, not the layer centre).
    i_arr = np.arange(Nt).astype(float)
    prephase = np.exp(1j * 2.0 * np.pi * i_arr * float(n_start) / float(Nt))
    win_pp = window_full.astype(np.complex128) * prephase     # (Nt,)

    # kappa_no_dt -- caller adds the missing sqrt(data_dt).
    kappa = 2.0 * np.sqrt(np.pi) / float(Nf)

    out = np.zeros((nch, Nm, N_sparse), dtype=np.complex128)

    # The polyphase identity:
    #   lisatools_after_ifft[m, n_g] = (1/Nt) * sum_j X[j] exp(+i 2*pi*j*n_g/Nt)
    # for n_g = n_start + n_layer * stride and X = (windowed * prephase) FD.
    # We can rewrite as (1/stride) * IFFT_Nt_layer(Y)[n_layer] where
    # Y[r] = sum_q X[q*Nt_layer + r] is the polyphase-folded length-Nt_layer
    # sequence. Plus the standard (-1)^n_g from the FD-centring shift.

    # Polyphase identity (no fftshift in input; no (-1)^n factor):
    #   lisatools_after_ifft[n_g] = (1/stride) * ifft_Nt_layer(Y)[n_layer]
    # where Y[r] = sum_q (windowed_FD * prephase)[q*Nt_layer + r].
    n_g = n_global_grid                                   # (N_sparse,)
    sign_scale = np.full(len(n_g), 1.0 / float(stride))   # (N_sparse,)

    for im in range(Nm):
        m_int = int(m_layers[im])
        k_arr = m_int * half_Nt + j_arr                  # (Nt,) FD bin indices
        # Hermitian wraparound (real TD -> rfft)
        k_use = k_arr.copy()
        neg = k_arr < 0
        over = k_arr > (N // 2)
        k_use[neg] = -k_arr[neg]
        k_use[over] = N - k_arr[over]
        conj_mask = neg | over
        in_range = (k_use >= 0) & (k_use <= (N // 2))

        h_vals = np.zeros((nch, Nt), dtype=np.complex128)
        if np.any(in_range):
            h_vals[:, in_range] = fd_rfft[:, k_use[in_range]]
        if np.any(conj_mask):
            h_vals[:, conj_mask] = np.conj(h_vals[:, conj_mask])

        weighted = h_vals * win_pp[None, :]              # (nch, Nt)

        # Polyphase fold: (nch, Nt) -> (nch, stride, Nt_layer) -> sum over stride.
        folded = weighted.reshape(nch, stride, Nt_layer).sum(axis=1)

        # IFFT of length Nt_layer
        ifft_out = np.fft.ifft(folded, axis=-1)          # (nch, Nt_layer)

        ifft_sub = ifft_out[:, :N_sparse] * sign_scale[None, :]

        # Apply lisatools wdm coefficient layout.
        m_plus_n = (m_int + n_g) & 1                     # (N_sparse,) 0/1
        conj_cmn = np.where(m_plus_n == 0, 1.0 + 0.0j, 0.0 - 1.0j)
        sign_mn = (-1.0) ** ((m_int + 1) * n_g)
        coef = kappa * sign_mn * conj_cmn                # (N_sparse,) complex

        out[:, im, :] = ifft_sub * coef[None, :]

    return out


# ---------------------------------------------------------------------------
# Sparse generator class (calibrates Nt_layer + holds reused scaffolding).
# ---------------------------------------------------------------------------


class GBSparseComplexWDMGen:
    """Generate sparse complex WDM coefs per active m-layer via minimal iFFT.

    Parameters
    ----------
    real_td_callable
        ``params9 -> (nch, N) real`` TD synthesis. In a production setup
        this is replaced by a fast FD-direct GB generator -- for the
        prototype any callable returning the TD signal works; per-call
        cost is then dominated by TD synth + rfft, but the polyphase
        WDM math is what v2 actually contributes, and it is tiny.
    wdm_set_complex
        ``WDMSettings`` with ``is_complex=True``.
    data_dt
        Underlying TD sample step in seconds (same as the WDMSettings).
    ind_min_t, Nt_active
        Local active n range in the Nt grid.
    m_active_half_width
        Active m-band half width; the per-call active m-layers are
        ``m_floor +/- m_active_half_width`` (default 2 -> 5 layers).
    Nt_layer
        Polyphase iFFT length per active m-layer = number of sparse n
        outputs per layer. Must divide ``Nt`` and be even. Smaller
        Nt_layer -> wider strides in time -> coarser sparse-n grid.
        The bound is the heterodyne linearisation: ``stride = Nt //
        Nt_layer`` must be small enough that ``r(t) = c1/c0`` is well
        approximated by a line over ``stride`` dense pixels (for typical
        GB MCMC proposals, ``stride <= ~16`` works -- mirrors v1).
        Polyphase ITSELF is exact at any Nt_layer.
    """

    def __init__(
        self,
        real_td_callable: Callable[[np.ndarray], np.ndarray],
        wdm_set_complex,
        data_dt: float,
        ind_min_t: int,
        Nt_active: int,
        Nt_layer: int,
        m_active_half_width: int = 2,
        f0_param_idx: int = 1,
    ):
        if not getattr(wdm_set_complex, "is_complex", False):
            raise ValueError("wdm_set_complex must have is_complex=True.")
        self.real_td_callable = real_td_callable
        self.wdm_set = wdm_set_complex
        self.Nf = int(wdm_set_complex.Nf)
        self.Nt = int(wdm_set_complex.Nt)
        self.N = self.Nf * self.Nt
        self.data_dt = float(data_dt)
        self.ind_min_t = int(ind_min_t)
        self.Nt_active = int(Nt_active)
        self.ind_min_f = int(wdm_set_complex.ind_min_f)
        self.Nf_active_total = int(
            wdm_set_complex.ind_max_f - wdm_set_complex.ind_min_f + 1
        )
        self.m_active_half_width = int(m_active_half_width)
        self.N_layers_active = 2 * self.m_active_half_width + 1
        self.layer_df = float(wdm_set_complex.layer_df)
        self.f0_param_idx = int(f0_param_idx)

        # Validate Nt_layer: must be even, divide Nt, stride <= Nt_active.
        Nt_layer = int(Nt_layer)
        if Nt_layer <= 0 or Nt_layer % 2 != 0:
            raise ValueError(f"Nt_layer={Nt_layer} must be a positive even int.")
        if self.Nt % Nt_layer != 0:
            raise ValueError(
                f"Nt_layer={Nt_layer} must divide Nt={self.Nt}."
            )
        stride = self.Nt // Nt_layer
        if stride > self.Nt_active:
            raise ValueError(
                f"stride={stride} > Nt_active={self.Nt_active}; pick a larger Nt_layer."
            )

        # Window is fixed -- always at full Nt width (its natural support).
        self.window_full = _build_full_window(self.Nf, self.Nt)

        # Sparse grid (bin centres at iFFT output positions).
        N_sparse = self.Nt_active // stride
        n_local = stride // 2 + np.arange(N_sparse) * stride
        n_global = self.ind_min_t + n_local

        self.Nt_layer = int(Nt_layer)
        self.stride = int(stride)
        self.N_sparse_t = int(N_sparse)
        self.n_sparse_local = n_local
        self.n_sparse_global = n_global
        self.kappa_full = 2.0 * np.sqrt(np.pi * self.data_dt) / float(self.Nf)
        self.sqrt_dt = float(np.sqrt(self.data_dt))

    # ------------------------------------------------------------------
    def _td_to_rfft(self, params9: np.ndarray) -> np.ndarray:
        td = self.real_td_callable(np.asarray(params9, dtype=float).reshape(9))
        td = np.asarray(td)
        if td.ndim == 1:
            td = td[None, :]
        return np.fft.rfft(td, axis=-1)             # (nch, N//2 + 1)

    def _active_m_global(self, params9: np.ndarray) -> np.ndarray:
        f0 = float(np.asarray(params9, dtype=float).reshape(9)[self.f0_param_idx])
        # Floor convention -- matches the canonical mm5/mm2 definition and
        # ``gb_chunked_prior_draws.py``. Using round() shifts the active band
        # by one layer when f_frac > 0.5, leaving v2's m_active centred one
        # row above the source's true carrier layer and producing spurious
        # r-spikes wherever |c0| drops near the envelope crossings.
        m_floor = int(np.floor(f0 / self.layer_df))
        m_act = m_floor + np.arange(
            -self.m_active_half_width, self.m_active_half_width + 1
        )
        m_act = np.clip(
            m_act,
            self.ind_min_f,
            self.ind_min_f + self.Nf_active_total - 1,
        )
        return m_act.astype(int)

    def active_m_local(self, params9: np.ndarray) -> np.ndarray:
        """Active m-layer indices in LOCAL active-band coordinates (0-based from ind_min_f)."""
        return self.active_m_global_at(params9) - self.ind_min_f

    def active_m_global_at(self, params9: np.ndarray) -> np.ndarray:
        return self._active_m_global(params9)

    # ------------------------------------------------------------------
    def verify_against_dense(self, x: np.ndarray, c_dense_complex: np.ndarray
                             ) -> float:
        """Per-pixel reldiff of v2 sparse vs lisatools dense at the sparse grid.

        Polyphase is mathematically exact -- this should return < 1e-12 at
        any valid ``Nt_layer``. Useful as a one-shot sanity check during
        construction.
        """
        ref = np.asarray(c_dense_complex)
        if ref.shape[1] != self.Nf_active_total or ref.shape[2] != self.Nt_active:
            raise ValueError(
                f"c_dense_complex shape {ref.shape} != "
                f"(nch, {self.Nf_active_total}, {self.Nt_active})"
            )
        sparse, m_local = self(x)
        ref_sparse = ref[:, m_local, :][:, :, self.n_sparse_local]
        diff = np.abs(sparse - ref_sparse)
        scale = max(float(np.abs(ref_sparse).max()), 1e-300)
        return float(diff.max()) / scale

    # ------------------------------------------------------------------
    def __call__(self, params9: np.ndarray):
        """Return ``(c1_active, m_local)`` for the candidate ``params9``.

        ``c1_active`` shape ``(nch, N_layers_active, N_sparse_t)`` complex.
        ``m_local`` shape ``(N_layers_active,)`` -- active m indices in
        local active-band coordinates (0-based from ``ind_min_f``).
        """
        fd_rfft = self._td_to_rfft(params9)
        m_active = self._active_m_global(params9)
        m_local = m_active - self.ind_min_f
        sparse = _compute_sparse_complex_wdm(
            fd_rfft, m_active, self.Nf, self.Nt, self.Nt_layer,
            self.window_full, self.n_sparse_global, self.stride,
        )
        sparse *= self.sqrt_dt
        return sparse, m_local

    def sparse_from_rfft(self, fd_rfft: np.ndarray, params9: np.ndarray):
        """Sparse complex-WDM at the active m-band given a precomputed rfft.

        Skips the internal TD-synth + rfft -- callers that already hold
        the rfft (or that source it from a fast FD-direct generator) call
        this directly to time-isolate the polyphase WDM math.
        """
        m_active = self._active_m_global(params9)
        m_local = m_active - self.ind_min_f
        sparse = _compute_sparse_complex_wdm(
            fd_rfft, m_active, self.Nf, self.Nt, self.Nt_layer,
            self.window_full, self.n_sparse_global, self.stride,
        )
        sparse *= self.sqrt_dt
        return sparse, m_local


# ---------------------------------------------------------------------------
# v2 GetLL -- inherits v1, replaces sparse grid + dense regen.
# ---------------------------------------------------------------------------


class GBSignalHetWDMGetLLv2(GBSignalHetWDMGetLL):
    """v2 of :class:`gb_signal_het_wdm.GBSignalHetWDMGetLL`.

    Same heterodyne / bin-fold / linearisation machinery; replaces the
    dense per-call ``gb_complex_wdm_gen`` with a calibrated minimal-iFFT
    :class:`GBSparseComplexWDMGen`. Sparse n grid is forced to align with
    the iFFT output stride.
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
        sparse_gen: GBSparseComplexWDMGen,
        reference_params,
        tdi_type: str = "XYZ",
        force_backend: Optional[str] = None,
    ):
        # Bypass v1's __init__ (it would partition bins via linspace);
        # call the parallelbase init directly.
        FastLISAResponseParallelModule.__init__(self, force_backend=force_backend)
        xp = self.backend.xp

        if tdi_type not in {"XYZ", "AE", "AET"}:
            raise ValueError("tdi_type must be 'XYZ', 'AE', or 'AET'.")
        self.wdm_set = wdm_set_complex
        self.tdi_type = tdi_type
        self.sparse_gen = sparse_gen
        # v1's gb_complex_wdm_gen attr is unused but kept to ease debugging.
        self.gb_complex_wdm_gen = None
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
        if Nf_act != sparse_gen.Nf_active_total:
            raise ValueError(
                f"sparse_gen Nf_active_total={sparse_gen.Nf_active_total} != "
                f"data Nf_act={Nf_act}"
            )
        if Nt_active != sparse_gen.Nt_active:
            raise ValueError(
                f"sparse_gen Nt_active={sparse_gen.Nt_active} != data Nt_active={Nt_active}"
            )

        # --- c0 (reference, complex WDM) ---------------------------------
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

        # --- sparse grid from sparse_gen ---------------------------------
        self.N_sparse_t = int(sparse_gen.N_sparse_t)
        stride = int(sparse_gen.stride)
        n_b_idx_np = np.asarray(sparse_gen.n_sparse_local, dtype=int)
        if n_b_idx_np.shape != (self.N_sparse_t,):
            raise ValueError("sparse_gen.n_sparse_local has unexpected shape.")
        # Bin partition: bin b covers pixels [b*stride, (b+1)*stride) in local
        # active coords for b < N_sparse_t-1, and [(N_sparse_t-1)*stride,
        # Nt_active) for the last bin (absorbs any leftover when Nt_active
        # is not a multiple of stride). Centre n_b_idx_local[b] stays at the
        # iFFT output position (stride/2 + b*stride).
        bin_edges = np.arange(self.N_sparse_t + 1) * stride
        bin_edges[-1] = Nt_active
        bin_sizes = np.diff(bin_edges).astype(int)
        self._bin_edges = xp.asarray(bin_edges)
        self._bin_sizes = xp.asarray(bin_sizes, dtype=float)
        self.n_b_idx_local = xp.asarray(n_b_idx_np)
        bin_idx_np = np.repeat(np.arange(self.N_sparse_t), bin_sizes)
        assert bin_idx_np.shape[0] == Nt_active
        self._bin_idx = xp.asarray(bin_idx_np)
        n_off_per_pixel = (
            np.arange(Nt_active) - n_b_idx_np[bin_idx_np]
        ).astype(float)
        self._n_off_per_pixel = xp.asarray(n_off_per_pixel)
        self._mean_Dn = float(np.mean(bin_sizes))

        # c0 cache at sparse centres
        self.c0_sparse = self.c0[:, :, self.n_b_idx_local]
        c0_mag = xp.abs(self.c0_sparse)
        floor = 1e-12 * c0_mag.max(axis=-1, keepdims=True)
        floor = xp.maximum(floor, 1e-300)
        self._c0_mask = c0_mag > floor

        # bin-fold A0/A1/B0/B1 (inherits v1's method)
        self._build_coefficients()

        d_d_raw = self._ip_complex_sum(self.data, self.data)
        h0_h0_raw = self._ip_complex_sum(self.c0, self.c0)
        d_h0_raw = self._ip_complex_sum(self.data, self.c0)
        self.ref_d_d = 0.5 * float(xp.real(d_d_raw))
        self.ref_h0_h0 = 0.5 * float(xp.real(h0_h0_raw))
        self.ref_d_h0 = 0.5 * float(xp.real(d_h0_raw))
        self.ref_ll = (
            self.ref_d_h0 - 0.5 * self.ref_h0_h0 - 0.5 * self.ref_d_d
        )

        self.d_h_out = None
        self.h_h_out = None
        self.r_out = None

    # ------------------------------------------------------------------
    def _r_and_dr(self, params_x1):
        """Override v1: c1 sparse from minimal iFFT, active-m only.

        Returns the FULL ``(nch, Nf_act, N_sparse_t)`` shape (zero-padded
        outside the active m-band) so the v1 ``get_ll`` body works unchanged.
        """
        xp = self.backend.xp
        c1_active, m_local = self.sparse_gen(params_x1)
        c1_active = xp.asarray(c1_active).astype(xp.complex128, copy=False)
        # Zero-pad to full Nf_act, only active rows non-zero.
        c1_sparse_full = xp.zeros(
            (self.nch, self.Nf_act, self.N_sparse_t), dtype=xp.complex128
        )
        m_local_xp = xp.asarray(m_local, dtype=int)
        c1_sparse_full[:, m_local_xp, :] = c1_active

        denom = xp.where(self._c0_mask, self.c0_sparse, 1.0)
        r = xp.where(self._c0_mask, c1_sparse_full / denom, 0.0 + 0.0j)
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
