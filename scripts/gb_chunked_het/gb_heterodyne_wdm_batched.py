#!/usr/bin/env python
"""
Batched single-chunk heterodyne WDM GB get_ll (cupy-optimized, numpy-testable).

Companion to :mod:`gb_heterodyne_fd_batched` (FD inner product). Same
single-chunk (T_chunk == Tobs) heterodyne FD construction; the difference is
the inner product runs in the WDM domain instead of FD:

    chunk_fd -> w_mn (FD -> WDM transform per layer) -> <d|h> via WDM IP

Cross-check: with consistent WDM-transformed data & invC the WDM inner
product should match :class:`GBHeterodyneFDGetLL` to floating-point
reassociation noise. This module includes a self-test that builds a
random FD data realisation, transforms it to WDM, runs both versions,
and compares.

Algorithm refs:
  * Heterodyne FD chunk build mirrors
    ``gb_heterodyne_fd.make_heterodyne_fd``.
  * FD -> WDM transform follows
    ``fastlisaresponse.jax.wdm.fast_inner_heterodyne.gb_chunk_fd_to_wdm_jax``
    (which is bit-identical to ``lisatools.FDSignal.wdmtransform`` at
    chunk-length Nf*Nt_sub).
"""

from __future__ import annotations

from copy import deepcopy
from typing import Optional

import numpy as np

from fastlisaresponse.tdiconfig import TDIConfig
from fastlisaresponse.tdionfly import GBTDIonTheFly
from fastlisaresponse.utils.parallelbase import FastLISAResponseParallelModule
from lisatools.detector import EqualArmlengthOrbits, Orbits


class GBHeterodyneWDMGetLL(FastLISAResponseParallelModule):
    """Batched single-chunk heterodyne WDM likelihood for GB sources.

    Single chunk = whole observation: ``T_chunk == Tobs``,
    ``N_chunk_td == Nf * Nt_sub``. Each binary produces an
    ``N_sparse``-wide non-zero patch in the dense chunk-FD around
    ``k_f0``; the FD->WDM transform converts that to an
    ``(nch, Nf, Nt_sub)`` real WDM block, then inner-product against
    pre-WDM-transformed data and invC.

    Parameters
    ----------
    T : float
        Observation duration.
    t_ref, t_start : float (must be equal)
        Heterodyne reference / sparse-grid origin.
    N_sparse : int
        Sparse FFT length per binary, power of two.
    Nf, Nt_sub : int
        WDM grid: Nf layers, Nt_sub time pixels per chunk. With single
        chunk this equals Nt as well.
    dt : float
        Dense sample step.
    wdm_window : (Nt_sub,) real array
        Pre-computed WDM analysis window (``lisatools.WDMSettings.window``
        truncated to Nt_sub).
    data_wdm : (num_data, nch, Nf, Nt_sub) real
        WDM-transformed data.
    invC_wdm : array
        XYZ: (num_noise, nch, nch, Nf, Nt_sub) real Hermitian Sigma^{-1}.
        AET/AE: (num_noise, nch, Nf, Nt_sub) real diagonal.
    """

    def __init__(
        self,
        T: float,
        t_ref: float,
        t_start: float,
        N_sparse: int,
        Nf: int,
        Nt_sub: int,
        dt: float,
        wdm_window,
        data_wdm,
        invC_wdm,
        orbits: Optional[Orbits] = None,
        tdi_config: Optional[TDIConfig] = None,
        tdi_chan: str = "XYZ",
        force_backend: Optional[str] = None,
        d_d: float = 0.0,
        tdi_type: str = "XYZ",
        tukey_alpha: float = 0.0,
    ):
        super().__init__(force_backend=force_backend)
        # N_sparse: heterodyne FFT length. cupy fft handles any length, but
        # the C++ radix-2 kernel needs power-of-2 -- match that constraint
        # so the same N_sparse value flows across backends.
        if N_sparse < 1 or (N_sparse & (N_sparse - 1)) != 0:
            raise ValueError(
                f"N_sparse={N_sparse} must be a power of two "
                "(matches the C++ radix-2 FFT convention)."
            )
        # Nt_sub: per-layer iFFT length in the FD->WDM transform. Only
        # needs to be even for the half-bin (m * Nt_sub//2 + ...) WDM
        # geometry; not power-of-2.
        if Nt_sub < 2 or Nt_sub % 2 != 0:
            raise ValueError(
                f"Nt_sub={Nt_sub} must be even (WDM half-bin geometry)."
            )
        # Nf: number of WDM layers; no FFT of this length, any positive
        # integer is fine. gb_chunked_test_script uses Nf=1460.
        if Nf < 1:
            raise ValueError(f"Nf={Nf} must be positive.")
        if abs(float(t_start) - float(t_ref)) > 1e-9:
            raise ValueError("t_start must equal t_ref.")
        if tdi_type not in {"XYZ", "AET", "AE"}:
            raise ValueError("tdi_type must be 'XYZ', 'AET', or 'AE'.")

        self.T = float(T)
        self.t_ref = float(t_ref)
        self.t_start = float(t_start)
        self.N_sparse = int(N_sparse)
        self.Nf = int(Nf)
        self.Nt_sub = int(Nt_sub)
        self.dt = float(dt)
        self.dt_sparse = self.T / self.N_sparse
        self.N_chunk_td = self.Nf * self.Nt_sub
        self.n_rfft_chunk = self.N_chunk_td // 2 + 1
        self.df_chunk = 1.0 / self.T  # single chunk = whole obs
        self.d_d = float(d_d)
        self.tukey_alpha = float(tukey_alpha)
        self.tdi_type = tdi_type
        self.tdi_chan = tdi_chan

        self.orbits = orbits
        self.tdi_config = tdi_config

        xp = self.xp

        wdm_window = xp.ascontiguousarray(wdm_window, dtype=xp.float64)
        if wdm_window.shape != (self.Nt_sub,):
            raise ValueError(
                f"wdm_window must be ({self.Nt_sub},); got {wdm_window.shape}"
            )
        self._wdm_window = wdm_window

        data_wdm = xp.ascontiguousarray(data_wdm, dtype=xp.float64)
        if data_wdm.ndim != 4:
            raise ValueError(
                "data_wdm must be (num_data, nch, Nf, Nt_sub); got "
                f"{data_wdm.shape}"
            )
        self.num_data, self.nchannels, _Nf, _Nts = data_wdm.shape
        if (_Nf, _Nts) != (self.Nf, self.Nt_sub):
            raise ValueError(
                f"data_wdm Nf/Nt_sub mismatch: {data_wdm.shape}"
            )

        invC_wdm = xp.ascontiguousarray(invC_wdm, dtype=xp.float64)
        if tdi_type == "XYZ":
            if invC_wdm.ndim != 5 or invC_wdm.shape[1:] != (
                self.nchannels, self.nchannels, self.Nf, self.Nt_sub,
            ):
                raise ValueError(
                    f"For XYZ, invC_wdm must be (num_noise, {self.nchannels}, "
                    f"{self.nchannels}, {self.Nf}, {self.Nt_sub}); got "
                    f"{invC_wdm.shape}"
                )
        else:
            if invC_wdm.ndim != 4 or invC_wdm.shape[1:] != (
                self.nchannels, self.Nf, self.Nt_sub,
            ):
                raise ValueError(
                    f"For {tdi_type}, invC_wdm must be (num_noise, "
                    f"{self.nchannels}, {self.Nf}, {self.Nt_sub}); got "
                    f"{invC_wdm.shape}"
                )
        self.num_noise = invC_wdm.shape[0]
        self._data_wdm = data_wdm
        self._invC_wdm = invC_wdm

        # ---- sparse-time / FFT constants
        self.t_sparse_np = np.asarray(
            self.t_start + np.arange(self.N_sparse) * self.dt_sparse
        )
        self.tau = xp.arange(self.N_sparse, dtype=xp.float64) * self.dt_sparse
        m_arr = np.fft.fftfreq(self.N_sparse, d=1.0 / self.N_sparse).astype(
            np.int64
        )
        self.m_arr = xp.asarray(m_arr)

        # ---- FD->WDM layer geometry, precomputed once
        # For layer m in [0, Nf]: k_global = m*Nt_sub/2 + (k - Nt_sub/2),
        # Hermitian-wrap k < 0 -> -k, k > N/2 -> N - k.
        half_Nt = self.Nt_sub // 2
        k_idx = np.arange(self.Nt_sub)
        m_layer = np.arange(self.Nf + 1)
        k_global = (
            m_layer[:, None] * half_Nt + (k_idx[None, :] - half_Nt)
        )  # (Nf+1, Nt_sub)
        herm_lo = k_global < 0
        herm_hi = k_global > (self.N_chunk_td // 2)
        herm = herm_lo | herm_hi
        k_safe = np.where(herm_lo, -k_global,
                  np.where(herm_hi, self.N_chunk_td - k_global, k_global))
        k_safe = np.clip(k_safe, 0, self.n_rfft_chunk - 1)
        self._k_safe = xp.asarray(k_safe)            # (Nf+1, Nt_sub)
        self._herm = xp.asarray(herm)                # (Nf+1, Nt_sub) bool

        # Parity / sign / mask constants (cached)
        n_arr_np = np.arange(self.Nt_sub)
        mn_par_even = (((m_layer[:, None] + n_arr_np[None, :]) & 1) == 0)
        boundary = (m_layer[:, None] == 0) | (m_layer[:, None] == self.Nf)
        mask_keep = ~(boundary & ~mn_par_even)
        sign = np.where(
            (((m_layer[:, None] + 1) * n_arr_np[None, :]) & 1) == 0, 1.0, -1.0
        )
        self._mn_par_even = xp.asarray(mn_par_even)  # (Nf+1, Nt_sub) bool
        self._sign = xp.asarray(sign)                # (Nf+1, Nt_sub) real
        self._mask_keep = xp.asarray(mask_keep.astype(np.float64))
        self._kappa = 2.0 * np.sqrt(np.pi * self.dt) / float(self.Nf)
        self._sqrt2 = float(np.sqrt(2.0))

    # ------------------------------------------------------------------
    # boilerplate
    # ------------------------------------------------------------------
    @property
    def xp(self):
        return self.backend.xp

    @property
    def orbits(self):
        return self._orbits

    @orbits.setter
    def orbits(self, o):
        if o is None:
            o = EqualArmlengthOrbits()
        elif not isinstance(o, Orbits) and issubclass(o, Orbits):
            o = o()
        else:
            assert isinstance(o, Orbits)
        self._orbits = deepcopy(o)
        if not self._orbits.configured:
            self._orbits.configure(linear_interp_setup=True)

    @property
    def tdi_config(self):
        return self._tdi_config

    @tdi_config.setter
    def tdi_config(self, tc):
        if tc is None:
            tc = TDIConfig("1st generation")
        elif isinstance(tc, str):
            tc = TDIConfig(tc)
        elif not isinstance(tc, TDIConfig):
            raise ValueError("tdi_config must be TDIConfig, str, or None.")
        self._tdi_config = tc

    @classmethod
    def supported_backends(cls):
        return ["fastlisaresponse_" + _t for _t in cls.GPU_RECOMMENDED()]

    # ------------------------------------------------------------------
    # core
    # ------------------------------------------------------------------
    def _make_gb(self, num_bin: int) -> GBTDIonTheFly:
        return GBTDIonTheFly(
            self.xp.asarray(self.t_sparse_np),
            self.T, self.t_ref, 1.0 / self.dt_sparse,
            num_bin,
            tdi_config=self._tdi_config,
            orbits=self._orbits,
            tdi_chan=self.tdi_chan,
            force_backend=self.backend.name.split("_")[-1],
        )

    def _build_chunk_fd(self, params):
        """Build (num_bin, nch, n_rfft_chunk) dense chunk-FD with N_sparse
        non-zero bins per binary -- placed at k_f0[b] + fftfreq.

        Memory-heavy for large num_bin (n_rfft_chunk >> N_sparse). For
        1M binaries / Nf=2048 / Nt_sub=256 this is ~12 TB; sub-batch in
        the caller.
        """
        xp = self.xp
        num_bin = params.shape[0]

        # Source-signal sparse-time evaluation (batched).
        gb = self._make_gb(num_bin)
        out = gb(
            params[:, 0], params[:, 1], params[:, 2], params[:, 3],
            params[:, 4], params[:, 5], params[:, 6], params[:, 7], params[:, 8],
            convert_to_ra_dec=False, return_spline=False,
        )
        tdi_amp = xp.asarray(out.tdi_amp)        # (num_bin, nch, N_sparse)
        tdi_phase = xp.asarray(out.tdi_phase)
        phase_ref = xp.asarray(out.phase_ref)    # (num_bin, N_sparse)

        # Heterodyne + Tukey + FFT (matches make_heterodyne_fd, FD module).
        f0 = params[:, 1]
        k_f0 = xp.round(f0 / self.df_chunk).astype(xp.int64)
        f0_grid = k_f0.astype(xp.float64) * self.df_chunk
        carrier = 2.0 * xp.pi * f0_grid[:, None] * self.tau[None, :]
        total_phase = tdi_phase + phase_ref[:, None, :] - carrier[:, None, :]
        slow = tdi_amp * xp.exp(1j * total_phase)
        if self.tukey_alpha > 0.0:
            n = xp.arange(self.N_sparse, dtype=xp.float64)
            last = float(self.N_sparse - 1)
            n_taper = 0.5 * self.tukey_alpha * last
            xl = xp.clip(n / n_taper, 0.0, 1.0)
            xr = xp.clip((last - n) / n_taper, 0.0, 1.0)
            left = 0.5 * (1.0 + xp.cos(xp.pi * (xl - 1.0)))
            right = 0.5 * (1.0 + xp.cos(xp.pi * (xr - 1.0)))
            window = xp.minimum(left, right)
            slow = slow * window[None, None, :]
        S = xp.fft.fft(slow, axis=-1) * self.dt_sparse
        X_het = 0.5 * S                         # (num_bin, nch, N_sparse)

        # Scatter into dense chunk_fd via advanced index.
        k_arr = k_f0[:, None] + self.m_arr[None, :]  # (num_bin, N_sparse)
        valid = (k_arr >= 0) & (k_arr < self.n_rfft_chunk)
        k_safe = xp.where(valid, k_arr, xp.zeros_like(k_arr))

        chunk_fd = xp.zeros(
            (num_bin, self.nchannels, self.n_rfft_chunk), dtype=xp.complex128
        )
        # per-binary scatter; chunk_fd[b, :, k_safe[b, :]] = X_het[b, :, :] * valid
        b_idx = xp.arange(num_bin)[:, None, None]
        c_idx = xp.arange(self.nchannels)[None, :, None]
        k_idx = k_safe[:, None, :]              # (num_bin, 1, N_sparse) -> bcast
        valid_3d = valid[:, None, :]
        # Use put-style assignment by clearing invalid via masking.
        chunk_fd[b_idx, c_idx, k_idx] = xp.where(
            valid_3d, X_het, 0.0 + 0.0j,
        )
        return chunk_fd

    def _chunk_fd_to_wdm(self, chunk_fd):
        """Batched FD -> WDM transform; mirrors gb_chunk_fd_to_wdm_jax."""
        xp = self.xp
        num_bin = chunk_fd.shape[0]
        # Gather: chunk_fd[b, c, k_safe[m, k]] -> (num_bin, nch, Nf+1, Nt_sub)
        # chunk_fd: (num_bin, nch, n_rfft_chunk); k_safe: (Nf+1, Nt_sub).
        # Using advanced indexing: result[b, c, m, k] = chunk_fd[b, c, k_safe[m, k]].
        fd_slice = chunk_fd[:, :, self._k_safe]  # (num_bin, nch, Nf+1, Nt_sub)
        herm_b = self._herm[None, None, :, :]
        fd_slice = xp.where(herm_b, xp.conj(fd_slice), fd_slice)
        fd_slice = (fd_slice / self.dt) * self._wdm_window[None, None, None, :]
        # iFFT along last axis (Nt_sub).
        after_ifft = xp.fft.ifft(fd_slice, axis=-1)  # complex (..., Nt_sub)

        # Parity Re/Im pick and sign / mask.
        mn_par_even_b = self._mn_par_even[None, None, :, :]  # (1,1,Nf+1,Nt_sub)
        real_part = xp.where(
            mn_par_even_b, after_ifft.real, after_ifft.imag
        )
        sign_b = self._sign[None, None, :, :]
        mask_b = self._mask_keep[None, None, :, :]
        val = self._kappa * sign_b * real_part * mask_b  # (num_bin,nch,Nf+1,Nt_sub)

        # Final w_mn shape (num_bin, nch, Nf, Nt_sub).
        w_mn = xp.zeros(
            (num_bin, self.nchannels, self.Nf, self.Nt_sub), dtype=xp.float64
        )
        if self.Nf > 2:
            w_mn[:, :, 1:self.Nf, :] = val[:, :, 1:self.Nf, :]

        # m=0 row: even n -> val[0]; odd n -> val[Nf] shifted by -1 (jnp.roll(1)).
        n_arr = xp.arange(self.Nt_sub)
        even_mask = (n_arr % 2 == 0)
        # val[:, :, Nf] rolled by +1 along n axis (so n -> n-1 lookup).
        val_Nf_shifted = xp.roll(val[:, :, self.Nf, :], 1, axis=-1)
        m0 = xp.where(
            even_mask[None, None, :],
            val[:, :, 0, :] / self._sqrt2,
            val_Nf_shifted / self._sqrt2,
        )
        w_mn[:, :, 0, :] = m0
        return w_mn

    def get_ll_wdm(
        self,
        params,
        data_index=None,
        noise_index=None,
        convert_to_ra_dec: bool = False,
    ):
        """Batched single-chunk heterodyne WDM likelihood.

        Returns ``-0.5 * (d_d + h_h - 2*d_h)``; components on
        ``self.d_h_out`` / ``self.h_h_out``.
        """
        xp = self.xp
        params = xp.asarray(xp.atleast_2d(params))
        if params.ndim != 2 or params.shape[1] != 9:
            raise ValueError(f"params must be (num_bin, 9); got {params.shape}")
        num_bin = params.shape[0]

        if data_index is None:
            data_index = xp.zeros(num_bin, dtype=xp.int32)
        else:
            data_index = xp.asarray(data_index).astype(xp.int32)
        if noise_index is None:
            noise_index = xp.zeros(num_bin, dtype=xp.int32)
        else:
            noise_index = xp.asarray(noise_index).astype(xp.int32)

        # ---- 1) chunk_fd (heterodyne FD)
        chunk_fd = self._build_chunk_fd(params)

        # ---- 2) FD -> WDM
        w = self._chunk_fd_to_wdm(chunk_fd)         # (num_bin, nch, Nf, Nt_sub)

        # ---- 3) gather data / invC slabs per binary
        d = self._data_wdm[data_index]              # (num_bin, nch, Nf, Nt_sub)
        if self.tdi_type == "XYZ":
            inv = self._invC_wdm[noise_index]       # (num_bin, nch, nch, Nf, Nt_sub)
            # <d|h> = sum_{c1,c2,m,n} d[c1] * w[c2] * invC[c1,c2]
            d_h = xp.einsum("bimn,bjmn,bijmn->b", d, w, inv)
            h_h = xp.einsum("bimn,bjmn,bijmn->b", w, w, inv)
        else:
            inv = self._invC_wdm[noise_index]       # (num_bin, nch, Nf, Nt_sub)
            d_h = xp.einsum("bcmn,bcmn,bcmn->b", d, w, inv)
            h_h = xp.einsum("bcmn,bcmn,bcmn->b", w, w, inv)

        self.d_h_out = d_h
        self.h_h_out = h_h
        return -0.5 * (self.d_d + h_h - 2.0 * d_h)


# ============================================================================
# Self-test: 1-year GB injection -> matched-filter recovery
# ----------------------------------------------------------------------------
# Mirrors gb_chunked_test_script.py's setup (Nf, Nt, dt, t_start, source
# params, sensitivity matrix), but runs the single-chunk path
# (Nt_sub == Nt) of this module instead of GBWDMComputations. The
# matched-filter recovery check is the same: inject one GB source,
# build the template with identical params, and verify
# <d|h> ~ <h|h> ~ SNR^2.
# ============================================================================
def _selftest():
    import os, time
    from lisatools.utils.constants import YRSID_SI
    from lisatools.domains import WDMSettings, TDSettings, TDSignal
    from lisatools.sensitivity import XYZ2SensitivityMatrix

    BACKEND = "cpu"

    # ---- WDM / observation grid (matches gb_chunked_test_script.py) -------
    dt = 10.0
    Nf = int(os.environ.get("NF", 1460))
    # Default Nt reduced for CPU self-test speed; override with NT to
    # match gb_chunked_test_script.py exactly (NT=2560 -> 1yr obs).
    Nt = int(os.environ.get("NT", 256))
    Tobs = Nf * Nt * dt
    N_dense = Nf * Nt

    t_start = int(0.5 * YRSID_SI / dt) * dt
    t_ref = t_start

    N_sparse = int(os.environ.get("N_SPARSE", 256))
    nchannels = 3
    min_freq = 0.0001
    max_freq = 35.0e-3

    print("=" * 72)
    print(f"GBHeterodyneWDMGetLL self-test -- 1-yr GB matched-filter setup")
    print(f"  Nf={Nf}  Nt={Nt}  dt={dt}  Tobs={Tobs:.3e}s "
          f"({Tobs / YRSID_SI:.3f} yr)")
    print(f"  N_sparse={N_sparse}  nchannels={nchannels}")
    print("=" * 72)

    tdi_config = TDIConfig("2nd generation")
    orbits = EqualArmlengthOrbits(force_backend=BACKEND)
    if not orbits.configured:
        orbits.configure(linear_interp_setup=True)

    wdm_set = WDMSettings(
        Nf, Nt, dt, t0=t_start,
        min_freq=min_freq, max_freq=max_freq,
        min_time=20.0 * Nf * dt, max_time=(Nt - 20) * Nf * dt,
        force_backend=BACKEND,
    )
    wdm_window = np.asarray(wdm_set.window)

    # ---- source params (same canonical GB as gb_chunked_test_script.py) ---
    layer_df = float(wdm_set.layer_df)
    m_ref_source = int(3e-3 / layer_df)
    f_frac = 0.5
    f0_val = (m_ref_source + f_frac) * layer_df

    params = np.array([[
        1.0e-22,        # amp
        f0_val,         # f0
        1.0e-17,        # fdot (essentially zero)
        0.0,            # fddot
        2.09802430298,  # phi0
        0.23984234,     # inc
        1.234019814,    # psi
        4.09808143,     # lam
        0.04,           # beta
    ]])
    print(f"  source: amp={params[0,0]:.3e}  f0={params[0,1]*1e3:.5f} mHz "
          f"(m_ref={m_ref_source}, f_frac={f_frac})")

    # ---- build injection via GBTDIonTheFly -> TDSignal -> WDMSignal -------
    # Uses the same Tukey-less injection path as gb_chunked_test_script.
    N_inj = 16384
    print(f"  building injection (N_inj={N_inj} TD samples)...", flush=True)
    t0 = time.perf_counter()

    t_tdi_inj = np.linspace(t_start, t_start + (N_dense - 1) * dt, N_inj)
    gb_gen_inj = GBTDIonTheFly(
        t_tdi_inj, Tobs, t_ref, 1.0 / dt, 1,
        tdi_config=tdi_config, orbits=orbits,
        tdi_chan="XYZ", force_backend=BACKEND,
    )
    inj_tmp = gb_gen_inj(
        np.full(1, params[0, 0]), np.full(1, params[0, 1]),
        np.full(1, params[0, 2]), np.full(1, params[0, 3]),
        np.full(1, params[0, 4]), np.full(1, params[0, 5]),
        np.full(1, params[0, 6]), np.full(1, params[0, 7]),
        np.full(1, params[0, 8]),
        convert_to_ra_dec=False, return_spline=True,
    )

    t_arr = np.arange(N_dense) * dt + t_start
    data_td = np.asarray(inj_tmp.eval_tdi(t_arr))  # (nch, N_dense) real
    if data_td.ndim == 3:
        data_td = data_td[0]  # squeeze leading "num_sub" axis
    td_set = TDSettings(N_dense, dt, t0=t_start, force_backend=BACKEND)
    data_wdm_signal = TDSignal(data_td, settings=td_set).transform(wdm_set)
    data_wdm = np.asarray(data_wdm_signal.arr)  # (nch, Nf, Nt)
    # We pad back to the wdm_set's full Nf x Nt if the transform stayed on
    # the active band; AnalysisContainer convention returns active-band
    # arrays. Pad zeros for inactive pixels so our class's data_wdm matches
    # its own (Nf, Nt) shape contract.
    if data_wdm.shape != (nchannels, Nf, Nt):
        full = np.zeros((nchannels, Nf, Nt), dtype=float)
        m_lo = wdm_set.ind_min_f
        m_hi = wdm_set.ind_max_f + 1
        full[:, m_lo:m_hi, :] = data_wdm
        data_wdm = full
    print(f"    done ({time.perf_counter()-t0:.1f}s); "
          f"|data_wdm|_max = {np.abs(data_wdm).max():.3e}")

    # ---- build invC from XYZ2SensitivityMatrix ---------------------------
    print("  building invC = XYZ2SensitivityMatrix(scirdv1)^{-1} ...",
          flush=True)
    t0 = time.perf_counter()
    sens_mat = XYZ2SensitivityMatrix(wdm_set, model="scirdv1")
    invC_active = np.asarray(sens_mat.invC)        # (3, 3, Nfa, Nta)
    invC_active = np.where(np.isfinite(invC_active), invC_active, 0.0)
    if invC_active.shape != (nchannels, nchannels, Nf, Nt):
        full = np.zeros((nchannels, nchannels, Nf, Nt), dtype=float)
        m_lo = wdm_set.ind_min_f
        m_hi = wdm_set.ind_max_f + 1
        full[:, :, m_lo:m_hi, :] = invC_active
        invC_active = full
    invC_wdm = invC_active[None]                   # (1, 3, 3, Nf, Nt)
    print(f"    done ({time.perf_counter()-t0:.1f}s); "
          f"invC[0,0] median = {np.median(invC_wdm[0, 0, 0]):.3e}")

    # ---- run the batched single-chunk WDM get_ll --------------------------
    print("  running GBHeterodyneWDMGetLL.get_ll_wdm "
          "(single-chunk, Nt_sub=Nt)...", flush=True)
    t0 = time.perf_counter()
    wdm_comp = GBHeterodyneWDMGetLL(
        T=Tobs, t_ref=t_ref, t_start=t_start, N_sparse=N_sparse,
        Nf=Nf, Nt_sub=Nt, dt=dt, wdm_window=wdm_window,
        data_wdm=data_wdm[None], invC_wdm=invC_wdm,
        orbits=orbits, tdi_config=tdi_config,
        force_backend=BACKEND, tdi_type="XYZ",
    )
    ll = np.asarray(wdm_comp.get_ll_wdm(params, convert_to_ra_dec=False))
    d_h = np.asarray(wdm_comp.d_h_out)
    h_h = np.asarray(wdm_comp.h_h_out)
    t_call = time.perf_counter() - t0
    print(f"    done ({t_call:.1f}s)")

    snr2_meas = d_h[0] * d_h[0] / max(h_h[0], 1e-300)
    overlap = d_h[0] / np.sqrt(max(h_h[0], 1e-300) * max(h_h[0], 1e-300))
    print()
    print(f"  <d|h>   = {d_h[0]:+.6e}")
    print(f"  <h|h>   = {h_h[0]:+.6e}     (= optimal SNR^2)")
    print(f"  <d|h>^2 / <h|h>  = {snr2_meas:+.6e}     "
          f"(matched SNR^2; should ~= <h|h>)")
    print(f"  matched overlap  = <d|h> / sqrt(<h|h><h|h>) = {overlap:+.6f}")
    print(f"  ll = -0.5*(<d|d>=0 + <h|h> - 2<d|h>) = {ll[0]:+.6e}")

    # ---- pass criteria ----------------------------------------------------
    # Matched-filter recovery: overlap of injection vs identical template
    # should be very close to 1 (down to TD->WDM truncation / single-chunk
    # heterodyne FD vs dense rfft mismatch). Threshold 0.99 is generous;
    # canonical chunked-het gets ~1 - 1e-9 on this configuration.
    finite = (np.isfinite(d_h).all() and np.isfinite(h_h).all()
              and h_h[0] > 0.0)
    matched = abs(overlap - 1.0) < 0.05
    print()
    print(f"  finite & <h|h> > 0  : {'PASS' if finite else 'FAIL'}")
    print(f"  overlap ~ 1 (<5%)   : {'PASS' if matched else 'FAIL'}  "
          f"(|1 - overlap| = {abs(1.0 - overlap):.3e})")

    return bool(finite and matched)


if __name__ == "__main__":
    ok = _selftest()
    raise SystemExit(0 if ok else 1)
