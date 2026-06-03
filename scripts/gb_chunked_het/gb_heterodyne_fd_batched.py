#!/usr/bin/env python
"""
Batched single-chunk heterodyne FD GB get_ll (cupy-optimized, numpy-testable).

Mirrors the algorithm in :func:`gb_heterodyne_fd.make_heterodyne_fd` but
processes many binaries at once via the batched :class:`GBTDIonTheFly`
(``num_sub == num_bin``). All non-source-signal work is pure xp
(numpy or cupy); the only C++ call is the source signal itself.

Designed for cupy's large-batch global-memory regime (10K-1M binaries).
Test path: instantiate with ``force_backend='cpu'`` so ``xp == numpy``,
then run against :class:`fastlisaresponse.gbcomps.GBFDComputations`
for cross-validation. The two implementations should match
``d_h``, ``h_h`` to floating-point reassociation noise.

Conventions
-----------
* Single chunk = whole observation. ``T_chunk == Tobs``.
* FD inner product (matches lisatools / GBFDComputations C++ kernel)::

      <a|b> = 4 * df * Re sum_{c1,c2,k} conj(a_c1[k]) * b_c2[k]
                                        * invC[c1,c2,k]

* For ``tdi_type == "XYZ"`` ``invC`` is the full Hermitian (real)
  Sigma^{-1} of shape ``(num_noise, 3, 3, n_rfft)``.
* For ``tdi_type in {"AET", "AE"}`` ``invC`` is diagonal of shape
  ``(num_noise, nchannels, n_rfft)``.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Optional

import numpy as np

from fastlisaresponse.tdiconfig import TDIConfig
from fastlisaresponse.tdionfly import GBTDIonTheFly
from fastlisaresponse.utils.parallelbase import FastLISAResponseParallelModule
from lisatools.detector import EqualArmlengthOrbits, Orbits


class GBHeterodyneFDGetLL(FastLISAResponseParallelModule):
    """Batched single-chunk heterodyne FD likelihood for GB sources.

    Constructor mirrors :class:`GBFDComputations`. The data / invC are
    stashed at construction time and stay resident in xp memory
    (one device-side global slab); ``get_ll`` is then called per
    parameter batch.

    Parameters
    ----------
    T : float
        Observation duration in seconds.
    t_ref, t_start : float
        Reference and sparse-grid origin times. ``t_ref == t_start`` is
        required (matches :class:`GBFDComputations`).
    N_sparse : int
        Sparse-time grid length; must be a power of two.
    df : float
        Dense-rfft frequency step (``df == 1/T``).
    data_fd : (num_data, nchannels, n_rfft) complex xp array
        Dense rfft of the data.
    invC : xp array
        ``(num_noise, nch, nch, n_rfft)`` if XYZ, else
        ``(num_noise, nch, n_rfft)``.
    orbits, tdi_config, tdi_chan, force_backend : as :class:`GBFDComputations`.
    """

    def __init__(
        self,
        T: float,
        t_ref: float,
        t_start: float,
        N_sparse: int,
        df: float,
        data_fd,
        invC,
        orbits: Optional[Orbits] = None,
        tdi_config: Optional[TDIConfig] = None,
        tdi_chan: str = "XYZ",
        force_backend: Optional[str] = None,
        d_d: float = 0.0,
        tdi_type: str = "XYZ",
    ):
        super().__init__(force_backend=force_backend)
        if N_sparse < 1 or (N_sparse & (N_sparse - 1)) != 0:
            raise ValueError("N_sparse must be a power of two.")
        if abs(float(t_start) - float(t_ref)) > 1e-9:
            raise ValueError(
                "t_start must equal t_ref (heterodyne phase = 1 only then)."
            )
        if tdi_type not in {"XYZ", "AET", "AE"}:
            raise ValueError("tdi_type must be 'XYZ', 'AET', or 'AE'.")

        self.T = float(T)
        self.t_ref = float(t_ref)
        self.t_start = float(t_start)
        self.N_sparse = int(N_sparse)
        self.df = float(df)
        self.dt_sparse = self.T / self.N_sparse
        self.d_d = float(d_d)
        self.tdi_type = tdi_type
        self.tdi_chan = tdi_chan

        self.orbits = orbits
        self.tdi_config = tdi_config

        xp = self.xp
        data_fd = xp.ascontiguousarray(data_fd)
        if data_fd.ndim != 3:
            raise ValueError(
                "data_fd must be (num_data, nchannels, n_rfft); got "
                f"{data_fd.shape}"
            )
        self.num_data, self.nchannels, self.n_rfft = data_fd.shape

        invC = xp.ascontiguousarray(invC, dtype=xp.float64)
        if tdi_type == "XYZ":
            if invC.ndim != 4 or invC.shape[1:] != (
                self.nchannels, self.nchannels, self.n_rfft,
            ):
                raise ValueError(
                    f"For XYZ, invC must be (num_noise, {self.nchannels}, "
                    f"{self.nchannels}, {self.n_rfft}); got {invC.shape}"
                )
        else:
            if invC.ndim != 3 or invC.shape[1:] != (
                self.nchannels, self.n_rfft,
            ):
                raise ValueError(
                    f"For {tdi_type}, invC must be (num_noise, "
                    f"{self.nchannels}, {self.n_rfft}); got {invC.shape}"
                )
        self.num_noise = invC.shape[0]
        self._data_fd = data_fd
        self._invC = invC

        # Sparse-time grid (one row -- GBTDIonTheFly tiles across num_sub).
        self.t_sparse_np = np.asarray(
            self.t_start + np.arange(self.N_sparse) * self.dt_sparse
        )
        # tau values and the fftfreq integer offsets for the gather.
        self.tau = xp.arange(self.N_sparse, dtype=xp.float64) * self.dt_sparse
        m_arr = np.fft.fftfreq(self.N_sparse, d=1.0 / self.N_sparse).astype(
            np.int64
        )
        self.m_arr = xp.asarray(m_arr)

    # ------------------------------------------------------------------
    # boilerplate -- mirrors GBFDComputations
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
        """Construct a per-call GBTDIonTheFly sized to this batch.

        ``num_sub`` is the C++-side batch length; reconstructing per call is
        cheap (it stashes pointers to existing orbit / tdi_config slabs).
        """
        return GBTDIonTheFly(
            self.xp.asarray(self.t_sparse_np),
            self.T, self.t_ref, 1.0 / self.dt_sparse,
            num_bin,
            tdi_config=self._tdi_config,
            orbits=self._orbits,
            tdi_chan=self.tdi_chan,
            force_backend=self.backend.name.split("_")[-1],
        )

    def get_ll_fd(
        self,
        params,
        data_index=None,
        noise_index=None,
        convert_to_ra_dec: bool = False,
    ):
        """Batched single-chunk heterodyne FD likelihood.

        Returns
        -------
        ll : (num_bin,) array
            ``-0.5 * (d_d + h_h - 2*d_h)``; ``self.d_h_out`` / ``self.h_h_out``
            hold the components (matches :meth:`GBFDComputations.get_ll_fd`).
        """
        xp = self.xp
        params = xp.asarray(xp.atleast_2d(params))
        if params.ndim != 2 or params.shape[1] != 9:
            raise ValueError(
                f"params must be (num_bin, 9); got {params.shape}"
            )
        if convert_to_ra_dec:
            # Match GBFDComputations._prep_params behaviour.
            from fastlisaresponse.utils.utility import ecliptic_to_icrs
            p = params.copy()
            lam, beta = ecliptic_to_icrs(p[:, -2].copy(), p[:, -1].copy())
            p = p.at[:, -2].set(lam) if hasattr(p, "at") else p
            if not hasattr(p, "at"):
                p[:, -2] = lam
                p[:, -1] = beta
            params = p
        num_bin = params.shape[0]

        if data_index is None:
            data_index = xp.zeros(num_bin, dtype=xp.int32)
        else:
            data_index = xp.asarray(data_index).astype(xp.int32)
        if noise_index is None:
            noise_index = xp.zeros(num_bin, dtype=xp.int32)
        else:
            noise_index = xp.asarray(noise_index).astype(xp.int32)

        # --- 1) Source-signal sparse-time evaluation, batched.
        gb = self._make_gb(num_bin)
        out = gb(
            params[:, 0], params[:, 1], params[:, 2], params[:, 3],
            params[:, 4], params[:, 5], params[:, 6], params[:, 7], params[:, 8],
            convert_to_ra_dec=False, return_spline=False,
        )
        tdi_amp = xp.asarray(out.tdi_amp)        # (num_bin, nch, N_sparse)
        tdi_phase = xp.asarray(out.tdi_phase)    # (num_bin, nch, N_sparse)
        phase_ref = xp.asarray(out.phase_ref)    # (num_bin, N_sparse)

        # --- 2) Heterodyne carrier (snap each binary's f0 to rfft grid).
        f0 = params[:, 1]
        k_f0 = xp.round(f0 / self.df).astype(xp.int64)         # (num_bin,)
        f0_grid = k_f0.astype(xp.float64) * self.df            # (num_bin,)
        carrier = 2.0 * xp.pi * f0_grid[:, None] * self.tau[None, :]
        # carrier: (num_bin, N_sparse)

        # --- 3) Slow signal, FFT, scale (matches make_heterodyne_fd).
        total_phase = tdi_phase + phase_ref[:, None, :] - carrier[:, None, :]
        slow = tdi_amp * xp.exp(1j * total_phase)
        S = xp.fft.fft(slow, axis=-1) * self.dt_sparse
        X_het = 0.5 * S                                         # (num_bin, nch, N_sparse)

        # --- 4) Inner product: gather data / invC at the per-binary bins.
        # k_arr[b, m] = k_f0[b] + m_arr[m]; mask out-of-rfft bins.
        k_arr = k_f0[:, None] + self.m_arr[None, :]              # (num_bin, N_sparse)
        valid = (k_arr >= 0) & (k_arr < self.n_rfft)
        k_safe = xp.where(valid, k_arr, xp.zeros_like(k_arr))    # safe gather idx
        valid_f = valid.astype(xp.float64)

        # Data gather: data_fd[data_index[b], c, k_safe[b, m]]
        # Build per-bin index arrays and use advanced indexing.
        # data_fd: (num_data, nch, n_rfft) -> we want (num_bin, nch, N_sparse).
        b_idx = xp.arange(num_bin)[:, None]                      # (num_bin, 1)
        # Index per (b, m): data_fd[data_index[b], :, k_safe[b, m]]
        # Result d_at: (num_bin, N_sparse, nch) -> moveaxis to (num_bin, nch, N_sparse)
        d_at = self._data_fd[data_index[:, None], :, k_safe]     # (num_bin, N_sparse, nch)
        d_at = xp.moveaxis(d_at, -1, 1)                          # (num_bin, nch, N_sparse)
        d_at = d_at * valid_f[:, None, :]                        # zero invalid bins

        # h := X_het (already zero outside [0, n_rfft) because we mask via valid_f)
        h = X_het * valid_f[:, None, :]

        if self.tdi_type == "XYZ":
            # invC: (num_noise, nch, nch, n_rfft)
            inv_at = self._invC[noise_index[:, None], :, :, k_safe]
            # shape: (num_bin, N_sparse, nch, nch) -> (num_bin, nch, nch, N_sparse)
            inv_at = xp.transpose(inv_at, (0, 2, 3, 1))
            inv_at = inv_at * valid_f[:, None, None, :]          # zero invalid bins

            # <d|h> = 4 df Re sum_{c1,c2,k} conj(d[c1]) * h[c2] * invC[c1,c2]
            # Re(conj(d) * h) = d.real*h.real + d.imag*h.imag
            # Pre-stack real/imag along channel axes.
            dh_per_k = (
                xp.einsum(
                    "bik,bjk,bijk->bk",
                    d_at.real, h.real, inv_at,
                )
                + xp.einsum(
                    "bik,bjk,bijk->bk",
                    d_at.imag, h.imag, inv_at,
                )
            )
            hh_per_k = (
                xp.einsum(
                    "bik,bjk,bijk->bk",
                    h.real, h.real, inv_at,
                )
                + xp.einsum(
                    "bik,bjk,bijk->bk",
                    h.imag, h.imag, inv_at,
                )
            )
            d_h = 4.0 * self.df * dh_per_k.sum(axis=-1)
            h_h = 4.0 * self.df * hh_per_k.sum(axis=-1)
        else:
            # Diagonal invC: (num_noise, nch, n_rfft)
            inv_at = self._invC[noise_index[:, None], :, k_safe]   # (num_bin, N_sparse, nch)
            inv_at = xp.moveaxis(inv_at, -1, 1)                    # (num_bin, nch, N_sparse)
            inv_at = inv_at * valid_f[:, None, :]
            dh_per = (d_at.real * h.real + d_at.imag * h.imag) * inv_at
            hh_per = (h.real ** 2 + h.imag ** 2) * inv_at
            d_h = 4.0 * self.df * dh_per.sum(axis=(1, 2))
            h_h = 4.0 * self.df * hh_per.sum(axis=(1, 2))

        self.d_h_out = d_h
        self.h_h_out = h_h
        return -0.5 * (self.d_d + h_h - 2.0 * d_h)


# ============================================================================
# Self-test against GBFDComputations (cpu numpy drop-in).
# ============================================================================
def _selftest():
    """Compare batched cupy-optimized path vs. C++ GBFDComputations.get_ll_fd."""
    import time
    from lisatools.utils.constants import YRSID_SI
    from fastlisaresponse.gbcomps import GBFDComputations

    BACKEND = "cpu"
    rng = np.random.default_rng(0)

    # ---- setup ----
    dt = 15.0
    Tobs = int(round(0.5 * YRSID_SI / dt)) * dt  # short obs to keep test fast
    N_dense = int(round(Tobs / dt))
    n_rfft = N_dense // 2 + 1
    df = 1.0 / Tobs
    t_start = 0.0
    t_ref = 0.0
    N_sparse = 256
    nchannels = 3

    tdi_config = TDIConfig("2nd generation")
    orbits = EqualArmlengthOrbits(force_backend=BACKEND)

    # Synthetic data + invC (diagonal real per channel, just for shape).
    data_fd = (rng.standard_normal((1, nchannels, n_rfft))
               + 1j * rng.standard_normal((1, nchannels, n_rfft))).astype(complex)
    invC_xyz = np.zeros((1, nchannels, nchannels, n_rfft), dtype=float)
    # Diagonal Sigma^{-1}: 1/Sn per channel.
    inv_sn = (rng.uniform(0.5, 1.5, size=(nchannels, n_rfft)) ** 2)
    for c in range(nchannels):
        invC_xyz[0, c, c, :] = inv_sn[c]
    # Don't put DC noise weight in -- match standard FD convention.
    invC_xyz[:, :, :, 0] = 0.0

    # ---- 2 binaries with distinct f0 ----
    num_bin = 2
    params = np.array([
        # amp,    f0,         fdot,   fddot, phi0, inc,  psi,  lam,  beta
        [8.0e-23, 20.0e-3, 1.0e-14, 0.0, 2.098, 0.240, 1.234, 4.098, 0.090],
        [6.0e-23, 21.0e-3, 8.0e-15, 0.0, 1.500, 0.500, 0.800, 3.500, -0.200],
    ])

    # ---- reference: GBFDComputations.get_ll_fd ----
    fd_comp = GBFDComputations(
        T=Tobs, t_ref=t_ref, t_start=t_start, N_sparse=N_sparse, df=df,
        data_fd=data_fd, invC=invC_xyz,
        orbits=orbits, tdi_config=tdi_config,
        force_backend=BACKEND, tdi_type="XYZ",
    )
    t0 = time.perf_counter()
    ll_ref = fd_comp.get_ll_fd(params, convert_to_ra_dec=False)
    t_ref_call = time.perf_counter() - t0
    d_h_ref = np.asarray(fd_comp.d_h_out)
    h_h_ref = np.asarray(fd_comp.h_h_out)

    # ---- new: batched xp version ----
    batched = GBHeterodyneFDGetLL(
        T=Tobs, t_ref=t_ref, t_start=t_start, N_sparse=N_sparse, df=df,
        data_fd=data_fd, invC=invC_xyz,
        orbits=orbits, tdi_config=tdi_config,
        force_backend=BACKEND, tdi_type="XYZ",
    )
    t0 = time.perf_counter()
    ll_new = batched.get_ll_fd(params, convert_to_ra_dec=False)
    t_new_call = time.perf_counter() - t0
    d_h_new = np.asarray(batched.d_h_out)
    h_h_new = np.asarray(batched.h_h_out)

    print("=" * 70)
    print(f"GBHeterodyneFDGetLL self-test  (num_bin={num_bin}, N_sparse={N_sparse})")
    print("=" * 70)
    print(f"  C++ GBFDComputations.get_ll_fd : {t_ref_call*1e3:7.1f} ms")
    print(f"  xp  GBHeterodyneFDGetLL.get_ll_fd: {t_new_call*1e3:7.1f} ms")
    print()
    print(f"  <d|h>   ref = {d_h_ref}")
    print(f"  <d|h>   new = {d_h_new}")
    rd = np.abs(d_h_new - d_h_ref) / np.maximum(np.abs(d_h_ref), 1e-300)
    print(f"  reldiff     = {rd}")
    print()
    print(f"  <h|h>   ref = {h_h_ref}")
    print(f"  <h|h>   new = {h_h_new}")
    rh = np.abs(h_h_new - h_h_ref) / np.maximum(np.abs(h_h_ref), 1e-300)
    print(f"  reldiff     = {rh}")
    print()
    print(f"  ll      ref = {np.asarray(ll_ref)}")
    print(f"  ll      new = {np.asarray(ll_new)}")
    rl = np.abs(np.asarray(ll_new) - np.asarray(ll_ref)) / np.maximum(
        np.abs(np.asarray(ll_ref)), 1e-300
    )
    print(f"  reldiff     = {rl}")
    print()
    ok = (rd.max() < 1e-10) and (rh.max() < 1e-10)
    print("  PASS" if ok else "  FAIL: reldiff exceeds 1e-10")
    return ok


if __name__ == "__main__":
    _selftest()
