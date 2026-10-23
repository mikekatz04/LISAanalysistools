"""JAX port of ``WaveletLookupTable`` (formerly ``TDIonTheFly.cu:455-591``).

The C++ per-pixel ``WaveletLookupTable`` path was retired at Phase 3L
(replaced by the chunked-heterodyne kernels; see ``gbgpu/cutils/gb_tdi_on_the_fly.hh``),
so this JAX class is now the only implementation; its consumer is
``gbgpu.jax.wdm.kernels`` / ``gbgpu.jax.wdm.computation_group``. The line
references in this module point at the pre-absorption lisa-on-gpu file.

Two table layouts are supported, matching the C++ ``LookupKind`` enum:

* ``LOOKUP_PER_N = 0`` -- table is ``(Nt, num_fdot, num_f)``; the
  per-n axis is offset into at lookup time.
* ``LOOKUP_N_REF_ONLY = 1`` -- table is ``(num_fdot, num_f)`` (built
  at a single ``(m_ref, n_ref)`` pixel); the ``(-1)^(layer_n - n_ref)``
  correction is applied in :func:`get_w_mn_lookup`.

The per-pixel WDM coefficient construction matches
``WaveletLookupTable::get_w_mn_lookup`` byte-for-byte (modulo FMA
re-orderings)::

    w_mn = c_nm * Re(M') + s_nm * Im(M')

with the ``(m+n)``-parity sin/cos swap and the
``(layer_m - m_ref)``-parity sign correction folded in.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


# C++ ``LookupKind`` enum values.
LOOKUP_PER_N = 0
LOOKUP_N_REF_ONLY = 1


class WaveletLookupTableWrapJAX:
    """Pure-JAX analog of ``WaveletLookupTableWrap``.

    Takes 20 constructor arguments (the two tables plus 18 scalars). The two
    table arrays are converted to ``jnp.ndarray`` through the host
    (``np.asarray``); the scalar / integer parameters stay as Python
    attributes so that downstream JIT traces them as static. Every argument is
    stored under its own name, plus ``Nf_active`` / ``Nt_active``.

    Args:
        c_nm_all (array-like): Cosine-coefficient table, ``(Nt, num_fdot, num_f)``
            for ``LOOKUP_PER_N`` or ``(num_fdot, num_f)`` for ``LOOKUP_N_REF_ONLY``.
        s_nm_all (array-like): Sine-coefficient table, same layout.
        num_f: Table points along the scaled frequency.
        num_fdot: Table points along fdot (1 = 1-D interpolation in f).
        df_interp: Scaled-frequency step of the table [Hz].
        dfdot_interp: fdot step of the table [Hz/s].
        min_f_scaled: First scaled frequency of the table [Hz].
        min_fdot: First fdot of the table [Hz/s].
        layer_df: WDM frequency-layer width [Hz].
        layer_dt: WDM time-pixel width [s].
        Nf: Number of frequency layers.
        Nt: Number of time pixels.
        num_channel: Number of TDI channels.
        ind_min_t: First time pixel of the active band.
        ind_max_t: Last time pixel of the active band (inclusive).
        ind_min_f: First frequency layer of the active band.
        ind_max_f: Last frequency layer of the active band.
        m_ref: Reference layer of the table.
        n_ref: Reference time pixel of the table (``LOOKUP_N_REF_ONLY``).
        kind: ``LOOKUP_PER_N`` (0) or ``LOOKUP_N_REF_ONLY`` (1).
    """

    def __init__(
        self,
        c_nm_all,
        s_nm_all,
        num_f: int,
        num_fdot: int,
        df_interp: float,
        dfdot_interp: float,
        min_f_scaled: float,
        min_fdot: float,
        layer_df: float,
        layer_dt: float,
        Nf: int,
        Nt: int,
        num_channel: int,
        ind_min_t: int,
        ind_max_t: int,
        ind_min_f: int,
        ind_max_f: int,
        m_ref: int,
        n_ref: int,
        kind: int,
    ):
        # Convert via numpy first (handles cupy gracefully).
        self.c_nm_all = jnp.asarray(np.asarray(c_nm_all))
        self.s_nm_all = jnp.asarray(np.asarray(s_nm_all))

        self.num_f = int(num_f)
        self.num_fdot = int(num_fdot)
        self.df_interp = float(df_interp)
        self.dfdot_interp = float(dfdot_interp)
        self.min_f_scaled = float(min_f_scaled)
        self.min_fdot = float(min_fdot)
        self.layer_df = float(layer_df)
        self.layer_dt = float(layer_dt)
        self.Nf = int(Nf)
        self.Nt = int(Nt)
        self.num_channel = int(num_channel)
        self.ind_min_t = int(ind_min_t)
        self.ind_max_t = int(ind_max_t)
        self.ind_min_f = int(ind_min_f)
        self.ind_max_f = int(ind_max_f)
        self.Nf_active = self.ind_max_f - self.ind_min_f + 1
        self.Nt_active = self.ind_max_t - self.ind_min_t + 1
        self.m_ref = int(m_ref)
        self.n_ref = int(n_ref)
        self.kind = int(kind)

    # ----------------------------------------------------------------
    # Linear interpolation in (f_scaled, fdot) on the lookup table.
    # Mirrors ``WaveletLookupTable::linear_interp``.
    # ----------------------------------------------------------------
    def _linear_interp(self, f_scaled: jnp.ndarray, fdot: jnp.ndarray,
                       z_vals: jnp.ndarray, layer_n: jnp.ndarray) -> jnp.ndarray:
        """Bilinear (or 1D linear if num_fdot == 1) interp on the table.

        Designed to be called from inside ``jax.vmap`` (one pixel at a
        time); ``f_scaled``, ``fdot``, ``layer_n`` are scalars from
        the caller's perspective. The vmap rebroadcasts to whatever
        batch shape the caller uses. Points outside the table return 0.
        In the bilinear branch the f-axis knots are ``df_interp * index``
        without the ``min_f_scaled`` offset (verbatim from the C++), while
        the 1-D branch includes the offset.
        """
        # Pick the (num_fdot, num_f) slice for this layer_n.
        if self.kind == LOOKUP_PER_N:
            # PER_N: z_vals shape (Nt, num_fdot, num_f); pull slice [layer_n].
            # Use jnp.take so layer_n can be a tracer.
            z_slice = jnp.take(z_vals, layer_n, axis=0)   # shape (num_fdot, num_f)
        else:
            # N_REF_ONLY: z_vals shape (num_fdot, num_f) already.
            z_slice = z_vals

        if self.num_fdot > 1:
            f_index = jnp.floor((f_scaled - self.min_f_scaled) / self.df_interp).astype(jnp.int32)
            fdot_index = jnp.floor((fdot - self.min_fdot) / self.dfdot_interp).astype(jnp.int32)
            in_bounds = (f_index >= 0) & (f_index < self.num_f - 1) & \
                        (fdot_index >= 0) & (fdot_index < self.num_fdot - 1)
            fi = jnp.clip(f_index, 0, self.num_f - 2)
            fdi = jnp.clip(fdot_index, 0, self.num_fdot - 2)

            # NOTE: the C++ uses x1 = df_interp * f_index (no
            # min_f_scaled offset). We match the C++ verbatim so the
            # parity is bit-tight against the reference path.
            x1 = self.df_interp * fi
            x2 = self.df_interp * (fi + 1)
            # fdot axis: its own spacing AND offset (was df_interp with no min_fdot;
            # inert while every table had num_fdot == 1).
            y1 = self.min_fdot + self.dfdot_interp * fdi
            y2 = self.min_fdot + self.dfdot_interp * (fdi + 1)

            # z_slice is a (num_fdot, num_f) 2-D array; (fdi, fi) are
            # scalars so this gather is just basic indexing.
            z11 = z_slice[fdi, fi]
            z12 = z_slice[fdi + 1, fi]
            z21 = z_slice[fdi, fi + 1]
            z22 = z_slice[fdi + 1, fi + 1]

            # Bilinear. f_x_y2 is the row fdi+1, i.e. corners z12 and z22 (the
            # historical copy read z21, a typo mirrored from the C++; fix the C++
            # copy too if it is revived).
            f_x_y1 = (x2 - f_scaled) / (x2 - x1) * z11 + (f_scaled - x1) / (x2 - x1) * z21
            f_x_y2 = (x2 - f_scaled) / (x2 - x1) * z12 + (f_scaled - x1) / (x2 - x1) * z22
            f_xy = (y2 - fdot) / (y2 - y1) * f_x_y1 + (fdot - y1) / (y2 - y1) * f_x_y2
            return jnp.where(in_bounds, f_xy, 0.0)
        else:
            # num_fdot == 1: 1D linear interp in f.
            f_index = jnp.floor((f_scaled - self.min_f_scaled) / self.df_interp).astype(jnp.int32)
            in_bounds = (f_index >= 0) & (f_index < self.num_f - 1)
            fi = jnp.clip(f_index, 0, self.num_f - 2)
            x1 = self.df_interp * fi + self.min_f_scaled
            x2 = self.df_interp * (fi + 1) + self.min_f_scaled
            z1 = z_slice[0, fi]
            z2 = z_slice[0, fi + 1]
            f_y = z1 + (f_scaled - x1) * (z2 - z1) / (x2 - x1)
            return jnp.where(in_bounds, f_y, 0.0)

    # ----------------------------------------------------------------
    # get_w_mn_lookup -- evaluate WDM coefficient at pixel (m, n).
    # ----------------------------------------------------------------
    def get_w_mn_lookup(self, tdi_channel_val: jnp.ndarray,
                        f: jnp.ndarray, fdot: jnp.ndarray,
                        layer_m: jnp.ndarray, layer_n: jnp.ndarray) -> jnp.ndarray:
        """Return the WDM coefficient ``w_mn`` of one channel value at pixel (m, n).

        Mirrors ``WaveletLookupTable::get_w_mn_lookup`` (formerly
        ``TDIonTheFly.cu:520-575``).

        Args:
            tdi_channel_val: Complex value ``M'`` (C++ convention, already
                rotated by ``conj(M * exp(-i pi/2))``; see
                :func:`~lisatools.jax.wdm.fast_inner.fast_wdm_inner_jax`).
            f: Instantaneous frequency [Hz].
            fdot: Frequency derivative [Hz/s].
            layer_m: Frequency layer ``m``.
            layer_n: Time pixel ``n``.

        Returns:
            jnp.ndarray: The real coefficient.
        """
        f_scaled = f - layer_m * self.layer_df
        _c_nm = self._linear_interp(f_scaled, fdot, self.c_nm_all, layer_n)
        _s_nm = self._linear_interp(f_scaled, fdot, self.s_nm_all, layer_n)

        # N_REF_ONLY parity sign on (layer_n - n_ref).
        if self.kind == LOOKUP_N_REF_ONLY:
            dn_odd = ((layer_n - self.n_ref) & 1) != 0
            _c_nm = jnp.where(dn_odd, -_c_nm, _c_nm)
            _s_nm = jnp.where(dn_odd, -_s_nm, _s_nm)

        # (m+n)-parity sin/cos swap.
        is_m_plus_n_even = ((layer_m + layer_n) % 2 == 0)
        s_nm = jnp.where(is_m_plus_n_even, _c_nm, _s_nm)
        c_nm = jnp.where(is_m_plus_n_even, _s_nm, _c_nm)

        # w_mn = c_nm * M'.real + s_nm * M'.imag.
        w_mn = c_nm * tdi_channel_val.real + s_nm * tdi_channel_val.imag

        # m-parity flip on (layer_m - m_ref).
        dm_odd = ((layer_m - self.m_ref) & 1) != 0
        w_mn = jnp.where(dm_odd, -w_mn, w_mn)
        return w_mn

    def get_wdm_in_channel_over_layers(self, tdi_channel_val: jnp.ndarray,
                                       f: jnp.ndarray, fdot: jnp.ndarray,
                                       m: jnp.ndarray, n: jnp.ndarray) -> jnp.ndarray:
        """Mirrors ``WaveletLookupTable::get_wdm_in_channel_over_layers``.

        :meth:`get_w_mn_lookup` with a band guard: returns 0 if ``m`` is
        outside ``[ind_min_f, ind_max_f)`` (upper end exclusive) to match the
        C++ guard.

        Args:
            tdi_channel_val: Complex ``M'`` value.
            f: Instantaneous frequency [Hz].
            fdot: Frequency derivative [Hz/s].
            m: Frequency layer.
            n: Time pixel.

        Returns:
            jnp.ndarray: The real coefficient (0 outside the band).
        """
        in_band = (m >= self.ind_min_f) & (m < self.ind_max_f)
        w_mn = self.get_w_mn_lookup(tdi_channel_val, f, fdot, m, n)
        return jnp.where(in_band, w_mn, 0.0)
