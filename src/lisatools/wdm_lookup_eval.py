# src/lisatools/wdm_lookup_eval.py
"""Vectorized evaluation of an ``n_ref`` WDM lookup table on numpy or cupy arrays.

The table (:class:`lisatools.domains.WDMLookupTable`, build kinds ``n_ref_only`` /
``n_ref_complex``) stores the WDM response at one reference pixel ``(m_ref, n_ref)`` of a unit
linear chirp as a function of the carrier's offset inside its layer (``f_vals_norm``, uniform,
support ``[-(k+1), k+1)`` layers for ``num_layers_diff = k``) and its chirp rate (``fdot_vals``,
uniform). This module evaluates
that table for arrays of pixels in one shot -- the vectorized twin of
``WDMLookupTable.get_wdm_coeffs`` (whose per-layer Python loop and scipy interpolators are the
EMRI reference path, docs/emri-direct-wdm.md) -- with the same evaluation rule
(``BASIS_CYCLE = "quarter_turn"``):

1. ``delta = f - m * layer_df``; read ``c`` (cos response) and ``s`` (sin response) at
   ``(fdot, delta)``, ``s`` from the seam-continuous table (the build's ``(-1)^block`` bake
   undone at the NODES, never per pixel);
2. if ``(m_ref + n_ref)`` is odd: ``c -> -c``;
3. if the absolute pixel parity ``(m + n)`` is odd: ``(c, s) -> (s, -c)``;
4. ``w[m, n] = amp * (c cos(phi) - s sin(phi))``, ``phi`` the carrier phase at the pixel centre
   ``t_n = n * layer_dt`` (signal ``amp * cos(phi)``, ``phi`` increasing in time).

Interpolation on the uniform ``(fdot, f_norm)`` grid is the table's uniform cubic B-spline
(``"spline"``, default: coefficients prefiltered once with mirror boundaries, then 16 gathers with
B-spline weights == ``scipy.ndimage.map_coordinates(order=3, mode="mirror", prefilter=False)``,
the semantics of ``WDMLookupTable`` ``INTERP_METHOD = "spline"`` and of the fused C++/CUDA lookup
kernels), Keys cubic convolution (``"cubic"``, a = -1/2) or bilinear (``"linear"``), all written
with array-module gathers so the same code runs on numpy and cupy. Linear interpolation across
the table's peaked response left a 2.5e-4 amplitude deficit on the EMRI; on the SOBBH 6-month
gate the B-spline is equal to or better than Keys (two of six sources halve their mismatch).
Entries outside the table support are zero (and flagged).

Used by the SOBBH direct-to-WDM lookup likelihood (:mod:`lisatools.sources.sobbh.wdm_direct`,
``SOBBH_LIKELIHOOD=lookup`` on the 9-month / 12-month runs) and the GB lookup engine
(``gbgpu.gblookupcomputations``).
"""

from __future__ import annotations

import numpy as np

from .utils.utility import asnumpy


def _bspline_weights(t):
    """Uniform cubic B-spline weights for the coefficients at offsets -1, 0, 1, 2; ``t`` in
    [0, 1)."""
    t2 = t * t
    t3 = t2 * t
    s = 1.0 - t
    return (
        s * s * s / 6.0,
        (3.0 * t3 - 6.0 * t2 + 4.0) / 6.0,
        (-3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0) / 6.0,
        t3 / 6.0,
    )


def _mirror_index(xp, i, n):
    """``scipy.ndimage`` ``mode="mirror"`` index (whole-sample symmetric about 0 and ``n - 1``).

    The taps of an in-support point lie in ``[-1, n + 1]``, so ONE reflection is exact for
    ``n >= 3``; the final clip only keeps out-of-support points (masked by the caller) inside
    the array."""
    if n == 1:
        return xp.zeros_like(i)
    i = xp.where(i < 0, -i, i)
    i = xp.where(i > n - 1, 2 * (n - 1) - i, i)
    return xp.clip(i, 0, n - 1)


def _keys_weights(t):
    """Catmull-Rom (Keys, a = -1/2) weights for the nodes at offsets -1, 0, 1, 2; ``t`` in
    [0, 1)."""
    t2 = t * t
    t3 = t2 * t
    return (
        0.5 * (-t3 + 2.0 * t2 - t),
        0.5 * (3.0 * t3 - 5.0 * t2 + 2.0),
        0.5 * (-3.0 * t3 + 4.0 * t2 + t),
        0.5 * (t3 - t2),
    )


class WDMLookupEvaluator:
    """Vectorized ``n_ref`` lookup-table evaluation (see the module docstring for the rule).

    Args:
        table: a built or loaded :class:`~lisatools.domains.WDMLookupTable` of build kind
            ``n_ref_only`` or ``n_ref_complex``.
        interp: ``"spline"`` (default: the table's uniform cubic B-spline, prefiltered with
            mirror boundaries == ``WDMLookupTable`` ``INTERP_METHOD = "spline"``),
            ``"cubic"`` (Keys cubic convolution) or ``"linear"`` (bilinear).
        force_backend: backend NAME the table arrays are placed on (``"cpu"``, ``"cuda12x"``,
            ...); ``None`` keeps the table's own backend. Chosen here, never per call. A
            backend object is also accepted (its name is used).

    Raises:
        ValueError: For another table kind, an unknown ``interp``, or a non-uniform table
            axis.

    Attributes:
        layer_df (float): Layer spacing of the table grid (Hz).
        layer_dt (float): Pixel duration of the table grid (s).
        m_ref (int): Reference layer of the table build.
        n_ref (int): Reference pixel of the table build.
        f_min, f_max, df (float): The offset axis (Hz).
        fdot_min, fdot_max, dfdot (float): The fdot axis (Hz/s; ``dfdot = 1`` for a
            one-node axis).
        tab_cos, tab_sin: The cos / seam-continuous sin tables on the backend (B-spline
            coefficients for ``"spline"``).

    Example:
        .. code-block:: python

            from lisatools.domains import WDMLookupTable
            from lisatools.wdm_lookup_eval import WDMLookupEvaluator

            ev = WDMLookupEvaluator(WDMLookupTable.from_file(path), interp="spline",
                                    force_backend="cpu")
            m = ev.layers_for(f, num_m_layers=2)              # (..., 5) target layers
            w, ok = ev.coeffs(amp[..., None], phi[..., None], f[..., None],
                              fdot[..., None], n[..., None], m)
    """

    INTERPS = ("linear", "cubic", "spline")
    #: ``"no_parity_turn"`` skips step 3 of the rule. NEGATIVE CONTROL ONLY (tests).
    BASIS_CYCLES = ("quarter_turn", "no_parity_turn")

    def __init__(self, table, interp="spline", force_backend=None):
        kind = getattr(table, "build_kind", None)
        if kind not in ("n_ref_only", "n_ref_complex"):
            raise ValueError(
                f"WDMLookupEvaluator needs an n_ref_only / n_ref_complex table, got {kind!r}"
            )
        if interp not in self.INTERPS:
            raise ValueError(f"interp must be one of {self.INTERPS}, got {interp!r}")
        self.interp = interp
        self.basis_cycle = "quarter_turn"
        if force_backend is None:
            self._backend_name = table.backend.name.split("_")[-1]
        elif isinstance(force_backend, str):
            self._backend_name = force_backend
        else:
            self._backend_name = force_backend.name.split("_")[-1]

        f_norm = np.asarray(asnumpy(table.f_vals_norm), dtype=float).ravel()
        fdot = np.asarray(asnumpy(table.fdot_vals), dtype=float).ravel()
        if kind == "n_ref_complex":
            cx = np.asarray(asnumpy(table.table_cx))
            tab_cos, tab_sin = np.real(cx), np.imag(cx)
        else:
            tab_cos = np.asarray(asnumpy(table.table_cos), dtype=float)
            tab_sin = np.asarray(asnumpy(table.table_sin), dtype=float)
        tab_cos = np.ascontiguousarray(tab_cos.reshape(fdot.size, f_norm.size), dtype=float)
        tab_sin = np.ascontiguousarray(tab_sin.reshape(fdot.size, f_norm.size), dtype=float)

        self.layer_df = float(table.layer_df)
        self.layer_dt = float(table.layer_dt)
        self.m_ref = int(table.m_ref)
        self.n_ref = int(table.n_ref)
        # undo the build's (-1)^block bake on the chirp (sin) term at the NODES so the stored
        # function is continuous across block seams (WDMLookupTable._sin_unbaked_coeffs)
        block = np.floor(f_norm / self.layer_df + 1e-9).astype(int)
        tab_sin = tab_sin * np.where(block % 2 != 0, -1.0, 1.0)[None, :]

        df = float(f_norm[1] - f_norm[0])
        if not np.allclose(np.diff(f_norm), df, rtol=1e-6, atol=0.0):
            raise ValueError("table f_vals_norm axis is not uniform")
        self.nf = int(f_norm.size)
        self.f_min, self.f_max, self.df = float(f_norm[0]), float(f_norm[-1]), df
        self.nfdot = int(fdot.size)
        if self.nfdot > 1:
            dfd = float(fdot[1] - fdot[0])
            if not np.allclose(np.diff(fdot), dfd, rtol=1e-6, atol=0.0):
                raise ValueError("table fdot_vals axis is not uniform")
        else:
            dfd = 1.0
        self.fdot_min, self.fdot_max, self.dfdot = float(fdot[0]), float(fdot[-1]), dfd

        if interp == "spline":
            # the table's own uniform cubic B-spline (WDMLookupTable INTERP_METHOD "spline",
            # _UniformCubicSpline): coefficients prefiltered ONCE with mirror boundaries, here
            # on the host (scipy), then evaluated with explicit weights + mirrored indices,
            # == ndimage.map_coordinates(order=3, mode="mirror", prefilter=False)
            from scipy import ndimage

            if self.nfdot > 1:
                tab_cos = ndimage.spline_filter(tab_cos, order=3, mode="mirror")
                tab_sin = ndimage.spline_filter(tab_sin, order=3, mode="mirror")
            else:
                tab_cos = ndimage.spline_filter1d(tab_cos, order=3, axis=1, mode="mirror")
                tab_sin = ndimage.spline_filter1d(tab_sin, order=3, axis=1, mode="mirror")
        xp = self.xp
        self.tab_cos = xp.asarray(tab_cos)
        self.tab_sin = xp.asarray(tab_sin)

    @property
    def backend(self):
        """The lisatools backend the tables live on (resolved from its name on each access)."""
        import lisatools

        return lisatools.get_backend(self._backend_name)

    @property
    def xp(self):
        """Array module of :attr:`backend` (numpy or cupy)."""
        return self.backend.xp

    def layers_for(self, f_ref, num_m_layers):
        """Target layers ``floor(f_ref / layer_df) + [-k .. k]`` -> int array ``(..., 2k + 1)``.

        Purely geometric: for ``f_ref < k * layer_df`` this can return negative layers. The
        caller owns the valid layer-range mask for its own grid (this evaluator is built from
        one table and has no notion of any particular caller's ``Nf``).

        Args:
            f_ref: Reference frequencies (Hz), any shape.
            num_m_layers: ``k``, layers on each side.

        Returns:
            ``xp.ndarray``: int64 layers of shape ``f_ref.shape + (2k + 1,)``.
        """
        xp = self.xp
        k = int(num_m_layers)
        m0 = xp.floor(xp.asarray(f_ref, dtype=float) / self.layer_df).astype(xp.int64)
        return m0[..., None] + xp.arange(-k, k + 1, dtype=xp.int64)

    def _interp(self, tab, u, v):
        """Interpolate ``tab[j, i]`` at fractional indices ``(v, u)`` (fdot, f_norm).

        Bilinear (clipped), uniform cubic B-spline on mirrored indices, or Keys cubic
        (clipped indices), per :attr:`interp`.
        """
        xp = self.xp
        nf, nfd = self.nf, self.nfdot
        i0 = xp.floor(u).astype(xp.int64)
        j0 = xp.floor(v).astype(xp.int64)
        if self.interp == "linear":
            i0c = xp.clip(i0, 0, max(nf - 2, 0))
            j0c = xp.clip(j0, 0, max(nfd - 2, 0))
            tu = u - i0c
            tv = v - j0c
            i1 = xp.minimum(i0c + 1, nf - 1)
            j1 = xp.minimum(j0c + 1, nfd - 1)
            z00 = tab[j0c, i0c]
            z01 = tab[j0c, i1]
            z10 = tab[j1, i0c]
            z11 = tab[j1, i1]
            return (1.0 - tv) * ((1.0 - tu) * z00 + tu * z01) + tv * ((1.0 - tu) * z10 + tu * z11)
        if self.interp == "spline":
            wu = _bspline_weights(u - i0)
            wv = _bspline_weights(v - j0)

            def idx_u(off):
                return _mirror_index(xp, i0 + off, nf)

            def idx_v(off):
                return _mirror_index(xp, j0 + off, nfd)

        else:
            wu = _keys_weights(u - i0)
            wv = _keys_weights(v - j0)

            def idx_u(off):
                return xp.clip(i0 + off, 0, nf - 1)

            def idx_v(off):
                return xp.clip(j0 + off, 0, nfd - 1)

        out = None
        for a in range(4):
            ja = idx_v(a - 1)
            row = None
            for b in range(4):
                ib = idx_u(b - 1)
                term = wu[b] * tab[ja, ib]
                row = term if row is None else row + term
            term = wv[a] * row
            out = term if out is None else out + term
        return out

    def coeffs(self, amp, phi, f, fdot, n, m):
        """WDM coefficients of carriers ``amp cos(phi)`` at pixels ``(m, n)``.

        All arguments broadcast together (``m`` and ``n`` integer-valued). Returns ``(w, ok)``:
        ``w`` the coefficient (zero where ``ok`` is False) and ``ok`` the in-support mask
        (offset ``f - m * layer_df`` inside the table's offset axis and, for a table with an
        fdot axis, ``fdot`` inside that axis). ``ok`` does not check that ``m`` itself is a
        valid layer on any particular grid (e.g. ``0 <= m < Nf``) -- this evaluator does not
        know the calling grid's ``Nf``, so that mask is the caller's responsibility.

        Args:
            amp: Carrier amplitudes.
            phi: Carrier phases at the pixel times ``n * layer_dt`` (rad).
            f: Instantaneous frequencies (Hz).
            fdot: Frequency derivatives (Hz/s).
            n: Absolute time-pixel indices.
            m: Absolute layer indices (e.g. from :meth:`layers_for`).

        Returns:
            tuple: ``(w, ok)``, broadcast shape; ``w`` float, ``ok`` bool.
        """
        xp = self.xp
        amp, phi, f, fdot, n, m = xp.broadcast_arrays(
            *(xp.asarray(a) for a in (amp, phi, f, fdot, n, m))
        )
        m = m.astype(xp.int64)
        n = n.astype(xp.int64)
        delta = f - m * self.layer_df
        ok = (delta >= self.f_min) & (delta <= self.f_max)
        u = (delta - self.f_min) / self.df
        if self.nfdot > 1:
            ok = ok & (fdot >= self.fdot_min) & (fdot <= self.fdot_max)
            v = (fdot - self.fdot_min) / self.dfdot
        else:
            v = xp.zeros_like(u)
        c = self._interp(self.tab_cos, u, v)
        s = self._interp(self.tab_sin, u, v)
        if (self.m_ref + self.n_ref) % 2:
            c = -c
        if self.basis_cycle == "quarter_turn":
            odd = ((m + n) % 2) != 0  # ABSOLUTE pixel parity
            cc = xp.where(odd, s, c)
            ss = xp.where(odd, -c, s)
        else:  # negative control: no quarter turn
            cc, ss = c, s
        w = amp * (cc * xp.cos(phi) - ss * xp.sin(phi))
        return xp.where(ok, w, 0.0), ok


__all__ = ["WDMLookupEvaluator"]
