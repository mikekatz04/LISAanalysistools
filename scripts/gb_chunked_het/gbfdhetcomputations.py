"""GBFDHetComputations -- frequency-domain GB likelihood built on the SAME
waveform the WDM (chunked-het / signal-het) paths start with: the full rfft of
the ``GBTDIonTheFly`` TD-on-the-fly signal. Because it is the identical waveform
and grid, its FD inner products reproduce the WDM-domain mismatch
(``mm ~ 1e-11`` at injection), unlike ``gbgpu.gbcomps.GBFDComputations`` which
uses GBGPU's separate fast-FD waveform (``mm ~ 1e-4`` vs mojito).

It mirrors the method surface of the other GB computations classes so it is a
drop-in alongside ``GBWDMComputations`` / ``GBSignalHetComputations``:

  ``get_ll`` / ``get_ll_fd``  -- FD log-likelihood for candidate params (batched)
  ``fill_global``            -- scatter the FD template into a global FD buffer
  ``get_swap_ll``            -- the 5 swap inner products (add/remove)

The template is the full rfft of the GB TD on the DATA grid (no heterodyne
carrier, no sparse-time approximation), so the GB phase reference ``t_ref`` is
used exactly as in the WDM paths (``t_ref = REF`` for mojito) -- there is no
``t_start == t_ref`` constraint and no parameter re-referencing.

Inner-product convention (matches lisatools / GBFDComputations):
``(a|b) = 4 df Re sum_{c,c',k in band} conj(a_c[k]) invC_{cc'}[k] b_{c'}[k]``.
"""
from __future__ import annotations

import numpy as np
from scipy.signal.windows import tukey as _tukey

from lisatools.domains import FDSettings
from lisatools.sensitivity import XYZ2SensitivityMatrix
from lisatools.response.tdiconfig import TDIConfig
from lisatools.response.tdionfly import GBTDIonTheFly


class GBFDHetComputations:
    """Dense-FD GB likelihood on the GBTDIonTheFly waveform.

    Args:
        data_td (np.ndarray): ``(3, Nf*Nt)`` time-domain TDI data.
        ref_params (np.ndarray): length-9 reference params (kept for API
            symmetry with ``GBSignalHetComputations``; not used by the dense
            FD path, which needs no heterodyne reference).
        Nf, Nt, dt, t0, t_ref: WDM grid / data sampling + GB phase reference.
        orbits, tdi_config: response orbits + TDI config (str or TDIConfig).
        min_freq, max_freq: active FD band [Hz] for the inner product.
        sens_model, tukey_alpha, n_nodes, batch, force_backend, tdi_type:
            as in ``GBSignalHetComputations`` (defaults match).
    """

    def __init__(self, data_td, ref_params, *, Nf, Nt, dt, t0, t_ref,
                 orbits, tdi_config, min_freq, max_freq, sens_model="scirdv1",
                 tukey_alpha=0.05, edge_cut=0, n_nodes=16384, batch=8,
                 force_backend="cpu", tdi_type="XYZ"):
        if tdi_type != "XYZ":
            raise ValueError("GBFDHetComputations currently supports tdi_type='XYZ'.")
        self.ref_params = np.asarray(ref_params, dtype=float).reshape(9)
        self.Nf = int(Nf); self.Nt = int(Nt); self.dt = float(dt)
        self.tdi_type = tdi_type; self.batch = int(batch)
        Nobs = self.Nf * self.Nt; self.Nobs = Nobs
        self.Tobs = Nobs * self.dt; self.df = 1.0 / self.Tobs
        self.n_rfft = Nobs // 2 + 1
        self.t0 = float(t0); self.t_ref = float(t_ref)
        self.t_arr = np.arange(Nobs) * self.dt + self.t0
        # Time window applied to BOTH data and templates: tukey, optionally x an
        # edge-cut boxcar that restricts to [edge_cut, Nt-edge_cut] layers (the
        # WDM grid's active time region). edge_cut=0 -> full Tobs.
        self.window = (_tukey(Nobs, tukey_alpha).astype(float) if tukey_alpha > 0
                       else np.ones(Nobs))
        self.edge_cut = int(edge_cut)
        if self.edge_cut > 0:
            ec = np.zeros(Nobs); ec[self.edge_cut * self.Nf:(self.Nt - self.edge_cut) * self.Nf] = 1.0
            self.window = self.window * ec
        self._t_nodes = np.linspace(self.t_arr[0], self.t_arr[-1], n_nodes)

        if isinstance(tdi_config, str):
            tdi_config = TDIConfig(tdi_config, force_backend=force_backend)
        self._tdi_config = tdi_config; self._orbits = orbits
        self._backend = force_backend

        # --- data FD (full grid) on the SAME window the WDM data uses --------
        self.data_fd = np.ascontiguousarray(
            np.fft.rfft(np.asarray(data_td)[:3] * self.window[None, :], axis=-1) * self.dt)
        self.nch = self.data_fd.shape[0]

        # --- inverse noise covariance (full grid, DC bin zeroed) ------------
        iC = np.asarray(XYZ2SensitivityMatrix(
            FDSettings(self.n_rfft, self.df, force_backend=force_backend),
            model=sens_model).invC).copy()
        iC = np.where(np.isfinite(iC), iC, 0.0); iC[:, :, 0] = 0.0
        self.invC = iC

        # --- active band + <d|d> --------------------------------------------
        self.ind_min = max(1, int(min_freq / self.df))
        self.ind_max = min(self.n_rfft - 1, int(max_freq / self.df))
        self._sl = slice(self.ind_min, self.ind_max + 1)
        self.d_d = self._dd()

        self._keep_alive = dict(orbits=orbits, tdi_config=tdi_config)
        self.d_h_out = None; self.h_h_out = None

    # ------------------------------------------------------------------
    def _dd(self):
        sl = self._sl
        return 4.0 * self.df * float(np.real(np.einsum(
            "ck,cdk,dk->", np.conj(self.data_fd[:, sl]), self.invC[:, :, sl],
            self.data_fd[:, sl])))

    def _make_gb(self, num_bin):
        return GBTDIonTheFly(self._t_nodes, self.Tobs, self.t_ref, 1.0 / self.dt,
                             num_bin, tdi_config=self._tdi_config, orbits=self._orbits,
                             tdi_chan="XYZ", force_backend=self._backend)

    def _iter_fd(self, params, convert_to_ra_dec=False):
        """Yield ``(start, end, h_chunk)`` with ``h_chunk`` the full-grid FD
        templates ``(chunk, nch, n_rfft)``, processed in ``batch``-sized chunks
        to bound memory (the full grid is large)."""
        x = np.atleast_2d(np.asarray(params, dtype=float))
        if x.shape[1] != 9:
            raise ValueError(f"params must be (N, 9); got {x.shape}")
        N = x.shape[0]
        for s in range(0, N, self.batch):
            e = min(s + self.batch, N); c = e - s
            gb = self._make_gb(c)
            sp = gb(*[x[s:e, j].copy() for j in range(9)],
                    convert_to_ra_dec=convert_to_ra_dec, return_spline=True)
            td = np.asarray(sp.eval_tdi(self.t_arr))
            if td.ndim == 2:
                td = td[None]
            h = np.fft.rfft(td * self.window[None, None, :], axis=-1) * self.dt
            yield s, e, h

    # ------------------------------------------------------------------
    def get_ll(self, params, convert_to_ra_dec=False):
        """FD logL for candidate ``params`` (length-9 or ``(N, 9)``)."""
        N = np.atleast_2d(np.asarray(params, dtype=float)).shape[0]
        d_h = np.empty(N); h_h = np.empty(N)
        sl = self._sl; iC = self.invC[:, :, sl]; dsl = np.conj(self.data_fd[:, sl])
        for s, e, h in self._iter_fd(params, convert_to_ra_dec):
            hb = h[:, :, sl]
            d_h[s:e] = 4.0 * self.df * np.real(np.einsum("ck,cdk,ndk->n", dsl, iC, hb))
            h_h[s:e] = 4.0 * self.df * np.real(np.einsum("nck,cdk,ndk->n",
                                                         np.conj(hb), iC, hb))
        self.d_h_out = d_h; self.h_h_out = h_h
        return -0.5 * self.d_d + d_h - 0.5 * h_h

    def get_ll_fd(self, params, **kw):  # name parity with GBFDComputations
        return self.get_ll(params, **kw)

    def fill_global(self, params, templates, data_index=None, factors=None,
                    convert_to_ra_dec=False):
        """Scatter the FD templates into ``templates`` (num_templates, nch,
        n_rfft) complex: ``templates[data_index[b]] += factors[b] * h_b``."""
        x = np.atleast_2d(np.asarray(params, dtype=float)); N = x.shape[0]
        if data_index is None:
            data_index = np.zeros(N, dtype=int)
        if factors is None:
            factors = np.ones(N)
        for s, e, h in self._iter_fd(params, convert_to_ra_dec):
            for b in range(s, e):
                templates[int(data_index[b])] += factors[b] * h[b - s]

    def get_swap_ll(self, params_add, params_remove, convert_to_ra_dec=False):
        """5 swap inner products on the FD templates: returns
        ``(like_add, like_remove, d_h_add, d_h_remove, hh_add, hh_remove, hh_cross)``."""
        sl = self._sl; iC = self.invC[:, :, sl]; dsl = np.conj(self.data_fd[:, sl])
        pa = np.atleast_2d(np.asarray(params_add, float))
        pr = np.atleast_2d(np.asarray(params_remove, float))
        N = pa.shape[0]
        ha = np.empty((N, self.nch, sl.stop - sl.start), dtype=complex)
        hr = np.empty_like(ha)
        for s, e, h in self._iter_fd(pa, convert_to_ra_dec):
            ha[s:e] = h[:, :, sl]
        for s, e, h in self._iter_fd(pr, convert_to_ra_dec):
            hr[s:e] = h[:, :, sl]
        k = 4.0 * self.df
        d_h_a = k * np.real(np.einsum("ck,cdk,ndk->n", dsl, iC, ha))
        d_h_r = k * np.real(np.einsum("ck,cdk,ndk->n", dsl, iC, hr))
        aa = k * np.real(np.einsum("nck,cdk,ndk->n", np.conj(ha), iC, ha))
        rr = k * np.real(np.einsum("nck,cdk,ndk->n", np.conj(hr), iC, hr))
        ar = k * np.real(np.einsum("nck,cdk,ndk->n", np.conj(ha), iC, hr))
        like_add = -0.5 * (self.d_d + aa - 2.0 * d_h_a)
        like_rem = -0.5 * (self.d_d + rr - 2.0 * d_h_r)
        return like_add, like_rem, d_h_a, d_h_r, aa, rr, ar
