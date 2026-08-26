"""Warm-start GB RJ-birth proposal from a previous run's clustered posterior.

Workstream B of the warm-start pipeline (``docs/warm-start-gb-proposal.md``,
``docs/6mo-run-prep.md``): the offline fitter
``scripts/gb/warmstart_fit_from_store.py`` turns a finished run's cold-chain
GB leaf table into Gaussian components (one per recovered source mode) with
inclusion probabilities ``p``; :class:`WarmStartComponents` loads that npz and
serves it as an eryn duck-typed RJ birth distribution over the FULL 9-column
sampled GB basis::

    0 dist [kpc], 1 f0 [mHz], 2 Mc [Msun], 3 phi0 [rad, 2pi), 4 cos_iota,
    5 psi [rad, pi), 6 alpha [rad, 2pi), 7 sin_delta, 8 fdot_astro_ratio

``rvs(size) -> (size, 9)`` and ``logpdf((n, 9)) -> (n,)`` are mutually
consistent BY CONSTRUCTION (the same mixture), which is what the RJ
Metropolis-Hastings factors require. Key design points:

* **Weights ~ p** (renormalized; optional ``p_floor``): the inclusion
  probability from the previous posterior IS the birth weight -- the
  warm start's value over the F-stat proposal. ``mult`` (leaf multiplicity)
  is loaded but IGNORED for weighting (PROVISIONAL v1 policy; a dup-detector
  diagnostic only).
* **Cholesky draws**: ``x = mean + L z`` with ``L = cholesky(cov)`` --
  never ``mean + cov @ z`` (the historical gmm.py rvs covariance bug).
* **Circular columns** (phi0/alpha period 2pi, psi period pi) are WRAPPED
  into ``[0, period)`` on draw; ``logpdf`` evaluates the Gaussian at the
  MINIMAL-IMAGE displacement ``d - period * round(d / period)``. This is the
  nearest-image approximation to the wrapped normal (summing the +-1 period
  images is overkill when ``sigma << period``); it is checked at load and a
  warning is issued for components whose circular sigma exceeds
  ``period / 6`` (for those, near-antipodal densities are underestimated --
  the proposal remains a valid, mutually-consistent MH proposal up to that
  density approximation, which is shared by rvs' images too).
* **f0 windowing is a SEARCH structure only** -- NOT a truncation of the
  mixture. ``rvs`` draws from the full mixture (all components, full mass);
  ``logpdf`` sums over the components whose f0 candidate window
  ``|f0 - mu_f0| <= window_df * df + 10 * sigma_f0`` covers the query point.
  Bound: an omitted component is >= 10 sigma away in f0 ALONE, so its
  density at the point is <= exp(-50) ~ 1.9e-22 of its own peak -- far
  below the 2.2e-16 double-precision resolution of the retained logsumexp
  whenever any component (or the uniform floor) is retained. The windowed
  logpdf therefore equals the full-mixture logpdf to machine precision
  wherever the density is not already negligible. ``window_df`` defaults to
  ``GB_WARM_START_F0_WINDOW_DF`` (10).
* **Uniform floor** (``floor_box`` + ``floor_eps``): BandSorter evaluates
  ``logpdf`` at EXISTING leaves for RJ death factors and multiplies by
  inds masks -- a ``-inf`` there produces NaN factors (same reason
  ``UniformFloorMixture`` is mandatory for the F-stat container). With a
  floor the density is ``(1 - eps) * mixture + eps * Uniform(box)``,
  finite everywhere inside the 9-D box. ``floor_eps=0`` (default) keeps the
  bare mixture (standalone/testing use).
* **Cross-Tobs v1 policy** (``docs/warm-start-gb-proposal.md``): components
  are proposed at their STORED widths -- conservative, wider = safer for MH;
  NO Fisher ``T``-rescale. Only the f0 candidate windows are re-derived
  against the NEW run's ``df = 1/new_tobs``.

Pickle/deepcopy safe (sprint rule): master tables are numpy; the cupy
device cache is dropped on ``__getstate__``.
"""

from __future__ import annotations

import json
import logging
import os
import typing
import warnings

import numpy as np

logger = logging.getLogger(__name__)

# The sampled GB basis the fitter writes (scripts/gb/warmstart_fit_from_store
# .py COLUMN_NAMES / CIRCULAR_COLS -- kept in lockstep by the npz meta check).
COLUMN_NAMES = [
    "dist", "f0", "Mc", "phi0", "cos_iota", "psi", "alpha",
    "sin_delta", "fdot_astro_ratio",
]
CIRCULAR_COLS = {3: 2.0 * np.pi, 5: np.pi, 6: 2.0 * np.pi}

#: default logpdf candidate window, in units of df = 1/Tobs (mHz)
DEFAULT_F0_WINDOW_DF = 10.0
#: per-component sigma guard added to the candidate window half-width; the
#: negligibility bound quoted in the module docstring is exp(-GUARD^2/2).
F0_WINDOW_GUARD_NSIGMA = 10.0
#: circular-sigma sanity threshold (fraction of the period) for the
#: minimal-image approximation warning.
CIRC_SIGMA_WARN_FRAC = 1.0 / 6.0


def _host(x):
    return x.get() if hasattr(x, "get") else np.asarray(x)


class WarmStartComponents:
    """f0-windowed Gaussian-mixture RJ birth proposal over the 9-col GB basis.

    Args:
        means: ``(n, 9)`` component means in the sampled basis (f0 in mHz).
        covs: ``(n, 9, 9)`` covariances (must be positive definite; the
            fitter's eigenvalue floors guarantee it).
        p: ``(n,)`` inclusion probabilities; mixture weights are
            ``max(p, p_floor)`` renormalized.
        new_tobs: THE NEW RUN's observation time [s]; f0 candidate windows
            are derived from ``df = 1/new_tobs`` (cross-Tobs v1 policy --
            widths themselves stay as stored).
        window_df: candidate window half-width in units of df. ``None`` ->
            ``GB_WARM_START_F0_WINDOW_DF`` env (default 10).
        p_floor: optional floor applied to ``p`` before renormalizing.
        floor_box: ``(lo, hi)`` 9-vector pair bounding the uniform floor
            component (normally the run's prior box). ``None`` with
            ``floor_eps > 0`` raises.
        floor_eps: uniform-floor mixture weight (0 = bare mixture).
        mult, n_members, island_id, f0_window_edges, meta: carried through
            from the npz for diagnostics; unused by rvs/logpdf (``mult`` is
            deliberately NOT a weight -- PROVISIONAL v1 policy).
        use_cupy: keep a device copy of the tables and return device draws
            (GPU runs). The master copy stays numpy (pickle rule).
        seed: RNG seed for :meth:`rvs`.
    """

    ndim = 9

    def __init__(self, means, covs, p, *, new_tobs, window_df=None,
                 p_floor: float = 0.0, floor_box=None, floor_eps: float = 0.0,
                 mult=None, n_members=None, island_id=None,
                 f0_window_edges=None, meta: typing.Optional[dict] = None,
                 use_cupy: bool = False, seed: typing.Optional[int] = None):
        means = np.array(means, dtype=np.float64, copy=True)
        covs = np.array(covs, dtype=np.float64, copy=True)
        p = np.asarray(p, dtype=np.float64).ravel()
        if means.ndim != 2 or means.shape[1] != self.ndim:
            raise ValueError(
                f"means must be (n, {self.ndim}); got {means.shape}."
            )
        n = means.shape[0]
        if n == 0:
            raise ValueError("WarmStartComponents needs >= 1 component.")
        if covs.shape != (n, self.ndim, self.ndim):
            raise ValueError(
                f"covs must be (n, {self.ndim}, {self.ndim}); got {covs.shape}."
            )
        if p.shape != (n,) or np.any(p <= 0):
            raise ValueError("p must be (n,) and strictly positive.")

        # wrap the stored circular means into their fundamental domain so the
        # minimal-image displacement in logpdf is measured from a canonical
        # representative (draws are wrapped the same way).
        for c, period in CIRCULAR_COLS.items():
            means[:, c] = means[:, c] % period

        # --- Gaussian machinery -----------------------------------------
        # Cholesky draws: x = mean + L z. NEVER mean + cov @ z (the gmm.py
        # rvs covariance bug). LinAlgError here = a non-PD component, which
        # the schema forbids.
        chol = np.linalg.cholesky(covs)
        eye = np.eye(self.ndim)
        # L^{-1} per component (9x9 solves; n is O(10^3) -- trivial).
        chol_inv = np.linalg.solve(chol, np.broadcast_to(eye, covs.shape))
        logdet = 2.0 * np.sum(np.log(np.diagonal(chol, axis1=1, axis2=2)),
                              axis=1)
        log_norm = -0.5 * (self.ndim * np.log(2.0 * np.pi) + logdet)

        # --- weights ~ p (renormalized; optional floor) ------------------
        w = np.maximum(p, float(p_floor))
        self.weights = w / w.sum()
        log_w = np.log(self.weights)

        # --- f0 candidate windows vs the NEW run's df --------------------
        self.tobs = float(new_tobs)
        if self.tobs <= 0:
            raise ValueError(f"new_tobs must be positive; got {new_tobs}.")
        self.df_mhz = 1e3 / self.tobs  # stored f0 is mHz
        if window_df is None:
            window_df = float(os.environ.get(
                "GB_WARM_START_F0_WINDOW_DF", str(DEFAULT_F0_WINDOW_DF)))
        self.window_df = float(window_df)
        sigma_f0 = np.sqrt(covs[:, 1, 1])
        # SEARCH structure only: the guard keeps every omitted component
        # >= GUARD sigma away in f0, making the windowed logpdf equal to the
        # full-mixture logpdf to machine precision (module docstring bound).
        self.window_halfwidth_mhz = (
            self.window_df * self.df_mhz
            + F0_WINDOW_GUARD_NSIGMA * sigma_f0
        )
        win_lo = means[:, 1] - self.window_halfwidth_mhz
        win_hi = means[:, 1] + self.window_halfwidth_mhz
        order = np.argsort(win_lo, kind="stable")
        lo_sorted = win_lo[order]
        hi_sorted_all = np.sort(win_hi)
        depth = np.arange(1, n + 1) - np.searchsorted(
            hi_sorted_all, lo_sorted, side="left")
        self._overlap_depth = int(max(1, depth.max()))

        # --- circular-sigma sanity (minimal-image approximation) ---------
        bad = 0
        for c, period in CIRCULAR_COLS.items():
            sig = np.sqrt(covs[:, c, c])
            bad += int(np.sum(sig > CIRC_SIGMA_WARN_FRAC * period))
        if bad:
            warnings.warn(
                f"WarmStartComponents: {bad} circular (column, component) "
                f"pairs have sigma > period/6; the minimal-image logpdf "
                f"underestimates their near-antipodal density (wrapped-"
                f"normal tail images neglected). Proposal remains usable; "
                f"see module docstring.",
                RuntimeWarning, stacklevel=2,
            )

        # --- uniform floor ------------------------------------------------
        self.floor_eps = float(floor_eps)
        if self.floor_eps < 0 or self.floor_eps >= 1:
            raise ValueError(f"floor_eps must be in [0, 1); got {floor_eps}.")
        if self.floor_eps > 0:
            if floor_box is None:
                raise ValueError("floor_eps > 0 requires floor_box=(lo, hi).")
            lo_b = np.asarray(floor_box[0], dtype=np.float64).ravel()
            hi_b = np.asarray(floor_box[1], dtype=np.float64).ravel()
            if lo_b.shape != (self.ndim,) or hi_b.shape != (self.ndim,):
                raise ValueError("floor_box lo/hi must each have 9 entries.")
            if np.any(hi_b <= lo_b):
                raise ValueError("floor_box needs hi > lo on every axis.")
            self.floor_lo, self.floor_hi = lo_b, hi_b
            self._floor_log_vol = float(np.sum(np.log(hi_b - lo_b)))
        else:
            self.floor_lo = self.floor_hi = None
            self._floor_log_vol = 0.0

        # --- master (numpy) tables; device copies are cached lazily -------
        self._tables = dict(
            means=means,
            chol=chol,
            chol_inv=chol_inv,
            log_norm=log_norm,
            log_w=log_w,
            win_lo=win_lo,
            win_hi=win_hi,
            order=order.astype(np.int64),
            lo_sorted=lo_sorted,
        )
        self._dev_cache: dict = {}

        # diagnostics / provenance (never consulted by rvs/logpdf)
        self.p = p
        self.mult = None if mult is None else np.asarray(mult, dtype=float)
        self.n_members = (None if n_members is None
                          else np.asarray(n_members, dtype=np.int64))
        self.island_id = (None if island_id is None
                          else np.asarray(island_id, dtype=np.int64))
        # stored at the PREVIOUS run's df; informational only (the pruning
        # windows above are the ones re-derived against new_tobs).
        self.f0_window_edges = (None if f0_window_edges is None
                                else np.asarray(f0_window_edges, dtype=float))
        self.meta = dict(meta) if meta else {}

        self.use_cupy = bool(use_cupy)
        self._rng = np.random.default_rng(seed)
        if self.use_cupy:
            self._t(self.xp)  # build the device cache up-front

    # ------------------------------------------------------------------
    @property
    def n_components(self) -> int:
        return int(self._tables["means"].shape[0])

    @property
    def xp(self):
        if self.use_cupy:
            import cupy as cp

            return cp
        return np

    def _t(self, xp):
        """The tables on ``xp`` (device copies cached; numpy = the masters)."""
        if xp is np:
            return self._tables
        if "dev" not in self._dev_cache:
            self._dev_cache["dev"] = {
                k: xp.asarray(v) for k, v in self._tables.items()
            }
        return self._dev_cache["dev"]

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_dev_cache"] = {}  # device arrays never pickle
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._dev_cache = {}

    # ------------------------------------------------------------------
    @classmethod
    def from_npz(cls, path: str, new_tobs: typing.Optional[float] = None,
                 use_cupy: bool = False, **kwargs) -> "WarmStartComponents":
        """Load a fitter npz (``warmstart_fit_from_store.py`` writer schema).

        ``new_tobs=None`` falls back to the npz's OWN ``meta['tobs']`` (same-
        Tobs reuse); pass the new run's Tobs [s] for the cross-Tobs policy.
        Remaining ``kwargs`` forward to the constructor (``floor_box``,
        ``floor_eps``, ``window_df``, ``p_floor``, ``seed``...).
        """
        with np.load(path, allow_pickle=False) as d:
            required = ("means", "covs", "p", "mult", "n_members",
                        "island_id", "f0_window_edges", "meta")
            missing = [k for k in required if k not in d]
            if missing:
                raise ValueError(
                    f"warm-start npz {path} is missing keys {missing} "
                    f"(has {sorted(d.keys())}); expected the "
                    f"scripts/gb/warmstart_fit_from_store.py writer schema."
                )
            meta = json.loads(str(d["meta"]))
            # basis lockstep checks against the writer
            cols = list(meta.get("column_names", []))
            if cols and cols != COLUMN_NAMES:
                raise ValueError(
                    f"npz column_names {cols} != expected {COLUMN_NAMES}."
                )
            if str(meta.get("f0_units", "mHz")) != "mHz":
                raise ValueError(
                    f"npz f0_units {meta.get('f0_units')!r} != 'mHz'."
                )
            circ = {int(k): float(v)
                    for k, v in dict(meta.get("circular_cols", {})).items()}
            if circ and any(
                abs(circ.get(c, -1.0) - per) > 1e-12
                for c, per in CIRCULAR_COLS.items()
            ):
                raise ValueError(
                    f"npz circular_cols {circ} != expected {CIRCULAR_COLS}."
                )
            if new_tobs is None:
                new_tobs = float(meta["tobs"])
            obj = cls(
                d["means"], d["covs"], d["p"], new_tobs=float(new_tobs),
                mult=d["mult"], n_members=d["n_members"],
                island_id=d["island_id"],
                f0_window_edges=d["f0_window_edges"], meta=meta,
                use_cupy=use_cupy, **kwargs,
            )
        logger.info(
            "WarmStartComponents: %d components from %s (fit tobs=%.6g s, "
            "new tobs=%.6g s -> df=%.6g mHz; window +-%g df + %g sigma_f0; "
            "floor_eps=%g; weights ~ p, mult IGNORED [v1]).",
            obj.n_components, path, float(meta.get("tobs", new_tobs)),
            obj.tobs, obj.df_mhz, obj.window_df, F0_WINDOW_GUARD_NSIGMA,
            obj.floor_eps,
        )
        return obj

    # ------------------------------------------------------------------
    def rvs(self, size=1):
        """Draws from the FULL mixture (+ floor); ``size + (9,)`` array.

        Component choice ~ ``weights`` (~ p); per-component draw
        ``mean + L z`` (Cholesky); circular columns wrapped to
        ``[0, period)``. Host RNG; returned on ``xp`` (cupy iff use_cupy).
        """
        if isinstance(size, int):
            size = (size,)
        n = int(np.prod(size))
        t = self._tables  # draws are host-side; upload once at the end
        out = np.empty((n, self.ndim), dtype=np.float64)

        is_floor = np.zeros(n, dtype=bool)
        if self.floor_eps > 0:
            is_floor = self._rng.random(n) < self.floor_eps
            n_floor = int(is_floor.sum())
            if n_floor:
                out[is_floor] = self._rng.uniform(
                    self.floor_lo, self.floor_hi,
                    size=(n_floor, self.ndim))

        n_mix = int((~is_floor).sum())
        if n_mix:
            k = self._rng.choice(self.n_components, size=n_mix,
                                 p=self.weights)
            z = self._rng.standard_normal((n_mix, self.ndim))
            draws = t["means"][k] + np.einsum(
                "nij,nj->ni", t["chol"][k], z)
            for c, period in CIRCULAR_COLS.items():
                draws[:, c] %= period
            out[~is_floor] = draws

        out = out.reshape(size + (self.ndim,))
        return self.xp.asarray(out) if self.use_cupy else out

    # ------------------------------------------------------------------
    def logpdf(self, x):
        """Mixture (+ floor) log density at ``x`` of shape ``(n, 9)``.

        Computed over the f0-windowed candidate set (search structure; equal
        to the full mixture to machine precision -- module docstring bound).
        Dispatches on the INPUT's array module; never returns NaN.
        """
        from ..utils.utility import get_array_module

        xp = get_array_module(x)
        t = self._t(xp)
        x = xp.atleast_2d(xp.asarray(x, dtype=xp.float64))
        n = x.shape[0]
        f0 = x[:, 1]

        # candidate gather over the (actual) max f0-window overlap depth D
        # (the StackedFStatProposal4D pattern).
        D = self._overlap_depth
        j = xp.searchsorted(t["lo_sorted"], f0, side="right")
        cand_pos = j[:, None] - 1 - xp.arange(D)[None, :]  # (n, D)
        valid = cand_pos >= 0
        cand_pos = xp.clip(cand_pos, 0, self.n_components - 1)
        kk = t["order"][cand_pos]                          # (n, D)
        valid = valid & (f0[:, None] >= t["win_lo"][kk]) \
            & (f0[:, None] <= t["win_hi"][kk])

        lp = xp.full((n, D), -np.inf, dtype=xp.float64)
        rows, cols = xp.where(valid)
        if rows.shape[0]:
            k_sel = kk[rows, cols]
            diff = x[rows] - t["means"][k_sel]
            # minimal-image displacement on the circular columns
            for c, period in CIRCULAR_COLS.items():
                d = diff[:, c]
                diff[:, c] = d - period * xp.round(d / period)
            y = xp.einsum("nij,nj->ni", t["chol_inv"][k_sel], diff)
            maha = xp.sum(y * y, axis=1)
            lp[rows, cols] = (t["log_w"][k_sel] + t["log_norm"][k_sel]
                              - 0.5 * maha)

        m = xp.max(lp, axis=1)
        m_safe = xp.where(xp.isfinite(m), m, 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            lp_mix = xp.log(xp.sum(xp.exp(lp - m_safe[:, None]), axis=1)) \
                + m_safe
        lp_mix = xp.where(xp.isfinite(m), lp_mix, -xp.inf)

        if self.floor_eps <= 0:
            return lp_mix

        # (1 - eps) * mixture + eps * Uniform(box); the box test wraps the
        # circular columns first (leaves arrive wrapped, but be safe).
        xb = x.copy()
        for c, period in CIRCULAR_COLS.items():
            xb[:, c] = xb[:, c] % period
        lo = xp.asarray(self.floor_lo)
        hi = xp.asarray(self.floor_hi)
        inside = xp.all((xb >= lo) & (xb <= hi), axis=1)
        lp_floor = xp.where(inside, -self._floor_log_vol, -xp.inf)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = xp.logaddexp(np.log1p(-self.floor_eps) + lp_mix,
                               np.log(self.floor_eps) + lp_floor)
        return xp.where(xp.isnan(out), -xp.inf, out)
