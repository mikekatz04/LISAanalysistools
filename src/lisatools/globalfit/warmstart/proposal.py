"""Warm-start GB RJ-birth proposal from a previous run's clustered posterior.

Workstream B of the warm-start pipeline (``docs/warm-start-gb-proposal.md``,
``docs/6mo-run-prep.md``): the offline fitter
``lisatools.globalfit.warmstart.fit_from_store`` turns a finished run's cold-chain
GB leaf table into clusters (one per recovered source) with inclusion
probabilities ``p`` -- a Gaussian mixture per cluster in the production
(observable-basis) format, one Gaussian per cluster in the legacy one --
the referee prunes them, and :class:`WarmStartComponents` loads the refereed
npz and serves it as an eryn duck-typed RJ birth distribution over the FULL
9-column sampled GB basis::

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
* **Two component formats** (2026-09-18). ``meta["basis"]`` selects:

  - ``"sampling"`` (or NO ``basis`` key at all -- the shipped
    ``gf_prod_3mo_v8_10w_refereed.npz``): ``means``/``covs`` are one
    Gaussian per cluster in the 9 columns above. Unchanged in every respect.
  - ``"observable"``: the packed ``gmm_*`` arrays hold a Gaussian MIXTURE
    per cluster in the OBSERVABLE columns
    ``(lnA, f_mid [Hz], fdot, phi0, cos_iota, psi, alpha, sin_delta, Mc)``.
    ``rvs`` draws there and returns ``from_internal(z)`` -- sampling
    columns, as every caller expects -- and ``logpdf`` scores
    ``mixture_logpdf(to_internal(x)) - log_jacobian(x)``.

  **The Jacobian SUBTRACTS.** ``log_jacobian`` is ``ln|dy/dz|`` and a
  density transforms with the determinant of the inverse map,
  ``q_y = q_z |dz/dy|``. The sign lives in
  :func:`.basis.log_density_to_sampling` so there is one place to get it
  right; flipped, ``rvs`` and ``logpdf`` disagree by ``2 log_jacobian``,
  which varies across a component and therefore biases the RJ birth/death
  factors rather than cancelling as a constant would.

Pickle/deepcopy safe (sprint rule): master tables are numpy; the per-device
cupy cache is dropped on ``__getstate__``.

WHERE IT SITS. ``recipe.build_gb_moves`` loads the npz named by
``GB_WARM_START_COMPONENTS`` twice with :meth:`WarmStartComponents.from_npz`:
the ``rj_warm_search`` container (minimal-image charge) and the
``rj_warm_pe`` container (``circ_images = GB_WARM_START_CIRC_IMAGES``, the
exact wrapped-normal density PE needs), both at ``new_tobs`` = this run's
Tobs, ``floor_box`` = the run's 9-D prior box (f0 over the interior bands),
``floor_eps = GB_WARM_START_FLOOR_EPS`` and ``p_floor =
GB_WARM_START_P_FLOOR``, and per-rank seeds. With ``GB_WARM_SELF=1`` each is
then wrapped by :class:`~lisatools.globalfit.warmstart.self_refit.WarmSelfMixer`,
which loads every self-refit epoch with the same options. The
``BandSorter`` of the GB RJ moves calls :meth:`~WarmStartComponents.logpdf`
on device-resident leaves for the death factors.

Env knobs:

* ``GB_WARM_START_FLOOR_EPS`` (recipe, default ``"0.05"``; launchers 0.05),
  ``GB_WARM_START_P_FLOOR`` (recipe, default ``"0"``; unset) and
  ``GB_WARM_START_CIRC_IMAGES`` (recipe, the PE twin and the opt-in
  para-search container, default ``"3"``; launchers 3) -- read by the
  recipe and passed in.
* ``GB_WARM_START_F0_WINDOW_DF`` -- the logpdf candidate-window half-width
  in units of ``1/Tobs`` when ``window_df=None``; default 10; unset.
* ``GB_WARM_START_LOGPDF_MAX_BYTES`` -- byte budget of one logpdf chunk,
  read at import; default ``2**30``; unset.
"""

from __future__ import annotations

import json
import logging
import os
import typing
import warnings

import numpy as np

from ...utils.device import current_device, device_context

logger = logging.getLogger(__name__)

# The sampled GB basis the fitter writes (warmstart.fit_from_store
# .py COLUMN_NAMES / CIRCULAR_COLS -- kept in lockstep by the npz meta check).
COLUMN_NAMES = [
    "dist", "f0", "Mc", "phi0", "cos_iota", "psi", "alpha",
    "sin_delta", "fdot_astro_ratio",
]
CIRCULAR_COLS = {3: 2.0 * np.pi, 5: np.pi, 6: 2.0 * np.pi}

# Byte budget for one chunk of the windowed-mixture logpdf, which scores
# (coord, candidate) PAIRS and whose chol_inv gather is (n_pair, ndim, ndim).
# Default 1 GiB, the same value and spelling as the sig-het fold cap
# (GB_SIGHET_FOLD_MAX_BYTES). Raise it only with measured headroom: the pair
# count is n x D x images, and a 6-month run hit 19.7M pairs = 12.8 GB in a
# single unchunked allocation. See _mixture_logpdf.
_LOGPDF_MAX_BYTES = int(
    os.environ.get("GB_WARM_START_LOGPDF_MAX_BYTES", 1 << 30))

# Bytes the candidate-gather stage of the logpdf holds PER (coord, depth)
# cell before any pair is scored: cand_pos + kk (int64), the win_lo / win_hi
# takes and lp (float64), the exp(lp - m) temporary (float64), the bool
# masks (~3 bytes). Seven words is a round upper bound. The same 1 GiB
# budget above divides into rows of the coordinate block (see
# _mixture_logpdf): the 9mo full_pe (job 757, 2026-10-12) scored 20,000
# coordinates against a depth-29,832 candidate set and asked for
# 4.77 GB per (n, D) array on a card already holding 92 GB.
_LOGPDF_BYTES_PER_CANDIDATE = 56

#: default logpdf candidate window, in units of df = 1/Tobs (mHz)
DEFAULT_F0_WINDOW_DF = 10.0
#: per-component sigma guard added to the candidate window half-width; the
#: negligibility bound quoted in the module docstring is exp(-GUARD^2/2).
F0_WINDOW_GUARD_NSIGMA = 10.0
#: circular-sigma sanity threshold (fraction of the period) for the
#: minimal-image approximation warning.
CIRC_SIGMA_WARN_FRAC = 1.0 / 6.0


def _host(x):
    """cupy-or-numpy -> numpy."""
    return x.get() if hasattr(x, "get") else np.asarray(x)


class WarmStartComponents:
    """f0-windowed Gaussian-mixture RJ birth proposal over the 9-col GB basis.

    The ``rj_proposal_distribution`` of the ``rj_warm_search`` /
    ``rj_warm_pe`` moves (directly, or inside a
    :class:`~lisatools.globalfit.warmstart.self_refit.WarmSelfMixer`); also
    the container of every self-refit epoch and of the para-search epochs.
    Normally built with :meth:`from_npz`. ``rvs`` returns and ``logpdf``
    takes SAMPLING-basis rows ``(dist [kpc], f0 [mHz], Mc [Msun], phi0,
    cos_iota, psi, alpha, sin_delta, fdot_astro_ratio)``, ICRS sky, whatever
    basis the components were fitted in.

    Args:
        means (array-like): ``(n, 9)`` component means in the fitted basis:
            the sampled basis (f0 in mHz) for ``basis="sampling"``, the
            observable basis (f_mid in Hz) for ``basis="observable"``.
        covs (array-like): ``(n, 9, 9)`` covariances in the same basis (must
            be positive definite; the fitter's eigenvalue floors guarantee
            it).
        p (array-like): ``(n,)`` strictly positive weights (inclusion
            probabilities; :meth:`from_npz` passes cluster ``p`` x the
            component's share for a mixture set); mixture weights are
            ``max(p, p_floor)`` renormalized.
        new_tobs (float): THE NEW RUN's observation time [s]; f0 candidate
            windows are derived from ``df = 1/new_tobs`` (cross-Tobs v1
            policy -- widths themselves stay as stored).
        window_df (float, optional): candidate window half-width in units of
            df. ``None`` -> ``GB_WARM_START_F0_WINDOW_DF`` env (default 10).
        p_floor: optional floor applied to ``p`` before renormalizing.
            Default ``0.0``.
        floor_box (tuple, optional): ``(lo, hi)`` 9-vector pair in the
            SAMPLING basis bounding the uniform floor component (normally the
            run's prior box). ``None`` with ``floor_eps > 0`` raises.
        floor_eps: uniform-floor mixture weight in ``[0, 1)`` (0 = bare
            mixture). Default ``0.0``.
        circ_images: wrapped-normal image count for the circular columns.
            0 (default) charges the minimal-image approximation -- the
            SEARCH behavior, bit-identical to the pre-knob code. K > 0
            makes logpdf the EXACT density of the wrapped draws (the
            ``rj_warm_pe`` detailed-balance requirement): per component
            and circular column the full multivariate normal is summed
            over the +-K_c period images around the minimal image, with
            ``K_c = min(K, ceil(6 sigma_c / period + 0.5) - 1)`` so
            narrow columns stay single-image (their neglected mass is
            < e^-18) and only genuinely wide columns pay. Draws are
            unaffected (rvs wraps either way; only the CHARGE changes).
        bounded_cols: ``{column: (lo, hi)}`` box-truncated columns of the
            FITTED basis (at most 2, never circular): ``logpdf`` carries each
            component's rectangle normalization and is ``-inf`` outside,
            ``rvs`` rejects and redraws. Default ``None`` = none.
        mult, n_members, island_id, f0_window_edges, meta: carried through
            from the npz for diagnostics; unused by rvs/logpdf (``mult`` is
            deliberately NOT a weight -- PROVISIONAL v1 policy).
        use_cupy: keep a device copy of the tables and return device draws
            (GPU runs). The master copy stays numpy (pickle rule).
        seed: RNG seed for :meth:`rvs`.
        basis: ``"sampling"`` (default) or ``"observable"`` -- the basis of
            ``means`` / ``covs``.
        map_params (dict, optional): The stored ``meta["map_params"]``;
            required for ``basis="observable"`` (the map is rebuilt from it at
            construction).
        gmm_ncomp (array-like, optional): Per-cluster component counts of a
            mixture set (partitions the flat component arrays).

    Attributes:
        weights (np.ndarray): ``(n,)`` normalized mixture weights.
        tobs (float): ``new_tobs``.
        df_mhz (float): ``1e3 / new_tobs``.
        df_col1 (float): ``1/new_tobs`` in column-1 units of the fitted
            basis (mHz sampling, Hz observable).
        window_df (float): The resolved candidate half-width in df units.
        window_halfwidth_mhz (np.ndarray): ``(n,)`` per-component window
            half-widths, in column-1 units despite the name.
        obs_map (GBObservableFiberBasis or None): The observable map
            (``None`` for a sampling set).
        floor_eps, circ_images, bounded_cols, basis, use_cupy: As given.
        p, mult, n_members, island_id, f0_window_edges, meta: Diagnostics.

    Raises:
        ValueError: On a shape mismatch, non-positive ``p``, a bad
            ``bounded_cols`` / ``floor_box`` / ``floor_eps`` /
            ``circ_images`` / ``new_tobs``, or ``basis="observable"`` without
            ``map_params``.
        NotImplementedError: For more than 2 ``bounded_cols``.
        numpy.linalg.LinAlgError: For a covariance that is not positive
            definite.

    Example:
        >>> import numpy as np
        >>> from lisatools.globalfit.warmstart.proposal import WarmStartComponents
        >>> means = np.array([[1.0, 3.0, 0.3, 1.0, 0.1, 1.0, 2.0, 0.2, 0.0]])
        >>> covs = np.diag([1e-2, 1e-8, 1e-4, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2])[None]
        >>> comps = WarmStartComponents(means, covs, np.array([1.0]), new_tobs=3.15e7, seed=1)
        >>> comps.rvs(4).shape
        (4, 9)
        >>> comps.logpdf(comps.rvs(2)).shape
        (2,)
    """

    ndim = 9

    def __init__(self, means, covs, p, *, new_tobs, window_df=None,
                 p_floor: float = 0.0, floor_box=None, floor_eps: float = 0.0,
                 circ_images: int = 0,
                 bounded_cols: typing.Optional[dict] = None,
                 mult=None, n_members=None, island_id=None,
                 f0_window_edges=None, meta: typing.Optional[dict] = None,
                 use_cupy: bool = False, seed: typing.Optional[int] = None,
                 basis: str = "sampling", map_params=None, gmm_ncomp=None):
        if basis not in ("sampling", "observable"):
            raise ValueError(
                f"basis must be 'sampling' or 'observable', got {basis!r}")
        #: which coordinates ``means``/``covs`` live in. ``"observable"``
        #: means rvs converts back with ``from_internal`` and logpdf carries
        #: the map's log Jacobian.
        self.basis = str(basis)
        self._map_params = dict(map_params) if map_params else None
        #: per-CLUSTER mixture component counts; partitions the flat
        #: component arrays (``None`` for a legacy one-Gaussian file).
        self.gmm_ncomp = (None if gmm_ncomp is None
                          else np.asarray(gmm_ncomp, dtype=np.int64))
        self.obs_map = None
        if self.basis == "observable":
            if not self._map_params:
                raise ValueError(
                    "basis='observable' needs map_params to rebuild the "
                    "GBObservableFiberBasis.")
            from . import basis as _wb

            # eager: the stored params carry their own input_basis, so the
            # object is usable straight from from_npz rather than waiting
            # for someone to remember attach_transform().
            self.obs_map = _wb.build_map_from_stored_params(self._map_params)
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

        # --- bounded columns: box-truncated mixture (ruling 2026-09-11) ---
        # The joint Gaussian is truncated on the box of the listed columns
        # (production: 4 = cos_iota, 8 = fdot_astro_ratio): logpdf carries
        # each component's rectangle normalization -log Z_k and is -inf
        # outside the box; rvs rejects+redraws out-of-box rows. Everything
        # stays in the sampled x-basis, so the fitter/referee/apply chain
        # is unchanged. Benched 2026-09-11 (2-fold CV test log-density on
        # the v8 24w gated fit): truncated normal beats the plain Gaussian
        # on 92% (cos_iota) / 97% (fdot_ratio) of components and beats the
        # atanh infinite-basis refit on both columns.
        self.bounded_cols = {}
        if bounded_cols:
            for c, (blo, bhi) in dict(bounded_cols).items():
                c = int(c)
                blo, bhi = float(blo), float(bhi)
                if not 0 <= c < self.ndim:
                    raise ValueError(f"bounded_cols column {c} out of range.")
                if c in CIRCULAR_COLS:
                    raise ValueError(
                        f"bounded_cols column {c} is circular; the wrapped "
                        f"treatment already owns it.")
                if not bhi > blo:
                    raise ValueError(
                        f"bounded_cols[{c}] needs hi > lo; got "
                        f"({blo}, {bhi}).")
                self.bounded_cols[c] = (blo, bhi)
        if len(self.bounded_cols) > 2:
            raise NotImplementedError(
                "bounded_cols supports at most 2 columns (the rectangle "
                f"normalization is 1-D/2-D); got {len(self.bounded_cols)}.")
        if self.bounded_cols:
            from scipy.stats import multivariate_normal, norm as _norm

            bc = sorted(self.bounded_cols)
            if len(bc) == 1:
                c0 = bc[0]
                blo, bhi = self.bounded_cols[c0]
                sg = np.sqrt(covs[:, c0, c0])
                z = (_norm.cdf((bhi - means[:, c0]) / sg)
                     - _norm.cdf((blo - means[:, c0]) / sg))
            else:
                c0, c1 = bc
                lo0, hi0 = self.bounded_cols[c0]
                lo1, hi1 = self.bounded_cols[c1]
                z = np.empty(n)
                for k in range(n):
                    mu = means[k, [c0, c1]]
                    cv = covs[k][np.ix_([c0, c1], [c0, c1])]
                    f = multivariate_normal(mean=mu, cov=cv,
                                            allow_singular=True).cdf
                    z[k] = (f([hi0, hi1]) - f([lo0, hi1])
                            - f([hi0, lo1]) + f([lo0, lo1]))
            log_trunc_z = np.log(np.maximum(z, 1e-12))
        else:
            log_trunc_z = np.zeros(n)

        # --- weights ~ p (renormalized; optional floor) ------------------
        w = np.maximum(p, float(p_floor))
        self.weights = w / w.sum()
        log_w = np.log(self.weights)

        # --- f0 candidate windows vs the NEW run's df --------------------
        self.tobs = float(new_tobs)
        if self.tobs <= 0:
            raise ValueError(f"new_tobs must be positive; got {new_tobs}.")
        self.df_mhz = 1e3 / self.tobs  # stored f0 is mHz
        # UNITS: column 1 is f0 [mHz] in the sampling basis but f_mid [Hz]
        # in the observable one, so the candidate window's df must be taken
        # in the units of the basis the components were FITTED in. Reusing
        # the mHz width against Hz means would make every window 1000x too
        # wide -- harmless to correctness (the window is a search structure
        # bounded by the 10-sigma guard) but it would gather the whole band
        # as candidates at every logpdf call.
        self.df_col1 = (1.0 / self.tobs if self.basis == "observable"
                        else self.df_mhz)
        if window_df is None:
            window_df = float(os.environ.get(
                "GB_WARM_START_F0_WINDOW_DF", str(DEFAULT_F0_WINDOW_DF)))
        self.window_df = float(window_df)
        sigma_f0 = np.sqrt(covs[:, 1, 1])
        # SEARCH structure only: the guard keeps every omitted component
        # >= GUARD sigma away in f0, making the windowed logpdf equal to the
        # full-mixture logpdf to machine precision (module docstring bound).
        self.window_halfwidth_mhz = (
            self.window_df * self.df_col1
            + F0_WINDOW_GUARD_NSIGMA * sigma_f0
        )
        win_lo = means[:, 1] - self.window_halfwidth_mhz
        win_hi = means[:, 1] + self.window_halfwidth_mhz

        # --- wrapped-normal image expansion (circ_images) ----------------
        # The logpdf gather runs over CANDIDATES: component k contributes
        # one candidate per period-image shift of its circular columns, so
        # the mixture logsumexp computes sum_k w_k sum_j N(d_min + s_j) --
        # the wrapped density -- with no change to the window machinery
        # (shifts never touch f0). circ_images == 0 gives the identity
        # expansion: one zero-shift candidate per component, numerically
        # identical to the pre-knob minimal-image code.
        self.circ_images = int(circ_images)
        if self.circ_images < 0:
            raise ValueError(f"circ_images must be >= 0; got {circ_images}.")
        if self.circ_images > 0:
            import itertools

            circ_cols = list(CIRCULAR_COLS.items())
            # per-component per-column image count: images whose nearest
            # approach to the minimal image exceeds 6 sigma are dropped.
            k_per = np.zeros((n, len(circ_cols)), dtype=np.int64)
            for ci, (c, period) in enumerate(circ_cols):
                sig = np.sqrt(covs[:, c, c])
                k_per[:, ci] = np.minimum(
                    self.circ_images,
                    np.maximum(
                        0, np.ceil(6.0 * sig / period + 0.5).astype(np.int64)
                        - 1),
                )
            parents, shifts = [], []
            for k in range(n):
                ranges = [range(-kk_c, kk_c + 1) for kk_c in k_per[k]]
                for combo in itertools.product(*ranges):
                    s = np.zeros(self.ndim)
                    for ci, (c, period) in enumerate(circ_cols):
                        s[c] = combo[ci] * period
                    parents.append(k)
                    shifts.append(s)
            cand_parent = np.asarray(parents, dtype=np.int64)
            cand_shift = np.asarray(shifts, dtype=np.float64)
        else:
            cand_parent = np.arange(n, dtype=np.int64)
            cand_shift = np.zeros((n, self.ndim), dtype=np.float64)
        self._n_candidates = int(len(cand_parent))

        cand_win_lo = win_lo[cand_parent]
        cand_win_hi = win_hi[cand_parent]
        order = np.argsort(cand_win_lo, kind="stable")
        lo_sorted = cand_win_lo[order]
        hi_sorted_all = np.sort(cand_win_hi)
        depth = np.arange(1, self._n_candidates + 1) - np.searchsorted(
            hi_sorted_all, lo_sorted, side="left")
        self._overlap_depth = int(max(1, depth.max()))

        # --- circular-sigma sanity (minimal-image approximation) ---------
        # With circ_images > 0 the wide pairs are exactly what the image
        # sum covers, so the minimal-image warning no longer applies.
        bad = 0
        if self.circ_images == 0:
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
            log_trunc_z=log_trunc_z,
            log_w=log_w,
            # per-CANDIDATE (image) arrays; parents index the arrays above
            win_lo=cand_win_lo,
            win_hi=cand_win_hi,
            order=order.astype(np.int64),
            lo_sorted=lo_sorted,
            cand_parent=cand_parent,
            cand_shift=cand_shift,
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
        """Flat mixture component count (all clusters' components)."""
        return int(self._tables["means"].shape[0])

    @property
    def xp(self):
        """``cupy`` when ``use_cupy``, else ``numpy`` (derived, never stored)."""
        if self.use_cupy:
            import cupy as cp

            return cp
        return np

    def _t(self, xp):
        """The tables on ``xp`` (numpy = the masters; device copies cached PER DEVICE).

        MULTI-GPU (2026-10-09, 9mo job 751): the cache used to be keyed by
        the literal ``"dev"``, so the device current on the FIRST call (GPU 0:
        ``__init__`` builds it up-front) owned the tables for good and a rank
        on GPU 1 read them across devices on every warm-start proposal (cupy
        "array resides (0) ... current device (1)" at the searchsorted and
        the gathers of :meth:`_mixture_logpdf`). Keyed by the current device
        id now, each copy uploaded from the host masters with that device
        current.
        """
        if xp is np:
            return self._tables
        dev = current_device(xp)
        tables = self._dev_cache.get(dev)
        if tables is None:
            with device_context(xp, dev):
                tables = {k: xp.asarray(v) for k, v in self._tables.items()}
            self._dev_cache[dev] = tables
        return tables

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
        """Load a fitter / refereed npz (the :mod:`.fit_from_store` writer schema).

        ``new_tobs=None`` falls back to the npz's OWN ``meta['tobs']`` (same-
        Tobs reuse); pass the new run's Tobs [s] for the cross-Tobs policy.
        ``meta["basis"]`` selects the format (absent = the legacy sampling
        set). For an observable set the packed ``gmm_*`` arrays are
        flattened to one entry per mixture component, with weight = cluster
        ``p`` x the component's within-cluster weight (floored at ``1e-12``)
        and the per-cluster diagnostics repeated per component. The
        ``bounded_cols`` in the meta apply unless passed explicitly. Also
        loads the self-refit and para-search epoch npz files (same schema).

        Args:
            path: The npz.
            new_tobs: This run's Tobs [s]. Default ``None`` = the file's.
            use_cupy: Device tables / draws. Default ``False``.
            **kwargs: Forwarded to the constructor (``floor_box``,
                ``floor_eps``, ``window_df``, ``p_floor``, ``seed``,
                ``circ_images``, ``bounded_cols``).

        Returns:
            WarmStartComponents: The loaded container.

        Raises:
            ValueError: When a required key is missing, or the meta's
                ``column_names`` / ``f0_units`` / ``circular_cols`` disagree
                with this module's basis.

        Example:
            .. code-block:: python

                comps = WarmStartComponents.from_npz(
                    "warmstart/gf_prod_6mo_v9_4gpu_refereed.npz",
                    new_tobs=23328000.0, floor_box=(lo, hi), floor_eps=0.05,
                    circ_images=3, seed=1)
        """
        with np.load(path, allow_pickle=False) as d:
            meta = json.loads(str(d["meta"])) if "meta" in d else {}
            # A file with no ``basis`` key is the pre-2026-09-18 layout:
            # sampling basis, ONE Gaussian per cluster. That is what the
            # shipped gf_prod_3mo_v8_10w_refereed.npz is, and it must keep
            # loading and drawing exactly as before.
            basis = str(meta.get("basis", "sampling"))
            per_cluster = ("p", "mult", "n_members", "island_id",
                           "f0_window_edges", "meta")
            if basis == "observable":
                required = ("gmm_ncomp", "gmm_weights", "gmm_means",
                            "gmm_covs", "gmm_invcovs", "gmm_dets",
                            "gmm_mins", "gmm_maxs") + per_cluster
            else:
                required = ("means", "covs") + per_cluster
            missing = [k for k in required if k not in d]
            if missing:
                raise ValueError(
                    f"warm-start npz {path} is missing keys {missing} "
                    f"(has {sorted(d.keys())}); expected the "
                    f"warmstart.fit_from_store writer schema "
                    f"(basis={basis!r})."
                )
            # basis lockstep checks against the writer
            want_cols = COLUMN_NAMES
            want_units = "mHz"
            if basis == "observable":
                from . import basis as _wb

                want_cols = _wb.OBSERVABLE_COLUMN_NAMES
                want_units = "Hz"
            cols = list(meta.get("column_names", []))
            if cols and cols != want_cols:
                raise ValueError(
                    f"npz column_names {cols} != expected {want_cols}."
                )
            if str(meta.get("f0_units", want_units)) != want_units:
                raise ValueError(
                    f"npz f0_units {meta.get('f0_units')!r} != "
                    f"{want_units!r} (basis {basis!r})."
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
            # bounded columns ride in the fitter meta (absent = the
            # pre-2026-09-11 all-Gaussian behavior, bit-identical); an
            # explicit ctor kwarg wins.
            if "bounded_cols" not in kwargs:
                bc_meta = dict(meta.get("bounded_cols", {}) or {})
                kwargs["bounded_cols"] = {
                    int(c): (float(v[0]), float(v[1]))
                    for c, v in bc_meta.items()
                } or None
            extra = {}
            if basis == "observable":
                from lisatools.sampling.fstat_proposal import (
                    unpack_gmm_components)

                comps = unpack_gmm_components(d)
                ncomp = np.asarray(d["gmm_ncomp"], dtype=np.int64)
                means = np.concatenate([np.asarray(m) for m in comps[1]],
                                       axis=0)
                covs = np.concatenate([np.asarray(c) for c in comps[2]],
                                      axis=0)
                w_within = np.concatenate([np.asarray(w) for w in comps[0]])
                # FLAT component weight = the cluster's inclusion
                # probability x the component's share of that cluster, so
                # the mixture still weights ~ p across clusters (v1 policy)
                # while distributing each cluster's mass over its modes.
                # Clamped strictly positive: a collapsed mixture component
                # can carry a zero weight, which the schema forbids.
                p_flat = (np.repeat(np.asarray(d["p"], dtype=float), ncomp)
                          * np.maximum(w_within, 1e-12))
                # per-CLUSTER diagnostics are broadcast to per-COMPONENT so
                # every consumer indexing alongside means stays aligned.
                rep = {k: np.repeat(np.asarray(d[k]), ncomp, axis=0)
                       for k in ("mult", "n_members", "island_id")}
                extra = dict(basis="observable", gmm_ncomp=ncomp,
                             map_params=meta.get("map_params"))
            else:
                means, covs, p_flat = d["means"], d["covs"], d["p"]
                rep = {k: d[k] for k in ("mult", "n_members", "island_id")}
            obj = cls(
                means, covs, p_flat, new_tobs=float(new_tobs),
                mult=rep["mult"], n_members=rep["n_members"],
                island_id=rep["island_id"],
                f0_window_edges=d["f0_window_edges"], meta=meta,
                use_cupy=use_cupy, **extra, **kwargs,
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
    # TODO check: keep/remove/document -- no caller outside tests (from_npz builds the map eagerly)
    def attach_transform(self, transform_container):
        """Rebind the observable map against a REAL transform container.

        :meth:`from_npz` already builds the map from the stored
        ``map_params`` (which carry their own ``input_basis``), so this is
        not required for the object to work. What it adds is the check that
        the RUN's sampling basis matches the one the components were fitted
        against -- a mismatch raises here rather than mis-indexing every
        column silently. No-op on a sampling-basis set.
        """
        if self.basis != "observable":
            return
        from . import basis as _wb

        self.obs_map = _wb.build_map_from_params(
            transform_container, self._map_params)

    # ------------------------------------------------------------------
    def rvs(self, size=1):
        """Draws from the FULL mixture (+ floor); ``size + (9,)`` array.

        Component choice ~ ``weights`` (~ p); per-component draw
        ``mean + L z`` (Cholesky); bounded columns rejected and redrawn (up
        to 500 passes, then clipped); circular columns wrapped to
        ``[0, period)``; an observable set is mapped back with
        ``from_internal``. A fraction ``floor_eps`` of the draws is uniform
        in ``floor_box``. Host RNG; returned on ``xp`` (cupy iff use_cupy).

        Args:
            size (int or tuple of int, optional): Draw shape. Default ``1``.

        Returns:
            ``xp.ndarray``: ``size + (9,)`` float64 SAMPLING-basis rows (f0
            in mHz); cupy when ``use_cupy``.
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
            if self.bounded_cols:
                # truncated law: reject + redraw out-of-box rows (same
                # component). Acceptance per row is its Z_k, so a few
                # passes clear everything; the cap is a pathology guard.
                for _ in range(500):
                    bad = np.zeros(n_mix, dtype=bool)
                    for c, (blo, bhi) in self.bounded_cols.items():
                        bad |= (draws[:, c] < blo) | (draws[:, c] > bhi)
                    if not bad.any():
                        break
                    kb = k[bad]
                    zb = self._rng.standard_normal((int(bad.sum()),
                                                    self.ndim))
                    draws[bad] = t["means"][kb] + np.einsum(
                        "nij,nj->ni", t["chol"][kb], zb)
                else:
                    for c, (blo, bhi) in self.bounded_cols.items():
                        draws[:, c] = np.clip(draws[:, c], blo, bhi)
            for c, period in CIRCULAR_COLS.items():
                draws[:, c] %= period
            if self.basis == "observable":
                # back to the SAMPLING columns every caller expects. The
                # circular columns pass through the map untouched (indices
                # 3/5/6 are identical in both bases), so wrapping them
                # above is equivalent to wrapping them after.
                draws = np.asarray(self.obs_map.from_internal(draws),
                                   dtype=np.float64)
            out[~is_floor] = draws

        out = out.reshape(size + (self.ndim,))
        return self.xp.asarray(out) if self.use_cupy else out

    # ------------------------------------------------------------------
    def logpdf(self, x):
        """Mixture (+ floor) log density at ``x`` of shape ``(n, 9)``.

        Computed over the f0-windowed candidate set (search structure; equal
        to the full mixture to machine precision -- module docstring bound).
        Dispatches on the INPUT's array module; never returns NaN. For an
        observable set the density is transported to the sampling basis
        (:func:`.basis.log_density_to_sampling`). Rows are processed in
        chunks sized by ``GB_WARM_START_LOGPDF_MAX_BYTES``.

        Args:
            x (array-like): ``(n, 9)`` (or ``(9,)``) SAMPLING-basis rows,
                numpy or cupy.

        Returns:
            ``xp.ndarray``: ``(n,)`` log density on the input's array module;
            ``-inf`` outside the support.

        Raises:
            RuntimeError: For an observable set with no map.
        """
        from ...utils.utility import get_array_module

        xp = get_array_module(x)
        x = xp.atleast_2d(xp.asarray(x, dtype=xp.float64))

        if self.basis == "observable":
            if self.obs_map is None:
                raise RuntimeError(
                    "observable components have no map; call "
                    "attach_transform() with the run's GB transform.")
            from . import basis as _wb

            z = self.obs_map.to_internal(x)
            # ln q_y(y) = ln q_z(z(y)) - ln|dy/dz|  -- see
            # basis.log_density_to_sampling for why it subtracts.
            lp_mix = _wb.log_density_to_sampling(
                self._mixture_logpdf(z), x, self.obs_map)
            lp_mix = xp.asarray(lp_mix)
        else:
            lp_mix = self._mixture_logpdf(x)

        if self.floor_eps <= 0:
            return lp_mix
        return self._add_uniform_floor(lp_mix, x, xp)

    # ------------------------------------------------------------------
    def _mixture_logpdf(self, x):
        """Windowed Gaussian-mixture log density in the FITTED basis.

        ``x`` is in whatever coordinates ``means``/``covs`` live in: the
        sampling columns for a legacy set, the observable ones for a
        mixture set. The caller owns the change of variables.
        """
        from ...utils.utility import get_array_module

        xp = get_array_module(x)
        t = self._t(xp)
        x = xp.atleast_2d(xp.asarray(x, dtype=xp.float64))
        n = x.shape[0]
        # CHUNKED OVER ROWS (2026-10-12), on top of the pair chunking inside
        # _mixture_logpdf_rows. The candidate gather there materialises
        # several (n, D) arrays at once (cand_pos, kk, the win_lo / win_hi
        # takes, lp): with the production candidate set D is tens of
        # thousands, so n = 20,000 coordinates (every GB slot of a walker,
        # BandSorter scores them all) is 4.77 GB PER ARRAY. 9mo job 757 died
        # on exactly that in full_pe's first rj_warm_pe, after the PE F-stat
        # refit had left ~70 GB resident:
        #
        #   rj_warm_pe -> BandSorter.__init__ -> rj_prop.logpdf
        #     -> _mixture_logpdf -> t["win_hi"][kk]
        #   OutOfMemoryError: allocating 4,773,120,000 bytes
        #                     (allocated so far: 92,007,001,088)
        #
        # Every row's density depends on that row alone, so slicing the
        # coordinate block is float-exact (tests/test_warmstart_logpdf_row_
        # chunking.py checks bit identity at every chunk size).
        rows = self._logpdf_rows_per_chunk(self._overlap_depth)
        if n <= rows:
            return self._mixture_logpdf_rows(x, t, xp)
        out = xp.empty(n, dtype=xp.float64)
        for s in range(0, n, rows):
            e = min(s + rows, n)
            out[s:e] = self._mixture_logpdf_rows(x[s:e], t, xp)
        return out

    @staticmethod
    def _logpdf_rows_per_chunk(depth, budget=None):
        """Rows of one logpdf block so its ``(rows, D)`` temporaries fit the byte budget.

        The budget is ``GB_WARM_START_LOGPDF_MAX_BYTES`` (``budget=None``) at
        ``_LOGPDF_BYTES_PER_CANDIDATE`` bytes per (row, candidate) cell; at
        least one row, however small the budget or large the depth.
        """
        budget = _LOGPDF_MAX_BYTES if budget is None else int(budget)
        per_row = _LOGPDF_BYTES_PER_CANDIDATE * max(int(depth), 1)
        return max(1, int(budget // max(per_row, 1)))

    def _mixture_logpdf_rows(self, x, t, xp):
        """One block of :meth:`_mixture_logpdf`.

        ``x`` is ``(n, ndim)`` in the fitted basis on ``xp``, ``t`` the
        tables on that module. Returns the ``(n,)`` windowed mixture log
        density (``-inf`` outside the bounded box or with no candidate).
        """
        n = x.shape[0]
        f0 = x[:, 1]

        # candidate gather over the (actual) max f0-window overlap depth D
        # (the StackedFStatProposal4D pattern). Candidates are period
        # IMAGES of the components (identity when circ_images == 0); the
        # logsumexp over candidates therefore sums each component's
        # wrapped-normal images along with the mixture.
        D = self._overlap_depth
        j = xp.searchsorted(t["lo_sorted"], f0, side="right")
        cand_pos = j[:, None] - 1 - xp.arange(D)[None, :]  # (n, D)
        valid = cand_pos >= 0
        cand_pos = xp.clip(cand_pos, 0, self._n_candidates - 1)
        kk = t["order"][cand_pos]                          # (n, D)
        valid = valid & (f0[:, None] >= t["win_lo"][kk]) \
            & (f0[:, None] <= t["win_hi"][kk])

        lp = xp.full((n, D), -np.inf, dtype=xp.float64)
        rows, cols = xp.where(valid)
        n_pair = int(rows.shape[0])
        if n_pair:
            # CHUNKED OVER (coord, candidate) PAIRS (2026-09-19). Every pair
            # is independent -- each writes one cell of ``lp`` and reads only
            # its own component -- so slicing the pair list is a float-exact
            # transformation, not an approximation.
            #
            # It has to be chunked because ``t["chol_inv"][k_sel]`` is a fancy
            # -index TAKE that materialises (n_pair, ndim, ndim) float64 in
            # one allocation. The pair count is n x D, not n: every coord
            # contributes one pair per overlapping window, and
            # GB_WARM_START_CIRC_IMAGES multiplies the candidate set by the
            # wrapped-normal image count on top of that. A 6-month production
            # run reached ~19.7 MILLION pairs and asked cupy for a single
            # 12.8 GB block on a card already holding 90.7 GB:
            #
            #   rj_warm_pe -> BandSorter.__init__ -> rj_prop.logpdf
            #     -> _mixture_logpdf -> einsum("nij,nj->ni", chol_inv[k_sel])
            #   OutOfMemoryError: allocating 12,768,211,456 bytes
            #
            # Budget knob mirrors the sig-het fold cap's spelling and its
            # 1 GiB default. Bytes per pair: the chol_inv take (ndim^2) plus
            # diff/y/maha (~3 x ndim), all float64.
            per_pair = 8 * (self.ndim * self.ndim + 3 * self.ndim)
            step = max(1, int(_LOGPDF_MAX_BYTES // max(per_pair, 1)))
            for s in range(0, n_pair, step):
                e = min(s + step, n_pair)
                r_c, c_c = rows[s:e], cols[s:e]
                k_img = kk[r_c, c_c]              # candidate (image) index
                k_sel = t["cand_parent"][k_img]   # parent component
                diff = x[r_c] - t["means"][k_sel]
                # minimal-image displacement on the circular columns ...
                for c, period in CIRCULAR_COLS.items():
                    d = diff[:, c]
                    diff[:, c] = d - period * xp.round(d / period)
                # ... plus this candidate's period-image shift (zero when
                # circ_images == 0 -- float-exact no-op).
                diff = diff + t["cand_shift"][k_img]
                y = xp.einsum("nij,nj->ni", t["chol_inv"][k_sel], diff)
                maha = xp.sum(y * y, axis=1)
                lp[r_c, c_c] = (t["log_w"][k_sel] + t["log_norm"][k_sel]
                                - t["log_trunc_z"][k_sel] - 0.5 * maha)
                del k_img, k_sel, diff, y, maha

        m = xp.max(lp, axis=1)
        m_safe = xp.where(xp.isfinite(m), m, 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            lp_mix = xp.log(xp.sum(xp.exp(lp - m_safe[:, None]), axis=1)) \
                + m_safe
        lp_mix = xp.where(xp.isfinite(m), lp_mix, -xp.inf)

        # box-truncation support: the mixture is ZERO outside the bounded
        # box (the floor below keeps whatever support it was configured
        # with -- in production its box coincides on these columns).
        for c, (blo, bhi) in self.bounded_cols.items():
            lp_mix = xp.where((x[:, c] >= blo) & (x[:, c] <= bhi),
                              lp_mix, -xp.inf)
        return lp_mix

    # ------------------------------------------------------------------
    def _add_uniform_floor(self, lp_mix, x, xp):
        """``(1 - eps) * mixture + eps * Uniform(box)``.

        ``x`` and ``floor_box`` are both in the SAMPLING basis, which is
        where the mixture density has been transported to by the time this
        runs -- so the floor means the same thing in both formats.
        """
        # the box test wraps the circular columns first (leaves arrive
        # wrapped, but be safe).
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
