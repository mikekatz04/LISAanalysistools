# -*- coding: utf-8 -*-
"""Per-leaf eigen proposal tables for the single-source PE moves.

Builds the ``(axes, sigmas)`` tables that :class:`eryn.moves.EigenAxisMove`
consumes, from an information matrix in the SAMPLING basis. Two builders,
one shared post-processing pipeline:

* :func:`eigen_table_from_ll` — likelihood second differences via
  :func:`lisatools.info_matrix_ll.information_matrix_from_ll`. The right
  route when the likelihood is cheap and batched (SOBBH chunked scoring).
* :func:`eigen_table_from_waveform` — the Gram form
  ``<d_i h | d_j h>`` via the general waveform machinery
  :func:`lisatools.diagnostic.info_matrix` (positive semi-definite by
  construction, ``parameter_transforms`` keeps the derivatives in the
  sampling basis). The right route when one likelihood row is expensive
  but waveform builds are not (MBH, EMRI).

Pipeline (the GB lesson, ``gb_prior_box_scales``: a relative eigen floor is
not scale invariant): whiten the matrix by the prior box widths, run the
generic :func:`eryn.moves.eigenaxis.eigen_axis_set` (whitened
``sigma_max=1`` = one prior width), map the axes back to the sampling
basis with their curvature widths, and cap each width by
:func:`eryn.moves.eigenaxis.axis_prior_bounds`.

Any failure — a non-finite likelihood, a raising engine, a broken prior —
degrades to the identity-axes / 1%-of-prior-width fallback with a logged
warning. A refresh must never crash the sampler.

The step-scale adaptation (``{BRANCH}_EIGEN_AXIS_ADAPT``, user ruling
2026-10-14) lives here too: :func:`cold_unit_deviations` ->
:func:`empirical_axis_widths` -> :func:`adapt_axis_sigmas` widen each axis
of a built table to the chain's own robust width along it (the axes are
kept); the add/remove move owns the ring buffer that feeds them.
"""

import logging

import numpy as np

from eryn.moves.eigenaxis import (
    axis_prior_bounds,
    eigen_axis_set,
    prior_box_scales,
)

from ...diagnostic import info_matrix
from ...info_matrix_ll import information_matrix_from_ll

__all__ = [
    "prior_box_widths",
    "prior_box_bounds",
    "nudge_inside",
    "eigen_table_from_ll",
    "eigen_tables_from_ll_batch",
    "eigen_table_from_waveform",
    "temper_sigmas",
    "axis_caps",
    "wrap_periodic",
    "cold_unit_deviations",
    "empirical_axis_widths",
    "carried_axis_widths",
    "adapt_axis_sigmas",
]

logger = logging.getLogger(__name__)


def _prior_entries(prob_dist_container):
    """``[(column indices, distribution), ...]``, key-spelling agnostic.

    Read the container's PARSED ``priors`` list — eryn's
    :class:`~eryn.priors.probdist.ProbDistContainer` normalises every key
    spelling it accepts (``int``, ``str``, and tuples of either) into
    ``[column_index_array, dist]`` entries there, so this is the only view
    that sees a string-keyed container's columns at all.

    ``priors_in`` is the fallback for duck-typed containers that carry only
    the raw mapping; it can resolve INTEGER keys only, because a string key
    carries no column index outside the parser.
    """
    parsed = getattr(prob_dist_container, "priors", None)
    if parsed is not None:
        return [(np.atleast_1d(np.asarray(inds)), dist) for inds, dist in parsed]
    return [
        (np.atleast_1d(np.asarray(key)), dist)
        for key, dist in prob_dist_container.priors_in.items()
        if isinstance(key, (int, np.integer))
    ]


def temper_sigmas(sigmas_cold, betas):
    """Widen a COLD-chain 1-sigma table for each rung of a ladder.

    ``sigmas_cold`` is any array whose LEADING axis is broadcastable
    against ``betas`` -- typically ``(1, nwalkers, nleaves, ndim)`` from a
    cold-row table about to be shared up the ladder. Returns
    ``sigmas_cold / sqrt(beta)`` with ``betas`` on axis 0.

    **Why this is required whenever a cold table is reused at a hot rung.**
    The target at inverse temperature ``beta`` is proportional to
    ``exp(beta * L)``, so its covariance is ``(beta * F)^-1`` and its width
    is ``1 / sqrt(beta)`` times the cold width. ``EigenAxisMove`` applies
    NO temperature scaling of its own -- ``draw_axis_step`` is
    ``jump_factor * sigma * z`` and ``beta`` appears nowhere in
    ``eryn/moves/eigenaxis.py`` -- so a cold sigma handed to a hot rung
    proposes steps ``sqrt(T)`` too small. On a 12-rung ladder reaching
    ``beta ~ 1e-4`` that is 100x too small at the hot end, which shows up
    as near-1.0 in-model acceptance with a stationary chain: the step is
    tiny, so ``delta_lnL`` is tiny, so ``exp(beta * delta_lnL)`` accepts
    almost everything and nothing travels.

    ``beta <= 0`` (a fully flattened rung) has no finite width; those rungs
    keep the cold sigma rather than producing ``inf``, and the axis cap in
    :func:`_tables_from_info_batch` already bounds every sigma by the prior
    box, so a widened table can never step outside the prior.
    """
    sig = np.asarray(sigmas_cold, dtype=float)
    b = np.asarray(betas, dtype=float).reshape((-1,) + (1,) * (sig.ndim - 1))
    scale = np.where(b > 0.0, 1.0 / np.sqrt(np.where(b > 0.0, b, 1.0)), 1.0)
    return sig * scale


def prior_box_widths(prob_dist_container, ndim):
    """Per-column prior box widths from an eryn prior container.

    Mirrors the GB move's ``_eigen_axis_widths`` reader: eryn's uniform
    exposes ``minimum``/``maximum`` (the ``min_val``/``max_val`` spelling
    belongs to other distributions — try both rather than silently falling
    back to unit widths). Columns without a scalar distribution, and any
    reader failure, fall back to width 1.0 so the table build degrades
    instead of crashing.

    **Read the columns through** :func:`_prior_entries`, never off
    ``priors_in`` keys. Half the stock branches spell their prior dict with
    LABELS rather than column indices — ``psd`` (``psd_prior_dict``:
    ``r"$S_{\\rm oms}$"``), ``mbh`` (``"logM"``, ...), ``emri`` and ``sobbh``
    (their ``input_basis`` names) — and an index-keyed reader silently
    resolves NONE of those columns. Unit widths there are not a harmless
    default: the psd levels live at ~1e-11 / ~1e-14 in boxes 1.9e-10 /
    2.0e-13 wide, so width 1.0 overstates the box by 5e9x / 5e12x, which
    puts every finite-difference corner AND every capped eigen step far
    outside the prior (2026-09-16 one-walker defect: psd in-model
    acceptance exactly 0, "All points entering likelihood have a log prior
    of minus inf" on every repeat).

    A column the container DOES cover but whose distribution exposes no
    FINITE, positive box (eryn distributions default ``minimum`` /
    ``maximum`` to -inf / +inf, and :func:`prior_box_scales` maps such a
    width back to 1.0) is reported once per ``(ndim, columns)`` rather than
    defaulting silently — that silence is what hid the defect above.
    Columns the container does not cover at all (fixed / per-leaf-filled
    parameters) keep width 1.0 quietly, as before. A reader failure keeps
    the columns read up to that point (unit widths elsewhere) and warns.
    """
    lo_r, hi_r, covered, unread, exc = _read_prior_box(prob_dist_container, ndim)
    if exc is not None:
        logger.warning(
            "[eigen_refresh] prior box unavailable (%r); keeping the columns "
            "read so far, unit widths elsewhere", exc,
        )
    lo = np.zeros(ndim)
    hi = np.ones(ndim)
    lo[covered] = lo_r[covered]
    hi[covered] = hi_r[covered]
    width = hi - lo
    unread.extend(c for c in covered if not np.isfinite(width[c]) or width[c] <= 0)
    if unread:
        _warn_unread(ndim, tuple(sorted(set(unread))))
    return prior_box_scales(lo, hi)


def prior_box_bounds(prob_dist_container, ndim):
    """Per-column ``(lo, hi)`` prior box, ``(-inf, +inf)`` where unknown.

    The same one-pass reader as :func:`prior_box_widths` (columns through
    :func:`_prior_entries`, so label-keyed dicts resolve). A column gets its
    distribution's ``(minimum, maximum)`` only when both are finite and
    ``maximum > minimum``; uncovered, unreadable, unbounded and degenerate
    columns get ``(-inf, +inf)``, which :func:`nudge_inside` leaves alone.

    Quiet by design: the refresh reads :func:`prior_box_widths` off the same
    container first, and that reports unread columns and reader failures.
    """
    lo_r, hi_r, covered, _, _ = _read_prior_box(prob_dist_container, ndim)
    lo = np.full(ndim, -np.inf)
    hi = np.full(ndim, np.inf)
    ok = np.zeros(ndim, dtype=bool)
    ok[covered] = True
    with np.errstate(invalid="ignore"):
        ok &= np.isfinite(lo_r) & np.isfinite(hi_r) & (hi_r - lo_r > 0)
    lo[ok] = lo_r[ok]
    hi[ok] = hi_r[ok]
    return lo, hi


def nudge_inside(x0, steps, lo, hi):
    """``x0`` moved so the ``x0 +/- steps`` stencil stays in ``[lo, hi]``.

    Per column with a positive step, clip to ``[lo + step, hi - step]``; a
    box narrower than ``2 * step`` takes the box centre instead. Infinite
    bounds and steps ``<= 0`` (a frozen column in
    :func:`~lisatools.info_matrix_ll.information_matrix_from_ll`) leave the
    column alone. ``x0`` is ``(ndim,)`` or ``(n, ndim)``; ``steps`` / ``lo``
    / ``hi`` broadcast against it. Returns a new array; ``x0`` is not
    mutated.

    Why (9mo job 751, 2026-10-09): an EMRI walker at ``cos(qK) =
    -0.9999770663`` in a ``uniform(-0.99999, 0.99999)`` box, 1.3e-5 from the
    bound, with central-difference steps of 2e-4 and up. The ``-`` row left
    the box, the cos -> arccos transform gave a NaN template, the Gram build
    failed every refresh and the likelihood fallback scored a "no signal"
    corner -- finite, so its table was silently wrong.
    """
    x = np.array(x0, dtype=float)
    s = np.broadcast_to(np.asarray(steps, dtype=float), x.shape)
    lo = np.broadcast_to(np.asarray(lo, dtype=float), x.shape)
    hi = np.broadcast_to(np.asarray(hi, dtype=float), x.shape)
    live = s > 0
    a = lo + s
    b = hi - s
    # fl(fl(lo + s) - s) can land one ulp below lo (and fl(fl(hi - s) + s)
    # one above hi); pull the limit in by that ulp so the stencil row, built
    # the same way, is inside
    a = np.where(a - s < lo, np.nextafter(a, np.inf), a)
    b = np.where(b + s > hi, np.nextafter(b, -np.inf), b)
    out = np.where(live, np.minimum(np.maximum(x, a), b), x)
    narrow = live & (a > b)
    out[narrow] = 0.5 * (lo[narrow] + hi[narrow])
    return out


def _read_prior_box(prob_dist_container, ndim):
    """One pass over :func:`_prior_entries`: the per-column box as read.

    Returns ``(lo, hi, covered, unread, exc)``: ``lo`` / ``hi`` carry the
    distributions' ``minimum`` / ``maximum`` on the ``covered`` columns (NaN
    elsewhere; a column covered twice keeps the later entry), ``unread``
    lists covered-but-unreadable columns, ``exc`` is the reader failure
    (``None`` if the pass completed; the columns read before it are kept).
    Shared by :func:`prior_box_widths` and :func:`prior_box_bounds` so the
    two never disagree on a column.
    """
    lo = np.full(ndim, np.nan)
    hi = np.full(ndim, np.nan)
    covered = []
    unread = []
    exc = None
    try:
        for cols, dist in _prior_entries(prob_dist_container):
            cols_all = [int(c) for c in cols.ravel()]
            keep = np.array([0 <= c < ndim for c in cols_all], dtype=bool)
            if not keep.any():
                continue
            cols_in = [c for c, k in zip(cols_all, keep) if k]
            _mn = getattr(dist, "minimum", getattr(dist, "min_val", None))
            _mx = getattr(dist, "maximum", getattr(dist, "max_val", None))
            if _mn is None or _mx is None:
                unread.extend(cols_in)
                continue
            try:
                # a scalar bound on a multi-column (tuple-keyed) entry
                # applies to each of its columns; a per-column array must
                # match the entry's columns one for one — masked TOGETHER
                # with the columns, so an out-of-range column cannot shift
                # the others' bounds
                mn = np.broadcast_to(np.asarray(_mn, dtype=float), (len(cols_all),))[keep]
                mx = np.broadcast_to(np.asarray(_mx, dtype=float), (len(cols_all),))[keep]
            except (TypeError, ValueError):
                unread.extend(cols_in)
                continue
            lo[cols_in] = mn
            hi[cols_in] = mx
            covered.extend(cols_in)
    except Exception as err:  # never break the sampler on an exotic prior
        exc = err
    return lo, hi, covered, unread, exc


# (ndim, columns) already reported by _warn_unread: addremove calls
# prior_box_widths once per leaf per refresh, so an unbounded column would
# otherwise log one identical line per leaf.
_UNREAD_WARNED = set()


def _warn_unread(ndim, cols):
    key = (int(ndim), tuple(cols))
    if key in _UNREAD_WARNED:
        return
    _UNREAD_WARNED.add(key)
    logger.warning(
        "[eigen_refresh] prior columns %s expose no finite positive "
        "(minimum, maximum) box; using unit width there — the eigen steps "
        "and the prior-box cap on those columns are NOT in the parameter's "
        "own units.", list(cols),
    )


def _fallback_table(widths):
    """Identity axes with 1%-of-prior-width steps — always usable."""
    widths = np.asarray(widths, dtype=float)
    return np.eye(widths.size), 1e-2 * widths


def _identity_tables(widths, n):
    """Batched identity fallback: ``n`` copies of :func:`_fallback_table`."""
    widths = np.asarray(widths, dtype=float)
    d = widths.size
    return (np.broadcast_to(np.eye(d), (n, d, d)).copy(),
            np.broadcast_to(1e-2 * widths, (n, d)).copy())


def _tables_from_info_batch(info, widths, sigma_max_frac=1.0):
    """Whiten -> eigen axes -> un-whiten -> prior cap, batched ``(n, d, d)``.

    Raises on non-finite input or output (callers catch and fall back).
    """
    widths = np.asarray(widths, dtype=float)
    info = np.asarray(info, dtype=float)
    if not np.all(np.isfinite(info)):
        raise ValueError("non-finite information matrix")
    # diag(w) @ info @ diag(w): the whitened matrix whose spectrum reflects
    # curvature, not unit choice
    info_y = info * widths[None, :, None] * widths[None, None, :]
    # whitened sigma_max=1.0 == one prior width along any axis
    axes_y, sig_y = eigen_axis_set(info_y, sigma_max=1.0)
    a_x = widths[None, :, None] * axes_y
    norms = np.linalg.norm(a_x, axis=1)
    norms = np.where(norms > 0, norms, 1.0)
    axes = a_x / norms[:, None, :]
    sigmas = sig_y * norms
    sigmas = np.minimum(sigmas, axis_caps(axes, widths, sigma_max_frac))
    if not (np.all(np.isfinite(axes)) and np.all(np.isfinite(sigmas))):
        raise ValueError("non-finite eigen table")
    return axes, sigmas


def axis_caps(axes, widths, sigma_max_frac=1.0):
    """The per-axis sigma cap every table is built with.

    ``sigma_max_frac`` x :func:`eryn.moves.eigenaxis.axis_prior_bounds` (the
    largest 1-sigma step along a unit axis that stays inside the prior box).
    ``axes`` is ``(..., ndim, ndim)`` (column ``k`` = axis ``k``); returns
    ``(..., ndim)``. One function for the table build
    (:func:`_tables_from_info_batch`) and the step-scale adaptation
    (:func:`adapt_axis_sigmas`), so an adapted axis is capped exactly as the
    information-matrix table caps it.
    """
    axes = np.asarray(axes, dtype=float)
    lead = axes.shape[:-2]
    d = axes.shape[-1]
    bounds = axis_prior_bounds(
        axes.reshape((-1, d, d)), np.asarray(widths, dtype=float)
    )
    return float(sigma_max_frac) * bounds.reshape(lead + (d,))


# ---------------------------------------------------------------------------
# Step-scale adaptation from the chain ({BRANCH}_EIGEN_AXIS_ADAPT)
# ---------------------------------------------------------------------------
#
# Why (user ruling 2026-10-14; 9mo job 763, rows 453-561, T_max 6 EMRI
# ladder): every EMRI samples a long degenerate intrinsic ridge (ln M, mu, a,
# p0, e0). The information-matrix table is the LOCAL curvature at one point,
# so the eigen move steps one table sigma along the right direction, but the
# chain's own cold spread along that direction was 1.4-56x the table sigma.
# A random walk whose step is r times smaller than the ridge needs ~r^2
# iterations to cross it; the four cold walkers sat at different positions on
# one connected ridge. The fix keeps the DIRECTIONS (axes) and widens the
# SCALE per axis to the chain's own robust width along it, never below the
# information-matrix sigma and never past the table's own cap.

#: 1.4826 x MAD is the standard deviation of a Gaussian.
MAD_TO_SIGMA = 1.4826


def wrap_periodic(dev, periodic_inds=(), periods=()):
    """``dev`` with its periodic columns wrapped to ``[-P/2, P/2)``.

    ``dev`` is ``(..., ndim)``; ``periodic_inds`` / ``periods`` are the
    branch's periodic columns and their periods (the move's
    :class:`eryn.utils.PeriodicContainer`). A copy is returned; non-periodic
    columns pass through unchanged. A phase on ``[0, 2 pi)`` maps to
    ``[-pi, pi)``.
    """
    out = np.array(dev, dtype=float, copy=True)
    inds = np.asarray(periodic_inds, dtype=int).ravel()
    if inds.size == 0:
        return out
    per = np.asarray(periods, dtype=float).ravel()
    out[..., inds] = np.mod(out[..., inds] + 0.5 * per, per) - 0.5 * per
    return out


def cold_unit_deviations(rung_samples, periodic_inds=(), periods=()):
    """Every buffered sample as a deviation in COLD units, one array per rung.

    ``rung_samples`` is a list over rungs of ``(X_k (m_k, ndim), beta_k
    (m_k,))``: rung ``k``'s pooled samples (every walker of the rank, every
    buffered visit) and the inverse temperature each was drawn at. Returns
    the list of ``(m_k, ndim)`` deviations (rungs with nothing usable
    dropped), or ``None`` when nothing usable remains;
    :func:`empirical_axis_widths` re-centres each rung in the axis basis.

    * **Centre: the rung's OWN robust centre**, never the cold rung's. At
      stationarity both coincide, but a hot rung that sits elsewhere
      (exploring another part of a ridge or another island, a fresh cold ->
      hot rotation copy, a re-anchored rung, burn-in) would add its centre
      OFFSET x sqrt(beta_k) to every one of its deviations -- a between-rung
      displacement, not a width -- and on a multi-modal leaf it would size
      the cold step to the distance between modes. With the rung's own
      centre the mapping is exact for a tempered Gaussian wherever each rung
      sits. Here the centre is the per-column median (for a periodic column
      the median of the wrapped offsets from the rung's first sample, so a
      phase straddling 0 / 2 pi is centred correctly); it only has to be
      good enough to wrap the periodic columns -- the final centring is the
      per-rung median of the AXIS-BASIS coordinates
      (:func:`empirical_axis_widths`), because a column-space centre error
      set by a wide axis would leak, rung to rung, into a narrow one.
    * **Wrap before projecting**: periodic deviations go to ``[-P/2, P/2)``
      (:func:`wrap_periodic`); an unwrapped phase near the period boundary
      would inflate every axis it feeds.
    * **Cold units**: x ``sqrt(beta_k)``. The target at inverse temperature
      ``beta`` is ``exp(beta lnL)``, whose Gaussian width is ``1 /
      sqrt(beta)`` times the cold one. Rungs with ``beta <= 0`` (no finite
      width) and non-finite rows are dropped.
    """
    devs = []
    for X, b in rung_samples:
        X = np.asarray(X, dtype=float)
        b = np.broadcast_to(np.asarray(b, dtype=float), (X.shape[0],))
        keep = (b > 0.0) & np.all(np.isfinite(X), axis=1) & np.isfinite(b)
        if not np.any(keep):
            continue
        X, b = X[keep], b[keep]
        ref = X[0]
        off = wrap_periodic(X - ref, periodic_inds, periods)
        centre = ref + np.median(off, axis=0)
        dev = wrap_periodic(X - centre, periodic_inds, periods)
        devs.append(dev * np.sqrt(b)[:, None])
    return devs or None


def empirical_axis_widths(axes, deviations):
    """Robust 1-sigma width of ``deviations`` along each table axis.

    ``axes`` is ``(..., ndim, ndim)`` with column ``k`` = axis ``k`` (the
    orientation of :func:`_tables_from_info_batch` and of
    :func:`eryn.moves.eigenaxis.draw_axis_step`, which steps
    ``sigma_k * axes[:, k]``); ``deviations`` is the per-rung list of
    cold-unit deviations (:func:`cold_unit_deviations`), or one ``(m,
    ndim)`` array (a single group). Returns ``(..., ndim)``.

    **Projection = the coordinates in the axis basis**, ``c = A^-1 dev``,
    not the dot products ``A^T dev``. The table's axes are unit vectors but
    NOT orthogonal in the sampling basis (they are orthonormal in the
    prior-box-whitened basis and un-whitened by the box widths). An eigen
    step ``x -> x + sigma_k z a_k`` changes ``c_k`` by ``sigma_k z`` and
    leaves every other ``c_j`` unchanged, so ``c_k``'s spread is exactly the
    scale ``sigma_k`` must be compared with. For orthonormal axes the two
    projections coincide.

    **Per-rung centre in the axis basis**: each group's ``c`` is re-centred
    on its own per-axis median before pooling (the column-space centre of
    :func:`cold_unit_deviations` carries an error set by the WIDE axes, ~
    ``1.25 sigma_wide / sqrt(m)``, and rung-to-rung differences of it would
    land on a narrow axis as spurious width -- 70 % too wide on a 1:500
    axis pair with 1500 samples per rung).

    **Robust width**: ``1.4826 x MAD`` of each pooled ``c_k`` (the Gaussian
    sigma; 50 % breakdown), not the plain standard deviation -- the buffer
    carries transient outliers (a hot rung freshly re-anchored or rotated,
    a temperature swap from far away). A singular axis matrix falls back to
    the pseudo-inverse.
    """
    axes = np.asarray(axes, dtype=float)
    groups = deviations if isinstance(deviations, (list, tuple)) else [deviations]
    try:
        inv = np.linalg.inv(axes)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(axes)
    coefs = []
    for D in groups:
        c = np.einsum("...ij,mj->...mi", inv, np.asarray(D, dtype=float))  # (..., m, ndim)
        coefs.append(c - np.median(c, axis=-2, keepdims=True))
    coef = np.concatenate(coefs, axis=-2)
    med = np.median(coef, axis=-2, keepdims=True)
    return MAD_TO_SIGMA * np.median(np.abs(coef - med), axis=-2)


def carried_axis_widths(axes, axes_old, sigmas_old):
    """Widths along ``axes`` implied by an OLDER adapted table.

    Reads the old table as the Gaussian proposal covariance it encodes,
    ``C = sum_k sigma_k^2 a_k a_k^T``, and returns the 1-sigma spread of the
    axis-basis coordinates ``A^-1 x`` under it (``sqrt(diag(A^-1 C
    A^-T))``) -- exactly ``sigmas_old`` when the axes did not change. Used
    after a restart, while the ring buffer refills: the sidecar carries the
    adapted scale across the process boundary, and a table refresh that
    lands before the buffer holds ``{BRANCH}_EIGEN_ADAPT_MIN_VISITS`` visits
    maps that scale onto the new axes instead of dropping it. Returns
    ``None`` when the shapes do not line up.
    """
    axes = np.asarray(axes, dtype=float)
    a_old = np.asarray(axes_old, dtype=float)
    s_old = np.asarray(sigmas_old, dtype=float)
    C = np.einsum("...ik,...k,...jk->...ij", a_old, s_old ** 2, a_old)
    try:
        inv = np.linalg.inv(axes)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(axes)
    V = inv @ C @ np.swapaxes(inv, -1, -2)
    out = np.sqrt(np.clip(np.diagonal(V, axis1=-2, axis2=-1), 0.0, None))
    if out.shape != axes.shape[:-1]:
        return None
    return out


def adapt_axis_sigmas(axes, sigmas_fisher, widths_emp, prior_widths,
                      sigma_max_frac=1.0):
    """``max(sigma_fisher, min(sigma_empirical, cap))`` per axis.

    The information-matrix sigma is the FLOOR (the adaptation only ever
    widens a step); the empirical width is capped exactly as the table is
    (:func:`axis_caps`), so an axis the matrix could not measure -- railed at
    the cap -- stays at the cap. For every table the builders produce
    (``sigma_fisher <= cap``) this is ``min(max(sigma_fisher,
    sigma_empirical), cap)``; a custom table above the cap keeps its own
    sigma. Non-finite empirical widths count as no information (the Fisher
    sigma). The axes are not touched.
    """
    sf = np.asarray(sigmas_fisher, dtype=float)
    se = np.asarray(widths_emp, dtype=float)
    se = np.where(np.isfinite(se), se, 0.0)
    cap = axis_caps(axes, prior_widths, sigma_max_frac)
    return np.maximum(sf, np.minimum(se, cap))


def _table_from_info(info, widths, sigma_max_frac=1.0):
    """Single-matrix wrapper over :func:`_tables_from_info_batch`."""
    axes, sigmas = _tables_from_info_batch(
        np.asarray(info, dtype=float)[None], widths,
        sigma_max_frac=sigma_max_frac,
    )
    return axes[0], sigmas[0]


def eigen_table_from_ll(call_ll, x0, widths, *, eps_rel=1e-4,
                        sigma_max_frac=1.0, xp=np, bounds=None):
    """``(axes, sigmas)`` from likelihood second differences at ``x0``.

    ``call_ll(params_2d) -> ll_1d`` scores rows in the SAMPLING basis (wrap
    any transform inside it) and must not mutate the residual. The
    per-parameter step is ``eps_rel`` of the prior box width; a point within
    one step of a bound would still put a corner outside the prior, so
    ``bounds=(lo, hi)`` (:func:`prior_box_bounds`) first moves ``x0`` that
    step inside (:func:`nudge_inside`). ``None`` uses ``x0`` as given.
    """
    widths = np.asarray(widths, dtype=float)
    try:
        x0 = np.asarray(x0, dtype=float)
        param_eps = float(eps_rel) * widths
        if bounds is not None:
            x0 = nudge_inside(x0, param_eps, *bounds)
        info = information_matrix_from_ll(
            call_ll, x0[None, :], xp=xp, param_eps=param_eps
        )
        info = np.asarray(info)[0]
        return _table_from_info(info, widths, sigma_max_frac=sigma_max_frac)
    except Exception as exc:
        logger.warning(
            "[eigen_refresh] information-matrix build from the likelihood "
            "failed (%r); using the identity fallback table", exc,
        )
        return _fallback_table(widths)


def eigen_tables_from_ll_batch(call_ll, x0s, widths, *, eps_rel=1e-4,
                               sigma_max_frac=1.0, xp=np, bounds=None):
    """One ``(axes, sigmas)`` table per row of ``x0s``, ONE batched sweep.

    ``x0s`` is ``(n, ndim)`` — e.g. every (temperature, walker) point of a
    leaf. All corners of all points go through ``call_ll`` together, so a
    batched scorer pays one dispatch. Rows reach ``call_ll`` as whole
    n-point blocks in the ``x0s`` row order (the
    :func:`~lisatools.info_matrix_ll.information_matrix_from_ll` batching
    invariant), so per-point metadata — a per-walker ``data_index`` —
    must be TILED by ``rows // n``. ``bounds=(lo, hi)`` nudges each row
    inside the prior box first, as in :func:`eigen_table_from_ll`.

    Returns ``axes (n, ndim, ndim)``, ``sigmas (n, ndim)``; any failure
    degrades to identity tables for every point with a logged warning.
    """
    widths = np.asarray(widths, dtype=float)
    x0s = np.atleast_2d(np.asarray(x0s, dtype=float))
    try:
        param_eps = float(eps_rel) * widths
        if bounds is not None:
            x0s = nudge_inside(x0s, param_eps, *bounds)
        info = information_matrix_from_ll(
            call_ll, x0s, xp=xp, param_eps=param_eps
        )
        return _tables_from_info_batch(
            np.asarray(info), widths, sigma_max_frac=sigma_max_frac
        )
    except Exception as exc:
        logger.warning(
            "[eigen_refresh] batched information-matrix build failed (%r); "
            "using identity fallback tables", exc,
        )
        return _identity_tables(widths, x0s.shape[0])


def eigen_table_from_waveform(waveform_model, params, widths, *,
                              eps_rel=1e-4, sigma_max_frac=1.0,
                              deriv_inds=None, parameter_transforms=None,
                              inner_product_kwargs=None,
                              waveform_kwargs=None, more_accurate=False):
    """``(axes, sigmas)`` from the Gram-form waveform information matrix.

    Delegates to :func:`lisatools.diagnostic.info_matrix` — ``params`` and
    the ``eps_rel``-of-prior-width derivative steps are in the SAMPLING
    basis, with ``parameter_transforms`` carrying the map to the waveform
    basis (fills included). ``inner_product_kwargs`` must weight by the
    PSD the run is actually using.
    """
    widths = np.asarray(widths, dtype=float)
    try:
        eps = float(eps_rel) * widths
        info = info_matrix(
            eps,
            waveform_model,
            np.asarray(params, dtype=float).copy(),
            deriv_inds=deriv_inds,
            inner_product_kwargs=inner_product_kwargs or {},
            parameter_transforms=parameter_transforms,
            waveform_kwargs=waveform_kwargs or {},
            more_accurate=more_accurate,
        )
        return _table_from_info(
            np.asarray(info), widths, sigma_max_frac=sigma_max_frac
        )
    except Exception as exc:
        logger.warning(
            "[eigen_refresh] information-matrix build from the waveform "
            "failed (%r); using the identity fallback table", exc,
        )
        return _fallback_table(widths)
