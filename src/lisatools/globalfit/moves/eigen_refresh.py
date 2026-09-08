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
    "eigen_table_from_ll",
    "eigen_tables_from_ll_batch",
    "eigen_table_from_waveform",
]

logger = logging.getLogger(__name__)


def prior_box_widths(prob_dist_container, ndim):
    """Per-column prior box widths from an eryn prior container.

    Mirrors the GB move's ``_eigen_axis_widths`` reader: eryn's uniform
    exposes ``minimum``/``maximum`` (the ``min_val``/``max_val`` spelling
    belongs to other distributions — try both rather than silently falling
    back to unit widths). Columns without a scalar distribution, and any
    reader failure, fall back to width 1.0 so the table build degrades
    instead of crashing.
    """
    lo = np.zeros(ndim)
    hi = np.ones(ndim)
    try:
        pri = prob_dist_container.priors_in
        for col, dist in pri.items():
            idx = col if isinstance(col, (int, np.integer)) else None
            if idx is None or not (0 <= int(idx) < ndim):
                continue
            _mn = getattr(dist, "minimum", getattr(dist, "min_val", None))
            _mx = getattr(dist, "maximum", getattr(dist, "max_val", None))
            if _mn is None or _mx is None:
                continue
            lo[int(idx)] = float(_mn)
            hi[int(idx)] = float(_mx)
    except Exception as exc:  # never break the sampler on an exotic prior
        logger.warning(
            "[eigen_refresh] prior box unavailable (%r); falling back to "
            "unit widths", exc,
        )
    return prior_box_scales(lo, hi)


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
    bounds = axis_prior_bounds(axes, widths)
    sigmas = np.minimum(sigmas, float(sigma_max_frac) * bounds)
    if not (np.all(np.isfinite(axes)) and np.all(np.isfinite(sigmas))):
        raise ValueError("non-finite eigen table")
    return axes, sigmas


def _table_from_info(info, widths, sigma_max_frac=1.0):
    """Single-matrix wrapper over :func:`_tables_from_info_batch`."""
    axes, sigmas = _tables_from_info_batch(
        np.asarray(info, dtype=float)[None], widths,
        sigma_max_frac=sigma_max_frac,
    )
    return axes[0], sigmas[0]


def eigen_table_from_ll(call_ll, x0, widths, *, eps_rel=1e-4,
                        sigma_max_frac=1.0, xp=np):
    """``(axes, sigmas)`` from likelihood second differences at ``x0``.

    ``call_ll(params_2d) -> ll_1d`` scores rows in the SAMPLING basis (wrap
    any transform inside it) and must not mutate the residual. The
    per-parameter step is ``eps_rel`` of the prior box width, so the
    corners stay well inside the prior for any reasonable ``x0``.
    """
    widths = np.asarray(widths, dtype=float)
    try:
        x0 = np.asarray(x0, dtype=float)
        param_eps = float(eps_rel) * widths
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
                               sigma_max_frac=1.0, xp=np):
    """One ``(axes, sigmas)`` table per row of ``x0s``, ONE batched sweep.

    ``x0s`` is ``(n, ndim)`` — e.g. every (temperature, walker) point of a
    leaf. All corners of all points go through ``call_ll`` together, so a
    batched scorer pays one dispatch. Rows reach ``call_ll`` as whole
    n-point blocks in the ``x0s`` row order (the
    :func:`~lisatools.info_matrix_ll.information_matrix_from_ll` batching
    invariant), so per-point metadata — a per-walker ``data_index`` —
    must be TILED by ``rows // n``.

    Returns ``axes (n, ndim, ndim)``, ``sigmas (n, ndim)``; any failure
    degrades to identity tables for every point with a logged warning.
    """
    widths = np.asarray(widths, dtype=float)
    x0s = np.atleast_2d(np.asarray(x0s, dtype=float))
    try:
        info = information_matrix_from_ll(
            call_ll, x0s, xp=xp, param_eps=float(eps_rel) * widths
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
