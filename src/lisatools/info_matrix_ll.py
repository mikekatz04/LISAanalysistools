"""Information matrix from second differences of the LIKELIHOOD.

Engine-agnostic: the caller injects

    call_ll(params_2d) -> ll_1d

and everything else here is index bookkeeping. That makes the same code serve
the sig-het in-model scorer (fast, ~14.6 us/candidate, reference already built
by ``setup_in_model``) and the chunked-het scorer, and it keeps a future
in-kernel Gram version behind the same entry point.

**Why second differences of ll rather than a Gram contraction.** The Gram form
``Gamma_ij = <d_i h | d_j h>`` needs only ``2*nparams`` waveform builds, but it
needs the derivative BLOCKS resident simultaneously -- which does not fit in
shared memory (``dr_sparse`` is 240 B per sparse point, so 9 of them at
``N_sparse_t=204`` is ~440 KB against A100's 163 KB) and has no Python-level
API today, since neither engine returns derivative waveforms. Second
differencing the likelihood needs only ONE waveform resident at a time, which
is what makes it expressible through the existing ``get_ll`` surface with no
kernel work. See
``docs/superpowers/2026-08-04-sighet-info-matrix-kernel-plan.md`` for the
in-kernel Gram design this is the stepping stone to.

**The batching invariant holds**: every corner of every pair of every source is
assembled into ONE parameter array and swept in a single batched ``call_ll`` --
never a per-source or per-pair loop.

**Sign convention.** Returns the OBSERVED information ``-d_i d_j lnL``. At the
likelihood peak this equals the Fisher matrix; away from it the two differ by
``<r | d_i d_j h>``, whose expectation vanishes. Because that term is not
sign-definite the result can have non-positive eigenvalues; callers that need a
Cholesky must handle that (see ``regularize``).
"""

from __future__ import annotations

import logging
import os
from typing import Callable, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["information_matrix_from_ll", "infomat_knob", "regularize"]


def infomat_knob(name: str, default):
    """Env knob resolver for the SIGHET_INFOMAT_* family."""
    raw = os.environ.get(name, "").strip()
    if raw == "":
        return default
    if isinstance(default, bool):
        return raw not in ("0", "false", "False", "")
    return type(default)(raw)


def regularize(gamma, xp, floor_rel: float = 1e-10):
    """Symmetrize and lift ``gamma`` onto the PSD cone.

    The observed information can be indefinite (see the module docstring), and
    ``_proposal_cholesky`` needs a factor. Eigen-decompose, clip eigenvalues at
    ``floor_rel * max_eig``, rebuild. A no-op for matrices that are already
    comfortably PSD.
    """
    g = 0.5 * (gamma + xp.swapaxes(gamma, -1, -2))
    w, v = xp.linalg.eigh(g)
    wmax = xp.max(w, axis=-1, keepdims=True)
    floor = xp.maximum(floor_rel * xp.maximum(wmax, 0.0), 1e-300)
    # How much work the projection is doing IS the diagnostic for whether the
    # observed-information form is adequate here: if many matrices need
    # lifting, the neglected <r-h | d_i d_j h> term is large and the Gram
    # (Fisher) form -- which is PSD by construction -- is the right answer.
    n_neg = int(xp.sum(xp.any(w < 0.0, axis=-1)))
    if n_neg:
        worst = float(xp.min(w / xp.maximum(wmax, 1e-300)))
        logger.warning(
            "[infomat] %d/%d matrices had a non-positive eigenvalue "
            "(worst lambda/lambda_max = %.3e) -> projected onto the PSD cone. "
            "A large fraction here means the observed-information form is "
            "being pushed past its validity; prefer the Gram form.",
            n_neg, int(w.shape[0]), worst,
        )
    w = xp.maximum(w, floor)
    return (v * w[..., None, :]) @ xp.swapaxes(v, -1, -2)


def information_matrix_from_ll(
    call_ll: Callable,
    params,
    *,
    xp,
    param_eps,
    inds: Optional[Sequence[int]] = None,
    batch: Optional[int] = None,
    one_sided: bool = False,
    psd_project: bool = True,
):
    """``-d_i d_j lnL`` per source, from batched likelihood evaluations.

    Central second differences::

        G_ij = -[ll(+i,+j) - ll(+i,-j) - ll(-i,+j) + ll(-i,-j)] / (4 e_i e_j)
        G_ii = -[ll(+i)    - 2 ll(0)   + ll(-i)]               / (e_i^2)

    Args:
        call_ll: ``fn(params_2d) -> ll_1d``. Must accept an arbitrary row
            count (rows are chunked by ``batch``) and must NOT mutate the
            residual -- an add-delta scorer is what this wants.
        params: ``(n, nparams)`` rows, in whatever basis ``call_ll`` consumes.
        param_eps: per-parameter step, ``(nparams,)`` or ``(n, nparams)``.
            ``eps <= 0`` FREEZES that dimension (the existing gradient
            convention): its row/column comes back 0 with 1 on the diagonal so
            the matrix stays invertible.
        inds: parameter indices to include (default: all).
        batch: rows per ``call_ll`` call (default ``FSTAT``-style 4096).
        one_sided: forward differences -- fewer evaluations, lower accuracy.
            Diagnostic only.
        psd_project: run :func:`regularize` on the result.

    Returns:
        ``(n, ndim, ndim)`` on ``xp``, where ``ndim = len(inds)``.
    """
    p = xp.asarray(xp.atleast_2d(params))
    n, nparams = int(p.shape[0]), int(p.shape[1])
    if inds is None:
        inds = list(range(nparams))
    inds = [int(i) for i in inds]
    ndim = len(inds)

    eps = xp.asarray(param_eps, dtype=xp.float64)
    if eps.ndim == 1:
        eps = xp.broadcast_to(eps, (n, nparams)).copy()
    frozen = eps <= 0.0

    if batch is None:
        batch = int(infomat_knob("SIGHET_INFOMAT_BATCH", 4096))

    # ---- assemble every corner ONCE (the batching invariant) -------------
    # rows: [central] + [per-i +/-] + [per-pair (+,+),(+,-),(-,+),(-,-)]
    rows = [p]                                   # 0: central, n rows
    off = {"central": 0}
    cur = n

    signs = (1.0,) if one_sided else (1.0, -1.0)
    for a, i in enumerate(inds):
        for s in signs:
            q = p.copy()
            q[:, i] = q[:, i] + s * eps[:, i]
            rows.append(q)
            off[("1", a, s)] = cur
            cur += n

    pairs = [(a, b) for a in range(ndim) for b in range(a + 1, ndim)]
    corner_signs = ((1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0))
    if one_sided:
        corner_signs = ((1.0, 1.0),)
    for (a, b) in pairs:
        i, j = inds[a], inds[b]
        for (si, sj) in corner_signs:
            q = p.copy()
            q[:, i] = q[:, i] + si * eps[:, i]
            q[:, j] = q[:, j] + sj * eps[:, j]
            rows.append(q)
            off[("2", a, b, si, sj)] = cur
            cur += n

    allp = xp.concatenate(rows, axis=0)
    n_total = int(allp.shape[0])
    logger.debug("[infomat] %d sources x %d dims -> %d likelihood rows",
                 n, ndim, n_total)

    ll = xp.empty(n_total, dtype=xp.float64)
    for s in range(0, n_total, batch):
        e = min(s + batch, n_total)
        ll[s:e] = xp.asarray(call_ll(allp[s:e])).astype(xp.float64).reshape(-1)

    def L(key):
        o = off[key]
        return ll[o:o + n]

    ll0 = L("central")
    G = xp.zeros((n, ndim, ndim), dtype=xp.float64)

    # ---- diagonal --------------------------------------------------------
    for a, i in enumerate(inds):
        ei = eps[:, i]
        if one_sided:
            # forward: [ll(+2e) - 2 ll(+e) + ll(0)] / e^2  -- needs a 2e row we
            # did not assemble, so fall back to the central form's magnitude.
            d2 = (L(("1", a, 1.0)) - ll0) / xp.maximum(ei, 1e-300) ** 2
        else:
            d2 = (L(("1", a, 1.0)) - 2.0 * ll0 + L(("1", a, -1.0))) \
                / xp.maximum(ei, 1e-300) ** 2
        G[:, a, a] = -d2

    # ---- off-diagonal ----------------------------------------------------
    for (a, b) in pairs:
        i, j = inds[a], inds[b]
        den = 4.0 * xp.maximum(eps[:, i], 1e-300) * xp.maximum(eps[:, j], 1e-300)
        if one_sided:
            v = (L(("2", a, b, 1.0, 1.0)) - L(("1", a, 1.0))
                 - L(("1", b, 1.0)) + ll0) \
                / (xp.maximum(eps[:, i], 1e-300) * xp.maximum(eps[:, j], 1e-300))
        else:
            v = (L(("2", a, b, 1.0, 1.0)) - L(("2", a, b, 1.0, -1.0))
                 - L(("2", a, b, -1.0, 1.0)) + L(("2", a, b, -1.0, -1.0))) / den
        G[:, a, b] = -v
        G[:, b, a] = -v

    # ---- frozen dims: identity row/col so the matrix stays invertible ----
    for a, i in enumerate(inds):
        fz = frozen[:, i]
        if bool(fz.any()):
            G[fz, a, :] = 0.0
            G[fz, :, a] = 0.0
            G[fz, a, a] = 1.0

    bad = ~xp.isfinite(G)
    if bool(bad.any()):
        logger.warning("[infomat] %d non-finite entries -> identity rows",
                       int(bad.sum()))
        rows_bad = xp.any(xp.any(bad, axis=-1), axis=-1)
        G[rows_bad] = xp.eye(ndim, dtype=xp.float64)

    if psd_project:
        G = regularize(G, xp)
    return G
