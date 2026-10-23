"""Fused CPU transfer kernel for unequal-arm TDI-2 instrument noise.

The generated closed forms in :mod:`lisatools._unequal_arm_expressions` are a
fully expanded version of the same calculation.  This module keeps the
factorisation instead: six link-delay phases are shared by all XYZ elements,
then the OMS and test-mass mixing matrices are projected once each.

Entry point: :func:`unequal_arm_tdi2_unit_covariances`, called by the unequal-arm instrument
noise in :mod:`lisatools.sensitivity` (the production noise model, ``UNEQUAL_ARM=1``) once per
locally stationary WDM time column to build the unit OMS / acceleration bases that each PSD
proposal rescales by ``Soms_d`` and ``Sa_a``. numpy only.
"""

from __future__ import annotations

import numpy as np

from .utils.constants import C_SI


# Directed link order of the ``ltts`` argument (link ``ij`` = light travelling
# from spacecraft j to spacecraft i); same order as lisatools.detector.LINKS.
LINKS = (12, 23, 31, 13, 32, 21)
_LINK_INDEX = {link: i for i, link in enumerate(LINKS)}

# X2 in eta measurements.  Y2 and Z2 are the two cyclic permutations.  These
# are the stock PyTDI second-generation Michelson terms, represented as
# (measurement link, delay links, coefficient).
_X2_TERMS = (
    (13, (), +1),
    (31, (13,), +1),
    (12, (13, 31), +1),
    (21, (13, 31, 12), +1),
    (12, (), -1),
    (21, (12,), -1),
    (13, (12, 21), -1),
    (31, (12, 21, 13), -1),
    (12, (13, 31, 12, 21), +1),
    (21, (13, 31, 12, 21, 12), +1),
    (13, (13, 31, 12, 21, 12, 21), +1),
    (31, (13, 31, 12, 21, 12, 21, 13), +1),
    (13, (12, 21, 13, 31), -1),
    (31, (12, 21, 13, 31, 13), -1),
    (12, (12, 21, 13, 31, 13, 31), -1),
    (21, (12, 21, 13, 31, 13, 31, 12), -1),
)


def _cyclic_link(link: int, permutation: int) -> int:
    """Shift both spacecraft indices of link ``ij`` cyclically by ``permutation`` (1->2->3->1)."""
    first, second = divmod(int(link), 10)
    first = (first - 1 + permutation) % 3 + 1
    second = (second - 1 + permutation) % 3 + 1
    return 10 * first + second


_XYZ_TERMS = tuple(
    tuple(
        (
            _LINK_INDEX[_cyclic_link(link, channel)],
            tuple(_LINK_INDEX[_cyclic_link(delay, channel)] for delay in delays),
            coefficient,
        )
        for link, delays, coefficient in _X2_TERMS
    )
    for channel in range(3)
)
_REVERSE_INDEX = np.asarray(
    [_LINK_INDEX[int(str(link)[::-1])] for link in LINKS], dtype=np.intp
)


def unequal_arm_tdi2_unit_covariances(f, ltts):
    """Return unit OMS and acceleration XYZ covariances on ``f``.

    Builds the TDI-2 Michelson transfer from the six link noises with all six light travel
    times independent (no equal-arm assumption), in relative-frequency units with the same
    OMS / test-mass shapes as :mod:`lisatools._unequal_arm_expressions`. Grids longer than
    4096 points use the compact X/Y/Z factorisation of the 16 TDI-2 terms, with the phases
    from a blockwise recurrence (re-anchored every 1024 samples) when the grid is also
    regularly spaced; grids of at most 4096 points evaluate the 16 terms directly with
    explicit phases. Diagonal elements are made exactly real and the lower triangle is the
    conjugate of the upper one, so the result is Hermitian by construction.

    Args:
        f (array-like): Frequencies in Hz, shape ``(Nf,)``. ``f = 0`` yields non-finite
            values (divide warnings suppressed).
        ltts (array-like): Directed light travel times in seconds, shape ``(6,)``, in
            :data:`LINKS` order.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``(B_oms, B_acc)``, complex128, each with shape
        ``(3, 3, Nf)`` (XYZ x XYZ x frequency). Multiplying by ``LISAModel.Soms_d`` and
        ``LISAModel.Sa_a`` respectively and summing gives the physical covariance.

    Raises:
        ValueError: If ``f`` is not one-dimensional or ``ltts`` does not have shape ``(6,)``.

    Example:
        >>> import numpy as np
        >>> from lisatools._unequal_arm_fused import unequal_arm_tdi2_unit_covariances
        >>> f = np.linspace(1e-4, 1e-2, 50)
        >>> b_oms, b_acc = unequal_arm_tdi2_unit_covariances(f, np.full(6, 8.34))
        >>> b_oms.shape
        (3, 3, 50)
    """
    f = np.asarray(f, dtype=float)
    ltts = np.asarray(ltts, dtype=float)
    if f.ndim != 1:
        raise ValueError(f"f must be one-dimensional; got shape {f.shape}.")
    if ltts.shape != (6,):
        raise ValueError(f"ltts must have shape (6,); got {ltts.shape}.")

    # The only complex exponentials in the calculation: every compound TDI
    # delay is a product of these six phases.  The expanded formulas evaluate
    # hundreds of equivalent exponentials independently for every matrix
    # element and noise basis.
    if f.size > 1:
        df = float(f[1] - f[0])
        regular = np.allclose(np.diff(f), df, rtol=1e-12, atol=0.0)
    else:
        regular = False
    if regular and f.size > 4096:
        # WDM's compact fold frequencies are a contiguous rFFT interval.  A
        # blockwise recurrence replaces ~3 million transcendental evaluations
        # per time column by complex multiplies, while re-anchoring often
        # enough to keep phase drift below floating-point cancellation noise.
        block_size = 1024
        step = np.exp(-2j * np.pi * ltts * df)
        within = np.empty((6, block_size), dtype=np.complex128)
        within[:, 0] = 1.0
        within[:, 1:] = np.cumprod(
            np.broadcast_to(step[:, None], (6, block_size - 1)), axis=1
        )
        phases = np.empty((6, f.size), dtype=np.complex128)
        for start in range(0, f.size, block_size):
            stop = min(start + block_size, f.size)
            anchor = np.exp(-2j * np.pi * ltts * f[start])
            phases[:, start:stop] = anchor[:, None] * within[:, : stop - start]
    else:
        phases = np.exp(-2j * np.pi * ltts[:, None] * f[None, :])
    eta_to_xyz = np.zeros((3, 6, f.size), dtype=np.complex128)
    if f.size <= 4096:
        # Keep the unsimplified term ordering on tiny/direct FD grids.  Near a
        # transfer null it matches the historical generated expressions'
        # cancellation rounding more closely; its cost is irrelevant here.
        for channel, terms in enumerate(_XYZ_TERMS):
            for measurement, delays, coefficient in terms:
                delayed = np.ones(f.size, dtype=np.complex128)
                for delay in delays:
                    delayed *= phases[delay]
                eta_to_xyz[channel, measurement] += coefficient * delayed
    else:
        for channel in range(3):
            # Compact factorisation of the 16 TDI-2 Michelson terms.  For X,
            # let a=D13 D31 and b=D12 D21.  The four nonzero coefficients are
            # eta13: (1-b)(1-ab), eta31: D13 times that, eta12:
            # (a-1)(1-ab), and eta21: D12 times that.  Y/Z are cyclic.
            a_forward = _LINK_INDEX[_cyclic_link(13, channel)]
            a_reverse = _LINK_INDEX[_cyclic_link(31, channel)]
            b_forward = _LINK_INDEX[_cyclic_link(12, channel)]
            b_reverse = _LINK_INDEX[_cyclic_link(21, channel)]
            a = phases[a_forward] * phases[a_reverse]
            b = phases[b_forward] * phases[b_reverse]
            common = 1.0 - a * b
            coefficient_b = (1.0 - b) * common
            coefficient_a = (a - 1.0) * common
            eta_to_xyz[channel, a_forward] = coefficient_b
            eta_to_xyz[channel, a_reverse] = phases[a_forward] * coefficient_b
            eta_to_xyz[channel, b_forward] = coefficient_a
            eta_to_xyz[channel, b_reverse] = phases[b_forward] * coefficient_a

    # eta_ij = N_oms_ij + N_tm_ij + D_ij N_tm_ji.  Hence a given N_tm_ij
    # enters eta_ij directly and eta_ji after D_ji.
    tm_to_xyz = np.empty_like(eta_to_xyz)
    for link_index, reverse_index in enumerate(_REVERSE_INDEX):
        tm_to_xyz[:, link_index] = (
            eta_to_xyz[:, link_index]
            + eta_to_xyz[:, reverse_index] * phases[reverse_index]
        )

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        oms_shape = (2.0 * np.pi * f / C_SI) ** 2 * (1.0 + (2.0e-3 / f) ** 4)
        acc_shape = (1.0 / (2.0 * np.pi * f * C_SI)) ** 2
        acc_shape *= (1.0 + (0.4e-3 / f) ** 2) * (1.0 + (f / 8.0e-3) ** 4)

    oms = np.einsum(
        "ikf,jkf,f->ijf", eta_to_xyz, eta_to_xyz.conj(), oms_shape, optimize=True
    )
    acc = np.einsum(
        "ikf,jkf,f->ijf", tm_to_xyz, tm_to_xyz.conj(), acc_shape, optimize=True
    )
    # Auto spectra are mathematically real.  Removing their roundoff-sized
    # imaginary parts preserves the exact Hermitian assignment contract.
    for channel in range(3):
        oms[channel, channel] = oms[channel, channel].real
        acc[channel, channel] = acc[channel, channel].real
    # Cross spectra are mathematically exact conjugate pairs -- the einsum is
    # ``A S A^dagger`` with real ``S``, and each summand of ``(j, i)`` is the
    # bitwise conjugate of the matching ``(i, j)`` summand.  But einsum
    # accumulates the two elements as INDEPENDENT reductions, so their last
    # bits agree only when the BLAS happens to sum symmetric pairs in the same
    # order.  It does on Accelerate/ARM and does not on the cluster's x86 BLAS
    # (observed 2026-08-28: ~1 ulp in Im, 56% of elements).  Mirror the upper
    # triangle so Hermiticity is a construction guarantee on every platform,
    # exactly like the diagonal-real step above; the WDM fold then yields an
    # exactly symmetric real covariance too.  No-op where the BLAS was already
    # symmetric, so validated numbers are unchanged there.
    for i in range(3):
        for j in range(i + 1, 3):
            oms[j, i] = oms[i, j].conj()
            acc[j, i] = acc[i, j].conj()
    return oms, acc


__all__ = ["LINKS", "unequal_arm_tdi2_unit_covariances"]
