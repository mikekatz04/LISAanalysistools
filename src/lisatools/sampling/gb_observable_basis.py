r"""Observable-basis bijection for GB proposals: ``y <-> z``.

The 9-column GB sampling basis
``y = [dist (kpc), f0 (mHz), Mc (Msun), phi0, cos_iota, psi, alpha, sin_delta, r]``
(ICRS sky; ``r = fdot_astro_ratio``) is the right thing to SAMPLE in -- ``f0``
anchored at ``t_ref`` is stable as
the mission's observation span grows, and bands / cap cells / storage are all
keyed on it. It is the wrong thing to PROPOSE in, for two independent reasons
measured on the flagship (20.380377 mHz, SNR 46) in 2026-08/09:

1. **3->2 redundancy.** ``(dist, Mc, r)`` enter the waveform only through
   ``(A, fdot)`` -- ``A = A(dist, f0, Mc)`` has no ``r`` and
   ``fdot = fdot_gr(f0, Mc)(1 + r)`` has no ``dist`` -- so one direction is
   EXACTLY likelihood-flat (``t^T F t / lam_max = 4.7e-26``). Worse, ``Mc``
   and ``r`` both drive ``fdot``, so a proposal can move both and leave
   ``fdot_total`` untouched: one measured eigen-axis moves ``r`` by 0.61 and
   ``ln(fdot)`` by 0.0062.

2. **The shear.** ``f0`` is the frequency at the START of the data; the data
   constrains the frequency at the MIDDLE (the mean frequency of a linear
   chirp is the frequency at its midpoint, since the accumulated cycle count
   is ``T * f(T/2)``). ``f0 = f_mid - fdot*T/2`` is a shear, and a shear turns
   an uncorrelated pair into a correlated one. The induced ``f0`` offset is
   ``-(T/2)*fdot*T``, i.e. ``propto fdot*T^2``: 0.04 bins at 6.8 mHz, 3.1 bins
   at 20.4 mHz (verified across the band, measured/analytic ratio 0.95-1.06).
   **That scaling is why high frequency has been the hard region.**

So proposals convert here, propose in
``z = [lnA, f_mid (Hz), fdot (Hz/s), phi0, cos_iota, psi, alpha, sin_delta, Mc]``
where ``fdot`` is a coordinate (nothing can cancel in it) and ``f_mid``
decorrelates the shear, then convert back. ``Mc`` is the fiber coordinate: the
exactly-flat direction, along which the likelihood cannot change. The map is

.. math::

    \ln A = \ln A(f_0, M_c, d), \qquad
    f_{\rm mid} = f_0 + c_t\,\dot f, \qquad
    \dot f = \dot f_{\rm GR}(f_0, M_c)\,(1 + r), \qquad c_t = s\,T_{\rm obs},

with shear fraction ``s = 0.5`` by default and the extrinsics passed through.

**The measure.**
``|dy/dz| = dist / fdot_gr(f0, Mc)`` -- with ``y = (dist, f0, Mc, r)`` and
``z = (lnA, f_mid, fdot, Mc)`` the Jacobian is block-triangular and expanding on
the ``lnA`` column gives ``(-dist) * (-1/fdot_gr)``:

.. math::

    \left|\frac{\partial y}{\partial z}\right|
        = \frac{d}{\dot f_{\rm GR}(f_0, M_c)}.

Verified three ways:
analytically; numerically (ratio to analytic 0.9999987, and **identical for
shear coefficients 0, T/2, 0.41T, T, -3T** -- the shear is unit-determinant so
it contributes nothing); and by prior-invariance simulation under a flat
likelihood, where the correct sign preserves all nine marginals (min KS
p = 0.084) while each of four injected defects collapses at least one below
1e-6. See ``tests/test_gb_observable_basis_invariance.py`` -- note that no
SINGLE marginal catches every defect, so the controls take a minimum over
``{dist, Mc, r}``.

**A wrong shear coefficient costs efficiency only, never correctness.** That
is what the determinant result buys, and it is why ``Tobs`` may be recomputed
per run from the current span with no risk.

Consumed by the in-model proposal (chain side, via the class) and by the
F-stat grid (scalar half only -- grid rows are the physical waveform layout
and carry no ``transform_container``).

**The VGB restriction.**
:class:`VGBObservableBasis` is the SAME map with fewer columns, not a second
implementation: the verification-binary branch pins ``f0``, ``alpha``,
``sin_delta`` (and, in the default basis, ``Mc``) per LEAF in the transform
container's per-leaf ``fill_dict``, so its sampled side is
``y = [dist, phi0, cos_iota, psi, r]`` and its observable side is
``z = [lnA, fdot, phi0, cos_iota, psi]``. Every formula below is shared; the
subclass only says which columns are sampled and where the pinned constants
come from. Two structural consequences, both deliberate:

* **No f_mid.** The shear exists to decorrelate a SAMPLED ``f0`` from
  ``fdot``. With ``f0`` pinned there is nothing to decorrelate and a
  ``f_mid`` column would be ``fdot`` in disguise (``f_mid = f0 + c_t*fdot``
  at constant ``f0``).
* **No fiber.** ``(dist, Mc, r) -> (A, fdot)`` is 3->2 only when all three
  are sampled. With ``Mc`` pinned it is 2->2, ``FIBER_INDEX`` is ``None``,
  and the generic (no-fiber) eigen set applies. (Under
  ``VGB_CHIRP_MASS_BASIS=1`` ``Mc`` IS sampled, the derived layout grows the
  ``Mc`` column back, and the fiber returns -- same code path.)

``|dy/dz| = dist / fdot_gr(f0, Mc)`` is UNCHANGED by the restriction: with
``f0``/``Mc`` constant the map is block-DIAGONAL (``ddist/dlnA = -dist``,
``dr/dfdot = 1/fdot_gr``) rather than block-triangular, and the determinant
comes out to the same expression -- which is why both bases call the one
:meth:`GBObservableFiberBasis.log_jacobian`.

**Entry points and callers** (this module reads no env knob itself):

* :class:`GBObservableFiberBasis` / :class:`VGBObservableBasis` and
  :func:`gb_observable_step_scales` -- the in-model observable proposal of
  ``GBSpecialStretchMove`` / ``VGBSpecialStretchMove``
  (:mod:`lisatools.globalfit.moves.gbspecialstretch`), selected by
  ``GB_INMODEL_PROPOSAL=observable`` / ``VGB_INMODEL_PROPOSAL=observable`` in the
  9mo/12mo launchers. The move builds the map with ``Tobs = 1/df`` and
  ``shear = GB_INMODEL_OBSERVABLE_SHEAR`` (launchers 0.5) and sizes the steps
  from ``GB_INMODEL_OBSERVABLE_MC_STEP`` (launchers 0.05) and
  ``GB_INMODEL_OBSERVABLE_JUMP`` / ``VGB_INMODEL_OBSERVABLE_JUMP`` (launchers 1.5 /
  1.0). The warm-start tools (:mod:`lisatools.globalfit.warmstart.basis`) build
  the same map.
* :func:`fdot_gr`, :func:`fdot_axis_bounds`, :func:`n_fdot_nodes`,
  :func:`fdot_coherence_width`, :func:`mc_floor_for_fdot`, :func:`r_from_fdot` --
  the F-stat grid's linear fdot axis (``FSTAT_FDOT_AXIS=1``, the launcher value):
  :func:`lisatools.sampling.fstat_gridfit.run_stacked_stage_b`,
  :func:`~lisatools.sampling.fstat_gridfit.enumerate_center_nodes`,
  :class:`~lisatools.sampling.fstat_proposal.FdotAxisBirth` and
  :class:`~lisatools.sampling.fstat_proposal.CombIntrinsicProposal`.
* :func:`sky_doppler_alias_jump_batch` -- the search-only sky-Doppler jump
  (``GB_DOPPLER_JUMP_EVERY``, default 0; off in the launchers).
"""
from __future__ import annotations

import numpy as np

from ..utils.utility import get_array_module

__all__ = [
    "FDOT_K",
    "GB_INTERNAL_BASIS",
    "OBSERVABLE_CD_FLOORS",
    "OBSERVABLE_CD_FLOOR_DEFAULT",
    "GBObservableFiberBasis",
    "VGBObservableBasis",
    "f0_from_f_mid",
    "f_mid_from_f0",
    "fdot_axis_bounds",
    "fdot_coherence_width",
    "fdot_gr",
    "fdot_shear_hz",
    "gb_observable_step_scales",
    "mc_floor_for_fdot",
    "n_fdot_nodes",
    "r_from_fdot",
    "sky_doppler_alias_jump",
]

# Constants taken from the SAME source as gbgpu.utils.utility.get_fdot /
# get_amplitude so the grid rows and this module cannot drift. Pinned by
# test_fdot_gr_matches_gbgpu.
try:                                        # pragma: no cover - import shape
    from gbgpu.utils.utility import MSUN_SI as _MSUN_SI, G_SI as _G_SI, C_SI as _C_SI
except ImportError:                         # pragma: no cover
    from ..utils.constants import MSUN_SI as _MSUN_SI, G_SI as _G_SI, C_SI as _C_SI

#: ``fdot_gr = FDOT_K * Mc[Msol]**(5/3) * f0[Hz]**(11/3)``, with
#: ``FDOT_K = (96/5) pi^(8/3) (G Msun / c^3)^(5/3)`` (lisaconstants values via
#: gbgpu). Meant to absorb the two duplicate definitions in
#: ``sampling/fstat_proposal.py`` (``_fdot_gr`` / ``_MSUN_S`` and ``_FDOT_K``),
#: which still exist there (2026-10-19).
FDOT_K = (96.0 / 5.0) * np.pi ** (8 / 3) * (_G_SI * _MSUN_SI / _C_SI ** 3) ** (5 / 3)

_TWO_POW_1_5 = 2.0 ** (1.0 / 5.0)

# Analytic 1/rho step-scale coefficients for the (lnA, f_mid, fdot) block.
# EFFICIENCY ONLY -- these set a symmetric step size, not a Jacobian, so a
# wrong value costs acceptance and never correctness. Do not "fix" one under
# time pressure and then worry about bias; there is none.
#
# All three are the FULL marginal 1-sigma at signal-to-noise ``rho``, from
# the same one-radian criterion, so they stay comparable:
#   lnA :  rho propto A          => sigma(lnA) = 1/rho
#   f_mid, fdot: with the decorrelated form ``Q = a'**2/12 + b**2/720``
#     (``a' = df_mid*T``, ``b = dfdot*T**2``; see fdot_coherence_width) the
#     rms phase residual ``2*pi*sqrt(Q)`` equals ``1/rho`` at one sigma, so
#       sigma(a') = sqrt(12)/(2 pi rho)  = 0.5513/rho   [bins]
#       sigma(b)  = sqrt(720)/(2 pi rho) = 4.2705/rho   [bins per Tobs]
# NOTE 2026-09-01: STEP_C_FDOT was 2.14, i.e. HALF its marginal while the
# other two carried their full one. That is the coordinate the whole change
# exists to move, so the inconsistency cost exactly the wrong thing.
STEP_C_LNA = 1.0        # dimensionless
STEP_C_FMID = 0.5513    # in frequency bins (1/Tobs)
STEP_C_FDOT = 4.2705    # in bins per Tobs (1/Tobs**2)

#: Internal (observable) coordinate names of the FULL 9-column GB basis.
#: Every consumer of a reduced layout must read the map's own
#: ``INTERNAL_BASIS`` instead of assuming these positions -- the VGB
#: restriction drops ``f_mid`` / ``alpha`` / ``sin_delta`` / ``Mc``, so a
#: hard-coded "column 2 is fdot" is right for GB and wrong for VGB.
GB_INTERNAL_BASIS = ("lnA", "f_mid", "fdot", "phi0", "cos_iota", "psi",
                     "alpha", "sin_delta", "Mc")

#: Pass-through (extrinsic) coordinates, in canonical internal order. A map
#: carries exactly the ones its sampling basis carries -- the VGB branch
#: pins ``alpha``/``sin_delta`` per leaf, so its layout has neither.
_EXTRINSIC_NAMES = ("phi0", "cos_iota", "psi", "alpha", "sin_delta")

#: ABSOLUTE floors for a central-difference step in the internal basis,
#: keyed by NAME. ``fdot`` lives at ~1e-16 and ``lnA`` at ~1e1, so one
#: relative step size cannot serve both; the angle columns are pure
#: passthrough in both directions of the map, so a tiny step is exact
#: there. Keyed by name rather than by position precisely so a REDUCED
#: layout cannot silently inherit the 9-column ordering (``[:ndim]`` of a
#: positional list hands the VGB ``fdot`` column ``f_mid``'s floor).
OBSERVABLE_CD_FLOORS = {"lnA": 1e-1, "f_mid": 1e-6, "fdot": 1e-22}
OBSERVABLE_CD_FLOOR_DEFAULT = 1e-3


def fdot_gr(f0_hz, mc):
    r"""GR-driven ``fdot`` from ``(f0[Hz], Mc[Msol])``. Elementwise, xp-agnostic.

    .. math::

        \dot f_{\rm GR} = K\, M_c^{5/3} f_0^{11/3}, \qquad
        K = \frac{96}{5}\pi^{8/3}\left(\frac{G M_\odot}{c^3}\right)^{5/3}

    with ``K`` = :data:`FDOT_K`. The same expression as
    :func:`gbgpu.utils.utility.get_fdot` (pinned by ``test_fdot_gr_matches_gbgpu``).
    Used by the observable map, the F-stat fdot
    axis, :class:`~lisatools.sampling.fstat_proposal.FdotAxisBirth` and
    :class:`~lisatools.sampling.fstat_proposal.CombIntrinsicProposal`, and by
    ``gbspecialstretch``.

    Args:
        f0_hz (float or xp.ndarray): GW frequency in Hz (NOT the mHz sampling
            column).
        mc (float or xp.ndarray): Chirp mass in solar masses.

    Returns:
        float or xp.ndarray: ``fdot_gr`` in Hz/s, broadcast over the inputs, on
        their array module.
    """
    return FDOT_K * mc ** (5.0 / 3.0) * f0_hz ** (11.0 / 3.0)


def _shear_coeff(tobs, c_t=None):
    """Shear coefficient in seconds: ``tobs / 2`` when ``c_t`` is None, else ``c_t``."""
    return 0.5 * float(tobs) if c_t is None else float(c_t)


# TODO check: keep/remove/document -- not reached by the 9mo/12mo runs; only caller is tests/test_gb_observable_basis.py (GBObservableFiberBasis and FdotAxisBirth inline the same shear)
def f_mid_from_f0(f0_hz, fdot, tobs, *, c_t=None):
    """``f_mid = f0 + c_t * fdot`` with ``c_t = T/2`` by default.

    The mean frequency of a linear chirp over ``[0, T]``; measured
    ``<t> = 0.498 T``, i.e. uniform weighting.
    """
    return f0_hz + _shear_coeff(tobs, c_t) * fdot


# TODO check: keep/remove/document -- not reached by the 9mo/12mo runs; only caller is tests/test_gb_observable_basis.py
def f0_from_f_mid(f_mid_hz, fdot, tobs, *, c_t=None):
    """Exact inverse of :func:`f_mid_from_f0` at the same ``fdot``."""
    return f_mid_hz - _shear_coeff(tobs, c_t) * fdot


# TODO check: keep/remove/document -- not reached by the 9mo/12mo runs; only caller is tests/test_gb_observable_basis.py
def fdot_shear_hz(mc, f0_ref_hz, tobs, *, c_t=None):
    """The ``f0`` offset a GR chirp of this ``(f0_ref, Mc)`` induces.

    ``c_t * fdot_gr(f0_ref, mc)``. ``f0_ref`` is a per-box CONSTANT reference
    for the F-stat grid, never the node's own ``f0`` -- using the node's own
    value makes the grid shear's determinant ``1 - 2.6e-4`` instead of exactly
    1, a needless inexactness.
    """
    return _shear_coeff(tobs, c_t) * fdot_gr(f0_ref_hz, mc)


# TODO check: keep/remove/document -- not reached by the 9mo/12mo runs; only caller is tests/test_sky_doppler_jump.py (scalar reference for the batch version)
def sky_doppler_alias_jump(f0_hz, fdot_hz_s, alpha, sin_delta, *, sky_step,
                           v_over_c, times):
    """Sky-Doppler degeneracy jump: shift the sky, return the compensating (f0, fdot).

    A GB's OBSERVED instantaneous frequency carries the orbital-Doppler (Roemer)
    term ``f_obs(t) = f0 + fdot*t + f0*(v(t)·n̂)/c``. Over a short (< 1 yr) window
    the quarter-annual ``v·n̂`` is ~offset+slope in ``t`` -- indistinguishable
    from an f0/fdot change -- so a source has a *second mode* at a shifted sky
    whose Doppler mimics the true chirp (verified on band 1100: 23 deg sky shift
    reproduces the +7.6-bin f0 offset).

    This shifts the sky by ``sky_step`` and returns the intrinsic ``(Δf0, Δfdot)``
    that cancel the OFFSET and SLOPE of the induced Doppler over ``times``, so
    the jumped template shares the same observed track to first order and lands
    on the alias. The residual curvature (what a linear f0/fdot cannot cancel)
    is the imperfect overlap -- the measured ~9-nat gap the MH ratio then judges.
    It is a deterministic map given ``sky_step``; the reverse jump (negated step
    from the new point) returns the original to first order. An exact-detailed-
    balance caller multiplies by the sky-draw Jacobian; a search caller may drop
    it (search waiver).

    Args:
        f0_hz, fdot_hz_s, alpha, sin_delta: current source parameters (scalars).
        sky_step: ``(Δalpha, Δsin_delta)``.
        v_over_c: ``(N, 3)`` constellation velocity / c over the window, in the
            SAME frame as ``alpha``/``sin_delta`` (ICRS for the production run).
        times: ``(N,)`` window times (s), aligned with ``v_over_c`` rows.

    Returns:
        ``(f0_new, fdot_new, alpha_new, sin_delta_new)``.
    """
    xp = get_array_module(v_over_c)
    da, dsd = float(sky_step[0]), float(sky_step[1])
    sd1 = min(1.0, max(-1.0, float(sin_delta)))
    sd2 = min(1.0, max(-1.0, float(sin_delta) + dsd))
    a2 = alpha + da

    def _nh(al, sd):
        cd = np.sqrt(max(0.0, 1.0 - sd * sd))
        return xp.asarray([cd * np.cos(al), cd * np.sin(al), sd],
                          dtype=getattr(v_over_c, "dtype", float))

    dn = _nh(a2, sd2) - _nh(alpha, sd1)          # Δn̂ (3,)
    D = v_over_c @ dn                             # (N,) = Δ(v·n̂)/c
    rhs = -float(f0_hz) * D                        # target: Δf0 + Δfdot*t = rhs
    A = xp.stack([xp.ones_like(times), times], axis=-1)
    coef = xp.linalg.lstsq(A, rhs, rcond=None)[0]
    df0 = float(coef[0])
    dfdot = float(coef[1])
    return f0_hz + df0, fdot_hz_s + dfdot, a2, sd2


# TODO check: keep/remove/document -- only reachable with GB_DOPPLER_JUMP_EVERY>0 (default 0), which no launcher sets; caller gbspecialstretch._doppler_jump_proposal
def sky_doppler_alias_jump_batch(f0_hz, fdot_hz_s, alpha, sin_delta, *,
                                 sky_step, v_over_c, times):
    """Vectorized :func:`sky_doppler_alias_jump` over a batch of ``n`` sources.

    ``f0_hz``/``fdot_hz_s``/``alpha``/``sin_delta`` are ``(n,)``; ``sky_step`` is
    ``(n, 2)`` (``[Δalpha, Δsin_delta]``); ``v_over_c`` is ``(N, 3)`` and
    ``times`` ``(N,)`` (shared window). Returns four ``(n,)`` arrays
    ``(f0_new, fdot_new, alpha_new, sin_delta_new)``. Matches the scalar
    function row-for-row; used by the GB in-model Doppler-jump proposal.
    """
    xp = get_array_module(v_over_c)
    f0 = xp.asarray(f0_hz)
    fd = xp.asarray(fdot_hz_s)
    al = xp.asarray(alpha)
    sd = xp.asarray(sin_delta)
    st = xp.asarray(sky_step)
    da = st[:, 0]
    dsd = st[:, 1]
    sd1 = xp.clip(sd, -1.0, 1.0)
    sd2 = xp.clip(sd + dsd, -1.0, 1.0)
    al2 = al + da

    def _nh(a, s):
        cd = xp.sqrt(xp.clip(1.0 - s * s, 0.0, None))
        return xp.stack([cd * xp.cos(a), cd * xp.sin(a), s], axis=-1)  # (n,3)

    dn = _nh(al2, sd2) - _nh(al, sd1)             # (n, 3) Δn̂
    D = dn @ v_over_c.T                            # (n, N) = Δ(v·n̂)/c
    rhs = -(f0[:, None]) * D                        # (n, N): Δf0 + Δfdot t = rhs
    A = xp.stack([xp.ones_like(times), times], axis=-1)   # (N, 2)
    pinvA = xp.linalg.pinv(A)                      # (2, N); shared across sources
    coef = (pinvA @ rhs.T).T                        # (n, 2) -> [Δf0, Δfdot]
    return f0 + coef[:, 0], fd + coef[:, 1], al2, sd2


def fdot_coherence_width(tobs, *, aligned=False, eta=1.0):
    r"""``fdot`` node spacing at the one-radian criterion.

    With ``a = df0*T``, ``b = dfdot*T**2`` and uniform weighting the residual
    phase variance (in cycles squared) is

    .. math::

        Q = \frac{a^2}{12} + \frac{ab}{12} + \frac{b^2}{45}
          = \frac{a'^2}{12} + \frac{b^2}{720}, \qquad a' = a + \frac{b}{2}

    (the shear) -- the cross term vanishes exactly and ``det`` is preserved
    (``1/8640`` both ways), an independent confirmation of the unit-determinant
    result.

    So the *irreducible* fdot residual is ``b**2/720`` rather than the raw
    ``b**2/45 + cross``, and one radian at the end of the observation allows a
    node spacing ``sqrt(720)/(2 pi)`` instead of ``1/pi`` -- **13.4x coarser at
    the same criterion**.

    ``aligned=False`` reproduces the legacy Mc-axis rule of
    :func:`lisatools.sampling.fstat_proposal.fstat_n_mc` (``eta / (pi T^2)``)
    byte-identically; the fdot axis (:func:`n_fdot_nodes`, the production
    ``FSTAT_FDOT_AXIS=1`` path) uses ``aligned=True``.

    Args:
        tobs (float): Observation span ``T`` in seconds.
        aligned (bool, optional): Use the sheared width ``eta sqrt(720) /
            (2 pi T^2)`` instead of ``eta / (pi T^2)``. Default ``False``.
        eta (float, optional): Node spacing in units of the width. Default ``1.0``.

    Returns:
        float: The fdot node spacing in Hz/s.
    """
    t2 = float(tobs) ** 2
    if aligned:
        return float(eta) * np.sqrt(720.0) / (2.0 * np.pi * t2)
    return float(eta) / (np.pi * t2)


# ---------------------------------------------------------------------------
# F-stat grid: fdot as a FIRST-CLASS axis, and the conversion back.
#
# The grid's Mc axis is already a fdot axis -- just a bad one. It assembles
# rows as ``fdot = fdot_gr(f0, Mc_node)``, i.e. the ``r = 0`` manifold, so
# uniform-in-Mc nodes are NON-uniform in fdot (``fdot ~ Mc^(5/3)``) while
# StackedFStatProposal4D hard-assumes uniform axes, and negative fdot is
# UNREPRESENTABLE -- against 40% of low-f and 21% of high-f v7 leaves that
# carry ``fdot < 0``. Replacing it with a LINEAR fdot axis in Hz/s fixes
# coverage, spacing and the uniform-axis assumption at once.
# ---------------------------------------------------------------------------

def fdot_axis_bounds(f0_ref_hz, mc_hi, ratio_max):
    """Reachable ``fdot`` over the whole prior box. ``(lo, hi)`` in Hz/s.

    ``fdot = fdot_gr(f0, Mc) * (1 + r)`` with ``Mc`` in ``[mc_lo, mc_hi]``
    and ``r`` in ``[-M, M]``. ``fdot_gr > 0`` and increasing in ``Mc``, so
    both extremes sit at ``Mc = mc_hi``: ``(1 -+ M) * fdot_gr(f0, mc_hi)``.
    Called per peak box by
    :func:`lisatools.sampling.fstat_gridfit.run_stacked_stage_b` under
    ``FSTAT_FDOT_AXIS=1`` (the group's axis is the intersection of its boxes'
    bounds).

    **``mc_lo`` deliberately does not appear.** The tempting form
    ``[fdot_gr(mc_lo)(1-M), fdot_gr(mc_hi)(1+M)]`` looks symmetric and is
    wrong: at the production ``mc_lo = 0.001`` its lower bound is 1e-5 of
    the upper, so the axis would come out effectively one-sided and the
    negative-fdot defect would survive the redesign untouched.

    ``f0_ref_hz`` is a per-GROUP CONSTANT (never the node's own f0). That
    is what keeps ``fdot`` independent of ``f0``, which in turn makes
    ``f0 = f_mid - (T/2) fdot`` an exact unit-determinant shear. With the
    node's own f0 the determinant is ``1 - 2.6e-4`` -- a needless
    inexactness in a term that reaches the RJ acceptance ratio.

    Args:
        f0_ref_hz (float): Reference frequency in Hz (a per-group constant).
        mc_hi (float): Upper chirp-mass bound (Msun).
        ratio_max (float): Ratio half-width ``M`` (launchers:
            ``FSTAT_FDOT_RATIO_MAX=5.0``).

    Returns:
        tuple[float, float]: ``((1 - M) g, (1 + M) g)`` with
        ``g = fdot_gr(f0_ref_hz, mc_hi)``, in Hz/s.
    """
    g = fdot_gr(float(f0_ref_hz), float(mc_hi))
    return (1.0 - float(ratio_max)) * g, (1.0 + float(ratio_max)) * g


def n_fdot_nodes(fdot_lo, fdot_hi, tobs, *, eta=1.0, aligned=True,
                 clip=(3, 96)):
    """Node count for the fdot axis: one per coherence width across it.

    Same one-radian criterion as the Mc rule it replaces, so the two are
    directly comparable -- but evaluated on the ALIGNED width
    (``sqrt(720)/(2 pi T^2)``, 13.4x coarser; see
    :func:`fdot_coherence_width`) and across the FULL prior range rather
    than the positive ``r = 0`` sliver. At the flagship that is ~50 nodes
    over 10x the coverage, against ~70-96 for the old Mc rule.

    The ``[3, 96]`` clamp is inherited verbatim. The floor matters at low
    frequency, where ``fdot`` is simply not measurable over 90 d and the
    span is under one width -- 3 nodes there is the correct answer, not a
    degradation.

    Called per peak box by
    :func:`lisatools.sampling.fstat_gridfit.run_stacked_stage_b` with
    ``aligned=True`` and ``eta = FSTAT_MC_ETA`` (default 1.0).

    Args:
        fdot_lo (float): Lower axis bound (Hz/s).
        fdot_hi (float): Upper axis bound (Hz/s).
        tobs (float): Observation span in seconds.
        eta (float, optional): Spacing in coherence widths. Default ``1.0``.
        aligned (bool, optional): Use the sheared width (see
            :func:`fdot_coherence_width`). Default ``True``.
        clip (tuple[int, int], optional): ``(min, max)`` node count. Default
            ``(3, 96)``.

    Returns:
        int: ``clip(round(|hi - lo| / width) + 1, *clip)``.
    """
    span = abs(float(fdot_hi) - float(fdot_lo))
    width = fdot_coherence_width(tobs, aligned=aligned, eta=eta)
    n = int(np.clip(round(span / width) + 1, int(clip[0]), int(clip[1])))
    return n


def mc_floor_for_fdot(fdot, f0_hz, ratio_max, mc_lo):
    r"""Smallest ``Mc`` that can carry ``fdot`` with ``|r| <= ratio_max``.

    ``1 + r = fdot / fdot_gr(f0, Mc)`` and ``fdot_gr`` grows with ``Mc``,
    so a large ``|fdot|`` needs a large ``Mc``. The binding constraint is
    ``|1 + r| <= c``, with ``c = 1 + M`` when ``fdot >= 0`` (the ``1 - M``
    side is slack once ``M > 1``) and ``c = |M - 1|`` when ``fdot < 0``:

    .. math::

        M_{c,\min} = \max\!\left[\left(\frac{|\dot f|}{c\,K f_0^{11/3}}\right)^{3/5},
            \; m_{c,\rm lo}\right]

    with ``K = FDOT_K``. ``fdot = 0`` gives ``r = -1`` for any ``Mc``, so the floor is just
    ``mc_lo``.

    This is derived independently of the ``|r| <= M`` check, and
    ``Mc >= mc_floor`` must be EXACTLY that condition -- pinned by
    ``test_the_mc_floor_is_exactly_the_r_in_range_condition``. If the two
    ever disagree, one of the derivations is wrong.

    Callers: :meth:`lisatools.sampling.fstat_proposal.FdotAxisBirth.rvs` /
    ``logpdf`` (the feasible ``Mc`` interval of a birth) and
    :func:`lisatools.sampling.fstat_gridfit.enumerate_center_nodes`.

    Args:
        fdot (float or xp.ndarray): ``fdot`` in Hz/s, per row.
        f0_hz (float or xp.ndarray): ``f0`` in Hz, per row (broadcast against
            ``fdot``).
        ratio_max (float): Ratio half-width ``M``.
        mc_lo (float): Chirp-mass prior floor (Msun).

    Returns:
        xp.ndarray: float64 ``Mc`` floor (Msun) on the array module of ``fdot``,
        shaped like the broadcast inputs.
    """
    xp = get_array_module(fdot)
    fd = xp.asarray(fdot, dtype=xp.float64)
    # f0 is PER ROW in every real caller (the birth block derives it from
    # each row's own f_mid and fdot). An earlier version took float(f0_hz)
    # and worked only because the unit tests all passed a scalar.
    f0 = xp.asarray(f0_hz, dtype=xp.float64)
    M = float(ratio_max)
    c = xp.where(fd >= 0.0, 1.0 + M, abs(M - 1.0))
    denom = xp.maximum(c, 1e-300) * FDOT_K * xp.abs(f0) ** (11.0 / 3.0)
    need = xp.abs(fd) / xp.maximum(denom, 1e-300)
    return xp.maximum(need ** (3.0 / 5.0), float(mc_lo))


def r_from_fdot(fdot, f0_hz, mc):
    """``r = fdot / fdot_gr(f0, Mc) - 1``, the fiber conversion.

    Used by :meth:`lisatools.sampling.fstat_proposal.FdotAxisBirth.rvs` to turn
    a drawn ``(fdot, Mc)`` into the sampled ratio column.

    Args:
        fdot (float or xp.ndarray): ``fdot`` in Hz/s.
        f0_hz (float or xp.ndarray): ``f0`` in Hz.
        mc (float or xp.ndarray): Chirp mass in Msun.

    Returns:
        xp.ndarray: float64 ``fdot_astro_ratio`` on the array module of ``fdot``
        (``fdot_gr`` is floored at ``1e-300`` before the division).
    """
    xp = get_array_module(fdot)
    g = fdot_gr(xp.asarray(f0_hz, dtype=xp.float64),
                xp.asarray(mc, dtype=xp.float64))
    return xp.asarray(fdot, dtype=xp.float64) / xp.maximum(g, 1e-300) - 1.0


def gb_observable_step_scales(snr, tobs, *, extrinsic_scales, mc_step,
                              jump=1.0, snr_clip=(1.0, 1.0e4),
                              internal_basis=GB_INTERNAL_BASIS,
                              axis_mult=None):
    """Per-column step scales in the INTERNAL basis. ``(n,) -> (n, n_z)``.

    **This function deliberately cannot see ``coords``.** State-dependence
    belongs in the coordinate change, never in the step size: a scale that
    depends on the current point breaks the proposal's symmetry, which makes
    ``factors = Jacobian only`` wrong -- while leaving the acceptance rate
    looking perfectly healthy. Making the signature incapable of expressing it
    is cheaper than remembering not to.

    ``axis_mult`` is an optional ``(n, n_z)`` PER-ROW PER-COLUMN
    multiplier applied on top of ``jump``. It exists for the per-source
    adaptive step scales, and it does NOT weaken the rule above: a
    multiplier is a per-leaf CONSTANT for the whole propose, fixed
    before any draw and not a function of the current point, so the
    proposal stays symmetric and ``factors = Jacobian only`` stays
    correct. The signature still cannot express coord-dependence, which
    is the property that matters -- ``axis_mult`` is supplied by the
    caller, the same way ``extrinsic_scales`` already is.

    ``(lnA, f_mid, fdot)`` are analytic and go as ``1/rho``; the extrinsic
    block is supplied by the caller (from the information matrix, whose
    unreliability is confined to derivatives through ``f0``); ``Mc`` is
    prior-set because the likelihood is flat along the fiber.

    ``internal_basis`` names the columns, so a REDUCED layout (the VGB
    restriction: no ``f_mid``, no ``alpha``/``sin_delta``, no ``Mc``) gets
    each coordinate's own rule rather than the 9-column positions. The
    default is the GB layout and reproduces the pre-generalization array
    value for value; ``extrinsic_scales`` supplies the pass-through columns
    the layout actually carries, in order, and ``mc_step`` is consumed only
    by a layout that HAS an ``Mc`` column.

    Per column, before ``jump`` and ``axis_mult`` (``rho`` = clipped ``|snr|``,
    ``T`` = ``tobs``):

    * ``lnA``: ``STEP_C_LNA / rho``;
    * ``f_mid``: ``(STEP_C_FMID / rho) / T`` (Hz);
    * ``fdot``: ``(STEP_C_FDOT / rho) / T**2`` (Hz/s);
    * ``Mc``: ``mc_step``;
    * every other name: the next column of ``extrinsic_scales``.

    Called per in-model propose by the observable proposal of
    ``GBSpecialStretchMove`` / ``VGBSpecialStretchMove``
    (:mod:`lisatools.globalfit.moves.gbspecialstretch`), which passes
    ``mc_step = GB_INMODEL_OBSERVABLE_MC_STEP`` x the Mc prior width and
    ``jump = GB_INMODEL_OBSERVABLE_JUMP`` (or ``VGB_INMODEL_OBSERVABLE_JUMP``).

    Args:
        snr (xp.ndarray): ``(n,)`` per-row SNR; ``|snr|`` is clipped to
            ``snr_clip``.
        tobs (float): Observation span in seconds.
        extrinsic_scales (xp.ndarray): ``(n, n_pass)`` or ``(n_pass,)`` step
            scales of the pass-through columns, in the order ``internal_basis``
            carries them.
        mc_step (float): Absolute ``Mc`` step (Msun); read only when the layout
            has an ``Mc`` column.
        jump (float, optional): Global multiplier. Default ``1.0``.
        snr_clip (tuple[float, float], optional): ``(lo, hi)`` clip on ``|snr|``.
            Default ``(1.0, 1.0e4)``.
        internal_basis (tuple[str, ...], optional): Column names of the
            internal layout. Default :data:`GB_INTERNAL_BASIS`.
        axis_mult (array_like, optional): ``(n, n_z)`` per-row per-column
            multiplier, or ``None`` (the production value: no move defines the
            table yet). Default ``None``.

    Returns:
        xp.ndarray: ``(n, n_z)`` float64 step scales on the array module of
        ``snr``.

    Raises:
        ValueError: ``extrinsic_scales`` does not supply exactly one column per
            pass-through name, or ``axis_mult`` does not have shape ``(n, n_z)``.
    """
    xp = get_array_module(snr)
    rho = xp.clip(xp.abs(xp.asarray(snr, dtype=xp.float64)),
                  float(snr_clip[0]), float(snr_clip[1]))
    n = rho.shape[0]
    bin_hz = 1.0 / float(tobs)
    names = tuple(internal_basis)
    ex = xp.asarray(extrinsic_scales, dtype=xp.float64)
    ex = ex if ex.ndim == 2 else ex[None, :]
    out = xp.zeros((n, len(names)), dtype=xp.float64)
    k = 0
    for j, nm in enumerate(names):
        if nm == "lnA":
            out[:, j] = STEP_C_LNA / rho
        elif nm == "f_mid":
            out[:, j] = (STEP_C_FMID / rho) * bin_hz
        elif nm == "fdot":
            out[:, j] = (STEP_C_FDOT / rho) * bin_hz / float(tobs)
        elif nm == "Mc":
            out[:, j] = float(mc_step)
        else:
            out[:, j] = ex[:, k]
            k += 1
    if k != int(ex.shape[-1]):
        raise ValueError(
            f"extrinsic_scales has {int(ex.shape[-1])} columns but the "
            f"internal basis {names} carries {k} pass-through coordinates."
        )
    out = out * float(jump)
    if axis_mult is None:
        return out
    am = np.asarray(axis_mult, dtype=out.dtype)
    if am.shape != out.shape:
        raise ValueError(
            f"axis_mult has shape {am.shape} but the step scales are "
            f"{out.shape}. A silently broadcast multiplier would scale "
            f"the wrong axis for every row.")
    return out * am


class GBObservableFiberBasis:
    """``y <-> z`` for the 9-column GB sampling basis.

    Column indices are resolved from the transform container's
    ``input_basis`` and never hard-coded (the pattern from
    :class:`lisatools.sampling.ridge_fiber.McRatioDistFiber`); an ineligible
    basis raises at construction rather than mis-indexing at run time.

    The internal layout is DERIVED from the basis: ``lnA``, then ``f_mid`` (only
    when ``f0`` is sampled), ``fdot``, the pass-through extrinsics the basis
    carries (canonical order ``phi0, cos_iota, psi, alpha, sin_delta``), then
    ``Mc`` (only when sampled). For the full GB basis that is
    :data:`GB_INTERNAL_BASIS` with ``FIBER_INDEX = 8``. Units of ``z``: ``lnA``
    natural log of the strain amplitude, ``f_mid`` Hz, ``fdot`` Hz/s, ``Mc`` Msun
    (or ``ln Msun`` for ``fiber_coord="lnMc"``); the sampling side keeps ``f0``
    in mHz and ``dist`` in kpc.

    Built (and cached) by ``GBSpecialBase._observable_map`` in
    :mod:`lisatools.globalfit.moves.gbspecialstretch` with ``Tobs = 1/df``,
    ``shear = GB_INMODEL_OBSERVABLE_SHEAR`` (launchers 0.5) and
    ``fiber_coord="Mc"``, and by :func:`lisatools.globalfit.warmstart.basis.build_map`.

    Args:
        transform_container: supplies ``input_basis`` (and, for the VGB
            subclass, the per-leaf ``original_fill_dict``).
        Tobs (float): observation span in seconds. **Required, no default** -- a
            wrong value is efficiency-only by the determinant result, but a
            silently defaulted one is untraceable. Callers must pass
            ``1.0 / self.df``, NOT ``basis_settings.Tobs``: the latter does not
            exist on ``FDSettings`` and an unconditional read has already broken
            every FD-domain GB flow once.
        shear (float, optional): coefficient as a fraction of ``Tobs``
            (0.5 -> ``f_mid``). Default ``0.5``.
        fiber_coord (str, optional): ``"Mc"`` (linear) or ``"lnMc"``. Explicit
            because a silent mix-up tilts the ``Mc`` marginal by ``Mc**-1``,
            which reads as "the sampler prefers low Mc" and gets rationalised as
            physics. Default ``"Mc"``.

    Raises:
        ValueError: A required column (``dist, f0, Mc, fdot_astro_ratio`` for
            GB) is missing, ``fiber_coord`` is not ``"Mc"``/``"lnMc"``, ``Tobs``
            is not finite and positive, or (VGB) a pinned constant has no
            per-leaf fill value.

    Attributes:
        input_basis (list[str]): Sampling-basis column names.
        INTERNAL_BASIS (tuple[str, ...]): Derived internal column names.
        dist_index (int): Sampling column of ``dist``.
        f0_index (int or None): Sampling column of ``f0``; ``None`` = pinned per
            leaf (VGB).
        mc_index (int or None): Sampling column of ``Mc``; ``None`` = pinned per
            leaf.
        ratio_index (int): Sampling column of ``fdot_astro_ratio``.
        Tobs (float): Observation span (s).
        shear (float): Shear fraction ``s``; ``c_t = s * Tobs``.
        fiber_coord (str): ``"Mc"`` or ``"lnMc"``.

    Example:
        >>> from types import SimpleNamespace
        >>> import numpy as np
        >>> from lisatools.sampling.gb_observable_basis import GBObservableFiberBasis
        >>> tc = SimpleNamespace(input_basis=["dist", "f0", "Mc", "phi0", "cos_iota",
        ...                                   "psi", "alpha", "sin_delta",
        ...                                   "fdot_astro_ratio"])
        >>> m = GBObservableFiberBasis(tc, Tobs=3.15e7)
        >>> y = np.array([[8.0, 3.0, 0.3, 1.0, 0.2, 0.5, 1.0, 0.1, 0.4]])
        >>> z = m.to_internal(y)
        >>> bool(np.allclose(m.from_internal(z), y))
        True
    """

    _REQUIRED = ("dist", "f0", "Mc", "fdot_astro_ratio")
    #: Output-basis names this map takes from the container's PER-LEAF fills
    #: rather than from a sampled column (empty for GB: everything is
    #: sampled). Only entries absent from ``input_basis`` are looked up.
    _PINNED = ()
    INTERNAL_BASIS = GB_INTERNAL_BASIS
    FIBER_INDEX = 8

    def __init__(self, transform_container, *, Tobs, shear=0.5,
                 fiber_coord="Mc"):
        basis = list(getattr(transform_container, "input_basis", []) or [])
        missing = [n for n in self._REQUIRED if n not in basis]
        if missing:
            raise ValueError(
                f"{type(self).__name__} requires "
                f"{', '.join(self._REQUIRED)} in the sampling basis; "
                f"input_basis {basis} is missing {missing}. 8-column "
                "(A / fdot) bases are not eligible."
            )
        if fiber_coord not in ("Mc", "lnMc"):
            raise ValueError(f"fiber_coord must be 'Mc' or 'lnMc', got {fiber_coord!r}")
        if not np.isfinite(Tobs) or float(Tobs) <= 0:
            raise ValueError(f"Tobs must be finite and positive, got {Tobs!r}")
        self.input_basis = basis
        self.dist_index = basis.index("dist")
        # ``None`` = pinned per leaf rather than sampled (the VGB
        # restriction); every read below goes through _f0_hz / _mc so the
        # two cases share one expression.
        self.f0_index = basis.index("f0") if "f0" in basis else None
        self.mc_index = basis.index("Mc") if "Mc" in basis else None
        self.ratio_index = basis.index("fdot_astro_ratio")
        self._extrinsic_names = tuple(n for n in _EXTRINSIC_NAMES
                                      if n in basis)
        self._extrinsic = [basis.index(n) for n in self._extrinsic_names]
        self.Tobs = float(Tobs)
        self.shear = float(shear)
        self.fiber_coord = fiber_coord
        self._pinned = self._read_pinned(transform_container)
        # DERIVED internal layout. For the full GB basis this reproduces
        # ``GB_INTERNAL_BASIS`` and ``FIBER_INDEX = 8`` exactly (pinned by
        # test_gb_observable_basis); a reduced basis simply carries fewer
        # names, in the same canonical order, whatever order it SAMPLES in.
        names = ["lnA"]
        if self.f0_index is not None:
            names.append("f_mid")           # no sampled f0 => no shear
        names.append("fdot")
        names.extend(self._extrinsic_names)
        if self.mc_index is not None:
            names.append("Mc")              # the fiber coordinate
        self.INTERNAL_BASIS = tuple(names)
        self._z = {n: i for i, n in enumerate(self.INTERNAL_BASIS)}
        #: Index of ``Mc`` in ``z`` (int); ``None`` when the layout has no fiber
        #: (``Mc`` pinned): the consumer then uses the generic no-fiber eigen set.
        self.FIBER_INDEX = self._z.get("Mc")

    def _read_pinned(self, transform_container):
        """``{name: (nleaves,) values}`` for the per-leaf pinned constants.

        Empty for GB (``_PINNED = ()``), so the container is not touched at
        all -- a stand-in exposing only ``input_basis`` stays valid. The VGB
        restriction reads the SAMPLING-unit fill values (f0 in mHz) out of
        the container's per-leaf ``fill_dict`` list, which is the single
        source of those constants; rows are selected by ``leaf_inds`` at
        call time exactly as ``TransformContainer.fill_values`` selects them.

        Args:
            transform_container: The container; its ``original_fill_dict``
                (a dict or a per-leaf list of dicts) is read only when a
                ``_PINNED`` name is not a sampled column.

        Returns:
            dict[str, np.ndarray]: ``{name: (nleaves,) float}``; empty when
            nothing is pinned.

        Raises:
            ValueError: A pinned name is needed but the container has no
                fill dict, or the fill dict lacks the key.
        """
        need = [k for k in self._PINNED if k not in self.input_basis]
        if not need:
            return {}
        fills = getattr(transform_container, "original_fill_dict", None)
        if isinstance(fills, dict):
            fills = [fills]
        if not fills:
            raise ValueError(
                f"{type(self).__name__} needs per-leaf fill values for "
                f"{need} (they are not sampled columns of {self.input_basis}) "
                "but the transform container carries no fill_dict."
            )
        absent = [k for k in need if k not in fills[0]]
        if absent:
            raise ValueError(
                f"{type(self).__name__}: {absent} are neither sampled "
                f"columns of {self.input_basis} nor per-leaf fill keys "
                f"{list(fills[0])}."
            )
        return {k: np.asarray([float(d[k]) for d in fills], dtype=float)
                for k in need}

    @property
    def n_leaves(self):
        """Number of pinned leaves, or ``None`` when nothing is pinned."""
        for v in self._pinned.values():
            return int(v.shape[0])
        return None

    @property
    def _c_t(self):
        """Shear coefficient ``c_t = shear * Tobs`` in seconds."""
        return self.shear * self.Tobs

    def _pinned_row(self, key, coords, leaf_inds):
        """Per-ROW value of a pinned constant, selected by leaf index.

        Args:
            key (str): Pinned name (``"f0"`` in mHz or ``"Mc"`` in Msun).
            coords (xp.ndarray): ``(n, ...)`` rows (only ``shape[0]`` and the
                array module are used).
            leaf_inds (array_like or None): ``(n,)`` leaf index per row; ``None``
                is allowed only when exactly one leaf is pinned.

        Returns:
            xp.ndarray: ``(n,)`` pinned values.

        Raises:
            ValueError: ``leaf_inds`` is ``None`` and more than one leaf is
                pinned.
        """
        xp = get_array_module(coords)
        vals = self._pinned[key]
        if leaf_inds is None:
            if vals.shape[0] == 1:
                return xp.full(coords.shape[0], float(vals[0]))
            raise ValueError(
                f"{type(self).__name__} pins {key!r} per leaf "
                f"({vals.shape[0]} leaves); pass leaf_inds (shape "
                "coords.shape[:-1]) -- a row mapped with another leaf's "
                "constants is a different physical source."
            )
        return xp.asarray(vals)[xp.asarray(leaf_inds).astype(int)]

    def _f0_hz(self, coords, leaf_inds=None):
        """``f0`` in Hz: the sampled mHz column, or the pinned constant."""
        f0_ms = (coords[:, self.f0_index] if self.f0_index is not None
                 else self._pinned_row("f0", coords, leaf_inds))
        return f0_ms * 1e-3                                  # mHz -> Hz

    def _mc(self, coords, leaf_inds=None):
        """``Mc`` (Msol): the sampled column, or the pinned constant."""
        if self.mc_index is not None:
            return coords[:, self.mc_index]
        return self._pinned_row("Mc", coords, leaf_inds)

    # ---- the bijection -------------------------------------------------
    def to_internal(self, coords, leaf_inds=None):
        """``(n, ndim)`` sampling -> ``(n, n_z)`` internal.

        ``leaf_inds`` is required only when the map pins a per-leaf constant
        (the VGB restriction); the GB map ignores it.

        Args:
            coords (xp.ndarray): ``(n, ndim)`` sampling rows (``f0`` in mHz,
                ``dist`` in kpc).
            leaf_inds (array_like, optional): ``(n,)`` leaf index per row (VGB
                only). Default ``None``.

        Returns:
            xp.ndarray: ``(n, n_z)`` float64 rows in :attr:`INTERNAL_BASIS`
            order (``f_mid`` in Hz, ``fdot`` in Hz/s), on the array module of
            ``coords``.
        """
        xp = get_array_module(coords)
        f0 = self._f0_hz(coords, leaf_inds)
        mc = self._mc(coords, leaf_inds)
        fd = fdot_gr(f0, mc) * (1.0 + coords[:, self.ratio_index])
        amp = self._amp(f0, mc, coords[:, self.dist_index])
        z = xp.zeros((coords.shape[0], len(self.INTERNAL_BASIS)),
                     dtype=xp.float64)
        z[:, self._z["lnA"]] = xp.log(amp)
        if "f_mid" in self._z:
            z[:, self._z["f_mid"]] = f0 + self._c_t * fd
        z[:, self._z["fdot"]] = fd
        for name, c in zip(self._extrinsic_names, self._extrinsic):
            z[:, self._z[name]] = coords[:, c]
        if "Mc" in self._z:
            z[:, self._z["Mc"]] = (xp.log(mc) if self.fiber_coord == "lnMc"
                                   else mc)
        return z

    def from_internal(self, z, template=None, leaf_inds=None):
        """``(n, n_z)`` internal -> ``(n, ndim)`` sampling.

        Inverts :meth:`to_internal`: ``f0 = f_mid - c_t fdot``,
        ``r = fdot / fdot_gr(f0, Mc) - 1`` and ``dist = A(f0, Mc, 1 kpc) / A``.

        Args:
            z (xp.ndarray): ``(n, n_z)`` internal rows.
            template (xp.ndarray, optional): ``(n, ndim)`` rows copied first;
                every column the map owns is then overwritten, so only columns
                outside the map would carry through. ``None`` starts from
                zeros. Default ``None``.
            leaf_inds (array_like, optional): ``(n,)`` leaf index per row (VGB
                only). Default ``None``.

        Returns:
            xp.ndarray: ``(n, ndim)`` float64 sampling rows (``f0`` in mHz).
        """
        xp = get_array_module(z)
        ndim = len(self.input_basis)
        out = (xp.zeros((z.shape[0], ndim), dtype=xp.float64)
               if template is None else xp.array(template, dtype=xp.float64))
        if "Mc" in self._z:
            _zmc = z[:, self._z["Mc"]]
            mc = xp.exp(_zmc) if self.fiber_coord == "lnMc" else _zmc
        else:
            mc = self._pinned_row("Mc", z, leaf_inds)
        fd = z[:, self._z["fdot"]]
        if "f_mid" in self._z:
            f0 = z[:, self._z["f_mid"]] - self._c_t * fd
        else:
            f0 = self._pinned_row("f0", z, leaf_inds) * 1e-3
        if self.f0_index is not None:
            out[:, self.f0_index] = f0 * 1e3                  # Hz -> mHz
        if self.mc_index is not None:
            out[:, self.mc_index] = mc
        out[:, self.ratio_index] = fd / fdot_gr(f0, mc) - 1.0
        # A is strictly propto 1/dist, so dist = A(f0, Mc, 1 kpc) / A.
        out[:, self.dist_index] = (self._amp(f0, mc, 1.0)
                                   / xp.exp(z[:, self._z["lnA"]]))
        for name, c in zip(self._extrinsic_names, self._extrinsic):
            out[:, c] = z[:, self._z[name]]
        return out

    # ---- the measure ---------------------------------------------------
    def log_jacobian(self, coords, leaf_inds=None):
        """``ln|dy/dz|`` at a SAMPLING point, up to an additive constant.

        ``= ln(dist) - ln(fdot_gr(f0, Mc))``, plus ``+ln(Mc)`` when the fiber
        coordinate is ``lnMc``. Only differences are ever used, so the
        constant is irrelevant -- but it must be the SAME constant at both
        ends, which is why both ends call this one function.

        The reduced (pinned-f0/Mc) map has the SAME expression: the map is
        then block-diagonal instead of block-triangular and the determinant
        is unchanged. The ``fdot_gr`` term is a per-leaf constant there, so
        it cancels in :meth:`factors` -- correctly, and without a second
        formula.

        Args:
            coords (xp.ndarray): ``(n, ndim)`` sampling rows.
            leaf_inds (array_like, optional): ``(n,)`` leaf index per row (VGB
                only). Default ``None``.

        Returns:
            xp.ndarray: ``(n,)`` log Jacobian; non-positive ``dist``/``Mc`` give
            ``nan``/``-inf`` (numpy warnings suppressed).
        """
        xp = get_array_module(coords)
        f0 = self._f0_hz(coords, leaf_inds)
        mc = self._mc(coords, leaf_inds)
        with np.errstate(invalid="ignore", divide="ignore"):
            lj = xp.log(coords[:, self.dist_index]) - xp.log(fdot_gr(f0, mc))
            if self.fiber_coord == "lnMc" and self.mc_index is not None:
                lj = lj + xp.log(mc)
        return lj

    def factors(self, old_coords, new_coords, leaf_inds=None):
        """MH log-factor for the move ``old -> new``: **NEW minus OLD**.

        Sign confirmed by prior-invariance simulation -- correct preserves all
        nine marginals, flipped collapses ``dist`` and ``Mc`` (KS p underflows
        to 0 and 2.5e-253 respectively).

        Non-finite rows are clamped to a large FINITE negative rather than
        ``nan``: this is evaluated before the prior gate, so ``ln`` of a
        non-positive ``dist``/``Mc`` is reachable, and ``nan`` comparison
        semantics need not agree between the NumPy accept path and the CUDA
        kernel. A finite ``-1e300`` is rejected identically by both.

        Args:
            old_coords (xp.ndarray): ``(n, ndim)`` current sampling rows.
            new_coords (xp.ndarray): ``(n, ndim)`` proposed sampling rows.
            leaf_inds (array_like, optional): ``(n,)`` leaf index per row (VGB
                only). Default ``None``.

        Returns:
            xp.ndarray: ``(n,)`` ``log_jacobian(new) - log_jacobian(old)``, with
            non-finite entries replaced by ``-1e300``.
        """
        xp = get_array_module(new_coords)
        f = (self.log_jacobian(new_coords, leaf_inds)
             - self.log_jacobian(old_coords, leaf_inds))
        return xp.where(xp.isfinite(f), f, -1e300)

    # ---- internals -----------------------------------------------------
    @staticmethod
    def _amp(f0_hz, mc, dist_kpc):
        """``A`` through the installed helper, so the constant set matches.

        Calls :func:`gbgpu.utils.utility.get_amplitude` with equal component
        masses ``m = Mc * 2**(1/5)`` (whose chirp mass is ``Mc``).

        Args:
            f0_hz (xp.ndarray): Frequency (Hz).
            mc (xp.ndarray): Chirp mass (Msun).
            dist_kpc (float or xp.ndarray): Distance (kpc).

        Returns:
            xp.ndarray: Strain amplitude ``A``.
        """
        from gbgpu.utils.utility import get_amplitude
        m = mc * _TWO_POW_1_5
        return get_amplitude(m, m, f0_hz, dist_kpc)


class VGBObservableBasis(GBObservableFiberBasis):
    """The VGB restriction of :class:`GBObservableFiberBasis`.

    Same map, same measure, same code -- only the column set differs. The
    verification-binary branch is fixed-dimensional and its sources are
    KNOWN, so ``f0``, ``alpha``, ``sin_delta`` (and ``Mc``, unless
    ``VGB_CHIRP_MASS_BASIS=1``) are pinned PER LEAF in the transform
    container's per-leaf ``fill_dict``. That leaves

        y = [dist, phi0, cos_iota, psi, r]   ->
        z = [lnA,  fdot, phi0, cos_iota, psi]

    (see the module docstring for why there is no ``f_mid`` and no fiber).
    Every coordinate read routes through the base class's ``_f0_hz`` /
    ``_mc``, which take the pinned constant when the column is absent --
    so ``leaf_inds`` must be supplied on every call, the same contract
    ``TransformContainer.fill_values`` imposes on the transform path.

    WHY THIS MATTERS BEYOND THE PROPOSAL SHAPE. In the SAMPLING basis the
    only physical quantity ``r`` drives is ``fdot = fdot_gr(f0, Mc)(1 + r)``
    -- and with ``Mc`` pinned, ``Mc`` occupies the physical ``fdot`` slot in
    the container's ``key_map``, so ``fdot`` sits OUTSIDE ``test_inds`` and
    ``r``'s only scored target is the dead ``fddot`` slot (identically
    zero). The information matrix therefore had EXACTLY zero curvature in
    the ``r`` direction and its eigen step was the prior-box width: a blind
    jump. Proposing in ``z``, where ``fdot`` is a raw coordinate, is half
    the fix; the other half is asking the engine for the physical ``fdot``
    slot in the first place (see ``_infomat_phys_inds`` in
    ``globalfit/moves/gbspecialstretch.py``).

    Constructor arguments are those of :class:`GBObservableFiberBasis`
    (``shear`` is accepted and never consulted). The required sampled columns
    are ``dist`` and ``fdot_astro_ratio``; ``f0`` (mHz) and ``Mc`` (Msun) come
    from the per-leaf fills unless sampled. Used by the ``vgb_pe`` in-model
    proposal (``VGB_INMODEL_PROPOSAL=observable``); under
    ``VGB_CHIRP_MASS_BASIS=1`` (the 9mo/12mo launcher value) ``Mc`` is sampled,
    so the derived layout is ``[lnA, fdot, phi0, cos_iota, psi, Mc]`` with
    ``FIBER_INDEX = 5``.

    Example:
        .. code-block:: python

            from lisatools.sampling.gb_observable_basis import VGBObservableBasis

            m = VGBObservableBasis(vgb_transform_container, Tobs=1.0 / df)
            z = m.to_internal(coords, leaf_inds=leaf_inds)
            back = m.from_internal(z, leaf_inds=leaf_inds)
    """

    #: ``f0`` is pinned, so it is NOT required in the sampling basis (the
    #: base's requirement list would reject the reduced basis outright).
    _REQUIRED = ("dist", "fdot_astro_ratio")
    _PINNED = ("f0", "Mc")
    #: Documented default layout (``VGB_CHIRP_MASS_BASIS=0``). The instance
    #: attribute is DERIVED in ``__init__`` from the actual basis, so the
    #: chirp-mass variant grows its ``Mc`` column (and its fiber) back
    #: without a second class.
    INTERNAL_BASIS = ("lnA", "fdot", "phi0", "cos_iota", "psi")
    FIBER_INDEX = None

    # NOTE: ``shear`` is accepted (inherited signature, one construction
    # path) and never consulted -- with ``f0`` pinned there is no ``f_mid``
    # column for it to act on, so ``_c_t`` is unreachable in this layout.
