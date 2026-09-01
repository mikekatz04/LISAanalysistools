"""Observable-basis bijection for GB proposals: ``y <-> (u, v)``.

The 9-column GB sampling basis
``y = [dist, f0(mHz), Mc, phi0, cos_iota, psi, alpha, sin_delta, r]``
is the right thing to SAMPLE in -- ``f0`` anchored at ``t_ref`` is stable as
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
``z = [lnA, f_mid, fdot, phi0, cos_iota, psi, alpha, sin_delta, Mc]``
where ``fdot`` is a coordinate (nothing can cancel in it) and ``f_mid``
decorrelates the shear, then convert back. ``Mc`` is the fiber coordinate: the
exactly-flat direction, along which the likelihood cannot change.

THE MEASURE
-----------
``|dy/dz| = dist / fdot_gr(f0, Mc)`` -- with ``y = (dist, f0, Mc, r)`` and
``z = (lnA, f0, fdot, Mc)`` the Jacobian is block-triangular and expanding on
the ``lnA`` column gives ``(-dist) * (-1/fdot_gr)``. Verified three ways:
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
"""
from __future__ import annotations

import numpy as np

from ..utils.utility import get_array_module

__all__ = [
    "FDOT_K",
    "GBObservableFiberBasis",
    "f0_from_f_mid",
    "f_mid_from_f0",
    "fdot_coherence_width",
    "fdot_gr",
    "fdot_shear_hz",
    "gb_observable_step_scales",
]

# Constants taken from the SAME source as gbgpu.utils.utility.get_fdot /
# get_amplitude so the grid rows and this module cannot drift. Pinned by
# test_fdot_gr_matches_gbgpu.
try:                                        # pragma: no cover - import shape
    from gbgpu.utils.utility import MSUN_SI as _MSUN_SI, G_SI as _G_SI, C_SI as _C_SI
except ImportError:                         # pragma: no cover
    from lisaconstants import GM_SUN, c as _C_SI
    _G_SI, _MSUN_SI = 6.674e-11, GM_SUN / 6.674e-11

#: ``fdot_gr = FDOT_K * Mc[Msol]**(5/3) * f0[Hz]**(11/3)``. Absorbs the two
#: duplicate definitions in ``sampling/fstat_proposal.py``.
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


def fdot_gr(f0_hz, mc):
    """GR-driven ``fdot`` from ``(f0[Hz], Mc[Msol])``. Elementwise, xp-agnostic."""
    return FDOT_K * mc ** (5.0 / 3.0) * f0_hz ** (11.0 / 3.0)


def _shear_coeff(tobs, c_t=None):
    return 0.5 * float(tobs) if c_t is None else float(c_t)


def f_mid_from_f0(f0_hz, fdot, tobs, *, c_t=None):
    """``f_mid = f0 + c_t * fdot`` with ``c_t = T/2`` by default.

    The mean frequency of a linear chirp over ``[0, T]``; measured
    ``<t> = 0.498 T``, i.e. uniform weighting.
    """
    return f0_hz + _shear_coeff(tobs, c_t) * fdot


def f0_from_f_mid(f_mid_hz, fdot, tobs, *, c_t=None):
    """Exact inverse of :func:`f_mid_from_f0` at the same ``fdot``."""
    return f_mid_hz - _shear_coeff(tobs, c_t) * fdot


def fdot_shear_hz(mc, f0_ref_hz, tobs, *, c_t=None):
    """The ``f0`` offset a GR chirp of this ``(f0_ref, Mc)`` induces.

    ``c_t * fdot_gr(f0_ref, mc)``. ``f0_ref`` is a per-box CONSTANT reference
    for the F-stat grid, never the node's own ``f0`` -- using the node's own
    value makes the grid shear's determinant ``1 - 2.6e-4`` instead of exactly
    1, a needless inexactness.
    """
    return _shear_coeff(tobs, c_t) * fdot_gr(f0_ref_hz, mc)


def fdot_coherence_width(tobs, *, aligned=False, eta=1.0):
    """``fdot`` node spacing at the one-radian criterion.

    With ``a = df0*T``, ``b = dfdot*T**2`` and uniform weighting the residual
    phase variance is ``Q = a**2/12 + a*b/12 + b**2/45``. Substituting
    ``a' = a + b/2`` (the shear) gives ``Q = a'**2/12 + b**2/720`` -- the cross
    term vanishes exactly and ``det`` is preserved (``1/8640`` both ways), an
    independent confirmation of the unit-determinant result.

    So the *irreducible* fdot residual is ``b**2/720`` rather than the raw
    ``b**2/45 + cross``, and one radian at the end of the observation allows a
    node spacing ``sqrt(720)/(2 pi)`` instead of ``1/pi`` -- **13.4x coarser at
    the same criterion**.

    ``aligned=False`` reproduces the live rule byte-identically.
    """
    t2 = float(tobs) ** 2
    if aligned:
        return float(eta) * np.sqrt(720.0) / (2.0 * np.pi * t2)
    return float(eta) / (np.pi * t2)


def gb_observable_step_scales(snr, tobs, *, extrinsic_scales, mc_step,
                              jump=1.0, snr_clip=(1.0, 1.0e4)):
    """Per-column step scales in the INTERNAL basis. ``(n,) -> (n, 9)``.

    **This function deliberately cannot see ``coords``.** State-dependence
    belongs in the coordinate change, never in the step size: a scale that
    depends on the current point breaks the proposal's symmetry, which makes
    ``factors = Jacobian only`` wrong -- while leaving the acceptance rate
    looking perfectly healthy. Making the signature incapable of expressing it
    is cheaper than remembering not to.

    ``(lnA, f_mid, fdot)`` are analytic and go as ``1/rho``; the extrinsic
    block is supplied by the caller (from the information matrix, whose
    unreliability is confined to derivatives through ``f0``); ``Mc`` is
    prior-set because the likelihood is flat along the fiber.
    """
    xp = get_array_module(snr)
    rho = xp.clip(xp.abs(xp.asarray(snr, dtype=xp.float64)),
                  float(snr_clip[0]), float(snr_clip[1]))
    n = rho.shape[0]
    bin_hz = 1.0 / float(tobs)
    out = xp.zeros((n, 9), dtype=xp.float64)
    out[:, 0] = STEP_C_LNA / rho
    out[:, 1] = (STEP_C_FMID / rho) * bin_hz
    out[:, 2] = (STEP_C_FDOT / rho) * bin_hz / float(tobs)
    ex = xp.asarray(extrinsic_scales, dtype=xp.float64)
    out[:, 3:8] = ex if ex.ndim == 2 else ex[None, :]
    out[:, 8] = float(mc_step)
    return out * float(jump)


class GBObservableFiberBasis:
    """``y <-> z`` for the 9-column GB sampling basis.

    Column indices are resolved from the transform container's
    ``input_basis`` and never hard-coded (the pattern from
    :class:`lisatools.sampling.ridge_fiber.McRatioDistFiber`); an ineligible
    basis raises at construction rather than mis-indexing at run time.

    Args:
        transform_container: supplies ``input_basis``.
        Tobs: observation span in seconds. **Required, no default** -- a wrong
            value is efficiency-only by the determinant result, but a silently
            defaulted one is untraceable. Callers must pass ``1.0 / self.df``,
            NOT ``basis_settings.Tobs``: the latter does not exist on
            ``FDSettings`` and an unconditional read has already broken every
            FD-domain GB flow once.
        shear: coefficient as a fraction of ``Tobs`` (0.5 -> ``f_mid``).
        fiber_coord: ``"Mc"`` (linear) or ``"lnMc"``. Explicit because a
            silent mix-up tilts the ``Mc`` marginal by ``Mc**-1``, which reads
            as "the sampler prefers low Mc" and gets rationalised as physics.
    """

    _REQUIRED = ("dist", "f0", "Mc", "fdot_astro_ratio")
    INTERNAL_BASIS = ("lnA", "f_mid", "fdot", "phi0", "cos_iota", "psi",
                      "alpha", "sin_delta", "Mc")
    FIBER_INDEX = 8

    def __init__(self, transform_container, *, Tobs, shear=0.5,
                 fiber_coord="Mc"):
        basis = list(getattr(transform_container, "input_basis", []) or [])
        missing = [n for n in self._REQUIRED if n not in basis]
        if missing:
            raise ValueError(
                "GBObservableFiberBasis requires the (dist, f0, Mc, "
                f"fdot_astro_ratio) GB sampling basis; input_basis {basis} is "
                f"missing {missing}. 8-column (A / fdot) and VGB bases are not "
                "eligible."
            )
        if fiber_coord not in ("Mc", "lnMc"):
            raise ValueError(f"fiber_coord must be 'Mc' or 'lnMc', got {fiber_coord!r}")
        if not np.isfinite(Tobs) or float(Tobs) <= 0:
            raise ValueError(f"Tobs must be finite and positive, got {Tobs!r}")
        self.input_basis = basis
        self.dist_index = basis.index("dist")
        self.f0_index = basis.index("f0")
        self.mc_index = basis.index("Mc")
        self.ratio_index = basis.index("fdot_astro_ratio")
        self._extrinsic = [basis.index(n) for n in
                           ("phi0", "cos_iota", "psi", "alpha", "sin_delta")
                           if n in basis]
        self.Tobs = float(Tobs)
        self.shear = float(shear)
        self.fiber_coord = fiber_coord

    @property
    def _c_t(self):
        return self.shear * self.Tobs

    # ---- the bijection -------------------------------------------------
    def to_internal(self, coords):
        """``(n, ndim)`` sampling -> ``(n, 9)`` internal."""
        xp = get_array_module(coords)
        f0 = coords[:, self.f0_index] * 1e-3                 # mHz -> Hz
        mc = coords[:, self.mc_index]
        fd = fdot_gr(f0, mc) * (1.0 + coords[:, self.ratio_index])
        amp = self._amp(f0, mc, coords[:, self.dist_index])
        z = xp.zeros((coords.shape[0], 9), dtype=xp.float64)
        z[:, 0] = xp.log(amp)
        z[:, 1] = f0 + self._c_t * fd
        z[:, 2] = fd
        for k, c in enumerate(self._extrinsic):
            z[:, 3 + k] = coords[:, c]
        z[:, 8] = xp.log(mc) if self.fiber_coord == "lnMc" else mc
        return z

    def from_internal(self, z, template=None):
        """``(n, 9)`` internal -> ``(n, ndim)`` sampling."""
        xp = get_array_module(z)
        ndim = len(self.input_basis)
        out = (xp.zeros((z.shape[0], ndim), dtype=xp.float64)
               if template is None else xp.array(template, dtype=xp.float64))
        mc = xp.exp(z[:, 8]) if self.fiber_coord == "lnMc" else z[:, 8]
        fd = z[:, 2]
        f0 = z[:, 1] - self._c_t * fd
        out[:, self.f0_index] = f0 * 1e3                      # Hz -> mHz
        out[:, self.mc_index] = mc
        out[:, self.ratio_index] = fd / fdot_gr(f0, mc) - 1.0
        # A is strictly propto 1/dist, so dist = A(f0, Mc, 1 kpc) / A.
        out[:, self.dist_index] = self._amp(f0, mc, 1.0) / xp.exp(z[:, 0])
        for k, c in enumerate(self._extrinsic):
            out[:, c] = z[:, 3 + k]
        return out

    # ---- the measure ---------------------------------------------------
    def log_jacobian(self, coords):
        """``ln|dy/dz|`` at a SAMPLING point, up to an additive constant.

        ``= ln(dist) - ln(fdot_gr(f0, Mc))``, plus ``+ln(Mc)`` when the fiber
        coordinate is ``lnMc``. Only differences are ever used, so the
        constant is irrelevant -- but it must be the SAME constant at both
        ends, which is why both ends call this one function.
        """
        xp = get_array_module(coords)
        f0 = coords[:, self.f0_index] * 1e-3
        mc = coords[:, self.mc_index]
        with np.errstate(invalid="ignore", divide="ignore"):
            lj = xp.log(coords[:, self.dist_index]) - xp.log(fdot_gr(f0, mc))
            if self.fiber_coord == "lnMc":
                lj = lj + xp.log(mc)
        return lj

    def factors(self, old_coords, new_coords):
        """MH log-factor for the move ``old -> new``: **NEW minus OLD**.

        Sign confirmed by prior-invariance simulation -- correct preserves all
        nine marginals, flipped collapses ``dist`` and ``Mc`` (KS p underflows
        to 0 and 2.5e-253 respectively).

        Non-finite rows are clamped to a large FINITE negative rather than
        ``nan``: this is evaluated before the prior gate, so ``ln`` of a
        non-positive ``dist``/``Mc`` is reachable, and ``nan`` comparison
        semantics need not agree between the NumPy accept path and the CUDA
        kernel. A finite ``-1e300`` is rejected identically by both.
        """
        xp = get_array_module(new_coords)
        f = self.log_jacobian(new_coords) - self.log_jacobian(old_coords)
        return xp.where(xp.isfinite(f), f, -1e300)

    # ---- internals -----------------------------------------------------
    @staticmethod
    def _amp(f0_hz, mc, dist_kpc):
        """``A`` through the installed helper, so the constant set matches."""
        from gbgpu.utils.utility import get_amplitude
        m = mc * _TWO_POW_1_5
        return get_amplitude(m, m, f0_hz, dist_kpc)
