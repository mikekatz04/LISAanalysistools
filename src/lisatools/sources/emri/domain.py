"""FEW domain-of-validity helpers for EMRI sampling.

The stock EMRI priors are independent boxes on ``(a, p0, e0, ...)`` whose
corners extend beyond the FEW Kerr eccentric-equatorial interpolation grid, so
prior draws / proposals can hit parameters the waveform cannot generate. FEW
rejects those with bare ``ValueError`` ("... outside of our domain of
validity.") at init or ``AssertionError`` ("Interpolation: p ... out of
bounds.") from the trajectory. Two layers keep the sampler safe:

- :func:`few_domain_guard` converts those errors into the typed
  :class:`~lisatools.utils.exceptions.WaveformDomainError` that the global-fit
  likelihood machinery catches (scoring the proposal at the ``-1e300`` floor).
- :func:`emri_kerr_domain_mask` is a cheap (no-waveform) vectorized mirror of
  ``KerrEccentricEquatorial.sanity_check_init`` plus the trajectory's
  separatrix buffer, used by the EMRI prior
  (:class:`~lisatools.sampling.prior.EMRIKerrDomainPrior`) to zero support
  (``-inf``) outside the domain before any waveform is attempted.

``few`` is imported lazily (inside the mask) so importing this module does not
require it, matching :mod:`.waveform`.
"""

from __future__ import annotations

import contextlib

import numpy as np

from ...utils.exceptions import WaveformDomainError

__all__ = [
    "FEW_DOMAIN_ERROR_PATTERNS",
    "few_domain_guard",
    "emri_kerr_domain_mask",
]

#: Substrings identifying FEW's out-of-domain ``ValueError``/``AssertionError``
#: messages (``few.utils.baseclasses.sanity_check_init`` and the
#: ``few.trajectory.ode`` ``isvalid_*`` checks).
FEW_DOMAIN_ERROR_PATTERNS = (
    "out of bounds",  # trajectory/interpolant isvalid_* checks (p, e, a, x)
    "outside of our domain of validity",  # sanity_check_init (a, p0, e0)
    "too close to the separatrix",  # sanity_check_init NaN forward map
    "not allowed",  # Schwarzschild e0 > 0.75
    "not physical",  # e0 < 0
    "Initial p0 is too large",  # Schwarzschild outer p0 bound
)


@contextlib.contextmanager
def few_domain_guard():
    """Convert FEW out-of-domain errors into :class:`WaveformDomainError`.

    Wrap the FEW waveform call with this. Any ``ValueError`` or
    ``AssertionError`` whose message matches :data:`FEW_DOMAIN_ERROR_PATTERNS`
    is re-raised as a :class:`WaveformDomainError` (chained); anything else
    propagates untouched.
    """
    try:
        yield
    except (AssertionError, ValueError) as exc:
        msg = str(exc)
        if any(pattern in msg for pattern in FEW_DOMAIN_ERROR_PATTERNS):
            raise WaveformDomainError(msg) from exc
        raise


def emri_kerr_domain_mask(a, p0, e0, xI=1.0):
    """Vectorized validity mask for the FEW Kerr ecc-eq domain at ``t=0``.

    Mirrors ``KerrEccentricEquatorial.sanity_check_init`` (the forward-map
    ``u``/``w`` grid checks and the ``|a| <= 0.999`` bound) plus the
    trajectory's start requirement ``p0 >= p_sep + 2*DELTAPMIN``
    (``KerrEccEqFlux.separatrix_buffer_dist``), without generating a waveform.
    Mid-trajectory grid exits cannot be predicted here — those are handled by
    :func:`few_domain_guard` at waveform-call time.

    Args:
        a: Dimensionless spin (broadcastable array or scalar).
        p0: Initial semi-latus rectum.
        e0: Initial eccentricity.
        xI: Cosine of inclination, ``+1`` or ``-1`` (scalar fill value).

    Returns:
        Boolean ``np.ndarray`` (broadcast shape), ``True`` where the
        combination is inside the domain of validity.
    """
    from few.utils.geodesic import get_separatrix
    from few.utils.mappings.kerrecceq import DELTAPMIN, kerrecceq_forward_map

    a, p0, e0 = np.broadcast_arrays(
        np.atleast_1d(np.asarray(a, dtype=float)),
        np.atleast_1d(np.asarray(p0, dtype=float)),
        np.atleast_1d(np.asarray(e0, dtype=float)),
    )

    # sanity_check_init's sign convention: negative xI flips both signs.
    xI = float(xI)
    a_sign = a * xI
    xI_in = abs(xI)

    ok = (np.abs(a_sign) <= 0.999) & (e0 >= 0.0) & (e0 < 1.0) & (p0 > 0.0)
    if not np.any(ok):
        return ok

    u = np.full(a.shape, np.nan)
    w = np.full(a.shape, np.nan)
    psep = np.full(a.shape, np.inf)
    # Off-grid corners produce NaNs via log/sqrt of negatives — expected here.
    with np.errstate(all="ignore"):
        grid_coords = kerrecceq_forward_map(
            a_sign[ok], p0[ok], e0[ok], np.full(int(ok.sum()), xI_in)
        )
        u[ok] = grid_coords[0]
        w[ok] = grid_coords[1]
        asign = np.sign(a_sign[ok])
        asign[asign == 0] = 1.0
        psep[ok] = get_separatrix(np.abs(a_sign[ok]), e0[ok], asign * xI_in)

    ok &= ~np.isnan(u)
    ok &= u <= 1.000001
    ok &= (w >= -1e-6) & (w <= 1.000001)
    ok &= p0 >= psep + 2.0 * DELTAPMIN
    return ok
