"""Composite (additive, optionally time-modulated) sensitivity matrices.

The total noise covariance is built as a sum of independent components. Each
component contributes a covariance in the domain's basis; a component may also
carry a per-element time modulation, so that

.. math::

    C_{ij}[\\text{basis}] = \\sum_c \\big(\\text{base}_{c,ij}[\\text{basis}]\\big)
                            \\cdot M_{c,ij}[\\text{time}]

where ``M_c`` is ``1`` for stationary components. This mirrors the GLASS noise
model (``generate_full_dynamic_covariance_matrix``): a stationary instrument
term, a time-modulated galactic foreground, and a stationary SGWB summed into
one covariance.

**Domain-agnostic by design.** The spectral part of every component goes through
:func:`~lisatools.sensitivity.get_sensitivity`, which dispatches on the domain
settings (FD → ``Sn(f_arr)``, WDM → folded wavelet PSD, …), so the same
component classes work in any domain ``get_sensitivity`` supports. A *constant*
modulation works in every domain; a *time-varying* modulation requires the
domain to have a time axis (WDM, STFT, TD) — see :func:`_basis_time_axis`.

The assembled ``(nch, nch, *basis_shape_active)`` array is handed to
:class:`~lisatools.sensitivity.SensitivityMatrixBase`, which computes ``detC`` /
``invC`` — so :func:`~lisatools.diagnostic.inner_product` and
:func:`~lisatools.diagnostic.noise_likelihood_term` consume the result with no
changes.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np

try:
    import cupy as cp
except (ModuleNotFoundError, ImportError):
    import numpy as cp

from . import domains
from .sensitivity import (
    SensitivityMatrixBase,
    get_sensitivity,
    X1TDISens,
    Y1TDISens,
    Z1TDISens,
    XY1TDISens,
    ZX1TDISens,
    YZ1TDISens,
    X2TDISens,
    Y2TDISens,
    Z2TDISens,
    XY2TDISens,
    ZX2TDISens,
    YZ2TDISens,
)
from .stochastic import HyperbolicTangentGalacticForeground, check_stochastic
from .utils.utility import get_array_module

# Upper-triangle covariance elements, in the order used throughout this module.
ELEMENTS = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
ELEMENT_NAMES = ["XX", "YY", "ZZ", "XY", "XZ", "YZ"]

# TDI XYZ element sensitivity classes per generation, matching ELEMENTS order.
_XYZ_ELEMENT_SENS = {
    1: [X1TDISens, Y1TDISens, Z1TDISens, XY1TDISens, ZX1TDISens, YZ1TDISens],
    2: [X2TDISens, Y2TDISens, Z2TDISens, XY2TDISens, ZX2TDISens, YZ2TDISens],
}


def _basis_time_axis(settings: domains.DomainSettingsBase) -> Optional[int]:
    """Index of the time axis within ``basis_shape_active``, or ``None``.

    Domains without a time axis (e.g. frequency-domain) return ``None``; only a
    constant modulation is meaningful for those.
    """
    if isinstance(settings, domains.WDMSettings):
        return 1  # basis_shape_active = (Nf, Nt)
    if isinstance(settings, domains.STFTSettings):
        return 0  # basis_shape_active = (NT, NF)
    if isinstance(settings, domains.TDSettings):
        return 0  # basis_shape_active = (N,)
    return None  # FDSettings and anything else with no time axis


def modulation_from_elements(
    elements: dict, nchannels: int = 3
) -> Callable[[np.ndarray], np.ndarray]:
    """Build a ``(nch, nch, Ntime)`` modulation callable from six per-element entries.

    Args:
        elements: Mapping from element name (``"XX"``, ``"YY"``, ``"ZZ"``,
            ``"XY"``, ``"XZ"``, ``"YZ"``) to a callable ``t_arr -> (Ntime,)`` or a
            precomputed length-``Ntime`` array (or a scalar).
        nchannels: Number of channels (3 for XYZ).

    Returns:
        A callable ``t_arr -> (nch, nch, Ntime)`` filling the symmetric matrix.
    """

    def _mod(t_arr):
        xp = get_array_module(t_arr)
        nt = t_arr.shape[0]
        M = xp.zeros((nchannels, nchannels, nt))
        for name, (i, j) in zip(ELEMENT_NAMES, ELEMENTS):
            val = elements[name]
            arr = val(t_arr) if callable(val) else xp.asarray(val)
            M[i, j] = arr
            M[j, i] = arr
        return M

    return _mod


class NoiseComponent:
    """Base class for an additive contribution to the noise covariance.

    Subclasses return the full ``(nch, nch, *basis_shape_active)`` contribution
    for the given domain settings. Use :class:`SeparableComponent` for the common
    factorised case (a base covariance times a per-element modulation).
    """

    name: str = "component"
    nchannels: int = 3

    def covariance(self, settings: domains.DomainSettingsBase) -> np.ndarray:
        raise NotImplementedError


class SeparableComponent(NoiseComponent):
    """A component whose covariance factorises as ``base_ij[basis] * M_ij[time]``.

    Subclasses provide :meth:`base_covariance` (the stationary covariance in the
    domain basis) and may override :meth:`time_modulation` (per-element factor;
    defaults to ``None`` = stationary). The modulation may be:

    * ``None`` — stationary (the base covariance is returned unchanged);
    * a ``(nch, nch)`` constant matrix — applied in any domain;
    * a ``(nch, nch, Ntime)`` array — requires a time axis; broadcast along it.
    """

    def base_covariance(self, settings: domains.DomainSettingsBase) -> np.ndarray:
        """Stationary covariance in the domain basis: ``(nch, nch, *basis_shape_active)``."""
        raise NotImplementedError

    def time_modulation(self, settings: domains.DomainSettingsBase):
        """Per-element modulation: ``None``, ``(nch,nch)``, or ``(nch,nch,Ntime)``."""
        return None

    def covariance(self, settings: domains.DomainSettingsBase) -> np.ndarray:
        base = self.base_covariance(settings)  # (nch, nch, *basis)
        mod = self.time_modulation(settings)
        if mod is None:
            return base

        xp = settings.xp
        mod = xp.asarray(mod)
        nbasis = len(settings.basis_shape_active)

        if mod.ndim == 2:
            # constant matrix: broadcast over all basis axes (works in any domain)
            idx = (slice(None), slice(None)) + (None,) * nbasis
            return base * mod[idx]

        if mod.ndim == 3:
            time_axis = _basis_time_axis(settings)
            if time_axis is None:
                raise ValueError(
                    f"{type(settings).__name__} has no time axis; a time-varying "
                    "modulation (nch, nch, Ntime) is not allowed — use a constant "
                    "(nch, nch) modulation instead."
                )
            ntime = mod.shape[2]
            if ntime != settings.basis_shape_active[time_axis]:
                raise ValueError(
                    f"modulation time length {ntime} != basis time length "
                    f"{settings.basis_shape_active[time_axis]}."
                )
            # place Ntime on the (channel-offset) time axis, size-1 elsewhere
            shape = list(base.shape[:2]) + [1] * nbasis
            shape[2 + time_axis] = ntime
            return base * mod.reshape(shape)

        raise ValueError("modulation must be 2D (nch,nch) or 3D (nch,nch,Ntime).")


class InstrumentNoise(SeparableComponent):
    """Stationary TDI instrument-noise covariance (no time modulation).

    Args:
        tdi_generation: 1 (TDI 1.5) or 2 (TDI 2.0).
        model: LISA noise model (name or :class:`~lisatools.detector.LISAModel`).
        fill_nans: Passed to :func:`get_sensitivity` (default ``np.nan``, matching
            the stock matrices, leaves the ``f=0`` bin non-finite).
    """

    name = "instrument"

    def __init__(self, tdi_generation: int = 2, model="sangria", fill_nans: float = np.nan):
        if tdi_generation not in _XYZ_ELEMENT_SENS:
            raise ValueError(f"tdi_generation must be 1 or 2, got {tdi_generation!r}.")
        self.tdi_generation = tdi_generation
        self.model = model
        self.fill_nans = fill_nans
        self.element_sens_fns = _XYZ_ELEMENT_SENS[tdi_generation]

    def base_covariance(self, settings: domains.DomainSettingsBase) -> np.ndarray:
        xp = settings.xp
        nch = self.nchannels
        elems = [
            get_sensitivity(settings, sens_fn=fn, model=self.model, fill_nans=self.fill_nans)
            for fn in self.element_sens_fns
        ]
        C = xp.zeros((nch, nch) + tuple(settings.basis_shape_active), dtype=elems[0].dtype)
        for (i, j), arr in zip(ELEMENTS, elems):
            C[i, j] = arr
            C[j, i] = arr
        return C


class GalacticForeground(SeparableComponent):
    """Galactic confusion foreground with a per-element time modulation.

    Base covariance (GLASS convention): the foreground *magnitude*
    ``Sgal_mag[basis]`` — the auto-channel (XX) foreground in the domain basis —
    is placed on every element, computed domain-agnostically as the difference of
    the X-channel sensitivity with and without the stochastic foreground. The
    per-element structure (including the off-diagonal sign) and the slow time
    variation live entirely in the modulation.

    Args:
        foreground_params: Parameters for ``stochastic_fn`` (for the default
            :class:`HyperbolicTangentGalacticForeground` these are
            ``(amp, fk, alpha, s1, s2)``).
        modulation: One of: ``None`` (the isotropic/stationary limit — diagonals
            ``1``, off-diagonals ``-1/2`` — which reproduces the stationary
            foreground); a ``(nch, nch)`` constant matrix; a ``(nch, nch, Ntime)``
            array; or a callable ``t_arr -> (nch, nch, Ntime)`` (build one from
            six per-element functions with :func:`modulation_from_elements`).
        tdi_generation: 1 or 2 (used to pick the X-channel sensitivity used to
            extract the foreground magnitude).
        model: LISA noise model (must match the instrument component).
        stochastic_fn: Stochastic foreground model (class or name).
    """

    name = "galactic_foreground"

    def __init__(
        self,
        foreground_params: Sequence[float],
        modulation: Optional[object] = None,
        tdi_generation: int = 2,
        model="sangria",
        stochastic_fn=HyperbolicTangentGalacticForeground,
    ):
        if tdi_generation not in _XYZ_ELEMENT_SENS:
            raise ValueError(f"tdi_generation must be 1 or 2, got {tdi_generation!r}.")
        self.foreground_params = tuple(foreground_params)
        self._modulation = modulation
        self.tdi_generation = tdi_generation
        self.model = model
        self.stochastic_fn = check_stochastic(stochastic_fn)

    def base_covariance(self, settings: domains.DomainSettingsBase) -> np.ndarray:
        xp = settings.xp
        nch = self.nchannels
        Xsens = _XYZ_ELEMENT_SENS[self.tdi_generation][0]
        # foreground magnitude in the domain basis = (instrument + foreground) - instrument
        with_fg = get_sensitivity(
            settings,
            sens_fn=Xsens,
            model=self.model,
            stochastic_params=self.foreground_params,
            stochastic_function=self.stochastic_fn,
            fill_nans=0.0,
        )
        instr = get_sensitivity(settings, sens_fn=Xsens, model=self.model, fill_nans=0.0)
        mag = with_fg - instr
        C = xp.zeros((nch, nch) + tuple(settings.basis_shape_active), dtype=mag.dtype)
        for (i, j) in ELEMENTS:
            C[i, j] = mag
            C[j, i] = mag
        return C

    def time_modulation(self, settings: domains.DomainSettingsBase):
        xp = settings.xp
        nch = self.nchannels

        if self._modulation is None:
            # isotropic / stationary limit: diag = 1, off-diag = -1/2 (constant)
            M = xp.full((nch, nch), -0.5)
            for i in range(nch):
                M[i, i] = 1.0
            return M

        if callable(self._modulation):
            time_axis = _basis_time_axis(settings)
            if time_axis is None:
                raise ValueError(
                    f"{type(settings).__name__} has no time axis; cannot evaluate a "
                    "callable (time-varying) foreground modulation here."
                )
            return self._modulation(settings.t_arr)

        return xp.asarray(self._modulation)


class CompositeSensitivityMatrix(SensitivityMatrixBase):
    """Sensitivity matrix built as a sum of :class:`NoiseComponent` objects.

    Args:
        settings: Domain settings the matrix is evaluated on (FD, WDM, …).
        components: :class:`NoiseComponent` list to sum. All must produce the same
            ``(nch, nch, *basis_shape_active)`` shape.
        skip_inv_det: Skip determinant/inverse computation (e.g. for slicing).
    """

    def __init__(
        self,
        settings: domains.DomainSettingsBase,
        components: Sequence[NoiseComponent],
        skip_inv_det: bool = False,
    ):
        SensitivityMatrixBase.__init__(self, settings, skip_inv_det=skip_inv_det)
        self.components = list(components)
        if not self.components:
            raise ValueError("CompositeSensitivityMatrix needs at least one component.")
        # Per-component contribution cache: a param change only recomputes the
        # touched component before re-summing (the expensive det/inv runs once).
        self._contrib_cache: dict[int, np.ndarray] = {}
        self.rebuild()

    def rebuild(self, indices: Optional[Sequence[int]] = None) -> None:
        """(Re)compute component contributions and re-sum into ``sens_mat``.

        Args:
            indices: Component indices to recompute. ``None`` recomputes all;
                cached contributions are reused for the rest.
        """
        if indices is None:
            indices = range(len(self.components))
        for i in indices:
            self._contrib_cache[i] = self.components[i].covariance(self.basis_settings)

        C = None
        for i in range(len(self.components)):
            contrib = self._contrib_cache[i]
            C = contrib if C is None else C + contrib
        self.sens_mat = C  # triggers detC / invC in SensitivityMatrixBase

    def update_component(self, index: int) -> None:
        """Recompute a single component (after changing its params) and re-sum."""
        self.rebuild(indices=[index])
