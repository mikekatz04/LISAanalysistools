"""EMRI TDI waveform: the LISA response around FEW's ``GenerateEMRIWaveform``.

**Alignment guarantee (2026-07-12).** A bare :class:`EMRITDIWaveform` builds
its generator through
:func:`lisatools.sources.emri.response.get_emri_response_wrapper` — the *same*
builder the stock ``all_sources`` global fit uses for its default EMRI branch
(see ``lisatools.globalfit.stock.erebor.source_runtime.SourceEMRISettings`` and
``get_emri_wave_wrap``). Its defaults therefore match the global-fit EMRI path
exactly:

* ``FastKerrEccentricEquatorialFlux`` intrinsic model,
* 2nd-generation TDI, ``XYZ`` channels, response ``order=40``,
* ``flip_hx=True``, ``remove_garbage="zero"``, ``is_ecliptic_latitude=False``
  (ecliptic-polar sky: ``index_beta=7`` carries ``qS``),
* the validated SPECIAL frame (``special_frame=True``) that converts the
  ecliptic-polar sky to the ICRS orbit frame per call.

Constructing this class and the global fit's EMRI ``signal_gen`` with the same
run essentials (``T``, ``dt``, ``t0``, ``orbits``) yields byte-identical raw
TDI channels — the generator is even shared through the module-level cache in
``get_emri_response_wrapper``.

The response ``order`` default lives in :data:`EMRI_STOCK_RESPONSE_ORDER`; the
alignment test ``tests/test_stock_waveform_alignment.py`` asserts it stays equal
to the erebor ``SourceEMRISettings.response_order``.
"""

from __future__ import annotations

from typing import Any, Optional

from ...utils.constants import YRSID_SI

# imports
from ..waveformbase import AETTDIWaveform

# Stock EMRI response defaults — the sources-side single source of truth,
# mirrored by ``SourceEMRISettings.response_order`` and the
# ``get_emri_response_wrapper`` signature. Keep these equal to the erebor
# values (enforced by tests/test_stock_waveform_alignment.py).
EMRI_STOCK_RESPONSE_ORDER = 40
EMRI_STOCK_TDI_GENERATION = "2nd generation"
EMRI_STOCK_TDI_CHAN = "XYZ"


class EMRITDIWaveform(AETTDIWaveform):
    """Generate EMRI waveforms with the LISA TDI response (global-fit aligned).

    Delegates to :func:`get_emri_response_wrapper`, so a bare instance
    reproduces the stock ``all_sources`` EMRI generator (see the module
    docstring for the alignment guarantee).

    Args:
        T: Observation time in years.
        dt: Time cadence in seconds.
        t0: Epoch the intrinsic FEW phases (``Phi_phi0`` ...) are referenced to
            (the response ``t0``). For mojito data this is the catalogue
            reference time; ``0.0`` for a stand-alone waveform.
        tdi_gen: TDI generation string (``"2nd generation"`` — the global-fit
            default).
        tdi_chan: TDI channel set (``"XYZ"`` — the global-fit default).
        order: Lagrange-interpolation order of the pyResponseTDI response
            (``40`` — matches ``SourceEMRISettings.response_order``).
        t_buffer: Response garbage buffer (seconds).
        orbits: :class:`~lisatools.detector.Orbits` for the response. ``None``
            builds a fresh :class:`~lisatools.detector.EqualArmlengthOrbits` on
            ``force_backend`` inside :func:`get_emri_response_wrapper` (no heavy
            object is constructed at module import).
        force_backend: Backend selection at construction (``"cpu"`` /
            ``"cuda12x"`` / ...); never passed as a per-call kwarg.
        special_frame: When ``True`` (default, the validated mojito /
            ``all_sources`` convention) the generator is wrapped so every call
            converts the ecliptic-polar sky to the ICRS orbit frame. Set
            ``False`` for a plain ecliptic-frame smoke waveform.
        t0_shift_to_data: Sub-sample (``|.| < dt``) alignment of the
            REF-anchored waveform onto the data grid (global-fit internal use).
        inspiral_kwargs, sum_kwargs, mode_selector_kwargs: Optional overrides
            for the FEW generator; ``None`` keeps the shared stock defaults
            (``EMRI_INSPIRAL_KWARGS`` / ``EMRI_SUM_KWARGS`` /
            ``EMRI_MODE_SELECTOR_KWARGS``).
        response_kwargs: **Deprecated** back-compat dict. If given, its
            ``t0`` / ``order`` / ``tdi_chan`` / ``orbits`` entries override the
            corresponding keyword arguments.
    """

    def __init__(
        self,
        T: Optional[float] = 1.0,
        dt: Optional[float] = 10.0,
        *,
        t0: float = 0.0,
        tdi_gen: str = EMRI_STOCK_TDI_GENERATION,
        tdi_chan: str = EMRI_STOCK_TDI_CHAN,
        order: int = EMRI_STOCK_RESPONSE_ORDER,
        t_buffer: float = 3e4,
        orbits: Optional[Any] = None,
        force_backend: str = "cpu",
        special_frame: bool = True,
        t0_shift_to_data: float = 0.0,
        inspiral_kwargs: Optional[dict] = None,
        sum_kwargs: Optional[dict] = None,
        mode_selector_kwargs: Optional[dict] = None,
        response_kwargs: Optional[dict] = None,
    ) -> None:
        # Lazy import keeps the (few-importing) response module out of the way
        # of a plain ``import lisatools.sources.emri.waveform`` and avoids any
        # package-init ordering surprises.
        from ...response.tdiconfig import TDIConfig
        from .response import get_emri_response_wrapper

        # Back-compat: legacy callers passed a single ``response_kwargs`` dict.
        # Pull the still-meaningful knobs out of it so old call sites keep
        # working against the aligned builder.
        if response_kwargs:
            t0 = response_kwargs.get("t0", t0)
            order = response_kwargs.get("order", order)
            tdi_chan = response_kwargs.get("tdi_chan", tdi_chan)
            orbits = response_kwargs.get("orbits", orbits)

        self._dt = dt
        tdi_config = TDIConfig(tdi_gen, force_backend=force_backend)
        self.response = get_emri_response_wrapper(
            Tobs=T * YRSID_SI,
            dt=dt,
            t_start=t0,
            tdi_config=tdi_config,
            tdi_chan=tdi_chan,
            order=order,
            t_buffer=t_buffer,
            orbits=orbits,
            force_backend=force_backend,
            t0_shift_to_data=t0_shift_to_data,
            special_frame=special_frame,
            inspiral_kwargs=inspiral_kwargs,
            sum_kwargs=sum_kwargs,
            mode_selector_kwargs=mode_selector_kwargs,
        )

    @property
    def dt(self) -> float:
        """Timestep in seconds."""
        return self.response.dt

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Generate the EMRI TDI waveform by delegating to the wrapped response.

        Args:
            *args: Positional arguments forwarded to the response generator
                (the FEW 14-parameter waveform basis).
            **kwargs: Keyword arguments forwarded to the response generator.

        Returns:
            The list of per-channel TDI arrays from the underlying response.
        """
        # Typed re-raise of FEW's bare out-of-domain errors (see .domain).
        from .domain import few_domain_guard

        with few_domain_guard():
            return self.response(*args, **kwargs)
