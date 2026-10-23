"""Generic default keyword arguments for the LISA TDI response wrapper.

Holds :class:`DefaultResponseKwargs`, a small bundle of :class:`~lisatools.response.ResponseWrapper`
settings for stand-alone examples. The global fit does not use it: its waveform builders set
their own response options (XYZ, TDI 2nd generation, the run's unequal-arm orbits).
"""

from dataclasses import dataclass

from ..detector import EqualArmlengthOrbits, Orbits


@dataclass
class DefaultResponseKwargs:
    """Default response keyword arguments.

    Bundles generic kwargs for :class:`~lisatools.response.directresponse.ResponseWrapper`
    (formerly ``fastlisaresponse.ResponseWrapper``, absorbed into ``lisatools.response``):

    - ``t0 = 30000.0``
    - ``order = 25``
    - ``tdi = "1st generation"``
    - ``tdi_chan = "AET"``
    - ``orbits = EqualArmlengthOrbits()``

    The values are class attributes; ``orbits`` is one
    :class:`~lisatools.detector.EqualArmlengthOrbits` instance built when the class is defined
    (at import) on the default backend, and shared by every :meth:`get_dict` call.

    Note:
        These are legacy example defaults (TDI 1st generation, AET, equal-arm analytic orbits).
        They are not the production model: the global fit uses XYZ TDI 2nd generation on the
        unequal-arm :class:`~lisatools.detector.L1Orbits`.

    Example:
        .. code-block:: python

            from lisatools.response.directresponse import ResponseWrapper
            from lisatools.sources.defaultresponse import DefaultResponseKwargs

            kw = DefaultResponseKwargs.get_dict()
            wrap = ResponseWrapper(waveform_gen, Tobs, dt, index_lambda, index_beta, **kw)
    """

    t0 = 30000.0
    order = 25
    tdi = "1st generation"
    tdi_chan = "AET"
    orbits = EqualArmlengthOrbits()

    @classmethod
    def get_dict(cls) -> dict:
        """Return the default response kwargs as a dictionary.

        Returns:
            dict: ``{"t0", "order", "tdi", "tdi_chan", "orbits"}`` with the class values (the
            ``orbits`` entry is the shared class-level instance, not a copy).
        """
        return dict(
            t0=cls.t0,
            order=cls.order,
            tdi=cls.tdi,
            tdi_chan=cls.tdi_chan,
            orbits=cls.orbits,
        )
