from dataclasses import dataclass

from ..detector import EqualArmlengthOrbits, Orbits


@dataclass
class DefaultResponseKwargs:
    """Default response kwargs

    Default response kwargs for
    `fastlisaresponse.ResponseWrapper <https://mikekatz04.github.io/lisa-on-gpu/html/user/main.html#response-function-wrapper>`_.

    ``t0=30000.0``
    ``order=25``
    ``tdi="1st generation"``
    ``tdi_chan="AET"``
    ``orbits=EqualArmlengthOrbits()``

    """

    t0 = 30000.0
    order = 25
    tdi = "1st generation"
    tdi_chan = "AET"
    orbits = EqualArmlengthOrbits()

    @classmethod
    def get_dict(cls) -> dict:
        """Return default dictionary"""
        return dict(
            t0=cls.t0,
            order=cls.order,
            tdi=cls.tdi,
            tdi_chan=cls.tdi_chan,
            orbits=cls.orbits,
        )
