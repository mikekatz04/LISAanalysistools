from ..detector import EqualArmlengthOrbits

default_response_kwargs = dict(
    t0=30000.0,
    order=25,
    tdi="1st generation",
    tdi_chan="AET",
    orbits=EqualArmlengthOrbits(),
)
