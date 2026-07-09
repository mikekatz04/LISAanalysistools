"""Stellar-origin BBH (SOBBH) sources for the LISA global fit.

Provides the PN inspiral waveform (TaylorT3-style, 3.5PN aligned-spin) and
a thin LISA-side wrapper so SOBBHs slot into the global-fit pipeline the
same way EMRIs do.
"""

from .waveform import (  # noqa: F401
    SOBBHWaveform,
    phase,
    frequency,
    frequency_derivative,
    tau_to_x,
    time_to_merger,
    waveform_generate_h_plus_cross,
)
from .response import (  # noqa: F401
    SOBBHWaveWrap,
    SOBBHTDIonFlyWaveWrap,
    get_sobbh_response_wrapper,
    get_sobbh_tdionfly_gen,
)
