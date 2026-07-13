"""Stellar-origin BBH (SOBBH) sources for the LISA global fit.

Provides the PN inspiral waveform (TaylorT3-style, 3.5PN aligned-spin) and
thin LISA-side wrappers so SOBBHs slot into the global-fit pipeline the
same way EMRIs do.

**Stock-aligned entry.** The default ``all_sources`` SOBBH branch is
TDI-on-the-fly (``SourceSOBBHSettings.use_tdionfly=True``). The single
documented builder that reproduces it exactly is
:func:`build_sobbh_stock_waveform` (``get_sobbh_tdionfly_gen`` +
:class:`SOBBHTDIonFlyWaveWrap`). The legacy pyResponse path
(:func:`get_sobbh_response_wrapper` + :class:`SOBBHWaveWrap`) is kept for the
``use_tdionfly=False`` fallback only.
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
    SOBBH_STOCK_BUFFER_TIME,
    SOBBH_STOCK_N_GRID,
    SOBBH_STOCK_RESPONSE_ORDER,
    SOBBHTDIonFlyWaveWrap,
    SOBBHWaveWrap,
    build_sobbh_stock_waveform,
    get_sobbh_response_wrapper,
    get_sobbh_tdionfly_gen,
)
