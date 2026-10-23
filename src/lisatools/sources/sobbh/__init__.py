"""Stellar-origin BBH (SOBBH) sources for the LISA global fit.

Provides the PN inspiral waveform (TaylorT3-style, 3.5PN aligned-spin) and
thin LISA-side wrappers so SOBBHs slot into the global-fit pipeline the
same way EMRIs do.

**Production (9mo / 12mo runs).** ``SOBBH_LIKELIHOOD=lookup``: scoring AND the engine's
template fills use the direct-to-WDM lookup scorer
:class:`~lisatools.sources.sobbh.wdm_direct.SOBBHLookupComputations` (behind
:class:`~lisatools.sources.sobbh.wdm_direct.SOBBHLookupRouter`), which calls the free PN
functions of :mod:`~lisatools.sources.sobbh.waveform` row-vectorized. The slow
TDI-on-the-fly wrap is kept for the move's fast-vs-slow cross-check (``SOBBH_CHECK_LL``,
off in production) and as the move's fallback generator for a container that lacks the
branch. ``SOBBH_CHUNKED_FILL=0`` (unset in production, where the batched fill is on)
does not reach it: on the stock path that escape runs the base move's per-row
expose/fold through the container's installed generator, i.e. the same fast comp's
fill one row at a time. Under ``SOBBH_LIKELIHOOD=full`` the slow wrap is the generator of
both the engine and the stock per-row move.

**Stock-aligned waveform entry.** The ``all_sources`` SOBBH waveform wrap is
TDI-on-the-fly (``SourceSOBBHSettings.use_tdionfly=True``). The single
documented builder that reproduces it exactly is
:func:`~lisatools.sources.sobbh.response.build_sobbh_stock_waveform` (``get_sobbh_tdionfly_gen``
+ :class:`~lisatools.sources.sobbh.response.SOBBHTDIonFlyWaveWrap`). The legacy pyResponse path
(:func:`~lisatools.sources.sobbh.response.get_sobbh_response_wrapper` +
:class:`~lisatools.sources.sobbh.response.SOBBHWaveWrap`) is kept for the
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
