"""Deprecation shim: the GB band likelihood engines moved to GBGPU.

The engine layer wraps the GB fast-waveform computation objects
(``gbgpu.gbcomps.GBFDComputations`` / ``GBWDMComputations``), so it lives
with them in :mod:`gbgpu.gb_likelihood` (2026-07 GB-proposal rework).
This module re-exports the same class objects so existing imports keep
working; new code should import from ``gbgpu.gb_likelihood``.
"""

import warnings

from gbgpu.gb_likelihood import (  # noqa: F401
    BandLikelihoodEngine,
    FDBandLikelihoodEngine,
    SwapLLResult,
    WDMBandLikelihoodEngine,
    make_band_likelihood_engine,
)

warnings.warn(
    "lisatools.globalfit.moves.gb_likelihood moved to gbgpu.gb_likelihood; "
    "this shim re-exports the same objects and will be removed.",
    DeprecationWarning,
    stacklevel=2,
)
