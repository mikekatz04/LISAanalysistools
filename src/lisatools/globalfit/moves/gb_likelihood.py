"""Deprecation shim: the GB band likelihood engines moved to GBGPU.

The engine layer wraps the GB fast-waveform computation objects
(``gbgpu.gbcomps.GBFDComputations`` / ``GBWDMComputations``), so it lives
with them in :mod:`gbgpu.gb_likelihood` (2026-07 GB-proposal rework).
This module re-exports the same class objects so existing imports keep
working; new code should import from ``gbgpu.gb_likelihood``.

Re-exported names: ``BandLikelihoodEngine``, ``FDBandLikelihoodEngine``,
``WDMBandLikelihoodEngine``, ``SwapLLResult`` and
``make_band_likelihood_engine``. Importing the module emits a
:class:`DeprecationWarning`. The global-fit moves do not use it:
:mod:`lisatools.globalfit.moves.gbbands` imports ``gbgpu.gb_likelihood``
directly.
"""

# TODO check: keep/remove/document -- deprecation shim; only tests/ and scripts/validation import it
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
