"""Waveform generators and calculation controllers for LISA source classes.

Subpackages, one per source class:

- :mod:`lisatools.sources.gb`: galactic binaries (stand-alone GBGPU FD templates; the global
  fit's GB likelihood lives in GBGPU itself).
- :mod:`lisatools.sources.bbh`: massive black hole binaries (phentax IMRPhenomTHM with LAT's
  ``pyResponseTDI``; the grid-aligned batched generators).
- :mod:`lisatools.sources.emri`: EMRIs (FEW; the direct-to-WDM lookup template and the legacy
  ``ResponseWrapper`` path).
- :mod:`lisatools.sources.sobbh`: stellar-origin BBHs (3.5PN inspiral; the WDM lookup scorer).

Shared modules: :mod:`~lisatools.sources.waveformbase` (time-domain waveform base classes),
:mod:`~lisatools.sources.batching` (batched ``signal_gen`` adapters),
:mod:`~lisatools.sources.utils` (ICRS / ecliptic conversions and the calculation
controllers) and :mod:`~lisatools.sources.defaultresponse`.

The global-fit production paths are the EMRI direct-to-WDM template
(:mod:`lisatools.sources.emri.wdm_direct`, ``EMRI_LIKELIHOOD=direct``), the SOBBH lookup scorer
(:mod:`lisatools.sources.sobbh.wdm_direct`, ``SOBBH_LIKELIHOOD=lookup``) and the windowed
grid-aligned MBH generator (:mod:`lisatools.sources.bbh.gridaligned` through
:class:`~lisatools.sources.batching.MBHWindowedWDMSignalGen`, ``MBH_LIKELIHOOD=batched``).

This package re-exports the calculation controllers from :mod:`~lisatools.sources.utils`.
"""

from .utils import (
    BBHCalculationController,
    CalculationController,
    EMRICalculationController,
    GBCalculationController,
)
