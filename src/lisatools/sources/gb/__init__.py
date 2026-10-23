"""Galactic binary (GB) waveform generators.

:class:`~lisatools.sources.gb.waveform.GBXYZTDIWaveform` is the primary stand-alone GB
waveform (XYZ, TDI 2nd generation). :class:`~lisatools.sources.gb.waveform.GBAETWaveform` is
the legacy AET wrapper kept for back-compat. The global fit's GB likelihood runs in GBGPU
(signal-heterodyne / WDM lookup computations), not through these classes.
"""

from .waveform import GBXYZTDIWaveform, GBAETWaveform
