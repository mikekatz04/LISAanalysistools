"""Galactic binary (GB) waveform generators.

:class:`GBXYZTDIWaveform` is the primary, global-fit-aligned GB waveform (XYZ,
TDI 2nd generation). :class:`GBAETWaveform` is the legacy AET wrapper kept for
back-compat.
"""

from .waveform import GBXYZTDIWaveform, GBAETWaveform
