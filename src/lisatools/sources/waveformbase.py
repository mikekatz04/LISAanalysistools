from abc import ABC
from typing import Union, Tuple
import numpy as np


class AETTDIWaveform(ABC):
    """Base class for an AET TDI Waveform."""

    @property
    def dt(self) -> float:
        """Timestep in seconds."""
        return None

    @property
    def f_arr(self) -> np.ndarray:
        """Frequency array."""
        return None

    @property
    def df(self) -> float:
        """Frequency bin size."""
        return None


class SNRWaveform(ABC):
    """Base class for a waveform built in a simpler fashion for SNR calculations."""

    @property
    def dt(self) -> float:
        """Timestep in seconds."""
        return None

    @property
    def f_arr(self) -> np.ndarray:
        """Frequency array."""
        return None

    @property
    def df(self) -> float:
        """Frequency bin size."""
        return None
