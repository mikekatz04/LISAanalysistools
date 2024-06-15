import numpy as np
from typing import Optional, Any, Tuple
from copy import deepcopy

from few.waveform import GenerateEMRIWaveform

# imports
from fastlisaresponse import ResponseWrapper
from lisatools.detector import EqualArmlengthOrbits
from ..waveformbase import AETTDIWaveform

from gbgpu.gbgpu import GBGPU


class GBAETWaveform(AETTDIWaveform):
    def __init__(self) -> None:
        # wave generating class
        self.wave_gen = GBGPU()

    @property
    def f_arr(self) -> np.ndarray:
        return self._f_arr

    @f_arr.setter
    def f_arr(self, f_arr: np.ndarray) -> None:
        self._f_arr = f_arr

    def __call__(
        self, *params: Any, **kwargs: Any
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.wave_gen.run_wave(
            *params,
            **kwargs,
        )

        self.f_arr = self.wave_gen.freqs[0]
        A = self.wave_gen.A[0]
        E = self.wave_gen.E[0]
        T = self.wave_gen.X[0]

        return (A, E, T)
