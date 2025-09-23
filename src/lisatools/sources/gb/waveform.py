from __future__ import annotations

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
    """Stock AET waveform for Galactic Binaries.

    Args:
        *args: Arguments for :class:`GBGPU`.
        **kwargs: Arguments for :class:`GBGPU`.

    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # wave generating class
        self.wave_gen = GBGPU(*args, **kwargs)

    @property
    def f_arr(self) -> np.ndarray:
        """Frequency array."""
        return self._f_arr

    @f_arr.setter
    def f_arr(self, f_arr: np.ndarray) -> None:
        """Set the frequency array."""
        self._f_arr = f_arr

    def __call__(
        self, *params: Any, return_array: Optional[bool] = False, **kwargs: Any
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray] | np.ndarray:
        """Generate waveform.

        Args:
            *params: Parameters going into :meth:`GBGPU.run_wave`.
            return_array: If ``True``, return ``array([A, E, T]).
                If ``False``, return (A, E, T).
            **kwargs: Keyword arguments going into :meth:`GBGPU.run_wave`.

        Returns:
            Output waveform.

        """
        self.wave_gen.run_wave(
            *params,
            **kwargs,
        )

        # prepare outputs
        self.f_arr = self.wave_gen.freqs[0]
        A = self.wave_gen.A[0]
        E = self.wave_gen.E[0]
        T = self.wave_gen.X[0]

        if return_array:
            return np.array([A, E, T])
        else:
            return (A, E, T)
