import numpy as np
from typing import Optional, Any, Tuple
from copy import deepcopy

# imports
from fastlisaresponse import ResponseWrapper
from bbhx.waveformbuild import BBHWaveformFD
from ...utils.constants import *

from ..defaultresponse import default_response_kwargs
from ..waveformbase import SNRWaveform


class BBHSNRWaveform(SNRWaveform):
    def __init__(
        self,
        bbh_waveform_kwargs: Optional[dict] = {"run_phenomd": False},
        response_kwargs: Optional[dict] = {"TDItag": "AET"},
    ) -> None:
        # wave generating class
        self.wave_gen = BBHWaveformFD(
            amp_phase_kwargs=bbh_waveform_kwargs,
            response_kwargs=response_kwargs,
        )

    @property
    def f_arr(self) -> np.ndarray:
        return self._f_arr

    @f_arr.setter
    def f_arr(self, f_arr: np.ndarray) -> None:
        self._f_arr = f_arr

    def __call__(
        self, *params: Any, **kwargs: Any
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        m1 = params[0]
        m2 = params[1]

        min_f = 1e-4 / (MTSUN_SI * (m1 + m2))
        max_f = 0.6 / (MTSUN_SI * (m1 + m2))

        self.f_arr = np.logspace(np.log10(min_f), np.log10(max_f), 1024)
        AET = self.wave_gen(
            *params,
            direct=True,
            combine=True,
            freqs=self.f_arr,
            **kwargs,
        )[0]

        return (AET[0], AET[1], AET[2])
