from __future__ import annotations
from lisatools.detector import EqualArmlengthOrbits
import numpy as np
from typing import Optional, Any
from copy import deepcopy

from few.waveform import GenerateEMRIWaveform

# imports
from ..waveformbase import AETTDIWaveform
from fastlisaresponse import ResponseWrapper
from ...detector import EqualArmlengthOrbits

default_response_kwargs = dict(
    t0=30000.0,
    order=25,
    tdi="1st generation",
    tdi_chan="AET",
    orbits=EqualArmlengthOrbits(),
)


class EMRITDIWaveform(AETTDIWaveform):
    """Generate EMRI waveforms with the TDI LISA Response.

    Args:
        T: Observation time in years.
        dt: Time cadence in seconds.
        emri_waveform_args: Arguments for :class:`GenerateEMRIWaveforms`.
        emri_waveform_kwargs: Keyword arguments for :class:`GenerateEMRIWaveforms`.
        response_kwargs: Keyword arguments for :class:`ResponseWrapper`.

    """

    def __init__(
        self,
        T: Optional[float] = 1.0,
        dt: Optional[float] = 10.0,
        emri_waveform_args: Optional[tuple] = ("FastKerrEccentricEquatorialFlux",),
        emri_waveform_kwargs: Optional[dict] = {},
        response_kwargs: Optional[dict] = default_response_kwargs,
    ):
        # sky parameters in GenerateEMRIWaveform
        index_lambda = 8
        index_beta = 7

        for key in default_response_kwargs:
            response_kwargs[key] = response_kwargs.get(
                key, default_response_kwargs[key]
            )
        gen_wave = GenerateEMRIWaveform(
            *emri_waveform_args,
            sum_kwargs=dict(pad_output=True),
            **emri_waveform_kwargs,
        )

        response_kwargs_in = deepcopy(response_kwargs)
        # parameters
        self.response = ResponseWrapper(
            gen_wave,
            T,
            dt,
            index_lambda,
            index_beta,
            flip_hx=True,  # set to True if waveform is h+ - ihx
            remove_sky_coords=False,
            is_ecliptic_latitude=False,
            remove_garbage=True,  # removes the beginning of the signal that has bad information
            **response_kwargs_in,
        )

    @property
    def dt(self) -> float:
        """timestep"""
        return self.response.dt

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        __doc__ = ResponseWrapper.__call__.__doc__
        return self.response(*args, **kwargs)
