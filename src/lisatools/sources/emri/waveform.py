"""EMRI waveform classes wrapping ``few``'s :class:`GenerateEMRIWaveform` with the LISA TDI response."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Optional, Tuple

from lisatools.utils.typing import NDArrayLike
import numpy as np
from few.waveform.waveform import GenerateEMRIWaveform

from ..waveformbase import AETTDIWaveform, TDPyResponseWaveformBase
from ..utils import icrs_to_ecliptic
from ...detector import EqualArmlengthOrbits
from ...response.directresponse import ResponseWrapper
from ...utils.constants import YRSID_SI
from ...utils.utility import get_array_module

# try:
#     import cupy as cp
# except ImportError:
#     pass


# _ = few.get_backend('cuda12x')


default_response_kwargs = dict(
    t0=30000.0,
    order=25,
    tdi="1st generation",
    tdi_chan="AET",
    orbits=EqualArmlengthOrbits(),
)


class LegacyEMRITDIWaveform(AETTDIWaveform):
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
            response_kwargs[key] = response_kwargs.get(key, default_response_kwargs[key])
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
            **response_kwargs_in,
        )

    @property
    def dt(self) -> float:
        """Timestep in seconds."""
        return self.response.dt

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Generate the EMRI TDI waveform by delegating to the wrapped response.

        Args:
            *args: Positional arguments forwarded to :meth:`ResponseWrapper.__call__`.
            **kwargs: Keyword arguments forwarded to :meth:`ResponseWrapper.__call__`.

        Returns:
            The output of the underlying :class:`ResponseWrapper` call (TDI channels).
        """
        __doc__ = ResponseWrapper.__call__.__doc__
        try:
            return self.response(*args, **kwargs)
        except Exception as e:
            print(e)
            breakpoint()

class EMRITDIWaveform(TDPyResponseWaveformBase):
    """Generate EMRI waveforms with the TDI LISA Response.

    Args:
        waveform_class: The waveform class to use for generating the EMRI waveform. Propagated to :class:`few.waveform.waveform.GenerateEMRIWaveform`.
        waveform_kwargs: Keyword arguments for :class:`few.waveform.waveform.GenerateEMRIWaveform`.

        *args: Additional positional arguments forwarded to :class:`TDPyResponseWaveformBase`.
        **kwargs: Additional keyword arguments forwarded to :class:`TDPyResponseWaveformBase`.
    """

    default_kwargs = {
        "frame": "detector",
        "return_list": False,
    }

    def __init__(
        self,
        waveform_class: str = "FastKerrEccentricEquatorialFlux",
        waveform_kwargs: Optional[dict] = None,
        *args: Any,
        **kwargs: Any
    ):
        if waveform_kwargs is None:
            waveform_kwargs = {}

        super().__init__(*args, **kwargs)

        # update the input kwargs with the default kwargs
        for key, value in self.default_kwargs.items():
            kwargs[key] = kwargs.get(key, value)

        self.waveform_class = waveform_class
        self.waveform_kwargs = waveform_kwargs

        self.waveform = GenerateEMRIWaveform(
            self.waveform_class,
            force_backend=self.force_backend,
            **self.waveform_kwargs,
        )

    @property
    def kwargs(self):
        wrapper_kwargs = self.wrapper_kwargs # get the TD TDI wrapper-specific kwargs

        return {
            "waveform_class": self.waveform_class,
            "waveform_kwargs": self.waveform_kwargs,
            **wrapper_kwargs,
        }
    
    def convert_sky_coords(self, ra_or_lambda: float, dec_or_beta: float) -> tuple[float, float]:
        """
        Convert sky from the frame required by the response to the one required by FEW. 
        
        If the orbits are in ecliptic coordinates, convert from ecliptic to polar and azimuthal.
        If the orbits are in ICRS coordinates, convert from ICRS to polar and azimuthal.

        Args:
            ra_or_lambda: Right ascension or ecliptic longitude in radians.
            dec_or_beta: Declination or ecliptic latitude in radians.
        
        Returns:
            tuple[float, float]: Polar angle (theta) and azimuthal angle (phi) in radians.
        """
        if self.orbits.frame == "icrs":
            ra_or_lambda, dec_or_beta = icrs_to_ecliptic(ra_or_lambda, dec_or_beta)
        
        # now go to polar angle 
        qS = np.pi / 2 - dec_or_beta
        phiS = ra_or_lambda
        return qS, phiS
    
    def wave_gen(
            self, 
            m1: float,
            m2: float,
            a: float,
            p0: float,
            e0: float,
            x0: float,
            distance: float,
            qK: float,
            phiK: float,
            Phi_phi0: float,
            Phi_theta0: float,
            Phi_r0: float,
            ra: float,
            dec: float,
            merger_time: float,
            **kwargs
            ) -> tuple[NDArrayLike, NDArrayLike, NDArrayLike]:
        """
        Generate the EMRI waveform's polarizations for a single source.

        Args:
            m1: Mass of the primary black hole in solar masses.
            m2: Mass of the secondary object in solar masses.
            a: Spin of the primary black hole (dimensionless).
            p0: Initial semi-latus rectum of the orbit.
            e0: Initial eccentricity of the orbit.
            x0: Initial inclination parameter.
            distance: Distance to the source in Gpc.
            qK: Polarization angle in radians.
            phiK: Azimuthal angle in radians.
            Phi_phi0: Initial phase of the azimuthal motion in radians.
            Phi_theta0: Initial phase of the polar motion in radians.
            Phi_r0: Initial phase of the radial motion in radians.
            ra: Right ascension of the source in radians.
            dec: Declination of the source in radians.
            merger_time: Time of merger in seconds.
            **kwargs: Additional keyword arguments forwarded to :class:`few.waveform.waveform.GenerateEMRIWaveform`.

        Returns:
            tuple[NDArrayLike, NDArrayLike, NDArrayLike]: The time array and the generated waveform polarizations.

        """

        qS, phiS = self.convert_sky_coords(ra, dec)

        kwargs["T"] = int(merger_time / self.dt) * self.dt / YRSID_SI  # convert to years
        kwargs["dt"] = self.dt

        wave_out = self.waveform(m1, m2, a, p0, e0, x0, distance, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0, **kwargs)

        times = self.xp.arange(0, wave_out.shape[0]) * self.dt
        hplus = wave_out.real
        hcross = -1 * wave_out.imag

        return times, hplus, hcross

    def wave_gen_batch(
            self, 
            m1: NDArrayLike,
            m2: NDArrayLike,
            a: NDArrayLike,
            p0: NDArrayLike,
            e0: NDArrayLike,
            x0: NDArrayLike,
            distance: NDArrayLike,
            qK: NDArrayLike,
            phiK: NDArrayLike,
            Phi_phi0: NDArrayLike,
            Phi_theta0: NDArrayLike,
            Phi_r0: NDArrayLike,
            ra: NDArrayLike,
            dec: NDArrayLike,
            merger_time: float,
            **kwargs
            ) -> tuple[NDArrayLike, NDArrayLike, NDArrayLike]:
        """
        Generate the EMRI waveform's polarizations for a batch of sources, looping over the batch dimension.

        Args:
            m1: Mass of the primary black hole in solar masses.
            m2: Mass of the secondary object in solar masses.
            a: Spin of the primary black hole (dimensionless).
            p0: Initial semi-latus rectum of the orbit.
            e0: Initial eccentricity of the orbit.
            x0: Initial inclination parameter.
            distance: Distance to the source in Gpc.
            qK: Polarization angle in radians.
            phiK: Azimuthal angle in radians.
            Phi_phi0: Initial phase of the azimuthal motion in radians.
            Phi_theta0: Initial phase of the polar motion in radians.
            Phi_r0: Initial phase of the radial motion in radians.
            ra: Right ascension of the source in radians.
            dec: Declination of the source in radians.
            merger_time: Time of merger in seconds.
            **kwargs: Additional keyword arguments forwarded to :class:`few.waveform.waveform.GenerateEMRIWaveform`.

        Returns:
            tuple[NDArrayLike, NDArrayLike, NDArrayLike]: The time array and the generated waveform polarizations.

        """

        times = []
        hplus = []
        hcross = []

        for i in range(len(m1)):
            t, hplus_i, hcross_i = self.wave_gen(
                m1[i], m2[i], a[i], p0[i], e0[i], x0[i], distance[i],
                qK[i], phiK[i], Phi_phi0[i], Phi_theta0[i], Phi_r0[i],
                ra[i], dec[i], merger_time, **kwargs
            )
            times.append(t)
            hplus.append(hplus_i)
            hcross.append(hcross_i)
        
        return self.xp.array(times), self.xp.array(hplus), self.xp.array(hcross)

    def compute_tdi_channels(
        self,
        *args,
        ra: float | NDArrayLike = None,
        dec: float | NDArrayLike = None,
        merger_time: float | NDArrayLike = None,
        **kwargs,
    ) -> Tuple[NDArrayLike, NDArrayLike]:
        """Time domain TDI channels computation. In the case of multiple sources, the TDI response is applied sequentially to each source and the results are stacked together.

        Args:
            *args: Arguments for the wave_gen / wave_gen_batch method.
            ra: Right ascension in radians.  Scalar for single source, 1-D array for batch.
            dec: Declination in radians.  Same shape as ``ra``.
            merger_time: Time of merger in seconds.  Same shape as ``ra``. Ignored for the moment in the emri case, the merger time is set to cover the emtire duration from t_ref to the end of the data time settings.
            **kwargs: Keyword arguments for the wave_gen / wave_gen_batch method.

        Returns:
            Tuple of (times, channels) where `times` is the time array after shifting and padding, and `channels` are the TDI variables with shape (num_channels, num_times) for single source or (num_bin, num_channels, num_times) for batch.
        """
        if ra is None or dec is None:
            *args, ra, dec = args
        
        merger_time = float(self.domain_settings.t_end - self.waveform_t0)

        if np.ndim(ra) >= 1:
            return self._call_batched(*args, ra=ra, dec=dec, merger_time=merger_time, **kwargs)
        return self._call_single(*args, ra=ra, dec=dec, merger_time=merger_time, **kwargs)

    def _apply_response(
        self,
        t_arr: NDArrayLike,
        h_plus: NDArrayLike,
        h_cross: NDArrayLike,
        ra: float | NDArrayLike,
        dec: float | NDArrayLike,
        merger_time: float | NDArrayLike,
    ) -> Tuple[NDArrayLike, NDArrayLike]:
        """Apply the TDI response to a batch of sources.

        Args:
            t_arr: Time array relative to zero, shape (Ntimes,) or (Nbatch, Ntimes).
            h_plus: Plus polarization, shape (Ntimes,) or (Nbatch, Ntimes).
            h_cross: Cross polarization, shape (Ntimes,) or (Nbatch, Ntimes).
            ra: Right ascension in radians, float or shape (Nbatch,).
            dec: Declination in radians, float or shape (Nbatch,).
            merger_time: Time of merger in seconds (relative to waveform_t0), float or shape (Nbatch,). Ignored for emris, will be set to 0.0. 

        Returns:
            Tuple of (times_batch, channels_batch) where times_batch is the time array after shifting and padding with shape (Nbatch, Ntimes), and channels_batch is the TDI response with shape (Nbatch, num_channels, num_times).
        """

        merger_time = 0.0 if isinstance(merger_time, float) else get_array_module(merger_time).zeros_like(merger_time)

        return super()._apply_response(t_arr, h_plus, h_cross, ra, dec, merger_time)
