"""Calculation-controller helpers for SNR and information-matrix computations across source classes."""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from eryn.utils import TransformContainer

from ..diagnostic import covariance, plot_covariance_contour, plot_covariance_corner

from ..detector import LISAModel
from ..diagnostic import snr as snr_func
from ..sensitivity import A1TDISens, Sensitivity
from ..utils.constants import *
from .waveformbase import AETTDIWaveform, SNRWaveform


class CalculationController:
    """Wrapper class to control investigative computations.

    Provides a common interface for computing SNRs (and, in subclasses, information /
    covariance matrices) for a particular source class given a template waveform
    generator and a LISA noise model.

    Args:
        aet_template_gen: Template waveform generator that returns AET TDI channels.
        model: LISA noise model passed to the sensitivity function.
        psd_kwargs: Keyword arguments forwarded to :func:`lisatools.sensitivity.get_sensitivity`.
        Tobs: Observation time in **years**.
        dt: Timestep in seconds.
        psd: Sensitivity curve type to use. Default is :class:`A1TDISens`
            because we ignore ``T`` in these simplified calculations and
            the ``A`` and ``E`` sensitivities are equivalent.

    """

    def __init__(
        self,
        aet_template_gen: SNRWaveform | AETTDIWaveform,
        model: LISAModel,
        psd_kwargs: dict,
        Tobs: float,
        dt: float,
        psd: Sensitivity = A1TDISens,
    ) -> None:

        # Store everything.
        self.aet_template_gen = aet_template_gen
        self.psd_kwargs = psd_kwargs
        self.model = model
        self.psd = psd
        self.Tobs = Tobs
        self.dt = dt

    @property
    def parameter_transforms(self) -> TransformContainer:
        """Transform parameters from sampling basis to waveform basis."""
        return self._parameter_transforms

    @parameter_transforms.setter
    def parameter_transforms(self, parameter_transforms: TransformContainer) -> None:
        """Set the parameter transform container."""
        assert isinstance(parameter_transforms, TransformContainer)
        self._parameter_transforms = parameter_transforms

    def get_snr(self, *params: Any, **kwargs: Any) -> float:
        """Compute the SNR.

        Args:
            *params: Parameters to go into waveform generator.
            **kwargs: Kwargs for waveform generator.

        Returns:
            SNR.

        """
        # generate waveform
        a_chan, e_chan, t_chan = self.aet_template_gen(*params, **kwargs)

        # ignore t channel for snr computation
        # compute SNR
        opt_snr = snr_func(
            [a_chan, e_chan],
            psd=self.psd,
            psd_kwargs={**self.psd_kwargs, "model": self.model},
            dt=self.aet_template_gen.dt,
            f_arr=self.aet_template_gen.f_arr,
            df=self.aet_template_gen.df,
        )

        # prepare outputs
        self.f_arr = self.aet_template_gen.f_arr
        self.last_output = (a_chan, e_chan)

        return opt_snr


def mT_q_to_m1_m2(mT: float, q: float) -> Tuple[float, float]:
    """Convert total mass and mass ratio to component masses.

    Args:
        mT: Total mass :math:`M = m_1 + m_2`.
        q: Mass ratio :math:`q = m_2 / m_1 \\le 1`.

    Returns:
        ``(m1, m2)`` component masses in the same units as ``mT``.
    """
    return (mT / (1 + q), (q * mT) / (1 + q))


def dist_convert(x: float) -> float:
    """Convert a distance from Gpc to meters."""
    return x * 1e9 * PC_SI


def time_convert(x: float) -> float:
    """Convert a time from years (sidereal) to seconds."""
    return x * YRSID_SI


class BBHCalculationController(CalculationController):
    """Calculation controller for BBHs.

    Args:
        *args: Args for :class:`CalculationController`.
        *kwargs: Kwargs for :class:`CalculationController`.

    """

    def __init__(self, *args: Any, **kwargs: Any):

        # transforms from information matrix basis
        parameter_transforms = {
            0: np.exp,
            4: dist_convert,
            7: np.arccos,
            9: np.arcsin,
            11: time_convert,
            (0, 1): mT_q_to_m1_m2,
        }

        input_basis = list(range(12))
        output_basis = list(range(12))
        self.transform_fn = TransformContainer(
            input_basis,
            output_basis,
            parameter_transforms=parameter_transforms,
            fill_dict=None,  # fill_dict
        )

        super(BBHCalculationController, self).__init__(*args, **kwargs)

    def get_snr(self, *args: Any, **kwargs: Any) -> float:
        """Compute the SNR.

        Args:
            *params: Parameters to go into waveform generator.
            **kwargs: Kwargs for waveform generator.

        Returns:
            SNR.

        """
        # adjust kwargs to simplify calculation
        if "t_obs_start" not in kwargs:
            kwargs["shift_t_limits"] = True
            kwargs["t_obs_start"] = 0.0
            kwargs["t_obs_end"] = self.Tobs
        # compute snr
        return super(BBHCalculationController, self).get_snr(*args, **kwargs)

    def get_cov(
        self,
        *params: Any,
        more_accurate: bool = False,
        eps: float = 1e-9,
        deriv_inds: np.ndarray = None,
        precision: bool = False,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get covariance matrix.

        Args:
            *params: Parameters for BBH. Must include ``f_ref``.
            more_accurate: If ``True``, run a more accurate derivate requiring 2x more waveform generations.
            eps: Absolute **derivative** step size. See :func:`lisatools.diagnostic.info_matrix`.
            deriv_inds: Subset of parameters of interest for which to calculate the information matrix, by index.
                If ``None``, it will be ``np.arange(len(params))``.
            precision: If ``True``, uses 500-dps precision to compute the information matrix inverse (requires `mpmath <https://mpmath.org>`_).
                This is typically a good idea as the information matrix can be highly ill-conditioned.
            **kwargs: Kwargs for waveform generation.

        Returns:
            Parameters and covariance matrix.

        """

        # setup all bbh specific quantities.
        assert len(params) == 12

        if isinstance(params, tuple):
            params = list(params)

        params = np.asarray(params)

        m1 = params[0]
        m2 = params[1]
        mT = m1 + m2

        if m2 > m1:
            tmp = m2
            m2 = m1
            m1 = tmp

        q = m2 / m1

        params[0] = mT
        params[1] = q

        params[0] = np.log(params[0])
        params[4] = params[4] / 1e9 / PC_SI
        params[7] = np.cos(params[7])
        params[9] = np.sin(params[9])
        params[11] = params[11] / YRSID_SI

        # default deriv inds
        if deriv_inds is None:
            deriv_inds = np.delete(np.arange(12), 6)

        # remove f_ref derivative
        if 6 in deriv_inds:
            deriv_inds = np.delete(deriv_inds, np.where(deriv_inds == 6)[0])

        kwargs["return_array"] = True

        if "t_obs_start" not in kwargs:
            kwargs["shift_t_limits"] = True
            kwargs["t_obs_start"] = 0.0
            kwargs["t_obs_end"] = self.Tobs

        # compute covariance
        cov = covariance(
            eps,
            self.aet_template_gen,
            params,
            parameter_transforms=self.transform_fn,
            inner_product_kwargs=dict(
                psd=self.psd,
                psd_kwargs={**self.psd_kwargs, "model": self.model},
                dt=self.aet_template_gen.dt,
                f_arr=self.aet_template_gen.f_arr,
                df=self.aet_template_gen.df,
            ),
            waveform_kwargs=kwargs,
            more_accurate=more_accurate,
            deriv_inds=deriv_inds,
            precision=precision,
        )

        # return parameters and their covariance
        return params[deriv_inds], cov


class GBCalculationController(CalculationController):
    """Calculation controller for GBs.

    Args:
        *args: Args for :class:`CalculationController`.
        *kwargs: Kwargs for :class:`CalculationController`.

    """

    def __init__(self, *args: Any, **kwargs: Any):

        # parameter transforms from sampling basis to waveform basis
        parameter_transforms = {
            0: lambda x: x * 1e-23,
            1: lambda x: x / 1e3,
            2: lambda x: x * 1e-18,
            5: np.arccos,
            8: np.arcsin,
            # (1, 2, 3): lambda x, y, z: (x, y, 11.0 / 3.0 * y**2 / x),
        }

        input_basis = list(range(9))
        output_basis = list(range(9))
        self.transform_fn = TransformContainer(
            input_basis,
            output_basis,
            parameter_transforms=parameter_transforms,
            fill_dict=None,  # fill_dict
        )

        super(GBCalculationController, self).__init__(*args, **kwargs)
        # convert back to seconds
        self.Tobs *= YRSID_SI

    def get_cov(
        self,
        *params: np.ndarray | list,
        more_accurate: bool = False,
        eps: float = 1e-9,
        deriv_inds: np.ndarray = None,
        precision: bool = False,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get covariance matrix.

        Args:
            *params: Parameters for GB. Must include ``fddot``.
            more_accurate: If ``True``, run a more accurate derivate requiring 2x more waveform generations.
            eps: Absolute **derivative** step size. See :func:`lisatools.diagnostic.info_matrix`.
            deriv_inds: Subset of parameters of interest for which to calculate the information matrix, by index.
                If ``None``, it will be ``np.arange(len(params))``.
            precision: If ``True``, uses 500-dps precision to compute the information matrix inverse (requires `mpmath <https://mpmath.org>`_).
                This is typically a good idea as the information matrix can be highly ill-conditioned.
            **kwargs: Kwargs for waveform generation.

        Returns:
            Parameters and covariance matrix.

        """
        assert len(params) == 9

        if isinstance(params, tuple):
            params = list(params)

        params = np.asarray(params)

        # params[0] = np.log(params[0])
        params[0] = params[0] / 1e-23
        params[1] = params[1] * 1e3
        params[2] = params[2] / 1e-18

        if params[3] != 0.0:
            raise NotImplementedError("This class has not been implemented for fddot != 0 yet.")

        params[5] = np.cos(params[5])
        params[8] = np.sin(params[8])

        if deriv_inds is None:
            deriv_inds = np.delete(np.arange(9), 3)

        # remove fddot for now
        if 3 in deriv_inds:
            deriv_inds = np.delete(deriv_inds, np.where(deriv_inds == 3)[0])

        kwargs["return_array"] = True

        kwargs["dt"] = self.dt
        kwargs["T"] = self.Tobs

        cov = covariance(
            eps,
            self.aet_template_gen,
            params,
            parameter_transforms=self.transform_fn,
            inner_product_kwargs=dict(
                psd=self.psd,
                psd_kwargs={**self.psd_kwargs, "model": self.model},
                dt=self.aet_template_gen.dt,
                f_arr=self.aet_template_gen.f_arr,
                df=self.aet_template_gen.df,
            ),
            waveform_kwargs=kwargs,
            more_accurate=more_accurate,
            deriv_inds=deriv_inds,
            precision=precision,
        )

        return params[deriv_inds], cov

    def get_snr(self, *args: Any, **kwargs: Any) -> float:
        """Compute the SNR.

        Args:
            *params: Parameters to go into waveform generator.
            **kwargs: Kwargs for waveform generator.

        Returns:
            SNR.

        """
        # make sure it is TDI 2
        if "tdi2" not in kwargs:
            kwargs["tdi2"] = True

        kwargs["dt"] = self.dt
        kwargs["T"] = self.Tobs

        # ensures tdi2 is added correctly for GBGPU
        return super(GBCalculationController, self).get_snr(*args, **kwargs)


class EMRICalculationController(CalculationController):
    """Calculation controller for EMRIs.

    Args:
        *args: Args for :class:`CalculationController`.
        *kwargs: Kwargs for :class:`CalculationController`.

    """

    def __init__(self, *args: Any, **kwargs: Any):

        # parameter transforms for EMRIs
        parameter_transforms = {
            0: np.exp,
            5: np.arccos,
            7: np.arccos,
            9: np.arccos,
            # (1, 2, 3): lambda x, y, z: (x, y, 11.0 / 3.0 * y**2 / x),
        }

        input_basis = list(range(14))
        output_basis = list(range(14))
        self.transform_fn = TransformContainer(
            input_basis,
            output_basis,
            parameter_transforms=parameter_transforms,
            fill_dict=None,  # fill_dict
        )

        super(EMRICalculationController, self).__init__(*args, **kwargs)

    def get_cov(
        self,
        *params: np.ndarray | list,
        more_accurate: bool = False,
        eps: float = 1e-9,
        deriv_inds: np.ndarray = None,
        precision: bool = False,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get covariance matrix.

        Args:
            *params: Parameters for EMRIs.
            more_accurate: If ``True``, run a more accurate derivate requiring 2x more waveform generations.
            eps: Absolute **derivative** step size. See :func:`lisatools.diagnostic.info_matrix`.
            deriv_inds: Subset of parameters of interest for which to calculate the information matrix, by index.
                If ``None``, it will be ``np.arange(len(params))``.
            precision: If ``True``, uses 500-dps precision to compute the information matrix inverse (requires `mpmath <https://mpmath.org>`_).
                This is typically a good idea as the information matrix can be highly ill-conditioned.
            **kwargs: Kwargs for waveform generation.

        Returns:
            Parameters and covariance matrix.

        """
        assert len(params) == 14

        if isinstance(params, tuple):
            params = list(params)

        params = np.asarray(params)

        params[0] = np.log(params[0])
        params[5] = np.cos(params[5])
        params[7] = np.cos(params[7])
        params[9] = np.cos(params[9])

        kwargs["return_array"] = True

        assert self.aet_template_gen.response.dt == self.dt
        assert self.aet_template_gen.response.T == self.Tobs

        cov = covariance(
            eps,
            self.aet_template_gen,
            params,
            parameter_transforms=self.transform_fn,
            inner_product_kwargs=dict(
                psd=self.psd,
                psd_kwargs={**self.psd_kwargs, "model": self.model},
                dt=self.aet_template_gen.dt,
                f_arr=self.aet_template_gen.f_arr,
                df=self.aet_template_gen.df,
            ),
            waveform_kwargs=kwargs,
            more_accurate=more_accurate,
            deriv_inds=deriv_inds,
            precision=precision,
        )

        return params[deriv_inds], cov


# Equatorial (ICRS) pole expressed in ecliptic cartesian coordinates,
# computed lazily through the same astropy frame used for the position
# conversions so the polarization rotation is exactly consistent with them.
_EQ_POLE_ECL_XYZ = None


def _equatorial_pole_ecliptic_xyz() -> np.ndarray:
    """Unit vector of the ICRS pole in (barycentric true) ecliptic cartesian coords."""
    global _EQ_POLE_ECL_XYZ
    if _EQ_POLE_ECL_XYZ is None:
        pole = SkyCoord(
            ra=0.0 * u.rad, dec=(np.pi / 2.0) * u.rad, frame="icrs"
        ).barycentrictrueecliptic
        lam_p = pole.lon.rad
        beta_p = pole.lat.rad
        _EQ_POLE_ECL_XYZ = np.array(
            [
                np.cos(beta_p) * np.cos(lam_p),
                np.cos(beta_p) * np.sin(lam_p),
                np.sin(beta_p),
            ]
        )
    return _EQ_POLE_ECL_XYZ


def psi_rotation_icrs_to_ecliptic(lam_ecl, beta_ecl):
    """Polarization-basis rotation ``chi`` such that ``psi_ecl = psi_icrs - chi``.

    Both polarization angles are assumed measured from the respective
    frame's local north (direction of increasing latitude) toward local
    east, with the same rotational sense about the line of sight, so only
    the relative rotation of the two local-north directions matters. The
    rotation is position-dependent only and is computed in ecliptic
    cartesian coordinates with the standard vector construction (local
    north = frame pole projected onto the sky plane), using the same
    astropy frames as the position conversion.

    Args:
        lam_ecl: Ecliptic longitude(s) in radians.
        beta_ecl: Ecliptic latitude(s) in radians.

    Returns:
        ``chi`` in radians with the same shape as the inputs.
    """
    lam = np.asarray(lam_ecl)
    beta = np.asarray(beta_ecl)
    n = np.stack(
        [np.cos(beta) * np.cos(lam), np.cos(beta) * np.sin(lam), np.sin(beta)],
        axis=-1,
    )  # line of sight (unit)
    z_ecl = np.array([0.0, 0.0, 1.0])
    p_eq = _equatorial_pole_ecliptic_xyz()

    # Local-north directions (pole minus its line-of-sight component).
    # Normalization is unnecessary: |cross(north_icrs, n)| == |north_icrs|
    # since north_icrs is perpendicular to n, so the common positive factor
    # |north_icrs| * |north_ecl| cancels inside arctan2.
    north_ecl = z_ecl - (n @ z_ecl)[..., None] * n
    north_icrs = p_eq - (n @ p_eq)[..., None] * n
    east_icrs = np.cross(north_icrs, n)
    chi = np.arctan2(
        np.sum(east_icrs * north_ecl, axis=-1),
        np.sum(north_icrs * north_ecl, axis=-1),
    )
    return chi


def icrs_to_ecliptic(psi, ra, dec) -> Tuple[float, float, float]:
    """Convert ICRS sky coordinates and polarisation angle to ecliptic coordinates.

    Vectorized: scalar or array inputs are both supported.

    Args:
        psi: Polarisation angle in radians, defined in the ICRS frame.
        ra: Right ascension in radians.
        dec: Declination in radians.

    Returns:
        Tuple ``(psi_ecliptic, lambda, beta)`` of polarisation angle, ecliptic
        longitude, and ecliptic latitude (all in radians).
    """
    coord = SkyCoord(ra=ra * u.rad, dec=dec * u.rad, frame='icrs')
    ecliptic_coord = coord.barycentrictrueecliptic

    lambd = ecliptic_coord.lon.rad
    beta = ecliptic_coord.lat.rad

    psi_ecliptic = psi - psi_rotation_icrs_to_ecliptic(lambd, beta)

    return psi_ecliptic, lambd, beta


def ecliptic_to_icrs(psi_ecl, lam, beta) -> Tuple[float, float, float]:
    """Convert ecliptic sky coordinates and polarisation angle to ICRS (inverse of :func:`icrs_to_ecliptic`).

    Vectorized: scalar or array inputs are both supported.

    Args:
        psi_ecl: Polarisation angle in radians, defined in the ecliptic frame.
        lam: Ecliptic longitude in radians.
        beta: Ecliptic latitude in radians.

    Returns:
        Tuple ``(psi_icrs, ra, dec)`` of polarisation angle, right
        ascension, and declination (all in radians).
    """
    coord = SkyCoord(
        lon=lam * u.rad, lat=beta * u.rad, frame="barycentrictrueecliptic"
    )
    icrs_coord = coord.icrs

    psi_icrs = psi_ecl + psi_rotation_icrs_to_ecliptic(lam, beta)

    return psi_icrs, icrs_coord.ra.rad, icrs_coord.dec.rad
