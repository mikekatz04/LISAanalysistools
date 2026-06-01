from __future__ import annotations

from typing import Any, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
from eryn.utils import TransformContainer

from lisatools.diagnostic import covariance, plot_covariance_contour, plot_covariance_corner

from ..detector import LISAModel
from ..diagnostic import snr as snr_func
from ..sensitivity import A1TDISens, Sensitivity
from ..utils.constants import *
from ..utils.utility import get_array_module
from .waveformbase import AETTDIWaveform, SNRWaveform

if TYPE_CHECKING:
    try:
        import cupy as cp
    except (ImportError, ModuleNotFoundError):
        import numpy as cp

class CalculationController:
    """Wrapper class to controll investigative computations.

    Args:
        aet_template_gen: Template waveform generator.
        model: Model for LISA.
        psd_kwargs: psd_kwargs for :func:`lisatools.sensitivity.get_sensitivity`.
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
        """Set parameter transforms."""
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
    """
    q <= 1.0
    """
    return (mT / (1 + q), (q * mT) / (1 + q))


def dist_convert(x: float) -> float:
    return x * 1e9 * PC_SI


def time_convert(x: float) -> float:
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


def return_input(x: Any) -> Any:
    """Return the input."""
    return x


def return_float(x: np.ndarray | float) -> float:
    """Return the input as a float."""
    return float(x)


def icrs_to_ecliptic(
    ra: float | np.ndarray | cp.ndarray, dec: float | np.ndarray | cp.ndarray, psi: Optional[float | np.ndarray | cp.ndarray] = None
) -> (
    Tuple[float | np.ndarray | cp.ndarray, float | np.ndarray | cp.ndarray]
    | Tuple[float | np.ndarray | cp.ndarray, float | np.ndarray | cp.ndarray, float | np.ndarray | cp.ndarray]
):
    """
    Convert ICRS angles (ra, dec, optional psi) to ecliptic coordinates
    (lambda, beta, optional psi_ecliptic).

    The sky angles are converted according to the convention document LISA-DDPC-SEG-TN-007, sec 5.4.2:
    .. math::

        \\sin(\\beta) = \\sin(\\delta) \\cos(\\epsilon) - \\cos(\\delta) \\sin(\\epsilon) \\sin(\\alpha)
        \\cos(\\lambda) = \\cos(\\delta) \\cos(\\alpha) / \\cos(\\beta)
        \\sin(\\lambda) = (\\sin(\\delta) \\sin(\\epsilon) + \\cos(\\delta) \\cos(\\epsilon) \\sin(\\alpha)) / \\cos(\\beta).

    The polarization angle is converted according to:
    ..math::
        \\psi_{\rm ecliptic} = \\psi - \\delta \\psi
        \\cos(\\delta \\psi) = \\frac{1}{\\cos(\\beta)} (\\sin(\\epsilon) \\sin(\\delta) \\sin(\\alpha) + \\cos(\\epsilon) \\cos(\\delta))
        \\sin(\\delta \\psi) = - \\frac{1}{\\cos(\\beta)} \\sin(\\epsilon) \\cos(\\alpha)

    Inputs are broadcast following NumPy rules. If all inputs are scalar,
    scalar outputs are returned.

    Args:
        ra: Right ascension in radians.
        dec: Declination in radians.
        psi: Optional polarization angle in radians. If provided, the returned tuple will include the converted polarization angle in ecliptic coordinates.

    Returns:
        If `psi` is not provided, returns a tuple of (lambda, beta) in radians.
        If `psi` is provided, returns a tuple of (lambda, beta, psi_ecliptic) in radians.
    """
    scalar_output = (
        isinstance(ra, float) and isinstance(dec, float) and (psi is None or isinstance(psi, float))
    )
    out_fun = return_float if scalar_output else return_input

    xp = np if scalar_output else get_array_module(ra)

    ra = xp.asarray(ra, dtype=float)
    dec = xp.asarray(dec, dtype=float)

    cos_dec = xp.cos(dec)
    sin_dec = xp.sin(dec)
    cos_ra = xp.cos(ra)
    sin_ra = xp.sin(ra)
    cos_eps = xp.cos(EPS_RAD)
    sin_eps = xp.sin(EPS_RAD)

    sin_beta = sin_dec * cos_eps - cos_dec * sin_eps * sin_ra
    beta = xp.arcsin(sin_beta)

    cos_beta = xp.cos(beta)
    eps_cos_beta = xp.finfo(float).eps
    safe_cos_beta = xp.where(
        xp.abs(cos_beta) < eps_cos_beta,
        xp.copysign(eps_cos_beta, cos_beta),
        cos_beta,
    )
    inv_cos_beta = 1.0 / safe_cos_beta

    cos_lambda = cos_dec * cos_ra * inv_cos_beta
    sin_lambda = (sin_dec * sin_eps + cos_dec * cos_eps * sin_ra) * inv_cos_beta
    lambd = xp.arctan2(sin_lambda, cos_lambda) % (2 * xp.pi)

    if psi is not None:
        psi = xp.asarray(psi, dtype=float)

        cosdeltapsi = inv_cos_beta * (sin_eps * sin_dec * sin_ra + cos_eps * cos_dec)
        sindeltapsi = -inv_cos_beta * sin_eps * cos_ra

        deltapsi = xp.arctan2(sindeltapsi, cosdeltapsi)
        psi_ecliptic = (psi - deltapsi) % xp.pi

        return out_fun(lambd), out_fun(beta), out_fun(psi_ecliptic)

    return out_fun(lambd), out_fun(beta)


def ecliptic_to_icrs(
    lambd: float | np.ndarray | cp.ndarray,
    beta: float | np.ndarray | cp.ndarray,
    psi_ecliptic: Optional[float | np.ndarray | cp.ndarray] = None,
) -> (
    Tuple[float | np.ndarray |cp.ndarray, float | np.ndarray |cp.ndarray]
    | Tuple[float | np.ndarray |cp.ndarray, float | np.ndarray |cp.ndarray, float | np.ndarray |cp.ndarray]
):
    """
    Convert ecliptic coordinates (lambda, beta, optional psi_ecliptic) to ICRS angles
    (ra, dec, optional psi).

    The sky angles are converted according to the convention document LISA-DDPC-SEG-TN-007, sec 5.4.2:
    ..math::

        \\sin(\\delta) = \\sin(\\beta) \\cos(\\epsilon) + \\cos(\\beta) \\sin(\\epsilon) \\sin(\\lambda)
        \\cos(\\alpha) = \\cos(\\beta) \\cos(\\lambda) / \\cos(\\delta)
        \\sin(\\alpha) = (-\\sin(\\beta) \\sin(\\epsilon) + \\cos(\\beta) \\cos(\\epsilon) \\sin(\\lambda)) / \\cos(\\delta).

    The polarization angle is converted according to:
    ..math::
        \\psi = \\psi_{\\rm ecliptic} + \\delta \\psi
        \\cos(\\delta \\psi) = \\frac{1}{\\cos(\\beta)} (\\sin(\\epsilon) \\sin(\\delta) \\sin(\\alpha) + \\cos(\\epsilon) \\cos(\\delta))
        \\sin(\\delta \\psi) = - \\frac{1}{\\cos(\\beta)} \\sin(\\epsilon) \\cos(\\alpha)

    Inputs are broadcast following NumPy rules. If all inputs are scalar,
    scalar outputs are returned.

    Args:
        lambd: Ecliptic longitude in radians.
        beta: Ecliptic latitude in radians.
        psi_ecliptic: Optional polarization angle in radians. If provided, the returned tuple will include the converted polarization angle in ICRS coordinates.

    Returns:
        If `psi_ecliptic` is not provided, returns a tuple of (ra, dec) in radians.
        If `psi_ecliptic` is provided, returns a tuple of (ra, dec, psi) in radians.
    """
    scalar_output = (
        isinstance(lambd, float)
        and isinstance(beta, float)
        and (psi_ecliptic is None or isinstance(psi_ecliptic, float))
    )
    out_fun = return_float if scalar_output else return_input
    xp = np if scalar_output else get_array_module(lambd)

    lambd = xp.asarray(lambd, dtype=float)
    beta = xp.asarray(beta, dtype=float)

    cos_beta = xp.cos(beta)
    sin_beta = xp.sin(beta)
    cos_lambda = xp.cos(lambd)
    sin_lambda = xp.sin(lambd)
    cos_eps = xp.cos(EPS_RAD)
    sin_eps = xp.sin(EPS_RAD)

    sin_dec = sin_beta * cos_eps + cos_beta * sin_eps * sin_lambda
    dec = xp.arcsin(sin_dec)
    cos_dec = xp.cos(dec)
    eps_float = xp.finfo(float).eps
    safe_cos_dec = xp.where(
        xp.abs(cos_dec) < eps_float,
        xp.copysign(eps_float, cos_dec),
        cos_dec,
    )
    inv_cos_dec = 1.0 / safe_cos_dec

    cos_ra = cos_beta * cos_lambda * inv_cos_dec
    sin_ra = (-sin_beta * sin_eps + cos_beta * cos_eps * sin_lambda) * inv_cos_dec
    ra = xp.arctan2(sin_ra, cos_ra) % (2 * xp.pi)

    if psi_ecliptic is not None:
        psi_ecliptic = xp.asarray(psi_ecliptic, dtype=float)

        cos_beta = xp.cos(beta)
        safe_cos_beta = xp.where(
            xp.abs(cos_beta) < eps_float,
            xp.copysign(eps_float, cos_beta),
            cos_beta,
        )
        inv_cos_beta = 1.0 / safe_cos_beta

        cosdeltapsi = inv_cos_beta * (sin_eps * sin_dec * sin_ra + cos_eps * cos_dec)
        sindeltapsi = -inv_cos_beta * sin_eps * cos_ra

        deltapsi = xp.arctan2(sindeltapsi, cosdeltapsi)
        psi = (psi_ecliptic + deltapsi) % xp.pi

        return out_fun(ra), out_fun(dec), out_fun(psi)

    return out_fun(ra), out_fun(dec)


def evolve_galactic_binary(
    t_start: float, 
    t_end: float, 
    f_start: float | np.ndarray, 
    phi_start: float | np.ndarray, 
    fdot_start: float | np.ndarray, 
    fddot: float | np.ndarray = 0.0,
    phase_sign: int = 1,
    Mc: Optional[float | np.ndarray] = None
) -> (
    Tuple[float | np.ndarray, float | np.ndarray, float | np.ndarray]
):
    """Evolves one or more galactic binaries from its initial starting parameters to some set of final parameters for a given starting and end time.
    If Mc is provided, the binary is evolved using gravitationally driven quadrupolar radiation. Otherwise, a linear approximation is used.

    Args:
        t_start (float): Start/initial time of the binaries before evolving.
        t_end (float): End/final time of the binaries after evolving.
        f_start (float | np.ndarray): Start/initial frequency of the binaries before evolving.
        phi_start (float | np.ndarray): Start/initial phase of the binaries before evolving.
        fdot_start (float | np.ndarray): Start/initial frequency derivative of the binary before evolving. Only evolves when chirp mass Mc is provided or fddot is not zero.
        fddot (float | np.ndarray): Second order frequency derivative of the binary. Standard value is 0.0
        phase_sign (int): Sign of the phase evolution.  JaxGB assumes -1, while the GBGPU implementation assumes +1. This enters as: :math:`\\phi(t) = \\text{sign} \\phi_0 + ...`
        Mc (float | np.ndarray | None): The chrip mass of the binary to evolve the binary using gravitationally driven quadrupolar radiation. Standard is None.
        
    Returns:
        Tuple of (f_end, phi_end, fdot_end).
        (Tuple[float | np.ndarray, float | np.ndarray, float | np.ndarray])
    """
    assert phase_sign in [1, -1], "phase_sign must be either +1 or -1"
    
    if Mc is None:
       delta_t = t_end - t_start
       fdot_end = fdot_start + fddot * delta_t
       f_end = f_start + fdot_start * delta_t + 1/2 * fddot * delta_t ** 2
       phi_end = phi_start + phase_sign * 2 * np.pi * (f_start * delta_t + 1/2 * fdot_start * delta_t ** 2 + 1/6 * fddot * delta_t ** 3) 
       phi_end = phi_end % (2 * np.pi)
       return f_end, phi_end, fdot_end
       
    else:
        raise NotImplementedError("Currently, the galactic binaries can only be evolved using the linear approximation")