from __future__ import annotations

import numpy as np
from typing import Any, Tuple, List, Optional

from ..diagnostic import snr as snr_func
from lisatools.diagnostic import (
    covariance,
    plot_covariance_corner,
    plot_covariance_contour,
)
from ..sensitivity import A1TDISens, Sensitivity
from .waveformbase import SNRWaveform, AETTDIWaveform
from ..detector import LISAModel
from ..utils.constants import *
from eryn.utils import TransformContainer


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
        self.transform_fn = TransformContainer(
            parameter_transforms=parameter_transforms, fill_dict=None  # fill_dict
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
        **kwargs: Any
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
        self.transform_fn = TransformContainer(
            parameter_transforms=parameter_transforms, fill_dict=None  # fill_dict
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
        **kwargs: Any
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
            raise NotImplementedError(
                "This class has not been implemented for fddot != 0 yet."
            )

        params[5] = np.cos(params[5])
        params[8] = np.cos(params[5])

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
        self.transform_fn = TransformContainer(
            parameter_transforms=parameter_transforms, fill_dict=None  # fill_dict
        )

        super(EMRICalculationController, self).__init__(*args, **kwargs)

    def get_cov(
        self,
        *params: np.ndarray | list,
        more_accurate: bool = False,
        eps: float = 1e-9,
        deriv_inds: np.ndarray = None,
        precision: bool = False,
        **kwargs: Any
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
