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
    def __init__(
        self,
        aet_template_gen: SNRWaveform | AETTDIWaveform,
        model: LISAModel,
        psd_kwargs: dict,
        psd: Sensitivity = A1TDISens,
    ) -> None:
        self.aet_template_gen = aet_template_gen
        self.psd_kwargs = psd_kwargs
        self.model = model
        self.psd = psd

    @property
    def parameter_transforms(self) -> TransformContainer:
        return self._parameter_transforms

    @parameter_transforms.setter
    def parameter_transforms(self, parameter_transforms: TransformContainer) -> None:
        assert isinstance(parameter_transforms, TransformContainer)
        self._parameter_transforms = parameter_transforms

    def get_snr(self, *params: np.ndarray | list, **kwargs: Any) -> float:
        a_chan, e_chan, t_chan = self.aet_template_gen(*params, **kwargs)

        # ignore t channel for snr computation
        opt_snr = snr_func(
            [a_chan, e_chan],
            psd=self.psd,
            psd_kwargs={**self.psd_kwargs, "model": self.model},
            dt=self.aet_template_gen.dt,
            f_arr=self.aet_template_gen.f_arr,
            df=self.aet_template_gen.df,
        )

        self.f_arr = self.aet_template_gen.f_arr
        self.last_output = (a_chan, e_chan)

        return opt_snr


class BBHCalculatorController(CalculationController):
    def __init__(self, *args: Any, **kwargs: Any):
        # fill_dict = {
        #     "ndim_full": 12,
        #     "fill_values": np.array([0.0]),
        #     "fill_inds": np.array([6]),
        # }
        parameter_transforms = {
            0: np.exp,
            4: lambda x: x * 1e9 * PC_SI,
            7: np.arccos,
            9: np.arcsin,
            11: lambda x: x * YRSID_SI,
        }
        self.transform_fn = TransformContainer(
            parameter_transforms=parameter_transforms, fill_dict=None  # fill_dict
        )

        super(BBHCalculatorController, self).__init__(*args, **kwargs)

    def get_cov(
        self,
        *params: np.ndarray | list,
        precision: bool = True,
        more_accurate: bool = False,
        eps: float = 1e-9,
        deriv_inds: np.ndarray = None,
        **kwargs: Any
    ) -> Tuple[np.ndarray, np.ndarray]:

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

        if deriv_inds is None:
            deriv_inds = np.delete(np.arange(12), 6)

        if 6 in deriv_inds:
            deriv_inds = np.delete(deriv_inds, np.where(deriv_inds == 6)[0])

        kwargs["return_array"] = True

        # ignore t channel for snr computation
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
        )

        return params[np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11])], cov


class GBCalculatorController(CalculationController):
    def __init__(self, *args: Any, **kwargs: Any):
        # fill_dict = {
        #     "ndim_full": 12,
        #     "fill_values": np.array([0.0]),
        #     "fill_inds": np.array([6]),
        # }
        parameter_transforms = {
            0: np.exp,
            1: lambda x: x / 1e3,
            2: np.exp,
            5: np.arccos,
            8: np.arcsin,
            # (1, 2, 3): lambda x, y, z: (x, y, 11.0 / 3.0 * y**2 / x),
        }
        self.transform_fn = TransformContainer(
            parameter_transforms=parameter_transforms, fill_dict=None  # fill_dict
        )

        super(GBCalculatorController, self).__init__(*args, **kwargs)

    def get_cov(
        self,
        *params: np.ndarray | list,
        precision: bool = True,
        more_accurate: bool = False,
        eps: float = 1e-9,
        deriv_inds: np.ndarray = None,
        **kwargs: Any
    ) -> Tuple[np.ndarray, np.ndarray]:

        assert len(params) == 9

        if isinstance(params, tuple):
            params = list(params)

        params = np.asarray(params)

        params[0] = np.log(params[0])
        params[1] = params[1] * 1e3
        params[2] = np.log(params[2])

        if params[3] != 0.0:
            raise NotImplementedError(
                "This class has not been implemented for fddot != 0 yet."
            )

        params[5] = np.cos(params[5])
        params[8] = np.cos(params[5])

        if deriv_inds is None:
            deriv_inds = np.delete(np.arange(9), 3)

        if 3 in deriv_inds:
            deriv_inds = np.delete(deriv_inds, np.where(deriv_inds == 3)[0])

        kwargs["return_array"] = True

        # ignore t channel for snr computation
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
        )

        return params[np.array([0, 1, 2, 4, 5, 6, 7, 8])], cov

    def get_snr(self, *args: Any, **kwargs: Any) -> float:

        if "tdi2" not in kwargs:
            kwargs["tdi2"] = True

        # ensures tdi2 is added correctly for GBGPU
        return super(GBCalculatorController, self).get_snr(*args, **kwargs)
