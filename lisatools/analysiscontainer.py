import warnings
from abc import ABC
from typing import Any, Tuple, Optional, List

import math
import numpy as np
from numpy.typing import ArrayLike
from scipy import interpolate
import matplotlib.pyplot as plt

try:
    import cupy as cp

except (ModuleNotFoundError, ImportError):
    import numpy as cp

from . import detector as lisa_models
from .utils.utility import AET, get_array_module
from .utils.constants import *
from .stochastic import (
    StochasticContribution,
    FittedHyperbolicTangentGalacticForeground,
)
from .datacontainer import DataResidualArray
from .sensitivity import SensitivityMatrix
from .diagnostic import (
    noise_likelihood_term,
    residual_full_source_and_noise_likelihood,
    residual_source_likelihood_term,
    inner_product,
)


class AnalysisContainer:
    """Combinatorial container that combines sensitivity and data information.

    Args:
        data_res_arr: Data / Residual / Signal array.
        sens_mat: Sensitivity information.

    """

    def __init__(
        self, data_res_arr: DataResidualArray, sens_mat: SensitivityMatrix
    ) -> None:
        self.data_res_arr = data_res_arr
        self.sens_mat = sens_mat

    @property
    def data_res_arr(self) -> DataResidualArray:
        """Data information."""
        return self._data_res_arr

    @data_res_arr.setter
    def data_res_arr(self, data_res_arr: DataResidualArray) -> None:
        """Set data."""
        assert isinstance(data_res_arr, DataResidualArray)
        self._data_res_arr = data_res_arr

    @property
    def sens_mat(self) -> SensitivityMatrix:
        """Sensitivity information."""
        return self._sens_mat

    @sens_mat.setter
    def sens_mat(self, sens_mat: SensitivityMatrix) -> None:
        "Set sensitivity information."
        assert isinstance(sens_mat, SensitivityMatrix)
        self._sens_mat = sens_mat

    def loglog(self) -> Tuple[plt.Figure, plt.Axes]:
        """Produce loglog plot of both source and sensitivity information.

        Returns:
            Matplotlib figure and axes object in a 2-tuple.

        """
        fig, ax = self.sens_mat.loglog(char_strain=True)
        if self.sens_mat.ndim == 3:
            # 3x3 most likely
            for i in range(self.sens_mat.shape[0]):
                for j in range(i, self.sens_mat.shape[1]):
                    # char strain
                    ax[i * self.sens_mat.shape[1] + j].loglog(
                        self.data_res_arr.f_arr,
                        self.data_res_arr.f_arr * self.data_res_arr[i],
                    )
                    ax[i * self.sens_mat.shape[1] + j].loglog(
                        self.data_res_arr.f_arr,
                        self.data_res_arr.f_arr * self.data_res_arr[j],
                    )
        else:
            for i in range(self.sens_mat.shape[0]):
                ax[i].loglog(
                    self.data_res_arr.f_arr,
                    self.data_res_arr.f_arr * self.data_res_arr[i],
                )
        return (fig, ax)

    def inner_product(self, **kwargs: dict) -> float | complex:
        """Return the inner product of the current set of information

        Args:
            **kwargs: Inner product keyword arguments.

        Returns:
            Inner product value.

        """
        if "psd" in kwargs:
            kwargs.pop("psd")

        return inner_product(self.data_res_arr, self.data_res_arr, psd=self.sens_mat)

    def template_inner_product(
        self, template: DataResidualArray, **kwargs: dict
    ) -> float | complex:
        """Calculate the inner product of a template with the data.

        Args:
            template: Template signal.
            **kwargs: Keyword arguments to pass to :func:`lisatools.diagnostic.inner_product`.

        Returns:
            Inner product value.

        """
        if "psd" in kwargs:
            kwargs.pop("psd")

        ip_val = inner_product(self.data_res_arr, template, psd=self.sens_mat, **kwargs)
        return ip_val

    def template_snr(
        self, template: DataResidualArray, phase_maximize: bool = False, **kwargs: dict
    ) -> Tuple[float, float]:
        """Calculate the SNR of a template, both optimal and detected.

        Args:
            template: Template signal.
            phase_maximize: If ``True``, maximize over an overall phase.
            **kwargs: Keyword arguments to pass to :func:`lisatools.diagnostic.inner_product`.

        Returns:
            ``(optimal snr, detected snr)``.

        """
        kwargs_in = kwargs.copy()
        if "psd" in kwargs:
            kwargs_in.pop("psd")

        if "complex" in kwargs_in:
            kwargs_in.pop("complex")

        # TODO: should we cache?
        h_h = inner_product(template, template, psd=self.sens_mat, **kwargs_in)
        non_marg_d_h = inner_product(
            self.data_res_arr, template, psd=self.sens_mat, complex=True, **kwargs_in
        )
        d_h = np.abs(non_marg_d_h) if phase_maximize else non_marg_d_h.copy()
        self.non_marg_d_h = non_marg_d_h

        opt_snr = np.sqrt(h_h.real)
        det_snr = d_h.real / opt_snr
        return (opt_snr, det_snr)

    def template_likelihood(
        self,
        template: DataResidualArray,
        include_psd_info: bool = False,
        phase_maximize: bool = False,
        **kwargs: dict
    ) -> float:
        """Calculate the Likelihood of a template against the data.

        Args:
            template: Template signal.
            include_psd_info: If ``True``, add the PSD term to the Likelihood value.
            phase_maximize: If ``True``, maximize over an overall phase.
            **kwargs: Keyword arguments to pass to :func:`lisatools.diagnostic.inner_product`.

        Returns:
            Likelihood value.

        """
        kwargs_in = kwargs.copy()
        if "psd" in kwargs_in:
            kwargs_in.pop("psd")

        if "complex" in kwargs_in:
            kwargs_in.pop("complex")

        # TODO: should we cache?
        d_d = inner_product(
            self.data_res_arr, self.data_res_arr, psd=self.sens_mat, **kwargs_in
        )
        h_h = inner_product(template, template, psd=self.sens_mat, **kwargs_in)
        non_marg_d_h = inner_product(
            self.data_res_arr, template, psd=self.sens_mat, complex=True, **kwargs_in
        )
        d_h = np.abs(non_marg_d_h) if phase_maximize else non_marg_d_h.copy()
        self.non_marg_d_h = non_marg_d_h
        like_out = -1 / 2 * (d_d + h_h - 2 * d_h).real

        if include_psd_info:
            # add noise term if requested
            like_out += self.likelihood(noise_only=True)

        return like_out

    def likelihood(
        self, source_only: bool = False, noise_only: bool = False, **kwargs: dict
    ) -> float | complex:
        """Return the likelihood of the current arangement.

        Args:
            source_only: If ``True`` return the source-only Likelihood.
            noise_only: If ``True``, return the noise part of the Likelihood alone.
            **kwargs: Keyword arguments to pass to :func:`lisatools.diagnostic.inner_product`.

        Returns:
            Likelihood value.

        """
        if noise_only and source_only:
            raise ValueError("noise_only and source only cannot both be True.")
        elif noise_only:
            return noise_likelihood_term(self.sens_mat)
        elif source_only:
            return residual_source_likelihood_term(
                self.data_res_arr, self.sens_mat, **kwargs
            )
        else:
            return residual_full_source_and_noise_likelihood(
                self.data_res_arr, self.sens_mat, **kwargs
            )
