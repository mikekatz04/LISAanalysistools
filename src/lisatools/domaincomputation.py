"""Python wrapper around C++ STFTDomainWrap / FDDomainWrap for batched
likelihood computation of (d|h) and (h|h) inner products."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from .utils.parallelbase import LISAToolsParallelModule

if TYPE_CHECKING:
    from .analysiscontainer import AnalysisContainerArray
    from .domains import STFTSettings, FDSettings

logger = logging.getLogger(__name__)

class BaseDomainComputationGroup(LISAToolsParallelModule):
    """Wraps C++ DomainWrap for batched likelihood computation on the AnalysisContainerArray data.

    One instance per GPU split.  Holds references to the linearized arrays
    to prevent GC from invalidating the C++ domain's pointers.
    """

    def __init__(
        self,
        acs: AnalysisContainerArray = None,
        split_index: int = 0,
        data_arr: np.ndarray | None = None,
        invC_arr: np.ndarray | None = None,
        num_data: int = None,
        num_noise: int = None,
        num_channels: int = None,
        settings: STFTSettings | FDSettings = None,
        tdi_type: str = "XYZ",
        force_backend:str ='cpu',
    ):
        super().__init__(force_backend=force_backend)
        self.tdi_type = tdi_type
        from .domains import STFTSettings, FDSettings
        assert isinstance(settings, (STFTSettings, FDSettings)), f"settings must be an instance of STFTSettings or FDSettings. No other domain is currently supported. Got {type(settings)}"

        if acs is not None:
            self.extract_from_acs(acs, split_index)
        else:
            #todo not sure if I want to keep this or make the class only work with acs, which would be cleaner. 
            for param in [
                data_arr,
                invC_arr,
                num_data,
                num_noise,
                num_channels,
                settings,
            ]:
                if param is None:
                    raise ValueError("All parameters must be provided if acs is not given.")
        # Keep references alive so the C++ pointers remain valid. We do not copy to always point to the same memory.
            self.data_arr = data_arr
            self.invC_arr = invC_arr
            self.num_channels = num_channels
            self.num_data = num_data
            self.num_noise = num_noise
            self.settings = settings

    def extract_from_acs(self, acs: AnalysisContainerArray, split_index: int):
        """
        Extracts the necessary arrays and parameters from the given AnalysisContainerArray for the specified split index.

        """
        self._acs = acs
        self.split_index = split_index
        self.data_arr = acs.linear_data_arr[split_index]
        self.invC_arr = acs.linear_psd_arr[split_index]
        self.num_channels = acs.nchannels
        self.num_data = len(acs.gpu_splits[split_index])
        self.num_noise = len(acs.gpu_splits[split_index])
        self.settings = acs.settings

    @property
    def acs(self):
        return self._acs
    @acs.setter
    def acs(self, value):
        self._acs = value

    @property
    def data_arr(self):
        return self._data_arr
    @data_arr.setter
    def data_arr(self, value):
        self._data_arr = value

    @property
    def invC_arr(self):
        return self._invC_arr
    @invC_arr.setter
    def invC_arr(self, value):
        self._invC_arr = value

    @property
    def num_channels(self):
        return self._num_channels
    @num_channels.setter
    def num_channels(self, value):
        self._num_channels = value

    @property
    def num_data(self):
        return self._num_data
    @num_data.setter
    def num_data(self, value):
        self._num_data = value

    @property
    def num_noise(self):
        return self._num_noise
    @num_noise.setter
    def num_noise(self, value):
        self._num_noise = value

    @property
    def settings(self):
        return self._settings
    @settings.setter
    def settings(self, value):
        self._settings = value

    @property
    def d_d(self):
        return self._d_d
    @d_d.setter
    def d_d(self, value):
        self._d_d = value

    @property
    def xp(self):
        return self.backend.xp
    
    @property
    def pycpp_domain(self):
        if not hasattr(self, "_pycpp_domain"):
            self._pycpp_domain = self._create_pycpp_domain()
        return self._pycpp_domain

    def _create_pycpp_domain(self):
        raise NotImplementedError("Subclasses must implement _create_pycpp_domain")


    def compute_d_d_term(self, out=False, **kwargs):
        """
        Compute (d|d) term for the current data and noise arrays. We can use the AnalysisContainerArray method for this.

        Parameters
        ----------
        out : bool, optional
            If True, return the computed (d|d) term. Otherwise, store it in the instance variable `self.d_d`. Default is False.
        **kwargs
            Additional keyword arguments to pass to the `inner_product` method of the AnalysisContainerArray.
        """ 
        self.d_d = self.acs.inner_product(**kwargs)
        if out:
            return self.d_d



class STFTComputationGroup(BaseDomainComputationGroup):
    """Wraps C++ STFTDomainWrap for batched likelihood computation."""
    
    def __init__(self,
                 *args,
                 settings: STFTSettings = None,
                 **kwargs):
        from .domains import STFTSettings
        if settings is None or not isinstance(settings, STFTSettings):
            raise ValueError("settings must be an instance of STFTSettings for STFTComputationGroup.")
        super().__init__(*args, settings=settings, **kwargs)

    @property
    def domain_args(self):
        return [
            self.settings.NT,
            self.settings.NF,
            self.num_channels,
            self.settings.t0,
            self.settings.min_freq,
            self.settings.max_freq,
            self.settings.dt,
            self.settings.df,
            self.data_arr,
            self.invC_arr,
            self.num_data,
            self.num_noise,
            self.backend.TDITypeDict[self.tdi_type],
        ]

    def _create_pycpp_domain(self):
        domain = self.backend.STFTDomainWrap(*self.domain_args)
        logger.debug("Initialized STFTDomainWrap with arguments: %s", self.domain_args)
        return domain

    def compute_likelihood_terms(
        self,
        template_vals,
        start_times,
        start_freqs,
        data_index,
        noise_index,
        **kwargs
    ):
        """Compute (d|h) and (h|h) for a batch of binaries.

        Parameters
        ----------
        template_vals : complex array
            Shape ``(num_binaries, num_channels, n_t, n_f)``.
        start_times : double array, shape ``(num_binaries,)``
        start_freqs : double array, shape ``(num_binaries,)``
        data_index : int array, shape ``(num_binaries,)``
        noise_index : int array, shape ``(num_binaries,)``

        Returns
        -------
        d_h_out : complex array, shape ``(num_binaries,)``
        h_h_out : complex array, shape ``(num_binaries,)``
        """
        num_binaries, _, num_times, num_freqs = template_vals.shape

        d_h_out = self.xp.zeros(num_binaries, dtype=self.xp.complex128)
        h_h_out = self.xp.zeros(num_binaries, dtype=self.xp.complex128)

        start_freqs = self.xp.ascontiguousarray(start_freqs, dtype=self.xp.float64)
        start_times = self.xp.ascontiguousarray(start_times, dtype=self.xp.float64)
        data_index = self.xp.ascontiguousarray(data_index, dtype=self.xp.int32)
        noise_index = self.xp.ascontiguousarray(noise_index, dtype=self.xp.int32)

        self.pycpp_domain.compute_likelihood_terms(
            d_h_out,
            h_h_out,
            template_vals.ravel(),
            start_times,
            start_freqs,
            num_binaries,
            data_index,
            noise_index,
            num_times,
            num_freqs,
        )

        return d_h_out, h_h_out

    def compute_likelihood(
        self,
        template_vals,
        start_times,
        start_freqs,
        data_index,
        noise_index,
        **kwargs
    ):
        """Compute the log-likelihood for a batch of binaries.

        Returns
        -------
        like_out : double array, shape ``(num_binaries,)``
        """
        d_h_out, h_h_out = self.compute_likelihood_terms(
            template_vals, start_times, start_freqs, data_index, noise_index, **kwargs
        )
        like_out = -1. / 2. * (self.d_d + h_h_out - 2 * d_h_out).real
        return like_out

class FDComputationGroup(BaseDomainComputationGroup):
    """
    Wraps C++ FDDomainWrap for batched likelihood computation.
    """
    def __init__(self,
                 *args,
                 settings: FDSettings = None,
                 **kwargs):
        from .domains import FDSettings
        if settings is None or not isinstance(settings, FDSettings):
            raise ValueError("settings must be an instance of FDSettings for FDComputationGroup.")
        super().__init__(*args, settings=settings, **kwargs)

    @property
    def domain_args(self):
        return [
            self.settings.N,
            self.num_channels,
            self.settings.min_freq,
            self.settings.max_freq,
            self.settings.df,
            self.data_arr,
            self.invC_arr,
            self.num_data,
            self.num_noise,
            self.backend.TDITypeDict[self.tdi_type],
        ]

    def _create_pycpp_domain(self):
        domain = self.backend.FDDomainWrap(*self.domain_args)
        logger.debug("Initialized FDDomainWrap with arguments: %s", self.domain_args)
        return domain

    def compute_likelihood_terms(
        self,
        template_vals,
        start_freqs,
        data_index,
        noise_index,
        **kwargs
    ):
        """Compute (d|h) and (h|h) for a batch of binaries.

        Parameters
        ----------
        template_vals : complex array
            Shape ``(num_binaries, num_channels, n_f)``.
        start_freqs : double array, shape ``(num_binaries,)``
        data_index : int array, shape ``(num_binaries,)``
        noise_index : int array, shape ``(num_binaries,)``

        Returns
        -------
        d_h_out : complex array, shape ``(num_binaries,)``
        h_h_out : complex array, shape ``(num_binaries,)``
        """
        num_binaries, _, num_freqs = template_vals.shape

        d_h_out = self.xp.zeros(num_binaries, dtype=self.xp.complex128)
        h_h_out = self.xp.zeros(num_binaries, dtype=self.xp.complex128)

        start_freqs = self.xp.ascontiguousarray(start_freqs, dtype=self.xp.float64)
        data_index = self.xp.ascontiguousarray(data_index, dtype=self.xp.int32)
        noise_index = self.xp.ascontiguousarray(noise_index, dtype=self.xp.int32)

        self.pycpp_domain.compute_likelihood_terms(
            d_h_out,
            h_h_out,
            template_vals.ravel(),
            start_freqs,
            num_binaries,
            data_index,
            noise_index,
            num_freqs,
        )

        return d_h_out, h_h_out

    def compute_likelihood(
        self,
        template_vals,
        start_freqs,
        data_index,
        noise_index,
        **kwargs
    ):
        """Compute the log-likelihood for a batch of binaries.

        Returns
        -------
        like_out : double array, shape ``(num_binaries,)``
        """
        d_h_out, h_h_out = self.compute_likelihood_terms(
            template_vals, start_freqs, data_index, noise_index, **kwargs
        )
        like_out = -1. / 2. * (self.d_d + h_h_out - 2 * d_h_out).real
        return like_out