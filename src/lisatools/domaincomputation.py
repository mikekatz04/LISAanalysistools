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
    try:
        import cupy as cp
    except (ModuleNotFoundError, ImportError):
        import numpy as cp

logger = logging.getLogger(__name__)

class BaseDomainComputationGroup(LISAToolsParallelModule):
    """Wraps C++ DomainWrap for batched likelihood computation on the AnalysisContainerArray data.

    One instance per GPU split.  Holds references to the linearized arrays
    to prevent GC from invalidating the C++ domain's pointers.

    Args:
        acs : AnalysisContainerArray, optional
            The AnalysisContainerArray containing the data and noise arrays. If not provided, the necessary arrays and Args: must be provided directly.
        split_index : int, optional
            The index of the GPU split to use from the AnalysisContainerArray. Only used if `acs` is provided. Default is 0.
        data_arr : np.ndarray, optional
            The linearized data array for the current split. Only used if `acs` is not provided.
        invC_arr : np.ndarray, optional
            The linearized inverse noise PSD array for the current split. Only used if `acs` is not provided.
        num_data : int, optional
            The number of data points for the current split. Only used if `acs` is not provided.
        num_noise : int, optional
            The number of noise points for the current split. Only used if `acs` is not provided.
        num_channels : int, optional
            The number of channels for the current split. Only used if `acs` is not provided.
        settings : STFTSettings or FDSettings, optional
            The settings for the domain computation. Must be an instance of STFTSettings for STFTComputationGroup or FDSettings for FDComputationGroup. Only used if `acs` is not provided.
        tdi_type : str, optional
            The TDI type to use for the likelihood computation. Default is "XYZ". Must be a key in the backend's TDITypeDict.
        force_backend : str, optional   
            If provided, forces the use of the specified backend. Must be one of the supported backends for the domain computation. Default is 'cpu'.   
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
                    raise ValueError("All Args: must be provided if acs is not given.")
        # Keep references alive so the C++ pointers remain valid. We do not copy to always point to the same memory.
            self.data_arr = data_arr
            self.invC_arr = invC_arr
            self.num_channels = num_channels
            self.num_data = num_data
            self.num_noise = num_noise
            self.settings = settings

    def extract_from_acs(self, acs: AnalysisContainerArray, split_index: int):
        """
        Extracts the necessary arrays and Args: from the given AnalysisContainerArray for the specified split index.

        Args:
            acs : AnalysisContainerArray
                The AnalysisContainerArray containing the data and noise arrays.
            split_index : int
                The index of the GPU split to use from the AnalysisContainerArray.
        """
        self.split_index = split_index
        self.data_arr = acs.linear_data_arr[split_index]
        self.invC_arr = acs.linear_psd_arr[split_index]
        self.num_channels = acs.nchannels
        self.num_data = len(acs.gpu_splits[split_index])
        self.num_noise = len(acs.gpu_splits[split_index])
        self.settings = acs.settings

        # Store references to the containers in this split for (d|d) computation.
        # Do NOT wrap them in a new AnalysisContainerArray — that would call
        # reset_linear_data_arr / reset_linear_psd_arr, rebinding each AC's
        # internal ._arr to a new buffer and breaking the C++ pointer contract.
        all_acs = acs.acs.flatten()
        split_container_ids = acs.gpu_splits[split_index]
        self.split_acs = [all_acs[i] for i in split_container_ids]

    @property
    def split_acs(self) -> list:
        if not hasattr(self, "_split_acs"):
            raise ValueError("Split ACs have not been set. Call extract_from_acs first.")
        return self._split_acs
    @split_acs.setter
    def split_acs(self, value: list):
        self._split_acs = value

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
        Compute (d|d) term for the containers in this split only.

        Args:
            out : bool, optional
                If True, return the computed (d|d) term. Otherwise, store it in the instance variable `self.d_d`. Default is False.
            **kwargs
                Additional keyword arguments to pass to the `inner_product` method of each AnalysisContainer.

        Returns:
            If `out` is True, returns a double array of shape ``(num_data,)`` containing the (d|d) term for each container in the split. Otherwise, returns None and stores the result in `self.d_d`.

        Notes
        -----
        The result ``self.d_d`` has shape ``(num_data,)`` — one value per
        container in the split, indexed by intra-split index.
        """
        if not hasattr(self, "_split_acs") or self._split_acs is None:
            raise ValueError("Split ACs are not set. Cannot compute (d|d) term. "
                             "Provide an AnalysisContainerArray to extract_from_acs first.")

        d_d = self.xp.zeros(self.num_data, dtype=self.xp.float64)
        for i, ac in enumerate(self._split_acs):
            d_d[i] = ac.inner_product(**kwargs)
        self.d_d = d_d

        if out:
            return self.d_d.copy()
        
    def compute_likelihood_terms(
        self, 
        data_index: np.ndarray | cp.ndarray,
        noise_index: np.ndarray | cp.ndarray,
        template_vals: np.ndarray | cp.ndarray,
        start_freqs: np.ndarray | cp.ndarray,
        **kwargs
        ) -> tuple[np.ndarray | cp.ndarray, np.ndarray | cp.ndarray]:
        """
        Compute the inner products :math:`\\langle d | h\\rangle` and :math:`\\langle h | h\\rangle` for the input set of binaries.

        Args:
            *args: positional arguments
        """
        raise NotImplementedError("The `compute_likelihood_terms` method must be implemented by subclasses")

    def compute_likelihood(
        self,
        data_index: np.ndarray | cp.ndarray,
        noise_index: np.ndarray | cp.ndarray,
        template_vals: np.ndarray | cp.ndarray,
        start_freqs: np.ndarray | cp.ndarray,
        start_times: np.ndarray | cp.ndarray = None,
        **kwargs
    ) -> np.ndarray | cp.ndarray:
        """
        Compute the log-likelihood for a batch of binaries.

        Args:
            template_vals : complex array
                Shape ``(num_binaries, num_channels, n_t, n_f)`` for STFT or ``(num_binaries, num_channels, n_f)`` for FD.
            data_index : int array, shape ``(num_binaries,)``
            noise_index : int array, shape ``(num_binaries,)``
            start_freqs : double array, shape ``(num_binaries,)``
            start_times : double array, shape ``(num_binaries,)``, optional
                 Only used for STFT. If not provided, defaults to None.
            **kwargs: additional keyword arguments to pass to the `compute_likelihood_terms` method. Kept for future extensibility.

        Returns:
            like_out : double array, shape ``(num_binaries,)``
        """
        d_h_out, h_h_out = self.compute_likelihood_terms(
            data_index=data_index, 
            noise_index=noise_index, 
            template_vals=template_vals, 
            start_freqs=start_freqs, 
            start_times=start_times,
            **kwargs
        )
        
        d_d_per_binary = self.d_d[data_index]
        like_out = -1. / 2. * (d_d_per_binary + h_h_out - 2 * d_h_out).real
        return like_out



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
        data_index: np.ndarray | cp.ndarray,
        noise_index: np.ndarray | cp.ndarray,
        template_vals: np.ndarray | cp.ndarray,
        start_freqs: np.ndarray | cp.ndarray,
        start_times: np.ndarray | cp.ndarray,
        **kwargs
    ) -> tuple[np.ndarray | cp.ndarray, np.ndarray | cp.ndarray]:
        """
        Compute (d|h) and (h|h) for a batch of binaries.

        Args:
            template_vals : complex array
                Shape ``(num_binaries, num_channels, n_t, n_f)``.
            data_index : int array, shape ``(num_binaries,)``
            noise_index : int array, shape ``(num_binaries,)``
            start_freqs : double array, shape ``(num_binaries,)``
            start_times : double array, shape ``(num_binaries,)``

        Returns:
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
        data_index: np.ndarray | cp.ndarray,
        noise_index: np.ndarray | cp.ndarray,
        template_vals: np.ndarray | cp.ndarray,
        start_freqs: np.ndarray | cp.ndarray,
        **kwargs
    ) -> tuple[np.ndarray | cp.ndarray, np.ndarray | cp.ndarray]:
        """
        Compute (d|h) and (h|h) for a batch of binaries.

        Args:
            template_vals : complex array
                Shape ``(num_binaries, num_channels, n_f)``.
            data_index : int array, shape ``(num_binaries,)``
            noise_index : int array, shape ``(num_binaries,)``
            start_freqs : double array, shape ``(num_binaries,)``

        Returns:
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