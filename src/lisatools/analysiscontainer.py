from __future__ import annotations

import math
import warnings
from abc import ABC
from typing import Any, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from eryn.utils import TransformContainer
from scipy import interpolate

from lisatools.domains import DomainBase, DomainBaseArray, DomainSettingsBase

from . import domains

try:
    import cupy as cp

except (ModuleNotFoundError, ImportError):
    import numpy as cp

from . import detector as lisa_models
from .datacontainer import DataResidualArray
from .diagnostic import (data_signal_full_source_and_noise_likelihood,
                         data_signal_source_likelihood_term, inner_product,
                         noise_likelihood_term,
                         residual_full_source_and_noise_likelihood,
                         residual_source_likelihood_term)
from .sensitivity import SensitivityMatrix, SensitivityMatrixBase
from .stochastic import (FittedHyperbolicTangentGalacticForeground,
                         StochasticContribution)
from .utils.constants import *
from .utils.utility import AET, get_array_module


class AnalysisContainer:
    """Combinatorial container that combines sensitivity and data information.

    Args:
        data_res_arr: Data / Residual / Signal array.
        sens_mat: Sensitivity information.
        signal_gen: Callable object that takes information through ``*args`` and ``**kwargs`` and
            generates a signal in the proper channel setup employed in ``data_res_arr`` and ``sens_mat``.

    """

    def __init__(
        self,
        data_res_arr: DataResidualArray,
        sens_mat: SensitivityMatrixBase,
        signal_gen: Optional[callable] = None,
    ) -> None:
        self.data_res_arr = data_res_arr
        self.sens_mat = sens_mat

        if signal_gen is not None:
            self.signal_gen = signal_gen

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
    def sens_mat(self) -> SensitivityMatrixBase:
        """Sensitivity information."""
        return self._sens_mat

    @sens_mat.setter
    def sens_mat(self, sens_mat: SensitivityMatrixBase) -> None:
        "Set sensitivity information."
        assert isinstance(sens_mat, SensitivityMatrixBase)
        self._sens_mat = sens_mat

    @property
    def signal_gen(self) -> callable:
        """Signal generator."""
        if not hasattr(self, "_signal_gen"):
            raise ValueError(
                "User must input signal_gen kwarg to use the signal generator."
            )
        return self._signal_gen

    @signal_gen.setter
    def signal_gen(self, signal_gen: callable):
        """Set signal generator."""
        assert hasattr(signal_gen, "__call__")
        self._signal_gen = signal_gen

    @property
    def start_freq_ind(self):
        return self.data_res_arr.start_freq_ind

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
                        self.data_res_arr.f_arr * np.abs(self.data_res_arr[i]),
                    )
                    ax[i * self.sens_mat.shape[1] + j].loglog(
                        self.data_res_arr.f_arr,
                        self.data_res_arr.f_arr * np.abs(self.data_res_arr[j]),
                    )
        else:
            for i in range(self.sens_mat.shape[0]):
                ax[i].loglog(
                    self.data_res_arr.f_arr,
                    self.data_res_arr.f_arr * np.abs(self.data_res_arr[i]),
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

        return inner_product(
            self.data_res_arr, self.data_res_arr, psd=self.sens_mat, **kwargs
        )

    def snr(self, **kwargs: dict) -> float:
        """Return the SNR of the current set of information

        Args:
            **kwargs: Inner product keyword arguments.

        Returns:
            SNR value.

        """
        return self.inner_product(**kwargs).real ** (1 / 2)

    def _slice_to_template(
        self, template: DataResidualArray
    ) -> Tuple[DataResidualArray, DataResidualArray, SensitivityMatrixBase]:
        """Slice the data residual array to the same shape as the template.

        This is used for calculating inner products and likelihoods with templates that are shorter than the data.

        Args:
            template: Template signal.
        """
        data_settings = self.data_res_arr.settings
        templ_settings = template.settings

        if type(data_settings) != type(templ_settings):
            raise ValueError(
                f"Data domain ({type(data_settings).__name__}) and template domain "
                f"({type(templ_settings).__name__}) must match."
            )

        # Fast path: settings identical → no slicing needed
        if data_settings == templ_settings:
            return self.data_res_arr, template, self.sens_mat

        elif isinstance(data_settings, domains.STFTSettings):
            return self._slice_stft_to_template(template)
        else:
            raise NotImplementedError(
                f"Automatic region slicing not yet implemented for "
                f"{type(data_settings).__name__}. Ensure template and data "
                f"have the same shape, or use STFT domain."
            )

    def _slice_stft_to_template(
        self, 
        template: DataResidualArray
        ) -> Tuple[DataResidualArray, DataResidualArray, SensitivityMatrixBase]:
        """
        Slice the data residual array and sensitivity matrix to the time and frequency region covered
        by the template, for the case of STFT domain settings.

        Args:
            template: Template signal.

        Returns:
            Tuple of (sliced data residual array, sliced template, sliced sensitivity matrix).
        """

        data_settings = self.data_res_arr.settings
        templ_settings = template.settings

        # validate grids
        if not np.isclose(data_settings.dt, templ_settings.dt):
            raise ValueError(
                f"Data segment duration ({data_settings.dt}) and template segment "
                f"duration ({templ_settings.dt}) must match in STFT."
            )
        if not np.isclose(data_settings.df, templ_settings.df):
            raise ValueError(
                f"Data df ({data_settings.df}) and template df ({templ_settings.df}) must match."
            )

        # find indices for slicing
        tmin, tmax = (
            templ_settings.t0,
            templ_settings.t0 + templ_settings.NT * templ_settings.dt,
        )
        fmin, fmax = templ_settings.f_arr[0], templ_settings.f_arr[-1]

        slices = data_settings.compute_slice_indices(tmin, tmax, fmin, fmax)
        sliced_data_res_arr = DataResidualArray(self.data_res_arr.data_res_arr.get_array_slice(slices))
        sliced_sens_mat = self.sens_mat.get_slice(slices)

        return sliced_data_res_arr, template, sliced_sens_mat

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

        if "include_psd_info" in kwargs:
            kwargs.pop("include_psd_info")

        data_res_arr_sliced, template_sliced, sens_mat_sliced = self._slice_to_template(
            template
        )

        ip_val = inner_product(
            data_res_arr_sliced, template_sliced, psd=sens_mat_sliced, **kwargs
        )
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

        sliced_data_res_arr, sliced_template, sliced_sens_mat = self._slice_to_template(
            template
        )

        # TODO: should we cache?
        h_h = inner_product(
            sliced_template, sliced_template, psd=sliced_sens_mat, **kwargs_in
        )
        non_marg_d_h = inner_product(
            sliced_data_res_arr,
            sliced_template,
            psd=sliced_sens_mat,
            complex=True,
            **kwargs_in,
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
        amp_maximize: bool = False,
        **kwargs: dict,
    ) -> float:
        """Calculate the Likelihood of a template against the data.

        Args:
            template: Template signal.
            include_psd_info: If ``True``, add the PSD term to the Likelihood value.
            phase_maximize: If ``True``, maximize over an overall phase.
            amp_maximize: If ``True``, maximize over an overall amplitude.
            **kwargs: Keyword arguments to pass to :func:`lisatools.diagnostic.inner_product`.

        Returns:
            Likelihood value.

        """
        kwargs_in = kwargs.copy()
        if "psd" in kwargs_in:
            kwargs_in.pop("psd")

        if "complex" in kwargs_in:
            kwargs_in.pop("complex")

        data_res_arr_sliced, template_sliced, sens_mat_sliced = self._slice_to_template(
            template
        )

        # when computing the <d|d> term we need the full data and sensitivity matrix.

        # TODO: should we cache?
        d_d = inner_product(
            self.data_res_arr, self.data_res_arr, psd=self.sens_mat, **kwargs_in
        )
        h_h = inner_product(
            template_sliced, template_sliced, psd=sens_mat_sliced, **kwargs_in
        )

        non_marg_d_h = inner_product(
            data_res_arr_sliced,
            template_sliced,
            psd=sens_mat_sliced,
            complex=True,
            **kwargs_in,
        )

        d_h = np.abs(non_marg_d_h) if phase_maximize else non_marg_d_h.copy()
        self.non_marg_d_h = non_marg_d_h

        if amp_maximize:
            amp_factor = d_h.real / h_h.real
            d_h *= amp_factor
            h_h *= amp_factor**2
        # breakpoint()
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
                self.data_res_arr, psd=self.sens_mat, **kwargs
            )
        else:
            return residual_full_source_and_noise_likelihood(
                self.data_res_arr, self.sens_mat, **kwargs
            )

    # TODO: make sure there is a way for backends to check TDI channel structure/domain is equivalent

    def _calculate_signal_operation(
        self,
        calc: str,
        *args: Any,
        source_only: bool = False,
        waveform_kwargs: Optional[dict] = {},
        data_res_arr_kwargs: Optional[dict] = {},
        transform_fn: Optional[TransformContainer] = None,
        signal_gen: Optional[callable] = None,
        **kwargs: dict,
    ) -> float | complex:
        """Return the likelihood of a generated signal with the data.

        Args:
            calc: Type of calculation to do. Options are ``"likelihood"``, ``"inner_product"``, or ``"snr"``.
            *args: Arguments to waveform generating function. Must include parameters.
            source_only: If ``True`` return the source-only Likelihood (leave out noise part).
            waveform_kwargs: Keyword arguments to pass to waveform generator.
            data_res_arr_kwargs: Keyword arguments for instantiation of :class:`DataResidualArray`.
                This can be used if any transforms are desired prior to the Likelihood computation. If it is not input,
                the kwargs are taken to be the same as those used to initalize ``self.data_res_arr``.
            transform_fn: Transform information for signal parameters if they
                are entered on a basis other than the waveform basis.
            signal_gen: In scope waveform generator. Replaces ``self.signal_gen`` if this input is not ``None``.
            **kwargs: Keyword arguments to pass to :func:`lisatools.diagnostic.inner_product`

        Returns:
            Likelihood value.

        """

        if data_res_arr_kwargs == {}:
            data_res_arr_kwargs = self.data_res_arr.init_kwargs

        if transform_fn is not None:
            args_tmp = np.asarray(args)
            args_in = tuple(transform_fn.both_transforms(args_tmp))
        else:
            args_in = args

        signal_gen_here = self.signal_gen if signal_gen is None else signal_gen

        template = DataResidualArray(
            signal_gen_here(*args_in, **waveform_kwargs), **data_res_arr_kwargs
        )

        args_2 = (template,)

        if "include_psd_info" in kwargs:
            assert kwargs["include_psd_info"] == (not source_only)
            kwargs.pop("include_psd_info")

        kwargs = dict(psd=self.sens_mat, **kwargs)

        if calc == "likelihood":
            kwargs["include_psd_info"] = not source_only
            return self.template_likelihood(*args_2, **kwargs)
        elif calc == "inner_product":
            return self.template_inner_product(*args_2, **kwargs)
        elif calc == "snr":
            return self.template_snr(*args_2, **kwargs)
        else:
            raise ValueError("`calc` must be 'likelihood', 'inner_product', or 'snr'.")

    def calculate_signal_likelihood(
        self,
        *args: Any,
        source_only: bool = False,
        waveform_kwargs: Optional[dict] = {},
        data_res_arr_kwargs: Optional[dict] = {},
        **kwargs: dict,
    ) -> float | complex:
        """Return the likelihood of a generated signal with the data.

        Args:
            params: Arguments to waveform generating function. Must include parameters.
            source_only: If ``True`` return the source-only Likelihood (leave out noise part).
            waveform_kwargs: Keyword arguments to pass to waveform generator.
            data_res_arr_kwargs: Keyword arguments for instantiation of :class:`DataResidualArray`.
                This can be used if any transforms are desired prior to the Likelihood computation.
            **kwargs: Keyword arguments to pass to :func:`lisatools.diagnostic.inner_product`

        Returns:
            Likelihood value.

        """

        return self._calculate_signal_operation(
            "likelihood",
            *args,
            source_only=source_only,
            waveform_kwargs=waveform_kwargs,
            data_res_arr_kwargs=data_res_arr_kwargs,
            **kwargs,
        )

    def calculate_signal_inner_product(
        self,
        *args: Any,
        source_only: bool = False,
        waveform_kwargs: Optional[dict] = {},
        data_res_arr_kwargs: Optional[dict] = {},
        **kwargs: dict,
    ) -> float | complex:
        """Return the inner product of a generated signal with the data.

        Args:
            *args: Arguments to waveform generating function. Must include parameters.
            source_only: If ``True`` return the source-only Likelihood (leave out noise part).
            waveform_kwargs: Keyword arguments to pass to waveform generator.
            data_res_arr_kwargs: Keyword arguments for instantiation of :class:`DataResidualArray`.
                This can be used if any transforms are desired prior to the Likelihood computation.
            **kwargs: Keyword arguments to pass to :func:`lisatools.diagnostic.inner_product`

        Returns:
            Inner product value.

        """

        return self._calculate_signal_operation(
            "inner_product",
            *args,
            source_only=source_only,
            waveform_kwargs=waveform_kwargs,
            data_res_arr_kwargs=data_res_arr_kwargs,
            **kwargs,
        )

    def calculate_signal_snr(
        self,
        *args: Any,
        source_only: bool = False,
        waveform_kwargs: Optional[dict] = {},
        data_res_arr_kwargs: Optional[dict] = {},
        **kwargs: dict,
    ) -> Tuple[float, float]:
        """Return the SNR of a generated signal with the data.

        Args:
            *args: Arguments to waveform generating function. Must include parameters.
            source_only: If ``True`` return the source-only Likelihood (leave out noise part).
            waveform_kwargs: Keyword arguments to pass to waveform generator.
            data_res_arr_kwargs: Keyword arguments for instantiation of :class:`DataResidualArray`.
                This can be used if any transforms are desired prior to the Likelihood computation.
            **kwargs: Keyword arguments to pass to :func:`lisatools.diagnostic.inner_product`

        Returns:
            Snr values (optimal, detected).

        """

        return self._calculate_signal_operation(
            "snr",
            *args,
            source_only=source_only,
            waveform_kwargs=waveform_kwargs,
            data_res_arr_kwargs=data_res_arr_kwargs,
            **kwargs,
        )

    def eryn_likelihood_function(
        self, x: np.ndarray | list | tuple, *args: Any, **kwargs: Any
    ) -> np.ndarray | float:
        """Likelihood function for Eryn sampler.

        This function is not vectorized.

        ``signal_gen`` must be set to use this function.

        Args:
            x: Parameters. Can be 1D list, tuple, array or 2D array.
                If a 2D array is input, the computation is done serially.
            *args: Likelihood args.
            **kwargs: Likelihood kwargs.

        Returns:
            Likelihood value(s).

        """
        assert self.signal_gen is not None

        if isinstance(x, list) or isinstance(x, tuple):
            x = np.asarray(x)

        if x.ndim == 1:
            input_vals = tuple(x) + tuple(args)
            return self.calculate_signal_likelihood(*input_vals, **kwargs)
        elif x.ndim == 2:
            likelihood_out = np.zeros(x.shape[0])
            for i in range(x.shape[0]):
                input_vals = tuple(x[i]) + tuple(args)
                likelihood_out[i] = self.calculate_signal_likelihood(
                    *input_vals, **kwargs
                )
            return likelihood_out

        else:
            raise ValueError("x must be a 1D or 2D array.")


class AnalysisContainerArray:
    """
    Container for multiple analysis containers. This is useful for parallelization and batching.

    Args:
        analysis_containers: Can be a single :class:`AnalysisContainer`, a 1D list of :class:`AnalysisContainer`, or a numpy object array of :class:`AnalysisContainer`. If a 2D or higher list/array is input, it will be flattened to 1D and the original shape will be stored in ``acs_shape``.
        gpus: If not ``None``, list of GPU ids to use for storing data and sensitivity information. The data and sensitivity information for each container will be split across the GPUs as evenly as possible. If ``None``, everything is stored on the CPU.
    """


    def __init__(self, 
                 analysis_containers: AnalysisContainer | List[AnalysisContainer] | np.ndarray, 
                 gpus: list | int | None = None
                 ) -> None:
        
        if isinstance(analysis_containers, AnalysisContainer):
            acs = np.array([analysis_containers], dtype=object)
            
        elif isinstance(analysis_containers, np.ndarray):
            assert analysis_containers.dtype == object
            assert np.all(
                [
                    isinstance(tmp, AnalysisContainer)
                    for tmp in analysis_containers.flatten()
                ]
            )
            acs = analysis_containers
        elif isinstance(analysis_containers, list):
            if isinstance(analysis_containers[0], list):
                raise ValueError(
                    "If inputing list of containers, must be 1D. Use a numpy object array for 2+D."
                )
            acs = np.asarray(analysis_containers, dtype=object)
        else:
            raise ValueError(
                "Analysis container must be single container, 1D list, or numpy object array."
            )

        self.acs = acs
        self.acs_shape = acs.shape
        self.acs_total_entries = np.prod(acs.shape)

        # generalize to a potential time-frequency input, where
        data_shape = acs.flatten()[0].data_res_arr.shape

        if len(data_shape) == 1:
            self.data_length = data_shape[0]
            self.nchannels = 1
            self.end_shape = (self.data_length,)
        elif len(data_shape) == 2:
            self.nchannels, self.data_length = data_shape
            self.end_shape = (self.data_length,)
        elif len(data_shape) == 3:
            self.nchannels, self.m, self.n = (
                data_shape  # let's call the external layer m and n for now. In the stft case, m would be the number of time segments and n would be the number of frequencies. In WDM it seems this is switched.
            )
            self.data_length = self.m * self.n
            self.end_shape = (self.m, self.n)

        if gpus is not None:
            self.xp = xp = cp
            if isinstance(gpus, list):
                if len(gpus) > 1:
                    raise NotImplementedError
                xp.cuda.runtime.setDevice(gpus[0])
            elif isinstance(gpus, int):
                xp.cuda.runtime.setDevice(gpus)
        else:
            xp = np
        # xp = get_array_module(acs.flatten()[0].data_res_arr[0])

        ac_tmp = acs.flatten()[0]
        self.shape_sens = shape_sens = ac_tmp.sens_mat.shape[
            : -len(ac_tmp.sens_mat.data_shape)
        ]

        if isinstance(ac_tmp.sens_mat.basis_settings, domains.WDMSettings):
            self.data_dtype = float
        else:
            self.data_dtype = complex

        assert np.all(np.asarray(shape_sens) < 5)  # makes sure it is not length of data
        # reset so that all data are linear in memory
        num_machines = 1 if gpus is None else len(gpus)

        split_num = int(np.ceil(self.acs_total_entries / num_machines))
        split_inds = np.arange(split_num, self.acs_total_entries, split_num)

        self.gpu_splits = gpu_splits = np.split(
            np.arange(self.acs_total_entries), split_inds
        )

        self.gpu_map = np.zeros(self.acs_total_entries, dtype=int)
        self.split_map = np.zeros(self.acs_total_entries, dtype=int)
        self.linear_data_arr = []
        self.linear_psd_arr = []
        for i, split in enumerate(gpu_splits):
            if gpus is not None:
                self.gpu_map[split] = gpus[i]
            else:
                self.gpu_map[split] = 0
            self.split_map[split] = i
            self.linear_data_arr.append(
                xp.zeros(
                    self.data_length * self.nchannels * len(split),
                    dtype=self.data_dtype,
                )
            )
            self.linear_psd_arr.append(
                xp.zeros(
                    self.data_length * np.prod(shape_sens) * len(split), dtype=complex
                )
            )

        self.num_acs = len(acs.flatten())
        self.gpus = gpus
        self.reset_linear_data_arr()
        self.reset_linear_psd_arr()

    def zero_out_data_arr(self):
        if self.gpus is not None:
            main_gpu = self.xp.cuda.runtime.getDevice()

        for gpu_i, gpu in enumerate(self.gpus):
            with self.xp.cuda.device.Device(gpu):
                self.linear_data_arr[gpu_i][:] = 0.0

        self.xp.cuda.runtime.setDevice(main_gpu)

    def reset_linear_data_arr(self):
        if self.gpus is not None:
            main_gpu = self.xp.cuda.runtime.getDevice()

        # settings = self.settings
        # signal_class = settings.associated_class

        for i, ac in enumerate(self.acs.flatten()):
            gpu = self.gpu_map[i]
            split = self.split_map[i]
            if self.gpus is not None:
                self.xp.cuda.runtime.setDevice(gpu)

            # following assumes everything is ordered purposefully
            intra_split_index = np.where(self.gpu_splits[split] == i)[0][0]
            start_index = intra_split_index * (self.nchannels * self.data_length)
            end_index = (intra_split_index + 1) * (self.nchannels * self.data_length)
            self.linear_data_arr[split][start_index:end_index] = self.xp.asarray(
                ac.data_res_arr.flatten()
            )
            # ac.data_res_arr._data_res_arr = signal_class(arr=self.linear_data_arr[split][start_index:end_index].reshape(self.nchannels, *self.data_shape), settings=settings)     #as todo check: are those 2 lines the same?
            ac.data_res_arr.data_res_arr._arr = self.linear_data_arr[split][
                start_index:end_index
            ].reshape((self.nchannels,) + self.end_shape)
            # TODO: add check to make sure changes are made inline along with protections
            if self.gpus is not None:
                self.xp.get_default_memory_pool().free_all_blocks()

        if self.gpus is not None:
            self.xp.cuda.runtime.setDevice(main_gpu)

    def reset_linear_psd_arr(self):
        if self.gpus is not None:
            main_gpu = self.xp.cuda.runtime.getDevice()

        for i, ac in enumerate(self.acs.flatten()):
            gpu = self.gpu_map[i]
            split = self.split_map[i]
            if self.gpus is not None:
                self.xp.cuda.runtime.setDevice(gpu)

            # TODO: should I not store this in memory?!?!?
            intra_split_index = np.where(self.gpu_splits[split] == i)[0][0]
            start_index = intra_split_index * (
                np.prod(self.shape_sens) * self.data_length
            )
            end_index = (intra_split_index + 1) * (
                np.prod(self.shape_sens) * self.data_length
            )
            self.linear_psd_arr[split][start_index:end_index] = self.xp.asarray(
                ac.sens_mat.invC.flatten()
            )
            ac.sens_mat.invC = self.linear_psd_arr[split][
                start_index:end_index
            ].reshape(self.shape_sens + (self.m, self.n))

            # TODO: add check to make sure changes are made inline along with protections
            if self.gpus is not None:
                self.xp.get_default_memory_pool().free_all_blocks()

        if self.gpus is not None:
            self.xp.cuda.runtime.setDevice(main_gpu)

    @property
    def settings(self) -> DomainSettingsBase:
        """Basis settings of the data residual array."""
        return self.acs[0].data_res_arr.settings

    @property
    def f_arr(self):
        return self.acs[0].data_res_arr.f_arr

    @property
    def df(self):
        return self.f_arr[1] - self.f_arr[0]

    def __len__(self) -> int:
        return len(self.acs)

    def _loop_operation(self, operation: str, **kwargs: Any) -> np.ndarray:
        for i, ac in enumerate(self.acs.flatten()):
            _tmp = getattr(ac, operation)
            if callable(_tmp):
                _tmp_output = _tmp(**kwargs)
            else:
                # must be property or attribute
                _tmp_output = _tmp

            if i == 0:
                output = np.zeros(self.acs_total_entries, dtype=_tmp_output.dtype)

            output[i] = _tmp_output

        return output.reshape(self.acs_shape)

    @property
    def start_freq_ind(self):
        return self._loop_operation("start_freq_ind")

    def inner_product(self, **kwargs):
        return self._loop_operation("inner_product", **kwargs)

    def likelihood(self, **kwargs):
        return self._loop_operation("likelihood", **kwargs)

    def snr(self, **kwargs):
        return self._loop_operation("snr", **kwargs)

    def __getitem__(self, index: Any) -> np.ndarray[AnalysisContainer]:
        return self.acs[index]

    # ------------------------------------------------------------------
    # Domain-specific helpers for signal_operation
    # ------------------------------------------------------------------

    def _apply_stft_signal(
        self,
        sign: int,
        template_arr,
        template_settings: domains.STFTSettings,
        data_res_arr: DataResidualArray,
    ) -> None:
        """Add or subtract an STFT template from *data_res_arr*.

        Handles partial time support (template may cover only a sub-range of
        the data time axis) and computes the frequency intersection between
        template and data active frequency ranges.

        Args:
            sign: +1 to add, -1 to subtract.
            template_arr: Array of shape ``(nchannels, NT_tmpl, NF_tmpl_active)``.
            template_settings: Settings for the template STFT grid.
            data_res_arr: Target data residual array.

        """
        data_settings = data_res_arr.settings

        if not np.isclose(data_settings.df, template_settings.df):
            raise ValueError(
                f"Data df ({data_settings.df}) and template df "
                f"({template_settings.df}) must match."
            )
        if data_settings.NF != template_settings.NF:
            raise ValueError(
                f"Data NF ({data_settings.NF}) and template NF "
                f"({template_settings.NF}) must match."
            )

        # --- time offset (in number of time bins) ---
        time_offset = int(
            round((template_settings.t0 - data_settings.t0) / data_settings.dt)
        )
        t_start_data = max(0, time_offset)
        t_end_data = min(data_settings.NT, time_offset + template_settings.NT)

        if t_start_data >= t_end_data:
            warnings.warn(
                f"STFT template time range does not overlap with data. Skipping. "
                f"Template t0={template_settings.t0}, NT={template_settings.NT}, "
                f"dt={template_settings.dt}; "
                f"Data t0={data_settings.t0}, NT={data_settings.NT}, "
                f"dt={data_settings.dt}."
            )
            return

        tmpl_t_start = t_start_data - time_offset
        tmpl_t_end = t_end_data - time_offset

        # --- frequency intersection (in active-freq-bin coordinates) ---
        data_f0 = data_settings.f_arr[0]
        tmpl_f0 = template_settings.f_arr[0]
        data_f1 = data_settings.f_arr[-1]
        tmpl_f1 = template_settings.f_arr[-1]

        f_lo = max(data_f0, tmpl_f0)
        f_hi = min(data_f1, tmpl_f1)

        if f_lo > f_hi:
            warnings.warn(
                "STFT template and data frequency ranges do not overlap. Skipping."
            )
            return

        f_start_data = int(round((f_lo - data_f0) / data_settings.df))
        f_end_data = int(round((f_hi - data_f0) / data_settings.df)) + 1
        f_start_tmpl = int(round((f_lo - tmpl_f0) / template_settings.df))
        f_end_tmpl = f_start_tmpl + (f_end_data - f_start_data)

        data_res_arr[
            :, t_start_data:t_end_data, f_start_data:f_end_data
        ] += sign * template_arr[:, tmpl_t_start:tmpl_t_end, f_start_tmpl:f_end_tmpl]

    def _apply_fd_signal(
        self,
        sign: int,
        template_arr,
        template_settings: domains.FDSettings,
        data_res_arr: DataResidualArray,
    ) -> None:
        """Add or subtract an FD template from *data_res_arr*.

        Computes the frequency intersection between template and data active
        frequency ranges and applies only in the overlapping region.

        Args:
            sign: +1 to add, -1 to subtract.
            template_arr: Array of shape ``(nchannels, NF_tmpl_active)``.
            template_settings: Settings for the template FD grid.
            data_res_arr: Target data residual array.

        """
        data_settings = data_res_arr.settings

        if not np.isclose(data_settings.df, template_settings.df):
            raise ValueError(
                f"Data df ({data_settings.df}) and template df "
                f"({template_settings.df}) must match."
            )

        data_f0 = data_settings.f_arr[0]
        tmpl_f0 = template_settings.f_arr[0]
        data_f1 = data_settings.f_arr[-1]
        tmpl_f1 = template_settings.f_arr[-1]

        f_lo = max(data_f0, tmpl_f0)
        f_hi = min(data_f1, tmpl_f1)

        if f_lo > f_hi:
            warnings.warn(
                "FD template and data frequency ranges do not overlap. Skipping."
            )
            return

        f_start_data = int(round((f_lo - data_f0) / data_settings.df))
        f_end_data = int(round((f_hi - data_f0) / data_settings.df)) + 1
        f_start_tmpl = int(round((f_lo - tmpl_f0) / template_settings.df))
        f_end_tmpl = f_start_tmpl + (f_end_data - f_start_data)

        data_res_arr[:, f_start_data:f_end_data] += (
            sign * template_arr[:, f_start_tmpl:f_end_tmpl]
        )

    def _apply_td_signal(
        self,
        sign: int,
        template_arr,
        template_settings: domains.TDSettings,
        data_res_arr: DataResidualArray,
    ) -> None:
        """Add or subtract a TD template from *data_res_arr*.

        Handles partial time support (template may cover only a sub-range of
        the data time axis).

        Args:
            sign: +1 to add, -1 to subtract.
            template_arr: Array of shape ``(nchannels, N_tmpl)``.
            template_settings: Settings for the template TD grid.
            data_res_arr: Target data residual array.

        """
        data_settings = data_res_arr.settings

        if not np.isclose(data_settings.dt, template_settings.dt):
            raise ValueError(
                f"Data dt ({data_settings.dt}) and template dt "
                f"({template_settings.dt}) must match."
            )

        time_offset = int(
            round((template_settings.t0 - data_settings.t0) / data_settings.dt)
        )
        t_start_data = max(0, time_offset)
        t_end_data = min(data_settings.N, time_offset + template_settings.N)

        if t_start_data >= t_end_data:
            warnings.warn(
                f"TD template time range does not overlap with data. Skipping. "
                f"Template t0={template_settings.t0}, N={template_settings.N}; "
                f"Data t0={data_settings.t0}, N={data_settings.N}."
            )
            return

        tmpl_t_start = t_start_data - time_offset
        tmpl_t_end = t_end_data - time_offset

        data_res_arr[:, t_start_data:t_end_data] += (
            sign * template_arr[:, tmpl_t_start:tmpl_t_end]
        )

    def _apply_wdm_signal(
        self,
        sign: int,
        template_arr,
        data_res_arr: DataResidualArray,
    ) -> None:
        """Add or subtract a WDM template (full support) from *data_res_arr*.

        Args:
            sign: +1 to add, -1 to subtract.
            template_arr: Array with the same shape as the data.
            data_res_arr: Target data residual array.

        """
        data_res_arr[:] += sign * template_arr

    # ------------------------------------------------------------------
    # Public signal operation API
    # ------------------------------------------------------------------

    def signal_operation(
        self,
        sign: int,
        templates,
        data_index: Optional[np.ndarray] = None,
        start_index=None,
    ) -> None:
        """Apply ``sign * template`` to each targeted data residual array.

        Templates are applied in a domain-aware manner: time or frequency
        offsets are inferred from the template's domain settings, and only
        the overlapping support is modified.

        Args:
            sign: ``+1`` to add, ``-1`` to subtract.
            templates: One of:

                * a :class:`~lisatools.domains.DomainBaseArray` (recommended),
                * a list of :class:`~lisatools.domains.DomainBase` objects,
                * a single :class:`~lisatools.domains.DomainBase` (possibly batched),
                * a raw ``np.ndarray`` / ``cp.ndarray`` (legacy – deprecated).

            data_index: 1-D integer array mapping ``templates[i]`` to
                ``self.acs.flatten()[data_index[i]]``.  When ``None``, a
                one-to-one mapping is assumed (requires
                ``len(templates) == self.acs_total_entries``).
            start_index: Kept for backward compatibility with raw-array calls.
                Ignored when domain-aware templates are supplied.

        """
        # ---- normalise *templates* to a flat list of DomainBase ----
        if isinstance(templates, DomainBaseArray):
            item_list = list(templates)

        elif isinstance(templates, list):
            item_list = templates

        elif isinstance(templates, DomainBase):
            if templates.is_batched:
                settings = templates.settings
                item_list = [
                    settings.associated_class(templates.arr[i], settings)
                    for i in range(templates.nbatch)
                ]
            else:
                item_list = [templates]

        elif isinstance(templates, (np.ndarray, cp.ndarray)):
            # ---- legacy path: raw array ----
            warnings.warn(
                "Passing a raw ndarray to signal_operation is deprecated. "
                "Wrap your templates in a DomainBase (or DomainBaseArray) instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            # Determine domain from the first AC
            ac0_settings = self.acs.flatten()[0].data_res_arr.settings
            if isinstance(ac0_settings, domains.WDMSettings):
                if templates.ndim == 2:
                    templates = templates[None, :]
                num_templates = templates.shape[0]
            else:
                if templates.ndim == 3:
                    templates = templates[None, :]
                num_templates = templates.shape[0]

            if data_index is None:
                assert num_templates == self.acs_total_entries
                data_index = np.arange(num_templates)
            else:
                data_index = np.asarray(data_index)

            if start_index is None:
                start_index = np.zeros(num_templates, dtype=int)
            else:
                start_index = np.asarray(start_index)

            template_length = int(np.prod(templates.shape[2:]))
            for i, (di, si) in enumerate(zip(data_index, start_index)):
                self.acs.flatten()[di].data_res_arr[
                    :, si : si + template_length
                ] += sign * templates[i]
            return

        else:
            raise TypeError(
                f"templates must be a DomainBase, list of DomainBase, "
                f"DomainBaseArray, or ndarray (legacy). Got {type(templates)}."
            )

        # ---- domain-aware path ----
        num_templates = len(item_list)

        if data_index is None:
            assert num_templates == self.acs_total_entries, (
                f"Number of templates ({num_templates}) must equal the number of "
                f"analysis containers ({self.acs_total_entries}) when data_index is None."
            )
            data_index = np.arange(num_templates)
        else:
            data_index = np.asarray(data_index)
            assert data_index.max() < self.acs_total_entries

        acs_flat = self.acs.flatten()
        for i, di in enumerate(data_index):
            signal = item_list[i]
            data_res_arr = acs_flat[di].data_res_arr
            template_arr = signal.arr
            template_settings = signal.settings

            if isinstance(template_settings, domains.STFTSettings):
                self._apply_stft_signal(
                    sign, template_arr, template_settings, data_res_arr
                )
            elif isinstance(template_settings, domains.FDSettings):
                self._apply_fd_signal(
                    sign, template_arr, template_settings, data_res_arr
                )
            elif isinstance(template_settings, domains.WDMSettings):
                self._apply_wdm_signal(sign, template_arr, data_res_arr)
            elif isinstance(template_settings, domains.TDSettings):
                self._apply_td_signal(
                    sign, template_arr, template_settings, data_res_arr
                )
            else:
                raise ValueError(
                    f"Unknown domain type for template {i}: {type(template_settings)}"
                )

    def add_signal_to_residual(self, templates, data_index=None, **kwargs) -> None:
        """Subtract templates from the residual (residual = data - signal).

        Args:
            templates: See :meth:`signal_operation`.
            data_index: See :meth:`signal_operation`.
            **kwargs: Passed through to :meth:`signal_operation`.

        """
        self.signal_operation(-1, templates, data_index=data_index, **kwargs)

    def remove_signal_from_residual(self, templates, data_index=None, **kwargs) -> None:
        """Add templates back into the residual.

        Args:
            templates: See :meth:`signal_operation`.
            data_index: See :meth:`signal_operation`.
            **kwargs: Passed through to :meth:`signal_operation`.

        """
        self.signal_operation(+1, templates, data_index=data_index, **kwargs)

    @property
    def data_shaped(self):
        out = []
        for i, tmp in enumerate(self.linear_data_arr):
            if self.gpus is not None:
                self.xp.cuda.runtime.setDevice(self.gpus[i])
            out.append(tmp.reshape(-1, self.nchannels, *self.data_shape))
        return out

    @property
    def psd_shaped(self):
        out = []
        for i, tmp in enumerate(self.linear_psd_arr):
            if self.gpus is not None:
                self.xp.cuda.runtime.setDevice(self.gpus[i])
            out.append(tmp.reshape(-1, *self.shape_sens, *self.data_shape))
        return out
