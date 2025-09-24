from __future__ import annotations


import warnings
from abc import ABC
from typing import Any, Tuple, Optional, List

import math
import numpy as np

from scipy import interpolate
import matplotlib.pyplot as plt

from eryn.utils import TransformContainer


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
    data_signal_source_likelihood_term,
    data_signal_full_source_and_noise_likelihood,
)


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
        sens_mat: SensitivityMatrix,
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
    def sens_mat(self) -> SensitivityMatrix:
        """Sensitivity information."""
        return self._sens_mat

    @sens_mat.setter
    def sens_mat(self, sens_mat: SensitivityMatrix) -> None:
        "Set sensitivity information."
        assert isinstance(sens_mat, SensitivityMatrix)
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

        return inner_product(self.data_res_arr, self.data_res_arr, psd=self.sens_mat)

    def snr(self, **kwargs: dict) -> float:
        """Return the SNR of the current set of information

        Args:
            **kwargs: Inner product keyword arguments.

        Returns:
            SNR value.

        """
        return self.inner_product(**kwargs).real ** (1 / 2)

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
        **kwargs: dict,
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

        else:
            raise ValueError("x must be a 1D or 2D array.")


class AnalysisContainerArray:
    def __init__(self, analysis_containers, gpus=None):
        if isinstance(analysis_containers, AnalysisContainer):
            acs = np.array([analysis_containers], dtype=object)
        elif isinstance(analysis_containers, np.ndarray):
            assert analysis_containers.dtype == object
            assert np.all([isinstance(tmp, AnalysisContainer) for tmp in analysis_containers.flatten()])
            acs = analysis_containers
        elif isinstance(analysis_containers, list):
            if isinstance(analysis_containers[0], list):
                raise ValueError("If inputing list of containers, must be 1D. Use a numpy object array for 2+D.")
            acs = np.asarray(analysis_containers, dtype=object)
        else:
            raise ValueError("Analysis container must be single container, 1D list, or numpy object array.")
        
        self.acs = acs
        self.acs_shape = acs.shape
        self.acs_total_entries = np.prod(acs.shape)
        try:
            self.nchannels, self.data_length = acs.flatten()[0].data_res_arr.shape
        except ValueError:
            self.data_length = acs.flatten()[0].data_res_arr.shape[0]
            self.nchannels = 1

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
        self.shape_sens = shape_sens = ac_tmp.sens_mat.shape[:-1]

        assert np.all(np.asarray(shape_sens) < 5)  # makes sure it is not length of data
        # reset so that all data are linear in memory
        num_machines = 1 if gpus is None else len(gpus)

        split_num = int(np.ceil(self.acs_total_entries / num_machines))
        split_inds = np.arange(split_num, self.acs_total_entries, split_num)

        self.gpu_splits = gpu_splits = np.split(np.arange(self.acs_total_entries), split_inds)

        self.gpu_map = np.zeros(self.acs_total_entries, dtype=int)
        self.split_map = np.zeros(self.acs_total_entries, dtype=int)
        self.linear_data_arr = []
        self.linear_psd_arr = []
        for i, split in enumerate(gpu_splits):
            self.gpu_map[split] = gpus[i]
            self.split_map[split] = i
            self.linear_data_arr.append(xp.zeros(self.data_length * self.nchannels * len(split), dtype=complex))
            self.linear_psd_arr.append(xp.zeros(self.data_length * np.prod(shape_sens) * len(split), dtype=float))

        self.num_acs = num_acs = len(acs.flatten())
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

        for i, ac in enumerate(self.acs.flatten()):
            gpu = self.gpu_map[i]
            split = self.split_map[i]
            if self.gpus is not None:
                self.xp.cuda.runtime.setDevice(gpu)

            # following assumes everything is ordered purposefully
            intra_split_index = np.where(self.gpu_splits[split] == i)[0][0]
            start_index = intra_split_index * (self.nchannels * self.data_length)
            end_index = (intra_split_index + 1) * (self.nchannels * self.data_length)
            self.linear_data_arr[split][start_index:end_index] = self.xp.asarray(ac.data_res_arr.flatten())
            ac.data_res_arr._data_res_arr = self.linear_data_arr[split][start_index:end_index].reshape(self.nchannels, self.data_length)
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
            start_index = intra_split_index * (np.prod(self.shape_sens) * self.data_length)
            end_index = (intra_split_index + 1) * (np.prod(self.shape_sens) * self.data_length)
            self.linear_psd_arr[split][start_index:end_index] = self.xp.asarray(ac.sens_mat.invC.flatten())
            ac.sens_mat.invC = self.linear_psd_arr[split][start_index:end_index].reshape(self.shape_sens + (self.data_length,))

            # TODO: add check to make sure changes are made inline along with protections
            if self.gpus is not None:
                self.xp.get_default_memory_pool().free_all_blocks()

        if self.gpus is not None:
            self.xp.cuda.runtime.setDevice(main_gpu)

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
                output = np.zeros(self.acs_total_entries, dtype=type(_tmp_output))

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

    def signal_operation(self, sign, templates, data_index=None, start_index=None):

        assert isinstance(templates, np.ndarray) or isinstance(templates, cp.ndarray)
        
        if templates.ndim == 2:
            _nchannels, template_length = templates.shape
            num_templates = 1
        elif templates.ndim == 3:
            num_templates, _nchannels, template_length = templates.shape

        if data_index is None:
            assert num_templates == self.acs_total_entries
            data_index = np.arange(num_templates)
        else: 
            assert data_index.max() < self.acs_total_entries

        if start_index is None:
            start_index = np.zeros_like(data_index)
        else:
            assert len(start_index) == num_templates
            assert start_index < self.data_length

        assert len(start_index) == len(data_index)
        for i, (di, si) in enumerate(zip(data_index, start_index)):
            self.acs[di].data_res_arr[:, si:si+template_length] += sign * templates[i]
        
    def add_signal_to_residual(self, *args, **kwargs):
        self.signal_operation(-1, *args, **kwargs)

    def remove_signal_from_residual(self, *args, **kwargs):
        self.signal_operation(+1, *args, **kwargs)

    @property
    def data_shaped(self):
        out = []
        for i, tmp in enumerate(self.linear_data_arr):
            if self.gpus is not None:
                self.xp.cuda.runtime.setDevice(self.gpus[i])
            out.append(tmp.reshape(-1, self.nchannels, self.data_length)) 
        return out
        
    @property
    def psd_shaped(self):
        out = []
        for i, tmp in enumerate(self.linear_psd_arr):
            if self.gpus is not None:
                self.xp.cuda.runtime.setDevice(self.gpus[i])
            out.append(tmp.reshape(-1, self.nchannels, self.data_length)) 
        return out

    