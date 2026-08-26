"""Container for LISA data, residuals, or templates and their basis settings."""

from __future__ import annotations

import math
import warnings
from abc import ABC
from typing import Any, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate, signal

try:
    import cupy as cp

except (ModuleNotFoundError, ImportError):
    import numpy as cp

import dataclasses

from . import detector as lisa_models
from . import domains
from .sensitivity import SensitivityMatrix
from .stochastic import FittedHyperbolicTangentGalacticForeground, StochasticContribution
from .utils.constants import *
from .utils.utility import AET, asnumpy, get_array_module


class DataResidualArray:
    """Deprecated container — kept as a thin shim around :class:`~lisatools.domains.DomainBase`.

    All capabilities (slicing, add/subtract signals, ``f_arr``/``start_freq_ind``/
    ``layer_df`` accessors, ``char_strain``, ``loglog``, etc.) have moved onto
    :class:`~lisatools.domains.DomainBase` and its concrete subclasses
    (:class:`~lisatools.domains.FDSignal`, :class:`~lisatools.domains.WDMSignal`,
    :class:`~lisatools.domains.TDSignal`, :class:`~lisatools.domains.STFTSignal`).
    Pass those directly to :class:`~lisatools.analysiscontainer.AnalysisContainer`.

    This class is retained as a transparent wrapper so existing code keeps
    working; a :class:`DeprecationWarning` is emitted at construction to flag
    call sites that should migrate. The constructor still accepts a raw NumPy /
    CuPy / JAX array (plus ``input_signal_domain``), an existing
    :class:`~lisatools.domains.DomainBase`, or another :class:`DataResidualArray`.

    Args:
        data_res_in: Data, residual, or template input.
        signal_domain: Target domain (defaults to ``input_signal_domain``).
        input_signal_domain: Domain of the raw input array (required for raw arrays).
        window: Optional window applied during a domain transform.
        **kwargs: For future compatibility.
    """

    def __init__(
        self,
        data_res_in: List[np.ndarray] | np.ndarray | DataResidualArray,
        signal_domain: Optional[domains.DomainSettingsBase] = None,
        input_signal_domain: Optional[domains.DomainSettingsBase] = None,
        window: np.ndarray | cp.ndarray | None = None,
        **kwargs: dict,
    ) -> None:
        warnings.warn(
            "DataResidualArray is deprecated. Pass a DomainBase child "
            "(FDSignal / WDMSignal / TDSignal / STFTSignal) directly to "
            "AnalysisContainer and downstream APIs instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.data_res_in_orig_input = data_res_in
        if isinstance(data_res_in, DataResidualArray):
            for key, item in data_res_in.__dict__.items():
                setattr(self, key, item)

        else:
            if not isinstance(data_res_in, domains.DomainBase):
                # Accept numpy, cupy, or jax arrays (the array-module
                # dispatch via ``get_array_module`` handles all three).
                try:
                    xp = get_array_module(data_res_in)
                except ValueError as e:
                    raise AssertionError(
                        "data_res_in must be a numpy / cupy / jax array, "
                        "DomainBase, or DataResidualArray."
                    ) from e
                data_res_in = xp.atleast_2d(data_res_in)
                if input_signal_domain is None:
                    raise ValueError(
                        "If inputing a basic array, must put in the input_signal_domain argument."
                    )
                assert isinstance(input_signal_domain, domains.DomainSettingsBase)
                data_res_in = input_signal_domain.associated_class(data_res_in, input_signal_domain)

            input_signal_domain = data_res_in.settings

            if signal_domain is None:
                if isinstance(input_signal_domain, domains.TDSettings):
                    # default for TD for now is in FD
                    # The rfft of a real length-N signal has N // 2 + 1 bins.
                    # This used to allocate a length-N ones array and run a
                    # full FFT on it just to read that number off the shape.
                    Nf = input_signal_domain.N // 2 + 1
                    df = 1. / (input_signal_domain.N * input_signal_domain.dt)
                    signal_domain = domains.FDSettings(Nf, df, force_backend=input_signal_domain.force_backend)

                else:
                    # default is same domain
                    signal_domain = input_signal_domain
             
            if signal_domain == input_signal_domain:
                self.data_res_arr = data_res_in
            else:
                self.data_res_arr = data_res_in.transform(signal_domain, window=window)

            self.nchannels = self.data_res_arr.nchannels
            self.nbatch = self.data_res_arr.nbatch
            self.is_batched = self.data_res_arr.is_batched
            self.data_shape = self.data_res_arr.settings.basis_shape

        self.init_kwargs = dict(
            signal_domain=signal_domain,
            input_signal_domain=input_signal_domain,
            **kwargs,
        )

    @property
    def init_kwargs(self) -> dict:
        """Initial keyword arguments passed at construction (``signal_domain``, ``input_signal_domain``, ...)."""
        return self._init_kwargs

    @init_kwargs.setter
    def init_kwargs(self, init_kwargs: dict) -> None:
        """Set initial kwargs."""
        self._init_kwargs = init_kwargs

    # def _check_inputs(
    #     self,
    #     dt: Optional[float] = None,
    #     f_arr: Optional[np.ndarray] = None,
    #     df: Optional[float] = None,
    # ):
    #     number_of_none = 0

    #     number_of_none += 1 if dt is None else 0
    #     number_of_none += 1 if f_arr is None else 0
    #     number_of_none += 1 if df is None else 0

    #     if number_of_none == 3:
    #         raise ValueError("Must provide either df, dt, or f_arr.")

    #     elif number_of_none == 1:
    #         raise ValueError(
    #             "Can only provide one of dt, f_arr, or df. Not more than one."
    #         )
    #     self.init_kwargs = dict(dt=dt, f_arr=f_arr, df=df)
    
    @property
    def start_freq_ind(self):
        """Index of the first frequency bin relative to a uniform ``df`` grid (or ``None``).

        For WDM data this is the first active *frequency-layer* index
        (``settings.ind_min_f``); for FD data it is the first bin index
        ``f_arr[0] / df``. Returns ``None`` when no uniform grid exists.
        """
        if isinstance(self.settings, domains.WDMSettings):
            return int(self.settings.ind_min_f)
        if self._df is not None:
            return int(self._f_arr[0] / self._df)
        return None

    @property
    def start_freq_layer_ind(self):
        """First active WDM frequency-layer index (``ind_min_f``); ``None`` if not WDM."""
        if isinstance(self.settings, domains.WDMSettings):
            return int(self.settings.ind_min_f)
        return None

    @property
    def start_time_layer_ind(self):
        """First active WDM time-layer index (``ind_min_t``); ``None`` if not WDM."""
        if isinstance(self.settings, domains.WDMSettings):
            return int(self.settings.ind_min_t)
        return None

    @property
    def layer_df(self):
        """WDM layer frequency spacing; ``None`` if not WDM."""
        if isinstance(self.settings, domains.WDMSettings):
            return float(self.settings.layer_df)
        return None

    @property
    def layer_dt(self):
        """WDM layer time spacing; ``None`` if not WDM."""
        if isinstance(self.settings, domains.WDMSettings):
            return float(self.settings.layer_dt)
        return None

    def _store_time_and_frequency_information(
        self,
        dt: Optional[float] = None,
        f_arr: Optional[np.ndarray] = None,
        df: Optional[float] = None,
    ):
        """Populate ``_dt``, ``_df``, ``_Tobs``, ``_fmax``, and ``_f_arr`` from one of ``dt``/``f_arr``/``df``."""
        if dt is not None:
            self._dt = dt
            self._Tobs = self.data_length * dt
            self._df = 1 / self._Tobs
            self._fmax = 1 / (2 * dt)
            xp = get_array_module(self.data_res_arr)
            self._f_arr = xp.asarray(np.fft.rfftfreq(self.data_length, dt))

            # transform data
            tmp = xp.fft.rfft(self.data_res_arr, axis=-1) * self._dt
            del self._data_res_arr
            self._data_res_arr = tmp
            self.data_length = self._data_res_arr.shape[-1]

        # THIS NEEDS TO BE BEFORE df CHECK
        elif f_arr is not None:
            self._f_arr = f_arr
            self._fmax = f_arr.max()
            # constant spacing
            if np.allclose(np.diff(f_arr), np.diff(f_arr)[0]):
                if df is None:
                    raise ValueError(
                        "When providing evenly spaced f_arr, need to also provide df to avoid numerical issues."
                    )
                # TODO: fix this up in the docs
                self._df = df  # np.diff(f_arr)[0].item()

                if f_arr[0] == 0.0:
                    # could be fft because of constant spacing and f_arr[0] == 0.0
                    self._Tobs = 1 / self._df
                    self._dt = 1 / (2 * self._fmax)

                else:
                    # cannot be fft basis
                    self._Tobs = None
                    self._dt = None

            else:
                self._df = None
                self._Tobs = None
                self._dt = None

        elif df is not None:
            self._df = df
            self._Tobs = 1 / self._df
            self._fmax = (self.data_length - 1) * df
            self._dt = 1 / (2 * self._fmax)
            self._f_arr = np.arange(0.0, self._fmax, self._df)

        if len(self.f_arr) != self.data_length:
            raise ValueError(
                "Entered or determined f_arr does not have the same length as the data channel inputs."
            )

    @property
    def settings(self) -> domains.DomainSettingsBase:
        """Basis settings of the data residual array."""
        return self.data_res_arr.settings

    @property
    def fmax(self):
        """Maximum frequency."""
        return self._fmax

    @property
    def f_arr(self):
        """Frequency array.

        For WDM data this is the active per-layer frequency grid
        (``settings.f_arr``); for FD data it is the precomputed FD grid
        stored at construction.
        """
        if isinstance(self.settings, domains.WDMSettings):
            return self.settings.f_arr
        return self._f_arr

    @property
    def dt(self):
        """Time step in seconds."""
        if self._dt is None:
            raise ValueError("dt cannot be determined from this f_arr input.")

        return self._dt

    @property
    def Tobs(self):
        """Observation time in seconds"""
        if self._Tobs is None:
            raise ValueError("Tobs cannot be determined from this f_arr input.")

        return self._Tobs

    @property
    def df(self):
        """Delta f.

        For WDM data this is the layer frequency spacing
        (``settings.layer_df``); for FD data it is the precomputed FD
        bin spacing.
        """
        if isinstance(self.settings, domains.WDMSettings):
            return float(self.settings.layer_df)
        if self._df is None:
            raise ValueError("df cannot be determined from this f_arr input.")
        return self._df

    @property
    def frequency_arr(self) -> np.ndarray:
        """Frequency array"""
        return self.settings.f_arr

    @property
    def data_res_arr(self) -> DomainBase:
        """Actual data residual array"""
        return self._data_res_arr

    @data_res_arr.setter
    def data_res_arr(self, data_res_arr: List[np.ndarray] | np.ndarray) -> None:
        """Set ``data_res_arr``."""
        self._data_res_arr = data_res_arr

    def __getitem__(self, index: tuple) -> np.ndarray:
        """Index this class directly in ``self.data_res_arr``."""
        return self.data_res_arr[index]

    def __setitem__(self, index: tuple, value: float | np.ndarray) -> np.ndarray:
        """Index this class directly in ``self.data_res_arr``."""
        self.data_res_arr[index] = value

    @property
    def ndim(self) -> int:
        """Number of dimensions in the `data_res_arr`."""
        return self.data_res_arr.ndim

    def flatten(self) -> np.ndarray:
        """Flatten the ``data_res_arr``."""
        return self.data_res_arr[:].flatten()

    @property
    def shape(self) -> tuple:
        """Shape of ``data_res_arr``."""
        return self.data_res_arr.shape

    def loglog(
        self,
        ax: Optional[List[plt.Axes] | plt.Axes] = None,
        fig: Optional[plt.Figure] = None,
        inds: Optional[List[int] | int] = None,
        char_strain: Optional[bool] = False,
        **kwargs: dict,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Produce a log-log plot of the data.

        Args:
            ax: Matplotlib Axes objects to add plots. Either a list of Axes objects or a single Axes object.
            fig: Matplotlib figure object.
            inds: Integer index to select out which data to add to a single access.
                A list can be provided if ax is a list. They must be the same length.
            char_strain: If ``True`` return plot in characteristic strain representation.
            **kwargs: Keyword arguments to be passed to ``loglog`` function in matplotlib.

        Returns:
            Matplotlib figure and axes objects in a 2-tuple.


        """

        assert isinstance(self.data_res_arr.data_res_arr.settings, domains.FDSettings)
        if ax is None and fig is None:
            nrows = 1
            ncols = self.shape[0]

            fig, ax = plt.subplots(nrows, ncols, sharex=True, sharey=True)
            ax = ax.ravel()
            inds_list = range(len(ax))

        elif ax is not None:
            if isinstance(ax, list):
                assert len(ax) == np.prod(self.shape[:-1])
                if inds is None:
                    inds_list = list(np.arange(np.prod(self.shape[:-1])))
                else:
                    assert isinstance(inds, list) and len(inds) == len(ax)
                    inds_list = inds

            elif isinstance(ax, plt.Axes):
                assert inds is not None and (isinstance(inds, tuple) or isinstance(inds, int))
                ax = [ax]
                inds_list = [inds]

        elif fig is not None:
            raise NotImplementedError

        _f_arr = asnumpy(self.settings.f_arr)
        _data_res_arr = asnumpy(self.data_res_arr.arr)

        for i, ax_tmp in zip(inds_list, ax):
            plot_in = np.abs(_data_res_arr[i])
            if char_strain:
                plot_in *= _f_arr
            ax_tmp.loglog(_f_arr, plot_in, **kwargs)

        return (fig, ax)

    @property
    def char_strain(self) -> np.ndarray:
        """Characteristic strain representation of the data."""
        return np.sqrt(self.f_arr) * np.abs(self.data_res_arr)
