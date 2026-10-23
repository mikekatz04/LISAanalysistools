"""Deprecated container for LISA data, residuals, or templates and their basis settings.

The module holds :class:`DataResidualArray`, the legacy wrapper that tagged an array with its
domain. Its capabilities now live on :class:`~lisatools.domains.DomainBase` and its concrete
subclasses (:class:`~lisatools.domains.FDSignal`, :class:`~lisatools.domains.WDMSignal`,
:class:`~lisatools.domains.TDSignal`, :class:`~lisatools.domains.STFTSignal`), which new code
passes directly to :class:`~lisatools.analysiscontainer.AnalysisContainer`. The global-fit
pipeline does not build :class:`DataResidualArray` objects; it only accepts one in
:func:`lisatools.globalfit.postprocessing.save_residuals`.
"""

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
    """Deprecated wrapper that tags a data/residual/template array with its domain.

    All capabilities (slicing, add/subtract signals, ``f_arr`` / ``start_freq_ind`` /
    ``layer_df`` accessors, ``char_strain``, ``loglog``, ...) have moved onto
    :class:`~lisatools.domains.DomainBase` and its concrete subclasses
    (:class:`~lisatools.domains.FDSignal`, :class:`~lisatools.domains.WDMSignal`,
    :class:`~lisatools.domains.TDSignal`, :class:`~lisatools.domains.STFTSignal`).
    Pass those directly to :class:`~lisatools.analysiscontainer.AnalysisContainer`
    (which still unwraps a :class:`DataResidualArray` transparently).

    The class is kept so existing code keeps working; every construction emits a
    :class:`DeprecationWarning`. The input is resolved as follows:

    - another :class:`DataResidualArray`: its instance attributes are copied as they are
      (no transform, ``signal_domain`` / ``window`` are not applied);
    - a :class:`~lisatools.domains.DomainBase`: used as the input signal;
    - a raw numpy / cupy / jax array: promoted with ``atleast_2d`` and wrapped in
      ``input_signal_domain.associated_class``.

    In the last two cases the signal is transformed to ``signal_domain`` when that differs
    from the input domain. With ``signal_domain=None`` the input domain is kept, except for
    :class:`~lisatools.domains.TDSettings` input, which goes to the rfft grid
    ``FDSettings(N // 2 + 1, 1 / (N * dt))`` (same ``force_backend``).

    Args:
        data_res_in: Data, residual, or template: a raw array (leading axes
            ``(nbatch?, nchannels)``, trailing axes the basis), a
            :class:`~lisatools.domains.DomainBase`, or another :class:`DataResidualArray`.
        signal_domain: Target domain. Default ``None``: keep the input domain (TD input:
            the matching FD grid, see above).
        input_signal_domain: Domain of a raw input array. Required for raw arrays; for a
            :class:`~lisatools.domains.DomainBase` input its ``settings`` are used instead.
        window: Window forwarded to the input signal's ``transform`` when a transform is
            needed. Default ``None``.
        **kwargs: Stored in :attr:`init_kwargs`; otherwise unused.

    Raises:
        AssertionError: ``data_res_in`` is not an array, DomainBase, or DataResidualArray, or
            ``input_signal_domain`` is not a :class:`~lisatools.domains.DomainSettingsBase`.
        ValueError: A raw array is given without ``input_signal_domain``.

    Attributes:
        nchannels (int): Number of channels of :attr:`data_res_arr`.
        nbatch (int | None): Batch size of :attr:`data_res_arr` (``None`` if unbatched).
        is_batched (bool): Whether :attr:`data_res_arr` has a batch axis.
        data_shape (tuple): ``data_res_arr.settings.basis_shape``.
        data_res_in_orig_input: The ``data_res_in`` object as passed (for a
            :class:`DataResidualArray` input, that object's own ``data_res_in_orig_input``,
            copied with its other attributes).

    Warning:
        The legacy FD grid accessors :attr:`fmax`, :attr:`dt` and :attr:`Tobs`, and the non-WDM
        branches of :attr:`f_arr`, :attr:`df` and :attr:`start_freq_ind`, read private
        attributes (``_f_arr``, ``_df``, ``_dt``, ``_Tobs``, ``_fmax``) that no constructor path
        sets any more, so they raise :class:`AttributeError`. Use :attr:`frequency_arr`,
        :attr:`settings`, or the accessors of :attr:`data_res_arr` instead.

    Example:
        Preferred form, then the legacy wrapper around the same array:

        .. code-block:: python

            import numpy as np
            from lisatools.domains import FDSettings, FDSignal
            from lisatools.datacontainer import DataResidualArray

            settings = FDSettings(N=1025, df=1e-5)
            arr = np.zeros((3, 1025), dtype=complex)   # (nchannels, N)
            sig = FDSignal(arr, settings)              # pass this to AnalysisContainer

            legacy = DataResidualArray(arr, input_signal_domain=settings)  # DeprecationWarning
            legacy.data_res_arr                        # the FDSignal it wraps
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
        """dict: Construction keywords (``signal_domain``, ``input_signal_domain``, kwargs)."""
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
        """int | None: Index of the first frequency bin relative to a uniform ``df`` grid.

        For WDM data this is the first active *frequency-layer* index
        (``settings.ind_min_f``). The non-WDM branch computes ``f_arr[0] / df`` from the
        legacy private ``_df`` / ``_f_arr`` attributes, which no constructor path sets, so it
        raises :class:`AttributeError`; use ``data_res_arr.start_freq_ind`` instead.
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

    # TODO check: keep/remove/document -- dead code: no callers; reads self.data_length, never set
    def _store_time_and_frequency_information(
        self,
        dt: Optional[float] = None,
        f_arr: Optional[np.ndarray] = None,
        df: Optional[float] = None,
    ):
        """Set ``_dt``, ``_df``, ``_Tobs``, ``_fmax``, ``_f_arr`` from one of the inputs."""
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
        """float: Maximum frequency (Hz) from the legacy private ``_fmax``.

        No constructor path sets ``_fmax`` any more, so this raises :class:`AttributeError`;
        use ``data_res_arr.fmax``.
        """
        return self._fmax

    @property
    def f_arr(self):
        """np.ndarray: Frequency array (Hz).

        For WDM data this is the active per-layer frequency grid (``settings.f_arr``). The
        non-WDM branch returns the legacy private ``_f_arr``, which no constructor path sets,
        so it raises :class:`AttributeError`; use :attr:`frequency_arr` instead.
        """
        if isinstance(self.settings, domains.WDMSettings):
            return self.settings.f_arr
        return self._f_arr

    @property
    def dt(self):
        """float: Time step (s) from the legacy private ``_dt``.

        Raises :class:`ValueError` when ``_dt`` is ``None``. No constructor path sets ``_dt``
        any more, so in practice this raises :class:`AttributeError`.
        """
        if self._dt is None:
            raise ValueError("dt cannot be determined from this f_arr input.")

        return self._dt

    @property
    def Tobs(self):
        """float: Observation time (s) from the legacy private ``_Tobs``.

        Raises :class:`ValueError` when ``_Tobs`` is ``None``. No constructor path sets
        ``_Tobs`` any more, so in practice this raises :class:`AttributeError`.
        """
        if self._Tobs is None:
            raise ValueError("Tobs cannot be determined from this f_arr input.")

        return self._Tobs

    @property
    def df(self):
        """float: Frequency spacing (Hz).

        For WDM data this is the layer frequency spacing (``settings.layer_df``). The non-WDM
        branch returns the legacy private ``_df`` (:class:`ValueError` if it is ``None``),
        which no constructor path sets, so it raises :class:`AttributeError`; for FD data use
        ``data_res_arr.df``.
        """
        if isinstance(self.settings, domains.WDMSettings):
            return float(self.settings.layer_df)
        if self._df is None:
            raise ValueError("df cannot be determined from this f_arr input.")
        return self._df

    @property
    def frequency_arr(self) -> np.ndarray:
        """Frequency array of the wrapped signal's basis (``settings.f_arr``), in Hz."""
        return self.settings.f_arr

    @property
    def data_res_arr(self) -> DomainBase:
        """The wrapped (possibly transformed) :class:`~lisatools.domains.DomainBase` signal."""
        return self._data_res_arr

    @data_res_arr.setter
    def data_res_arr(self, data_res_arr: List[np.ndarray] | np.ndarray) -> None:
        """Set ``data_res_arr``."""
        self._data_res_arr = data_res_arr

    def __getitem__(self, index: tuple) -> np.ndarray:
        """Index the underlying array of :attr:`data_res_arr` (``data_res_arr.arr[index]``)."""
        return self.data_res_arr[index]

    def __setitem__(self, index: tuple, value: float | np.ndarray) -> np.ndarray:
        """Assign into the underlying array of :attr:`data_res_arr` in place."""
        self.data_res_arr[index] = value

    @property
    def ndim(self) -> int:
        """Number of dimensions of the underlying array of :attr:`data_res_arr`."""
        return self.data_res_arr.ndim

    def flatten(self) -> np.ndarray:
        """Return the underlying array of :attr:`data_res_arr` flattened to 1D (a copy)."""
        return self.data_res_arr[:].flatten()

    @property
    def shape(self) -> tuple:
        """Shape of the underlying array of :attr:`data_res_arr`."""
        return self.data_res_arr.shape

    def loglog(
        self,
        ax: Optional[List[plt.Axes] | plt.Axes] = None,
        fig: Optional[plt.Figure] = None,
        inds: Optional[List[int] | int] = None,
        char_strain: Optional[bool] = False,
        **kwargs: dict,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot ``|arr|`` against frequency on log-log axes (FD data only).

        Prefer :meth:`lisatools.domains.DomainBase.loglog` on the wrapped signal.

        Args:
            ax: Axes to draw on: a list (length ``prod(shape[:-1])``, one per channel row)
                or a single Axes. Default ``None`` creates a figure with one panel per entry
                of ``shape[0]``.
            fig: Must be ``None``; passing a figure without ``ax`` raises
                :class:`NotImplementedError`.
            inds: Rows of the array to draw. With a list ``ax``: a list of the same length
                (default all rows); with a single Axes: required, an ``int`` or ``tuple``.
            char_strain: If ``True``, plot ``f * |arr|`` instead of ``|arr|``. Default ``False``.
            **kwargs: Forwarded to ``Axes.loglog``.

        Returns:
            tuple[plt.Figure | None, list[plt.Axes]]: The figure (``None`` when ``ax`` was
            passed) and the axes drawn on.

        Raises:
            AssertionError: The wrapped signal is not FD, or ``ax`` / ``inds`` are inconsistent.
            NotImplementedError: ``fig`` is given without ``ax``.
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
        """np.ndarray: Legacy ``sqrt(f) * |data_res_arr|``.

        This reads :attr:`f_arr` (raises for non-WDM data) and applies ``np.abs`` to the
        wrapped :class:`~lisatools.domains.DomainBase` object rather than its array; use
        ``data_res_arr.char_strain`` instead.
        """
        return np.sqrt(self.f_arr) * np.abs(self.data_res_arr)
