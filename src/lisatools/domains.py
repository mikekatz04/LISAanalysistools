"""Time, frequency, and time-frequency domain settings and array wrappers.

Every data, residual and template array in lisatools travels as a signal object that knows
its basis. This module holds the two halves of that pairing:

* settings classes (the basis grid, no data): :class:`DomainSettingsBase` and its concrete
  subclasses :class:`TDSettings` (time samples), :class:`FDSettings` (one-sided rFFT bins),
  :class:`STFTSettings` (short-time FFT segments) and :class:`WDMSettings` (the WDM wavelet
  grid of ``Nf`` frequency layers by ``Nt`` time pixels), plus the time-coarsened
  :class:`CoarseWDMSettings` model view;
* signal classes (an array plus its settings): :class:`DomainBase` and :class:`TDSignal`,
  :class:`FDSignal`, :class:`STFTSignal`, :class:`WDMSignal`. Each signal class inherits
  its settings class, so ``signal.df`` / ``signal.layer_df`` work directly, and
  :meth:`DomainBase.transform` moves a signal to any other domain
  (``td.transform(wdm_settings)``).

Array layout: the trailing axes of a signal array are the ACTIVE basis
(``settings.basis_shape_active``: ``(N,)`` TD, ``(N_active,)`` FD, ``(NT, NF_active)``
STFT, ``(Nf_active, Nt_active)`` WDM); at most two leading axes are allowed, read as
``(nchannels,)`` or ``(nbatch, nchannels)``.

Domains are never communicated by string: code dispatches on ``isinstance`` of the
settings class, and a run configuration takes either a settings instance or a
``(times, dt, force_backend)`` factory built by :meth:`FDSettings.make_factory`,
:meth:`STFTSettings.make_factory` or :meth:`WDMSettings.make_factory`. The 9-month and
12-month global-fit runs use the WDM domain (the stock factory sets ``min_freq`` /
``max_freq`` from the run band and ``min_time`` / ``max_time`` from the edge crop).

Also here: the time-domain grid helpers :func:`pad_td_signal` and
:func:`place_td_signal_on_grid`, :func:`get_stft_settings`, the WDM pixel lookup table
:class:`WDMLookupTable` (SOBBH lookup, EMRI direct-to-WDM and GB lookup likelihoods),
:class:`DomainBaseArray` and :func:`get_available_domains`.

Environment knobs read at import: ``LISATOOLS_WDM_CLEAR_FFT_CACHE`` (restore the cuFFT
plan-cache wipe inside the FD -> WDM transform; default off), ``WDM_BUILD_KIND``
(:attr:`WDMLookupTable.BUILD_KIND`, default ``"n_ref_only"``) and ``WDM_LOOKUP_INTERP``
(:attr:`WDMLookupTable.INTERP_METHOD`, default ``"linear"``).
"""

from __future__ import annotations

import copy
import math
import warnings
from abc import ABC
from typing import Any, Tuple, Optional, List
import os
import pickle
import math
import numpy as np

try:
    from cupyx.scipy import interpolate as interpolate_gpu
except (ModuleNotFoundError, ImportError):
    pass
from scipy import interpolate as interpolate_cpu
from scipy import signal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from scipy import interpolate, signal, special

try:
    import cupy as cp

    CUPY_AVAILABLE = True

except (ModuleNotFoundError, ImportError):
    import numpy as cp  # type: ignore

from . import detector as lisa_models
from .utils.utility import AET, asnumpy, get_array_module
from .utils.constants import *
from .utils.parallelbase import LISAToolsParallelModule
from . import cutils
import dataclasses
import logging

logger = logging.getLogger("lisatools.domains")

# Byte budget for the per-block complex intermediates inside
# ``FDSignal.wdmtransform``. Frequency layers are independent, so the transform
# walks them in blocks sized to keep the two temporary (outer..., block, Nt)
# complex arrays under this. 256 MiB is small enough to keep a large batch off
# the memory ceiling and large enough that the block loop stays short.
# Override per-settings with ``wdm_layer_budget_bytes`` / ``wdm_layer_chunk``.
_WDM_LAYER_BUDGET = 256 * 1024 * 1024

# Opt-in restoration of the per-transform cuFFT plan-cache wipe (see
# wdmtransform). Default OFF: it cost 172 ms/call and the cache never hit.
_WDM_CLEAR_FFT_CACHE = os.environ.get("LISATOOLS_WDM_CLEAR_FFT_CACHE", "0") not in ("0", "", "false", "False")


@dataclasses.dataclass
class DomainSettingsBase(LISAToolsParallelModule):
    """Base class for domain settings (TD, FD, STFT, WDM).

    Carries the ``force_backend`` selector that decides whether NumPy or CuPy
    is used for arrays in this domain (one settings instance = one backend).
    Concrete subclasses provide the basis description used everywhere else:

    * ``args`` / ``kwargs``: the reconstruction recipe,
      ``type(s)(*s.args, **s.kwargs)`` rebuilds an equal settings object (signal
      classes and per-device replicas rely on it);
    * ``basis_shape`` / ``basis_shape_active``: the full and the active
      (band-limited) basis shape;
    * ``differential_component``: the factor multiplying the inner-product sum;
    * ``total_terms``: the number of active basis elements;
    * ``associated_class``: the paired :class:`DomainBase` signal class;
    * :meth:`get_slice`: settings for a sub-box of the active basis.

    Args:
        force_backend: Backend name passed to
            :class:`~lisatools.utils.parallelbase.LISAToolsParallelModule`
            (e.g. ``"cpu"``, ``"cuda12x"``), or a backend object. ``None`` lets the
            parallel module pick its default.

    Attributes:
        force_backend: The value passed at construction.
    """

    force_backend: str = None

    def __init__(self, force_backend: str = None):
        self.force_backend = force_backend
        LISAToolsParallelModule.__init__(self, force_backend=force_backend)

    @classmethod
    def supported_backends(cls):
        """Return the list of backend names this settings class supports.

        Returns ``lisatools_*`` names (``lisatools_`` + each entry of
        ``GPU_RECOMMENDED()``). The old ``fastlisaresponse_*`` names were retired
        with the responselisa pybind11 module (Phase 3E).

        Returns:
            list[str]: Backend names, e.g. ``["lisatools_cuda12x", ..., "lisatools_cpu"]``.
        """
        return ["lisatools_" + _tmp for _tmp in cls.GPU_RECOMMENDED()]

    def get_slice(self, index: tuple) -> DomainSettingsBase:
        """Return a new settings object describing a sliced view of this domain.

        Implemented by the subclasses that support slicing (TD, STFT, WDM).

        Args:
            index: Domain-specific slice specification (see the subclass).

        Raises:
            NotImplementedError: Always, on the base class.
        """
        raise NotImplementedError("get_slice needs to be implemented for this signal type.")

    def evaluate_time_modulation(self, modulation):
        """Evaluate a time modulation on this domain's active time grid.

        Domain subclasses with a reduced time representation
        (:class:`CoarseWDMSettings`) override this hook to preserve cell averages
        instead of sampling at representative times.

        Args:
            modulation (callable or array-like): A function of time, called as
                ``modulation(self.t_arr)``, or an already evaluated array.

        Returns:
            ``xp.ndarray``: The evaluated modulation on this settings' backend.
        """
        if callable(modulation):
            return modulation(self.t_arr)
        return self.xp.asarray(modulation)

    # Whether the basis coefficients that enter the Gaussian likelihood are
    # *real* variables (TD, WDM) rather than the complex one-sided (FD/STFT
    # Whittle) coefficients. Subclasses flip this single flag; the numeric
    # convention lives in one place, :attr:`logdet_factor`.
    _real_basis: bool = False

    @property
    def logdet_factor(self) -> float:
        r"""Factor multiplying ``sum(log det C)`` in the Gaussian noise term.

        One factor for every domain, resolved from a single characteristic
        (:attr:`_real_basis`): ``0.5`` for real-basis domains (TD, real WDM),
        where each real coefficient contributes ``-0.5 log det C`` to the
        Gaussian density, and ``1.0`` for the complex one-sided (FD / Whittle)
        convention used historically throughout lisatools. Without the ``0.5``
        the noise-parameter posterior peaks at ``C_true / 2`` (validated
        empirically via a global covariance-scale scan).
        """
        return 0.5 if self._real_basis else 1.0


class DomainBase:
    """Base wrapper for an array tagged with its domain settings.

    The array's last ``len(basis_shape_active)`` dimensions correspond to the
    domain's active basis (frequency bins, time samples, time-frequency cells);
    the leading dimensions are read as ``(nchannels,)`` (unbatched) or
    ``(nbatch, nchannels)`` (batched). An array with no leading dimension gets a
    channel axis of length 1 added.

    Not used on its own: the concrete signal classes (:class:`TDSignal`,
    :class:`FDSignal`, :class:`STFTSignal`, :class:`WDMSignal`) combine it with
    their settings class and supply ``settings``. A :class:`DomainBase` child is
    also the data / residual / template object held by
    :class:`~lisatools.analysiscontainer.AnalysisContainer` (the former
    ``DataResidualArray`` capabilities live here).

    Args:
        arr: The underlying NumPy or CuPy array.

    Raises:
        ValueError: If ``arr`` has more than two dimensions in front of the basis.

    Attributes:
        outer_shape (tuple): The leading (non-basis) shape, ``(nchannels,)`` or
            ``(nbatch, nchannels)``.
        nchannels (int): Number of channels.
    """

    def __init__(self, arr):
        self.arr = arr

    @staticmethod
    def get(x: np.ndarray) -> np.ndarray:
        """Return ``x`` as a NumPy array (calls ``.get()`` for CuPy arrays).

        Thin wrapper around :func:`lisatools.utils.utility.asnumpy` kept
        for API stability — older call sites use ``self.get(x)``.
        """
        return asnumpy(x)

    @property
    def arr(self) -> np.ndarray | cp.ndarray:
        """Underlying NumPy or CuPy array, shape ``outer_shape + basis_shape_active``.

        Setting it re-derives ``outer_shape``, ``nchannels`` and ``nbatch`` (and
        adds a channel axis to an array that has none).
        """
        return self._arr

    @arr.setter
    def arr(self, arr: np.ndarray | cp.ndarray):
        """Set the underlying array and infer batch / channel dimensions."""
        if self.backend.uses_cupy:
            # deferred import: cupyx is only present alongside cupy
            import cupyx.scipy.signal as cupyx_signal

            self._stft = cupyx_signal.stft
        else:
            self._stft = signal.stft

        assert len(arr.shape) >= len(self.basis_shape_active)
        if len(arr.shape) == len(self.basis_shape_active):
            arr = arr[None, ...]

        self.outer_shape = arr.shape[: -len(self.basis_shape_active)]
        if len(self.outer_shape) > 2:
            raise ValueError(
                f"Too many dimensions outside of basis_shape. "
                f"Expected at most 2 outer dims (batch, channels), got {len(self.outer_shape)}: {self.outer_shape}."
            )
        elif len(self.outer_shape) == 2:
            # batched: shape is (nbatch, nchannels, *basis_shape)
            self._nbatch = self.outer_shape[0]
            self.nchannels = self.outer_shape[1]
        else:
            # unbatched: shape is (nchannels, *basis_shape)
            self._nbatch = None
            self.nchannels = self.outer_shape[0]
        self._arr = arr

    def __getitem__(self, index):
        return self.arr[index]

    def __setitem__(self, index, value):
        self.arr[index] = value

    @property
    def is_batched(self) -> bool:
        """Whether this signal has a batch dimension."""
        return self._nbatch is not None

    @property
    def nbatch(self) -> int | None:
        """Number of batch elements, or None if unbatched."""
        return self._nbatch

    def flatten(self) -> np.ndarray | cp.ndarray:
        """Return :attr:`arr` flattened to 1D."""
        return self.arr.flatten()

    def transform(self, new_domain: DomainSettingsBase, window: np.ndarray | cp.ndarray = None):
        """Transform this signal into ``new_domain`` (implemented by each signal class).

        Args:
            new_domain: Target settings; dispatch is on its class.
            window: Optional window, interpreted by the subclass.

        Raises:
            NotImplementedError: Always, on the base class.
        """
        raise NotImplementedError("Transform needs to be implemented for this signal type.")

    def with_backend(self, force_backend) -> "DomainBase":
        """Return this signal on another backend (one host<->device transfer).

        Same signal class and identical settings, reconstructed with
        ``force_backend``; the data array is uploaded (``xp.asarray``) or
        downloaded (:func:`asnumpy`) exactly once. Array entries of the settings
        recipe (e.g. the WDM ``window`` / ``omega``) are brought to the host
        before the rebuild.

        This is the sanctioned crossing point between backends: everything
        derived from the returned object lives on the new backend, keeping
        the sprint rule "one instance = one backend" intact while still
        letting CPU-loaded data (disk reads are host-side by nature) enter a
        GPU pipeline.

        Args:
            force_backend (str or backend object): A backend name like ``"cpu"`` /
                ``"cuda12x"`` or a backend object.

        Returns:
            DomainBase: A new signal of the same class, or ``self`` when the
            backend already matches.
        """
        # Cheap short-circuit when handed a Backend object (the common case,
        # e.g. ``other_settings.backend``): compare array modules directly
        # instead of paying a settings reconstruction.
        if getattr(force_backend, "xp", None) is self.xp:
            return self

        settings = self.settings
        # settings.kwargs is the reconstruction recipe for the settings
        # object (`type(settings)(*settings.args, **settings.kwargs)`). Most
        # entries are scalars/None, but some settings classes carry ARRAYS in
        # the recipe (WDMSettings' `window` and `omega`), and those arrays
        # live on THIS signal's current backend. Rebuilding the settings for
        # the other backend with a CuPy array still inside the recipe would
        # plant device memory inside a CPU settings object, which only blows
        # up later, on the first NumPy op that touches it. So real device
        # arrays are downloaded to host (asnumpy -> .get()) before the
        # rebuild. The CPU->GPU direction needs no mirror-image upload: the
        # GPU settings ctor receives host arrays exactly as it would from a
        # user constructing it directly, and coerces with its own xp where
        # needed. The `cp is not np` guard covers CuPy-less installs, where
        # this module aliases cp to numpy at import and the isinstance test
        # would otherwise match every plain NumPy array.
        new_kwargs = {
            k: asnumpy(v) if isinstance(v, cp.ndarray) and cp is not np else v
            for k, v in settings.kwargs.items()
        }
        new_kwargs["force_backend"] = force_backend
        new_settings = type(settings)(*settings.args, **new_kwargs)
        if new_settings.xp is self.xp:
            return self
        if new_settings.backend.uses_cupy:
            new_arr = new_settings.xp.asarray(self.arr)
        else:
            new_arr = asnumpy(self.arr)
        return type(self)(new_arr, new_settings)

    def _coerce_transform_backend(
        self, new_domain: DomainSettingsBase, window: np.ndarray | cp.ndarray = None
    ) -> tuple["DomainBase", np.ndarray | cp.ndarray]:
        """Move this signal (and ``window``) to ``new_domain``'s backend if needed.

        Layered transforms thread the signal's backend through every
        intermediate domain built under the hood, so a target settings object
        on a different backend used to raise here (a mixed NumPy/CuPy chain
        fails deep inside otherwise). Since 2026-07: the transform target
        defines where the work happens — when the backends differ, the signal
        and window are transferred ONCE at entry via :meth:`with_backend` and
        the whole chain (intermediates + output) runs on the target backend.
        That is also the memory-minimal placement: a single copy of the input
        crosses the device boundary; nothing mid-chain ever transfers.
        """
        if new_domain.xp is self.xp:
            return self, window
        logger.info(
            "transform target is on backend '%s' but signal is on '%s': "
            "transferring the signal (%s, %.0f MB) to the target backend.",
            new_domain.backend_name,
            self.backend_name,
            "x".join(map(str, self.arr.shape)),
            self.arr.nbytes / 1e6,
        )
        signal_on_target = self.with_backend(new_domain.backend)
        if window is not None and not isinstance(window, str):
            if new_domain.backend.uses_cupy:
                window = new_domain.xp.asarray(window)
            else:
                window = asnumpy(window)
        return signal_on_target, window

    @property
    def shape(self) -> tuple:
        """Shape of :attr:`arr`."""
        return self.arr.shape
    
    def __add__(self, other: DomainBase):
        if not isinstance(other, DomainBase):
            raise ValueError("Can only add another DomainBase object.")
        if self.settings != other.settings:
            raise ValueError("Can only add another DomainBase object with the same settings.")
        return self.__class__(self.arr + other.arr, settings=self.settings)
    
    def __sub__(self, other: DomainBase):
        if not isinstance(other, DomainBase):
            raise ValueError("Can only subtract another DomainBase object.")
        if self.settings != other.settings:
            raise ValueError("Can only subtract another DomainBase object with the same settings.")
        return self.__class__(self.arr - other.arr, settings=self.settings) 
    
    def __mul__(self, other: float):
        if not isinstance(other, (int, float)):
            raise ValueError("Can only multiply by a scalar.")
        return self.__class__(self.arr * other, settings=self.settings)

    def __truediv__(self, other: float):
        if not isinstance(other, (int, float)):
            raise ValueError("Can only divide by a scalar.")
        return self.__class__(self.arr / other, settings=self.settings)

    def get_array_slice(self, index: tuple) -> DomainBase:
        """Return a sliced signal with both the array and its settings restricted to ``index``.

        The array is indexed as ``arr[(Ellipsis,) + index]`` (the basis axes) and the
        settings through ``settings.get_slice(index)``, so the two stay in step.

        Args:
            index: Basis-axis slices, relative to the active box (for WDM
                ``(f_slice, t_slice)``; see the settings class's ``get_slice``).

        Returns:
            DomainBase: A new signal of the associated class.
        """
        new_arr = self.arr[(Ellipsis,) + index]
        new_settings = self.settings.get_slice(index)
        return self.settings.associated_class(new_arr, new_settings)

    # ------------------------------------------------------------------
    # Data-residual / signal-handling capabilities
    #
    # Folded in from the (deprecated) DataResidualArray so a DomainBase
    # child is sufficient on its own as the data/residual/template object
    # held by AnalysisContainer. The legacy
    # ``data_res_arr.data_res_arr.<attr>`` chain keeps working because
    # :attr:`data_res_arr` returns ``self`` (so chained access reaches the
    # same object).
    # ------------------------------------------------------------------

    @property
    def data_res_arr(self) -> "DomainBase":
        """Self-reference for legacy ``.data_res_arr.<attr>`` access chains."""
        return self

    @property
    def ndim(self) -> int:
        """Number of dimensions of the underlying array."""
        return self.arr.ndim

    @property
    def data_shape(self) -> tuple:
        """Active basis shape (``settings.basis_shape_active``)."""
        return self.settings.basis_shape_active

    # ---- frequency / time grid passthroughs --------------------------

    @property
    def f_arr(self) -> np.ndarray:
        """Frequency array of the underlying basis (FD bins / WDM layer centres).

        Note: concrete signal classes (FDSignal, WDMSignal, STFTSignal) inherit
        ``f_arr`` directly from their settings class, which is earlier in the
        MRO. This base-class definition is the fallback and supplies a uniform
        access path so ``isinstance(x, DomainBase)`` checks can rely on it.
        """
        return self.settings.f_arr

    @property
    def frequency_arr(self) -> np.ndarray:
        """Alias for :attr:`f_arr` (kept from DataResidualArray for API parity)."""
        return self.settings.f_arr

    # NB: ``df``, ``dt``, ``Tobs``, ``layer_df``, ``layer_dt`` are deliberately
    # NOT added here. Those names are *set* as instance attributes by the
    # ``*Settings`` constructors (TDSettings.__init__: self.dt = ..., etc.).
    # Concrete *Signal classes inherit ``Settings`` and ``DomainBase``; if
    # DomainBase declared these as @property descriptors, the descriptor would
    # win in the MRO and the Settings constructor would raise AttributeError.
    # Access them domain-specifically: ``td_signal.dt``, ``fd_signal.df``,
    # ``wdm_signal.layer_df`` -- all already available via settings
    # inheritance. For a uniform handle, use ``settings.differential_component``.

    @property
    def fmax(self) -> Optional[float]:
        """Highest frequency in the active band; ``None`` if not applicable."""
        if not hasattr(self.settings, "f_arr"):
            return None
        arr = self.settings.f_arr
        return float(arr.max())

    # ---- WDM-specific layer-index passthroughs -----------------------

    @property
    def start_freq_ind(self) -> Optional[int]:
        """First frequency-bin / frequency-layer index relative to a uniform grid."""
        if isinstance(self.settings, WDMSettings):
            return int(self.settings.ind_min_f)
        if isinstance(self.settings, FDSettings):
            return int(self.settings.ind_min)
        return None

    @property
    def start_freq_layer_ind(self) -> Optional[int]:
        """First active WDM frequency-layer index (``ind_min_f``); ``None`` if not WDM."""
        if isinstance(self.settings, WDMSettings):
            return int(self.settings.ind_min_f)
        return None

    @property
    def start_time_layer_ind(self) -> Optional[int]:
        """First active WDM time-layer index (``ind_min_t``); ``None`` if not WDM."""
        if isinstance(self.settings, WDMSettings):
            return int(self.settings.ind_min_t)
        return None

    # NB: ``layer_df`` and ``layer_dt`` are also deliberately NOT added here.
    # ``WDMSettings.__init__`` does ``self.layer_df = ...`` / ``self.layer_dt = ...``
    # as plain instance-attribute assignments, and ``WDMSignal`` inherits both
    # ``WDMSettings`` and ``DomainBase`` -- a property on ``DomainBase`` without
    # a setter would win in the MRO and raise during ``WDMSettings.__init__``.
    # Access ``wdm_signal.layer_df`` / ``wdm_signal.layer_dt`` directly via the
    # inherited ``WDMSettings`` attribute. For non-WDM domains those names
    # simply do not exist (raise ``AttributeError``), which is the historical
    # behaviour (the old ``DataResidualArray`` returned ``None``).

    # ---- characteristic strain / plotting ----------------------------

    @property
    def char_strain(self) -> np.ndarray:
        """Characteristic strain representation ``sqrt(f) * |arr|`` (FD only)."""
        return self.xp.sqrt(self.f_arr) * self.xp.abs(self.arr)

    def loglog(
        self,
        ax: Optional["list[plt.Axes] | plt.Axes"] = None,
        fig: Optional["plt.Figure"] = None,
        inds: Optional["list[int] | int"] = None,
        char_strain: bool = False,
        **kwargs: dict,
    ):
        """Produce a log-log plot of ``|arr|`` against frequency (FD signals only).

        Args:
            ax: Matplotlib Axes (or list of Axes) to draw on. If ``None``, a new
                figure with one panel per entry of the first array axis is created
                and every channel is drawn (``inds`` is ignored).
            fig: Matplotlib Figure. Returned unchanged when ``ax`` is given; replaced
                by the new figure when ``ax`` is ``None``.
            inds: Channel indices to draw, one per Axes. With a single Axes the
                default is channel 0; with a list of Axes the default is one channel
                per Axes in order.
            char_strain: If ``True``, plot ``f * |arr|`` instead of ``|arr|``.
            **kwargs: Forwarded to ``ax.loglog``.

        Returns:
            tuple: ``(fig, ax)``.

        Raises:
            AssertionError: If the signal is not in the FD domain.
        """
        assert isinstance(self.settings, FDSettings), \
            "loglog is only defined on FD-domain signals."
        if ax is None:
            fig, ax = plt.subplots(1, self.shape[0], sharex=True, sharey=True)
            ax = np.atleast_1d(ax).ravel()
            inds_list = list(range(len(ax)))
        elif isinstance(ax, plt.Axes):
            ax = [ax]
            inds_list = [0] if inds is None else ([inds] if isinstance(inds, int) else list(inds))
        else:
            inds_list = list(range(len(ax))) if inds is None else list(inds)

        _f = asnumpy(self.f_arr)
        _arr = asnumpy(self.arr)
        for i, ax_i in zip(inds_list, ax):
            plot_in = np.abs(_arr[i])
            if char_strain:
                plot_in *= _f
            ax_i.loglog(_f, plot_in, **kwargs)
        return (fig, ax)

    # ---- domain-aware add / subtract of a template -------------------

    def add_signal(
        self, template: "DomainBase", sign: int = +1, copy: bool = False,
    ) -> "DomainBase":
        """Add (``sign=+1``) or subtract (``sign=-1``) ``template`` from this array.

        Domain-aware: FD, TD and STFT templates are added over the overlap of
        their frequency / time range with ``self`` (``df`` / ``dt`` must match; no
        overlap warns and does nothing). A WDM template must be on the same
        wavelet grid and its active box must equal ``self``'s or lie inside it
        (added into that sub-box, see :meth:`WDMSettings.sub_box_slices`).

        Args:
            template: A :class:`DomainBase` of the same settings class as ``self``.
            sign: +1 to add, -1 to subtract. Use :meth:`subtract_signal` for clarity.
            copy: If ``True``, operate on a copy and return it. If ``False``
                (default), modify ``self`` in place and return ``self``.

        Returns:
            DomainBase: ``self`` (or the copy when ``copy=True``).

        Raises:
            ValueError: If ``sign`` is not +1 / -1, the settings classes differ,
                the grid spacings differ, or a WDM template box does not fit.
            TypeError: If ``template`` is not a :class:`DomainBase`.
        """
        if sign not in (+1, -1):
            raise ValueError("sign must be +1 or -1")
        if not isinstance(template, DomainBase):
            raise TypeError(
                f"template must be a DomainBase; got {type(template).__name__}"
            )

        target = self
        if copy:
            target = self.__class__(self.arr.copy(), self.settings)

        t_settings = template.settings
        s_settings = target.settings
        if type(t_settings) is not type(s_settings):
            raise ValueError(
                f"Template domain ({type(t_settings).__name__}) does not match "
                f"data domain ({type(s_settings).__name__})."
            )

        if isinstance(s_settings, FDSettings):
            _apply_fd_add(target, sign, template.arr, t_settings)
        elif isinstance(s_settings, TDSettings):
            _apply_td_add(target, sign, template.arr, t_settings)
        elif isinstance(s_settings, STFTSettings):
            _apply_stft_add(target, sign, template.arr, t_settings)
        elif isinstance(s_settings, WDMSettings):
            _apply_wdm_add(target, sign, template.arr, t_settings)
        else:
            raise ValueError(
                f"add_signal is not implemented for domain {type(s_settings).__name__}."
            )
        return target

    def subtract_signal(self, template: "DomainBase", copy: bool = False) -> "DomainBase":
        """Subtract ``template``: ``add_signal(template, sign=-1, copy=copy)``.

        In place unless ``copy=True`` (see :meth:`add_signal`).
        """
        return self.add_signal(template, sign=-1, copy=copy)


# ----------------------------------------------------------------------
# Domain-specific add helpers used by DomainBase.add_signal. Kept at
# module scope so DomainBase doesn't need a giant per-domain dispatch
# body. They each modify ``target.arr`` in place.
# ----------------------------------------------------------------------


def _apply_fd_add(target, sign, template_arr, template_settings):
    """Add ``sign * template_arr`` (FD) to ``target.arr`` over the f-band overlap."""
    data_settings = target.settings
    if not np.isclose(data_settings.df, template_settings.df):
        raise ValueError(
            f"Data df ({data_settings.df}) and template df "
            f"({template_settings.df}) must match for FD add_signal."
        )

    data_f0 = float(data_settings.f_arr[0])
    tmpl_f0 = float(template_settings.f_arr[0])
    data_f1 = float(data_settings.f_arr[-1])
    tmpl_f1 = float(template_settings.f_arr[-1])

    f_lo = max(data_f0, tmpl_f0)
    f_hi = min(data_f1, tmpl_f1)
    if f_lo > f_hi:
        warnings.warn("FD template and data frequency ranges do not overlap. Skipping.")
        return

    f_start_data = int(round((f_lo - data_f0) / data_settings.df))
    f_end_data = int(round((f_hi - data_f0) / data_settings.df)) + 1
    f_start_tmpl = int(round((f_lo - tmpl_f0) / template_settings.df))
    f_end_tmpl = f_start_tmpl + (f_end_data - f_start_data)

    target.arr[..., f_start_data:f_end_data] += (
        sign * template_arr[..., f_start_tmpl:f_end_tmpl]
    )


def _apply_td_add(target, sign, template_arr, template_settings):
    """Add ``sign * template_arr`` (TD) to ``target.arr`` over the time-range overlap."""
    data_settings = target.settings
    if not np.isclose(data_settings.dt, template_settings.dt):
        raise ValueError(
            f"Data dt ({data_settings.dt}) and template dt "
            f"({template_settings.dt}) must match for TD add_signal."
        )

    time_offset = int(round((template_settings.t0 - data_settings.t0) / data_settings.dt))
    t_start_data = max(0, time_offset)
    t_end_data = min(data_settings.N, time_offset + template_settings.N)
    if t_start_data >= t_end_data:
        warnings.warn("TD template and data time ranges do not overlap. Skipping.")
        return

    tmpl_t_start = t_start_data - time_offset
    tmpl_t_end = t_end_data - time_offset
    target.arr[..., t_start_data:t_end_data] += (
        sign * template_arr[..., tmpl_t_start:tmpl_t_end]
    )


def _apply_stft_add(target, sign, template_arr, template_settings):
    """Add ``sign * template_arr`` (STFT) to ``target.arr`` over the (t, f) overlap."""
    data_settings = target.settings
    if not np.isclose(data_settings.df, template_settings.df):
        raise ValueError(
            f"Data df ({data_settings.df}) and template df "
            f"({template_settings.df}) must match for STFT add_signal."
        )
    if data_settings.NF != template_settings.NF:
        raise ValueError(
            f"Data NF ({data_settings.NF}) and template NF "
            f"({template_settings.NF}) must match for STFT add_signal."
        )

    time_offset = int(round((template_settings.t0 - data_settings.t0) / data_settings.dt))
    t_start_data = max(0, time_offset)
    t_end_data = min(data_settings.NT, time_offset + template_settings.NT)
    if t_start_data >= t_end_data:
        warnings.warn("STFT template and data time ranges do not overlap. Skipping.")
        return

    tmpl_t_start = t_start_data - time_offset
    tmpl_t_end = t_end_data - time_offset

    data_f0 = float(data_settings.f_arr[0])
    tmpl_f0 = float(template_settings.f_arr[0])
    data_f1 = float(data_settings.f_arr[-1])
    tmpl_f1 = float(template_settings.f_arr[-1])
    f_lo = max(data_f0, tmpl_f0)
    f_hi = min(data_f1, tmpl_f1)
    if f_lo > f_hi:
        warnings.warn("STFT template and data frequency ranges do not overlap. Skipping.")
        return

    f_start_data = int(round((f_lo - data_f0) / data_settings.df))
    f_end_data = int(round((f_hi - data_f0) / data_settings.df)) + 1
    f_start_tmpl = int(round((f_lo - tmpl_f0) / template_settings.df))
    f_end_tmpl = f_start_tmpl + (f_end_data - f_start_data)

    target.arr[..., t_start_data:t_end_data, f_start_data:f_end_data] += (
        sign * template_arr[..., tmpl_t_start:tmpl_t_end, f_start_tmpl:f_end_tmpl]
    )


def _apply_wdm_add(target, sign, template_arr, template_settings):
    """Add ``sign * template_arr`` (WDM) to ``target.arr``.

    Same active box: plain in-place add (unchanged behaviour). A template whose
    active box is a SUB-BOX of the target's -- same grid (``Nf``, ``Nt``,
    ``dt``, ``t0``), ``ind_min/max`` inside -- is added into that box (the MBH
    batched windowed templates, 2026-09-29). Anything else raises.
    """
    ts = template_settings
    same_origin = (
        int(ts.ind_min_f) == int(target.ind_min_f)
        and int(ts.ind_min_t) == int(target.ind_min_t)
    )
    if target.arr.shape[-2:] == template_arr.shape[-2:] and same_origin:
        target.arr[...] += sign * template_arr
        return
    f_sl, t_sl = target.sub_box_slices(ts)
    if tuple(template_arr.shape[-2:]) != (f_sl.stop - f_sl.start, t_sl.stop - t_sl.start):
        raise ValueError(
            f"WDM add_signal: template array {tuple(template_arr.shape[-2:])} does "
            f"not match its box {(f_sl.stop - f_sl.start, t_sl.stop - t_sl.start)}."
        )
    target.arr[..., f_sl, t_sl] += sign * template_arr


class TDSettings(DomainSettingsBase):
    """Time-domain basis settings: ``N`` samples at ``t0 + k * dt``.

    The time domain has no band masking: the active basis is the full grid,
    ``basis_shape_active == (N,)``. Its coefficients are real, so
    :attr:`~DomainSettingsBase.logdet_factor` is ``0.5``.

    Args:
        N: Number of time samples.
        dt: Sample spacing in seconds.
        t0: Start time in seconds. Defaults to ``0.0``.
        **kwargs: Forwarded to :class:`DomainSettingsBase` (e.g. ``force_backend``).

    Attributes:
        N (int): Number of samples.
        dt (float): Sample spacing (s).
        t0 (float): Time of sample 0 (s).

    Example:
        >>> import numpy as np
        >>> from lisatools.domains import TDSettings, TDSignal
        >>> settings = TDSettings(N=1024, dt=5.0, force_backend="cpu")
        >>> td = TDSignal(np.zeros((3, 1024)), settings)
        >>> td.shape
        (3, 1024)
        >>> td.fft().shape
        (3, 513)
    """

    # Real Gaussian basis -> logdet_factor = 0.5 (see DomainSettingsBase).
    _real_basis: bool = True

    N: int
    dt: float
    t0: float = 0.0

    def __init__(self, N: int, dt: float, t0: float = 0.0, **kwargs):
        self.t0 = t0
        self.N = N
        self.dt = dt
        # TODO: include kwargs in kwargs property?
        super().__init__(**kwargs)

    @staticmethod
    def get_associated_class():
        """Return the :class:`DomainBase` subclass that pairs with these settings."""
        return TDSignal

    @property
    def associated_class(self):
        """The :class:`DomainBase` subclass that pairs with these settings."""
        return self.get_associated_class()

    @property
    def kwargs(self) -> dict:
        """Keyword arguments needed to reconstruct this settings object."""
        return dict(t0=self.t0, force_backend=self.backend)

    @property
    def args(self) -> tuple:
        """Positional arguments needed to reconstruct this settings object."""
        return (self.N, self.dt)

    @property
    def t_arr(self) -> np.ndarray:
        """Array of sample times: ``t0 + arange(N) * dt``."""
        return self.t0 + self.xp.arange(self.N) * self.dt

    @property
    def basis_shape(self) -> tuple:
        """Total basis shape ``(N,)``."""
        return (self.N,)

    @property
    def basis_shape_active(self) -> tuple:
        """Active basis shape (same as :attr:`basis_shape` for TD)."""
        # TODO: adjust this
        return (self.N,)

    def __repr__(self) -> str:
        return (
            f"TDSettings(t0={self.t0}, N={self.N}, dt={self.dt}, "
            f"backend={self.backend_name.split('_')[-1]})"
        )

    def __eq__(self, value):
        if not isinstance(value, TDSettings):
            return False

        return (
            (value.N == self.N) and (value.dt == self.dt) and (self.xp.isclose(value.t0, self.t0))
        )

    @property
    def differential_component(self) -> float:
        """Differential element used in inner-product summations (``dt`` in TD)."""
        return self.dt

    @property
    def total_terms(self) -> int:
        """Total number of basis elements (``N``)."""
        return self.N

    def compute_slice_indices(self, tmin: float, tmax: float) -> slice:
        """Return a ``slice`` along the time axis covering ``[tmin, tmax)``.

        Both edges are rounded to the nearest sample: ``round((t - t0) / dt)``.

        Args:
            tmin: Start time (s), ``>= t0``.
            tmax: End time (s), ``<= t0 + N * dt``.

        Returns:
            slice: ``slice(start_idx, end_idx)``.

        Raises:
            ValueError: If ``tmin`` or ``tmax`` lies outside the grid.
        """
        if tmin < self.t0:
            raise ValueError("tmin must be greater than or equal to t0.")
        if tmax > self.t0 + self.N * self.dt:
            raise ValueError("tmax must be less than or equal to t0 + N*dt.")

        start_idx = int(self.xp.round((tmin - self.t0) / self.dt))
        end_idx = int(self.xp.round((tmax - self.t0) / self.dt))

        return slice(start_idx, end_idx)

    def get_slice(self, index=None, tmin: float = None, tmax: float = None) -> TDSettings:
        """Return TD settings for a slice of the time samples.

        The new grid starts at ``t0 + start * dt`` and has
        ``(stop - start) // step`` samples. When ``index`` is given it is used and
        ``tmin`` / ``tmax`` are ignored; otherwise the slice comes from
        :meth:`compute_slice_indices`.

        Args:
            index (slice, optional): Slice of the time axis, e.g. ``slice(0, 10)``.
            tmin: Start time (s); used with ``tmax`` when ``index`` is ``None``.
            tmax: End time (s); used with ``tmin`` when ``index`` is ``None``.

        Returns:
            TDSettings: Settings for the sliced grid.

        Raises:
            ValueError: If ``index`` is ``None`` and ``tmin`` / ``tmax`` are not both given.
            TypeError: If ``index`` is not a ``slice``.

        Warning:
            The return statement passes ``(new_t0, new_N, self.dt)`` positionally to
            ``TDSettings(N, dt, t0)``, so the returned object currently has
            ``N = new_t0``, ``dt = new_N`` and ``t0 = self.dt``.
        """

        if index is None:
            if tmin is None or tmax is None:
                raise ValueError("If index is not provided, both tmin and tmax must be provided.")
            index = self.compute_slice_indices(tmin, tmax)

        if not isinstance(index, slice):
            raise TypeError("index must be a slice object")

        start, stop, step = index.indices(self.N)
        new_N = (stop - start) // step
        new_t0 = self.t0 + start * self.dt

        return TDSettings(new_t0, new_N, self.dt, force_backend=self.backend)


class TDSignal(DomainBase, TDSettings):
    """Time-domain array wrapper paired with :class:`TDSettings`.

    Inherits :class:`TDSettings`, so ``td.N``, ``td.dt``, ``td.t0`` and ``td.t_arr``
    are available directly. :meth:`transform` reaches every other domain
    (:meth:`fft`, :meth:`stft`, :meth:`wdmtransform`).

    Args:
        arr: NumPy or CuPy array of shape ``(N,)``, ``(nchannels, N)`` or
            ``(nbatch, nchannels, N)``; a 1-D array gets a channel axis.
        settings: :class:`TDSettings` describing the time grid.

    Example:
        >>> import numpy as np
        >>> from lisatools.domains import TDSettings, TDSignal, WDMSettings
        >>> N, dt = 64 * 512, 5.0
        >>> td = TDSignal(np.zeros((1, N)), TDSettings(N=N, dt=dt, force_backend="cpu"))
        >>> wdm = td.transform(WDMSettings(Nf=64, Nt=512, dt=dt, force_backend="cpu"))
        >>> wdm.arr.shape
        (1, 64, 512)
    """

    def __init__(self, arr, settings: TDSettings):
        TDSettings.__init__(self, *settings.args, **settings.kwargs)
        DomainBase.__init__(self, arr)

    @property
    def settings(self) -> TDSettings:
        """A fresh :class:`TDSettings` matching this signal's time grid."""
        return TDSettings(*self.args, **self.kwargs)

    def __repr__(self) -> str:
        return (
            f"TDSignal(t0={self.t0}, N={self.N}, dt={self.dt}, "
            f"backend={self.backend_name.split('_')[-1]})"
        )
    
    def fft(self, settings=None, window=None):
        """Forward FFT of the (optionally windowed) time-domain signal.

        Computes ``rfft(arr * window) * dt`` along the last axis (continuous
        Fourier-transform normalization), then keeps the target's active band.

        Args:
            settings (FDSettings, optional): Target grid. Its FFT length
                ``round(1 / (settings.df * dt))`` must equal ``N`` (the caller pads
                beforehand; nothing is padded here). If ``None``, a full-band
                :class:`FDSettings` with ``N // 2 + 1`` bins and ``df = 1 / (N * dt)``
                is built on this signal's backend.
            window (array-like, optional): Window multiplied into the signal before
                the FFT, broadcastable to ``arr``. Default all ones.

        Returns:
            FDSignal: Shape ``outer_shape + (N_active,)`` of the target settings.

        Raises:
            AssertionError: If ``settings`` is not an :class:`FDSettings` or its FFT
                length does not match ``N``.
        """
        if window is None:
            window = self.xp.ones(self.arr.shape, dtype=float)

        if settings is not None:
            assert isinstance(settings, FDSettings)
            # Integer-length check against the target df (robust vs the old
            # float df == df comparison); the caller must pre-pad the signal
            # so the FFT length matches -- no silent padding here.
            n_fft = round(1 / (settings.df * self.dt))
            assert self.N == n_fft, (
                f"Signal length ({self.N}) != target FFT length ({n_fft}). "
                f"Caller must pre-pad the signal."
            )
            fd_arr = self.xp.fft.rfft(self.arr * window, axis=-1) * self.dt
            fd_settings = settings

        else:
            fd_arr = self.xp.fft.rfft(self.arr * window, axis=-1) * self.dt
            df = 1 / (self.N * self.dt)
            fd_settings = FDSettings(fd_arr.shape[-1], df, force_backend=self.backend)

        return FDSignal(fd_arr[..., fd_settings.active_slice], fd_settings)

    def stft(self, settings=None, window=None):
        """Short-time Fourier transform of the time-domain signal.

        Splits the first ``NT * nperseg`` samples (``nperseg = settings.dt / dt``)
        into ``NT`` contiguous, non-overlapping segments and takes
        ``dt * rfft(window * segment)`` of each.

        Args:
            settings (STFTSettings): Segment grid (required). ``settings.dt`` must be
                an integer multiple of this signal's ``dt``.
            window (array-like, optional): Per-segment window of length ``nperseg``.
                Default all ones.

        Returns:
            STFTSignal: Shape ``outer_shape + (NT, NF_active)``, sliced to
            ``settings.active_slice``.

        Raises:
            ValueError: If ``settings`` is ``None`` or the signal has fewer than
                ``NT * nperseg`` samples.
            AssertionError: If ``settings`` is not an :class:`STFTSettings` or the
                segment length is not a multiple of ``dt``.
        """
        if settings is None:
            raise ValueError("Must provide STFTSettings for stft transform.")
        assert isinstance(settings, STFTSettings)
        big_dt = settings.dt

        # Validate that big_dt is an integer multiple of self.dt
        nperseg = settings.get_nperseg(self.dt)

        if window is None:
            window =self.xp.ones(nperseg, dtype=float)

        # Use NT from settings directly to ensure consistency
        Nsegments = settings.NT
        Nsegments_available = self.N // nperseg

        # Check we have enough data
        required_samples = Nsegments * nperseg

        if self.N < required_samples:
            raise ValueError(
                f"Not enough data: have {self.N} samples, need {required_samples} for {Nsegments} segments"
            )

        if Nsegments > Nsegments_available:
            # Need to pad
            pad_samples = required_samples - self.N

            # Pad with zeros at the end
            # pad_width format: ((before_axis0, after_axis0), (before_axis1, after_axis1), ...)
            pad_width = [(0, 0)] * len(self.outer_shape) + [(0, pad_samples)]
            _arr =self.xp.pad(self.arr, pad_width, mode="constant", constant_values=0)
        else:
            # Truncate to use only what we need
            _arr = self.arr[..., :required_samples]

        stft_arr = self.dt *self.xp.fft.rfft(
            window[None, :] * _arr.reshape(self.outer_shape + (Nsegments, nperseg)),
            axis=-1,
        )

        return STFTSignal(stft_arr[..., settings.active_slice], settings)  # (nchannels, NT, NF)

    def wdmtransform(self, settings=None, window=None):
        """Transform to the WDM wavelet basis: :meth:`fft` then :meth:`FDSignal.wdmtransform`.

        The signal must have exactly ``settings.N = Nf * Nt`` samples (no padding
        is done). The WDM analysis window is ``settings.window``; ``window`` here is
        a time-domain window applied before the FFT.

        Args:
            settings (WDMSettings): Target wavelet grid (required).
            window (array-like, optional): Time-domain window, broadcastable to
                ``arr``. Default all ones.

        Returns:
            WDMSignal: Shape ``outer_shape + (Nf_active, Nt_active)``.

        Raises:
            ValueError: If ``settings`` is ``None``.
            AssertionError: If ``settings`` is not a :class:`WDMSettings`.
        """
        if window is None:
            window =self.xp.ones(self.arr.shape, dtype=float)

        if settings is None:
            raise ValueError("Must provide WDMSettings for WDM transform.")
        assert isinstance(settings, WDMSettings)

        # go to frequency domain then wavelets
        return self.fft(settings=None, window=window).transform(settings)

    def transform(self, new_domain: DomainSettingsBase, window: np.ndarray = None):
        """Dispatch to :meth:`fft`, :meth:`stft`, or :meth:`wdmtransform` based on ``new_domain``.

        ``window=None`` is forwarded as-is: each target handles its own
        default (full-length ones for FD/WDM, per-segment ``nperseg`` ones
        for STFT — pre-filling a full-signal window here broke the STFT
        branch, whose window must be segment-length). A ``TDSettings`` target
        returns ``arr * window`` on this signal's own settings. When
        ``new_domain`` lives on another backend the signal is moved there once
        (:meth:`DomainBase.with_backend`) and the whole transform runs on the
        target backend.

        Args:
            new_domain: Target settings (TD, FD, STFT or WDM).
            window (array-like, optional): See above.

        Returns:
            DomainBase: The signal in the target domain.

        Raises:
            ValueError: If ``new_domain`` is not a recognized settings class.
        """
        signal, window = self._coerce_transform_backend(new_domain, window)
        if signal is not self:
            return signal.transform(new_domain, window=window)
        if isinstance(new_domain, TDSettings):
            if window is None:
                window = self.xp.ones(self.arr.shape, dtype=float)
            return self.settings.associated_class(self.arr * window, self.settings)

        elif isinstance(new_domain, FDSettings):
            return self.fft(settings=new_domain, window=window)

        elif isinstance(new_domain, STFTSettings):
            return self.stft(settings=new_domain, window=window)

        elif isinstance(new_domain, WDMSettings):
            return self.wdmtransform(settings=new_domain, window=window)
        else:
            raise ValueError(f"new_domain type is not recognized {type(new_domain)}.")


def pad_td_signal(
    times,
    signals,
    *,
    data_t0: float,
    dt: float,
    align_samples: int,
    target_n: int = None,
) -> Tuple[Any, Any]:
    """Zero-pad time-domain arrays so the start aligns with a data grid.

    Stock grid-alignment utility shared by the waveform wrappers
    (:class:`lisatools.sources.waveformbase.TDWaveformBase`) and the
    settings-file injection builders. Accepts either a single source or
    a batch:

    - Single:  ``times (num_times,)``,        ``signals (..., num_times)``
    - Batched: ``times (num_bin, num_times)``, ``signals (num_bin, ..., num_times)``

    Left-pads with zeros so that the number of samples between the (new)
    ``t0`` and ``data_t0`` is an integer multiple of ``align_samples``.
    For STFT this enforces segment-boundary alignment
    (``align_samples = nperseg``); for FD / WDM pass
    ``align_samples = target_n`` to align the start exactly to
    ``data_t0`` (full-grid placement).

    Then, if ``target_n`` is given, right-pads with zeros so the total
    number of samples reaches ``target_n`` (ensuring the correct ``df``
    after an FFT, or the full ``Nf * Nt`` grid for a WDM transform).

    The signal must start on the ``data_t0 + k * dt`` grid at or after
    ``data_t0``; callers are responsible for snapping each source's time
    grid to integer multiples of ``dt`` relative to ``data_t0`` (see
    ``TDWaveformBase.get_grid_time``). For batched inputs all sources
    must produce the same left padding.

    Args:
        times: Time array, shape ``(num_times,)`` or ``(num_bin, num_times)``.
        signals: Signal array whose trailing axis matches ``times``.
        data_t0: Start time of the data grid in seconds.
        dt: Sample spacing in seconds.
        align_samples: Left-padding granularity. The signal is extended so
            that ``round((signal_t0 - data_t0) / dt)`` becomes divisible by
            this value.
        target_n: If provided, right-pad to at least this many total samples.

    Returns:
        tuple: ``(padded_times, padded_signals)`` with the same leading dimensions
        as the inputs (the inputs themselves when no padding is needed).

    Raises:
        AssertionError: For batched input whose sources need different left padding.
    """
    xp = get_array_module(signals)

    if times.ndim == 1:
        n_to_data_t0 = round((float(times[0]) - data_t0) / dt)
        n_left = n_to_data_t0 % align_samples
    else:
        n_to_data_t0 = xp.rint((times[:, 0] - data_t0) / dt).astype(int)
        n_left_per_bin = n_to_data_t0 % align_samples
        assert xp.all(n_left_per_bin == n_left_per_bin[0]), (
            "Batched pad_td_signal: sources produce different n_left values — "
            "ensure time grids are snapped to the dt grid before padding."
        )
        n_left = int(n_left_per_bin[0])

    N = times.shape[-1]
    n_right = 0
    if target_n is not None:
        new_n = N + n_left
        if new_n < target_n:
            n_right = target_n - new_n

    if n_left == 0 and n_right == 0:
        return times, signals

    # Pad signals on the last (time) axis, preserving all leading dims.
    pad_width = [(0, 0)] * (signals.ndim - 1) + [(n_left, n_right)]
    padded_signals = xp.pad(signals, pad_width, mode="constant", constant_values=0)

    # Extend the time array.
    if times.ndim == 1:
        parts = []
        if n_left > 0:
            parts.append(times[0] - xp.arange(n_left, 0, -1) * dt)
        parts.append(times)
        if n_right > 0:
            parts.append(times[-1] + xp.arange(1, n_right + 1) * dt)
        padded_times = xp.concatenate(parts)
    else:
        parts = []
        if n_left > 0:
            parts.append(times[:, 0:1] - xp.arange(n_left, 0, -1)[None, :] * dt)
        parts.append(times)
        if n_right > 0:
            parts.append(times[:, -1:] + xp.arange(1, n_right + 1)[None, :] * dt)
        padded_times = xp.concatenate(parts, axis=-1)

    return padded_times, padded_signals


def place_td_signal_on_grid(
    signals,
    settings: TDSettings,
    times=None,
) -> TDSignal:
    """Place a time-domain signal onto the full grid described by ``settings``.

    Stock utility for full-grid placement: the output :class:`TDSignal`
    spans exactly ``[settings.t0, settings.t0 + settings.N * settings.dt)``
    — left-padded back to ``settings.t0``, right-padded with zeros to
    ``settings.N`` samples, and clipped at both ends if the input extends
    outside the grid (samples outside the data span are unobserved and
    are dropped). The result is ready for
    :meth:`TDSignal.transform` into any analysis domain (FD / STFT / WDM)
    or for direct summation into an injection data stream.

    Args:
        signals: Signal array of shape ``(..., num_times)``, or
            ``(nbatch, ..., num_times)`` with 2-D ``times``.
        settings: :class:`TDSettings` describing the target grid.
        times: Time array of shape ``(num_times,)`` (or ``(nbatch, num_times)``,
            one row per source, placed one at a time) giving the sample times
            of ``signals``. Must lie on the ``settings.t0 + k * settings.dt``
            grid. If ``None``, the signal is assumed to already start at
            ``settings.t0`` (right-pad / clip only).

    Returns:
        TDSignal: Shape ``signals.shape[:-1] + (settings.N,)`` on the ``settings`` grid.

    Raises:
        ValueError: If a batched ``signals`` and ``times`` disagree in batch size.
        NotImplementedError: If ``times`` has more than two dimensions.
    """
    xp = get_array_module(signals)
    N_target = settings.N
    dt = settings.dt

    if times is None:
        # Already aligned at settings.t0: right-pad or clip to N samples.
        n = signals.shape[-1]
        if n < N_target:
            pad_width = [(0, 0)] * (signals.ndim - 1) + [(0, N_target - n)]
            arr = xp.pad(signals, pad_width, mode="constant", constant_values=0)
        else:
            arr = signals[..., :N_target]
        return TDSignal(arr, settings)

    if times.ndim == 2:
        # Batched: ``times`` is (nbatch, num_times) and ``signals`` is
        # (nbatch, ..., num_times). Sources generally start at different
        # samples -- ``_apply_response`` crops the batch by a single shared
        # ``start_ind`` but each source keeps its own t0 -- so the placement
        # offset is per source and the sources are placed one at a time.
        #
        # This loop is bookkeeping, not compute: it is a slice-assign per
        # source into a preallocated grid, with no waveform or response work
        # in it. The expensive stages upstream (waveform, TDI response) and
        # downstream (WDM transform, inner products) are all genuinely
        # batched, so this stays in the noise.
        nbatch = times.shape[0]
        if signals.shape[0] != nbatch:
            raise ValueError(
                f"Batched place_td_signal_on_grid: signals batch dim "
                f"{signals.shape[0]} != times batch dim {nbatch}."
            )
        out = xp.zeros(signals.shape[:-1] + (N_target,), dtype=signals.dtype)
        for b in range(nbatch):
            out[b] = place_td_signal_on_grid(
                signals[b], settings, times=times[b]
            ).arr
        return TDSignal(out, settings)

    if times.ndim != 1:
        raise NotImplementedError(
            f"place_td_signal_on_grid handles 1D (single source) or 2D "
            f"(batched) time arrays; got ndim={times.ndim}."
        )

    # Drop leading samples before the grid start (unobserved).
    n_clip = max(0, round((settings.t0 - float(times[0])) / dt))
    if n_clip > 0:
        times = times[n_clip:]
        signals = signals[..., n_clip:]
        if times.shape[-1] == 0:
            return TDSignal(
                xp.zeros(signals.shape[:-1] + (N_target,), dtype=signals.dtype),
                settings,
            )

    padded_times, padded_signals = pad_td_signal(
        times,
        signals,
        data_t0=settings.t0,
        dt=dt,
        align_samples=N_target,
        target_n=N_target,
    )
    # Clip any overrun past the grid end (e.g. merger + response buffer).
    return TDSignal(padded_signals[..., :N_target], settings)


class FDSettings(DomainSettingsBase):
    """Frequency-domain basis settings on a uniform one-sided grid.

    The full grid is ``arange(0, N) * df`` (the natural rFFT grid of ``2 * (N - 1)``
    time samples when ``N`` comes from an rFFT). The active band is the bin range
    ``[ind_min, ind_max]``, snapped inward from the requested edges:
    ``ind_min = ceil(min_freq / df)`` (``0`` for ``None``) and
    ``ind_max = int(max_freq / df)`` (floor; ``N - 1`` for ``None``). The snapped
    indices are not clipped to the grid. Signal arrays carry only the active band,
    ``basis_shape_active == (N_active,)``. Coefficients are complex one-sided (FD
    Whittle convention), so :attr:`~DomainSettingsBase.logdet_factor` is ``1.0``.

    Args:
        N: Total number of frequency bins on the underlying ``arange(0, N) * df`` grid.
        df: Frequency spacing in Hz.
        min_freq: Lower edge of the active band (Hz, ``>= 0``). Defaults to ``0.0``.
        max_freq: Upper edge of the active band (Hz, ``>= 0``); if ``None``, the band
            runs to the last bin.
        **kwargs: Forwarded to :class:`DomainSettingsBase` (e.g. ``force_backend``).

    Raises:
        ValueError: If ``min_freq`` or ``max_freq`` is negative.

    Attributes:
        N (int): Total number of bins.
        df (float): Bin spacing (Hz).
        min_freq_input (float or None): ``min_freq`` exactly as given.
        max_freq_input (float or None): ``max_freq`` exactly as given.

    Example:
        >>> from lisatools.domains import FDSettings
        >>> fd = FDSettings(N=1025, df=1e-4, min_freq=1.55e-3, max_freq=2.045e-2,
        ...                 force_backend="cpu")
        >>> fd.ind_min, fd.ind_max
        (16, 204)
        >>> fd.N_active
        189
    """

    N: int
    df: float
    min_freq: Optional[float] = 0.0
    max_freq: Optional[float] = None

    def __init__(
        self,
        N: int,
        df: float,
        min_freq: Optional[float] = 0.0,
        max_freq: Optional[float] = None,
        **kwargs,
    ):
        self.N = N
        self.df = df
        self.min_freq = min_freq
        self.max_freq = max_freq
        super().__init__(**kwargs)

    @staticmethod
    def make_factory(min_freq: Optional[float] = 0.0, max_freq: Optional[float] = None):
        """Build a ``(times, dt, force_backend) -> FDSettings`` factory.

        The factory derives ``N = len(times)//2 + 1`` and ``df = 1/(len(times)*dt)``
        from its inputs so the active band lives on the natural rFFT grid of the
        loaded data. The global-fit engine calls it with keyword arguments once the
        data are loaded. The returned function carries
        ``domain_settings_class = FDSettings`` so a run configuration can dispatch
        on the domain before the factory is called.

        Args:
            min_freq: Lower band edge (Hz) passed to :class:`FDSettings`.
            max_freq: Upper band edge (Hz) passed to :class:`FDSettings`.

        Returns:
            callable: ``factory(times, dt, force_backend) -> FDSettings``.

        Example:
            >>> import numpy as np
            >>> from lisatools.domains import FDSettings
            >>> factory = FDSettings.make_factory(min_freq=1e-4, max_freq=1e-2)
            >>> fd = factory(times=np.arange(2048) * 5.0, dt=5.0, force_backend="cpu")
            >>> fd.N
            1025
        """
        def _factory(times, dt, force_backend):
            Nt = len(times)
            df = 1.0 / (Nt * dt)
            Nf = Nt // 2 + 1
            return FDSettings(
                N=Nf, df=df, min_freq=min_freq, max_freq=max_freq,
                force_backend=force_backend,
            )
        # The class the factory builds, readable BEFORE it is called (a run
        # configured with the factory can dispatch on its domain at config
        # time, e.g. the stock MBH_LIKELIHOOD=auto resolution).
        _factory.domain_settings_class = FDSettings
        return _factory

    @property
    def differential_component(self) -> float:
        """Differential element used in inner-product summations (``df`` in FD)."""
        return self.df
   
    @property
    def min_freq(self) -> float:
        """Requested lower band edge (Hz) or ``None``.

        Setting it snaps :attr:`ind_min` to ``ceil(min_freq / df)`` (``0`` for
        ``None``); negative values raise ``ValueError``.
        """
        return self._min_freq

    @min_freq.setter
    def min_freq(self, value: Optional[float]):
        if value is not None and value < 0:
            raise ValueError("min_freq must be non-negative.")

        # self._min_freq = value
        # set it to the closest frequency bin
        self.min_freq_input = value
        if value is not None:
            self.ind_min = int(np.ceil(value / self.df))
        else:
            self.ind_min = 0
        self._min_freq = value

    @property
    def max_freq(self) -> Optional[float]:
        """Requested upper band edge (Hz) or ``None``.

        Setting it snaps :attr:`ind_max` to ``int(max_freq / df)`` (``N - 1`` for
        ``None``); negative values raise ``ValueError``.
        """
        return self._max_freq

    @max_freq.setter
    def max_freq(self, value: Optional[float]):
        if value is not None and value < 0:
            raise ValueError("max_freq must be non-negative.")

        # self._max_freq = value
        # set it to the closest frequency bin
        self.max_freq_input = value
        if value is not None:
            self.ind_max = int(value / self.df)
        else:
            self.ind_max = (self.N - 1)
        self._max_freq = value

    @property
    def ind_min(self) -> int:
        """First active bin index (set via :attr:`min_freq`; ``None`` gives ``0``)."""
        return self._ind_min
    
    @ind_min.setter
    def ind_min(self, ind_min: int):
        if ind_min is None:
            ind_min = 0
        self._ind_min = ind_min

    @property
    def ind_max(self) -> int:
        """Last active bin index, inclusive (set via :attr:`max_freq`; ``None`` -> ``N - 1``)."""
        return self._ind_max
    
    @ind_max.setter
    def ind_max(self, ind_max: int):
        if ind_max is None:
            ind_max = self.N - 1
        self._ind_max = ind_max

    @staticmethod
    def get_associated_class():
        """Return the :class:`DomainBase` subclass that pairs with these settings."""
        return FDSignal

    @staticmethod
    def get_associated_group():
        """Return :class:`~lisatools.domaincomputation.FDComputationGroup` (lazy import)."""
        from .domaincomputation import FDComputationGroup
        return FDComputationGroup

    @property
    def associated_class(self):
        """The :class:`DomainBase` subclass that pairs with these settings."""
        return self.get_associated_class()

    @property
    def associated_group(self):
        """The batched C++ likelihood group for this domain (see :meth:`get_associated_group`)."""
        return self.get_associated_group()

    @property
    def kwargs(self) -> dict:
        """Keyword arguments needed to reconstruct this settings object."""
        return dict(
            min_freq=self.min_freq,
            max_freq=self.max_freq,
            force_backend=self.backend,
        )

    @property
    def args(self) -> tuple:
        """Positional arguments needed to reconstruct this settings object."""
        return (self.N, self.df)

    @property
    def basis_shape(self) -> tuple:
        """Total basis shape ``(N,)``."""
        return (self.N,)

    @property
    def basis_shape_active(self) -> tuple:
        """Active basis shape (after applying ``min_freq``/``max_freq`` masking)."""
        return (self.N_active,)

    @property
    def f_arr(self) -> np.ndarray:
        """Active-band frequencies ``k * df`` for ``k`` in ``[ind_min, ind_max]`` (Hz)."""
        _all_freqs = self.xp.arange(0, self.N) * self.df

        return _all_freqs[self.active_slice]

    def __repr__(self) -> str:
        return (
            f"FDSettings(N={self.N}, df={self.df}, "
            f"min_freq={self.min_freq}, max_freq={self.max_freq}, "
            f"backend={self.backend_name.split('_')[-1]})"
        )

    def __eq__(self, value):
        if not isinstance(value, FDSettings):
            return False
        return (
            (value.N == self.N)
            and (value.df == self.df)
            and ((value.min_freq is None) or (self.min_freq is None) or (self.xp.isclose(value.min_freq, self.min_freq)))
            and ((value.max_freq is None) or (self.max_freq is None) or (self.xp.isclose(value.max_freq, self.max_freq)))
        )

    @property
    def total_terms(self) -> int:
        """Total number of basis elements in the active band."""
        return self.N_active

    @property
    def active_slice(
        self,
    ) -> slice:
        """Slice along the frequency axis that selects ``[min_freq, max_freq]``."""
        return slice(self.ind_min, self.ind_max + 1)

    @property
    def N_active(self) -> int:
        """Number of frequency bins in the active band."""
        sl = self.active_slice
        return sl.stop - sl.start


# try:
#     from pywavelet.transforms.phi_computer import phitilde_vec_norm
#     from pywavelet.transforms.numpy.forward.from_freq import (
#         transform_wavelet_freq_helper
#     )

# from pywavelet.transforms.numpy.inverse.to_freq import (
#     inverse_wavelet_freq_helper_fast as inverse_wavelet_freq_helper,
# )


class FDSignal(FDSettings, DomainBase):
    """Frequency-domain array wrapper paired with :class:`FDSettings`.

    Inherits :class:`FDSettings`, so ``fd.df``, ``fd.f_arr`` and the band indices
    are available directly. :meth:`transform` reaches TD (:meth:`ifft`) and WDM
    (:meth:`wdmtransform`).

    Args:
        arr: NumPy or CuPy array of shape ``(..., N_active)``. A full-grid
            ``(nchannels, N)`` array is accepted and trimmed to the active band
            (the trim indexes ``arr[:, active_slice]``, i.e. assumes exactly one
            leading axis).
        settings: :class:`FDSettings` describing the frequency grid and band.

    Raises:
        AssertionError: If the last axis is neither ``N_active`` nor ``N``.

    Example:
        >>> import numpy as np
        >>> from lisatools.domains import FDSettings, FDSignal
        >>> settings = FDSettings(N=1025, df=1e-4, min_freq=1.55e-3,
        ...                       max_freq=2.045e-2, force_backend="cpu")
        >>> FDSignal(np.zeros((3, 1025), dtype=complex), settings).shape
        (3, 189)
    """

    def __init__(self, arr, settings: FDSettings):
        FDSettings.__init__(self, *settings.args, **settings.kwargs)
        DomainBase.__init__(self, arr)
        if self.arr.shape[-1] != self.N_active:
            assert arr.shape[-1] == self.N
            _arr = self._arr.copy()
            del self._arr
            self.arr = _arr[:, self.active_slice]
            # self.arr = 

    @property
    def settings(self) -> FDSettings:
        """A fresh :class:`FDSettings` matching this signal's frequency grid."""
        return FDSettings(*self.args, **self.kwargs)

    def pad_array(self, arr: np.ndarray) -> np.ndarray:
        """Zero-pad an active-band array along the frequency axis before an inverse transform.

        Pads the trailing (frequency) axis only, so it accepts both the
        unbatched ``(nchannels, N_active)`` layout and the batched
        ``(nbatch, nchannels, N_active)`` one.

        Args:
            arr: Array with at least two dimensions whose last axis is the active band.

        Returns:
            ``xp.ndarray``: The padded array.

        Warning:
            The pad widths are ``(ind_min - 1, N - 1 - ind_max)``, so the result has
            ``N - 1`` bins (not ``N``) and the band lands one bin low; with
            ``ind_min == 0`` the left width is negative and ``pad`` raises.
        """
        assert arr.ndim >= 2
        pad_width = [(0, 0)] * (arr.ndim - 1) + [
            (self.ind_min - 1, self.N - 1 - self.ind_max)
        ]
        _arr = self.xp.pad(arr, pad_width, mode="constant", constant_values=0.0)
        return _arr

    def ifft(self, settings=None, window=None):
        """Inverse FFT back to the time domain (zero-padding the active band if trimmed).

        Computes ``irfft(arr * window) / dt`` along the last axis (the inverse of
        :meth:`TDSignal.fft`'s normalization). A band-limited signal is first
        padded with :meth:`pad_array` and a warning is issued.

        Args:
            settings (TDSettings, optional): Target time grid (its ``dt`` sets the
                normalization). If ``None``, a grid with ``Nobs`` samples (the
                ``irfft`` output length) and ``dt = 1 / (df * Nobs)`` is built on this
                signal's backend.
            window (array-like, optional): Frequency-domain window multiplied in
                before the inverse FFT. Default all ones.

        Returns:
            TDSignal: Shape ``outer_shape + (Nobs,)``.
        """

        arr_in = self.arr.copy()
        
        if self.ind_min != 0 or self.ind_max != self.N - 1:
            warnings.warn("Doing an ifft with a trimmed frequency domain array. Zero-padding.")
            arr_in = self.pad_array(arr_in)

        if window is None:
            window = self.xp.ones(arr_in.shape, dtype=float)

        _tmp = self.xp.fft.irfft(arr_in * window, axis=-1)
        
        if settings is None:
            # Intermediate settings built under the hood MUST inherit this
            # signal's backend: layered transforms (e.g. WDM -> STFT goes
            # through wdm_to_fd().ifft(settings=None).stft()) otherwise
            # produce a CPU-default TDSettings mid-chain on GPU runs and the
            # next step mixes NumPy/CuPy.
            Tobs = 1 / self.df
            Nobs = _tmp.shape[-1]
            dt = Tobs / Nobs
            settings = TDSettings(Nobs, dt, force_backend=self.backend)

        td_arr = _tmp / settings.dt
        return TDSignal(td_arr, settings)

    def __repr__(self) -> str:
        return (
            f"FDSignal(N={self.N}, df={self.df}, "
            f"min_freq={self.min_freq}, max_freq={self.max_freq}, "
            f"backend={self.backend_name.split('_')[-1]})"
        )

    # TODO check: keep/remove/document -- always raises NotImplementedError; no callers
    def get_fd_window_for_wdm(self, settings):
        """Build the WDM analysis window in frequency space (currently unimplemented)."""
        # TODO/DOCS: this method computes only the first half of the WDM window
        # and then raises NotImplementedError before normalising; the active
        # WDM path uses ``settings.window`` instead. Verify whether this helper
        # is still needed.
        N = self.settings.N

        # solve for window
        N = (settings.Nf+1)

        # mini wavelet structure for basis covering just N layers
        T = settings.dt*settings.Nt
        domega = 2 * np.pi / T

        window = self.xp.zeros(self.N, dtype=complex)

        # wdm window function
        for i in range(0, int(settings.Nt / 2)):  # (i=0; i<=wdm->Nt/2; i++)
            omega = i*domega
            window[i] = settings.phitilde(omega)

        raise NotImplementedError

        # normalize
        # for(i=-wdm->Nt/2; i<= wdm->Nt/2; i++) norm += window[abs(i)]*window[abs(i)];
        # norm = sqrt(norm/wdm_temp->cadence);

        # for(i=0; i<=wdm->NT/2; i++) window[i] /= norm;

        # free(wdm_temp);

    def wdmtransform(
        self, settings=None, window=None, return_transpose_time_axis_first: bool = False, is_psd: bool = False
    ):
        """Transform the FD signal into the WDM wavelet basis.

        The input must be the rFFT grid of ``settings.N = Nf * Nt`` time samples
        (``N // 2 + 1`` bins, as produced by :meth:`TDSignal.fft`); a band-limited
        input is zero-padded first (with a warning, see :meth:`pad_array`). Only
        the active frequency layers ``[ind_min_f, ind_max_f]`` (plus layer ``Nf``
        when the band starts at ``m = 0``) are transformed, through the cached
        gather map :meth:`WDMSettings.fold_shift_map`, and only the active time
        columns are returned. Each layer is one length-``Nt`` inverse FFT; layers
        are processed in blocks bounded by a byte budget (256 MiB, overridable
        with a ``wdm_layer_budget_bytes`` or ``wdm_layer_chunk`` attribute on the
        settings). Batched ``(nbatch, nchannels, N)`` input is supported.

        On a multi-GPU run the transform executes on the device that owns
        ``arr``, with the settings' cached window / gather map copied there.

        Args:
            settings: :class:`WDMSettings` describing the wavelet grid (required).
            window: Unused in the current implementation; the WDM analysis window
                is taken from ``settings.window``.
            return_transpose_time_axis_first: Accepted and passed through, but
                unused: the output is always ``(..., Nf_active, Nt_active)``.
            is_psd: If ``True``, treat the input as a stationary PSD sampled on the
                rFFT grid and return its per-layer WDM fold (the value repeats along
                each layer's time columns; with ``ind_min_f == 0`` row ``m = 0``
                alternates the DC and Nyquist folds) instead of a transformed signal.

        Returns:
            WDMSignal: Real coefficients (complex when ``settings.is_complex``), shape
            ``outer_shape + (Nf_active, Nt_active)``. With ``is_psd=True`` a plain
            complex ``xp.ndarray`` of the same shape.

        Raises:
            ValueError: If ``settings`` is ``None``.
            AssertionError: If ``settings`` is not a :class:`WDMSettings`.
        """
        if settings is None:
            raise ValueError("Must provide WDMSettings for WDM transform.")
        assert isinstance(settings, WDMSettings)
        # MULTI-GPU (2026-10-09): run the whole transform on the device that
        # OWNS this signal, with the settings' cached arrays (window, fold
        # map) brought there through the host. They live on the settings
        # object's home device (GPU 0), and a GPU-1 rank transforming a GPU-1
        # signal used to read them across devices by peer access
        # ("PerformanceWarning: The device where the array resides (0) is
        # different from the current device (1)"): GPU work on two devices is
        # unordered, a race behind the 6mo relaunch's illegal-memory-access
        # crash that CUDA_LAUNCH_BLOCKING=1 made disappear. A strict no-op on
        # CPU / one GPU / an already-local signal.
        from .utils.device import device_context

        dev = getattr(getattr(self.arr, "device", None), "id", None)
        with device_context(self.xp, dev):
            return self._wdmtransform_on_device(
                settings, return_transpose_time_axis_first, is_psd)

    def _wdmtransform_on_device(self, settings, return_transpose_time_axis_first, is_psd):
        """:meth:`wdmtransform`'s body, run inside the signal's device context."""
        from .utils.device import to_current_device_cached

        # Layer selection + rFFT gather map: shared with the PSD-fold fast
        # path in sensitivity.get_sensitivity, and cached on the settings
        # (it depends only on the grid and the active band).
        # TODO: WITH ROBBIE CHECK SECOND TO TOP INDEX START AND END
        Nf_act = settings.Nf_active
        include_top = (settings.ind_min_f == 0)
        n_special = Nf_act + (1 if include_top else 0)
        m_special_1d, k, herm, _ = to_current_device_cached(
            self.xp, settings, "fold_shift_map", settings.fold_shift_map())
        (base_window,) = to_current_device_cached(
            self.xp, settings, "window", (settings.window,))
        base_window = base_window[:]

        # No .copy(): every use below is a READ. The gather ``arr_in[..., k]``
        # produces a new array and all arithmetic happens on that, so nothing
        # ever writes through to ``self.arr``. The copy duplicated the whole
        # frequency-domain array -- 1.2 GiB for a batch of 8 over two years,
        # the single largest allocation in this transform -- for nothing.
        # ``pad_array`` already returns a fresh array on the trimmed path.
        arr_in = self.arr

        if self.ind_min != 0 or self.ind_max != self.N - 1:
            warnings.warn("Doing an ifft with a trimmed frequency domain array. Zero-padding.")
            arr_in = self.pad_array(arr_in)

        # Every index below addresses the TRAILING axes -- (..., n_special, Nt)
        # for the layer grid, (..., N) for the frequency axis -- so the same
        # code serves the unbatched ``(nchannels, N)`` layout and the batched
        # ``(nbatch, nchannels, N)`` one. ``k``, ``herm`` and ``set_zero`` are
        # all (n_special, Nt) masks over the two trailing axes; indexing them
        # positionally (``arr[:, k]``) silently addressed the CHANNEL axis as
        # soon as a batch axis was present, which is why a batched WDM
        # transform used to raise IndexError instead of broadcasting.
        if is_psd:
            before_ifft = arr_in[..., k] / settings.data_dt
            tmp_arr = before_ifft.copy()
            tmp_arr *= base_window**2 * np.pi * settings.data_dt
            psd_sum_tmp = tmp_arr.sum(axis=-1)
            psd_sum_tmp /= settings.Nf * settings.Nt   # = N

            wdmpsd_active = self.xp.zeros(
                self.outer_shape + (Nf_act, settings.Nt), dtype=complex
            )
            if include_top:
                # row 0 == m=0, row -1 == m=Nf, rows 1..Nf_act-1 == m=1..ind_max_f
                wdmpsd_active[..., 1:, :] = psd_sum_tmp[..., 1:Nf_act, None]
                wdmpsd_active[..., 0, 0::2] = psd_sum_tmp[..., 0, None]
                wdmpsd_active[..., 0, 1::2] = psd_sum_tmp[..., -1, None]
            else:
                # rows 0..Nf_act-1 map directly to m=ind_min_f..ind_max_f
                wdmpsd_active[...] = psd_sum_tmp[..., :Nf_act, None]

            wdmpsd_out = wdmpsd_active[..., settings.active_slice_t]
            return wdmpsd_out

        is_complex = bool(getattr(settings, "is_complex", False))
        out_dtype = complex if is_complex else float
        # Only the ACTIVE time columns are ever returned -- the old code built
        # all Nt and threw the rest away at the last line. When min_time /
        # max_time narrow the box (an MBHB occupies ~100 of 1451 layers at two
        # years) that is most of two large arrays wasted. Keep the active
        # columns and, for the two rows whose assembly reads other columns
        # (m=0 pulls from m=Nf at n-1), keep those two rows at full length --
        # two rows out of n_special is nothing.
        _t_sl = settings.active_slice_t
        t_lo = 0 if _t_sl.start is None else int(_t_sl.start)
        t_hi = settings.Nt if _t_sl.stop is None else int(_t_sl.stop)
        if _t_sl.step not in (None, 1):
            # Strided active slices are not something the assembly below can
            # express column-wise; fall back to the full axis and slice at the
            # end, exactly as before.
            t_lo, t_hi = 0, settings.Nt
            _slice_at_end = True
        else:
            _slice_at_end = False
        Nt_keep = t_hi - t_lo

        tmp_w_mn = self.xp.zeros(
            self.outer_shape + (n_special, Nt_keep), dtype=out_dtype
        )
        # Rows the m=0 assembly reads from, kept over the full time axis.
        _full_rows = {} if not include_top else {0: None, n_special - 1: None}
        kappa = 2 * np.sqrt(np.pi * settings.data_dt) / settings.Nf

        # Frequency layers are INDEPENDENT: layer m needs only its own row of
        # ``k`` and its own length-Nt iFFT. Doing all n_special at once
        # materialises two (outer..., n_special, Nt) COMPLEX arrays at the same
        # time -- 2.6 GiB of the 4.0 GiB peak for a batch of 8 over 2 years,
        # for a template that occupies ~60 of 1451 time layers. Walking the
        # layer axis in blocks bounds those two intermediates without changing
        # a single arithmetic operation, so the result is bitwise identical.
        #
        # The block size is chosen from a byte budget rather than exposed as a
        # tuning knob: the whole point is that the caller should not have to
        # know n_special or Nt to avoid an out-of-memory failure.
        per_layer_bytes = int(np.prod(self.outer_shape)) * settings.Nt * 16
        budget = int(getattr(settings, "wdm_layer_budget_bytes", 0) or 0) or _WDM_LAYER_BUDGET
        layer_chunk = int(getattr(settings, "wdm_layer_chunk", 0) or 0) or max(
            1, min(n_special, budget // max(per_layer_bytes, 1))
        )

        for _lo in range(0, n_special, layer_chunk):
            _hi = min(_lo + layer_chunk, n_special)
            m_here = self.xp.repeat(m_special_1d[_lo:_hi, None], settings.Nt, axis=-1)
            n_here = self.xp.tile(self.xp.arange(settings.Nt), (_hi - _lo, 1))

            # Gather THIS block's frequency bins only, so the (outer, block, Nt)
            # complex array is the largest thing alive rather than the full
            # (outer, n_special, Nt).
            _k = k[_lo:_hi]
            _b = arr_in[..., _k] / settings.data_dt
            # ``herm`` is what fold_shift_map returns; it IS ``neg_k | over_k``,
            # both of which are now local to that method.
            _herm = herm[_lo:_hi]
            if _herm.any():
                _b[..., _herm] = self.xp.conj(_b[..., _herm])
            _b *= base_window
            _a = self.xp.fft.ifft(_b, axis=-1)
            del _b

            set_zero = ((m_here == settings.Nf) | (m_here == 0)) & (
                (m_here + n_here) % 2 != 0
            )
            projected = self.xp.conj(
                settings.get_Cmn(m_here[~set_zero], n_here[~set_zero])
            ) * _a[..., ~set_zero]
            del _a
            _phase = kappa * (-1) ** ((m_here + 1) * n_here)[~set_zero]

            # Write via an explicit block, not chained indexing: assigning into
            # ``tmp_w_mn[..., lo:hi, :][..., mask]`` relies on the first index
            # returning a view, which is true for basic slicing but is exactly
            # the sort of thing that silently stops being true.
            _blk = self.xp.zeros(
                self.outer_shape + (_hi - _lo, settings.Nt), dtype=out_dtype
            )
            _blk[..., ~set_zero] = (
                _phase * projected if is_complex else _phase * self.xp.real(projected)
            )
            for _r in _full_rows:
                if _lo <= _r < _hi:
                    _full_rows[_r] = _blk[..., _r - _lo, :].copy()
            tmp_w_mn[..., _lo:_hi, :] = _blk[..., t_lo:t_hi]
            del projected, _blk

        if self.backend.uses_cupy and _WDM_CLEAR_FFT_CACHE:
            # Wiping the whole cuFFT plan cache here costs 172 ms PER CALL
            # (measured: cProfile shows PlanCache.clear -> _clear_LinkedList at
            # 0.172 s of a 0.178 s transform, and the cache reports 0 hits
            # ever). That is ~30% of an entire MBH likelihood evaluation, so in
            # a sampler it dominates everything else.
            #
            # The original comment read only "some issue with cupy and
            # xp.real/imag", with no reproducer. Kept behind an opt-in env var
            # rather than deleted, so the workaround is recoverable if that
            # issue resurfaces: set LISATOOLS_WDM_CLEAR_FFT_CACHE=1.
            cache = self.xp.fft.config.get_plan_cache()
            cache.clear()

        w_mn_active = self.xp.zeros(
            self.outer_shape + (Nf_act, Nt_keep), dtype=out_dtype
        )
        if include_top:
            w_mn_active[..., 1:, :] = tmp_w_mn[..., 1:Nf_act, :]
            # Row 0 interleaves two sources by ABSOLUTE column parity: even
            # columns come from m=0 at the same column, odd columns from m=Nf
            # at column n-1. Windowing shifts the local index, so resolve the
            # parity against absolute indices and read the two special rows,
            # which were kept full length for exactly this reason.
            _abs = self.xp.arange(t_lo, t_hi)
            _even = (_abs % 2) == 0
            _r0 = _full_rows[0]
            _rN = _full_rows[n_special - 1]
            w_mn_active[..., 0, _even] = _r0[..., _abs[_even]] / np.sqrt(2.)
            w_mn_active[..., 0, ~_even] = _rN[..., _abs[~_even] - 1] / np.sqrt(2.)
        else:
            w_mn_active[...] = tmp_w_mn[..., :Nf_act, :]

        # Already restricted to the active columns above, unless a strided
        # active slice forced the old full-axis path.
        output = (w_mn_active[..., settings.active_slice_t]
                  if _slice_at_end else w_mn_active)

        return WDMSignal(output, settings=settings)

    def transform(self, new_domain: DomainSettingsBase, window: np.ndarray | cp.ndarray = None):
        """Dispatch to :meth:`ifft` or :meth:`wdmtransform` based on ``new_domain``.

        An ``FDSettings`` target returns ``arr * window`` on this signal's own
        settings; a WDM target always uses ``new_domain.window`` (``window`` is
        ignored there). A target on another backend moves the signal there first
        (:meth:`DomainBase.with_backend`).

        Args:
            new_domain: Target settings (FD, TD or WDM).
            window (array-like, optional): FD-shaped window for the FD / TD targets.
                Default all ones.

        Returns:
            DomainBase: The signal in the target domain.

        Raises:
            NotImplementedError: For an STFT target.
            ValueError: If ``new_domain`` is not a recognized settings class.
        """
        signal, window = self._coerce_transform_backend(new_domain, window)
        if signal is not self:
            return signal.transform(new_domain, window=window)
        if window is None:
            window = self.xp.ones(self.arr.shape, dtype=float)

        if isinstance(new_domain, FDSettings):
            return self.settings.associated_class(self.arr * window, self.settings)

        elif isinstance(new_domain, TDSettings):
            return self.ifft(settings=new_domain, window=window)

        elif isinstance(new_domain, STFTSettings):
            raise NotImplementedError
            return self.stft()

        elif isinstance(new_domain, WDMSettings):
            return self.wdmtransform(settings=new_domain, window=new_domain.window)
        else:
            raise ValueError(f"new_domain type is not recognized {type(new_domain)}.")

    def plot(
        self, channel: int = 0, ax: plt.Axes | None = None, filename: Optional[str] = None, **kwargs
    ) -> plt.Axes:
        """Plot ``|arr[channel]|**2`` against frequency on log-log axes.

        Args:
            channel: The channel index to visualize (first array axis; unbatched).
            ax: Matplotlib Axes to plot on. If ``None``, a new figure and axes are created.
            filename: If given, the figure is saved to this file.
            **kwargs: Forwarded to ``ax.loglog``.

        Returns:
            plt.Axes: The Axes containing the plot.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        f_arr = asnumpy(self.f_arr)
        arr_here = asnumpy(self.arr[channel])

        ax.loglog(f_arr, np.abs(arr_here) ** 2, **kwargs)

        ax.set_title(f"Frequency Spectrum for Channel {channel}")
        ax.set_xlabel("Frequency")
        ax.set_ylabel("Magnitude")
        ax.set_xlim(self.min_freq, self.max_freq)
        if filename is not None:
            plt.savefig(filename, bbox_inches="tight")
        return ax


class STFTSettings(DomainSettingsBase):
    """Short-time Fourier transform basis settings.

    ``NT`` contiguous segments of duration ``dt`` starting at ``t0``, each carrying the
    one-sided frequency grid ``arange(0, NF) * df``. The active band is snapped
    like :class:`FDSettings`: ``ind_min = ceil(min_freq / df)`` (``0`` for ``None``) and
    ``ind_max = int(max_freq / df)`` (``NF - 1`` for ``None``). Signal arrays have
    trailing axes ``(NT, NF_active)``. Usually built from a time array with
    :func:`get_stft_settings` (``NF = nperseg // 2 + 1``, ``df = 1 / dt``).

    Args:
        t0: Time of the first segment (seconds).
        dt: Segment cadence in seconds (spacing between successive STFT slices).
        df: Frequency bin spacing in Hz.
        NT: Number of time segments.
        NF: Number of frequency bins per segment.
        min_freq: Lower edge of the active band (Hz). Defaults to ``0.0``.
        max_freq: Upper edge of the active band (Hz). If ``None``, the full range
            is used.
        **kwargs: Forwarded to :class:`DomainSettingsBase` (e.g. ``force_backend``).

    Raises:
        ValueError: If ``min_freq`` or ``max_freq`` is negative.

    Attributes:
        t0 (float): Start time of segment 0 (s).
        dt (float): Segment duration / cadence (s).
        df (float): Bin spacing (Hz).
        NT (int): Number of segments.
        NF (int): Number of bins per segment on the full grid.
        min_freq_input (float or None): ``min_freq`` exactly as given.
        max_freq_input (float or None): ``max_freq`` exactly as given.

    Example:
        >>> from lisatools.domains import STFTSettings
        >>> st = STFTSettings(t0=0.0, dt=3600.0, df=1 / 3600.0, NT=120, NF=361,
        ...                   force_backend="cpu")
        >>> st.basis_shape_active
        (120, 361)
    """

    t0: float
    dt: float
    df: float
    NT: int
    NF: int
    min_freq: Optional[float] = 0.0
    max_freq: Optional[float] = None

    def __init__(
        self,
        t0: float,
        dt: float,
        df: float,
        NT: int,
        NF: int,
        min_freq: Optional[float] = 0.0,
        max_freq: Optional[float] = None,
        **kwargs,
    ):
        self.t0 = t0
        self.dt = dt
        self.df = df
        self.NT = NT
        self.NF = NF
        self.min_freq = min_freq
        self.max_freq = max_freq
        super().__init__(**kwargs)

    def __repr__(self):
        return (
            f"STFTSettings(t0={self.t0}, dt={self.dt}, df={self.df}, NT={self.NT}, NF={self.NF}, "
            f"min_freq={self.min_freq}, max_freq={self.max_freq}, backend={self.backend_name.split('_')[-1]})"
        )

    @staticmethod
    def make_factory(big_dt: float, min_freq: Optional[float] = 0.0, max_freq: Optional[float] = None):
        """Build a ``(times, dt, force_backend) -> STFTSettings`` factory.

        Delegates to :func:`get_stft_settings`, which fits ``big_dt`` to an
        integer multiple of the sample step of ``times`` and derives ``NT``/``NF``
        from there (the factory's ``dt`` argument is not used). The returned
        function carries ``domain_settings_class = STFTSettings``.

        Args:
            big_dt: Requested segment duration (s).
            min_freq: Lower band edge (Hz).
            max_freq: Upper band edge (Hz).

        Returns:
            callable: ``factory(times, dt, force_backend) -> STFTSettings``.
        """
        def _factory(times, dt, force_backend):
            return get_stft_settings(
                times=times, big_dt=big_dt,
                min_freq=min_freq, max_freq=max_freq,
                force_backend=force_backend,
            )
        _factory.domain_settings_class = STFTSettings  # see FDSettings.make_factory
        return _factory

    @staticmethod
    def get_associated_class():
        """Return :class:`STFTSignal`."""
        return STFTSignal

    @staticmethod
    def get_associated_group():
        """Return :class:`~lisatools.domaincomputation.STFTComputationGroup` (lazy import)."""
        from .domaincomputation import STFTComputationGroup
        return STFTComputationGroup

    @property
    def associated_class(self):
        """The paired signal class, :class:`STFTSignal`."""
        return self.get_associated_class()

    @property
    def associated_group(self):
        """The batched C++ likelihood group for this domain (see :meth:`get_associated_group`)."""
        return self.get_associated_group()

    @property
    def basis_shape(self) -> tuple:
        """``(NT, NF_active)``: same as :attr:`basis_shape_active` for STFT."""
        return (
            self.NT,
            self.NF_active,
        )  #! in the STFT domain, the basis shape is (# number of times segments, # number of frequencies)

    @property
    def basis_shape_active(self) -> tuple:
        """Active basis shape ``(NT, NF_active)`` — required by the
        :class:`DomainBase` array contract (trailing signal axes)."""
        return (self.NT, self.NF_active)

    @property
    def total_terms(self) -> int:
        """Number of active basis elements, ``NT * NF_active``."""
        return self.NT * self.NF_active

    @property
    def t_arr(self) -> np.ndarray:
        """Segment start times ``t0 + arange(NT) * dt`` (s)."""
        return self.t0 + self.xp.arange(self.NT) * self.dt
   
    @property
    def min_freq(self) -> float:
        """Requested lower band edge (Hz); setting it snaps ``ind_min = ceil(min_freq / df)``."""
        return self._min_freq

    @min_freq.setter
    def min_freq(self, value: Optional[float]):
        if value is not None and value < 0:
            raise ValueError("min_freq must be non-negative.")

        # self._min_freq = value
        # set it to the closest frequency bin
        self.min_freq_input = value
        if value is not None:
            self.ind_min = int(np.ceil(value / self.df))
        else:
            self.ind_min = 0
        self._min_freq = value

    @property
    def max_freq(self) -> Optional[float]:
        """Requested upper band edge (Hz); setting it snaps ``ind_max = int(max_freq / df)``."""
        return self._max_freq

    @max_freq.setter
    def max_freq(self, value: Optional[float]):
        if value is not None and value < 0:
            raise ValueError("max_freq must be non-negative.")

        # self._max_freq = value
        # set it to the closest frequency bin
        self.max_freq_input = value
        if value is not None:
            self.ind_max = int(value / self.df)
        else:
            self.ind_max = (self.NF - 1)
        self._max_freq = value

    @property
    def ind_min(self) -> int:
        """First active frequency bin (``None`` -> ``0``)."""
        return self._ind_min
    
    @ind_min.setter
    def ind_min(self, ind_min: int):
        if ind_min is None:
            ind_min = 0
        self._ind_min = ind_min

    @property
    def ind_max(self) -> int:
        """Last active frequency bin, inclusive (``None`` -> ``NF - 1``)."""
        return self._ind_max
    
    @ind_max.setter
    def ind_max(self, ind_max: int):
        if ind_max is None:
            ind_max = self.NF - 1
        self._ind_max = ind_max

    @property
    def f_arr(self) -> np.ndarray:
        """Active-band frequencies ``k * df`` for ``k`` in ``[ind_min, ind_max]`` (Hz)."""

        _all_freqs = self.xp.arange(0, self.NF) * self.df
        return _all_freqs[self.active_slice]

    @property
    def args(self) -> tuple:
        """Positional reconstruction arguments ``(t0, dt, df, NT, NF)``."""
        return (self.t0, self.dt, self.df, self.NT, self.NF)

    @property
    def kwargs(self) -> dict:
        """Keyword reconstruction arguments (``min_freq``, ``max_freq``, ``force_backend``)."""
        return dict(
            min_freq=self.min_freq, max_freq=self.max_freq, force_backend=self.backend
        )

    @property
    def f_arr_edges(self) -> np.ndarray:
        """Full-grid bin edges ``arange(NF + 1) * df`` (Hz; not band-limited)."""
        return self.xp.arange(self.NF + 1) * self.df

    @property
    def t_arr_edges(self) -> np.ndarray:
        """Segment edges ``arange(NT + 1) * dt`` (s; ``t0`` is not added)."""
        return self.xp.arange(self.NT + 1) * self.dt

    def __eq__(self, value):
        if not isinstance(value, STFTSettings):
            return False
        return (
            (value.NT == self.NT)
            and (value.NF == self.NF)
            and (value.dt == self.dt)
            and (value.df == self.df)
            and (self.xp.isclose(value.t0, self.t0))
            and (self.xp.isclose(value.min_freq, self.min_freq))
            and (self.xp.isclose(value.max_freq, self.max_freq))
        )

    @property
    def differential_component(self) -> float:
        """Differential element of the inner-product sum (``df``)."""
        return self.df

    @property
    def active_slice(
        self,
    ) -> slice:
        """Slice of the frequency axis selecting ``[ind_min, ind_max]``."""
        return slice(self.ind_min, self.ind_max + 1)

    @property
    def NF_active(self) -> int:
        """Number of active frequency bins."""
        sl = self.active_slice
        return sl.stop - sl.start

    def get_nperseg(self, small_dt: float):
        """Number of samples per segment given the underlying sample step ``small_dt``.

        Args:
            small_dt: Time-domain sample step (s).

        Returns:
            int: ``round(dt / small_dt)``.

        Raises:
            AssertionError: If ``dt`` is not an integer multiple of ``small_dt``.
        """
        nperseg = round(self.dt / small_dt)

        assert (
            abs(nperseg * small_dt - self.dt) < 1e-10 * self.dt
        ), f"big_dt={self.dt} must be an integer multiple of dt={small_dt}"

        return nperseg

    def compute_slice_indices(
        self, tmin: float, tmax: float, fmin: float, fmax: float
    ) -> Tuple[slice, slice]:
        """Compute time and frequency slices covering a ``(t, f)`` box.

        Time edges round to the nearest segment, ``round((t - t0) / dt)``. The
        frequency slice is relative to the ACTIVE band: start
        ``round((fmin - f_arr[0]) / df)``, stop ``floor((fmax - f_arr[0]) / df) + 1``.

        Args:
            tmin: Minimum time value for the slice (s).
            tmax: Maximum time value for the slice (s).
            fmin: Minimum frequency value for the slice (Hz).
            fmax: Maximum frequency value for the slice (Hz).

        Returns:
            tuple[slice, slice]: ``(time_slice, freq_slice)``, e.g.
            ``(slice(0, 10), slice(5, 15))``.

        Raises:
            ValueError: If the box extends outside the time grid or the active band.
        """

        if tmin < self.t0:
            raise ValueError("tmin must be greater than or equal to t0.")
        if tmax > self.t0 + self.NT * self.dt:
            raise ValueError("tmax must be less than or equal to t0 + NT*dt.")
        f_min_active = self.f_arr[0]
        f_max_active = self.f_arr[-1]
        if fmin < f_min_active:
            raise ValueError(
                f"fmin ({fmin}) must be >= the active minimum frequency ({f_min_active})."
            )
        if fmax > f_max_active:
            raise ValueError(
                f"fmax ({fmax}) must be <= the active maximum frequency ({f_max_active})."
            )

        time_start_idx = int(self.xp.round((tmin - self.t0) / self.dt))
        time_end_idx = int(self.xp.round((tmax - self.t0) / self.dt))
        freq_start_idx = int(self.xp.round((fmin - self.f_arr[0]) / self.df))
        freq_end_idx = int(self.xp.floor((fmax - self.f_arr[0]) / self.df)) + 1

        return slice(time_start_idx, time_end_idx), slice(freq_start_idx, freq_end_idx)

    def get_slice(self, index: tuple) -> STFTSettings:
        """Return STFT settings for a ``(time, frequency)`` sub-box.

        The time slice moves ``t0`` and sets ``NT``; the frequency slice (relative
        to the active band) becomes new ``min_freq`` / ``max_freq`` on the same
        full ``NF`` grid.

        Args:
            index: ``(time_slice, freq_slice)``, e.g. ``(slice(0, 10), slice(5, 15))``.

        Returns:
            STFTSettings: Settings for the sub-box.

        Raises:
            ValueError: If ``index`` is not a 2-tuple.
        """
        if not isinstance(index, tuple) or len(index) != 2:
            raise ValueError(
                "Index must be a tuple of two slices for time and frequency dimensions."
            )

        time_slice, freq_slice = index

        new_t0 = self.t0 + time_slice.start * self.dt
        new_NT = time_slice.stop - time_slice.start
        new_NF = self.NF

        new_min_freq = float(self.f_arr[freq_slice.start])
        new_max_freq = float(self.f_arr[freq_slice.stop - 1])

        return STFTSettings(
            t0=new_t0,
            dt=self.dt,
            df=self.df,
            NT=new_NT,
            NF=new_NF,
            min_freq=new_min_freq,
            max_freq=new_max_freq,
            force_backend=self.backend,
        )


def get_stft_settings(
    times: np.ndarray | cp.ndarray,
    big_dt: float,
    min_freq: Optional[float] = 0.0,
    max_freq: Optional[float] = None,
    **kwargs,
) -> STFTSettings:
    """Build STFT settings from a uniform time array and a desired segment length.

    With ``dt = times[1] - times[0]`` and ``N = len(times)``: the segment length is
    rounded down to a whole number of samples, ``nperseg = int(big_dt / dt)``,
    ``big_dt -> nperseg * dt``; then ``NT = floor(N / nperseg)``,
    ``NF = nperseg // 2 + 1``, ``df = 1 / big_dt`` and ``t0 = times[0]``.

    Args:
        times: Uniform time array (s).
        big_dt: Desired segment duration (s).
        min_freq: Lower band edge (Hz).
        max_freq: Upper band edge (Hz); ``None`` keeps the full band.
        **kwargs: Forwarded to :class:`STFTSettings` (e.g. ``force_backend``).

    Returns:
        STFTSettings: The settings for the STFT.

    Example:
        >>> import numpy as np
        >>> from lisatools.domains import get_stft_settings
        >>> st = get_stft_settings(np.arange(86400) * 5.0, big_dt=3600.0,
        ...                        force_backend="cpu")
        >>> st.NT, st.NF
        (120, 361)
    """

    t0 = float(times[0])
    N = len(times)
    dt = float(times[1] - times[0])

    big_dt = int(big_dt / dt) * dt  # make sure big_dt is an integer multiple of dt
    NT = int(np.floor(N / (big_dt / dt)))
    DF = 1 / big_dt
    nperseg = int(big_dt / dt)
    NF = nperseg // 2 + 1

    return STFTSettings(
        t0=t0, dt=big_dt, df=DF, NT=NT, NF=NF, min_freq=min_freq, max_freq=max_freq, **kwargs
    )


class STFTSignal(STFTSettings, DomainBase):
    """STFT array wrapper paired with :class:`STFTSettings`.

    Usually produced by :meth:`TDSignal.stft` (or ``td.transform(stft_settings)``).

    Args:
        arr: NumPy or CuPy array with trailing axes ``(NT, NF_active)``; an array
            whose last axis is the full ``NF`` is sliced to the active band.
        settings: :class:`STFTSettings` describing the time-frequency grid.

    Raises:
        AssertionError: If the last axis is neither ``NF_active`` nor ``NF``.

    Example:
        >>> import numpy as np
        >>> from lisatools.domains import TDSettings, TDSignal, get_stft_settings
        >>> times = np.arange(86400) * 5.0
        >>> td = TDSignal(np.zeros((3, 86400)), TDSettings(N=86400, dt=5.0, force_backend="cpu"))
        >>> stft = td.stft(settings=get_stft_settings(times, big_dt=3600.0, force_backend="cpu"))
        >>> stft.shape
        (3, 120, 361)
    """

    def __init__(self, arr, settings: STFTSettings):
        STFTSettings.__init__(self, *settings.args, **settings.kwargs)
        DomainBase.__init__(self, arr)

        # Frequency axis is the LAST one (trailing axes are (NT, NF_active)).
        # Accept a full-NF array and slice it down to the active band.
        if self.arr.shape[-1] != self.NF_active:
            assert arr.shape[-1] == self.NF, (
                f"STFTSignal array last axis must be NF_active "
                f"({self.NF_active}) or NF ({self.NF}); got {arr.shape[-1]}."
            )
            _arr = self._arr.copy()
            del self._arr
            self.arr = _arr[..., self.active_slice]

    @property
    def settings(self) -> STFTSettings:
        """A fresh :class:`STFTSettings` matching this signal's grid."""
        return STFTSettings(*self.args, **self.kwargs)

    def __repr__(self) -> str:
        return (
            f"STFTSignal(t0={self.t0}, dt={self.dt}, df={self.df}, NT={self.NT}, NF={self.NF}, "
            f"min_freq={self.min_freq}, max_freq={self.max_freq}, backend={self.backend_name.split('_')[-1]})"
        )

    def _plot_stft(self, channel=0, ax=None, **kwargs):
        """``pcolormesh`` of ``|arr[channel]|**2`` on (time, log frequency) axes, with colorbar."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        t_arr = asnumpy(self.t_arr)
        f_arr = asnumpy(self.f_arr)

        arr_here = asnumpy(self.arr[channel])
        cb = ax.pcolormesh(
            t_arr, f_arr, (np.abs(arr_here) ** 2).T, shading="auto", cmap="cividis", **kwargs
        )

        ax.set_yscale("log")
        ax.set_xlabel("Time")
        ax.set_ylabel("Frequency")
        ax.set_ylim(self.min_freq, self.max_freq)
        plt.colorbar(cb, ax=ax, label="Magnitude")
        return ax

    def _plot_fd(self, channel=0, ax=None, time_bin=0, **kwargs):
        """Log-log ``|arr[channel, time_bin]|**2`` against frequency."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        f_arr = asnumpy(self.f_arr)
        arr_here = asnumpy(self.arr[channel])

        ax.loglog(f_arr, np.abs(arr_here[time_bin]) ** 2, **kwargs)

        ax.set_title(
            f"STFT Frequency Spectrum for Time Bin {time_bin} (Time = {self.t_arr[time_bin]:.2f})"
        )
        ax.set_xlabel("Frequency")
        ax.set_ylabel("Magnitude")
        ax.set_xlim(self.min_freq, self.max_freq)
        return ax

    def _plot_td(self, channel=0, ax=None, freq_bin=0, **kwargs):
        """Real and imaginary parts of ``arr[channel, :, freq_bin]`` against segment time."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        t_arr = asnumpy(self.t_arr)
        arr_here = asnumpy(self.arr[channel])

        ax.plot(t_arr, arr_here[:, freq_bin].real, label="real part", **kwargs)
        ax.plot(t_arr, arr_here[:, freq_bin].imag, label="imag part", **kwargs)

        ax.legend()
        ax.set_title(
            f"STFT Time Series for Frequency Bin {freq_bin} (Frequency = {self.f_arr[freq_bin]:.2f})"
        )
        ax.set_xlabel("Time")
        ax.set_ylabel("Magnitude")
        return ax

    @property
    def ind_min(self) -> int:
        return self._ind_min

    @ind_min.setter
    def ind_min(self, ind_min: int):
        if ind_min is None:
            ind_min = 0
        self._ind_min = ind_min

    @property
    def ind_max(self) -> int:
        return self._ind_max

    @ind_max.setter
    def ind_max(self, ind_max: int):
        if ind_max is None:
            ind_max = self.NF - 1
        self._ind_max = ind_max

    def plot(
        self,
        channel: int = 0,
        ax: plt.Axes | None = None,
        plot_type: str = "stft",
        filename: Optional[str] = None,
        **kwargs,
    ) -> plt.Axes:
        """Plot one channel of the STFT as a time-frequency map, a spectrum, or a time series.

        Args:
            channel: The channel index to visualize (first array axis; unbatched).
            ax: Matplotlib Axes to plot on. If ``None``, a new figure and axes are created.
            plot_type: ``"stft"``: time vs (log) frequency map of ``|coeff|**2``;
                ``"fd"``: log-log ``|coeff|**2`` of one segment (``time_bin=`` keyword,
                default 0); ``"td"``: real and imaginary parts of one frequency bin
                across segments (``freq_bin=`` keyword, default 0).
            filename: If given, the figure is saved to this file.
            **kwargs: Forwarded to the plotting helper and on to matplotlib.

        Returns:
            plt.Axes: The Axes containing the plot.

        Raises:
            ValueError: If ``plot_type`` is not one of the three options.
        """
        if plot_type == "stft":
            ax = self._plot_stft(channel=channel, ax=ax, **kwargs)
        elif plot_type == "fd":
            ax = self._plot_fd(channel=channel, ax=ax, **kwargs)
        elif plot_type == "td":
            ax = self._plot_td(channel=channel, ax=ax, **kwargs)
        else:
            raise ValueError(
                f"Invalid plot_type {plot_type}. Must be one of 'stft', 'fd', or 'td'."
            )

        if filename is not None:
            plt.savefig(filename, bbox_inches="tight")

        return ax


WAVELET_BANDWIDTH = 6.51041666666667e-5
WAVELET_DURATION = 7680.0
WAVELET_FILTER_CONSTANT = 4


class WDMSettings(DomainSettingsBase):
    """Wavelet (WDM, Wilson-Daubechies-Meyer) basis settings: the global-fit domain.

    The WDM transform tiles the time-frequency plane of ``N = Nf * Nt`` time
    samples (step ``dt``) into ``Nf`` frequency layers by ``Nt`` time pixels:

    * ``layer_df = 1 / (2 * Nf * dt)``: layer spacing (Hz); layer ``m`` is centred
      at ``m * layer_df`` and ``Nf * layer_df`` is the Nyquist frequency;
    * ``layer_dt = Nf * dt``: pixel duration (s); pixel ``n`` sits at
      ``n * layer_dt`` from the grid start;
    * ``layer_df * layer_dt = 1/2`` (critical sampling), ``Tobs = N * dt``.

    The active box ``[ind_min_f, ind_max_f] x [ind_min_t, ind_max_t]`` is snapped
    inward from the requested edges and clipped to the grid:
    ``ind_min_f = ceil(min_freq / layer_df)``, ``ind_max_f = int(max_freq / layer_df)``
    (floor), ``ind_min_t = ceil(min_time / layer_dt)``,
    ``ind_max_t = int(max_time / layer_dt)``; ``None`` selects the grid edge. Times
    are measured from the grid start (``t0`` is not subtracted). Signal arrays carry
    only the active box, ``basis_shape_active == (Nf_active, Nt_active)``. The
    coefficients are real Gaussian variables, so
    :attr:`~DomainSettingsBase.logdet_factor` is ``0.5``.

    The stock global fits build this through :meth:`make_factory` with the run
    band (``min_freq`` / ``max_freq``) and an edge crop of
    ``min_time = edge_crop * layer_dt``, ``max_time = (Nt - edge_crop) * layer_dt``.

    Args:
        Nf: Number of frequency layers (an even integer; not checked here,
            :meth:`adjust_to_even_bins` picks an even pair).
        Nt: Number of time pixels per layer (an even integer).
        dt: Underlying time-domain sample step in seconds.
        t0: Start time in seconds. Defaults to ``0.0``. Carried through
            ``kwargs`` and compared by :meth:`sub_box_slices`; it does not shift
            :attr:`t_arr` or the time-band snapping.
        oversample: Stored and carried through ``kwargs`` / ``repr``; not used by
            the window construction. Defaults to ``16``.
        window: Pre-computed WDM analysis window of length ``Nt``. If provided,
            ``omega`` must also be supplied; otherwise the window is built by
            :meth:`setup_window`.
        omega: Pre-computed angular-frequency grid that pairs with ``window``.
        min_freq: Lower edge of the active frequency band (Hz). ``None`` selects
            layer 0.
        max_freq: Upper edge of the active frequency band (Hz). ``None`` selects
            layer ``Nf - 1``.
        min_time: Lower edge of the active time band (s from the grid start).
            ``None`` selects pixel 0.
        max_time: Upper edge of the active time band (s from the grid start).
            ``None`` selects pixel ``Nt - 1``.
        is_complex: If ``True``, coefficients are complex: the real WDM coefficient
            plus its quadrature (Hilbert-pair) companion as the imaginary part;
            :attr:`differential_component` is halved to keep inner products
            unchanged. Default ``False`` (the production real basis).
        **kwargs: Forwarded to :class:`DomainSettingsBase` (e.g. ``force_backend``).

    Raises:
        ValueError: If a band edge is negative.
        AssertionError: If ``window`` is given without ``omega`` or with a length
            other than ``Nt``.

    Attributes:
        Nf (int): Number of frequency layers.
        Nt (int): Number of time pixels.
        N (int): ``Nf * Nt``, the time-domain length the transform needs.
        data_dt (float): Time-domain sample step ``dt`` (s).
        Tobs (float): ``N * data_dt`` (s).
        layer_df (float): ``1 / (2 * Nf * data_dt)`` (Hz).
        layer_dt (float): ``Nf * data_dt`` (s).
        t0 (float): Grid start time (s).
        is_complex (bool): Quadrature (complex) mode.
        window (``xp.ndarray``): WDM analysis window, length ``Nt``.
        omega (``xp.ndarray``): Angular-frequency grid of ``window``.
        min_freq_input, max_freq_input, min_time_input, max_time_input: The band
            edges exactly as given.

    Example:
        >>> from lisatools.domains import WDMSettings
        >>> ws = WDMSettings(Nf=64, Nt=512, dt=5.0, min_freq=1e-3, max_freq=1e-2,
        ...                  force_backend="cpu")
        >>> ws.N, ws.layer_dt
        (32768, 320.0)
        >>> ws.ind_min_f, ws.ind_max_f
        (1, 6)
        >>> ws.basis_shape_active
        (6, 512)
    """

    # WDM coefficients entering the likelihood are real Gaussian variables
    # (true for the quadrature is_complex variant too, whose imaginary part is
    # dropped before the likelihood) -> logdet_factor = 0.5.
    _real_basis: bool = True

    def __init__(
        self,
        Nf: float,
        Nt: float,
        dt: float,
        t0: float = 0.0,
        oversample: int = 16,
        window: Optional[np.ndarray] = None,
        # norm: Optional[float] = None,
        omega: Optional[np.ndarray] = None,
        min_freq: Optional[float] = None,
        max_freq: Optional[float] = None,
        min_time: Optional[float] = None,
        max_time: Optional[float] = None,
        is_complex: bool = False,
        **kwargs
    ):
        DomainSettingsBase.__init__(self, **kwargs)
        self.Nt = Nt
        self.Nf = Nf
        self.data_dt = dt
        self.N = self.Nt * self.Nf
        self.data_dt = dt
        self.Tobs = self.N * self.data_dt
        self.layer_dt = self.Nf * self.data_dt
        self.layer_df = 1. / (2. * self.Nf * self.data_dt)
        self.t0 = t0
        # Complex/quadrature WDM mode -- when True the wavelet basis carries
        # both the standard real coefficient and its quadrature (Hilbert-pair)
        # companion as the imaginary part of a complex coefficient. The
        # differential_component is halved (0.125 instead of 0.25) so that the
        # diagnostic inner_product (which sums Re*Re + Im*Im via np.real of
        # sig1.conj()*sig2) recovers the same time-domain power as the
        # real-only WDM. NB: at the folded boundary layer m=0 (which packs
        # DC and Nyquist) the imag part is set to zero, so the correction is
        # exact for narrowband signals away from DC/Nyquist and slightly
        # over-corrects when boundary layers carry non-trivial power.
        self.is_complex = bool(is_complex)

        # these have to come after layer_df b/c setters
        # sets ind_min and ind_max
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.min_time = min_time
        self.max_time = max_time
        self.Nthalf = int(self.Nt / 2)
        self.oversample = oversample

        self.dOmega = 2 * np.pi * self.layer_df
        self.A = 0.0
        self.WAVELET_FILTER_CONSTANT = 4  # window roll-off parameter
        
        if window is None:
            self.setup_window()
        else:
            # assert norm is not None, "Must provide norm if providing window."
            assert omega is not None, "Must provide omega if providing window."
            assert len(window) == self.Nt
            self.window = window
            # self.norm = norm
            self.omega = omega
    
    @staticmethod
    def make_factory(
        Nf: int,
        Nt: int,
        min_freq: Optional[float] = None,
        max_freq: Optional[float] = None,
        oversample: int = 16,
        min_time: Optional[float] = None,
        max_time: Optional[float] = None,
    ):
        """Build a ``(times, dt, force_backend) -> WDMSettings`` factory.

        ``Nf`` / ``Nt`` and the band edges come from the caller; ``dt`` and
        ``force_backend`` are supplied at factory-call time by the global-fit
        engine once the data are loaded. ``times`` is ignored: the built settings
        have ``t0 = 0.0`` (array-space time) and ``is_complex=False``. The returned
        function carries ``domain_settings_class = WDMSettings`` so a run
        configuration can dispatch on the domain before the call. The stock global
        fits build their domain this way in
        :meth:`~lisatools.globalfit.stock.erebor.fit.EreborFit.make_domain_settings`
        (run band from ``MIN_FREQ`` / ``MAX_FREQ``, edge-cropped time box).

        Args:
            Nf: Number of frequency layers.
            Nt: Number of time pixels.
            min_freq: Lower band edge (Hz).
            max_freq: Upper band edge (Hz).
            oversample: Passed through to :class:`WDMSettings`.
            min_time: Lower time edge (s from the grid start).
            max_time: Upper time edge (s from the grid start).

        Returns:
            callable: ``factory(times, dt, force_backend) -> WDMSettings``.

        Example:
            >>> import numpy as np
            >>> from lisatools.domains import WDMSettings
            >>> factory = WDMSettings.make_factory(
            ...     Nf=64, Nt=512, min_freq=1e-3, max_freq=1e-2,
            ...     min_time=20 * 320.0, max_time=(512 - 20) * 320.0)
            >>> ws = factory(times=np.arange(64 * 512) * 5.0, dt=5.0, force_backend="cpu")
            >>> ws.ind_min_t, ws.ind_max_t
            (20, 492)
        """
        def _factory(times, dt, force_backend):
            return WDMSettings(
                Nf=Nf, Nt=Nt, dt=dt, oversample=oversample,
                min_freq=min_freq, max_freq=max_freq,
                min_time=min_time, max_time=max_time,
                force_backend=force_backend,
            )
        _factory.domain_settings_class = WDMSettings  # see FDSettings.make_factory
        return _factory

    @staticmethod
    def adjust_to_even_bins(t_min: float, t_max: float, dt: float, Tobs: float, num_linspace: Optional[int]=1000, verbose: Optional[bool] = False) -> Tuple[int, int, float]:
        """Pick a wavelet pixel duration in ``[t_min, t_max]`` that makes both ``Nf`` and ``Nt`` even.

        Scans ``num_linspace`` candidates from ``t_min`` upward; each is rounded down
        to a multiple of ``dt`` (``wavelet_duration``), then
        ``Nt = int(Tobs / wavelet_duration)`` and ``Nf = wavelet_duration / dt``
        (via the truncated ``Nt * wavelet_duration``). The first even pair wins, so
        the grid covers ``Nf * Nt * dt <= Tobs``. Used by the stock global fits
        (``derive_wdm_grid``) to size the run grid.

        Args:
            t_min: Lower bound on the wavelet pixel duration in seconds.
            t_max: Upper bound on the wavelet pixel duration in seconds.
            dt: Underlying time-sample step (the duration is rounded to a multiple).
            Tobs: Total observation time (seconds).
            num_linspace: Number of candidate durations to scan between ``t_min`` and ``t_max``.
            verbose: If ``True``, print each attempted duration.

        Returns:
            tuple[int, int, float]: ``(Nf, Nt, wavelet_duration)``; ``wavelet_duration``
            is the pixel duration ``layer_dt = Nf * dt`` (s).

        Raises:
            ValueError: If no candidate produces both ``Nf`` and ``Nt`` even.
        """
        Nf = -1
        Nt = -1

        found_wavelet = False
        wavelet_duration = -1.0
        # Evaluate every candidate against the ORIGINAL Tobs. (Mutating
        # Tobs inside the scan made it decay by up to t_max per iteration
        # -- harmless for year-long Tobs but a ZeroDivision crash on small
        # grids and a silent grid-shrink in general.)
        Tobs_in = Tobs
        for tmp in np.linspace(t_min, t_max, num_linspace):
            wavelet_duration = int(tmp / dt) * dt
            Nt = int(Tobs_in / wavelet_duration)
            if Nt == 0:
                continue  # candidate duration exceeds the observation time
            Tobs = Nt * wavelet_duration
            N = int(Tobs / dt)
            Nf = int(N / Nt)
            if verbose:
                print(f"Attempting wavelet duration: {tmp}, Nf: {Nf}, Nt: {Nt}")
            if (Nt % 2 == 0) and (Nf % 2 == 0):
                found_wavelet = True
                break
        
        if not found_wavelet:
            raise ValueError(f"Could not find suitable wavelet parameters for even numbered Nf and Nt in the range given ({t_min}, {t_max}).")
        
        return (Nf, Nt, wavelet_duration)
    
    @property
    def active_slice_f(
        self,
    ) -> slice:
        """Slice of the full layer axis selecting ``[ind_min_f, ind_max_f]``."""
        return slice(self.ind_min_f, self.ind_max_f + 1)
    
    @property
    def active_slice_t(
        self,
    ) -> slice:
        """Slice of the full pixel axis selecting ``[ind_min_t, ind_max_t]``."""
        return slice(self.ind_min_t, self.ind_max_t + 1)

    @property
    def active_slice(
        self,
    ) -> slice:
        """``(active_slice_f, active_slice_t)`` for a full ``(..., Nf, Nt)`` array."""
        return (self.active_slice_f, self.active_slice_t)
    
    @property
    def Nf_active(self) -> int:
        """Number of active frequency layers, ``ind_max_f - ind_min_f + 1``."""
        sl = self.active_slice_f
        return sl.stop - sl.start

    @property
    def Nt_active(self) -> int:
        """Number of active time pixels, ``ind_max_t - ind_min_t + 1``."""
        sl = self.active_slice_t
        return sl.stop - sl.start

    def __getstate__(self):
        """Exclude derived WDM gather/fold caches from copies and pickles."""
        state = dict(self.__dict__)
        state.pop("_fold_shift_map_cache", None)
        state.pop("_sparse_psd_fold_cache", None)
        state.pop("_sparse_psd_layer_fold_cache", None)
        state.pop("_wdm_layer_quadrature_cache", None)
        # Layer-mask caches (PR #82): device arrays keyed on the CURRENT
        # device; a copy on another device must rebuild them.
        state.pop("_freq_layer_mask_cache", None)
        state.pop("_time_layer_mask_cache", None)
        # per-device copies of the window / fold map (wdmtransform)
        state.pop("_device_copy_cache", None)
        return state

    def __eq__(self, value):
        if not isinstance(value, WDMSettings):
            return False
        return (
            (value.Nt == self.Nt) and (value.Nf == self.Nf)
            and (value.layer_dt == self.layer_dt) and (value.layer_df == self.layer_df)
            and (value.data_dt == self.data_dt)
            and (value.ind_min_t == self.ind_min_t)
            and (value.ind_max_t == self.ind_max_t)
            and (value.ind_min_f == self.ind_min_f)
            and (value.ind_max_f == self.ind_max_f)
            and (bool(getattr(value, "is_complex", False)) == bool(self.is_complex))
        )

    def eq_without_inds(self, value):
        """Whether ``value`` is a WDM grid equal to this one, ignoring the active box.

        Compares ``Nf``, ``Nt``, ``layer_dt``, ``layer_df``, ``data_dt`` and
        ``is_complex`` (not ``t0``; :meth:`sub_box_slices` checks ``t0`` itself).
        ``==`` additionally compares the four ``ind_*`` indices.

        Args:
            value: Object to compare.

        Returns:
            bool: ``True`` when the grids match.
        """
        if not isinstance(value, WDMSettings):
            return False
        return (
            (value.Nt == self.Nt) and (value.Nf == self.Nf)
            and (value.layer_dt == self.layer_dt) and (value.layer_df == self.layer_df)
            and (value.data_dt == self.data_dt)
            and (bool(getattr(value, "is_complex", False)) == bool(self.is_complex))
        )

    def sub_box_slices(self, other: "WDMSettings") -> Tuple[slice, slice]:
        """``(f_slice, t_slice)`` locating ``other``'s active box inside this one's.

        Slices are RELATIVE to this settings' active box (the axes of an
        ``(Nf_active, Nt_active)`` array), i.e. the index
        :meth:`get_slice` / :meth:`DomainBase.get_array_slice` take. The ONE
        place the sub-box rules live, shared by the WDM ``add_signal`` path and
        :meth:`AnalysisContainer._slice_wdm_to_template`: both must share the
        wavelet grid (``Nf``, ``Nt``, ``dt``, ``t0``) and ``other``'s box must
        lie inside this one's; anything else raises ``ValueError``.
        """
        if not self.eq_without_inds(other) or abs(
            float(other.t0) - float(self.t0)
        ) > 1e-6 * float(self.data_dt):
            raise ValueError(
                "WDM template and data must share the wavelet grid (Nf, Nt, dt, "
                f"t0); got template Nf={other.Nf} Nt={other.Nt} dt={other.data_dt} "
                f"t0={other.t0} vs data Nf={self.Nf} Nt={self.Nt} dt={self.data_dt} "
                f"t0={self.t0}."
            )
        f0 = int(other.ind_min_f) - int(self.ind_min_f)
        f1 = int(other.ind_max_f) - int(self.ind_min_f) + 1
        t0 = int(other.ind_min_t) - int(self.ind_min_t)
        t1 = int(other.ind_max_t) - int(self.ind_min_t) + 1
        if f0 < 0 or t0 < 0 or f1 > int(self.Nf_active) or t1 > int(self.Nt_active):
            raise ValueError(
                f"WDM template box f[{other.ind_min_f}:{other.ind_max_f}] "
                f"t[{other.ind_min_t}:{other.ind_max_t}] is not inside the data box "
                f"f[{self.ind_min_f}:{self.ind_max_f}] "
                f"t[{self.ind_min_t}:{self.ind_max_t}]."
            )
        return slice(f0, f1), slice(t0, t1)

    def get_slice(self, index: tuple) -> "WDMSettings":
        """Settings for a SUB-BOX of this settings' active box.

        ``index = (f_slice, t_slice)`` is RELATIVE to the active box -- the
        same index :meth:`DomainBase.get_array_slice` applies to the stored
        ``(Nf_active, Nt_active)`` array, so array and settings stay in step.
        The grid (``Nf``, ``Nt``, ``dt``, ``t0``, window) is unchanged; only
        ``ind_min/max_{f,t}`` narrow.

        The result is rebuilt through the PHYSICAL ``min/max_*`` setters,
        because every ``WDMSignal`` reconstructs its settings from
        ``(args, kwargs)`` and ``kwargs`` carries those inputs, not the
        indices: setting ``ind_*`` directly would be undone by the next
        reconstruction. Half-bin offsets make the setters' ceil/floor land
        exactly on the requested integers; the result is checked.
        """
        if not isinstance(index, tuple) or len(index) != 2:
            raise ValueError("WDMSettings.get_slice expects (f_slice, t_slice).")
        f_sl, t_sl = index
        for sl in (f_sl, t_sl):
            if not isinstance(sl, slice) or sl.step not in (None, 1):
                raise ValueError(
                    "WDMSettings.get_slice: slices must be contiguous (step 1)."
                )
        f_lo, f_hi, _ = f_sl.indices(int(self.Nf_active))
        t_lo, t_hi, _ = t_sl.indices(int(self.Nt_active))
        if f_hi <= f_lo or t_hi <= t_lo:
            raise ValueError(f"WDMSettings.get_slice: empty slice {index!r}.")
        want = (
            int(self.ind_min_f) + f_lo,
            int(self.ind_min_f) + f_hi - 1,
            int(self.ind_min_t) + t_lo,
            int(self.ind_min_t) + t_hi - 1,
        )
        kw = dict(self.kwargs)
        kw.update(
            min_freq=None if want[0] == 0 else (want[0] - 0.5) * self.layer_df,
            max_freq=(want[1] + 0.5) * self.layer_df,
            min_time=None if want[2] == 0 else (want[2] - 0.5) * self.layer_dt,
            max_time=(want[3] + 0.5) * self.layer_dt,
        )
        new = WDMSettings(*self.args, **kw)
        got = (int(new.ind_min_f), int(new.ind_max_f), int(new.ind_min_t), int(new.ind_max_t))
        if got != want:
            raise RuntimeError(
                f"WDMSettings.get_slice: rebuilt box {got} != requested {want}"
            )
        return new

    @property
    def basis_shape(self) -> tuple:
        """Full grid shape ``(Nf, Nt)``."""
        return (self.Nf, self.Nt)
    
    @property
    def basis_shape_active(self) -> tuple:
        """Active box shape ``(Nf_active, Nt_active)``: the trailing axes of a signal array."""
        return (self.Nf_active, self.Nt_active)
    
    @property
    def t_td_arr(self) -> np.ndarray:
        """Time-domain sample times ``k * data_dt`` inside ``[ind_min_t, ind_max_t] * layer_dt``."""
        _tmp = self.xp.arange(self.N) * self.data_dt
        tmp = _tmp[(_tmp >= self.ind_min_t * self.layer_dt) & (_tmp <= self.ind_max_t * self.layer_dt)]
        return tmp
    
    @property
    def t_arr(self) -> np.ndarray:
        """Times ``n * layer_dt`` of the active pixels (s from the grid start; ``t0`` not added).

        ``n * layer_dt`` is the time centre of wavelet ``n`` (the reference time
        the lookup-table builds and evaluators use).
        """
        return self.xp.arange(self.Nt)[self.active_slice_t] * self.layer_dt

    @property
    def f_arr(self) -> np.ndarray:
        """Centre frequencies ``m * layer_df`` of the active layers (Hz)."""
        return self.xp.arange(self.Nf)[self.active_slice_f] * self.layer_df
    
    @property
    def f_arr_edges(self) -> np.ndarray:
        """``(ind_min_f + k) * layer_df`` for ``k = 0 .. Nf_active``: ``pcolormesh`` edges (Hz)."""
        return (self.xp.arange(self.Nf_active + 1) + self.ind_min_f) * self.layer_df
    @property
    def t_arr_edges(self) -> np.ndarray:
        """``(ind_min_t + k) * layer_dt`` for ``k = 0 .. Nt_active``: ``pcolormesh`` edges (s)."""

        return (self.xp.arange(self.Nt_active + 1) + self.ind_min_t) * self.layer_dt

    def phitilde(self, omega, dOmega):
        r"""Meyer-type WDM window :math:`\tilde{\phi}(\omega)` in the frequency domain.

        With :math:`A` = ``self.A`` (set to ``dOmega / 4`` by :meth:`setup_window`),
        :math:`B = \Delta\Omega - 2A` and :math:`d` = ``WAVELET_FILTER_CONSTANT``
        (4):

        .. math::

            \tilde{\phi}(\omega) = \frac{1}{\sqrt{\Delta\Omega}} \times
            \begin{cases}
                1 & |\omega| < A \\
                \cos\left[\frac{\pi}{2}\, I_x(d, d)\right],
                    \ x = \frac{|\omega| - A}{B} & A \le |\omega| < A + B \\
                0 & \text{otherwise}
            \end{cases}

        where :math:`I_x(d, d)` is the regularized incomplete beta function
        (``scipy.special.betainc``).

        Args:
            omega (``xp.ndarray``): Angular frequencies, 1-D.
            dOmega (float): Angular layer width :math:`\Delta\Omega`.

        Returns:
            ``xp.ndarray``: :math:`\tilde{\phi}` at ``omega``, same length.
        """
        # TODO/DOCS: parameters ``A``, ``B`` and ``WAVELET_FILTER_CONSTANT``
        # control the window roll-off; the exact convention follows the WDM
        # transform of Cornish & Romano. Verify against the cited reference.
        insDOM = 1. / np.sqrt(dOmega)
        A = self.A
        B = dOmega - 2 * A

        z = self.xp.zeros(omega.shape[0])
        beta_inc_calc = (np.abs(omega) >= A) & (np.abs(omega) < A+B)
        # Clip to [0, 1] — at omega = ±(A+B) the mathematical value is 1.0 but
        # float arithmetic can overshoot by 1 ULP, and scipy.special.betainc
        # returns NaN for any x > 1.0.
        x = np.clip((np.abs(omega[beta_inc_calc])-A)/B, 0.0, 1.0)
        y = special.betainc(self.WAVELET_FILTER_CONSTANT, self.WAVELET_FILTER_CONSTANT, x)
        z[beta_inc_calc] = insDOM*np.cos(y*np.pi/2.0)
        z[(np.abs(omega) < A)] = insDOM
        #breakpoint()
        return z
    
    def get_Cmn(self, m: np.array[int], n: np.array[int]) -> np.array[int]:
        """WDM phase factor :math:`C_{mn}`: ``1`` for even ``(m + n)``, ``1j`` for odd.

        Args:
            m: Layer indices (broadcast with ``n``; scalars are promoted to 1-D).
            n: Pixel indices, same shape as ``m``.

        Returns:
            ``xp.ndarray``: Complex array of the shape of ``m``.
        """
        m_in = self.xp.atleast_1d(m)
        n_in = self.xp.atleast_1d(n)
        output = self.xp.zeros(m_in.shape, dtype=complex)
        is_even = ((m_in + n_in) % 2) == 0
        output[is_even] = 1.0
        output[~is_even] = 1j
        return output

    # TODO check: keep/remove/document -- raises NotImplementedError on entry; no callers
    def wavelet(self, N: int, in_fd: Optional[bool] = True) -> np.ndarray:
        """Single WDM wavelet (not implemented; raises ``NotImplementedError``)."""
        raise NotImplementedError
        # Nt * Nf is even 
        # assert (self.Nt * self.Nf) % 2 == 0
        base_window = self.window[:-1]
        omega_N = (self.xp.arange(self.N) - int(self.N / 2)) * self.domega
        wavelet_N = 1 / np.sqrt(2.) * self.phitilde(omega_N)
        
        if in_fd:
            return wavelet_N
        else:
            return self.xp.fft.ifft(wavelet_N) / self.norm

    def get_shift_map(self, m: np.ndarray[int]) -> np.ndarray:
        """Return the rFFT bin map ``m * Nt/2 + arange(-Nt/2, Nt/2)`` of each layer.

        Row ``i`` lists the (unfolded, possibly negative or past-Nyquist) Fourier bins
        that layer ``m[i]`` reads; :meth:`fold_shift_map` folds them into
        ``[0, N/2]`` and :meth:`WDMSignal.wdm_to_fd` scatters into them.

        Args:
            m: Layer indices, 1-D ``(n_layers,)`` or 2-D ``(n_layers, Nt)`` (each row
                constant, as the callers build it).

        Returns:
            ``xp.ndarray``: Integer array of shape ``(n_layers, Nt)``.

        Raises:
            ValueError: If ``m`` is not 1-D or 2-D.
        """
        if m.ndim == 1:
            m_in = m[:, None]
        elif m.ndim == 2:
            m_in = m
        else:
            raise ValueError("m must be 1D or 2D array.")

        return m_in * int(self.Nt / 2) + self.xp.arange(-int(self.Nt / 2),  int(self.Nt / 2))[None, :]

    def fold_shift_map(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Cached rFFT gather map used by the FD -> WDM transform.

        The transform only ever reads the Fourier-domain array at the bins
        named by this map, so it is both the index set
        :meth:`~lisatools.domains.FDSignal.wdmtransform` gathers with and the
        *only* frequencies a PSD has to be evaluated at before being folded
        (see :func:`lisatools.sensitivity.get_sensitivity`). For a narrow
        active band that is a small fraction of the full rFFT grid -- on the
        stock 768x1024 / 0.3-8 mHz noise grid, 30,720 of 393,217 bins.

        The map depends only on the wavelet grid and the active band, so it is
        computed once and cached; the cache is keyed on those values and
        recomputes if they are reassigned.

        Returns:
            ``(m_special_1d, k, herm, unique_k)`` -- the layer indices being
            transformed, the ``(n_special, Nt)`` rFFT bin map (already folded
            into ``[0, N/2]``), the mask of bins that were mirrored (and so
            need conjugating for non-PSD input), and the sorted unique bins.
        """
        key = (
            self.Nf, self.Nt, self.N,
            self.ind_min_f, self.ind_max_f,
        )
        cached = getattr(self, "_fold_shift_map_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]

        # Only transform layers inside the active band [ind_min_f, ind_max_f].
        # When ind_min_f == 0 we additionally need layer Nf because the m=0
        # odd-n slots of the final w_mn are sourced from layer Nf's even-n IFFT.
        if self.ind_min_f == 0:
            m_special_1d = self.xp.concatenate([
                self.xp.arange(self.ind_min_f, self.ind_max_f + 1),
                self.xp.array([self.Nf]),
            ])
        else:
            m_special_1d = self.xp.arange(self.ind_min_f, self.ind_max_f + 1)
        m_special = self.xp.repeat(m_special_1d[:, None], self.Nt, axis=-1)

        # removed zero frequency and mirrored
        k = self.get_shift_map(m_special)
        neg_k = (k < 0)
        over_k = (k > int(self.N / 2))
        k[neg_k] = self.xp.abs(k[neg_k])
        k[over_k] = self.N - k[over_k]
        herm = neg_k | over_k
        unique_k = self.xp.unique(k)

        out = (m_special_1d, k, herm, unique_k)
        self._fold_shift_map_cache = (key, out)
        return out

    @property
    def fold_frequency_indices(self) -> np.ndarray:
        """Sorted unique rFFT bins the WDM fold reads (see :meth:`fold_shift_map`)."""
        return self.fold_shift_map()[3]

    def sparse_psd_fold_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Cached data for folding a PSD sampled only where WDM reads it.

        A stationary PSD fold does not need an rFFT-sized, mostly-zero array or
        an :class:`FDSignal`.  It only needs the unique bins in
        :attr:`fold_frequency_indices`, a map from the WDM gather matrix back
        into that compact array, and the fixed squared WDM window.

        Returns:
            ``(frequencies, compact_gather, window_factor)`` where
            ``frequencies`` has length ``len(fold_frequency_indices)``,
            ``compact_gather`` has shape ``(n_special_layers, Nt)``, and
            ``window_factor = window**2 * pi * data_dt``.  All arrays live on
            the settings backend.

        Note:
            This cache is CPU-oriented for now.  Keeping the primitive backend
            agnostic makes a later single-GPU implementation local to this
            method rather than to every stochastic model that uses it.
        """
        key = (
            self.Nf,
            self.Nt,
            self.N,
            self.data_dt,
            self.ind_min_f,
            self.ind_max_f,
        )
        cached = getattr(self, "_sparse_psd_fold_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]

        _, k, _, unique_k = self.fold_shift_map()
        # ``rfftfreq`` uses arange(N//2+1) * (1 / (N*dt)); retain that
        # operation order so evaluating a PSD here produces the same floating
        # point frequencies as the dense reference path.
        frequencies = unique_k * (1.0 / (self.N * self.data_dt))
        compact_gather = self.xp.searchsorted(unique_k, k)
        window_factor = self.window ** 2 * np.pi * self.data_dt
        out = (frequencies, compact_gather, window_factor)
        self._sparse_psd_fold_cache = (key, out)
        return out

    def sparse_psd_fold_data_for_layers(
        self, frequency_indices
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compact exact-fold data for selected active WDM layers.

        The ordinary sparse fold evaluates every Fourier bin touched by every
        active layer.  A component known to matter only on a strict subband
        can use this view to evaluate only the Fourier bins touched by those
        layers.  The final array contains their indices in the full sparse
        frequency grid, which lets callers slice cached frequency invariants.
        Cached on the last requested index set.

        Args:
            frequency_indices (array-like of int): Active-layer indices in
                ``[0, Nf_active)``, 1-D.

        Returns:
            tuple: ``(frequencies, local_gather, window_factor, global_indices)``:
            the Fourier frequencies those layers read (Hz, NumPy), the
            ``(len(frequency_indices), Nt)`` gather map into them, the window factor
            of :meth:`sparse_psd_fold_data`, and the positions of ``frequencies``
            inside :attr:`fold_frequency_arr`.

        Raises:
            ValueError: If ``frequency_indices`` is not 1-D.
            IndexError: If an index lies outside ``[0, Nf_active)``.
        """
        indices = np.asarray(frequency_indices, dtype=np.int64)
        if indices.ndim != 1:
            raise ValueError("frequency_indices must be one-dimensional.")
        if indices.size and (
            np.any(indices < 0) or np.any(indices >= self.Nf_active)
        ):
            raise IndexError(
                f"frequency_indices must lie in [0, {self.Nf_active})."
            )
        key = tuple(map(int, indices))
        cached = getattr(self, "_sparse_psd_layer_fold_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]

        frequencies, compact_gather, window_factor = self.sparse_psd_fold_data()
        selected_gather = np.asarray(compact_gather)[indices]
        global_indices, local_gather = np.unique(
            selected_gather, return_inverse=True
        )
        local_gather = local_gather.reshape(selected_gather.shape)
        out = (
            np.asarray(frequencies)[global_indices],
            local_gather,
            window_factor,
            global_indices,
        )
        self._sparse_psd_layer_fold_cache = (key, out)
        return out

    @property
    def fold_frequency_arr(self) -> np.ndarray:
        """Frequencies at which an exact sparse stationary PSD is evaluated."""
        return self.sparse_psd_fold_data()[0]

    def fold_sparse_psd(
        self, psd: np.ndarray, *, frequency_indices=None
    ) -> np.ndarray:
        """Fold compact stationary PSD values into one WDM time column.

        Args:
            psd: PSD values whose last axis corresponds, in order, to
                :attr:`fold_frequency_arr`, or to the compact selected grid
                when ``frequency_indices`` is supplied. Leading batch/channel
                axes are supported.
            frequency_indices: Optional active-layer indices to fold.  Only
                Fourier bins touched by these layers are evaluated and the
                returned final axis has this selected length.

        Returns:
            ``xp.ndarray``: Real array with the same leading axes and a final axis of
            length :attr:`Nf_active` (or ``len(frequency_indices)``).  This is
            algebraically identical to the ``FDSignal.wdmtransform(..., is_psd=True)``
            column used previously, without allocating the full rFFT grid or the
            repeated WDM output.

        Raises:
            ValueError: If the last axis of ``psd`` does not match the frequency grid.
        """
        values = self.xp.asarray(psd)
        if frequency_indices is None:
            frequencies, compact_gather, window_factor = self.sparse_psd_fold_data()
        else:
            frequencies, compact_gather, window_factor, _ = (
                self.sparse_psd_fold_data_for_layers(frequency_indices)
            )
        if values.ndim == 0 or values.shape[-1] != frequencies.size:
            raise ValueError(
                "psd's last axis must have length len(fold_frequency_arr)="
                f"{frequencies.size}; got shape {values.shape}."
            )

        # Match FDSignal.wdmtransform(is_psd=True)'s arithmetic order: divide
        # by dt, multiply by window**2*pi*dt, sum, then divide by N.
        gathered = values[..., compact_gather] / self.data_dt
        folded = (gathered * window_factor).sum(axis=-1)
        folded /= self.N
        if frequency_indices is None:
            folded = folded[..., : self.Nf_active]
        return self.xp.real(folded)

    def wdm_layer_quadrature_data(
        self, node_count: int = 64
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Weighted frequency nodes for a fast proposal-dependent WDM fold.

        The exact stationary PSD fold contains ``Nt`` samples per frequency
        layer.  Partition the nonzero WDM window into ``node_count`` pieces,
        place one node at each piece's window-weighted mean frequency, and
        retain its exact total window weight.  This is deterministic and exact
        for a constant PSD, while resolving proposal-dependent curvature far
        better than a single layer-center sample. Under
        ``WDM_PSD_METHOD=layer_calibrated`` (set on the 9-month and 12-month runs) the
        stationary PSD columns built by ``sensitivity._wdm_stationary_psd_column`` (the
        sampled stochastic-foreground spectra) are evaluated this way, with 64 nodes;
        the fixed instrument bases use a one-off per-layer calibration instead. Cached
        on ``node_count`` and the grid.

        Args:
            node_count: Requested nodes per layer (positive integer). Default ``64``.

        Returns:
            tuple: ``(frequencies, weights)`` with shapes
            ``(Nf_active, K)`` and ``(K,)``, ``K = min(node_count, Nwindow)``,
            where ``Nwindow`` is the number of nonzero window samples. Multiplying a
            PSD evaluated at ``frequencies`` by ``weights`` and summing the last
            axis yields its calibrated WDM layer column.

        Raises:
            ValueError: If ``node_count`` is not a positive integer.
        """
        if isinstance(node_count, bool) or int(node_count) != node_count:
            raise ValueError("node_count must be a positive integer.")
        node_count = int(node_count)
        if node_count < 1:
            raise ValueError("node_count must be a positive integer.")
        key = (
            node_count,
            self.Nf,
            self.Nt,
            self.N,
            self.data_dt,
            self.ind_min_f,
            self.ind_max_f,
        )
        cached = getattr(self, "_wdm_layer_quadrature_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]

        frequencies, compact_gather, window_factor = self.sparse_psd_fold_data()
        frequencies = np.asarray(asnumpy(frequencies), dtype=float)
        compact_gather = np.asarray(asnumpy(compact_gather), dtype=np.int64)
        window_factor = np.asarray(asnumpy(window_factor), dtype=float)
        nonzero = np.flatnonzero(window_factor > 0.0)
        actual_node_count = min(node_count, nonzero.size)

        # ``fold_shift_map`` carries one extra Nyquist row when the active band
        # contains m=0.  A stationary per-layer column follows the established
        # sparse-fold convention and returns the ordinary active rows here.
        rows = frequencies[compact_gather][: self.Nf_active]
        node_frequencies = []
        node_weights = []
        for indices in np.array_split(nonzero, actual_node_count):
            weights = window_factor[indices]
            total = np.sum(weights)
            node_frequencies.append(
                np.sum(rows[:, indices] * weights[None, :], axis=1) / total
            )
            node_weights.append(total / (self.data_dt * self.N))
        out = (
            self.xp.asarray(np.stack(node_frequencies, axis=1)),
            self.xp.asarray(node_weights),
        )
        self._wdm_layer_quadrature_cache = (key, out)
        return out


    # def window_norm(self) -> float:
    #     dOmega_s = np.pi / self.Nf
    #     (2 * np.pi) / self.N 

    def setup_window(self):  # , forward: bool= True):
        """Build the default WDM analysis window and store it in :attr:`window` / :attr:`omega`.

        Sets ``omega = 2 pi / N * arange(-Nt/2, Nt/2)`` (length ``Nt``),
        ``A = pi / (4 Nf)`` and ``window = phitilde(omega, pi / Nf)`` (see
        :meth:`phitilde`). Called by the constructor when no ``window`` is given.
        """
        # *DX = (double*)malloc(sizeof(double)*(2*wdm->N))
        # zero frequency
        # REAL(DX,0) =  wdm->inv_root_dOmega
        # IMAG(DX,0) =  0.0
        T = self.data_dt * self.N
        dOmega_s = np.pi / self.Nf

        self.A = dOmega_s / 4.0
        self.omega = omega =  2 * np.pi / self.N * (self.xp.arange(-int(self.Nt / 2),  int(self.Nt / 2)))
        phif = self.phitilde(omega, dOmega_s)
        self.window = phif

        # 2 * sqrt(pi) / Nf

        # we apply this outside the window setup so the window is the same for forward and backward
        # if forward:
        #     self.window *= 2. / self.Nf

        assert 0.0 in omega

    @property
    def min_freq(self) -> float:
        """Requested lower band edge (Hz) or ``None``.

        Setting it snaps ``ind_min_f = max(0, ceil(min_freq / layer_df))`` (``0`` for
        ``None``); negative values raise ``ValueError``.
        """
        return self._min_freq

    @min_freq.setter
    def min_freq(self, value: Optional[float]):
        if value is not None and value < 0:
            raise ValueError("min_freq must be non-negative.")

        # self._min_freq = value
        # set it to the closest frequency bin
        self.min_freq_input = value
        if value is not None:
            self.ind_min_f = int(np.ceil(value / self.layer_df))
            if self.ind_min_f < 0:
                self.ind_min_f = 0
        else:
            self.ind_min_f = 0
        self._min_freq = value

    @property
    def max_freq(self) -> Optional[float]:
        """Requested upper band edge (Hz) or ``None``.

        Setting it snaps ``ind_max_f = min(Nf - 1, int(max_freq / layer_df))``
        (``Nf - 1`` for ``None``); negative values raise ``ValueError``.
        """
        return self._max_freq

    @max_freq.setter
    def max_freq(self, value: Optional[float]):
        if value is not None and value < 0:
            raise ValueError("max_freq must be non-negative.")

        # self._max_freq = value
        # set it to the closest frequency bin
        self.max_freq_input = value
        if value is not None:
            self.ind_max_f = int(value / self.layer_df)
            if self.ind_max_f > (self.Nf - 1):
                self.ind_max_f = (self.Nf - 1)
        else:
            self.ind_max_f = (self.Nf - 1)
        self._max_freq = value

    @property
    def min_time(self) -> float:
        """Requested lower time edge (s from the grid start) or ``None``.

        Setting it snaps ``ind_min_t = max(0, ceil(min_time / layer_dt))`` (``0`` for
        ``None``); negative values raise ``ValueError``.
        """
        return self._min_time

    @min_time.setter
    def min_time(self, value: Optional[float]):
        if value is not None and value < 0:
            raise ValueError("min_time must be non-negative.")

        # self._min_time = value
        # set it to the closest time bin
        self.min_time_input = value

        if value is not None:
            self.ind_min_t = int(np.ceil(value / self.layer_dt))
            if self.ind_min_t < 0:
                self.ind_min_t = 0
        else:
            self.ind_min_t = 0
        self._min_time = value

    @property
    def max_time(self) -> Optional[float]:
        """Requested upper time edge (s from the grid start) or ``None``.

        Setting it snaps ``ind_max_t = min(Nt - 1, int(max_time / layer_dt))``
        (``Nt - 1`` for ``None``); negative values raise ``ValueError``.
        """
        return self._max_time

    @max_time.setter
    def max_time(self, value: Optional[float]):
        if value is not None and value < 0:
            raise ValueError("max_time must be non-negative.")

        # self._max_time = value
        # set it to the closest time bin
        self.max_time_input = value
        if value is not None:
            self.ind_max_t = int(value / self.layer_dt)
            if self.ind_max_t > (self.Nt - 1):
                self.ind_max_t = (self.Nt - 1)
        else:
            self.ind_max_t = (self.Nt - 1)

        self._max_time = value

    @property
    def ind_min_f(self) -> int:
        """First active frequency layer (set via :attr:`min_freq`)."""
        return self._ind_min_f
    
    @ind_min_f.setter
    def ind_min_f(self, ind_min_f: int):
        if ind_min_f is None:
            ind_min_f = 0
        self._ind_min_f = ind_min_f

    @property
    def ind_max_f(self) -> int:
        """Last active frequency layer, inclusive (set via :attr:`max_freq`)."""
        return self._ind_max_f
    
    @ind_max_f.setter
    def ind_max_f(self, ind_max_f: int):
        if ind_max_f is None:
            ind_max_f = self.Nf - 1
        self._ind_max_f = ind_max_f

    @property
    def ind_min_t(self) -> int:
        """First active time pixel (set via :attr:`min_time`)."""
        return self._ind_min_t
    
    @ind_min_t.setter
    def ind_min_t(self, ind_min_t: int):
        if ind_min_t is None:
            ind_min_t = 0
        self._ind_min_t = ind_min_t

    @property
    def ind_max_t(self) -> int:
        """Last active time pixel, inclusive (set via :attr:`max_time`)."""
        return self._ind_max_t
    
    @ind_max_t.setter
    def ind_max_t(self, ind_max_t: int):
        if ind_max_t is None:
            ind_max_t = self.Nt - 1
        self._ind_max_t = ind_max_t

    def _mask_cache_device_tag(self) -> Optional[int]:
        """Current CUDA device id for the layer-mask caches; ``None`` on CPU."""
        cuda = getattr(self.xp, "cuda", None)
        if cuda is None:
            return None
        try:
            return int(cuda.runtime.getDevice())
        except Exception:
            return None

    @property
    def frequency_layer_mask(self) -> Optional[np.ndarray]:
        """Boolean ``(Nf,)`` mask of the active frequency layers (cached per CUDA device)."""
        # CACHED. This is a @property that allocated a fresh (Nf,) device array
        # on EVERY access, and diagnostic.inner_product touches it ~6x per call
        # (two existence checks, two inside np.array_equal -- which also forces a
        # device->host sync -- then two more selecting the branch). Measured 3780
        # allocations across 60 template_likelihood calls (63 each), costing
        # ~19.5 ms per likelihood INDEPENDENT of N_WIN (identical at N_WIN
        # 414,720 / 207,360 / 103,680). The mask depends only on (Nf,
        # active_slice_f), so it is cached and invalidated if either changes.
        #
        # The key also carries the CURRENT CUDA device (dev merge, 2026-09-29):
        # ``self.xp.zeros`` lands on whichever device is current, and one
        # settings object serves every GPU of a multi-device run, so a mask
        # cached on device 0 must not be handed to a kernel on device 1.
        # ``__getstate__`` drops both caches so copies rebuild them.
        key = (self.Nf, self.active_slice_f, self._mask_cache_device_tag())
        cached = getattr(self, "_freq_layer_mask_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        mask = self.xp.zeros(self.Nf, dtype=bool)
        mask[self.active_slice_f] = True
        try:
            object.__setattr__(self, "_freq_layer_mask_cache", (key, mask))
        except Exception:
            pass          # frozen/slotted settings: fall back to recomputing
        return mask
    
    @property
    def time_layer_mask(self) -> Optional[np.ndarray]:
        """Boolean ``(Nt,)`` mask of the active time pixels (cached per CUDA device)."""
        # Cached for the same reason as frequency_layer_mask above.
        key = (self.Nt, self.active_slice_t, self._mask_cache_device_tag())
        cached = getattr(self, "_time_layer_mask_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        mask = self.xp.zeros(self.Nt, dtype=bool)
        mask[self.active_slice_t] = True
        try:
            object.__setattr__(self, "_time_layer_mask_cache", (key, mask))
        except Exception:
            pass
        return mask
        
    @property
    def f_ind_array(self) -> np.ndarray:
        """Absolute layer indices of the active band, ``arange(ind_min_f, ind_max_f + 1)``."""
        return self.xp.arange(self.ind_min_f, self.ind_max_f + 1)

    @property
    def t_ind_array(self) -> np.ndarray:
        """Absolute pixel indices of the active band, ``arange(ind_min_t, ind_max_t + 1)``."""
        return self.xp.arange(self.ind_min_t, self.ind_max_t + 1)

    @staticmethod
    def get_associated_class():
        """Return :class:`WDMSignal`."""
        return WDMSignal

    @property
    def associated_class(self):
        """The paired signal class, :class:`WDMSignal`."""
        return self.get_associated_class()

    @property
    def kwargs(self) -> dict:
        """Keyword reconstruction arguments.

        ``t0``, ``oversample``, ``window``, ``omega``, the four band edges,
        ``is_complex`` and ``force_backend``; with :attr:`args`,
        ``WDMSettings(*s.args, **s.kwargs)`` rebuilds an equal object (the window
        is reused, not recomputed).
        """
        return dict(
            # t0 MUST ride along: every ``WDMSettings(*s.args, **s.kwargs)``
            # reconstruction (WDMSignal.__init__, the per-device settings
            # replicas in globalfit/stock/erebor/source_runtime.py) would
            # otherwise silently reset the data start time to 0.0. The stock
            # WDM global fits set ``wdm.t0 = data_t0`` before building the
            # chunked-het comps, whose ``t_obs_start``/``chunk_t_starts``
            # inherit it -- losing it here made every NON-PRIMARY GPU shard's
            # comp replica evaluate orbits + source phases at an epoch
            # shifted by -data_t0 (the 2026-08 multi-GPU VGB scoring bug:
            # in-model acceptance halved, shard-1 walker ll drift). Repro:
            # scripts/gb_chunked_het/gb_shard_inmodel_repro.py (Phase G).
            t0=self.t0,
            oversample=self.oversample,
            window=self.window,
            # norm=self.norm,
            omega=self.omega,
            min_freq=self.min_freq,
            max_freq=self.max_freq,
            min_time=self.min_time,
            max_time=self.max_time,
            is_complex=self.is_complex,
            force_backend=self.backend
        )

    @property
    def args(self) -> tuple:
        """Positional reconstruction arguments ``(Nf, Nt, data_dt)``."""
        return (self.Nf, self.Nt, self.data_dt)

    @property
    def differential_component(self) -> float:
        """Inner-product factor: ``0.25`` (real WDM) or ``0.125`` (``is_complex``).

        ``diagnostic.inner_product`` computes ``4 * sum(...) * differential_component``,
        so the real WDM inner product is the plain coefficient sum.
        """
        # Real-only WDM is a tight frame; inner_product uses
        # 4 * sum(...) * differential_component. The complex/quadrature WDM
        # sums Re*Re + Im*Im, which is approximately 2x the real-only power
        # (the Hilbert companion has matching variance), so halve the
        # differential to keep the inner-product value invariant.
        return 0.125 if getattr(self, "is_complex", False) else 0.25

    @property
    def total_terms(self) -> int:
        """Number of basis elements in the **active** band (``Nf_active * Nt_active``).

        Matches the FD semantics where ``total_terms`` is the active-band
        size, not the full grid. Sensitivity-matrix and related code use
        this to size per-pixel C++ buffers; allocating ``Nf*Nt`` instead
        produces a length mismatch against ``len(f_arr) * len(t_arr)``.
        """
        return self.Nf_active * self.Nt_active

    def apply_frequency_layer_mask(self, arr: np.ndarray) -> np.ndarray:
        """Restrict ``arr`` to the active frequency layers along the penultimate axis.

        An array whose penultimate axis already has ``Nf_active`` entries is
        returned unchanged; a 2-D full-grid ``(Nf, Nt)`` array is masked with
        :attr:`frequency_layer_mask`.

        Args:
            arr: Array whose penultimate axis is the WDM layer axis.

        Returns:
            ``xp.ndarray``: ``arr`` or its active-layer rows.

        Raises:
            IndexError: If ``arr`` is 1-D (``arr.shape[-2]`` is read first, so the
                ``ValueError`` branch for 1-D input is never reached).
            NotImplementedError: For a full-grid array with more than two dimensions.
        """
        if self.frequency_layer_mask is None or arr.shape[-2] == self.Nf_active:
            return arr
        elif arr.ndim == 1:
            raise ValueError("arr must be at least 2D to apply frequency layer mask.")
        elif arr.ndim == 2:
            return arr[self.frequency_layer_mask]
        elif arr.ndim > 2:
            raise NotImplementedError
            assert arr.shape[-2] == self.frequency_layer_mask.shape[0], "Last dimension of arr must match length of frequency_layer_mask."
            dims_transpose = tuple(np.roll(np.arange(arr.ndim)))
            _arr = arr.transpose(dims_transpose)
            new_arr = _arr[self.frequency_layer_mask]
            dims_back = tuple(np.roll(np.arange(arr.ndim), -1))
            new_arr = new_arr.transpose(dims_back)
            return new_arr


class CoarseWDMSettings(WDMSettings):
    """Time-coarsened model view of a fine :class:`WDMSettings` grid.

    The underlying WDM transform remains defined by ``Nf``/``Nt`` and the
    original window.  Only the active model time axis is reduced, into
    non-overlapping cells containing at most ``coarse_Q`` fine columns.  This
    object is therefore a sensitivity-model descriptor, not a WDM transform
    target; use :attr:`fine_settings` for transforms. It is the basis of the
    coarse real-WDM noise likelihood (:mod:`lisatools.coarsewdm`; stock knobs
    ``COARSE_Q`` / ``COARSE_GPU_MODE``, off on the 9-month and 12-month runs).

    Args:
        Nf: Number of frequency layers.
        Nt: Number of fine time pixels.
        dt: Time-domain sample step (s).
        *args: Further positional :class:`WDMSettings` arguments (``t0``, ...).
        coarse_Q: Keyword-only: fine active columns per coarse cell (``>= 1``).
            The last cell holds the remainder.
        **kwargs: Forwarded to :class:`WDMSettings` (band edges, ``force_backend``).

    Raises:
        TypeError: If ``coarse_Q`` is not an integer.
        ValueError: If ``coarse_Q < 1``.

    Attributes:
        coarse_Q (int): Columns per coarse cell.

    Example:
        >>> from lisatools.domains import CoarseWDMSettings, WDMSettings
        >>> fine = WDMSettings(Nf=64, Nt=512, dt=5.0, force_backend="cpu")
        >>> coarse = CoarseWDMSettings.from_fine(fine, 8)
        >>> coarse.basis_shape_active
        (64, 64)
    """

    def __init__(
        self,
        Nf: int,
        Nt: int,
        dt: float,
        *args,
        coarse_Q: int,
        **kwargs,
    ):
        if isinstance(coarse_Q, bool) or not isinstance(coarse_Q, (int, np.integer)):
            raise TypeError("coarse_Q must be an integer.")
        if int(coarse_Q) < 1:
            raise ValueError("coarse_Q must be >= 1.")
        self.coarse_Q = int(coarse_Q)
        super().__init__(Nf, Nt, dt, *args, **kwargs)

    @classmethod
    def from_fine(cls, settings: WDMSettings, Q: int) -> "CoarseWDMSettings":
        """Construct a coarse model view from a real fine WDM grid.

        The fine object is kept (``fine_settings`` returns it) so fine-grid caches
        are reused while the coarse view lives.

        Args:
            settings: A plain (not coarse) real :class:`WDMSettings`.
            Q: Columns per coarse cell.

        Returns:
            CoarseWDMSettings: The coarse view, same grid and active box.

        Raises:
            TypeError: If ``settings`` is not a plain :class:`WDMSettings`.
            ValueError: If ``settings.is_complex`` is set.
        """
        if not isinstance(settings, WDMSettings) or isinstance(settings, cls):
            raise TypeError("settings must be a plain WDMSettings instance.")
        if getattr(settings, "is_complex", False):
            raise ValueError("Coarse WDM likelihood currently supports real WDM only.")
        out = cls(*settings.args, coarse_Q=Q, **settings.kwargs)
        # Runtime identity lets expensive fine-grid component caches be reused
        # while this coarse view is alive. Reconstruction from args/kwargs still
        # works and simply recreates an equivalent plain fine settings object.
        out._fine_settings_source = settings
        return out

    @property
    def fine_settings(self) -> WDMSettings:
        """Reconstruct the plain fine-grid settings used by WDM transforms."""
        source = getattr(self, "_fine_settings_source", None)
        if source is not None:
            return source
        return WDMSettings(*super().args, **super().kwargs)

    @property
    def Ncoarse(self) -> int:
        """Number of coarse cells, ``ceil(Nt_active / coarse_Q)``."""
        return int(math.ceil(self.Nt_active / self.coarse_Q))

    @property
    def cell_starts(self) -> np.ndarray:
        """Fine active-time offsets at which coarse cells begin."""
        return np.arange(0, self.Nt_active, self.coarse_Q, dtype=int)

    @property
    def cell_sizes(self) -> np.ndarray:
        """Number of fine active columns in each coarse cell."""
        starts = self.cell_starts
        return np.minimum(self.coarse_Q, self.Nt_active - starts)

    @property
    def basis_shape_active(self) -> tuple:
        """Coarse active shape ``(Nf_active, Ncoarse)``."""
        return (self.Nf_active, self.Ncoarse)

    @property
    def fine_t_arr(self) -> np.ndarray:
        """Fine active pixel times (:attr:`WDMSettings.t_arr`)."""
        return super().t_arr

    @property
    def t_arr(self) -> np.ndarray:
        """Cell-mean time of each coarse cell (s from the grid start)."""
        return self.cell_mean(self.fine_t_arr)

    @property
    def t_arr_edges(self) -> np.ndarray:
        """Fine pixel edges at the cell boundaries, ``(Ncoarse + 1,)`` (s)."""
        fine_edges = super().t_arr_edges
        inds = np.concatenate([self.cell_starts, np.array([self.Nt_active])])
        return fine_edges[self.xp.asarray(inds)]

    @property
    def total_terms(self) -> int:
        """Number of coarse basis elements, ``Nf_active * Ncoarse``."""
        return self.Nf_active * self.Ncoarse

    def cell_mean(self, arr, axis: int = -1):
        """Average ``arr`` over coarse cells along a fine active-time axis.

        Args:
            arr (array-like): Array with ``Nt_active`` entries along ``axis``.
            axis: The fine time axis. Default the last.

        Returns:
            ``xp.ndarray``: Same shape with ``axis`` reduced to ``Ncoarse`` (the array
            module follows ``arr``).

        Raises:
            ValueError: If ``axis`` is out of range or does not have ``Nt_active``
                entries.
        """
        xp = get_array_module(arr)
        arr = xp.asarray(arr)
        axis = axis if axis >= 0 else arr.ndim + axis
        if axis < 0 or axis >= arr.ndim:
            raise ValueError(f"axis {axis} is out of bounds for an array of ndim {arr.ndim}.")
        if arr.shape[axis] != self.Nt_active:
            raise ValueError(
                f"cell_mean axis has length {arr.shape[axis]}; expected "
                f"Nt_active={self.Nt_active}."
            )
        pieces = []
        for start, size in zip(self.cell_starts, self.cell_sizes):
            index = [slice(None)] * arr.ndim
            index[axis] = slice(int(start), int(start + size))
            pieces.append(xp.mean(arr[tuple(index)], axis=axis))
        return xp.stack(pieces, axis=axis)

    def evaluate_time_modulation(self, modulation):
        """Cell-average a callable or fine-grid modulation exactly.

        A callable is evaluated at :attr:`fine_t_arr`. Arrays with at most two
        dimensions are returned unchanged. For higher-dimensional arrays the last
        axis may hold ``Nt`` (sliced to the active box first), ``Nt_active``
        (cell-averaged) or ``Ncoarse`` (returned as is) samples.

        Args:
            modulation (callable or array-like): Time modulation.

        Returns:
            ``xp.ndarray``: The modulation on the coarse time axis.

        Raises:
            ValueError: If the last axis matches none of the three lengths.
        """
        xp = self.xp
        values = modulation(self.fine_t_arr) if callable(modulation) else xp.asarray(modulation)
        values = xp.asarray(values)
        if values.ndim <= 2:
            return values
        if values.shape[-1] == self.Nt:
            values = values[..., self.active_slice_t]
        if values.shape[-1] == self.Ncoarse:
            return values
        if values.shape[-1] != self.Nt_active:
            raise ValueError(
                "time modulation must have fine Nt/Nt_active or coarse Ncoarse "
                f"samples; got {values.shape[-1]}."
            )
        return self.cell_mean(values)

    def __eq__(self, value):
        return (
            isinstance(value, CoarseWDMSettings)
            and self.coarse_Q == value.coarse_Q
            and super().__eq__(value)
        )

    def eq_without_inds(self, value):
        """:meth:`WDMSettings.eq_without_inds` plus a coarse view with equal ``coarse_Q``."""
        return (
            isinstance(value, CoarseWDMSettings)
            and self.coarse_Q == value.coarse_Q
            and super().eq_without_inds(value)
        )

    @property
    def kwargs(self) -> dict:
        """:attr:`WDMSettings.kwargs` plus ``coarse_Q``."""
        return {**super().kwargs, "coarse_Q": self.coarse_Q}


class WDMSignal(WDMSettings, DomainBase):
    """WDM wavelet array wrapper paired with :class:`WDMSettings`.

    The data / residual / template type of the WDM global fits. Inherits
    :class:`WDMSettings`, so ``wdm.layer_df``, ``wdm.f_arr`` and the active-box
    indices are available directly. Produced by :meth:`TDSignal.wdmtransform` /
    :meth:`FDSignal.wdmtransform` (or ``signal.transform(wdm_settings)``); goes back
    with :meth:`wdm_to_fd` / :meth:`wdm_to_td` (full grid only).

    Args:
        arr: NumPy or CuPy array with trailing axes ``(Nf_active, Nt_active)``:
            ``(nchannels, Nf_active, Nt_active)`` or
            ``(nbatch, nchannels, Nf_active, Nt_active)``. A full-grid
            ``(nchannels, Nf, Nt)`` array (either axis full) is sliced to the active
            box (the slice indexes ``arr[:, f, t]``, i.e. assumes exactly one leading
            axis).
        settings: :class:`WDMSettings` describing the wavelet grid.

    Raises:
        AssertionError: If a mismatched axis is not the full ``Nf`` / ``Nt``.

    Example:
        >>> import numpy as np
        >>> from lisatools.domains import WDMSettings, WDMSignal
        >>> ws = WDMSettings(Nf=64, Nt=512, dt=5.0, min_freq=1e-3, max_freq=1e-2,
        ...                  force_backend="cpu")
        >>> WDMSignal(np.zeros((3, 64, 512)), ws).shape
        (3, 6, 512)
    """

    # TEST back and forth
    # tmp_dat = np.zeros(wdm_set.N)
    # tmp_dat[0] = 1.0
    # _frq = np.fft.rfftfreq(wdm_set.N, wdm_set.data_dt)
    # tmp_dat_td = TDSignal(tmp_dat, TDSettings(wdm_set.N, wdm_set.data_dt))
    # tmp_dat_fd = tmp_dat_td.fft(FDSettings(_frq.shape[0], _frq[1] - _frq[0]))
    # check_1 = tmp_dat_td.fft().ifft()
    # tmp_dat_check_td_1 = tmp_dat_fd.wdmtransform(wdm_set).wdm_to_fd(tmp_dat_fd.settings).ifft()
    # tmp_dat_check_td = tmp_dat_td.wdmtransform(wdm_set).wdm_to_td()

    # assert np.allclose((_tmp := tmp_dat_check_td[:, 0]), np.ones_like(_tmp))
    # assert np.allclose((_tmp := tmp_dat_check_td[:, 1:]), np.zeros_like(_tmp))

    def __init__(self, arr, settings: WDMSettings):
        WDMSettings.__init__(self, *settings.args, **settings.kwargs)
        DomainBase.__init__(self, arr)

        # freq layers
        if self.arr.shape[-2] != self.Nf_active or self.arr.shape[-1] != self.Nt_active:
            if self.arr.shape[-2] != self.Nf_active:
                assert arr.shape[-2] == self.Nf
                f_slice = self.active_slice_f
            else:
                f_slice = slice(None)

            if self.arr.shape[-1] != self.Nt_active:
                assert arr.shape[-1] == self.Nt
                t_slice = self.active_slice_t
            else:
                t_slice = slice(None)

            _arr = self._arr.copy()
            del self._arr
            self.arr = _arr[:, f_slice, t_slice]

    @property
    def settings(self) -> WDMSettings:
        """A fresh :class:`WDMSettings` matching this signal's grid and active box."""
        return WDMSettings(*self.args, **self.kwargs)

    def __repr__(self) -> str:
        return (
            f"WDMSignal(Tobs={self.Tobs}, dt={self.data_dt}, t0={self.t0}, "
            f"Nt={self.Nt}, Nf={self.Nf}, oversample={self.oversample}, "
            f"backend={self.backend_name.split('_')[-1]})"
        )
    
    def wdm_to_td(self, settings=None, window=None):
        """Inverse-transform from WDM to time domain: :meth:`wdm_to_fd` then :meth:`FDSignal.ifft`.

        Args:
            settings (TDSettings, optional): Target time grid for the inverse FFT.
            window (array-like, optional): FD window passed to the inverse FFT.

        Returns:
            TDSignal: The time-domain signal.
        """
        return self.wdm_to_fd(settings=None).ifft(settings=settings, window=window)

    def wdm_to_fd(self, settings=None, window=None):
        """Inverse-transform from WDM to frequency domain.

        Requires the full grid: the frequency band is checked explicitly, and a
        narrowed time box fails on a shape mismatch. Works on an unbatched
        ``(nchannels, Nf, Nt)`` array. A complex (quadrature) signal is inverted
        from its real part with a warning. The monitor pads band-limited snapshots
        to the full grid before calling this.

        Args:
            settings: Optional target :class:`FDSettings`; it must be the rFFT grid of
                ``N`` samples (``N // 2 + 1`` bins, ``df = 1 / (N * data_dt)``). If
                ``None``, that grid is built on this signal's backend.
            window: Currently unused (the WDM analysis window stored on the
                signal is used internally).

        Returns:
            FDSignal: Shape ``(nchannels, N // 2 + 1)``.

        Raises:
            ValueError: If the active band is not the full ``[0, Nf - 1]``.
            AssertionError: If ``settings`` is not the rFFT grid of ``N`` samples.
        """
        if self.settings.Nf_active != self.settings.Nf:
            raise ValueError(
                "wdm_to_fd requires the WDM signal to span the full frequency "
                f"band [0, Nf-1]; got ind_min_f={self.settings.ind_min_f}, "
                f"ind_max_f={self.settings.ind_max_f} (Nf={self.settings.Nf})."
            )
        if getattr(self, "is_complex", False):
            warnings.warn(
                "wdm_to_fd called on a complex/quadrature WDMSignal; inverting "
                "the real part only (the quadrature companion is dropped). "
                "Convert explicitly to a real WDMSignal beforehand to silence "
                "this warning."
            )
            real_settings = WDMSettings(*self.args, **{**self.kwargs, "is_complex": False})
            real_signal = WDMSignal(self.xp.real(self.arr).copy(), real_settings)
            return real_signal.wdm_to_fd(settings=settings, window=window)
        if settings is None:
            _tmp_fd = np.fft.rfftfreq(self.N, self.data_dt)
            Nfd = len(_tmp_fd)
            df = _tmp_fd[1] - _tmp_fd[0]
            settings = FDSettings(Nfd, df, force_backend=self.backend)

        else:
            Nfd = len(_tmp_fd := np.fft.rfftfreq(self.N, self.data_dt))
            df = _tmp_fd[1] - _tmp_fd[0]
            assert settings.N == Nfd
            assert settings.df == df

        base_window = self.window.copy()

        m = self.xp.repeat(self.xp.arange(0, self.Nf)[:, None], self.Nt, axis=-1)
        n = self.xp.tile(self.xp.arange(self.Nt), (self.Nf, 1))

        m_special = self.xp.repeat(self.xp.arange(0, self.Nf + 1)[:, None], self.Nt, axis=-1)
        
        tmp_w_mn = self.xp.zeros((self.nchannels, self.Nf + 1, self.Nt))
        tmp_w_mn[:, 1:-1] = self.arr[:, 1:]
        
        # dc layer
        tmp_w_mn[:, 0, 0::2] = self.arr[:, 0, 0::2] * np.sqrt(2)
        # max freq layer
        tmp_w_mn[:, self.Nf, 0::2] = self.arr[:, 0, 1::2] * np.sqrt(2)

        lambda_coef = np.sqrt(np.pi / self.data_dt)

        m_here = self.xp.concatenate([m, self.xp.full((1, self.Nt), self.Nf)], axis=0)
        n_here = self.xp.concatenate([n, self.xp.array([self.xp.arange(self.Nt)])], axis=0)

        arr_fd = self.xp.zeros((self.nchannels, settings.N), dtype=complex)
        g_arr = tmp_w_mn * self.get_Cmn(m_here, n_here) * (-1) ** ((m_here + 1) * n_here)

        W_m = self.xp.fft.fft(g_arr, axis=-1)

        v = lambda_coef * base_window * W_m

        k = self.get_shift_map(m_special)
        
        k_even_m = k[0::2]
        k_odd_m = k[1::2]
        
        # for checking, but slow so commenting out
        # assert np.unique(k_even_m).shape[0] == np.prod(k_even_m.shape)
        # assert np.unique(k_odd_m).shape[0] == np.prod(k_odd_m.shape)
        keep_k_even = (k_even_m >= 0) & (k_even_m < settings.N)
        keep_k_odd = (k_odd_m >= 0) & (k_odd_m < settings.N)
        arr_fd[:, k_even_m[keep_k_even]] += v[:, 0::2][:, keep_k_even]
        arr_fd[:, k_odd_m[keep_k_odd]] += v[:, 1::2][:, keep_k_odd]

        # add dt factor for units
        arr_fd *= self.data_dt
        return FDSignal(arr_fd, settings)

    def transform(self, new_domain: DomainSettingsBase, window: np.ndarray | cp.ndarray = None):
        """Dispatch to the correct WDM-to-X conversion based on ``new_domain``.

        TD and STFT targets go through :meth:`wdm_to_fd` (and ``ifft``); a WDM
        target equal to this signal's settings returns ``self``, any other WDM
        target is reached through the FD domain (so this signal must span the full
        grid).

        ``window=None`` is forwarded as-is to the FINAL step of each chain:
        every step defaults its own window with the shape it needs.
        (Pre-filling a WDM-shaped ones array here broke the layered chains
        — e.g. it reached ``ifft`` on the FD intermediate, whose window must
        be FD-length. Same rule as :meth:`TDSignal.transform`.)

        Args:
            new_domain: Target settings (TD, FD, STFT or WDM).
            window (array-like, optional): Window for the final step.

        Returns:
            DomainBase: The signal in the target domain.

        Raises:
            ValueError: If ``new_domain`` is not a recognized settings class.
        """
        signal, window = self._coerce_transform_backend(new_domain, window)
        if signal is not self:
            return signal.transform(new_domain, window=window)

        if isinstance(new_domain, TDSettings):
            return self.wdm_to_fd(settings=None, window=None).ifft(settings=new_domain, window=window)

        elif isinstance(new_domain, FDSettings):
            return self.wdm_to_fd(settings=new_domain, window=window)

        elif isinstance(new_domain, STFTSettings):
            return self.wdm_to_fd(settings=None, window=None).ifft(settings=None, window=None).stft(settings=new_domain, window=window)

        elif isinstance(new_domain, WDMSettings):
            if new_domain == self.settings:
                return self
            else:
                return self.wdm_to_fd(settings=None, window=None).wdmtransform(
                    settings=new_domain, window=window
                )
        else:
            raise ValueError(f"new_domain type is not recognized {type(new_domain)}.")

    def heatmap(self, index: int = None, mag: bool = False, fig=None, ax=None, cax=None, add_cax=False, log: bool = False, **kwargs):
        """Produce a time-frequency heatmap of the WDM coefficients.

        Cells are drawn on :attr:`~WDMSettings.t_arr_edges` x
        :attr:`~WDMSettings.f_arr_edges`; rows are labelled ``X``, ``Y``, ``Z``.
        Unbatched signals only (``arr[i]`` is channel ``i``).

        Args:
            index: If given, plot only channel ``index`` on the supplied ``ax``.
                If ``None``, every channel is plotted on its own row.
            mag: If ``True``, plot ``|coeff|``; otherwise plot the signed value.
            fig: Existing :class:`matplotlib.figure.Figure`; required with ``index``,
                and with ``add_cax`` when ``ax`` is supplied (with neither ``fig`` nor
                ``ax`` given, a figure with one row per channel is created).
            ax: Existing axes; required when ``index`` is provided.
            cax: Optional axes for the colourbar.
            add_cax: If ``True``, create a new colourbar axes.
            log: If ``True``, plot ``log10(|coeff|)``.
            **kwargs: Forwarded to :func:`matplotlib.pyplot.pcolormesh` (default
                ``cmap`` is ``RdBu``).

        Returns:
            tuple: ``(fig, ax)``.
        """
        # if fig is not None or ax is not None:
        #     if fig is None or ax is None:
        #         raise ValueError("If providing fig or ax, must provide both.")

        # else:
        #     # fig and ax are None
        if "cmap" not in kwargs:
            kwargs["cmap"] = cm.RdBu

        if index is None:
            if fig is None and ax is None:
                fig, ax = plt.subplots(self.nchannels, 1, sharex=True, sharey=True)
            else:
                assert ax is not None

            try:
                len(ax)
            except TypeError:
                ax = [ax]
                
            for i, (ax_i, channel)  in enumerate(zip(ax, ["X", "Y", "Z"])):
                z = self.arr[i]
                x, y = self.t_arr_edges, self.f_arr_edges
                x = self.get(x)
                y = self.get(y)
                z = self.get(z)
                if mag:
                    z = np.abs(z)
                if log:
                    z = np.log10(np.abs(z))
                sc = ax_i.pcolormesh(
                    x, y, z, 
                    # extent=[self.t_arr.min(), self.t_arr.max(), self.f_arr.min(), self.f_arr.max()], 
                    **kwargs
                )
                ax_i.set_ylabel(channel)

        else:
            assert index is not None and fig is not None and ax is not None
            z = self.arr[index]
            x, y = self.t_arr_edges, self.f_arr_edges
            x = self.get(x)
            y = self.get(y)
            z = self.get(z)
            if mag:
                z = np.abs(z)
            if log:
                z = np.log10(np.abs(z))
            sc = ax.pcolormesh(
                x, y, z, 
                # extent=[self.t_arr.min(), self.t_arr.max(), self.f_arr.min(), self.f_arr.max()], 
                **kwargs
            )

        if add_cax:
            cax = fig.add_axes([0.9, 0.2, 0.05, 0.6])

        if add_cax or cax is not None:
            fig.colorbar(sc, cax=cax)
        
        plt.subplots_adjust(right=0.85, hspace=0.1)
        return fig, ax


# ---------------------------------------------------------------------------
# WDMLookupTable: restored verbatim from LAT f17d9a74 (domains.py:2498-3624) for the
# EMRI direct-to-WDM plan; only change: out-of-support f/fdot raise ValueError
# (the historical code dropped into breakpoint()).
# ---------------------------------------------------------------------------
import h5py

class _UniformCubicSpline:
    """Interpolating cubic spline on a uniform 1-D or 2-D grid, CPU (scipy) or GPU (cupyx).

    The B-spline coefficients are prefiltered ONCE (``ndimage.spline_filter``, mirror
    boundaries); a call is one ``ndimage.map_coordinates(order=3)``. Same API in scipy and
    cupyx, so the table evaluates on either backend (cupyx's RegularGridInterpolator has no
    cubic method). Called with ``pts`` of shape ``(..., 2)`` = ``(fdot, f_norm)`` for a 2-D
    table, or the ``f_norm`` array for a 1-D one. Points outside the grid return 0 (as the
    RegularGridInterpolator path's ``fill_value``).
    """

    def __init__(self, table, fdot_axis, f_axis, xp, gpu):
        if gpu:
            from cupyx.scipy import ndimage as ndi
        else:
            from scipy import ndimage as ndi
        self.ndi, self.xp = ndi, xp
        self.two_d = fdot_axis is not None
        tab = xp.asarray(table, dtype=xp.float64)
        if self.two_d:
            tab = tab.reshape(len(fdot_axis), len(f_axis))
        else:
            tab = tab.reshape(-1)
        self.axes = []
        for ax in ([fdot_axis] if self.two_d else []) + [f_axis]:
            ax = xp.asarray(ax, dtype=xp.float64)
            step = float(ax[1] - ax[0])
            if not bool(xp.allclose(xp.diff(ax), step, rtol=1e-8, atol=0.0)):
                raise ValueError("_UniformCubicSpline needs a uniform grid")
            self.axes.append((float(ax[0]), step, int(ax.size)))
        self.coeffs = ndi.spline_filter(tab, order=3, mode="mirror")

    def __call__(self, pts):
        xp = self.xp
        pts = xp.asarray(pts, dtype=xp.float64)
        cols = [pts[..., 0], pts[..., 1]] if self.two_d else [pts]
        idx, inside = [], None
        for c, (x0, step, n) in zip(cols, self.axes):
            i = (c - x0) / step
            ok = (i >= -1e-9) & (i <= n - 1 + 1e-9)
            inside = ok if inside is None else inside & ok
            idx.append(i)
        out = self.ndi.map_coordinates(self.coeffs, xp.stack([i.ravel() for i in idx]), order=3,
                                       mode="mirror", prefilter=False).reshape(idx[0].shape)
        return xp.where(inside, out, 0.0)


class WDMLookupTable(WDMSettings):
    """Pre-computed WDM pixel response of a unit chirping carrier (lookup table).

    The table stores, for a carrier ``cos`` / ``sin(2 pi (f t + fdot t^2 / 2))``, the
    WDM coefficient it produces at one reference pixel ``(m_ref, n_ref)``, as a
    function of the carrier's offset from that layer (``f_vals_norm``, spanning
    the ``m_diffs`` blocks of one layer each) and of its chirp rate (``fdot_vals``).
    By the shift structure of the WDM basis this one pixel gives every pixel's
    coefficient of a slowly varying signal (amplitude, phase, frequency and
    frequency derivative evaluated at the pixel time): see :meth:`get_wdm_coeffs`
    and the vectorized
    :class:`~lisatools.wdm_lookup_eval.WDMLookupEvaluator`. The table depends only
    on ``Nf`` and ``dt`` (the layer duration), not on ``Nt`` or the band.

    Production use (9-month / 12-month runs): the SOBBH lookup likelihood
    (``SOBBH_LIKELIHOOD=lookup``), the EMRI direct-to-WDM template
    (``EMRI_LIKELIHOOD=direct``) and the GB lookup engine read tables built by
    :mod:`lisatools.wdm_lookup_store` (recipe ``build_kind="n_ref_complex"``) and
    loaded with :meth:`from_file`.

    Construction loads ``store_path`` when the file exists, else builds the table
    there. The inherited :class:`WDMSettings` part is a copy of ``settings``.

    Args:
        settings: Underlying :class:`WDMSettings` defining the wavelet grid.
        nchannels: Number of channels stored in the table (metadata).
        m_ref: Reference frequency-layer index of the build (no default is
            computed here: pass the value returned by :meth:`apply_eps_frequency`).
        norm_freq_single_layer: In-layer frequency offsets ``[0, layer_df)`` (Hz),
            from :meth:`apply_eps_frequency`.
        m_diffs: Integer layer-block offsets covered by the table, from
            :meth:`apply_eps_frequency`.
        fdot_vals: Frequency-derivative grid (Hz/s), from :meth:`apply_eps_fdot`;
            ``[0.0]`` builds an fdot-free table.
        store_path: HDF5 file path (required in practice: ``None`` fails in
            ``os.path.exists``). If it exists, the table is loaded; otherwise it is
            built and saved here.
        batch_size_gen: Number of table entries generated per synthetic WDM
            transform batch (``-1`` means a single batch). Default ``20``.
        td_window: Optional time-domain window for the build. The ``n_ref`` build
            kinds require ``None`` (or all ones).
        verbose: Print build progress.
        build_kind: ``"n_ref_only"`` (one ``(fdot, f)`` table per quadrature, two
            real transforms), ``"n_ref_complex"`` (the same table from one complex
            transform, stored as ``table_cx``) or ``"per_n"`` (legacy per-pixel
            build, no fdot). ``None`` uses :attr:`BUILD_KIND` (env ``WDM_BUILD_KIND``,
            default ``"n_ref_only"``). A loaded file's kind overrides it.
        time_layers: ``Nt`` of the synthetic build transform for the ``n_ref``
            kinds (``None`` = ``settings.Nt``); its half must have the parity of
            ``settings.Nt // 2`` and be at least 4.

    Raises:
        ValueError: From the build, for an unknown ``build_kind``, a bad
            ``time_layers`` or a non-trivial ``td_window``.

    Attributes:
        m_ref (int): Reference layer of the build.
        n_ref (int): Reference pixel of the build (``sub_settings.Nt // 2``).
        sub_settings (WDMSettings): Grid of the synthetic build transform.
        fdot_vals, norm_freq_single_layer, m_diffs (``xp.ndarray``): The table axes.
        build_kind (str): Storage variant.
        BUILD_KIND (str): Class default build kind (env ``WDM_BUILD_KIND``).
        BASIS_CYCLE (str): Evaluation rule of :meth:`get_wdm_coeffs`
            (``"quarter_turn"``; ``"legacy_2way"`` is a negative control).
        INTERP_METHOD (str): Interpolation of the ``n_ref`` tables, ``"linear"``,
            ``"cubic"`` or ``"spline"`` (env ``WDM_LOOKUP_INTERP``, default
            ``"linear"``); see :meth:`set_interp_method`.

    Example:
        .. code-block:: python

            from lisatools.domains import WDMLookupTable
            from lisatools.wdm_lookup_store import ensure_lookup_table, lookup_table_path

            path = lookup_table_path(None, run_dir, Nf=1440, dt=2.5)
            ensure_lookup_table(path, Nf=1440, dt=2.5)        # build once if missing
            table = WDMLookupTable.from_file(path, force_backend="cpu")
            coeffs, m_map = table.get_wdm_coeffs(amp, phase, f, fdot, n_pix)
    """

    # Attribute / dataset keys owned by the lookup-table serialisation. Used by
    # :meth:`to_h5_group` so embedding the table into an HDF5 group that already
    # held an older copy stays clean instead of leaving stale keys.
    _H5_ATTR_KEYS = (
        "Nf",
        "Nt",
        "Nt_generate",
        "data_dt",
        "max_freq",
        "min_freq",
        "max_time",
        "min_time",
        "m_ref",
        "n_ref",
        "nchannels",
        "table_kind",
    )
    _H5_DATASET_KEYS = (
        "table_sin",
        "table_cos",
        "table_cx",
        "fdot_vals",
        "norm_freq_single_layer",
        "m_diffs",
    )

    def to_h5_group(self, group: h5py.Group) -> None:
        """Persist the lookup table into an open :class:`h5py.Group`.

        Idempotent: any pre-existing lookup-table attrs / datasets inside the
        group are removed before writing so the result is a clean replacement.
        Use this when embedding the table inside a larger HDF5 file (e.g. the
        global-fit backend). The standalone :meth:`to_file` is a thin wrapper
        around this helper.
        """
        for key in self._H5_ATTR_KEYS:
            if key in group.attrs:
                del group.attrs[key]
        for key in self._H5_DATASET_KEYS:
            if key in group:
                del group[key]

        group.attrs["Nf"] = self.Nf
        group.attrs["Nt"] = self.Nt
        group.attrs["Nt_generate"] = self.sub_settings.Nt
        group.attrs["data_dt"] = self.data_dt
        if self.max_freq is not None:
            group.attrs["max_freq"] = self.max_freq
        if self.min_freq is not None:
            group.attrs["min_freq"] = self.min_freq
        if self.max_time is not None:
            group.attrs["max_time"] = self.max_time
        if self.min_time is not None:
            group.attrs["min_time"] = self.min_time
        group.attrs["m_ref"] = self.m_ref
        group.attrs["n_ref"] = self.n_ref
        group.attrs["nchannels"] = self.nchannels
        group.attrs["table_kind"] = self.build_kind
        # Persist whichever storage variant the build produced. Real path
        # writes table_sin + table_cos; complex path writes table_cx.
        if self.build_kind == "n_ref_complex":
            group.create_dataset("table_cx", data=self.get(self.table_cx))
        else:
            group.create_dataset("table_sin", data=self.get(self.table_sin))
            group.create_dataset("table_cos", data=self.get(self.table_cos))
        group.create_dataset("fdot_vals", data=self.get(self.fdot_vals))
        group.create_dataset(
            "norm_freq_single_layer", data=self.get(self.norm_freq_single_layer)
        )
        group.create_dataset("m_diffs", data=self.get(self.m_diffs))

    @staticmethod
    def _settings_from_h5_group(
        group: h5py.Group, force_backend: Optional[str] = None
    ) -> Tuple["WDMSettings", int]:
        """Build the WDM grid + nchannels from a serialised lookup-table group.

        Handles the older ``ind_min`` / ``ind_max`` layout (converted to
        explicit ``min_freq`` / ``max_freq``) so files saved before that
        rewrite still load. Returns ``(settings, nchannels)``.
        """
        input_args = (
            int(group.attrs["Nf"]),
            int(group.attrs["Nt"]),
            float(group.attrs["data_dt"]),
        )
        nchannels = int(group.attrs["nchannels"])

        if "ind_min" in group.attrs:
            # Legacy layout: ``ind_min`` / ``ind_max`` index pixels rather than
            # carrying explicit frequency edges. Convert to ``min_freq`` /
            # ``max_freq`` using a throwaway WDMSettings to read ``layer_df``.
            _settings = WDMSettings(*input_args, force_backend=force_backend)
            min_freq = _settings.layer_df * int(group.attrs["ind_min"])
            max_freq = _settings.layer_df * int(group.attrs["ind_max"])
            input_kwargs = dict(
                min_freq=min_freq,
                max_freq=max_freq,
                force_backend=force_backend,
            )
        else:
            input_kwargs = dict(force_backend=force_backend)
            if "min_freq" in group.attrs:
                input_kwargs["min_freq"] = float(group.attrs["min_freq"])
            if "max_freq" in group.attrs:
                input_kwargs["max_freq"] = float(group.attrs["max_freq"])
            if "min_time" in group.attrs:
                input_kwargs["min_time"] = float(group.attrs["min_time"])
            if "max_time" in group.attrs:
                input_kwargs["max_time"] = float(group.attrs["max_time"])

        settings = WDMSettings(*input_args, **input_kwargs)
        return settings, nchannels

    @staticmethod
    def from_h5_group(
        group: h5py.Group, force_backend: Optional[str] = None
    ) -> "WDMLookupTable":
        """Reconstruct a :class:`WDMLookupTable` from an open :class:`h5py.Group`.

        Skips the constructor's build-or-load branch so no on-disk file is
        required; this is the counterpart to :meth:`to_h5_group`, for a table
        embedded in a larger HDF5 file. A group without ``table_kind`` is read as
        ``"per_n"``.

        Args:
            group: Group written by :meth:`to_h5_group`.
            force_backend: Backend for the table arrays (``None`` = default).

        Returns:
            WDMLookupTable: The loaded table (interpolators built on load).
        """
        settings, nchannels = WDMLookupTable._settings_from_h5_group(
            group, force_backend=force_backend
        )

        # Bypass ``__init__`` so we don't trigger the build-or-read-from-path
        # logic. ``WDMSettings.__init__`` still has to run to populate the
        # basis grid; everything else is filled in directly from ``group``.
        obj = WDMLookupTable.__new__(WDMLookupTable)
        WDMSettings.__init__(obj, *settings.args, **settings.kwargs)
        obj.nchannels = nchannels
        obj.store_path = None

        table_kind = group.attrs.get("table_kind", "per_n")
        if isinstance(table_kind, bytes):
            table_kind = table_kind.decode()
        obj.build_kind = table_kind

        obj.sub_settings = WDMSettings(
            int(group.attrs["Nf"]),
            int(group.attrs["Nt_generate"]),
            float(group.attrs["data_dt"]),
            force_backend=obj.settings.backend,
        )
        obj.fdot_vals = obj.xp.asarray(group["fdot_vals"][:])
        obj.m_ref = int(group.attrs["m_ref"])
        # n_ref is the BUILD's sub_n_ref (= sub_settings.Nt // 2). The build
        # enforces sub_n_ref parity == eval n_ref parity so the eval
        # (-1)^dn sign pattern matches the eval grid.
        obj.n_ref = int(obj.sub_settings.Nt / 2)
        obj.is_m_ref_n_ref_even = (obj.m_ref + obj.n_ref) % 2 == 0

        obj.norm_freq_single_layer = obj.xp.asarray(group["norm_freq_single_layer"][:])
        obj.m_diffs = obj.xp.asarray(group["m_diffs"][:])
        if table_kind == "n_ref_complex":
            obj.table_cx = obj.xp.asarray(group["table_cx"][:])
        else:
            obj.table_sin = obj.xp.asarray(group["table_sin"][:])
            obj.table_cos = obj.xp.asarray(group["table_cos"][:])
        return obj

    def to_file(self, fp: str) -> None:
        """Persist the lookup table to a standalone HDF5 file at ``fp`` (group ``"wdm"``).

        Raises:
            ValueError: If ``fp`` already exists.
        """
        if os.path.exists(fp):
            raise ValueError("Trying to write to file that exists.")
        with h5py.File(fp, "w") as f:
            self.to_h5_group(f.create_group("wdm"))

    @staticmethod
    def from_file(fp: str, force_backend: Optional[str] = None) -> "WDMLookupTable":
        """Construct a :class:`WDMLookupTable` from a standalone HDF5 file at ``fp``.

        This is how the global-fit source runtimes load the SOBBH / EMRI tables.

        Args:
            fp: Path of a file written by :meth:`to_file` (group ``"wdm"``).
            force_backend: Backend for the table arrays (``None`` = default).

        Returns:
            WDMLookupTable: The loaded table.
        """
        with h5py.File(fp, "r") as f:
            return WDMLookupTable.from_h5_group(f["wdm"], force_backend=force_backend)

    def from_file_internal(self, fp: str) -> None:
        """Repopulate this instance's tables and metadata from ``fp`` in-place.

        Used by ``__init__`` when ``store_path`` already exists on disk. Reads
        through :meth:`from_h5_group` and copies the populated fields over.
        """
        with h5py.File(fp, "r") as f:
            loaded = WDMLookupTable.from_h5_group(
                f["wdm"], force_backend=self.settings.backend
            )

        # Mirror the legacy ``from_file_internal`` field set so existing
        # callers see exactly the same instance attributes after this runs.
        self.build_kind = loaded.build_kind
        self.sub_settings = loaded.sub_settings
        self.fdot_vals = loaded.fdot_vals
        self.m_ref = loaded.m_ref
        self.n_ref = loaded.n_ref
        self.is_m_ref_n_ref_even = loaded.is_m_ref_n_ref_even
        self.nchannels = loaded.nchannels
        self.norm_freq_single_layer = loaded.norm_freq_single_layer
        self.m_diffs = loaded.m_diffs
        if loaded.build_kind == "n_ref_complex":
            self.table_cx = loaded.table_cx
        else:
            self.table_sin = loaded.table_sin
            self.table_cos = loaded.table_cos
            
    @staticmethod
    def apply_eps_fdot(eps: float, settings: WDMSettings, fdot_max_factor: float= 8.0) -> np.ndarray:
        r"""Build a symmetric :math:`\dot{f}` grid spaced by ``eps * layer_df / layer_dt``.

        The grid is ``k * delta`` for ``|k * delta| < fdot_max_factor * layer_df /
        layer_dt``, mirrored about (and including) zero.

        Args:
            eps: Step in units of ``layer_df / layer_dt``.
            settings: Grid supplying ``layer_df`` / ``layer_dt``.
            fdot_max_factor: Half-width in units of ``layer_df / layer_dt``.
                Default ``8.0``.

        Returns:
            np.ndarray: Sorted ``fdot`` values (Hz/s), odd length.
        """
        delta_fdot = eps * settings.layer_df / settings.layer_dt
        fdot_max_val = fdot_max_factor * settings.layer_df / settings.layer_dt
        _fdot = np.arange(0.0, fdot_max_val, delta_fdot)
        fdot_vals = np.concatenate([-_fdot[::-1][:-1], _fdot])
        return fdot_vals

    @staticmethod
    def apply_eps_frequency(eps: float, settings: WDMSettings, m_ref: Optional[int] = None, num_layers_diff: Optional[int] = 2) -> tuple:
        """Build sub-layer frequency offsets, layer-index offsets, and the reference layer.

        Args:
            eps: In-layer frequency step in units of ``layer_df``.
            settings: Grid supplying ``layer_df`` (and ``Nt`` for the default).
            m_ref: Reference layer. ``None`` uses ``int(settings.Nt / 2)`` (note: a
                time-pixel count, not ``Nf``; the production recipe passes 21).
            num_layers_diff: ``k``: the table spans ``2k + 2`` layer blocks.
                Default ``2``.

        Returns:
            tuple: ``(norm_freq_single_layer, m_diffs, m_ref)``:
            ``arange(0, layer_df, eps * layer_df)`` (Hz), the block offsets
            ``[-(k + 1), ..., k]`` and the reference layer.
        """
        delta_f = eps * settings.layer_df

        if m_ref is None:
            m_ref = int(settings.Nt / 2)

        norm_freq_single_layer = np.arange(0.0, settings.layer_df, delta_f)
        m_diffs = (_tmp := np.arange(2 * num_layers_diff + 2)) - _tmp[int(len(_tmp) / 2)]

        return norm_freq_single_layer, m_diffs, m_ref

    # Plan A vs Plan B switch. "n_ref_only" (default, see WDM_FDOT_LOOKUP_PLAN.md)
    # extracts a single (m_ref, n_ref) pixel and stores a 2-D
    # (fdot_steps, f_steps) table — handles fdot cleanly. "per_n" keeps
    # the older per-time-pixel / per-m_diff build that does not support
    # fdot at lookup time. Override per-instance via the build_kind kwarg
    # or globally via env var WDM_BUILD_KIND.
    BUILD_KIND = os.environ.get("WDM_BUILD_KIND", "n_ref_only")

    # n_ref eval rule: "quarter_turn" (derived, see get_wdm_coeffs) or "legacy_2way"
    # (the historical (-1)^dn + parity swap; negative-control only).
    BASIS_CYCLE = "quarter_turn"
    # (fdot, f) interpolation of the n_ref tables: "linear" (historical), "cubic"
    # (scipy RegularGridInterpolator; mid-node fdot error converges far faster) or
    # "spline" (_UniformCubicSpline: prefiltered cubic B-spline, CPU or GPU).
    INTERP_METHOD = os.environ.get("WDM_LOOKUP_INTERP", "linear")

    def __init__(self, settings: WDMSettings, nchannels: int, m_ref: int = None, norm_freq_single_layer: np.ndarray = None, m_diffs: np.ndarray = None, fdot_vals: np.ndarray = None, store_path: Optional[str] = None, batch_size_gen: Optional[int] = 20, td_window: Optional[np.ndarray] = None, verbose: bool = False, build_kind: Optional[str] = None, time_layers: Optional[int] = None):
        WDMSettings.__init__(self, *settings.args, **settings.kwargs)
        # TODO: CHECK FIRST AND LAST TIME LAYERS DUE TO TIME WINDOWING?

        self.nchannels = nchannels

        # Provisional build_kind; from_file_internal may overwrite it from disk.
        self.build_kind = build_kind if build_kind is not None else type(self).BUILD_KIND

        self.store_path = store_path
        if os.path.exists(self.store_path):
            self.from_file_internal(self.store_path)
        else:
            self.build_lookup_table(m_ref, m_diffs, norm_freq_single_layer, fdot_vals, store_path, batch_size_gen, td_window, verbose=verbose, time_layers=time_layers)

    def build_lookup_table(self, m_ref: int, m_diffs: np.ndarray, norm_freq_single_layer: np.ndarray, fdot_vals: np.ndarray, store_path: str, batch_size_gen: int, td_window: Optional[np.ndarray] = None, verbose: bool = False, time_layers: Optional[int] = None) -> None:
        """Generate the coefficient tables and (optionally) save them to ``store_path``.

        Dispatches on ``self.build_kind``: ``_build_lookup_table_n_ref`` (Plan A,
        ``"n_ref_only"``), ``_build_lookup_table_n_ref_complex`` (``"n_ref_complex"``)
        or ``_build_lookup_table_per_n`` (Plan B fallback, ``"per_n"``). Called by the
        constructor when ``store_path`` does not exist.

        For the ``n_ref`` kinds, ``time_layers`` controls the
        Nt of the synthetic transform used in the build; smaller values
        speed the build up linearly (the table only reads one pixel
        anyway). Must be even and large enough that the wavelet basis at
        ``time_layers // 2`` is clear of edge effects. ``None`` keeps the
        full Nt (slow).

        Args:
            m_ref, m_diffs, norm_freq_single_layer, fdot_vals: The table axes (see
                the class docstring).
            store_path: Output HDF5 path, or ``None`` to keep the table in memory.
            batch_size_gen: Entries per synthetic transform batch (``-1``: one batch).
            td_window: Build window (``None`` for the ``n_ref`` kinds).
            verbose: Print build progress.
            time_layers: See above.

        Raises:
            ValueError: For an unknown ``build_kind`` (and from the builders).
        """
        if self.build_kind == "n_ref_only":
            return self._build_lookup_table_n_ref(
                m_ref, m_diffs, norm_freq_single_layer, fdot_vals,
                store_path, batch_size_gen, td_window=td_window,
                time_layers=time_layers, verbose=verbose,
            )
        if self.build_kind == "n_ref_complex":
            return self._build_lookup_table_n_ref_complex(
                m_ref, m_diffs, norm_freq_single_layer, fdot_vals,
                store_path, batch_size_gen, td_window=td_window,
                time_layers=time_layers, verbose=verbose,
            )
        if self.build_kind == "per_n":
            return self._build_lookup_table_per_n(
                m_ref, m_diffs, norm_freq_single_layer, fdot_vals,
                store_path, batch_size_gen, td_window=td_window, verbose=verbose,
            )
        raise ValueError(f"Unknown WDMLookupTable.build_kind={self.build_kind!r}; expected 'n_ref_only', 'n_ref_complex', or 'per_n'.")

    def _build_lookup_table_per_n(self, m_ref: int, m_diffs: np.ndarray, norm_freq_single_layer: np.ndarray, fdot_vals: np.ndarray, store_path: str, batch_size_gen: int, td_window: Optional[np.ndarray] = None, verbose: bool = False) -> None:
        """Plan B (fallback) — per-(n, m_diff) build. fdot lookup not yet wired."""
        self.sub_settings = WDMSettings(self.Nf, self.Nt, self.data_dt, force_backend=self.force_backend)
        self.m_ref = m_ref
        self.n_ref = int(self.sub_settings.Nt / 2)
        self.is_m_ref_n_ref_even = (self.m_ref + self.n_ref) % 2 == 0
        self.m_diffs = self.xp.asarray(m_diffs).astype(self.xp.int32)
        self.fdot_vals = self.xp.asarray(fdot_vals)
        self.norm_freq_single_layer = self.xp.asarray(norm_freq_single_layer)

        total_f_fdot_vals = self.norm_f_steps * self.fdot_steps
        if self.run_fdot:
            _f_vals, _fdot_vals = self.xp.asarray([tmp.ravel() for tmp in self.xp.meshgrid(self.norm_freq_single_layer + self.f_ref, self.fdot_vals)])
        else:
            _f_vals = self.norm_freq_single_layer.copy() + self.f_ref
            _fdot_vals = self.xp.zeros_like(_f_vals)

        t_vals = self.xp.arange(self.sub_settings.N) * self.data_dt
        t_vals_wdm = self.sub_settings.t_arr
        t_diff_wdm = t_vals_wdm - self.t_ref
        t_diff = t_vals - self.t_ref

        if batch_size_gen == -1:
            batch_size_gen = total_f_fdot_vals

        batches = np.arange(0, total_f_fdot_vals, batch_size_gen)
        if batches[-1] < total_f_fdot_vals:
            batches = np.append(batches, np.array([total_f_fdot_vals]))

        if td_window is None:
            td_window = self.xp.ones_like(t_diff)

        _table_sin = self.xp.zeros((self.Nt, len(m_diffs), total_f_fdot_vals,))
        _table_cos = self.xp.zeros((self.Nt, len(m_diffs), total_f_fdot_vals,))

        self.td_window = td_window
        import time as _time
        _n_batches = len(batches) - 1
        _t_start = _time.perf_counter()
        if verbose:
            print(f"[build] starting: {_n_batches} batches × {len(m_diffs)} "
                  f"m_diffs × Nt={self.Nt} time pixels", flush=True)
        for _bi, (st_batch, end_batch) in enumerate(zip(batches[:-1], batches[1:])):
            _t_batch = _time.perf_counter()
            inds = np.arange(st_batch, end_batch)

            phase = 2 * np.pi * (_f_vals[inds, None] * t_diff[None, :] + 1. / 2. * _fdot_vals[inds, None] * t_diff[None, :] ** 2)
            wave_sin = self.xp.sin(phase)
            wave_cos = self.xp.cos(phase)

            wave_sin_wdm = TDSignal(wave_sin, TDSettings(self.sub_settings.N, self.sub_settings.data_dt, force_backend=self.force_backend)).wdmtransform(settings=self.sub_settings, window=self.td_window)
            wave_cos_wdm = TDSignal(wave_cos, TDSettings(self.sub_settings.N, self.sub_settings.data_dt, force_backend=self.force_backend)).wdmtransform(settings=self.sub_settings, window=self.td_window)

            # TODO this is transpose shape? should be fine
            phase_n = 2 * np.pi * (_f_vals[None, inds] * t_diff_wdm[:, None] + 1. / 2. * _fdot_vals[None, inds] * t_diff_wdm[:, None] ** 2)
            
            n_current = self.xp.arange(self.Nt)
            for m_i, m_diff in enumerate(m_diffs):
                m_current = self.m_ref + m_diff
                _sin_coeff_1 = wave_sin_wdm[:, self.m_ref - m_diff, :].T
                _cos_coeff_1 = wave_cos_wdm[:, self.m_ref - m_diff, :].T

                _cos_coeff = _cos_coeff_1 * np.cos(-phase_n)  - _sin_coeff_1 * np.sin(-phase_n)
                _sin_coeff = _cos_coeff_1 * np.sin(-phase_n)  + _sin_coeff_1 * np.cos(-phase_n)

                _f_norm = _f_vals[inds] - (self.m_ref - m_diff) * self.layer_df

                is_odd = ((m_current + n_current) % 2 == 1)[:, None]
                # sin_coeff = _sin_coeff * (~is_odd) + _cos_coeff * (is_odd)
                # cos_coeff = _sin_coeff * (is_odd) + _cos_coeff * (~is_odd)

                sin_coeff = _sin_coeff * (~is_odd) + _cos_coeff * (is_odd)
                cos_coeff = _sin_coeff * (is_odd) + _cos_coeff * (~is_odd)

                # Bake (-1)^(m_diff parity) into the stored values so that
                # adjacent m_diff blocks have matching signs at f_norm
                # boundaries. Without this, linear interp across block
                # boundaries blends opposite-sign samples and produces
                # garbage in a narrow window of pixels around each Doppler
                # crossing. The eval kernel undoes this with a matching
                # (-1)^(m_diff) factor (which combines with the original
                # FFT-mirror correction).
                # TODO: when fdot != 0, the sine term still needs review —
                # the current implementation only handles fdot_vals = [0.0],
                # and the parity argument for the sine branch under nonzero
                # fdot has not yet been worked out.
                if int(m_diff) & 1:
                    # with FDOT=0.0, sin_coeff is always ~0.0 here
                    sin_coeff = -sin_coeff
                    cos_coeff = -cos_coeff

                _table_sin[:, m_i, inds] = sin_coeff
                _table_cos[:, m_i, inds] = cos_coeff

            if verbose:
                _elapsed = _time.perf_counter() - _t_start
                _per_batch = _elapsed / (_bi + 1)
                _eta = _per_batch * (_n_batches - _bi - 1)
                print(f"[build] batch {_bi + 1:4d}/{_n_batches}  "
                      f"inds=[{int(st_batch):5d}, {int(end_batch):5d})  "
                      f"this_batch={_time.perf_counter() - _t_batch:.2f}s  "
                      f"elapsed={_elapsed/60.0:5.2f}m  ETA={_eta/60.0:5.2f}m",
                      flush=True)
        _table_sin = _table_sin.reshape(self.Nt, len(m_diffs), self.fdot_steps, self.norm_f_steps).transpose(0, 2, 1, 3).reshape(self.Nt, self.fdot_steps, self.f_steps).copy()
        _table_cos = _table_cos.reshape(self.Nt, len(m_diffs), self.fdot_steps, self.norm_f_steps).transpose(0, 2, 1, 3).reshape(self.Nt, self.fdot_steps, self.f_steps).copy()

        assert _table_sin.shape == (self.Nt, self.fdot_vals.shape[0], self.f_vals.shape[0])
        assert _table_cos.shape == (self.Nt, self.fdot_vals.shape[0], self.f_vals.shape[0])

        self.table_sin = _table_sin
        self.table_cos = _table_cos

        if store_path is not None:
            self.to_file(store_path)

    def _build_lookup_table_n_ref(self, m_ref: int, m_diffs: np.ndarray, norm_freq_single_layer: np.ndarray, fdot_vals: np.ndarray, store_path: str, batch_size_gen: int, td_window: Optional[np.ndarray] = None, time_layers: Optional[int] = None, verbose: bool = False) -> None:
        """Plan A — extract single ``(m_ref, n_ref)`` pixel, store ``(fdot_steps, f_steps)`` table.

        Each table entry is the WDM coefficient at pixel ``(m_ref, n_ref)``
        of a synthetic carrier ``sin/cos(2π (f·t + ½·fdot·t²))`` where
        ``f`` ranges over ``f_vals`` (one layer worth of ``norm_freq``
        concatenated across every ``m_diff`` block) and ``fdot`` over
        ``fdot_vals``. The ``(-1)^m_diff_block`` sign bake at build keeps
        adjacent ``m_diff`` blocks sign-aligned so the f_norm interpolator
        is smooth across layer boundaries — the eval kernel inverts that
        bake via the matching ``(-1)^((ms_to_use - m_ref) parity)``
        post-multiply in ``get_wdm_coeffs``.
        """
        # time_layers controls Nt of the synthetic transform used to read
        # the (m_ref, sub_n_ref) pixel. The WDM basis at (m, n) has time
        # support ~Nf samples (one layer_dt each side), so as long as
        # sub_n_ref is ≥ 2 layers from each edge the read is identical
        # to the full-Nt transform. The build cost scales linearly with
        # Nt → small time_layers is cheap.
        # Parity requirement: sub_n_ref must have the SAME parity as
        # the EVAL's n_ref (= self.Nt // 2) so the per-block parity
        # swap and m-rotation conventions match. We enforce
        # time_layers % 4 == 0 when self.Nt // 2 is even, etc.
        if time_layers is None:
            time_layers = int(self.Nt)
        time_layers = int(time_layers)
        _eval_n_ref = int(self.Nt) // 2
        _sub_n_ref = time_layers // 2
        if (_sub_n_ref % 2) != (_eval_n_ref % 2):
            raise ValueError(
                f"time_layers={time_layers} gives sub_n_ref={_sub_n_ref} "
                f"with parity {_sub_n_ref % 2}, but the EVAL's n_ref="
                f"{_eval_n_ref} has parity {_eval_n_ref % 2}. The build/eval "
                "parity conventions require these to match — adjust "
                "time_layers (e.g., bump to a multiple of 4 when the "
                "eval n_ref is EVEN)."
            )
        if _sub_n_ref < 4:
            raise ValueError(
                f"time_layers={time_layers} gives sub_n_ref={_sub_n_ref}; "
                "need at least 8 to keep the wavelet basis clear of edges."
            )

        self.sub_settings = WDMSettings(self.Nf, time_layers, self.data_dt, force_backend=self.force_backend)
        self.m_ref = m_ref
        # self.n_ref is the BUILD'S sub_n_ref (= sub_settings.Nt // 2)
        # — this is the actual cell whose parity drives the per-block
        # swap. The eval-side (-1)^dn correction uses (n_eval - n_ref)
        # too; we enforce sub_n_ref parity == eval_n_ref parity above,
        # so the sign pattern matches the eval grid's convention.
        self.n_ref = _sub_n_ref
        self.is_m_ref_n_ref_even = (self.m_ref + self.n_ref) % 2 == 0
        self.m_diffs = self.xp.asarray(m_diffs).astype(self.xp.int32)
        self.fdot_vals = self.xp.asarray(fdot_vals)
        self.norm_freq_single_layer = self.xp.asarray(norm_freq_single_layer)

        # Plan A reads a single pixel; any non-trivial window silently
        # re-weights the carrier and shifts the stored coefficient. The
        # outer TDSignal->WDM transform is where windowing belongs, not
        # the lookup table build.
        if td_window is not None:
            _tw = self.xp.asarray(td_window)
            if not bool(self.xp.allclose(_tw, 1.0)):
                raise ValueError(
                    "WDMLookupTable build_kind='n_ref_only' requires "
                    "td_window=None (or an all-ones array). A non-trivial "
                    "window biases the stored (m_ref, n_ref) pixel. "
                    "Apply windowing only at the outer TDSignal->WDM transform."
                )

        # Build time grid + t_diff use the SUB transform's reference
        # time (sub_n_ref * layer_dt) — this IS the wavelet's effective
        # center in Cornish-Romano WDM convention, despite t_arr[n]
        # reading as a "left edge" name. Empirically verified: shifting
        # build (and matching eval) to (sub_n_ref + 0.5)*layer_dt broke
        # the lookup (mm jumped from 2e-4 to ~1) because the new t_ref
        # was half a pixel off-center from the wavelet's actual support
        # midpoint, rotating every (f-dependent) table entry by huge
        # angles like 2π·f·0.5·layer_dt.
        _sub_t_ref = _sub_n_ref * self.sub_settings.layer_dt
        t_vals = self.xp.arange(self.sub_settings.N) * self.data_dt
        t_diff = t_vals - _sub_t_ref

        # Build the full f-axis: one layer worth of norm_freq, replicated
        # across every m_diff block (matches the self.f_vals property).
        # f_steps = norm_f_steps * len(m_diffs).
        total_f_fdot_vals = self.f_steps * self.fdot_steps
        if self.run_fdot:
            _f_grid, _fdot_grid = self.xp.meshgrid(self.f_vals, self.fdot_vals)
            _f_vals = _f_grid.ravel()
            _fdot_vals = _fdot_grid.ravel()
        else:
            _f_vals = self.f_vals.copy()
            _fdot_vals = self.xp.zeros_like(_f_vals)

        # Per-entry m_diff_block (which block of m_diffs each f_val lives in).
        # f_vals is concatenated as [norm_freq + m_diffs[0]*df, ..., norm_freq + m_diffs[-1]*df]
        # so block index is (i % f_steps) // norm_f_steps for the fdot-meshed layout.
        _block_per_fval = self.xp.repeat(self.m_diffs, self.norm_f_steps)
        if self.run_fdot:
            _block_per_entry = self.xp.tile(_block_per_fval, self.fdot_steps)
        else:
            _block_per_entry = _block_per_fval

        if batch_size_gen == -1:
            batch_size_gen = total_f_fdot_vals

        batches = np.arange(0, total_f_fdot_vals, batch_size_gen)
        if batches[-1] < total_f_fdot_vals:
            batches = np.append(batches, np.array([total_f_fdot_vals]))

        if td_window is None:
            td_window = self.xp.ones_like(t_diff)
        self.td_window = td_window

        _table_sin = self.xp.zeros((total_f_fdot_vals,))
        _table_cos = self.xp.zeros((total_f_fdot_vals,))

        import time as _time
        _n_batches = len(batches) - 1
        _t_start = _time.perf_counter()
        if verbose:
            print(f"[build n_ref_only] starting: {_n_batches} batches  "
                  f"f_steps={self.f_steps}  fdot_steps={self.fdot_steps}  "
                  f"total={total_f_fdot_vals}", flush=True)

        for _bi, (st_batch, end_batch) in enumerate(zip(batches[:-1], batches[1:])):
            _t_batch = _time.perf_counter()
            inds = np.arange(st_batch, end_batch)

            phase = 2 * np.pi * (
                _f_vals[inds, None] * t_diff[None, :]
                + 0.5 * _fdot_vals[inds, None] * t_diff[None, :] ** 2
            )
            wave_sin = self.xp.sin(phase)
            wave_cos = self.xp.cos(phase)

            wave_sin_wdm = TDSignal(
                wave_sin,
                TDSettings(self.sub_settings.N, self.sub_settings.data_dt, force_backend=self.force_backend),
            ).wdmtransform(settings=self.sub_settings, window=self.td_window)
            wave_cos_wdm = TDSignal(
                wave_cos,
                TDSettings(self.sub_settings.N, self.sub_settings.data_dt, force_backend=self.force_backend),
            ).wdmtransform(settings=self.sub_settings, window=self.td_window)

            # Read at the SUB transform's centre pixel, not the EVAL's
            # n_ref — by WDM time-shift invariance these are equivalent,
            # and the sub transform is small/fast.
            raw_sin = wave_sin_wdm[:, self.m_ref, _sub_n_ref]
            raw_cos = wave_cos_wdm[:, self.m_ref, _sub_n_ref]

            # 1) m-rotation: the WDM basis at (m, n) has a (m+n)π/2 phase
            # factor, so translating from (m_ref, n_ref) read to
            # (m_ref - k, n_ref) crosses |k| basis-parity boundaries.
            # Empirical test (diag_plan_a_fdot_per_pixel.py) shows for
            # source-with-fdot the rotation per ODD-parity-transition is
            # R(-90°) when (m_ref+n_ref) is ODD and R(+90°) when EVEN —
            # i.e., the direction is fixed by the m_ref/n_ref reference
            # parity, not by the SIGN of k. Crossing two boundaries
            # (k=±2) gives ODD→EVEN→ODD which cancels: net identity.
            #   k EVEN  -> identity        k ODD -> R(±90°)  (sign from m_ref+n_ref parity)
            # R(-90°)(a, b) = (b, -a),  R(+90°)(a, b) = (-b, a)
            _block_per_inds = _block_per_entry[inds].astype(int)
            _k_is_odd = (_block_per_inds & 1) != 0
            # Sign: -1 (i.e., R(-90°)) when (m_ref + n_ref) is ODD.
            if self.is_m_ref_n_ref_even:
                # EVEN reference: ODD k → R(+90°) = (-b, a)
                sin_rot = self.xp.where(_k_is_odd, -raw_cos, raw_sin)
                cos_rot = self.xp.where(_k_is_odd,  raw_sin, raw_cos)
            else:
                # ODD reference: ODD k → R(-90°) = (b, -a)
                sin_rot = self.xp.where(_k_is_odd,  raw_cos, raw_sin)
                cos_rot = self.xp.where(_k_is_odd, -raw_sin, raw_cos)

            # 2) Per-block parity swap mimicking per_n's per-cell swap at
            # n=n_ref. per_n swaps iff (m_current + n_current) is ODD with
            # m_current = m_ref + m_diff_BUILD. At n=n_ref this collapses
            # to a per-block boolean keyed by (m_ref + k + n_ref) parity.
            _is_odd_per_inds = (
                ((self.m_ref + _block_per_inds + self.n_ref) % 2) == 1
            )
            sin_pixel = self.xp.where(_is_odd_per_inds, cos_rot, sin_rot)
            cos_pixel = self.xp.where(_is_odd_per_inds, sin_rot, cos_rot)

            # 3) Bake (-1)^|k|. Together with the per-block swap this
            # keeps adjacent block samples sign-aligned at f_norm
            # boundaries. The eval-side FFT-mirror post-multiply
            # ((-1)^((ms_to_use - m_ref) parity)) undoes it per source.
            _block_sign = self.xp.where((_block_per_inds & 1) != 0, -1.0, 1.0)
            sin_pixel = sin_pixel * _block_sign
            cos_pixel = cos_pixel * _block_sign

            _table_sin[inds] = sin_pixel
            _table_cos[inds] = cos_pixel

            if verbose:
                _elapsed = _time.perf_counter() - _t_start
                _per_batch = _elapsed / (_bi + 1)
                _eta = _per_batch * (_n_batches - _bi - 1)
                print(f"[build n_ref_only] batch {_bi + 1:4d}/{_n_batches}  "
                      f"inds=[{int(st_batch):5d}, {int(end_batch):5d})  "
                      f"this_batch={_time.perf_counter() - _t_batch:.2f}s  "
                      f"elapsed={_elapsed/60.0:5.2f}m  ETA={_eta/60.0:5.2f}m",
                      flush=True)

        _table_sin = _table_sin.reshape(self.fdot_steps, self.f_steps).copy()
        _table_cos = _table_cos.reshape(self.fdot_steps, self.f_steps).copy()
        assert _table_sin.shape == (self.fdot_vals.shape[0], self.f_vals.shape[0])
        assert _table_cos.shape == (self.fdot_vals.shape[0], self.f_vals.shape[0])

        self.table_sin = _table_sin
        self.table_cos = _table_cos

        if store_path is not None:
            self.to_file(store_path)

    def _build_lookup_table_n_ref_complex(
        self, m_ref: int, m_diffs: np.ndarray, norm_freq_single_layer: np.ndarray,
        fdot_vals: np.ndarray, store_path: str, batch_size_gen: int,
        td_window: Optional[np.ndarray] = None, time_layers: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        """Plan A (complex variant) — same table layout as ``n_ref_only`` but built
        with ONE ``is_complex=True`` WDM transform instead of two real ones.

        Identical accuracy and identical eval-side mapping to the real path
        (the build's m-rotation / per-block parity-swap / ``(-1)^|k|``
        baking is retained, applied to Re/Im of the single complex
        transform). The single saving is the 2x build-time speedup from
        replacing the (sin-carrier, cos-carrier) pair of WDM transforms
        with one complex-WDM transform whose Re/Im carry both quadratures
        of the response.

        The stored complex value satisfies
        ``Re(table_cx) == table_cos`` and ``Im(table_cx) == table_sin``
        of the real path, so the existing real-path eval logic
        (``build_kind == 'n_ref_only'`` branch in ``get_wdm_coeffs``) can
        consume the Re/Im split without modification.
        """
        if time_layers is None:
            time_layers = int(self.Nt)
        time_layers = int(time_layers)
        _eval_n_ref = int(self.Nt) // 2
        _sub_n_ref = time_layers // 2
        if (_sub_n_ref % 2) != (_eval_n_ref % 2):
            raise ValueError(
                f"time_layers={time_layers} gives sub_n_ref={_sub_n_ref} "
                f"with parity {_sub_n_ref % 2}, but the EVAL's n_ref="
                f"{_eval_n_ref} has parity {_eval_n_ref % 2}."
            )
        if _sub_n_ref < 4:
            raise ValueError(
                f"time_layers={time_layers} gives sub_n_ref={_sub_n_ref}; "
                "need at least 8 to keep the wavelet basis clear of edges."
            )

        # Sub-build uses is_complex=True; we read the (m_ref, sub_n_ref) pixel
        # of the resulting complex WDM coefficient grid.
        self.sub_settings = WDMSettings(
            self.Nf, time_layers, self.data_dt,
            is_complex=True, force_backend=self.force_backend,
        )
        self.m_ref = m_ref
        self.n_ref = _sub_n_ref
        self.is_m_ref_n_ref_even = (self.m_ref + self.n_ref) % 2 == 0
        self.m_diffs = self.xp.asarray(m_diffs).astype(self.xp.int32)
        self.fdot_vals = self.xp.asarray(fdot_vals)
        self.norm_freq_single_layer = self.xp.asarray(norm_freq_single_layer)

        if td_window is not None:
            _tw = self.xp.asarray(td_window)
            if not bool(self.xp.allclose(_tw, 1.0)):
                raise ValueError(
                    "WDMLookupTable build_kind='n_ref_complex' requires "
                    "td_window=None (or an all-ones array)."
                )

        _sub_t_ref = _sub_n_ref * self.sub_settings.layer_dt
        t_vals = self.xp.arange(self.sub_settings.N) * self.data_dt
        t_diff = t_vals - _sub_t_ref

        total_f_fdot_vals = self.f_steps * self.fdot_steps
        if self.run_fdot:
            _f_grid, _fdot_grid = self.xp.meshgrid(self.f_vals, self.fdot_vals)
            _f_vals = _f_grid.ravel()
            _fdot_vals = _fdot_grid.ravel()
        else:
            _f_vals = self.f_vals.copy()
            _fdot_vals = self.xp.zeros_like(_f_vals)

        if batch_size_gen == -1:
            batch_size_gen = total_f_fdot_vals
        batches = np.arange(0, total_f_fdot_vals, batch_size_gen)
        if batches[-1] < total_f_fdot_vals:
            batches = np.append(batches, np.array([total_f_fdot_vals]))

        if td_window is None:
            td_window = self.xp.ones_like(t_diff)
        self.td_window = td_window

        # Per-entry m_diff_block (which block of m_diffs each f_val sits in).
        _block_per_fval = self.xp.repeat(self.m_diffs, self.norm_f_steps)
        if self.run_fdot:
            _block_per_entry = self.xp.tile(_block_per_fval, self.fdot_steps)
        else:
            _block_per_entry = _block_per_fval

        _table_cx = self.xp.zeros((total_f_fdot_vals,), dtype=complex)

        import time as _time
        _n_batches = len(batches) - 1
        _t_start = _time.perf_counter()
        if verbose:
            print(f"[build n_ref_complex] starting: {_n_batches} batches  "
                  f"f_steps={self.f_steps}  fdot_steps={self.fdot_steps}  "
                  f"total={total_f_fdot_vals}", flush=True)

        for _bi, (st_batch, end_batch) in enumerate(zip(batches[:-1], batches[1:])):
            _t_batch = _time.perf_counter()
            inds = np.arange(st_batch, end_batch)

            phase = 2 * np.pi * (
                _f_vals[inds, None] * t_diff[None, :]
                + 0.5 * _fdot_vals[inds, None] * t_diff[None, :] ** 2
            )
            # Single complex WDM transform: feeding the cos(phase) carrier
            # to ``is_complex=True`` yields a coefficient whose Re equals
            # the real cos-carrier WDM coefficient and Im equals the real
            # sin-carrier WDM coefficient at the same pixel (Hilbert
            # companion). This replaces the (sin, cos) pair of transforms
            # in ``_build_lookup_table_n_ref`` for the 2x build speedup.
            wave_in = self.xp.cos(phase)
            wave_wdm = TDSignal(
                wave_in,
                TDSettings(self.sub_settings.N, self.sub_settings.data_dt, force_backend=self.force_backend),
            ).wdmtransform(settings=self.sub_settings, window=self.td_window)

            raw_pix = wave_wdm[:, self.m_ref, _sub_n_ref]
            raw_cos = self.xp.real(raw_pix)
            raw_sin = self.xp.imag(raw_pix)

            # From here on the storage mirrors ``_build_lookup_table_n_ref``
            # so the eval-side (real-path) logic in ``get_wdm_coeffs`` can
            # consume Re/Im of ``table_cx`` as drop-in replacements for
            # (cos_table, sin_table). Each step below has the same meaning
            # as in the real build path; see that method for the detailed
            # derivation comments.
            _block_per_inds = _block_per_entry[inds].astype(int)
            _k_is_odd = (_block_per_inds & 1) != 0
            if self.is_m_ref_n_ref_even:
                sin_rot = self.xp.where(_k_is_odd, -raw_cos, raw_sin)
                cos_rot = self.xp.where(_k_is_odd,  raw_sin, raw_cos)
            else:
                sin_rot = self.xp.where(_k_is_odd,  raw_cos, raw_sin)
                cos_rot = self.xp.where(_k_is_odd, -raw_sin, raw_cos)
            _is_odd_per_inds = (
                ((self.m_ref + _block_per_inds + self.n_ref) % 2) == 1
            )
            sin_pixel = self.xp.where(_is_odd_per_inds, cos_rot, sin_rot)
            cos_pixel = self.xp.where(_is_odd_per_inds, sin_rot, cos_rot)
            _block_sign = self.xp.where(_k_is_odd, -1.0, 1.0)
            sin_pixel = sin_pixel * _block_sign
            cos_pixel = cos_pixel * _block_sign
            _table_cx[inds] = cos_pixel + 1j * sin_pixel

            if verbose:
                _elapsed = _time.perf_counter() - _t_start
                _per_batch = _elapsed / (_bi + 1)
                _eta = _per_batch * (_n_batches - _bi - 1)
                print(f"[build n_ref_complex] batch {_bi + 1:4d}/{_n_batches}  "
                      f"inds=[{int(st_batch):5d}, {int(end_batch):5d})  "
                      f"this_batch={_time.perf_counter() - _t_batch:.2f}s  "
                      f"elapsed={_elapsed/60.0:5.2f}m  ETA={_eta/60.0:5.2f}m",
                      flush=True)

        _table_cx = _table_cx.reshape(self.fdot_steps, self.f_steps).copy()
        assert _table_cx.shape == (self.fdot_vals.shape[0], self.f_vals.shape[0])
        self.table_cx = _table_cx

        if store_path is not None:
            self.to_file(store_path)

    @property
    def run_fdot(self) -> bool:
        """Whether the table has an fdot axis (``fdot_vals`` is not just ``[0.0]``)."""
        return not (len(self.fdot_vals) == 1 and self.fdot_vals[0] == 0.0)
    
    @property 
    def f_vals_norm(self) -> np.ndarray:
        """Table frequency axis relative to the reference layer, ``f_vals - f_ref`` (Hz)."""
        return self.f_vals - self.f_ref
    
    @property 
    def f_steps(self) -> np.ndarray:
        """Length of the frequency axis (int), ``len(f_vals)``."""
        return len(self.f_vals)
    
    @property 
    def norm_f_steps(self) -> np.ndarray:
        """Frequency samples per layer block (int), ``len(norm_freq_single_layer)``."""
        return len(self.norm_freq_single_layer)
    
    @property 
    def fdot_steps(self) -> np.ndarray:
        """Length of the fdot axis (int)."""
        return len(self.fdot_vals)
    
    @property 
    def t_ref(self) -> np.ndarray:
        """Reference time ``n_ref * layer_dt`` (s)."""
        return self.n_ref * self.layer_dt
    
    @property 
    def f_ref(self) -> np.ndarray:
        """Reference frequency ``m_ref * layer_df`` (Hz)."""
        return self.m_ref * self.layer_df
    
    @property
    def f_vals(self) -> np.ndarray:
        """Absolute table frequencies: one layer of offsets per ``m_diffs`` block (Hz, uniform)."""
        freq_single_layer = self.norm_freq_single_layer + self.f_ref
        f_vals = self.xp.concatenate([freq_single_layer + m_tmp * self.layer_df for m_tmp in self.m_diffs])
        assert self.xp.allclose(_tmp := self.xp.diff(f_vals), _tmp[0])
        return f_vals
        
    
    @staticmethod
    def get(x: np.ndarray) -> np.ndarray:
        """Return ``x`` as a NumPy array. Thin wrapper around :func:`asnumpy`."""
        return asnumpy(x)

    @property
    def points(self) -> np.ndarray:
        """Table nodes: ``(fdot_steps * f_steps, 2)`` ``(f, fdot)`` pairs, or ``f_vals``."""
        if self.run_fdot:
            return self.xp.asarray([tmp.ravel() for tmp in self.xp.meshgrid(self.f_vals, self.fdot_vals)]).T
        else:
            return self.f_vals
        
    @property
    def norm_points(self) -> np.ndarray:
        """As :attr:`points` with ``f_vals_norm`` in place of ``f_vals``."""
        if self.run_fdot:
            return self.xp.asarray([tmp.ravel() for tmp in self.xp.meshgrid(self.f_vals_norm, self.fdot_vals)]).T
        else:
            return self.f_vals_norm
        
    @property
    def table_sin(self) -> np.ndarray:
        """Response to the sin carrier, ``(fdot_steps, f_steps)`` (real kinds).

        Setting it rebuilds :attr:`table_sin_interpolate`.
        """
        return self._table_sin
    
    @property
    def table_sin_interpolate(self) -> np.ndarray:
        """Interpolator over :attr:`table_sin` (see :meth:`build_interpolator`)."""
        return self._table_sin_interpolate

    @table_sin.setter
    def table_sin(self, table_sin: np.ndarray):
        self._table_sin = table_sin
        self._table_sin_interpolate = self.build_interpolator(table_sin)

    @property
    def table_cos(self) -> np.ndarray:
        """Response to the cos carrier, ``(fdot_steps, f_steps)`` (real kinds).

        Setting it rebuilds :attr:`table_cos_interpolate`.
        """
        return self._table_cos
    
    @property
    def table_cos_interpolate(self) -> np.ndarray:
        """Interpolator over :attr:`table_cos` (see :meth:`build_interpolator`)."""
        return self._table_cos_interpolate
    
    @table_cos.setter
    def table_cos(self, table_cos: np.ndarray):
        self._table_cos = table_cos
        self._table_cos_interpolate = self.build_interpolator(table_cos)

    @property
    def table_cx(self) -> np.ndarray:
        """``n_ref_complex`` table, ``(fdot_steps, f_steps)``: ``Re`` = cos, ``Im`` = sin response.

        Setting it rebuilds the ``Re`` / ``Im`` interpolators.
        """
        return self._table_cx

    @property
    def table_cx_re_interpolate(self) -> np.ndarray:
        """Interpolator over ``Re(table_cx)`` (the cos response)."""
        return self._table_cx_re_interpolate

    @property
    def table_cx_im_interpolate(self) -> np.ndarray:
        """Interpolator over ``Im(table_cx)`` (the sin response)."""
        return self._table_cx_im_interpolate

    @table_cx.setter
    def table_cx(self, table_cx: np.ndarray):
        # Build_kind 'n_ref_complex' stores ONE complex 2-D table; we
        # interpolate Re and Im separately on the same grid (equivalent to
        # interpolating the complex value, but compatible with the same
        # scipy/cupyx Real interpolators used by the real path).
        self._table_cx = table_cx
        self._table_cx_re_interpolate = self.build_interpolator(
            self.xp.real(table_cx).copy()
        )
        self._table_cx_im_interpolate = self.build_interpolator(
            self.xp.imag(table_cx).copy()
        )

    @property
    def settings(self) -> TDSettings:
        """A plain :class:`WDMSettings` with this table's grid (the annotation is stale)."""
        return WDMSettings(*self.args, **self.kwargs)

    def build_interpolator(self, table: np.ndarray):
        """Construct a scipy/cupyx interpolator over ``table`` for fast template generation.

        Dispatches on ``self.build_kind`` and :attr:`INTERP_METHOD`:

        * ``"n_ref_only"`` / ``"n_ref_complex"`` with ``INTERP_METHOD == "spline"``:
          :class:`_UniformCubicSpline` over ``(fdot_vals, f_vals_norm)`` (or
          ``f_vals_norm`` alone), on CPU or GPU;
        * ``"n_ref_only"`` / ``"n_ref_complex"`` otherwise: 1-D ``interp1d`` over
          ``f_vals_norm`` (no fdot), or 2-D ``RegularGridInterpolator`` over
          ``(fdot_vals, f_vals_norm)`` with ``method=INTERP_METHOD`` (zero outside);
        * ``"per_n"``: legacy 1-D ``interp1d`` over the ``(f_norm, n)``-encoded
          coordinate; fdot lookup unimplemented.

        Args:
            table: Table values, ``(fdot_steps, f_steps)`` for the ``n_ref`` kinds.

        Returns:
            callable: The interpolator (called with ``(..., 2)`` ``(fdot, f_norm)``
            points, or ``f_norm`` without fdot).

        Raises:
            NotImplementedError: For ``"per_n"`` with an fdot axis.
            ValueError: For an unknown ``build_kind``.
        """
        if self.backend.uses_cupy:
            interpolate = interpolate_gpu
        else:
            interpolate = interpolate_cpu

        if self.INTERP_METHOD == "spline" and self.build_kind in ("n_ref_only", "n_ref_complex"):
            return _UniformCubicSpline(self.xp.asarray(table), self.fdot_vals if self.run_fdot else None,
                                       self.f_vals_norm, self.xp, self.backend.uses_cupy)

        if self.build_kind in ("n_ref_only", "n_ref_complex"):
            # Both kinds use the same (fdot, f_norm) grid; the complex
            # variant invokes this twice (Re and Im) on the same axes.
            if self.run_fdot:
                # table shape: (fdot_steps, f_steps)
                return interpolate.RegularGridInterpolator(
                    (self.fdot_vals, self.f_vals_norm),
                    table,
                    method=self.INTERP_METHOD,
                    bounds_error=False,
                    fill_value=0.0,
                )
            # table shape: (1, f_steps) — flatten for interp1d
            return interpolate.interp1d(self.f_vals_norm, table.ravel())

        if self.build_kind == "per_n":
            if self.run_fdot:
                raise NotImplementedError(
                    "build_kind='per_n' + run_fdot is not wired up; "
                    "use build_kind='n_ref_only' instead."
                )
            # TODO: is this an okay way to linear spline?
            # factor_spacing separates time-layer-encoded coordinates; must
            # exceed the f_vals_norm span so the (f_norm + factor_spacing*n)
            # encoding stays monotone across n.
            span = float(self.norm_points.max() - self.norm_points.min())
            self.factor_spacing = max(1e-3, 10.0 * span)
            assert span < self.factor_spacing

            norm_points_in = self.xp.tile(self.norm_points, (self.Nt, 1)).flatten()
            n_arr_in = self.xp.repeat(self.xp.arange(self.Nt)[:, None], self.norm_points.shape[0], axis=-1).flatten()
            x_points_in = self.get_x_points_no_fdot(norm_points_in, n_arr_in)
            y_points_in = table.flatten()
            return interpolate.interp1d(x_points_in, y_points_in)

        raise ValueError(f"Unknown WDMLookupTable.build_kind={self.build_kind!r}")

    def get_x_points_no_fdot(self, f_norm: np.ndarray, n_arr: np.ndarray) -> np.ndarray:
        """1D evaluation coordinate that encodes both ``f_norm`` and the time-pixel index (per_n build only)."""
        return (f_norm + self.factor_spacing * n_arr)

    def get_table_coeffs(self, f_norm: np.ndarray, fdot_arr: np.ndarray, n_arr: np.ndarray) -> np.ndarray:
        """Look up sin/cos coefficients at the requested ``(f_norm, fdot, n)`` points.

        For ``build_kind='n_ref_only'`` the ``n_arr`` argument is unused
        (the table has no time-pixel axis) but kept in the signature so
        callers do not need to branch. The ``n_ref_complex`` build packs
        cos into ``Re(table_cx)`` and sin into ``Im(table_cx)``, so the
        eval path just reads Re/Im from the same interpolators. NaNs (outside
        ``interp1d`` support) are set to zero.

        Args:
            f_norm: Offsets from the reference layer (Hz), 1-D.
            fdot_arr: Frequency derivatives (Hz/s), same shape (ignored without an
                fdot axis).
            n_arr: Pixel indices (used by ``"per_n"`` only).

        Returns:
            tuple: ``(sin_coeffs, cos_coeffs)``, each shaped like ``f_norm``.
        """
        if self.build_kind == "n_ref_only":
            if self.run_fdot:
                pts = self.xp.stack([fdot_arr, f_norm], axis=-1)
                sin_coeffs = self.table_sin_interpolate(pts)
                cos_coeffs = self.table_cos_interpolate(pts)
            else:
                sin_coeffs = self.table_sin_interpolate(f_norm)
                cos_coeffs = self.table_cos_interpolate(f_norm)
        elif self.build_kind == "n_ref_complex":
            if self.run_fdot:
                pts = self.xp.stack([fdot_arr, f_norm], axis=-1)
                cos_coeffs = self.table_cx_re_interpolate(pts)
                sin_coeffs = self.table_cx_im_interpolate(pts)
            else:
                cos_coeffs = self.table_cx_re_interpolate(f_norm)
                sin_coeffs = self.table_cx_im_interpolate(f_norm)
        elif self.build_kind == "per_n":
            if self.run_fdot:
                raise NotImplementedError(
                    "build_kind='per_n' + run_fdot is not wired up; "
                    "use build_kind='n_ref_only' instead."
                )
            x_points_in = self.get_x_points_no_fdot(f_norm, n_arr)
            sin_coeffs = self.table_sin_interpolate(x_points_in)
            cos_coeffs = self.table_cos_interpolate(x_points_in)
        else:
            raise ValueError(f"Unknown WDMLookupTable.build_kind={self.build_kind!r}")

        sin_coeffs[self.xp.isnan(sin_coeffs)] = 0.0
        cos_coeffs[self.xp.isnan(cos_coeffs)] = 0.0
        return (sin_coeffs, cos_coeffs)

    def set_interp_method(self, method: str) -> None:
        """Switch the (fdot, f) interpolation and rebuild the interpolators (built at load).

        ``"linear"`` (historical default), ``"cubic"`` or ``"spline"``. Cubic removes most
        of linear's bias across the table's peaked response (A9, CD1L EMRI 1: amplitude
        deficit 2.5e-4 -> 5e-6, mismatch 1e-6 -> 3.5e-8); it is CPU (scipy) only, since
        cupyx's RegularGridInterpolator has no cubic method. ``"spline"`` is the
        prefiltered uniform cubic B-spline (:class:`_UniformCubicSpline`), available on
        CPU and GPU. Sets the instance's :attr:`INTERP_METHOD` and drops the cached
        seam-continuous sin interpolator.

        Args:
            method: ``"linear"``, ``"cubic"`` or ``"spline"``.

        Raises:
            ValueError: For any other ``method``.
        """
        if method not in ("linear", "cubic", "spline"):
            raise ValueError(f"interp method must be 'linear', 'cubic' or 'spline', got {method!r}")
        self.INTERP_METHOD = method
        self._sin_unbaked_interp = None
        if self.build_kind == "n_ref_complex":
            self.table_cx = self._table_cx
        else:
            self.table_sin = self._table_sin
            self.table_cos = self._table_cos

    def _sin_unbaked_coeffs(self, f_norm, fdot):
        """Sin-table value with the build's (-1)^block bake undone at the nodes.

        block = floor(f_norm / layer_df) of each NODE; undoing it node-wise makes the
        stored function continuous across block seams, so linear interpolation is
        valid everywhere (built lazily, once).
        """
        if getattr(self, "_sin_unbaked_interp", None) is None:
            raw = self.xp.imag(self.table_cx) if self.build_kind == "n_ref_complex" else self.table_sin
            raw = self.xp.asarray(raw).copy()
            node_block = self.xp.floor(self.f_vals_norm / self.layer_df + 1e-9).astype(int)
            sign = self.xp.where(node_block % 2 != 0, -1.0, 1.0)
            raw = raw * sign.reshape((1,) * (raw.ndim - 1) + (-1,))
            self._sin_unbaked_interp = self.build_interpolator(raw)
        if self.run_fdot:
            out = self._sin_unbaked_interp(self.xp.stack([fdot, f_norm], axis=-1))
        else:
            out = self._sin_unbaked_interp(f_norm)
        out = self.xp.asarray(out)
        out[self.xp.isnan(out)] = 0.0
        return out

    def get_wdm_coeffs(self, amp_arr: np.ndarray, phi_arr: np.ndarray, f_arr: np.ndarray, fdot_arr: np.ndarray, n_arr: np.ndarray, num_m_layers: int = 1, out_of_support: str = "raise"):
        """WDM coefficients of carriers ``amp cos(phi)`` on the layers around each carrier.

        For element ``i`` (one source / channel / time pixel ``n_arr[i]``) the
        carrier layer is ``m0 = int(f_arr[i] / layer_df)`` and column ``j`` of the
        output is layer ``m0 + j - num_m_layers`` (``j = 0 .. 2 * num_m_layers``).
        Each value is read from the table at ``(fdot, f - m * layer_df)`` and mapped
        to pixel ``(m, n)`` by :attr:`BASIS_CYCLE` (``"quarter_turn"``, the
        production rule: the sin term comes from the seam-continuous table, an odd
        ``m_ref + n_ref`` flips ``c``, an odd absolute pixel parity ``m + n`` turns
        ``(c, s) -> (s, -c)``, then ``w = amp * (c cos(phi) - s sin(phi))``). This
        is the per-layer Python reference of the vectorized
        :meth:`lisatools.wdm_lookup_eval.WDMLookupEvaluator.coeffs`.

        Args:
            amp_arr: Carrier amplitudes, ``(N,)``.
            phi_arr: Carrier phases at the pixel times (rad), ``(N,)``.
            f_arr: Instantaneous frequencies (Hz), ``(N,)``.
            fdot_arr: Frequency derivatives (Hz/s), ``(N,)``.
            n_arr: Time-pixel indices, ``(N,)``.
            num_m_layers: Layers on each side of the carrier layer. Default ``1``
                (three columns).
            out_of_support: ``"raise"`` (default) raises when an offset or fdot falls
                outside the table; ``"zero"`` leaves out-of-support frequency offsets
                at zero.

        Returns:
            tuple: ``(wdm_coeffs_out, m_map)``, both ``(N, 2 * num_m_layers + 1)``:
            the real coefficients and the layer index of each column (``-1`` where
            the layer falls outside ``[0, Nf)`` or the entry was dropped).

        Raises:
            ValueError: With ``out_of_support="raise"``, for a frequency offset outside
                the table; in either mode, for an ``fdot`` outside the fdot axis.
        """
        ms = (f_arr / self.layer_df).astype(int)
        if getattr(self, "_f_norm_bounds", None) is None:     # f_vals rebuilds + checks on every access
            _fv = self.f_vals_norm
            self._f_norm_bounds = (float(_fv.min()), float(_fv.max()))
            self._fdot_bounds = (float(self.xp.min(self.xp.asarray(self.fdot_vals))),
                                 float(self.xp.max(self.xp.asarray(self.fdot_vals))))
        f_lo, f_hi = self._f_norm_bounds
        fd_lo, fd_hi = self._fdot_bounds
        wdm_coeffs_out = self.xp.zeros((amp_arr.shape[0], num_m_layers * 2 + 1))
        m_map = -self.xp.ones((amp_arr.shape[0], num_m_layers * 2 + 1), dtype=int)
        is_m_ref_n_ref_even = (self.m_ref + self.n_ref) % 2 == 0
        for i, m_diff in enumerate(range(-num_m_layers, num_m_layers + 1)):

            ms_to_use = (ms + m_diff).astype(int)
            keep_now = self.xp.arange(ms_to_use.shape[0])[(ms_to_use >= 0) & (ms_to_use < self.Nf)]

            f_norm = (f_arr[keep_now] - ms_to_use[keep_now] * self.layer_df)

            _in = (f_norm >= f_lo) & (f_norm <= f_hi)
            if out_of_support == "zero" and not bool(self.xp.all(_in)):
                # entries outside the table (far layers, ~1e-8 of the power) contribute 0
                keep_now = keep_now[_in]
                f_norm = f_norm[_in]
                if keep_now.size == 0:
                    continue
            if not bool(self.xp.all((f_norm >= f_lo) & (f_norm <= f_hi))):
                raise ValueError(
                    "WDMLookupTable.get_wdm_coeffs: f_norm outside the table's frequency support "
                    f"[{f_lo:.3e}, {f_hi:.3e}] Hz"
                )
            if not bool(self.xp.all((fdot_arr[keep_now] >= fd_lo) & (fdot_arr[keep_now] <= fd_hi))):
                raise ValueError("WDMLookupTable.get_wdm_coeffs: fdot outside the table's fdot axis")

            _sin_coeffs, _cos_coeffs = self.get_table_coeffs(f_norm, fdot_arr[keep_now], n_arr[keep_now])

            # The n_ref_complex storage packs (cos, sin) into (Re, Im) of
            # table_cx after the same build-time heroics; the eval path
            # below reuses the real-path branch unchanged.
            if self.build_kind in ("n_ref_only", "n_ref_complex") and self.BASIS_CYCLE == "quarter_turn":
                # Derived numerically (scripts/wdm/derive_basis_cycle.py; worst 1.4e-5 of
                # the peak over 2688 (pixel, offset, fdot) samples in 6 offset blocks):
                # the stored chirp (sin) term carries a (-1)^block bake, then pixels of odd
                # absolute parity (m + n) take a -90 degree turn (c, s) -> (s, -c); tables
                # with odd (m_ref + n_ref) first map (c, s) -> (-c, s) (checked for m_ref
                # 20 and 21: worst 1.4e-5 of peak each).
                # No other sign: the raw table values already carry the build's baking.
                # s from the seam-continuous table (block sign undone at the nodes): a
                # per-pixel flip of interpolated values is wrong inside the one-step
                # interval straddling each block seam (f exactly on a layer boundary).
                s_eff = self._sin_unbaked_coeffs(f_norm, fdot_arr[keep_now])
                # odd (m_ref + n_ref): the build baked a different rotation
                # (is_m_ref_n_ref_even) -> the table's (c, s) maps to (-c, s) first
                c_eff = -_cos_coeffs if (self.m_ref + self.n_ref) % 2 else _cos_coeffs
                odd = (ms_to_use[keep_now] + n_arr[keep_now]) % 2 != 0     # ABSOLUTE pixel parity
                cos_coeffs = self.xp.where(odd, s_eff, c_eff)
                sin_coeffs = self.xp.where(odd, -c_eff, s_eff)
                wdm_coeffs_out[keep_now, i] = amp_arr[keep_now] * (
                    cos_coeffs * self.xp.cos(phi_arr[keep_now]) - sin_coeffs * self.xp.sin(phi_arr[keep_now])
                )   # cos(x + phi) = cos x cos phi - sin x sin phi; (c, s) = responses to cos x, sin x
                m_map[keep_now, i] = ms_to_use[keep_now]
                continue

            if self.build_kind in ("n_ref_only", "n_ref_complex"):
                # legacy_2way (historical, stalled at mm ~0.5); kept ONLY as the
                # negative control of tests/test_wdm_lookup_basis_cycle.py.
                _dn_sign = self.xp.where(
                    ((n_arr[keep_now] - self.n_ref) % 2 == 1), -1.0, 1.0
                )
                _sin_coeffs = _sin_coeffs * _dn_sign
                _cos_coeffs = _cos_coeffs * _dn_sign

                is_m_plus_n_even = (((ms_to_use[keep_now] + n_arr[keep_now]) % 2 == 0))
                is_m_even = (ms_to_use[keep_now] % 2 == 0)

                sin_coeffs = self.xp.zeros_like(_sin_coeffs)
                cos_coeffs = self.xp.zeros_like(_cos_coeffs)

                sin_coeffs[~is_m_plus_n_even & is_m_even] = _sin_coeffs[~is_m_plus_n_even & is_m_even]
                cos_coeffs[~is_m_plus_n_even & is_m_even] = _cos_coeffs[~is_m_plus_n_even & is_m_even]

                sin_coeffs[~is_m_plus_n_even & ~is_m_even] = _sin_coeffs[~is_m_plus_n_even & ~is_m_even]
                cos_coeffs[~is_m_plus_n_even & ~is_m_even] = _cos_coeffs[~is_m_plus_n_even & ~is_m_even]

                sin_coeffs[is_m_plus_n_even & is_m_even] = _cos_coeffs[is_m_plus_n_even & is_m_even]
                cos_coeffs[is_m_plus_n_even & is_m_even] = _sin_coeffs[is_m_plus_n_even & is_m_even]

                sin_coeffs[is_m_plus_n_even & ~is_m_even] = _cos_coeffs[is_m_plus_n_even & ~is_m_even]
                cos_coeffs[is_m_plus_n_even & ~is_m_even] = _sin_coeffs[is_m_plus_n_even & ~is_m_even]
            else:
                is_m_plus_n_even = (((ms_to_use[keep_now] + n_arr[keep_now]) % 2 == 0))
                is_m_even = (ms_to_use[keep_now] % 2 == 0)

                sin_coeffs = self.xp.zeros_like(_sin_coeffs)
                cos_coeffs = self.xp.zeros_like(_cos_coeffs)

                sin_coeffs[~is_m_plus_n_even & is_m_even] = _sin_coeffs[~is_m_plus_n_even & is_m_even]
                cos_coeffs[~is_m_plus_n_even & is_m_even] = _cos_coeffs[~is_m_plus_n_even & is_m_even]

                sin_coeffs[~is_m_plus_n_even & ~is_m_even] = _sin_coeffs[~is_m_plus_n_even & ~is_m_even]
                cos_coeffs[~is_m_plus_n_even & ~is_m_even] = _cos_coeffs[~is_m_plus_n_even & ~is_m_even]

                sin_coeffs[is_m_plus_n_even & is_m_even] = _cos_coeffs[is_m_plus_n_even & is_m_even]
                cos_coeffs[is_m_plus_n_even & is_m_even] = _sin_coeffs[is_m_plus_n_even & is_m_even]

                sin_coeffs[is_m_plus_n_even & ~is_m_even] = _cos_coeffs[is_m_plus_n_even & ~is_m_even]
                cos_coeffs[is_m_plus_n_even & ~is_m_even] = _sin_coeffs[is_m_plus_n_even & ~is_m_even]

            wdm_coeffs_out[keep_now, i] = amp_arr[keep_now] * (sin_coeffs * self.xp.sin(phi_arr[keep_now]) + cos_coeffs * self.xp.cos(phi_arr[keep_now]))

            # Build now bakes (-1)^(m_diff_build) into the stored values so
            # that linear interp across f_norm block boundaries is smooth.
            # The block selected at lookup is m_diff_build = -m_diff_eval, so
            # the build flip imprints (-1)^(m_diff_eval) onto the looked-up
            # value. Combined with the original FFT-mirror correction
            # (-1)^(ms_source - m_ref), the net eval sign is
            # (-1)^((ms_to_use - m_ref) parity).
            sign = self.xp.where(((ms_to_use[keep_now] - self.m_ref) & 1) != 0, -1.0, 1.0)
            wdm_coeffs_out[keep_now, i] *= sign

            m_map[keep_now, i] = ms_to_use[keep_now]
        return wdm_coeffs_out, m_map


class DomainBaseArray:
    """Container for a collection of :class:`DomainBase` objects.

    When all signals share identical settings (uniform case), the signals are
    stacked into a single batched :class:`DomainBase`, enabling vectorized
    domain transforms (e.g. a single batched FFT instead of N sequential ones).
    Otherwise the class falls back to per-element processing.

    Args:
        signals: List of :class:`DomainBase` objects (unbatched, each
            ``(nchannels, *basis)``), or another :class:`DomainBaseArray` whose
            signals are reused.

    Raises:
        TypeError: If an element is not a :class:`DomainBase`.

    Attributes:
        signals (list[DomainBase]): The member signals.
        is_uniform (bool): Whether all members share equal settings.

    Example:
        .. code-block:: python

            arr = DomainBaseArray([td_a, td_b, td_c])        # same TDSettings
            wdm_list = arr.transform(wdm_settings)           # one batched transform
            wdm_list[0].arr.shape                            # (nchannels, Nf_active, Nt_active)
    """

    def __init__(self, signals: List[DomainBase] | DomainBaseArray) -> None:
        if isinstance(signals, DomainBaseArray):
            signals = signals.signals
            
        if not all(isinstance(s, DomainBase) for s in signals):
            raise TypeError("All elements of DomainBaseArray must be DomainBase instances.")
        self.signals = list(signals)

        if len(signals) > 1:
            s0 = signals[0].settings
            self.is_uniform = all(s.settings == s0 for s in signals[1:])
        else:
            self.is_uniform = True

        if self.is_uniform and len(signals) > 0:
            xp = get_array_module(signals[0].arr)
            arr_stacked = xp.stack([s.arr for s in signals], axis=0)
            settings = signals[0].settings
            self._batched = settings.associated_class(arr_stacked, settings)
        else:
            self._batched = None

    def __len__(self) -> int:
        return len(self.signals)

    def __iter__(self):
        return iter(self.signals)

    def __getitem__(self, index):
        return self.signals[index]

    def __add__(self, other: DomainBaseArray) -> "DomainBaseArray":
        """Concatenate the member lists of two :class:`DomainBaseArray` objects (not a sum).

        Raises:
            TypeError: If ``other`` is not a :class:`DomainBaseArray`.
        """

        if not isinstance(other, DomainBaseArray):
            raise TypeError("Can only add DomainBaseArray to another DomainBaseArray.")
        return DomainBaseArray(self.signals + other.signals)


    @property
    def batched(self) -> Optional[DomainBase]:
        """Batched :class:`DomainBase` (shape ``(nbatch, nchannels, *basis_shape)``),
        or ``None`` when settings are non-uniform."""
        return self._batched

    @property
    def settings(self) -> List[DomainSettingsBase]:
        """List of settings for each signal."""
        return [s.settings for s in self.signals]

    def transform(
        self,
        target_settings: DomainSettingsBase,
        window: Optional[np.ndarray] = None,
    ) -> "DomainBaseArray":
        """Transform all signals to *target_settings*.

        Uses a single vectorized call when all signals share the same settings
        (uniform case) and splits the batched result back into one signal per
        member; otherwise transforms each signal individually.

        Args:
            target_settings: Target domain settings.
            window: Optional window to apply during the transform.

        Returns:
            DomainBaseArray: The transformed signals, in order.
        """
        if self.is_uniform and self._batched is not None:
            transformed_batched = self._batched.transform(target_settings, window=window)
            # unstack along the batch axis
            transformed_signals = [
                target_settings.associated_class(transformed_batched.arr[i], target_settings)
                for i in range(len(self.signals))
            ]
            return DomainBaseArray(transformed_signals)
        else:
            return DomainBaseArray(
                [s.transform(target_settings, window=window) for s in self.signals]
            )

__available_domains__ = [TDSettings, FDSettings, STFTSettings, WDMSettings]

def get_available_domains() -> List[DomainSettingsBase]:
    """Return the list of :class:`DomainSettingsBase` subclasses supported by the package.

    Returns:
        list: ``[TDSettings, FDSettings, STFTSettings, WDMSettings]``.
    """
    return __available_domains__


# from .detector import LISAModel, ExtendedLISAModel


# class WDMSensitivityMatrix(WDMSettings, SensitivityMatrix):
#     def __init__(self, models, settings, base_sens_mat, psd_kwargs=None):
#         WDMSettings.__init__(self, *settings.args, **settings.kwargs)

#         if isinstance(models, LISAModel) or isinstance(models, ExtendedLISAModel):
#             models = [models for _ in range(self.Nt)]

#         for _tmp in models:
#             assert isinstance(_tmp, LISAModel) or isinstance(_tmp, ExtendedLISAModel)

#         if psd_kwargs is None:
#             psd_kwargs = [{} for _ in range(self.Nt)]
#         elif isinstance(psd_kwargs, dict):
#             psd_kwargs = [psd_kwargs for _ in range(self.Nt)]

#         for _tmp in psd_kwargs:
#             assert isinstance(_tmp, dict)

#         assert isinstance(models, list) and isinstance(psd_kwargs, list)
#         assert len(models) == len(psd_kwargs) == self.Nt

#         sens_mats = [base_sens_mat(settings.f_arr, model=model, **kwargs) for model, kwargs in zip(models, psd_kwargs)]
#         self.models = models
#         self.psd_kwargs = psd_kwargs

#         tmp_arr = xp.asarray([tmp_mat.sens_mat for tmp_mat in sens_mats])

#         SensitivityMatrix.__init__(self, settings.f_arr, tmp_arr)
