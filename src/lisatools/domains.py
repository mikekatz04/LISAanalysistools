"""Time, frequency, and time-frequency domain settings and array wrappers.

Defines :class:`DomainSettingsBase` (with its concrete subclasses :class:`TDSettings`,
:class:`FDSettings`, :class:`STFTSettings`, :class:`WDMSettings`) and the matching
:class:`DomainBase` array wrappers used throughout the package to keep arrays
tagged with their basis information so that transforms (FFT/iFFT, STFT, WDM)
can be performed automatically.
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

@dataclasses.dataclass
class DomainSettingsBase(LISAToolsParallelModule):
    """Base class for domain settings (TD, FD, STFT, WDM, ...).

    Carries the ``force_backend`` selector that decides whether NumPy or CuPy
    is used for arrays in this domain. Subclasses must implement
    :meth:`get_slice` and define an ``associated_class`` mapping the settings
    to a concrete :class:`DomainBase` subclass.

    Args:
        force_backend: Backend name passed to
            :class:`~lisatools.utils.parallelbase.LISAToolsParallelModule`
            (e.g. ``"cpu"``, ``"cuda12x"``).
    """

    force_backend: str = None

    def __init__(self, force_backend: str = None):
        self.force_backend = force_backend
        LISAToolsParallelModule.__init__(self, force_backend=force_backend)

    @classmethod
    def supported_backends(cls):
        """Return the list of backend names this settings class supports.

        Post-Phase-3E deprecation: returns ``lisatools_*`` names. The old
        ``fastlisaresponse_*`` names were retired when the responselisa
        pybind11 module was retired (see lisa-on-gpu's CLAUDE.md).
        """
        return ["lisatools_" + _tmp for _tmp in cls.GPU_RECOMMENDED()]

    def get_slice(self, index: tuple) -> DomainSettingsBase:
        """Return a new settings object describing a sliced view of this domain."""
        raise NotImplementedError("get_slice needs to be implemented for this signal type.")

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
    domain's basis (e.g. frequency bins, time bins, time-frequency cells); any
    leading dimensions are interpreted as ``(nbatch?, nchannels)``.

    Args:
        arr: The underlying NumPy or CuPy array.
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
        """Underlying NumPy or CuPy array."""
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
        """Transform this signal into ``new_domain`` (subclasses must implement)."""
        raise NotImplementedError("Transform needs to be implemented for this signal type.")

    def with_backend(self, force_backend) -> "DomainBase":
        """Return this signal on another backend (one host<->device transfer).

        Same signal class and identical settings, reconstructed with
        ``force_backend`` (a backend name like ``"cpu"``/``"cuda12x"`` or a
        Backend object); the data array is uploaded (``xp.asarray``) or
        downloaded (:func:`asnumpy`) exactly once. Returns ``self`` when the
        backend already matches.

        This is the sanctioned crossing point between backends: everything
        derived from the returned object lives on the new backend, keeping
        the sprint rule "one instance = one backend" intact while still
        letting CPU-loaded data (disk reads are host-side by nature) enter a
        GPU pipeline.
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
        """Return a sliced copy with both the array and its settings restricted to ``index``."""
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
        """Produce a log-log plot of the (FD-domain) signal.

        Args:
            ax: Matplotlib Axes (or list of Axes) to draw on. If ``None`` a new figure is created.
            fig: Matplotlib Figure. Not used directly when ``ax`` is created here.
            inds: Channel indices to draw. Defaults to all channels.
            char_strain: If ``True``, plot ``f * |arr|`` instead of ``|arr|``.
            **kwargs: Forwarded to ``ax.loglog``.

        Returns:
            ``(fig, ax)`` tuple.
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

        Domain-aware: handles partial overlap in time/frequency when the
        template covers a sub-range of ``self`` (FD, TD, STFT). For WDM
        signals the template must already share ``self``'s active grid.

        Args:
            template: A :class:`DomainBase` of the same domain family as ``self``.
            sign: +1 to add, -1 to subtract. Use :meth:`subtract_signal` for clarity.
            copy: If ``True``, operate on a copy and return it. If ``False``
                (default), modify ``self`` in place and return ``self``.

        Returns:
            ``self`` (or the copy when ``copy=True``).
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
        """In-place subtract ``template`` from this signal (see :meth:`add_signal`)."""
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
    """Add ``sign * template_arr`` (WDM) to ``target.arr``; shapes must already match."""
    if target.arr.shape[-2:] != template_arr.shape[-2:]:
        raise ValueError(
            f"WDM add_signal requires matching (Nf_active, Nt_active) shapes; "
            f"got data {target.arr.shape[-2:]} vs template {template_arr.shape[-2:]}."
        )
    target.arr[...] += sign * template_arr


class TDSettings(DomainSettingsBase):
    """Time-domain basis settings.

    Args:
        N: Number of time samples.
        dt: Sample spacing in seconds.
        t0: Start time in seconds. Defaults to ``0.0``.
        **kwargs: Forwarded to :class:`DomainSettingsBase` (e.g. ``force_backend``).
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
        """Return a ``slice`` along the time axis covering ``[tmin, tmax]``."""
        if tmin < self.t0:
            raise ValueError("tmin must be greater than or equal to t0.")
        if tmax > self.t0 + self.N * self.dt:
            raise ValueError("tmax must be less than or equal to t0 + N*dt.")

        start_idx = int(self.xp.round((tmin - self.t0) / self.dt))
        end_idx = int(self.xp.round((tmax - self.t0) / self.dt))

        return slice(start_idx, end_idx)

    def get_slice(self, index=None, tmin: float = None, tmax: float = None) -> TDSettings:
        """
        Return a new TDSettings object corresponding to the slice of the time points specified by index.

        Args:
            index: A slice object for the time dimension, e.g. slice(0, 10). If provided, this will be used to compute the new t0 and N for the sliced settings.
            tmin: Minimum time value for the slice. If provided, this will be used to compute the new t0 and N for the sliced settings.
            tmax: Maximum time value for the slice. If provided, this will be used to compute the new t0 and N for the sliced settings.

        If both index and (tmin, tmax) are provided, they must be consistent with each other (i.e. the time range specified by index must match the time range specified by tmin and tmax).

        Returns:
            A new TDSettings object corresponding to the slice of the time points specified by index or (tmin, tmax).
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

    Args:
        arr: NumPy or CuPy array with shape ``(..., N)``.
        settings: :class:`TDSettings` describing the time grid.
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

        Args:
            settings: Optional target :class:`FDSettings`; if ``None``, one is built
                from the inferred ``df`` and FFT length.
            window: Optional window applied before the transform.

        Returns:
            :class:`FDSignal` containing the (possibly trimmed) frequency-domain array.
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

        Args:
            settings: :class:`STFTSettings` describing the segment grid (required).
            window: Optional per-segment window (length ``nperseg``).

        Returns:
            :class:`STFTSignal` containing the time-frequency array sliced to
            ``settings.active_slice``.
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
        """Transform to the WDM wavelet basis via FFT then :meth:`FDSignal.wdmtransform`."""
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
        branch, whose window must be segment-length).
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
        ``(padded_times, padded_signals)`` with the same leading dimensions
        as the inputs.
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
        signals: Signal array of shape ``(..., num_times)``.
        settings: :class:`TDSettings` describing the target grid.
        times: Time array of shape ``(num_times,)`` giving the sample times
            of ``signals``. Must lie on the ``settings.t0 + k * settings.dt``
            grid. If ``None``, the signal is assumed to already start at
            ``settings.t0`` (right-pad / clip only).

    Returns:
        :class:`TDSignal` on the full ``settings`` grid.
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

    if times.ndim != 1:
        raise NotImplementedError(
            "place_td_signal_on_grid handles one source at a time; loop over "
            "the batch dimension for batched inputs."
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
    """Frequency-domain basis settings on a uniform grid.

    Args:
        N: Total number of frequency bins on the underlying ``arange(0, N) * df`` grid.
        df: Frequency spacing in Hz.
        min_freq: Lower edge of the active band; bins below are masked out via
            :attr:`active_slice`. Defaults to ``0.0``.
        max_freq: Upper edge of the active band; if ``None``, the full range is used.
        **kwargs: Forwarded to :class:`DomainSettingsBase` (e.g. ``force_backend``).
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
        from its inputs so the active band lives on the natural rFFT grid.
        """
        def _factory(times, dt, force_backend):
            Nt = len(times)
            df = 1.0 / (Nt * dt)
            Nf = Nt // 2 + 1
            return FDSettings(
                N=Nf, df=df, min_freq=min_freq, max_freq=max_freq,
                force_backend=force_backend,
            )
        return _factory

    @property
    def differential_component(self) -> float:
        """Differential element used in inner-product summations (``df`` in FD)."""
        return self.df
   
    @property
    def min_freq(self) -> float:
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
            ind_max = self.N - 1
        self._ind_max = ind_max

    @staticmethod
    def get_associated_class():
        """Return the :class:`DomainBase` subclass that pairs with these settings."""
        return FDSignal

    @staticmethod
    def get_associated_group():
        from .domaincomputation import FDComputationGroup
        return FDComputationGroup

    @property
    def associated_class(self):
        """The :class:`DomainBase` subclass that pairs with these settings."""
        return self.get_associated_class()

    @property
    def associated_group(self):
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
        """Active-band frequency array (Hz)."""
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

    Args:
        arr: NumPy or CuPy array. The trailing axis must have length ``N_active``
            (or ``N`` if no min/max masking is applied).
        settings: :class:`FDSettings` describing the frequency grid and band.
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
        """Zero-pad ``arr`` (2D) back to the full ``N``-bin grid before an inverse transform."""
        assert arr.ndim == 2
        _arr = self.xp.pad(arr, ((0, 0), (self.ind_min - 1, self.N - 1 - self.ind_max)), mode="constant", constant_values=0.0)
        return _arr

    def ifft(self, settings=None, window=None):
        """Inverse FFT back to the time domain (zero-padding the active band if trimmed)."""

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

        Args:
            settings: :class:`WDMSettings` describing the wavelet grid (required).
            window: Unused in the current implementation; the WDM analysis window
                is taken from ``settings.window``.
            return_transpose_time_axis_first: Currently has no effect (the
                transposed branch is commented out below).
            is_psd: If ``True``, treat the input as a PSD and follow the
                stationary-PSD shortcut (returns a raw NumPy/CuPy array instead
                of a :class:`WDMSignal`).

        Returns:
            :class:`WDMSignal` for ordinary signals, or a NumPy/CuPy array when
            ``is_psd is True``.
        """
        if settings is None:
            raise ValueError("Must provide WDMSettings for WDM transform.")
        assert isinstance(settings, WDMSettings)

        # Layer selection + rFFT gather map: shared with the PSD-fold fast
        # path in sensitivity.get_sensitivity, and cached on the settings
        # (it depends only on the grid and the active band).
        # TODO: WITH ROBBIE CHECK SECOND TO TOP INDEX START AND END
        Nf_act = settings.Nf_active
        include_top = (settings.ind_min_f == 0)
        n_special = Nf_act + (1 if include_top else 0)
        m_special_1d, k, herm, _ = settings.fold_shift_map()
        base_window = (settings.window[:])

        arr_in = self.arr.copy()

        if self.ind_min != 0 or self.ind_max != self.N - 1:
            warnings.warn("Doing an ifft with a trimmed frequency domain array. Zero-padding.")
            arr_in = self.pad_array(arr_in)

        before_ifft = arr_in[:, k] / settings.data_dt

        if not is_psd:
            if herm.any():
                before_ifft[:, herm] = self.xp.conj(before_ifft[:, herm])

        if is_psd:
            tmp_arr = before_ifft.copy()
            tmp_arr[:] *= (base_window[None, None, :]) ** 2 * np.pi * settings.data_dt
            psd_sum_tmp = tmp_arr.sum(axis=-1)
            psd_sum_tmp /= settings.Nf * settings.Nt   # = N

            wdmpsd_active = self.xp.zeros((self.nchannels, Nf_act, settings.Nt), dtype=complex)
            if include_top:
                # row 0 == m=0, row -1 == m=Nf, rows 1..Nf_act-1 == m=1..ind_max_f
                wdmpsd_active[:, 1:] = psd_sum_tmp[:, 1:Nf_act, None]
                wdmpsd_active[:, 0, 0::2] = psd_sum_tmp[:, 0, None]
                wdmpsd_active[:, 0, 1::2] = psd_sum_tmp[:, -1, None]
            else:
                # rows 0..Nf_act-1 map directly to m=ind_min_f..ind_max_f
                wdmpsd_active[:] = psd_sum_tmp[:, :Nf_act, None]

            wdmpsd_out = wdmpsd_active[:, :, settings.active_slice_t]
            return wdmpsd_out

        before_ifft[:] *= base_window[None, None, :]
        after_ifft = self.xp.fft.ifft(before_ifft, axis=-1)

        # TODO: fix this

        if self.backend.uses_cupy:
            # some issue with cupy and xp.real/imag
            cache = self.xp.fft.config.get_plan_cache()
            cache.clear()

        is_complex = bool(getattr(settings, "is_complex", False))
        out_dtype = complex if is_complex else float
        tmp_w_mn = self.xp.zeros((self.nchannels, n_special, settings.Nt), dtype=out_dtype)
        kappa = 2 * np.sqrt(np.pi * settings.data_dt) / settings.Nf
        m_here = self.xp.repeat(m_special_1d[:, None], settings.Nt, axis=-1)
        n_here = self.xp.tile(self.xp.arange(settings.Nt), (n_special, 1))
        set_zero = ((m_here == settings.Nf) | (m_here == 0)) & ((m_here + n_here) % 2 != 0)
        projected = self.xp.conj(settings.get_Cmn(m_here[~set_zero], n_here[~set_zero])) * after_ifft[:, ~set_zero]
        if is_complex:
            # keep both Re (standard WDM) and Im (Hilbert/quadrature companion)
            tmp_w_mn[:, ~set_zero] = (
                kappa * (-1) ** ((m_here + 1) * n_here)[~set_zero] * projected
            )
        else:
            tmp_w_mn[:, ~set_zero] = (
                kappa * (-1) ** ((m_here + 1) * n_here)[~set_zero] * self.xp.real(projected)
            )

        w_mn_active = self.xp.zeros((self.nchannels, Nf_act, settings.Nt), dtype=out_dtype)
        if include_top:
            w_mn_active[:, 1:] = tmp_w_mn[:, 1:Nf_act]
            w_mn_active[:, 0, 0::2] = tmp_w_mn[:, 0, 0::2] / np.sqrt(2.)
            w_mn_active[:, 0, 1::2] = tmp_w_mn[:, -1, 0::2] / np.sqrt(2.)
        else:
            w_mn_active[:] = tmp_w_mn[:, :Nf_act]

        output = w_mn_active[:, :, settings.active_slice_t]

        return WDMSignal(output, settings=settings)

    def transform(self, new_domain: DomainSettingsBase, window: np.ndarray | cp.ndarray = None):
        """Dispatch to :meth:`ifft`, :meth:`wdmtransform`, etc. based on ``new_domain``."""
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
        """
        Plot the squared amplitude of the FD signal for a given channel.

        Args:
            channel: The channel index to visualize.
            ax: An optional matplotlib Axes object to plot on. If None, a new figure and axes will be created.
            filename: An optional filename to save the plot to. If provided, the plot will be saved to this file.
            **kwargs: Additional keyword arguments to pass to the underlying plotting functions.

        Returns:
            The matplotlib Axes object containing the plot.
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
        integer multiple of ``dt`` and derives ``NT``/``NF`` from there.
        """
        def _factory(times, dt, force_backend):
            return get_stft_settings(
                times=times, big_dt=big_dt,
                min_freq=min_freq, max_freq=max_freq,
                force_backend=force_backend,
            )
        return _factory

    @staticmethod
    def get_associated_class():
        return STFTSignal

    @staticmethod
    def get_associated_group():
        from .domaincomputation import STFTComputationGroup
        return STFTComputationGroup

    @property
    def associated_class(self):
        return self.get_associated_class()

    @property
    def associated_group(self):
        return self.get_associated_group()

    @property
    def basis_shape(self) -> tuple:
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
        return self.NT * self.NF_active

    @property
    def t_arr(self) -> np.ndarray:
        return self.t0 + self.xp.arange(self.NT) * self.dt
   
    @property
    def min_freq(self) -> float:
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

    @property
    def f_arr(self) -> np.ndarray:

        _all_freqs = self.xp.arange(0, self.NF) * self.df
        return _all_freqs[self.active_slice]

    @property
    def args(self) -> tuple:
        return (self.t0, self.dt, self.df, self.NT, self.NF)

    @property
    def kwargs(self) -> dict:
        return dict(
            min_freq=self.min_freq, max_freq=self.max_freq, force_backend=self.backend
        )

    @property
    def f_arr_edges(self) -> np.ndarray:
        return self.xp.arange(self.NF + 1) * self.df

    @property
    def t_arr_edges(self) -> np.ndarray:
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
        return self.df

    @property
    def active_slice(
        self,
    ) -> slice:
        return slice(self.ind_min, self.ind_max + 1)

    @property
    def NF_active(self) -> int:
        sl = self.active_slice
        return sl.stop - sl.start

    def get_nperseg(self, small_dt: float):
        """Number of samples per segment given the underlying sample step ``small_dt``."""
        nperseg = round(self.dt / small_dt)

        assert (
            abs(nperseg * small_dt - self.dt) < 1e-10 * self.dt
        ), f"big_dt={self.dt} must be an integer multiple of dt={small_dt}"

        return nperseg

    def compute_slice_indices(
        self, tmin: float, tmax: float, fmin: float, fmax: float
    ) -> Tuple[slice, slice]:
        """
        Compute the slice indices for the time and frequency dimensions based on the provided min and max values.

        Args:
            tmin: Minimum time value for the slice.
            tmax: Maximum time value for the slice.
            fmin: Minimum frequency value for the slice.
            fmax: Maximum frequency value for the slice.

        Returns:
            A tuple of slices for the time and frequency dimensions, e.g. (slice(0, 10), slice(5, 15)).
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
        """
        Return a new STFTSettings object corresponding to the slice of the time and frequency points specified by index.

        Args:
            index: A tuple of slices for the time and frequency dimensions, e.g. (slice(0, 10), slice(5, 15)).

        Returns:
            STFTSettings: A new STFTSettings object corresponding to the slice.
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
    """
    Get STFT settings from time array and desired big_dt.

    Args:
        times: Time array.
        big_dt: Desired time resolution for STFT segments.
        min_freq: Minimum frequency to consider.
        max_freq: Maximum frequency to consider.
        **kwargs: Additional keyword arguments to pass to STFTSettings.

    Returns:
        STFTSettings: The settings for the STFT.
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

    Args:
        arr: NumPy or CuPy array with trailing axes ``(NT, NF_active)``.
        settings: :class:`STFTSettings` describing the time-frequency grid.
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
        """
        Visualize the STFT signal in either the time-frequency domain (stft), frequency domain (fd), or time domain (td).

        Args:
            channel: The channel index to visualize.
            ax: An optional matplotlib Axes object to plot on. If None, a new figure and axes will be created.
            plot_type: The type of plot to create. Must be one of 'stft', 'fd', or 'td'. 'stft' will create a time
                vs frequency plot of the magnitude squared of the STFT coefficients. 'fd' will create a log-log plot of the magnitude squared of the STFT coefficients for a single time bin. 'td' will create a plot of the magnitude squared of the real and imaginary parts of the STFT coefficients for a single frequency bin.
            filename: An optional filename to save the plot to. If provided, the plot will be saved to this file.
            **kwargs: Additional keyword arguments to pass to the underlying plotting functions.

        Returns:
            The matplotlib Axes object containing the plot.
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
    """Wavelet (WDM) basis settings for time-frequency analysis.

    Args:
        Nf: Number of frequency layers (must be even).
        Nt: Number of time pixels per layer (must be even).
        dt: Underlying time-domain sample step in seconds.
        t0: Start time in seconds. Defaults to ``0.0``.
        oversample: Frequency oversampling factor used when building the WDM
            window via :meth:`setup_window`. Defaults to ``16``.
        window: Pre-computed WDM analysis window of length ``Nt``. If provided,
            ``omega`` must also be supplied; otherwise the window is built by
            :meth:`setup_window`.
        omega: Pre-computed angular-frequency grid that pairs with ``window``.
        min_freq: Lower edge of the active frequency band (Hz). ``None`` selects
            the full range.
        max_freq: Upper edge of the active frequency band (Hz).
        min_time: Lower edge of the active time band (seconds).
        max_time: Upper edge of the active time band (seconds).
        **kwargs: Forwarded to :class:`DomainSettingsBase` (e.g. ``force_backend``).
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

        ``Nf``/``Nt`` come from the caller; ``dt`` is taken from the
        sample step of the input ``times`` array at factory-call time.
        """
        def _factory(times, dt, force_backend):
            return WDMSettings(
                Nf=Nf, Nt=Nt, dt=dt, oversample=oversample,
                min_freq=min_freq, max_freq=max_freq,
                min_time=min_time, max_time=max_time,
                force_backend=force_backend,
            )
        return _factory

    @staticmethod
    def adjust_to_even_bins(t_min: float, t_max: float, dt: float, Tobs: float, num_linspace: Optional[int]=1000, verbose: Optional[bool] = False) -> Tuple[int, int, float]:
        """Pick a wavelet pixel duration in ``[t_min, t_max]`` that makes both ``Nf`` and ``Nt`` even.

        Args:
            t_min: Lower bound on the wavelet pixel duration in seconds.
            t_max: Upper bound on the wavelet pixel duration in seconds.
            dt: Underlying time-sample step (the duration is rounded to a multiple).
            Tobs: Total observation time (seconds).
            num_linspace: Number of candidate durations to scan between ``t_min`` and ``t_max``.
            verbose: If ``True``, print each attempted duration.

        Returns:
            ``(Nf, Nt, wavelet_duration)``.

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
        return slice(self.ind_min_f, self.ind_max_f + 1)
    
    @property
    def active_slice_t(
        self,
    ) -> slice:
        return slice(self.ind_min_t, self.ind_max_t + 1)

    @property
    def active_slice(
        self,
    ) -> slice:
        return (self.active_slice_f, self.active_slice_t)
    
    @property
    def Nf_active(self) -> int:
        sl = self.active_slice_f
        return sl.stop - sl.start

    @property
    def Nt_active(self) -> int:
        sl = self.active_slice_t
        return sl.stop - sl.start

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
        if not isinstance(value, WDMSettings):
            return False
        return (
            (value.Nt == self.Nt) and (value.Nf == self.Nf)
            and (value.layer_dt == self.layer_dt) and (value.layer_df == self.layer_df)
            and (value.data_dt == self.data_dt)
            and (bool(getattr(value, "is_complex", False)) == bool(self.is_complex))
        )

    @property
    def basis_shape(self) -> tuple:
        return (self.Nf, self.Nt)
    
    @property
    def basis_shape_active(self) -> tuple:
        return (self.Nf_active, self.Nt_active)
    
    @property
    def t_td_arr(self) -> np.ndarray:
        _tmp = self.xp.arange(self.N) * self.data_dt
        tmp = _tmp[(_tmp >= self.ind_min_t * self.layer_dt) & (_tmp <= self.ind_max_t * self.layer_dt)]
        return tmp
    
    @property
    def t_arr(self) -> np.ndarray:
        return self.xp.arange(self.Nt)[self.active_slice_t] * self.layer_dt

    @property
    def f_arr(self) -> np.ndarray:
        return self.xp.arange(self.Nf)[self.active_slice_f] * self.layer_df
    
    @property
    def f_arr_edges(self) -> np.ndarray:
        return (self.xp.arange(self.Nf_active + 1) + self.ind_min_f) * self.layer_df
    @property
    def t_arr_edges(self) -> np.ndarray:

        return (self.xp.arange(self.Nt_active + 1) + self.ind_min_t) * self.layer_dt

    def phitilde(self, omega, dOmega):
        """Smooth WDM frequency-domain analysis function :math:`\\tilde{\\phi}(\\omega)`."""
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
        """Return ``1`` for even ``(m + n)`` and ``1j`` for odd ``(m + n)``."""
        m_in = self.xp.atleast_1d(m)
        n_in = self.xp.atleast_1d(n)
        output = self.xp.zeros(m_in.shape, dtype=complex)
        is_even = ((m_in + n_in) % 2) == 0
        output[is_even] = 1.0
        output[~is_even] = 1j
        return output

    def wavelet(self, N: int, in_fd: Optional[bool] = True) -> np.ndarray:
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
        """Return a 2D shift map ``m * Nt/2 + arange(-Nt/2, Nt/2)`` used by the WDM transform."""
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


    # def window_norm(self) -> float:
    #     dOmega_s = np.pi / self.Nf
    #     (2 * np.pi) / self.N 

    def setup_window(self):  # , forward: bool= True):
        """Build the default WDM analysis window and store it in :attr:`window` / :attr:`omega`."""
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
        return self._ind_min_f
    
    @ind_min_f.setter
    def ind_min_f(self, ind_min_f: int):
        if ind_min_f is None:
            ind_min_f = 0
        self._ind_min_f = ind_min_f

    @property
    def ind_max_f(self) -> int:
        return self._ind_max_f
    
    @ind_max_f.setter
    def ind_max_f(self, ind_max_f: int):
        if ind_max_f is None:
            ind_max_f = self.Nf - 1
        self._ind_max_f = ind_max_f

    @property
    def ind_min_t(self) -> int:
        return self._ind_min_t
    
    @ind_min_t.setter
    def ind_min_t(self, ind_min_t: int):
        if ind_min_t is None:
            ind_min_t = 0
        self._ind_min_t = ind_min_t

    @property
    def ind_max_t(self) -> int:
        return self._ind_max_t
    
    @ind_max_t.setter
    def ind_max_t(self, ind_max_t: int):
        if ind_max_t is None:
            ind_max_t = self.Nt - 1
        self._ind_max_t = ind_max_t

    @property
    def frequency_layer_mask(self) -> Optional[np.ndarray]:
        mask = self.xp.zeros(self.Nf, dtype=bool)
        mask[self.active_slice_f] = True
        return mask
    
    @property
    def time_layer_mask(self) -> Optional[np.ndarray]:
        mask = self.xp.zeros(self.Nt, dtype=bool)
        mask[self.active_slice_t] = True
        return mask
        
    @property
    def f_ind_array(self) -> np.ndarray:
        return self.xp.arange(self.ind_min_f, self.ind_max_f + 1)

    @property
    def t_ind_array(self) -> np.ndarray:
        return self.xp.arange(self.ind_min_t, self.ind_max_t + 1)

    @staticmethod
    def get_associated_class():
        return WDMSignal

    @property
    def associated_class(self):
        return self.get_associated_class()

    @property
    def kwargs(self) -> dict:
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
        return (self.Nf, self.Nt, self.data_dt)

    @property
    def differential_component(self) -> float:
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
        """Apply :attr:`frequency_layer_mask` to ``arr`` along the WDM frequency axis (penultimate dim)."""
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


class WDMSignal(WDMSettings, DomainBase):
    """WDM wavelet array wrapper paired with :class:`WDMSettings`.

    Args:
        arr: NumPy or CuPy array with trailing axes ``(Nf_active, Nt_active)``.
        settings: :class:`WDMSettings` describing the wavelet grid.
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
        return WDMSettings(*self.args, **self.kwargs)

    def __repr__(self) -> str:
        return (
            f"WDMSignal(Tobs={self.Tobs}, dt={self.data_dt}, t0={self.t0}, "
            f"Nt={self.Nt}, Nf={self.Nf}, oversample={self.oversample}, "
            f"backend={self.backend_name.split('_')[-1]})"
        )
    
    def wdm_to_td(self, settings=None, window=None):
        """Inverse-transform from WDM to time domain (via FD)."""
        return self.wdm_to_fd(settings=None).ifft(settings=settings, window=window)

    def wdm_to_fd(self, settings=None, window=None):
        """Inverse-transform from WDM to frequency domain.

        Args:
            settings: Optional target :class:`FDSettings`; if ``None``, one is
                derived from the underlying ``data_dt`` and ``N``.
            window: Currently unused (the WDM analysis window stored on the
                signal is used internally).

        Returns:
            :class:`FDSignal`.
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

        ``window=None`` is forwarded as-is to the FINAL step of each chain:
        every step defaults its own window with the shape it needs.
        (Pre-filling a WDM-shaped ones array here broke the layered chains
        — e.g. it reached ``ifft`` on the FD intermediate, whose window must
        be FD-length. Same rule as :meth:`TDSignal.transform`.)
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

        Args:
            index: If given, plot only channel ``index`` on the supplied ``ax``.
                If ``None``, every channel is plotted on its own row.
            mag: If ``True``, plot ``|coeff|``; otherwise plot the signed value.
            fig: Existing :class:`matplotlib.figure.Figure` (optional).
            ax: Existing axes; required when ``index`` is provided.
            cax: Optional axes for the colourbar.
            add_cax: If ``True``, create a new colourbar axes.
            log: If ``True``, plot ``log10(|coeff|)``.
            **kwargs: Forwarded to :func:`matplotlib.pyplot.pcolormesh`.

        Returns:
            ``(fig, ax)``.
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


class DomainBaseArray:
    """Container for a collection of :class:`DomainBase` objects.

    When all signals share identical settings (uniform case), the signals are
    stacked into a single batched :class:`DomainBase`, enabling vectorized
    domain transforms (e.g. a single batched FFT instead of N sequential ones).
    Otherwise the class falls back to per-element processing.

    Args:
        signals: List of :class:`DomainBase` objects.

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
        """
        Define how to add two DomainBaseArrays together. This will concatenate the signals from both arrays into a single array.
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
        (uniform case); otherwise transforms each signal individually.

        Args:
            target_settings: Target domain settings.
            window: Optional window to apply during the transform.

        Returns:
            :class:`DomainBaseArray` of transformed signals.

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
    """Return the list of :class:`DomainSettingsBase` subclasses supported by the package."""
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
