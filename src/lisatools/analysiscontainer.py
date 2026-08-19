"""High-level analysis containers combining data, sensitivity, and signal generation."""

from __future__ import annotations

import math
import os
import warnings
from abc import ABC
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
import logging
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from eryn.utils import TransformContainer
from scipy import interpolate

from .domains import DomainBase, DomainBaseArray, DomainSettingsBase

from . import domains

try:
    import cupy as cp

except (ModuleNotFoundError, ImportError):
    import numpy as cp

from . import detector as lisa_models
from .datacontainer import DataResidualArray
from .diagnostic import (
    data_signal_full_source_and_noise_likelihood,
    data_signal_source_likelihood_term,
    inner_product,
    noise_likelihood_term,
    residual_full_source_and_noise_likelihood,
    residual_source_likelihood_term,
)
from .sensitivity import SensitivityMatrix, SensitivityMatrixBase
from .utils.exceptions import WaveformDomainError
from .stochastic import FittedHyperbolicTangentGalacticForeground, StochasticContribution
from .utils.constants import *
from .utils.utility import AET, get_array_module, asnumpy


SignalGenSpec = Union[Callable, Mapping[str, Callable]]


def _coerce_to_domain_base(obj) -> DomainBase:
    """Return ``obj`` as a :class:`DomainBase` (unwrap a :class:`DataResidualArray`)."""
    if isinstance(obj, DomainBase):
        return obj
    if isinstance(obj, DataResidualArray):
        # ``DataResidualArray.data_res_arr`` is the underlying DomainBase.
        return obj.data_res_arr
    raise TypeError(
        f"Expected a DomainBase child (FDSignal, WDMSignal, TDSignal, STFTSignal) "
        f"or a DataResidualArray; got {type(obj).__name__}."
    )

def _same_device(src, dst) -> bool:
    """Whether ``src`` and ``dst`` are arrays sitting on the same device.

    Used by the repack paths (:meth:`AnalysisContainerArray.reset_linear_data_arr`
    / :meth:`~AnalysisContainerArray.reset_linear_psd_arr`) to tell a genuinely
    cross-device source -- which must round-trip through the host, because P2P
    copies are broken here -- from the common same-device case, which can stay
    resident. Returns ``False`` whenever either side lacks a ``device``
    attribute (e.g. NumPy < 2), which simply keeps the conservative host route.
    """
    src_dev = getattr(src, "device", None)
    dst_dev = getattr(dst, "device", None)
    if src_dev is None or dst_dev is None:
        return False
    try:
        return bool(src_dev == dst_dev)
    except Exception:  # pragma: no cover - exotic device objects
        return False


logger = logging.getLogger(__name__)

#: One WARNING per process for the multi-GPU split notice (see
#: AnalysisContainerArray.__init__); repeats demote to DEBUG.
_MULTI_GPU_SPLIT_WARNED = False

#: Keywords handled by :meth:`AnalysisContainer._calculate_signal_operation`
#: rather than forwarded to ``inner_product``. The batched likelihood path
#: cannot express these, so it declines any call carrying one.
_CONTAINER_LEVEL_KWARGS = frozenset(
    {"transform_fn", "apply_transform", "signal_gen", "per_model_per_signal"}
)


class AnalysisContainer:
    """Combinatorial container that combines sensitivity and data information.

    Args:
        data: Data / Residual / Signal information. May be a :class:`DomainBase`
            child (e.g. :class:`~lisatools.domains.FDSignal`,
            :class:`~lisatools.domains.WDMSignal`) or a legacy
            :class:`DataResidualArray` (transparently unwrapped).
        sens_mat: Sensitivity information. Accepts either:

            - an instance of :class:`~lisatools.sensitivity.SensitivityMatrixBase`
              (used directly), or
            - a subclass of :class:`~lisatools.sensitivity.SensitivityMatrixBase`
              -- the class is instantiated on the fly as
              ``sens_mat(data.settings, **sens_mat_kwargs)`` so callers don't
              have to repeat the settings extraction at every construction site.

        signal_gen: Either

            - a single ``Callable`` (legacy single-model usage), or
            - a ``dict`` mapping ``model_name -> Callable``. When a dict is
              given, the multi-model API in
              :meth:`calculate_signal_likelihood` / ``_inner_product`` /
              ``_snr`` is enabled: pass a ``dict`` of per-model parameter
              arrays (1D or 2D), and per-model templates are summed within
              each model along axis 0 and then across models before the
              inner-product step.

            In either case, the generator's output is automatically converted
            into ``data.settings`` (via :meth:`DomainBase.transform`) if the
            generator returns a :class:`DomainBase` in a different domain.

        sens_mat_kwargs: Extra keyword arguments forwarded when ``sens_mat`` is
            a class (the auto-instantiate path). Must be empty when
            ``sens_mat`` is already an instance.

        likelihood_source_only: Container-level default for the ``source_only``
            flag of the likelihood methods (:meth:`likelihood`,
            :meth:`calculate_signal_likelihood`, ...). When ``True``, calls
            that do not pass ``source_only`` explicitly return the
            source-only Likelihood ``-1/2 <r|r>`` and leave out the noise
            normalization term ``-sum(log|detC|)``. Only set this on runs
            where the PSD is FIXED (no psd sampling branch) — with a fixed
            PSD the noise term is an overall constant, so dropping it leaves
            all likelihood *differences* unchanged while making the readout
            directly interpretable as the residual inner product. An
            explicit ``source_only=`` at a call site always wins.

        data_res_arr: Deprecated alias of ``data`` (kept for backward compatibility).

    """

    def __init__(
        self,
        data: Union[DomainBase, DataResidualArray, None] = None,
        sens_mat: Union[SensitivityMatrixBase, type, None] = None,
        signal_gen: Optional[SignalGenSpec] = None,
        sens_mat_kwargs: Optional[dict] = None,
        likelihood_source_only: bool = False,
        *,
        data_res_arr: Union[DomainBase, DataResidualArray, None] = None,
    ) -> None:
        if data is None and data_res_arr is not None:
            # Backward-compat: old kw name was ``data_res_arr``.
            data = data_res_arr
        if data is None:
            raise TypeError("AnalysisContainer requires a ``data`` argument.")
        if sens_mat is None:
            raise TypeError("AnalysisContainer requires a ``sens_mat`` argument.")

        # 1. normalise data to a DomainBase
        self._data: DomainBase = _coerce_to_domain_base(data)

        # 2. normalise sens_mat (instance | subclass)
        sens_mat_kwargs = sens_mat_kwargs or {}
        if isinstance(sens_mat, type):
            if not issubclass(sens_mat, SensitivityMatrixBase):
                raise TypeError(
                    "If passing sens_mat as a class, it must subclass "
                    "SensitivityMatrixBase."
                )
            sens_mat = sens_mat(self._data.settings, **sens_mat_kwargs)
        elif sens_mat_kwargs:
            raise ValueError(
                "sens_mat_kwargs is only meaningful when ``sens_mat`` is a "
                "SensitivityMatrixBase subclass (the auto-instantiate path)."
            )

        if not isinstance(sens_mat, SensitivityMatrixBase):
            raise TypeError(
                "sens_mat must be a SensitivityMatrixBase instance or a "
                "SensitivityMatrixBase subclass class."
            )
        self._sens_mat = sens_mat

        # 3. signal generator (callable or dict)
        if signal_gen is not None:
            self.signal_gen = signal_gen

        # 4. run-level likelihood convention (see class docstring).
        self.likelihood_source_only = bool(likelihood_source_only)

        # 5. batched evaluation (see :meth:`eryn_likelihood_wrap`).
        #: Try to evaluate a 2D parameter block in ONE generator call rather
        #: than looping. Only ever engaged when the installed generator
        #: advertises ``supports_batch``, so this default is inert for every
        #: generator that does not opt in.
        self.batch_evaluation: bool = True
        #: Split a batch into chunks of at most this many rows (``None`` = one
        #: launch). Bounds peak memory, which grows linearly in the batch.
        self.batch_max_size: Optional[int] = None
        #: How many times a batched launch was refused/failed and fell back to
        #: the serial loop. A silent fallback is indistinguishable from a
        #: batched path that never ran, so this is counted and warned about.
        self.n_batch_fallbacks: int = 0
        #: The exception from the most recent fallback, for diagnosis.
        self.last_batch_error: Optional[BaseException] = None

    # ------------------------------------------------------------------
    # Data access (new + legacy)
    # ------------------------------------------------------------------

    @property
    def data(self) -> DomainBase:
        """The underlying :class:`DomainBase` data/residual/template signal."""
        return self._data

    @data.setter
    def data(self, value: Union[DomainBase, DataResidualArray]) -> None:
        self._data = _coerce_to_domain_base(value)

    @property
    def data_res_arr(self) -> DomainBase:
        """Legacy alias of :attr:`data`.

        Returns a :class:`DomainBase` directly (not a :class:`DataResidualArray`).
        The chain ``ac.data_res_arr.data_res_arr`` keeps working because
        :attr:`DomainBase.data_res_arr` is a self-reference.
        """
        return self._data

    @data_res_arr.setter
    def data_res_arr(self, value: Union[DomainBase, DataResidualArray]) -> None:
        self._data = _coerce_to_domain_base(value)

    @property
    def sens_mat(self) -> SensitivityMatrixBase:
        """Sensitivity information."""
        return self._sens_mat

    @sens_mat.setter
    def sens_mat(self, sens_mat: SensitivityMatrixBase) -> None:
        if not isinstance(sens_mat, SensitivityMatrixBase):
            raise TypeError(
                "sens_mat must be a SensitivityMatrixBase instance. To "
                "auto-instantiate from a class, pass it (and any "
                "``sens_mat_kwargs``) at AnalysisContainer construction."
            )
        self._sens_mat = sens_mat

    # ------------------------------------------------------------------
    # Signal generator (callable | dict[str, callable])
    # ------------------------------------------------------------------

    @property
    def signal_gen(self) -> SignalGenSpec:
        """Signal generator (callable or ``{model_name: callable}`` dict)."""
        if not hasattr(self, "_signal_gen"):
            raise ValueError("User must input signal_gen kwarg to use the signal generator.")
        return self._signal_gen

    @signal_gen.setter
    def signal_gen(self, signal_gen: SignalGenSpec) -> None:
        if isinstance(signal_gen, Mapping):
            if not signal_gen:
                raise ValueError("signal_gen dict cannot be empty.")
            for name, fn in signal_gen.items():
                if not callable(fn):
                    raise TypeError(
                        f"signal_gen[{name!r}] must be callable; got {type(fn).__name__}."
                    )
            # store a shallow copy so external mutation can't surprise us
            self._signal_gen = dict(signal_gen)
        else:
            if not callable(signal_gen):
                raise TypeError(
                    f"signal_gen must be callable or a dict of callables; got "
                    f"{type(signal_gen).__name__}."
                )
            self._signal_gen = signal_gen

    @contextmanager
    def _swapped_signal_gen(self, signal_gen: Optional[SignalGenSpec]):
        """Temporarily replace :attr:`signal_gen` for the duration of a call.

        ``None`` is a no-op (the installed generator, if any, stays in
        place). Restores the previous state on exit, including the
        "no generator installed" state.
        """
        if signal_gen is None:
            yield
            return
        prev_gen = self._signal_gen if hasattr(self, "_signal_gen") else None
        self.signal_gen = signal_gen
        try:
            yield
        finally:
            if prev_gen is None:
                del self._signal_gen
            else:
                self._signal_gen = prev_gen

    @property
    def is_multi_model(self) -> bool:
        """``True`` iff :attr:`signal_gen` is configured as a per-model dict."""
        return hasattr(self, "_signal_gen") and isinstance(self._signal_gen, Mapping)

    @property
    def model_names(self) -> List[str]:
        """List of model names when :attr:`signal_gen` is a dict (else empty list)."""
        return list(self._signal_gen.keys()) if self.is_multi_model else []

    # ------------------------------------------------------------------
    # Generator output -> data-domain DomainBase
    # ------------------------------------------------------------------

    def _to_data_domain(self, gen_out, gen_fn: Optional[Callable] = None) -> DomainBase:
        """Normalise the output of a signal generator into ``self.data.settings``.

        Accepts:

        - a :class:`DomainBase` (transformed to ``self.data.settings`` if its
          domain differs from the data's),
        - a :class:`DataResidualArray` (legacy; unwrapped then transformed),
        - a raw NumPy / CuPy / JAX array (interpreted as already living in
          ``self.data.settings``).
        """
        target = self._data.settings
        if isinstance(gen_out, DataResidualArray):
            gen_out = gen_out.data_res_arr
        if isinstance(gen_out, DomainBase):
            if gen_out.settings == target:
                return gen_out
            try:
                return gen_out.transform(target)
            except Exception as exc:
                raise ValueError(
                    f"Could not transform signal_gen output (domain "
                    f"{type(gen_out.settings).__name__}) into the data domain "
                    f"{type(target).__name__}: {exc}"
                ) from exc

        # Raw array path: trust the caller that this matches the data domain.
        try:
            arr = self._data.xp.asarray(gen_out)
        except Exception:
            arr = gen_out
        return target.associated_class(arr, target)

    # ------------------------------------------------------------------
    # Multi-model template assembly
    # ------------------------------------------------------------------

    def _call_single_model(self, fn: Callable, params, waveform_kwargs: dict) -> DomainBase:
        """Call ``fn(*params, **waveform_kwargs)`` (params may be 1D or 2D).

        For a 2D ``params`` of shape ``(n_signals, n_par)`` the call iterates
        over the rows and sums the per-row outputs (in the data domain).
        Returns a single :class:`DomainBase` in the data domain.
        """
        arr = np.asarray(params, dtype=object) if not hasattr(params, "ndim") else params
        if getattr(arr, "ndim", 1) == 1:
            return self._to_data_domain(fn(*tuple(params), **waveform_kwargs), fn)
        if arr.ndim != 2:
            raise ValueError(
                f"Per-model params must be 1D (single signal) or 2D "
                f"(batch of signals); got ndim={arr.ndim}."
            )
        accum: Optional[DomainBase] = None
        for row in arr:
            row_dom = self._to_data_domain(fn(*tuple(row), **waveform_kwargs), fn)
            if accum is None:
                accum = self._data.settings.associated_class(
                    row_dom.arr.copy(), self._data.settings,
                )
            else:
                accum.add_signal(row_dom, sign=+1)
        return accum

    def _build_batched_template(self, params, waveform_kwargs: dict):
        """Build a template stack with a leading SOURCE axis, without summing.

        This is deliberately NOT :meth:`_call_single_model`. That method also
        accepts a 2D ``params``, but it means something different: it iterates
        the rows and SUMS them into one combined template, which is the right
        answer for "several sources present in the data at once" and the wrong
        one for "several independent parameter proposals". Here every row must
        survive as its own template so the inner products can return one value
        per row.

        The generator is handed one ARRAY per parameter (a column of
        ``params``) rather than being called once per row; a generator that
        advertises ``supports_batch`` is one that accepts that.
        """
        arr = np.asarray(params)
        if arr.ndim != 2:
            raise ValueError(
                f"batched template build needs 2D params; got ndim={arr.ndim}."
            )
        cols = tuple(arr[:, k] for k in range(arr.shape[1]))
        return self._to_data_domain(
            self._signal_gen(*cols, **waveform_kwargs), self._signal_gen
        )

    def build_template(
        self,
        params: Union[tuple, list, np.ndarray, Mapping[str, Any]],
        waveform_kwargs: Union[dict, Mapping[str, dict], None] = None,
        per_model_per_signal: bool = False,
        signal_gen: Optional[SignalGenSpec] = None,
        apply_transform: Optional[bool] = None,
    ):
        """Build a combined template from ``params`` using :attr:`signal_gen`.

        - **Single-model**: ``signal_gen`` is a callable, ``params`` is a
          1D (single signal) or 2D (batch summed within model) parameter
          array. Returns a single :class:`DomainBase` template living in
          ``self.data.settings``.
        - **Multi-model**: ``signal_gen`` is a dict, ``params`` is a dict
          mapping ``model_name -> 1D or 2D params``. Per-model batch rows
          are summed (within-model), then summed across models. Returns a
          single :class:`DomainBase` (or, with ``per_model_per_signal=True``,
          a structured ``dict`` of per-model un-summed
          ``list[DomainBase]`` -- intended for diagnostics, not summed).

        Args:
            params: Parameters; structure must match :attr:`signal_gen`.
            waveform_kwargs: ``dict`` of kwargs forwarded to each generator
                (single-model), or ``{model_name: dict}`` of per-model kwargs
                (multi-model).
            per_model_per_signal: If ``True``, in the multi-model case return
                the per-model / per-signal :class:`DomainBase` list without
                summing -- handy for inspecting individual contributions.
            signal_gen: In-scope waveform generator (callable or dict).
                Replaces :attr:`signal_gen` for this call when given.
            apply_transform: Call-time transform control. When not ``None``,
                forwarded as an ``apply_transform=<bool>`` kwarg to every
                generator call. Generators following the engine convention
                (e.g. the stock ``SourceSignalGen``) apply their internal
                sampling→waveform transform by default; ``False`` tells them
                the caller already applied it (``params`` are waveform-basis),
                so the transform is applied exactly once. Leave ``None``
                (default) for generators that do not accept the kwarg.

        Returns:
            A :class:`DomainBase` template (or a structured dict when
            ``per_model_per_signal=True`` in multi-model mode).
        """
        with self._swapped_signal_gen(signal_gen):
            return self._build_template(
                params, waveform_kwargs, per_model_per_signal, apply_transform
            )

    def _build_template(
        self,
        params: Union[tuple, list, np.ndarray, Mapping[str, Any]],
        waveform_kwargs: Union[dict, Mapping[str, dict], None] = None,
        per_model_per_signal: bool = False,
        apply_transform: Optional[bool] = None,
    ):
        """:meth:`build_template` body against the currently-installed generator."""
        waveform_kwargs = waveform_kwargs or {}

        def _with_flag(kw: dict) -> dict:
            # inject the call-time transform control into a generator's kwargs
            if apply_transform is None:
                return kw
            return {**kw, "apply_transform": apply_transform}

        if not self.is_multi_model:
            if isinstance(params, Mapping):
                raise TypeError(
                    "signal_gen is a single callable, but params is a dict. "
                    "Either pass tuple/array params, or configure signal_gen "
                    "as a dict of per-model generators."
                )
            return self._call_single_model(
                self._signal_gen, params,
                _with_flag(waveform_kwargs if isinstance(waveform_kwargs, dict) else {}),
            )

        # multi-model path
        if not isinstance(params, Mapping):
            raise TypeError(
                "signal_gen is a dict of per-model generators; "
                "params must also be a dict {model_name: 1D/2D array}."
            )
        unknown = set(params) - set(self._signal_gen)
        if unknown:
            raise KeyError(
                f"Unknown model name(s) in params: {sorted(unknown)}. "
                f"Known models: {sorted(self._signal_gen)}."
            )
        if isinstance(waveform_kwargs, Mapping) and waveform_kwargs and all(
            isinstance(v, dict) for v in waveform_kwargs.values()
        ):
            per_model_kwargs = waveform_kwargs
        else:
            # treat as shared kwargs across all models
            per_model_kwargs = {name: dict(waveform_kwargs) for name in params}

        if per_model_per_signal:
            structured: Dict[str, List[DomainBase]] = {}
            for name, p in params.items():
                fn = self._signal_gen[name]
                wf = _with_flag(per_model_kwargs.get(name, {}))
                arr_like = np.asarray(p, dtype=object) if not hasattr(p, "ndim") else p
                if getattr(arr_like, "ndim", 1) == 1:
                    structured[name] = [self._to_data_domain(fn(*tuple(p), **wf), fn)]
                else:
                    structured[name] = [
                        self._to_data_domain(fn(*tuple(row), **wf), fn) for row in arr_like
                    ]
            return structured

        combined: Optional[DomainBase] = None
        for name, p in params.items():
            fn = self._signal_gen[name]
            wf = _with_flag(per_model_kwargs.get(name, {}))
            per_model = self._call_single_model(fn, p, wf)
            if combined is None:
                if len(params) == 1:
                    # single-entry dict: no cross-model summation to protect,
                    # so return the generator's template directly (matches the
                    # single-model path and avoids a full-grid copy — this is
                    # the hot path for per-branch residual/likelihood ops).
                    return per_model
                combined = self._data.settings.associated_class(
                    per_model.arr.copy(), self._data.settings,
                )
            else:
                combined.add_signal(per_model, sign=+1)
        return combined

    # ------------------------------------------------------------------
    # WDM/FD start-index passthroughs (now sourced from DomainBase)
    # ------------------------------------------------------------------

    @property
    def start_freq_ind(self):
        """Pass-through to :attr:`DomainBase.start_freq_ind`."""
        return self._data.start_freq_ind

    @property
    def start_freq_layer_ind(self):
        """Pass-through to :attr:`DomainBase.start_freq_layer_ind` (WDM only)."""
        return self._data.start_freq_layer_ind

    @property
    def start_time_layer_ind(self):
        """Pass-through to :attr:`DomainBase.start_time_layer_ind` (WDM only).

        For an :class:`AnalysisContainerArray` covering a WDM grid, every
        container shares the same active time range, so this value is the
        same across all containers.
        """
        return self._data.start_time_layer_ind

    @property
    def layer_df(self):
        """WDM layer frequency spacing (``None`` if data is not WDM)."""
        return getattr(self._data.settings, "layer_df", None)

    @property
    def layer_dt(self):
        """WDM layer time spacing (``None`` if data is not WDM)."""
        return getattr(self._data.settings, "layer_dt", None)

    # ------------------------------------------------------------------
    # Direct add/subtract of templates against the data residual
    # ------------------------------------------------------------------

    def add_signal_to_data(
        self,
        template,
        sign: int = +1,
        waveform_kwargs: Union[dict, Mapping[str, dict], None] = None,
        signal_gen: Optional[SignalGenSpec] = None,
        apply_transform: Optional[bool] = None,
    ) -> DomainBase:
        """Add ``template`` to ``self.data`` (in-place, domain-aware).

        ``template`` may be a :class:`DomainBase`, a :class:`DataResidualArray`,
        a tuple/array of parameters, or a multi-model params ``dict``. In the
        latter cases :meth:`build_template` is used to assemble the combined
        template from :attr:`signal_gen`; ``waveform_kwargs`` / ``signal_gen``
        / ``apply_transform`` are forwarded to it (call-time generator swap and
        transform control — see :meth:`build_template`).
        """
        if isinstance(template, (DomainBase, DataResidualArray)):
            tmpl = self._to_data_domain(template)
        else:
            tmpl = self.build_template(
                template,
                waveform_kwargs=waveform_kwargs,
                signal_gen=signal_gen,
                apply_transform=apply_transform,
            )
        self._data.add_signal(tmpl, sign=sign)
        return self._data

    def subtract_signal_from_data(self, template, **kwargs) -> DomainBase:
        """Subtract ``template`` from ``self.data`` (in-place, see :meth:`add_signal_to_data`)."""
        return self.add_signal_to_data(template, sign=-1, **kwargs)

    def loglog(self) -> Tuple[plt.Figure, plt.Axes]:
        """Produce loglog plot of both source and sensitivity information.

        Returns:
            Matplotlib figure and axes object in a 2-tuple.

        """
        assert isinstance(self._data.settings, domains.FDSettings)
        fig, ax = self.sens_mat.loglog(char_strain=True)
        f_arr = self._data.f_arr
        if self.sens_mat.ndim == 3:
            # 3x3 most likely
            for i in range(self.sens_mat.shape[0]):
                for j in range(i, self.sens_mat.shape[1]):
                    # char strain
                    ax[i * self.sens_mat.shape[1] + j].loglog(
                        f_arr,
                        f_arr * np.abs(self._data[i]),
                    )
                    ax[i * self.sens_mat.shape[1] + j].loglog(
                        f_arr,
                        f_arr * np.abs(self._data[j]),
                    )
        else:
            for i in range(self.sens_mat.shape[0]):
                ax[i].loglog(
                    f_arr,
                    f_arr * np.abs(self._data[i]),
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

        return inner_product(self._data, self._data, psd=self.sens_mat, **kwargs)

    def snr(self, **kwargs: dict) -> float:
        """Return the SNR of the current set of information

        Args:
            **kwargs: Inner product keyword arguments.

        Returns:
            SNR value.

        """
        return self.inner_product(**kwargs).real ** (1 / 2)

    def _slice_to_template(
        self, template: Union[DomainBase, DataResidualArray]
    ) -> Tuple[DomainBase, DomainBase, SensitivityMatrixBase]:
        """Slice the data to the same shape as the template.

        This is used for calculating inner products and likelihoods with
        templates that are shorter than the data.

        Args:
            template: Template signal (``DomainBase`` or legacy ``DataResidualArray``).
        """
        template = _coerce_to_domain_base(template)
        data_settings = self._data.settings
        templ_settings = template.settings

        if type(data_settings) is not type(templ_settings):
            raise ValueError(
                f"Data domain ({type(data_settings).__name__}) and template domain "
                f"({type(templ_settings).__name__}) must match."
            )

        # Fast path: settings identical → no slicing needed
        if data_settings == templ_settings:
            return self._data, template, self.sens_mat

        elif isinstance(data_settings, domains.STFTSettings):
            return self._slice_stft_to_template(template)
        else:
            raise NotImplementedError(
                f"Automatic region slicing not yet implemented for "
                f"{type(data_settings).__name__}. Ensure template and data "
                f"have the same shape, or use STFT domain."
            )

    def _slice_stft_to_template(
        self, template: DomainBase
    ) -> Tuple[DomainBase, DomainBase, SensitivityMatrixBase]:
        """STFT-specific slice helper used by :meth:`_slice_to_template`.

        Args:
            template: Template signal (already coerced to a :class:`DomainBase`).

        Returns:
            ``(sliced_data, sliced_template, sliced_sens_mat)`` as
            :class:`DomainBase` / :class:`DomainBase` / :class:`SensitivityMatrixBase`.
        """
        data_settings = self._data.settings
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

        templ_tmin, templ_tmax = (
            templ_settings.t0,
            templ_settings.t0 + templ_settings.NT * templ_settings.dt,
        )
        data_tmin, data_tmax = (
            data_settings.t0,
            data_settings.t0 + data_settings.NT * data_settings.dt,
        )
        tmin = max(templ_tmin, data_tmin)
        tmax = min(templ_tmax, data_tmax)

        fmin, fmax = templ_settings.f_arr[0], templ_settings.f_arr[-1]
        slices = data_settings.compute_slice_indices(tmin, tmax, fmin, fmax)
        sliced_data = self._data.get_array_slice(slices)
        sliced_sens_mat = self.sens_mat.get_slice(slices)

        templ_slice = templ_settings.compute_slice_indices(tmin, tmax, fmin, fmax)
        sliced_template = template.get_array_slice(templ_slice)

        return sliced_data, sliced_template, sliced_sens_mat

    def template_inner_product(
        self, template: Union[DomainBase, DataResidualArray], **kwargs: dict
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

        data_res_arr_sliced, template_sliced, sens_mat_sliced = self._slice_to_template(template)

        ip_val = inner_product(data_res_arr_sliced, template_sliced, psd=sens_mat_sliced, **kwargs)
        return ip_val

    def template_snr(
        self, template: Union[DomainBase, DataResidualArray], phase_maximize: bool = False, **kwargs: dict
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

        sliced_data_res_arr, sliced_template, sliced_sens_mat = self._slice_to_template(template)

        # TODO: should we cache?
        h_h = inner_product(sliced_template, sliced_template, psd=sliced_sens_mat, **kwargs_in)
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
        template: Union[DomainBase, DataResidualArray],
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

        data_res_arr_sliced, template_sliced, sens_mat_sliced = self._slice_to_template(template)

        # when computing the <d|d> term we need the full data and sensitivity matrix.

        # TODO: should we cache?
        d_d = inner_product(self._data, self._data, psd=self.sens_mat, **kwargs_in)
        h_h = inner_product(template_sliced, template_sliced, psd=sens_mat_sliced, **kwargs_in)

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
        # Stash the individual terms AS USED in the likelihood below (post
        # phase/amp maximization) -- readable after the call like
        # ``non_marg_d_h`` above, e.g. for per-source inner-product recording.
        self.last_d_h = d_h
        self.last_h_h = h_h
        self.last_d_d = d_d
        # breakpoint()
        like_out = -1 / 2 * (d_d + h_h - 2 * d_h).real

        if include_psd_info and not self.likelihood_source_only:
            # Add the noise normalization term. Skipped on source-only /
            # fixed-PSD containers (``likelihood_source_only=True``): there the
            # noise term is an overall constant that cancels in every
            # Likelihood difference, so dropping it keeps the readout as the
            # pure ``-1/2 <r|r>`` source Likelihood. ``source_only=False`` is
            # explicit so the noise_only call never collides with the
            # container-level source-only default.
            like_out += self.likelihood(noise_only=True, source_only=False)

        return like_out

    def likelihood(
        self, source_only: Optional[bool] = None, noise_only: bool = False, **kwargs: dict
    ) -> float | complex:
        """Return the likelihood of the current arangement.

        Args:
            source_only: If ``True`` return the source-only Likelihood
                (``-1/2 <r|r>``, no noise normalization term). ``None``
                (default) falls back to the container-level
                :attr:`likelihood_source_only` setting.
            noise_only: If ``True``, return the noise part of the Likelihood alone.
            **kwargs: Keyword arguments to pass to :func:`lisatools.diagnostic.inner_product`.

        Returns:
            Likelihood value.

        """
        source_only_explicit = source_only is not None
        if source_only is None:
            source_only = getattr(self, "likelihood_source_only", False)
        if noise_only and source_only:
            if source_only_explicit:
                raise ValueError(
                    "likelihood(noise_only=True, source_only=True) is "
                    "contradictory: the noise-only and source-only terms are "
                    "mutually exclusive."
                )
            raise ValueError(
                "likelihood(noise_only=True) was called on a container built "
                "with likelihood_source_only=True, so source_only defaulted to "
                "True and collides with noise_only. Pass source_only=False "
                "explicitly for the noise term, or don't request the noise "
                "term on a source-only (fixed-PSD) container."
            )
        elif noise_only:
            return noise_likelihood_term(self.sens_mat)
        elif source_only:
            return residual_source_likelihood_term(self._data, psd=self.sens_mat, **kwargs)
        else:
            return residual_full_source_and_noise_likelihood(
                self._data, self.sens_mat, **kwargs
            )

    # TODO: make sure there is a way for backends to check TDI channel structure/domain is equivalent

    def _calculate_signal_operation(
        self,
        calc: str,
        *args: Any,
        source_only: Optional[bool] = None,
        waveform_kwargs: Optional[Union[dict, Mapping[str, dict]]] = None,
        transform_fn: Optional[TransformContainer] = None,
        signal_gen: Optional[SignalGenSpec] = None,
        per_model_per_signal: bool = False,
        apply_transform: Optional[bool] = None,
        **kwargs: dict,
    ) -> float | complex:
        """Build a template from ``signal_gen`` and run a likelihood/SNR/inner-product op.

        Args:
            calc: One of ``"likelihood"``, ``"inner_product"``, ``"snr"``.
            *args: Parameter input. For the **single-model** case (signal_gen
                is a callable), this is the waveform's positional parameter
                list (1D), or a single 2D ndarray to batch-sum within model.
                For the **multi-model** case (signal_gen is a dict), pass
                exactly one positional argument: a ``dict`` mapping
                ``model_name -> 1D/2D params``.
            source_only: If ``True`` return the source-only Likelihood
                (leave out noise part). ``None`` (default) falls back to the
                container-level :attr:`likelihood_source_only` setting.
            waveform_kwargs: Keyword arguments forwarded to the generator(s).
                In multi-model mode this can also be a
                ``{model_name: dict}`` mapping.
            transform_fn: Optional :class:`~eryn.utils.TransformContainer`
                applied to the parameters before generation (single-model
                only).
            signal_gen: In-scope waveform generator (callable or dict).
                Replaces :attr:`signal_gen` for this call when given.
            per_model_per_signal: Multi-model only. If ``True``, skip the
                summation step and return a structured per-model /
                per-signal dict of results instead of a scalar.
            apply_transform: Call-time transform control forwarded to the
                generator calls (see :meth:`build_template`). ``False`` marks
                the parameters as already waveform-basis so an
                engine-convention generator skips its internal transform;
                ``None`` (default) injects nothing. Distinct from
                ``transform_fn``, which transforms the parameters *here*
                before generation.
            **kwargs: Forwarded to :func:`lisatools.diagnostic.inner_product`.

        Returns:
            Likelihood / inner-product / SNR value, or a structured dict
            (per_model_per_signal mode).
        """
        if source_only is None:
            source_only = getattr(self, "likelihood_source_only", False)

        # Temporarily swap in a per-call signal_gen if provided.
        with self._swapped_signal_gen(signal_gen):
            multi = self.is_multi_model

            if multi:
                if len(args) != 1 or not isinstance(args[0], Mapping):
                    raise TypeError(
                        "signal_gen is a dict; pass a single dict of per-model "
                        "params as the only positional argument."
                    )
                if transform_fn is not None:
                    raise NotImplementedError(
                        "transform_fn is not supported in multi-model mode; "
                        "apply parameter transforms inside each model's "
                        "signal_gen callable."
                    )
                params = args[0]
                template_or_struct = self._build_template(
                    params,
                    waveform_kwargs=waveform_kwargs or {},
                    per_model_per_signal=per_model_per_signal,
                    apply_transform=apply_transform,
                )

                if per_model_per_signal:
                    # Return per-model / per-signal results, evaluated by ``calc``.
                    return self._evaluate_structured_templates(
                        calc,
                        template_or_struct,
                        source_only=source_only,
                        **kwargs,
                    )
                template = template_or_struct
            else:
                if transform_fn is not None:
                    args_tmp = np.asarray(args)
                    args_in = tuple(transform_fn.both_transforms(args_tmp))
                else:
                    args_in = args
                template = self._build_template(
                    args_in,
                    waveform_kwargs=waveform_kwargs or {},
                    apply_transform=apply_transform,
                )

        if "include_psd_info" in kwargs:
            assert kwargs["include_psd_info"] == (not source_only)
            kwargs.pop("include_psd_info")

        kwargs = dict(psd=self.sens_mat, **kwargs)

        if calc == "likelihood":
            kwargs["include_psd_info"] = not source_only
            return self.template_likelihood(template, **kwargs)
        elif calc == "inner_product":
            return self.template_inner_product(template, **kwargs)
        elif calc == "snr":
            return self.template_snr(template, **kwargs)
        else:
            raise ValueError("`calc` must be 'likelihood', 'inner_product', or 'snr'.")

    def _evaluate_structured_templates(
        self,
        calc: str,
        structured: Dict[str, List[DomainBase]],
        source_only: bool = False,
        **kwargs: dict,
    ) -> Dict[str, List[Any]]:
        """Run ``calc`` per-model / per-signal on a structured template dict.

        Used by :meth:`_calculate_signal_operation` when
        ``per_model_per_signal=True``. Each individual template is evaluated
        against the data; the dict preserves the input shape so callers can
        inspect per-source contributions.
        """
        if "include_psd_info" in kwargs:
            kwargs.pop("include_psd_info")
        op_kwargs = dict(psd=self.sens_mat, **kwargs)
        if calc == "likelihood":
            op_kwargs["include_psd_info"] = not source_only
            op = self.template_likelihood
        elif calc == "inner_product":
            op = self.template_inner_product
        elif calc == "snr":
            op = self.template_snr
        else:
            raise ValueError("`calc` must be 'likelihood', 'inner_product', or 'snr'.")

        return {
            name: [op(tmpl, **op_kwargs) for tmpl in tmpl_list]
            for name, tmpl_list in structured.items()
        }

    def calculate_signal_likelihood(
        self,
        *args: Any,
        source_only: Optional[bool] = None,
        waveform_kwargs: Optional[Union[dict, Mapping[str, dict]]] = None,
        per_model_per_signal: bool = False,
        **kwargs: dict,
    ) -> Union[float, complex, Dict[str, List[Any]]]:
        """Return the likelihood of a generator-produced signal against the data.

        Single-model: ``*args`` are the waveform parameters (1D), or a single
        2D ndarray batch (summed within model).
        Multi-model (``signal_gen`` is a dict): pass exactly one positional
        argument, a ``dict`` of ``{model_name: 1D or 2D params}``.
        ``per_model_per_signal=True`` (multi-model) returns the un-summed
        per-source results instead of the combined-template scalar.
        ``source_only=None`` falls back to the container-level
        :attr:`likelihood_source_only` setting.
        """

        return self._calculate_signal_operation(
            "likelihood",
            *args,
            source_only=source_only,
            waveform_kwargs=waveform_kwargs,
            per_model_per_signal=per_model_per_signal,
            **kwargs,
        )

    def calculate_signal_inner_product(
        self,
        *args: Any,
        source_only: bool = False,
        waveform_kwargs: Optional[Union[dict, Mapping[str, dict]]] = None,
        per_model_per_signal: bool = False,
        **kwargs: dict,
    ) -> Union[float, complex, Dict[str, List[Any]]]:
        """Return the inner product of a generator-produced signal against the data.

        See :meth:`calculate_signal_likelihood` for the multi-model API and
        ``per_model_per_signal`` flag.
        """

        return self._calculate_signal_operation(
            "inner_product",
            *args,
            source_only=source_only,
            waveform_kwargs=waveform_kwargs,
            per_model_per_signal=per_model_per_signal,
            **kwargs,
        )

    def calculate_signal_snr(
        self,
        *args: Any,
        source_only: bool = False,
        waveform_kwargs: Optional[Union[dict, Mapping[str, dict]]] = None,
        per_model_per_signal: bool = False,
        **kwargs: dict,
    ) -> Union[Tuple[float, float], Dict[str, List[Tuple[float, float]]]]:
        """Return the SNR (optimal, detected) of a generator-produced signal.

        See :meth:`calculate_signal_likelihood` for the multi-model API and
        ``per_model_per_signal`` flag.
        """

        return self._calculate_signal_operation(
            "snr",
            *args,
            source_only=source_only,
            waveform_kwargs=waveform_kwargs,
            per_model_per_signal=per_model_per_signal,
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
                likelihood_out[i] = self.calculate_signal_likelihood(*input_vals, **kwargs)
            return likelihood_out

        else:
            raise ValueError("x must be a 1D or 2D array.")

    def _batched_likelihood(self, x, *, source_only: bool = False, **kwargs):
        """One generator launch for a whole 2D parameter block.

        Returns one log-likelihood per ROW. The per-row split is not done
        here: :func:`~lisatools.diagnostic.inner_product` keeps a leading
        source axis and reduces only the basis axes, so a batched template
        against the unbatched data yields a vector of inner products, and
        :meth:`template_likelihood` composes exactly those.

        ``batch_max_size`` chunks the launch. Memory grows linearly in the
        batch, so a walker cloud large enough to be worth batching is also
        large enough to run a device out of memory in one go.
        """
        arr = np.asarray(x)
        n = arr.shape[0]
        step = int(self.batch_max_size or n)
        if step < 1:
            raise ValueError(f"batch_max_size must be >= 1; got {self.batch_max_size!r}")

        waveform_kwargs = kwargs.pop("waveform_kwargs", None) or {}
        pieces = []
        for lo in range(0, n, step):
            template = self._build_batched_template(arr[lo:lo + step], waveform_kwargs)
            # MIRROR THE SERIAL ROUTE EXACTLY. ``template_likelihood`` has no
            # ``source_only`` parameter -- it takes ``include_psd_info``, and
            # forwards anything else straight to ``inner_product``, whose
            # keyword list is explicit. Passing ``source_only`` here raised
            # TypeError inside inner_product on EVERY call, which the
            # fallback below then swallowed, so the batched path silently
            # never ran. See _calculate_signal_operation, which derives the
            # flag the same way.
            vals = self.template_likelihood(
                template, include_psd_info=not source_only, **kwargs
            )
            pieces.append(np.atleast_1d(asnumpy(vals)).astype(float).ravel())

        out = np.concatenate(pieces) if len(pieces) > 1 else pieces[0]
        if out.shape != (n,):
            # The generator accepted the batch but did not give one value per
            # row. Returning this would silently misalign likelihoods with
            # walkers, so refuse and let the caller fall back to the loop.
            raise ValueError(
                f"batched likelihood returned shape {out.shape}, expected "
                f"({n},). The generator declares supports_batch but did not "
                f"preserve one template per parameter row."
            )
        return out

    def eryn_likelihood_wrap(
        self,
        x: np.ndarray | list | tuple,
        *args: Any,
        use_vmap: bool = False,
        source_only: bool = False,
        **kwargs: Any,
    ) -> np.ndarray | float:
        """Vectorized Eryn-compatible log-likelihood.

        Eryn's :class:`~eryn.ensemble.EnsembleSampler` calls its
        ``log_like_fn`` with all walkers at once when
        ``vectorize=True``. This method is the batched analog of
        :meth:`eryn_likelihood_function`: it accepts ``x`` of shape
        ``(nwalkers, n_params)`` (or ``(n_params,)`` for a single
        walker) and returns a vector of log-likelihoods.

        Two batching paths are supported:

        * ``use_vmap=True`` -- expects ``self.signal_gen`` to be a
          jax-traceable function returning a JAX array; the whole
          per-walker pipeline (signal generation +
          :meth:`template_likelihood`) is then batched through
          :func:`jax.vmap`. This is what you want for the JAX
          TDI-on-the-fly GB likelihood.
        * ``use_vmap=False`` (default) -- a Python ``for`` loop over
          the walker axis using :meth:`calculate_signal_likelihood`
          per walker (same code path as
          :meth:`eryn_likelihood_function`, just exposed under the
          name Eryn picks up via ``hasattr``).

        Args:
            x: Parameters. ``(n_params,)`` for a single walker or
                ``(nwalkers, n_params)`` for a batch.
            *args: Extra positional args forwarded to ``signal_gen``.
            use_vmap: Vectorize via :func:`jax.vmap`. Requires
                ``signal_gen`` to be jax-traceable and to return a JAX
                array of shape ``(nchannels, basis_length)``.
            source_only: Forwarded to :meth:`template_likelihood`.
            **kwargs: Forwarded to :meth:`template_likelihood`.

        Returns:
            Scalar (1D input) or ``(nwalkers,)`` array (2D input) of
            log-likelihood values.
        """
        assert self.signal_gen is not None

        if isinstance(x, (list, tuple)):
            x = np.asarray(x)

        if not hasattr(x, "ndim"):
            x = np.asarray(x)

        if x.ndim == 1:
            input_vals = tuple(x) + tuple(args)
            return self.calculate_signal_likelihood(
                *input_vals, source_only=source_only, **kwargs
            )

        if x.ndim != 2:
            raise ValueError("x must be a 1D or 2D array.")

        def _serial_loop():
            out = np.zeros(x.shape[0])
            for i in range(x.shape[0]):
                input_vals = tuple(x[i]) + tuple(args)
                out[i] = self.calculate_signal_likelihood(
                    *input_vals, source_only=source_only, **kwargs
                )
            return out

        if not use_vmap:
            # RESPONSE-LEVEL BATCH. Opt-in and default-deny: a generator that
            # does not declare ``supports_batch`` -- including one with no
            # common base class, which several sources here have -- is
            # excluded automatically and takes the loop below, byte for byte
            # as before.
            can_batch = (
                getattr(self, "batch_evaluation", False)
                and bool(getattr(self._signal_gen, "supports_batch", False))
                and not self.is_multi_model
                and x.shape[0] > 1
                and not args
                # Container-level keywords are consumed by
                # ``_calculate_signal_operation`` on the serial route, NOT by
                # ``template_likelihood``. The batched builder cannot express
                # them: ``transform_fn`` must be applied before generation,
                # ``apply_transform`` injected into the generator's kwargs,
                # and ``signal_gen`` installed via ``_swapped_signal_gen`` --
                # which _build_batched_template does not enter, so the gate
                # would also be reading a DIFFERENT generator from the one
                # building the template. Decline the batch and let the serial
                # route handle them correctly rather than half-honouring them.
                and not (_CONTAINER_LEVEL_KWARGS & kwargs.keys())
            )
            if can_batch:
                try:
                    return self._batched_likelihood(
                        x, source_only=source_only, **kwargs
                    )
                except TypeError as exc:
                    # A refusal is never a TypeError; this means THIS call
                    # site is wrong (a keyword the callee does not take), and
                    # swallowing it is what let a permanently-broken batched
                    # path look like a working one.
                    #
                    # Deliberately NOT AttributeError: a generator whose
                    # __call__ returns something the container cannot consume
                    # raises that, and falling back is the right answer there
                    # rather than killing the sampler.
                    raise RuntimeError(
                        f"batched likelihood is mis-wired, not refused: "
                        f"{type(exc).__name__}: {exc}. This is a bug in "
                        f"AnalysisContainer._batched_likelihood, not a "
                        f"property of the generator or the proposal."
                    ) from exc
                except Exception as exc:
                    # A REFUSAL IS NOT AN ERROR. A generator that reports
                    # "this batch cannot be launched as one" (merger times
                    # spread wider than the shared evaluation window, ragged
                    # node counts) is exercising designed control flow, and
                    # the serial loop is the correct answer.
                    #
                    # LOUD on purpose. A silent fallback is indistinguishable
                    # from a working batch that happens to be slow -- which is
                    # exactly how a broken batched path can look "identical to
                    # serial" while never having run at all.
                    self.n_batch_fallbacks += 1
                    self.last_batch_error = exc
                    warnings.warn(
                        f"batched likelihood failed and fell back to the "
                        f"serial loop ({type(exc).__name__}: {exc}). Results "
                        f"are still correct. Watch n_batch_fallbacks: if it "
                        f"keeps rising once the chain has settled, the batch "
                        f"geometry never becomes launchable and "
                        f"batch_evaluation is costing you the failed attempt "
                        f"each time; set batch_evaluation=False to stop "
                        f"trying.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
            return _serial_loop()

        # vmap path -- pull jax lazily so the import isn't required
        # when use_vmap=False.
        try:
            import jax
            import jax.numpy as jnp
        except (ImportError, ModuleNotFoundError) as e:
            raise RuntimeError(
                "use_vmap=True requires `jax` to be installed."
            ) from e

        basis_settings = self._data.settings
        x_jnp = jnp.asarray(x)

        # ``source_only`` is implemented through ``calculate_signal_likelihood``
        # (which dispatches to the right code path) and isn't a kwarg
        # of ``template_likelihood``. The vmap path uses
        # ``template_likelihood`` directly, so we translate the flag
        # via ``include_psd_info`` (the complement).
        template_likelihood_kwargs = dict(kwargs)
        if not source_only:
            template_likelihood_kwargs.setdefault("include_psd_info", False)

        if self.is_multi_model:
            raise NotImplementedError(
                "use_vmap=True is not currently supported with a multi-model "
                "signal_gen dict; build the combined template yourself."
            )

        def _ll_single(theta):
            template_arr = self._signal_gen(*theta, *args)
            template = basis_settings.associated_class(template_arr, basis_settings)
            return self.template_likelihood(
                template, **template_likelihood_kwargs,
            )

        return jax.vmap(_ll_single)(x_jnp)


def shard_lookup_maps(aca) -> tuple:
    """Cached ``(split_map_host, global_to_intra)`` lookup for a sharded ACA.

    Both arrays are host (numpy) int arrays of length ``acs_total_entries``:
    ``split_map_host[row]`` is the owning split and
    ``global_to_intra[row]`` the row's intra-shard position (the ``i`` such
    that ``gpu_splits[split][i] == row``). They are pure functions of the
    STATIC shard layout (``gpu_splits`` / ``split_map`` are assigned once in
    ``AnalysisContainerArray.__init__`` and never rebound — fixed-capacity
    ``resize_to`` re-binds cells, not shards), so they are computed once and
    cached on the ACA (``_shard_lookup_cache``). Rebuilding them per call
    was an O(capacity) host loop on EVERY routed engine call / BandView
    fancy index (production profiling, 2026-08). The cache key
    ``(n_rows, n_splits)`` guards the only ways the layout could legally
    look different (a different/duck-typed holder object reusing the
    attribute is keyed out by living on the instance itself).

    Works on any holder exposing ``acs_total_entries`` / ``split_map`` /
    ``gpu_splits`` (the multi-shard test fakes included). Callers must not
    mutate the returned arrays.
    """
    n = int(aca.acs_total_entries)
    n_splits = len(aca.gpu_splits)
    cache = getattr(aca, "_shard_lookup_cache", None)
    if cache is not None and cache[0] == n and cache[1] == n_splits:
        return cache[2], cache[3]
    split_map_host = np.asarray(asnumpy(aca.split_map), dtype=int)
    global_to_intra = np.empty(n, dtype=int)
    for rows in aca.gpu_splits:
        rr = np.asarray(asnumpy(rows), dtype=int)
        global_to_intra[rr] = np.arange(rr.shape[0])
    try:
        aca._shard_lookup_cache = (n, n_splits, split_map_host, global_to_intra)
    except AttributeError:
        pass  # slots/immutable holders just skip the cache
    return split_map_host, global_to_intra


class BandView:
    """Multi-shard view of a per-band buffer indexed by global band id.

    Drop-in replacement for the single-ndarray ``data_shaped[0]`` /
    ``psd_shaped[0]`` view used by the GB band-tree Buffer when the
    underlying :class:`AnalysisContainerArray` is sharded across multiple
    GPUs. Reads and writes by global band index are routed through
    ``aca.gpu_map`` to the owning shard at the right intra-shard
    position.

    Three operation classes are supported (these cover every Buffer
    callsite in ``gbspecialstretch.py`` today):

    1. **Per-band write** (``view[band_inds] = value``): partitions
       ``band_inds`` by ``aca.split_map``, enters each owning device
       context once, and writes either a scalar (broadcast) or
       per-band slabs ``value[k]`` into shard rows.

    2. **Per-band read** (``view[band_inds]``): inverse of (1); gathers
       the rows from each owning shard into a single contiguous
       ndarray on ``aca.gpus[0]`` (or the host when ``aca.gpus`` is
       None). Used by callers that pass the result to further compute
       (einsum, sum, etc.).

    3. **Whole-buffer gather** (``view.copy()`` / ``view.gather()``):
       same as ``view[arange(num_bands)]`` -- a single contiguous
       ``(num_bands, *per_band_shape)`` ndarray on ``aca.gpus[0]``.
       Used by the Buffer's ``likelihood()`` einsum block where the
       gathered copy is consumed by a single-GPU kernel.

    The view does NOT support arithmetic in place; callers that need to
    do math on the gathered buffer should call ``.copy()`` (or index by
    the full band range) to materialise a single ndarray first.
    """

    def __init__(
        self,
        aca: "AnalysisContainerArray",
        kind: str = "data",
        n_bands: Optional[int] = None,
    ):
        if kind not in ("data", "psd"):
            raise ValueError(f"kind must be 'data' or 'psd'; got {kind!r}")
        self._aca = aca
        self._kind = kind
        # Fixed-capacity front view (GB RJ SubBandBuffer, user ruling
        # 2026-08-14): when the backing ACA is allocated at capacity but only
        # the first ``n_bands`` slots are BOUND, the view exposes exactly
        # those slots — ``shape``/``len`` report ``n_bands`` and each shard
        # is front-sliced to its bound rows. ``gpu_splits`` entries are
        # sorted ascending, so a shard's bound rows (global ids < n_bands)
        # occupy exactly its leading intra-shard positions — a leading-axis
        # slice per shard. ``None`` (default) keeps today's full-ACA
        # behavior verbatim.
        if n_bands is not None:
            n_bands = int(n_bands)
            if not (0 < n_bands <= int(aca.acs_total_entries)):
                raise ValueError(
                    f"BandView n_bands={n_bands} outside (0, "
                    f"{int(aca.acs_total_entries)}]"
                )
        self._n_bands = n_bands

    @property
    def _num_bands_bound(self) -> int:
        """Bound band count: the front-view limit, or every ACA slot."""
        if self._n_bands is not None:
            return self._n_bands
        return int(self._aca.acs_total_entries)

    @property
    def _shards(self):
        shards = (
            self._aca.data_shaped if self._kind == "data" else self._aca.psd_shaped
        )
        if self._n_bands is None:
            return shards
        # Front-slice each shard to its bound rows (gpu_splits sorted =>
        # global ids < n_bands sit at intra positions 0..n_s-1). The
        # counts are cached on the ACA per n_bands (gpu_splits is static;
        # views are transient, so a view-local cache would never hit) —
        # this property is evaluated inside hot fill loops.
        counts = self._bound_shard_counts()
        return [sh[:n_s] for sh, n_s in zip(shards, counts)]

    def _bound_shard_counts(self) -> tuple:
        """Per-shard bound-row counts for the fixed-capacity front view.

        Cached on the ACA keyed by ``n_bands`` (``gpu_splits`` is assigned
        once at ACA construction and never rebound; ``resize_to`` changes
        the BOUND count — a new key — not the shard layout)."""
        aca = self._aca
        nb = int(self._n_bands)
        cache = getattr(aca, "_bandview_bound_counts", None)
        if cache is None:
            cache = {}
            try:
                aca._bandview_bound_counts = cache
            except AttributeError:
                cache = None
        if cache is not None and nb in cache:
            return cache[nb]
        counts = tuple(
            int(np.searchsorted(np.asarray(asnumpy(split)), nb))
            for split in aca.gpu_splits
        )
        if cache is not None:
            cache[nb] = counts
        return counts

    @property
    def shape(self) -> tuple:
        shards = (
            self._aca.data_shaped if self._kind == "data" else self._aca.psd_shaped
        )
        per_band = tuple(shards[0].shape[1:])
        return (self._num_bands_bound,) + per_band

    @property
    def dtype(self):
        shards = (
            self._aca.data_shaped if self._kind == "data" else self._aca.psd_shaped
        )
        return shards[0].dtype

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def __len__(self) -> int:
        return self._num_bands_bound

    def _resolve_scalar(self, band: int) -> tuple:
        """``(shard_id, intra_shard_index)`` for a single band id."""
        shard = int(self._aca.split_map[int(band)])
        intra = int(np.where(self._aca.gpu_splits[shard] == int(band))[0][0])
        return shard, intra

    def _resolve_array(self, band_inds) -> tuple:
        """Per-row ``(shard_ids, intra_indices)`` arrays for a 1-D band index."""
        band_inds = np.asarray(asnumpy(band_inds), dtype=int)
        # Cached static lookup (perf, 2026-08): the per-call
        # argsort/searchsorted of the static gpu_splits is replaced by two
        # O(1) table gathers (see shard_lookup_maps).
        split_map_host, global_to_intra = shard_lookup_maps(self._aca)
        shard_ids = split_map_host[band_inds]
        intra = global_to_intra[band_inds]
        return shard_ids, intra, band_inds

    def __getitem__(self, idx):
        # Tuple fancy index -> route to _fancy_get (band axis is idx[0]).
        if isinstance(idx, tuple):
            if len(idx) == 0:
                return self.gather()
            return self._fancy_get(idx[0], idx[1:])
        # Scalar -> shard slice (live view, no copy)
        if isinstance(idx, (int, np.integer)):
            shard, intra = self._resolve_scalar(int(idx))
            return self._shards[shard][intra]
        if isinstance(idx, slice):
            band_inds = np.arange(*idx.indices(self.shape[0]))
            return self._gather(band_inds)
        # Array idx (np or cp)
        band_inds = np.asarray(asnumpy(idx))
        if band_inds.ndim == 0:
            shard, intra = self._resolve_scalar(int(band_inds))
            return self._shards[shard][intra]
        if band_inds.dtype == bool:
            # Boolean mask over (num_bands,). Convert to int positions.
            if band_inds.shape[0] != self.shape[0]:
                raise IndexError(
                    f"boolean index shape {band_inds.shape} mismatches "
                    f"num_bands={self.shape[0]}"
                )
            band_inds = np.where(band_inds)[0]
        return self._gather(band_inds)

    def __setitem__(self, idx, val):
        if isinstance(idx, tuple):
            if len(idx) == 0:
                raise IndexError("BandView setitem with empty tuple")
            self._fancy_set(idx[0], idx[1:], val)
            return
        if isinstance(idx, (int, np.integer)):
            shard, intra = self._resolve_scalar(int(idx))
            self._write_to_shard(shard, intra, val)
            return
        if isinstance(idx, slice):
            band_inds = np.arange(*idx.indices(self.shape[0]))
            self._scatter(band_inds, val)
            return
        band_inds = np.asarray(asnumpy(idx))
        if band_inds.ndim == 0:
            shard, intra = self._resolve_scalar(int(band_inds))
            self._write_to_shard(shard, intra, val)
            return
        if band_inds.dtype == bool:
            if band_inds.shape[0] != self.shape[0]:
                raise IndexError(
                    f"boolean index shape {band_inds.shape} mismatches "
                    f"num_bands={self.shape[0]}"
                )
            band_inds = np.where(band_inds)[0]
        self._scatter(band_inds, val)

    def _fancy_get(self, band_idx, intra_axes: tuple):
        """Tuple-fancy ``view[band_idx, intra1, intra2, ...]`` read.

        ``band_idx`` indexes axis 0 (band id, global). ``intra_axes`` index
        the remaining per-band axes (channel, freq, etc.). All entries are
        broadcastable to a common shape per NumPy's fancy-indexing rules.

        The result is built by:

        1. broadcasting all index arrays to a common shape,
        2. partitioning the flat result rows by ``aca.split_map[band_id]``,
        3. running a per-shard fancy index ``shard[(intra_band, *intra_axes)]``
           against the shard's own ``(n_acs_on_shard, *per_band_shape)``
           reshape view,
        4. scattering each shard's contribution into the output (gathered
           onto ``gpus[0]`` for multi-GPU runs).

        Used by ``Buffer.fill_buffer_residual_and_psd_from_acs`` and other
        callers that previously had to ``gather_*_shaped()`` the whole
        outer ACA before fancy-indexing it.
        """
        return self._fancy_dispatch(band_idx, intra_axes, mode="get", val=None)

    def _fancy_set(self, band_idx, intra_axes: tuple, val) -> None:
        """Tuple-fancy ``view[band_idx, intra1, ...] = val`` write."""
        self._fancy_dispatch(band_idx, intra_axes, mode="set", val=val)

    def _fancy_dispatch(self, band_idx, intra_axes: tuple, mode: str, val):
        """Shared entry for tuple-fancy read (mode='get') / write (mode='set').

        Fast row-wise path (perf rework, 2026-08): when the band index
        varies only along axis 0 of the broadcast result (the shape every
        Buffer fill call site produces — ``band[:, None, ...]`` against
        open per-axis ramps), the flat rows partition by band, so each
        shard is fancy-indexed DEVICE-SIDE with the original (un-broadcast)
        index arrays and broadcasting left to the backend. The previous
        implementation ``asnumpy``'d every index array and broadcast it to
        the FULL target shape then ``reshape(-1)`` — hundreds of MB of host
        int64 per fill chunk, the fills' whole measured cost. Exotic
        patterns (band varying along a non-leading axis, scalar band)
        fall back to the bit-equivalent legacy broadcast path.
        """
        shapes = [np.shape(band_idx)] + [np.shape(a) for a in intra_axes]
        target_shape = np.broadcast_shapes(*shapes)
        band_shape = shapes[0]
        fast_ok = (
            len(target_shape) > 0
            and len(band_shape) == len(target_shape)
            and all(s == 1 for s in band_shape[1:])
        )
        if fast_ok:
            return self._fancy_rowwise(
                band_idx, intra_axes, target_shape, mode, val
            )
        return self._fancy_broadcast(band_idx, intra_axes, target_shape, mode, val)

    def _slice_axis0_on_device(self, arr, rows: np.ndarray):
        """``arr[rows]`` executed under ``arr``'s own device (numpy: plain)."""
        dev = getattr(getattr(arr, "device", None), "id", None)
        if dev is None:
            return arr[rows]
        with self._aca.xp.cuda.Device(int(dev)):
            return arr[rows]

    def _fancy_rowwise(self, band_idx, intra_axes, target_shape, mode, val):
        """Row-partitioned fancy dispatch (see :meth:`_fancy_dispatch`).

        Per shard: slice each index array along axis 0 only where it
        actually varies there, move the (small) slices onto the owning
        device, and let the backend's fancy indexing do the broadcasting
        on device. Bit-identical to the broadcast path — the same
        (band, *intra) coordinate reads/writes the same shard element.
        """
        xp = self._aca.xp
        nd = len(target_shape)
        n = int(target_shape[0])
        # Band ids per output row (host; n entries — small by construction).
        bands = np.asarray(asnumpy(band_idx)).reshape(-1).astype(int)
        if bands.shape[0] == 1 and n > 1:
            bands = np.broadcast_to(bands, (n,))
        split_map_host, global_to_intra = shard_lookup_maps(self._aca)
        shard_ids = split_map_host[bands]
        intra_all = global_to_intra[bands]
        shards = self._shards

        is_scalar_val = mode == "set" and (
            np.isscalar(val) or (hasattr(val, "ndim") and val.ndim == 0)
        )
        if mode == "set" and not is_scalar_val and not hasattr(val, "ndim"):
            val = np.asarray(val)  # list/tuple payloads slice like arrays
        val_shape = None if (mode != "set" or is_scalar_val) else np.shape(val)

        def _axis0_varies(shape) -> bool:
            return len(shape) == nd and shape[0] == n and n > 1

        out = None
        if mode == "get":
            if self._aca.gpus is None:
                out = np.zeros(target_shape, dtype=self.dtype)
            else:
                with self._aca.xp.cuda.Device(int(self._aca.gpus[0])):
                    out = xp.zeros(target_shape, dtype=self.dtype)

        main = (
            xp.cuda.runtime.getDevice() if self._aca.gpus is not None else None
        )
        try:
            for s in np.unique(shard_ids):
                rows = np.where(shard_ids == s)[0]
                idx0 = intra_all[rows].reshape((len(rows),) + (1,) * (nd - 1))
                # Slice axis-0-varying index arrays on their OWN device
                # before entering the shard context (cross-device slicing
                # is not legal cupy); broadcast-only arrays pass through.
                parts = [
                    self._slice_axis0_on_device(a, rows)
                    if _axis0_varies(np.shape(a)) else a
                    for a in intra_axes
                ]
                if mode == "set" and not is_scalar_val:
                    v_part = (
                        self._slice_axis0_on_device(val, rows)
                        if _axis0_varies(val_shape) else val
                    )
                if self._aca.gpus is None:
                    # asnumpy: tolerate device index/payload arrays handed
                    # to a CPU view (the legacy broadcast path did too).
                    shard_idx = tuple(
                        [idx0] + [np.asarray(asnumpy(p)) for p in parts]
                    )
                    if mode == "get":
                        out[rows] = shards[s][shard_idx]
                    elif is_scalar_val:
                        shards[s][shard_idx] = val
                    else:
                        shards[s][shard_idx] = np.asarray(asnumpy(v_part))
                else:
                    with xp.cuda.Device(int(self._aca.gpus[s])):
                        shard_idx = tuple(
                            [xp.asarray(idx0)] + [xp.asarray(p) for p in parts]
                        )
                        if mode == "get":
                            vals = shards[s][shard_idx]
                        elif is_scalar_val:
                            shards[s][shard_idx] = val
                        else:
                            shards[s][shard_idx] = xp.asarray(v_part)
                    if mode == "get":
                        with xp.cuda.Device(int(self._aca.gpus[0])):
                            out[rows] = xp.asarray(vals)
        finally:
            if self._aca.gpus is not None:
                xp.cuda.runtime.setDevice(main)
        return out if mode == "get" else None

    @staticmethod
    def _broadcast_fancy(band_idx, intra_axes):
        """Broadcast (band_idx, *intra_axes) to a common host-ndarray shape."""
        arrs = [np.asarray(asnumpy(band_idx))] + [
            np.asarray(asnumpy(a)) for a in intra_axes
        ]
        target_shape = np.broadcast_shapes(*[a.shape for a in arrs])
        return [np.broadcast_to(a, target_shape) for a in arrs], target_shape

    def _fancy_broadcast(self, band_idx, intra_axes, target_shape, mode, val):
        """Legacy full-broadcast fancy kernel (fallback for exotic index
        patterns the row-wise path does not cover)."""
        broadcast_arrays, target_shape = self._broadcast_fancy(band_idx, intra_axes)
        band_broadcast = broadcast_arrays[0]
        intra_broadcast = broadcast_arrays[1:]
        band_flat = band_broadcast.reshape(-1).astype(int)
        intra_flat = [a.reshape(-1) for a in intra_broadcast]
        split_map_host, global_to_intra = shard_lookup_maps(self._aca)
        shard_ids = split_map_host[band_flat]
        shards = self._shards

        is_scalar_val = mode == "set" and (
            np.isscalar(val) or (hasattr(val, "ndim") and val.ndim == 0)
        )
        if mode == "set" and not is_scalar_val:
            val_arr = np.asarray(asnumpy(val))
            val_broadcast = np.broadcast_to(val_arr, target_shape).reshape(-1)
        else:
            val_broadcast = None

        if mode == "get":
            if self._aca.gpus is None:
                out_flat = np.zeros(band_flat.shape[0], dtype=self.dtype)
            else:
                target = self._aca.gpus[0]
                with self._aca.xp.cuda.Device(int(target)):
                    out_flat = self._aca.xp.zeros(
                        band_flat.shape[0], dtype=self.dtype
                    )

        main = (
            self._aca.xp.cuda.runtime.getDevice() if self._aca.gpus is not None else None
        )
        try:
            for s in np.unique(shard_ids):
                rows = np.where(shard_ids == s)[0]
                intra_band = global_to_intra[band_flat[rows]]
                shard_idx = tuple(
                    [intra_band] + [arr[rows] for arr in intra_flat]
                )
                if self._aca.gpus is None:
                    if mode == "get":
                        out_flat[rows] = shards[s][shard_idx]
                    else:
                        if is_scalar_val:
                            shards[s][shard_idx] = val
                        else:
                            shards[s][shard_idx] = val_broadcast[rows]
                else:
                    if mode == "get":
                        with self._aca.xp.cuda.Device(int(self._aca.gpus[s])):
                            vals = shards[s][shard_idx]
                        with self._aca.xp.cuda.Device(int(self._aca.gpus[0])):
                            out_flat[rows] = self._aca.xp.asarray(vals)
                    else:
                        with self._aca.xp.cuda.Device(int(self._aca.gpus[s])):
                            if is_scalar_val:
                                shards[s][shard_idx] = val
                            else:
                                shards[s][shard_idx] = self._aca.xp.asarray(
                                    val_broadcast[rows]
                                )
        finally:
            if self._aca.gpus is not None:
                self._aca.xp.cuda.runtime.setDevice(main)

        if mode == "get":
            return out_flat.reshape(target_shape)
        return None

    def _write_to_shard(self, shard: int, intra: int, val) -> None:
        if self._aca.gpus is None:
            self._shards[shard][intra] = val
            return
        main = self._aca.xp.cuda.runtime.getDevice()
        try:
            with self._aca.xp.cuda.Device(int(self._aca.gpus[shard])):
                self._shards[shard][intra] = val
        finally:
            self._aca.xp.cuda.runtime.setDevice(main)

    def swap_rows(self, idx_a, idx_b) -> None:
        """Exchange rows ``idx_a[i] <-> idx_b[i]`` (parallel-resources plan P1).

        Same-shard pairs — the common case under band-grouped shard
        assignment, where a tempering swap pair shares its band — are
        swapped in place inside their owning device context with NO host
        hop. Cross-shard pairs fall back to the gather/scatter path.
        """
        idx_a = np.atleast_1d(np.asarray(asnumpy(idx_a), dtype=int))
        idx_b = np.atleast_1d(np.asarray(asnumpy(idx_b), dtype=int))
        if idx_a.shape != idx_b.shape:
            raise ValueError(
                f"swap_rows index shapes differ: {idx_a.shape} vs {idx_b.shape}"
            )
        if idx_a.size == 0:
            return
        sh_a, in_a, _ = self._resolve_array(idx_a)
        sh_b, in_b, _ = self._resolve_array(idx_b)
        same = sh_a == sh_b

        # Locality instrumentation (multi-GPU validation runbook): under
        # band-grouped shard assignment the cross count should stay ~0.
        self.swap_same_shard_count = getattr(self, "swap_same_shard_count", 0) + int(
            same.sum()
        )
        self.swap_cross_shard_count = getattr(self, "swap_cross_shard_count", 0) + int(
            (~same).sum()
        )
        if (~same).any():
            logger.debug(
                "BandView.swap_rows: %d cross-shard pair(s) this call "
                "(totals same=%d cross=%d)",
                int((~same).sum()),
                self.swap_same_shard_count,
                self.swap_cross_shard_count,
            )

        xp_mod = self._aca.xp
        shards = self._shards
        for s in np.unique(sh_a[same]):
            sel = same & (sh_a == s)
            shard = shards[s]
            if self._aca.gpus is None:
                ia, ib = in_a[sel], in_b[sel]
                tmp = shard[ia].copy()
                shard[ia] = shard[ib]
                shard[ib] = tmp
                continue
            main = xp_mod.cuda.runtime.getDevice()
            try:
                with xp_mod.cuda.Device(int(self._aca.gpus[s])):
                    ia = xp_mod.asarray(in_a[sel])
                    ib = xp_mod.asarray(in_b[sel])
                    tmp = shard[ia].copy()
                    shard[ia] = shard[ib]
                    shard[ib] = tmp
            finally:
                xp_mod.cuda.runtime.setDevice(main)

        if not same.all():
            # Cross-shard remainder (rare under band-grouped assignment):
            # the existing host-routed gather/scatter path stays correct.
            a, b = idx_a[~same], idx_b[~same]
            tmp = self._gather(a).copy()
            self._scatter(a, self._gather(b))
            self._scatter(b, tmp)

    def _gather(self, band_inds: np.ndarray):
        """Gather rows ``band_inds`` from every shard into one ndarray.

        Output lives on ``aca.gpus[0]`` when multi-GPU, on host when CPU.
        Each shard contributes its rows; cross-GPU rows are copied via
        ``xp.asarray`` inside the target device context.
        """
        shard_ids, intra, _ = self._resolve_array(band_inds)
        shards = self._shards
        per_band_shape = tuple(shards[0].shape[1:])
        out_shape = (len(band_inds),) + per_band_shape

        if self._aca.gpus is None:
            out = np.zeros(out_shape, dtype=self.dtype)
            for s in np.unique(shard_ids):
                rows = np.where(shard_ids == s)[0]
                out[rows] = shards[s][intra[rows]]
            return out

        target = self._aca.gpus[0]
        main = self._aca.xp.cuda.runtime.getDevice()
        try:
            with self._aca.xp.cuda.Device(int(target)):
                out = self._aca.xp.zeros(out_shape, dtype=self.dtype)
            for s in np.unique(shard_ids):
                rows = np.where(shard_ids == s)[0]
                with self._aca.xp.cuda.Device(int(self._aca.gpus[s])):
                    src = shards[s][intra[rows]]
                with self._aca.xp.cuda.Device(int(target)):
                    out[rows] = self._aca.xp.asarray(src)
            return out
        finally:
            self._aca.xp.cuda.runtime.setDevice(main)

    def _scatter(self, band_inds: np.ndarray, val) -> None:
        shard_ids, intra, _ = self._resolve_array(band_inds)
        shards = self._shards
        # Scalar broadcast vs per-row payload
        is_scalar = (
            np.isscalar(val)
            or (hasattr(val, "ndim") and val.ndim == 0)
        )

        if self._aca.gpus is None:
            for s in np.unique(shard_ids):
                rows = np.where(shard_ids == s)[0]
                if is_scalar:
                    shards[s][intra[rows]] = val
                else:
                    shards[s][intra[rows]] = val[rows]
            return

        main = self._aca.xp.cuda.runtime.getDevice()
        try:
            for s in np.unique(shard_ids):
                rows = np.where(shard_ids == s)[0]
                with self._aca.xp.cuda.Device(int(self._aca.gpus[s])):
                    if is_scalar:
                        shards[s][intra[rows]] = val
                    else:
                        sub = val[rows] if hasattr(val, "__getitem__") else val
                        shards[s][intra[rows]] = self._aca.xp.asarray(sub)
        finally:
            self._aca.xp.cuda.runtime.setDevice(main)

    def accumulate(self, idx, val) -> None:
        """In-place ``view[idx] += val`` with per-shard on-device adds.

        The generic ``view[idx] += val`` desugars to gather (all rows onto
        ``gpus[0]``), host-side add, scatter — the target bytes cross
        devices TWICE. This routes each row's add to its OWNING shard:
        only the matching rows of ``val`` move (once), and the ``+=`` runs
        in place on the shard buffer. Row semantics match ``__setitem__``
        with a per-row payload: ``idx`` is a 1-D global band index (bool
        masks accepted), ``val`` a scalar or an array whose leading axis
        matches ``idx``. Duplicate ids in ``idx`` collapse like fancy
        assignment (no double-accumulate), same as the gather/scatter
        path it replaces.
        """
        band_inds = np.atleast_1d(np.asarray(asnumpy(idx)))
        if band_inds.dtype == bool:
            if band_inds.shape[0] != self.shape[0]:
                raise IndexError(
                    f"boolean index shape {band_inds.shape} mismatches "
                    f"num_bands={self.shape[0]}"
                )
            band_inds = np.where(band_inds)[0]
        shard_ids, intra, _ = self._resolve_array(band_inds)
        shards = self._shards
        is_scalar = (
            np.isscalar(val)
            or (hasattr(val, "ndim") and val.ndim == 0)
        )

        if self._aca.gpus is None:
            for s in np.unique(shard_ids):
                rows = np.where(shard_ids == s)[0]
                if is_scalar:
                    shards[s][intra[rows]] += val
                else:
                    shards[s][intra[rows]] += val[rows]
            return

        main = self._aca.xp.cuda.runtime.getDevice()
        try:
            for s in np.unique(shard_ids):
                rows = np.where(shard_ids == s)[0]
                if is_scalar:
                    with self._aca.xp.cuda.Device(int(self._aca.gpus[s])):
                        shards[s][intra[rows]] += val
                    continue
                # Slice the payload on ITS device, add on the owning shard
                # (xp.asarray crosses devices directly, no host hop).
                sub = self._slice_axis0_on_device(val, rows)
                with self._aca.xp.cuda.Device(int(self._aca.gpus[s])):
                    shards[s][intra[rows]] += self._aca.xp.asarray(sub)
        finally:
            self._aca.xp.cuda.runtime.setDevice(main)

    def gather(self):
        """Materialise the full (num_bands, *per_band_shape) ndarray on gpus[0]."""
        return self._gather(np.arange(self.shape[0]))

    def copy(self):
        """Alias for :meth:`gather` -- matches ndarray's ``.copy()`` semantics."""
        return self.gather()


def band_gpu_assignment(
    num_bands: int,
    gpus: Optional[List[int]],
    mode: str = "striped",
    group_ids: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Return a per-slot GPU id assignment for use with :class:`AnalysisContainerArray`.

    Used by the GB special-move Buffer to shard its per-cell slabs across
    GPUs. ``num_bands`` is really the number of buffer SLOTS (one per active
    (temp, walker, band) cell).

    Parameters
    ----------
    num_bands:
        Number of buffer slots (== ``num_acs`` of the per-cell ACA).
    gpus:
        Available GPU ids. ``None`` or a single-GPU list returns ``None``
        (no custom assignment needed; legacy contiguous partition kicks in).
    mode:
        ``"striped"`` (default) — ``gpu_map[slot] = gpus[slot % len(gpus)]``.
        Fine-grained load balance over slots.

        ``"block"`` — contiguous-block split (legacy). Use only for testing;
        per the plan, this can violate the no-overlap invariant at shard
        seams when bands within a BandSorter pass touch their neighbours'
        edge buffers.
    group_ids:
        Optional per-slot BAND ids (parallel-resources plan P1): slots with
        the same band are placed on the SAME shard, so every cell of a band
        is device-local — in particular the tempering swap pairs (same
        band, adjacent temps), which under plain slot striping always
        landed on DIFFERENT shards (slot order puts a row's temps in
        consecutive slots) and paid a host-hop per swap. Bands round-robin
        across GPUs WITHIN each even/odd parity class (ranked by first
        appearance): the GB move activates bands in parity rotation, so
        ranking across classes could put all of one parity on a single
        device (e.g. active bands 3,4,7,8,... at ngpus=2). Overrides
        ``mode``.

    Returns
    -------
    ``np.ndarray`` of shape ``(num_bands,)`` with GPU ids, or ``None`` when
    no custom assignment is needed.
    """
    if gpus is None or len(gpus) <= 1:
        return None
    if group_ids is not None:
        group_ids = np.asarray(group_ids, dtype=int)
        if group_ids.shape != (num_bands,):
            raise ValueError(
                f"group_ids must have shape ({num_bands},); got {group_ids.shape}"
            )
        uniq, first_pos, inverse = np.unique(
            group_ids, return_index=True, return_inverse=True
        )
        # Rank each unique band within its own parity class, in first-
        # appearance order, so each parity pass spreads across all GPUs.
        ranks = np.empty(len(uniq), dtype=int)
        counters = [0, 0]
        for gi in np.argsort(first_pos):
            parity = int(uniq[gi] % 2)
            ranks[gi] = counters[parity]
            counters[parity] += 1
        return np.asarray(gpus, dtype=int)[ranks[inverse] % len(gpus)]
    if mode == "striped":
        return np.asarray([gpus[b % len(gpus)] for b in range(num_bands)], dtype=int)
    if mode == "block":
        per = int(np.ceil(num_bands / len(gpus)))
        return np.asarray(
            [gpus[min(b // per, len(gpus) - 1)] for b in range(num_bands)], dtype=int
        )
    raise ValueError(f"Unknown band_gpu_assignment mode: {mode!r}")


class AnalysisContainerArray:
    """Container for multiple :class:`AnalysisContainer` objects.

    Useful for parallelization and batching across many independent analyses
    (for example one container per source). Provides a flat view (``acs``) and
    preserves the original shape (``acs_shape``).

    Args:
        analysis_containers: Can be a single :class:`AnalysisContainer`, a 1D
            list of :class:`AnalysisContainer`, or a NumPy object array of
            :class:`AnalysisContainer`. If a 2D or higher list/array is input,
            it will be flattened to 1D and the original shape will be stored
            in ``acs_shape``.
        gpus: If not ``None``, list of GPU ids to use for storing data and
            sensitivity information. The data and sensitivity information for
            each container will be split across the GPUs as evenly as possible.
            If ``None``, everything is stored on the CPU.
        complex_psd: If ``True``, allocate a complex-valued PSD buffer (not yet
            implemented; raises ``NotImplementedError``).
        run_threaded: If ``True``, the array-level analysis ops
            (:meth:`_vectorized_dispatch` consumers — ``likelihood``,
            ``inner_product``, ``calculate_signal_*`` — and the domain-aware
            :meth:`signal_operation`) execute their per-split batches
            concurrently, one thread per split (GPU splits each enter their
            own device context; CPU splits rely on GIL-releasing work). The
            default ``False`` preserves the serial per-split loops.

    """

    @property
    def xp(self) -> object:
        """Return the active array module (``cupy`` if GPUs are configured, else ``numpy``)."""
        return cp if self.gpus is not None else np

    def __init__(
        self,
        analysis_containers: AnalysisContainer | List[AnalysisContainer] | np.ndarray,
        gpus: list | int | None = None,
        complex_psd: bool = False,
        gpu_assignment: Optional[np.ndarray] = None,
        run_threaded: bool = False,
    ) -> None:
        self.run_threaded = bool(run_threaded)
        self._thread_pool = None

        if isinstance(analysis_containers, AnalysisContainer):
            acs = np.array([analysis_containers], dtype=object)

        elif isinstance(analysis_containers, np.ndarray):
            assert analysis_containers.dtype == object
            assert np.all(
                [isinstance(tmp, AnalysisContainer) for tmp in analysis_containers.flatten()]
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

        if isinstance(gpus, int):
            gpus = [gpus]
        self.gpus = gpus
        if gpus is not None:
            # Sanity-check: each entry is a valid device id.
            for g in gpus:
                if not isinstance(g, (int, np.integer)):
                    raise TypeError(f"gpus entries must be int, got {type(g)}")
            if len(gpus) > 1:
                # Once per process at WARNING, then DEBUG: subclasses
                # (SubBandBuffer) are constructed per band-unit/proposal, so
                # repeating this mid-run reads as a re-split event when the
                # data plane hasn't moved (same demotion pattern as the
                # sensitivity invC report).
                global _MULTI_GPU_SPLIT_WARNED
                _log = logger.debug if _MULTI_GPU_SPLIT_WARNED else logger.warning
                _MULTI_GPU_SPLIT_WARNED = True
                _log(
                    "Multiple GPUs detected. Data and sensitivity information "
                    "will be split across the GPUs as evenly as possible."
                )
            # Allocations below run inside per-GPU `Device` contexts; we save
            # and restore the caller's main device so __init__ has no
            # side-effect on global CUDA state.
            self._main_device_at_init = self.xp.cuda.runtime.getDevice()

        ac_tmp = acs.flatten()[0]
        self.shape_sens = shape_sens = ac_tmp.sens_mat.shape[: -len(ac_tmp.sens_mat.data_shape)]

        if isinstance(ac_tmp.sens_mat.basis_settings, domains.WDMSettings):
            self.data_dtype = float
        else:
            self.data_dtype = complex

        self.noise_dtype = float if not complex_psd else complex
            
        assert np.all(np.asarray(shape_sens) < 5)  # makes sure it is not length of data
        # reset so that all data are linear in memory
        # Two partition modes:
        #   1. gpu_assignment is None (legacy): contiguous-block split of the
        #      ACs across the given GPUs in input order.
        #   2. gpu_assignment is provided: shape ``(acs_total_entries,)`` int
        #      array whose entries are GPU ids drawn from ``gpus``. Used by
        #      the GB band-tree Buffer to specify a striped/interleaved
        #      partition so neighbouring bands live on different GPUs.
        if gpu_assignment is not None and gpus is not None:
            gpu_assignment = np.asarray(gpu_assignment, dtype=int)
            if gpu_assignment.shape != (int(self.acs_total_entries),):
                raise ValueError(
                    f"gpu_assignment must have shape ({int(self.acs_total_entries)},); "
                    f"got {gpu_assignment.shape}"
                )
            unknown = set(np.unique(gpu_assignment).tolist()) - set(gpus)
            if unknown:
                raise ValueError(
                    f"gpu_assignment references GPU(s) {unknown} not in gpus={gpus}"
                )
            self.gpu_splits = gpu_splits = [
                np.where(gpu_assignment == g)[0] for g in gpus
            ]
        else:
            # One split on CPU; len(gpus) splits with GPUs. (The n_splits
            # CPU-thread splitting knob was removed — parallel-resources
            # plan P4: it duplicated kernel-level OMP threading above the
            # kernels with oversubscription risk and had no callers.)
            num_machines = 1 if gpus is None else len(gpus)
            # Load-balanced contiguous blocks (parallel-resources plan P1):
            # array_split keeps per-split sizes within 1 of each other. The
            # old ceil-stride split could leave the last device nearly idle
            # (e.g. 10 walkers on 4 devices -> 3/3/3/1 instead of 3/3/2/2).
            self.gpu_splits = gpu_splits = np.array_split(
                np.arange(self.acs_total_entries), num_machines
            )
            sizes = [len(s) for s in gpu_splits]
            if len(sizes) > 1 and max(sizes) != min(sizes):
                logger.info(
                    "AnalysisContainerArray split is uneven (%s entries per "
                    "split); the larger split(s) set the pace. Choose "
                    "nwalkers divisible by the device count to balance.",
                    sizes,
                )

        self.gpu_map = np.zeros(self.acs_total_entries, dtype=int)
        self.split_map = np.zeros(self.acs_total_entries, dtype=int)
        self.linear_data_arr = []
        self.linear_psd_arr = []
        for i, split in enumerate(gpu_splits):
            if gpus is not None:
                self.gpu_map[split] = gpus[i]
                # Allocate this shard's buffers on the owning GPU.
                with self.xp.cuda.Device(gpus[i]):
                    self.linear_data_arr.append(
                        self.xp.zeros(
                            self.data_length * self.nchannels * len(split),
                            dtype=self.data_dtype,
                        )
                    )
                    self.linear_psd_arr.append(
                        self.xp.zeros(
                            self.data_length * np.prod(shape_sens) * len(split),
                            dtype=self.noise_dtype,
                        )
                    )
            else:
                self.gpu_map[split] = 0
                self.linear_data_arr.append(
                    self.xp.zeros(
                        self.data_length * self.nchannels * len(split),
                        dtype=self.data_dtype,
                    )
                )
                self.linear_psd_arr.append(
                    self.xp.zeros(
                        self.data_length * np.prod(shape_sens) * len(split),
                        dtype=self.noise_dtype,
                    )
                )
            self.split_map[split] = i

        self.num_acs = len(acs.flatten())
        self.reset_linear_data_arr()
        self.reset_linear_psd_arr()
        if gpus is not None:
            # Restore whatever device the caller was on before __init__.
            self.xp.cuda.runtime.setDevice(self._main_device_at_init)

    def flatten(self):
        return self.acs.flatten()
    
    def synchronize(self):
        if self.gpus is not None:
            for gpu in self.gpus:
                with self.xp.cuda.device.Device(gpu):
                    self.xp.cuda.runtime.deviceSynchronize()

    def zero_out_data_arr(self):
        """Zero the linear (per-GPU) data buffers in place."""
        if self.gpus is None:
            for buf in self.linear_data_arr:
                buf[:] = 0.0
            return

        main_gpu = self.xp.cuda.runtime.getDevice()
        for gpu_i, gpu in enumerate(self.gpus):
            with self.xp.cuda.device.Device(gpu):
                self.linear_data_arr[gpu_i][:] = 0.0

        self.xp.cuda.runtime.setDevice(main_gpu)

    def snapshot_linear_data_arr(self):
        """Per-shard copy of ``linear_data_arr``.

        Returns a list of device-local buffers (one per shard), each copied
        inside its owning device context. Pair with
        :meth:`restore_linear_data_arr` for propose-scoped save/restore of
        the residual plane.
        """
        if self.gpus is None:
            return [b.copy() for b in self.linear_data_arr]
        main_gpu = self.xp.cuda.runtime.getDevice()
        try:
            out = []
            for i, gpu in enumerate(self.gpus):
                with self.xp.cuda.Device(int(gpu)):
                    out.append(self.linear_data_arr[i].copy())
            return out
        finally:
            self.xp.cuda.runtime.setDevice(main_gpu)

    def restore_linear_data_arr(self, snapshot):
        """In-place restore of every shard from the matching snapshot entry.

        Writes into the existing shard buffers (never reallocates) so views
        bound by :meth:`reset_linear_data_arr` stay valid.
        """
        if self.gpus is None:
            for buf, snap in zip(self.linear_data_arr, snapshot):
                buf[:] = snap[:]
            return
        main_gpu = self.xp.cuda.runtime.getDevice()
        try:
            for i, gpu in enumerate(self.gpus):
                with self.xp.cuda.Device(int(gpu)):
                    self.linear_data_arr[i][:] = snapshot[i][:]
        finally:
            self.xp.cuda.runtime.setDevice(main_gpu)

    def _free_pool_blocks_per_device(self):
        """Release the CuPy pool's free blocks once per device.

        ``MemoryPool.free_all_blocks()`` acts on the *current* device only, so
        the repack loops used to call it once per container (with that
        container's device selected). That wiped the pool ``nwalkers`` times
        per repack, pushing every subsequent large allocation back onto a
        synchronising ``cudaMalloc``; one sweep per device at the end gives the
        same release with a single call per arena. Set
        ``ACA_REPACK_FREE_BLOCKS=0`` to skip the sweep entirely.
        """
        if self.gpus is None:
            return
        if not bool(int(os.environ.get("ACA_REPACK_FREE_BLOCKS", "1"))):
            return
        for gpu in self.gpus:
            with self.xp.cuda.Device(int(gpu)):
                self.xp.get_default_memory_pool().free_all_blocks()

    def reset_linear_data_arr(self):
        """Repack each container's data residual into the contiguous per-GPU data buffer."""
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

            flat_data_res_here = ac.data_res_arr.flatten()
            dst = self.linear_data_arr[split][start_index:end_index]
            if _same_device(flat_data_res_here, dst):
                # source already lives on the target device: stay resident.
                dst[:] = flat_data_res_here
            else:
                # cross-device source: route through the host (broken P2P).
                if hasattr(flat_data_res_here, "get"):
                    flat_data_res_here = flat_data_res_here.get()
                dst[:] = self.xp.asarray(flat_data_res_here)
            # ac.data_res_arr._data_res_arr = signal_class(arr=self.linear_data_arr[split][start_index:end_index].reshape(self.nchannels, *self.data_shape), settings=settings)     #as todo check: are those 2 lines the same?
            ac.data_res_arr.data_res_arr._arr = self.linear_data_arr[split][
                start_index:end_index
            ].reshape((self.nchannels,) + self.end_shape)
            # TODO: add check to make sure changes are made inline along with protections

        if self.gpus is not None:
            self._free_pool_blocks_per_device()
            self.xp.cuda.runtime.setDevice(main_gpu)

    def reset_linear_psd_arr(self):
        """Repack each container's inverse-PSD into the contiguous per-GPU PSD buffer."""
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
            # invC may live on a different GPU (always computed on GPU 0).
            # Route through CPU to avoid broken P2P cross-device copies --
            # but only when the source really is on another device; the
            # same-device case stays resident (the D2H+H2D round trip was
            # pure overhead there).
            invC_src = ac.sens_mat.invC
            dst = self.linear_psd_arr[split][start_index:end_index]
            if _same_device(invC_src, dst):
                dst[:] = invC_src.reshape(-1)
            else:
                if hasattr(invC_src, 'get'):
                    invC_src = invC_src.get()
                dst[:] = self.xp.asarray(invC_src.flatten())
            ac.sens_mat.invC = self.linear_psd_arr[split][start_index:end_index].reshape(
                self.shape_sens + self.end_shape
            )

            # TODO: add check to make sure changes are made inline along with protections

        if self.gpus is not None:
            self._free_pool_blocks_per_device()
            self.xp.cuda.runtime.setDevice(main_gpu)

    @property
    def settings(self) -> DomainSettingsBase:
        """Basis settings of the data residual array."""
        return self.acs[0].data_res_arr.settings

    @property
    def f_arr(self):
        """Frequency array of the first analysis container."""
        return self.acs[0].data_res_arr.f_arr

    @property
    def df(self):
        """Frequency spacing inferred from :attr:`f_arr`."""
        return self.f_arr[1] - self.f_arr[0]

    def __len__(self) -> int:
        return len(self.acs)

    def flatten(self) -> "np.ndarray":
        """Return the underlying per-walker :class:`AnalysisContainer` object
        array flattened to 1D (delegates to ``self.acs.flatten()``)."""
        return self.acs.flatten()

    # ------------------------------------------------------------------
    # Vectorized dispatcher (replaces the legacy _loop_operation)
    # ------------------------------------------------------------------

    @staticmethod
    def _payload_leading(payload) -> int | None:
        """Leading axis of ``payload`` (single ndarray, dict of ndarrays, or list)."""
        if payload is None:
            return None
        if isinstance(payload, Mapping):
            n = None
            for v in payload.values():
                vn = v.shape[0] if hasattr(v, "shape") else len(v)
                if n is None:
                    n = vn
                elif vn != n:
                    raise ValueError(
                        "All dict payload entries must share the leading axis; "
                        f"got {n} and {vn}."
                    )
            return n
        if hasattr(payload, "shape"):
            return payload.shape[0]
        return len(payload)

    def _vectorized_dispatch(
        self,
        op_name: str,
        payload: Optional[Any] = None,
        index: Optional[np.ndarray] = None,
        op_kwargs: Optional[dict] = None,
        reshape_to_acs_shape: bool = True,
        domain_error_value: Optional[float] = None,
        payload_to_device: bool = True,
    ) -> np.ndarray | tuple:
        """Apply ``op_name`` on each row of ``payload`` against the AC named by ``index[row]``.

        This is the single dispatcher that replaces the legacy
        :meth:`_loop_operation`. It is the multi-GPU entry point for every
        analysis op on :class:`AnalysisContainerArray` (likelihood, inner
        product, SNR, calculate_signal_*, template_*, property reads).

        Parameters
        ----------
        op_name:
            Name of the attribute (callable method or scalar property) on
            each :class:`AnalysisContainer`.
        payload:
            Per-row input passed as the first positional argument to the op.
            May be ``None`` (data-only / property), a single ``np.ndarray`` /
            ``cp.ndarray`` whose leading axis is the row count, or a
            ``Mapping`` of ``{model_name: ndarray}`` (multi-model signal_gen
            case) with a consistent leading axis across entries. When
            ``index`` is None, the leading axis must equal ``num_acs``.
        index:
            Optional 1-D int array of length ``N`` with
            ``index[i] in [0, num_acs)``. When None, defaults to
            ``arange(num_acs)``; in that case the result is reshaped to
            ``acs_shape`` for backwards compatibility with the legacy
            ``_loop_operation`` return.
        op_kwargs:
            Extra keyword arguments forwarded to the op.
        reshape_to_acs_shape:
            When True and ``index`` was None, reshape the output from
            ``(num_acs,)`` to ``self.acs_shape``.

        Returns
        -------
        np.ndarray of shape ``(N,)`` (or ``acs_shape`` when ``index`` is
        None and reshape requested), or a tuple of such arrays when the op
        returns a tuple (e.g. ``snr -> (opt_snr, det_snr)``).
        """
        op_kwargs = op_kwargs or {}

        n_payload = self._payload_leading(payload)
        if index is None:
            if payload is not None:
                if n_payload != self.num_acs:
                    raise ValueError(
                        f"payload leading axis ({n_payload}) must equal "
                        f"num_acs ({self.num_acs}) when index is None"
                    )
            index_arr = np.arange(self.num_acs)
            index_was_none = True
        else:
            index_arr = np.asarray(index, dtype=int)
            if payload is not None and n_payload != len(index_arr):
                raise ValueError(
                    f"payload leading axis ({n_payload}) must equal "
                    f"len(index) ({len(index_arr)})"
                )
            if index_arr.size and (
                index_arr.min() < 0 or index_arr.max() >= self.num_acs
            ):
                raise ValueError(
                    f"index entries must be in [0, {self.num_acs}); "
                    f"got min={index_arr.min()} max={index_arr.max()}"
                )
            index_was_none = False

        N = len(index_arr)
        acs_flat = self.acs.flatten()
        results: list = [None] * N

        def _slice_row(p, r):
            if p is None:
                return None
            if isinstance(p, Mapping):
                return {k: v[r] for k, v in p.items()}
            return p[r]

        def _to_device(p):
            if p is None or self.gpus is None:
                return p
            if isinstance(p, Mapping):
                return {k: self.xp.asarray(v) for k, v in p.items()}
            if hasattr(p, "shape"):
                return self.xp.asarray(p)
            return p

        def _call_one(ac, p_row):
            attr = getattr(ac, op_name)
            if not callable(attr):
                return attr
            try:
                if p_row is None:
                    return attr(**op_kwargs)
                # Multi-model path (Mapping): AC expects a single dict
                # positional arg — same call shape as the plain-row path.
                return attr(p_row, **op_kwargs)
            except WaveformDomainError as exc:
                # Prior support can leak outside a waveform's domain of
                # validity; when the caller opted in (samplers pass the
                # -1e300 sentinel), score the row at the floor instead of
                # aborting the whole batch.
                if domain_error_value is None:
                    raise
                logger.debug(
                    "%s: waveform domain error -> %s: %s",
                    op_name, domain_error_value, exc,
                )
                return domain_error_value

        def _to_host(res):
            if isinstance(res, tuple):
                return tuple(asnumpy(x) if hasattr(x, "shape") else x for x in res)
            if hasattr(res, "shape"):
                return asnumpy(res)
            return res

        # Per-split execution (rows grouped by split_map so multi-GPU
        # parallelize the same way GPU splits do). results[r] slots are
        # distinct per row, so threaded workers never write the same slot.
        split_to_rows = self._split_rows(index_arr)

        if self.gpus is None:

            def _worker(split, rows):
                for r in rows:
                    ac_i = int(index_arr[int(r)])
                    res = _call_one(acs_flat[ac_i], _slice_row(payload, int(r)))
                    results[int(r)] = _to_host(res)

            self._run_per_split(_worker, split_to_rows)
        else:
            main_gpu = self.xp.cuda.runtime.getDevice()

            def _worker(split, rows):
                # cupy's current device is thread-local: each worker enters
                # its split's device itself.
                with self.xp.cuda.Device(int(self.gpus[split])):
                    for r in rows:
                        ac_i = int(index_arr[int(r)])
                        sub = _slice_row(payload, int(r))
                        # Params-based ops keep HOST rows (the engine
                        # generators do np.asarray(params)); template ops
                        # migrate their arrays to the owning device.
                        if payload_to_device:
                            sub = _to_device(sub)
                        res = _call_one(acs_flat[ac_i], sub)
                        results[int(r)] = _to_host(res)

            try:
                self._run_per_split(_worker, split_to_rows)
            finally:
                self.xp.cuda.runtime.setDevice(main_gpu)

        # Stack results. Handle tuple-valued ops (e.g. snr) separately.
        if N > 0 and isinstance(results[0], tuple):
            ntup = len(results[0])
            stacked = tuple(
                np.array([results[r][k] for r in range(N)])
                for k in range(ntup)
            )
            if reshape_to_acs_shape and index_was_none and self.acs_shape != (N,):
                stacked = tuple(s.reshape(self.acs_shape) for s in stacked)
            return stacked

        out = np.array(results)
        if reshape_to_acs_shape and index_was_none and self.acs_shape != (N,):
            out = out.reshape(self.acs_shape)
        return out

    @property
    def start_freq_ind(self):
        """Per-container ``start_freq_ind`` reshaped to :attr:`acs_shape`."""
        return self._vectorized_dispatch("start_freq_ind")

    @property
    def start_freq_layer_inds(self):
        """Per-container WDM ``start_freq_layer_ind`` reshaped to :attr:`acs_shape`.

        Each AC's view is ``Nf_active`` frequency layers wide centred on
        its layer of interest, so this array varies across containers
        (analogous to :attr:`start_freq_ind` in the FD case).
        """
        return self._vectorized_dispatch("start_freq_layer_ind")

    @property
    def start_time_layer_inds(self):
        """Per-container WDM ``start_time_layer_ind`` reshaped to :attr:`acs_shape`.

        Every container in the array covers the same active time range,
        so every entry is the same value. Kept as an array for API
        symmetry with :attr:`start_freq_layer_inds`.
        """
        return self._vectorized_dispatch("start_time_layer_ind")

    @property
    def layer_df(self):
        """WDM layer frequency spacing of the first container (shared across the array)."""
        return self.acs[0].layer_df

    @property
    def layer_dt(self):
        """WDM layer time spacing of the first container (shared across the array)."""
        return self.acs[0].layer_dt

    # ------------------------------------------------------------------
    # Per-split execution (serial or threaded) — shared by the vectorized
    # dispatcher and signal_operation. One Python orchestration layer
    # spreading work across threads and GPUs: GPU splits each enter their
    # own device context inside the worker (cupy's current device is
    # thread-local); CPU splits run as plain threads and
    # pay off where the per-row work releases the GIL.
    # ------------------------------------------------------------------

    @property
    def thread_pool(self) -> ThreadPoolExecutor:
        """Lazy ``ThreadPoolExecutor`` with one worker per split."""
        if self._thread_pool is None:
            self._thread_pool = ThreadPoolExecutor(
                max_workers=max(1, len(self.gpu_splits))
            )
        return self._thread_pool

    def _split_rows(self, index_arr: np.ndarray) -> dict:
        """Group flat row positions by owning split: ``{split: rows}``."""
        split_per_row = self.split_map[np.asarray(index_arr, dtype=int)]
        return {
            int(s): np.where(split_per_row == s)[0]
            for s in np.unique(split_per_row)
        }

    def _run_per_split(self, worker, split_to_rows: dict, run_threaded=None) -> None:
        """Run ``worker(split_index, rows)`` once per populated split.

        With ``run_threaded`` (default :attr:`run_threaded`) and more than
        one populated split, splits execute concurrently in
        :attr:`thread_pool`; worker exceptions propagate to the caller.
        The worker is responsible for entering its split's device context
        (GPU splits) — see the call sites.
        """
        if run_threaded is None:
            run_threaded = self.run_threaded
        items = [(s, rows) for s, rows in split_to_rows.items() if len(rows)]
        if run_threaded and len(items) > 1:
            futures = [
                self.thread_pool.submit(worker, s, rows) for s, rows in items
            ]
            for f in futures:
                f.result()  # re-raise worker exceptions in caller
        else:
            for s, rows in items:
                worker(s, rows)

    # ------------------------------------------------------------------
    # Vectorized analysis ops (sharded by ``index`` across GPUs)
    # ------------------------------------------------------------------
    # All of these accept an optional ``index`` array (1-D int, ``index[i]
    # in [0, num_acs)``) that maps each input row to an AC. When ``index``
    # is None, the leading axis of the payload (if any) must equal
    # ``num_acs`` and the implicit mapping is ``arange(num_acs)`` -- one
    # row per AC, in flat order. Output is reshaped back to ``acs_shape``
    # in that case for backwards compatibility with the legacy
    # ``_loop_operation`` return shape.

    def inner_product(self, index=None, **kwargs):
        """Data-only inner product per AC.

        Sharded by ``index``: each AC's call runs on its owning GPU.
        ``index=None`` evaluates every AC (legacy behaviour); pass an
        ``index`` array to evaluate a subset.
        """
        return self._vectorized_dispatch(
            "inner_product", payload=None, index=index, op_kwargs=kwargs
        )

    def likelihood(self, index=None, **kwargs):
        """Data-only likelihood per AC. See :meth:`inner_product` for ``index`` semantics."""
        return self._vectorized_dispatch(
            "likelihood", payload=None, index=index, op_kwargs=kwargs
        )

    def snr(self, index=None, **kwargs):
        """Data-only SNR per AC. See :meth:`inner_product` for ``index`` semantics."""
        return self._vectorized_dispatch(
            "snr", payload=None, index=index, op_kwargs=kwargs
        )

    def calculate_signal_likelihood(
        self, params, index=None, domain_error_value=None, **kwargs
    ):
        """Vectorized signal-likelihood across many (params, AC) rows.

        ``params`` is row-major with leading axis ``N`` (and may be a
        ``Mapping`` of ``{model_name: ndarray}`` in the multi-model
        ``signal_gen`` case). ``index[i]`` names the AC that row ``i``
        targets. When ``index`` is None, ``N == num_acs`` is required and
        the implicit mapping is one row per AC.

        ``domain_error_value``: when set (samplers pass ``-1e300``), a
        per-row :class:`WaveformDomainError` scores that row at this value
        instead of aborting the batch.

        Returns a host ``np.ndarray`` of per-row likelihoods. The
        dispatcher groups rows by ``self.gpu_map[index]`` so the per-AC
        compute runs on the owning GPU; CuPy stream concurrency lets
        multiple GPUs work in parallel from a single Python thread.
        """
        return self._vectorized_dispatch(
            "calculate_signal_likelihood",
            payload=params,
            index=index,
            op_kwargs=kwargs,
            domain_error_value=domain_error_value,
            # sampling-basis params stay on host: the engine-convention
            # generators (SourceSignalGen / MoveSignalGen) np.asarray them.
            payload_to_device=False,
        )

    def calculate_signal_inner_product(self, params, index=None, **kwargs):
        """Vectorized signal-inner-product across many (params, AC) rows.

        See :meth:`calculate_signal_likelihood` for the ``index`` and
        ``params`` semantics.
        """
        return self._vectorized_dispatch(
            "calculate_signal_inner_product",
            payload=params,
            index=index,
            op_kwargs=kwargs,
            payload_to_device=False,
        )

    def calculate_signal_snr(self, params, index=None, **kwargs):
        """Vectorized signal SNR (returns tuple-of-arrays: ``(opt, det)``).

        See :meth:`calculate_signal_likelihood` for the ``index`` and
        ``params`` semantics.
        """
        return self._vectorized_dispatch(
            "calculate_signal_snr",
            payload=params,
            index=index,
            op_kwargs=kwargs,
            payload_to_device=False,
        )

    def template_inner_product(self, template, index=None, **kwargs):
        """Vectorized template inner product (template-per-row).

        ``template`` is row-major with leading axis ``N`` along the row
        dimension; row ``i`` is passed as the single template argument
        to ``AC[index[i]].template_inner_product``. See
        :meth:`calculate_signal_likelihood` for ``index`` semantics.
        """
        return self._vectorized_dispatch(
            "template_inner_product",
            payload=template,
            index=index,
            op_kwargs=kwargs,
        )

    def template_likelihood(self, template, index=None, **kwargs):
        """Vectorized template likelihood (template-per-row).

        See :meth:`template_inner_product` for the ``index`` and
        ``template`` semantics.
        """
        return self._vectorized_dispatch(
            "template_likelihood",
            payload=template,
            index=index,
            op_kwargs=kwargs,
        )

    def template_snr(self, template, index=None, **kwargs):
        """Vectorized template SNR (template-per-row).

        See :meth:`template_inner_product` for the ``index`` and
        ``template`` semantics. Returns a tuple-of-arrays ``(opt, det)``.
        """
        return self._vectorized_dispatch(
            "template_snr",
            payload=template,
            index=index,
            op_kwargs=kwargs,
        )

    def __getitem__(self, index: Any) -> np.ndarray[AnalysisContainer]:
        return self.acs[index]

    # ------------------------------------------------------------------
    # Public signal operation API
    # ------------------------------------------------------------------

    def signal_operation(
        self,
        sign: int,
        templates: DomainBaseArray | List[DomainBase] | DomainBase | np.ndarray | cp.ndarray,
        data_index: Optional[np.ndarray] = None,
        start_index=None,
    ) -> None:
        """Apply ``sign * template`` to each targeted data residual array.

        Domain-aware: per-template add/subtract is delegated to
        :meth:`DomainBase.add_signal`, which handles partial time/frequency
        overlap for each domain.

        Args:
            sign: ``+1`` to add, ``-1`` to subtract.
            templates: One of:

                * a :class:`~lisatools.domains.DomainBaseArray` (recommended),
                * a list of :class:`~lisatools.domains.DomainBase` objects,
                * a single :class:`~lisatools.domains.DomainBase` (possibly batched),
                * a raw ``np.ndarray`` / ``cp.ndarray`` (legacy -- deprecated).

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
            # legacy raw-array path
            warnings.warn(
                "Passing a raw ndarray to signal_operation is deprecated. "
                "Wrap your templates in a DomainBase (or DomainBaseArray) instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            ac0_data = self.acs.flatten()[0].data
            ac0_settings = ac0_data.settings
            if isinstance(ac0_settings, domains.WDMSettings):
                if templates.ndim == 2:
                    templates = templates[None, :]
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
            acs_flat = self.acs.flatten()
            if self.gpus is None:
                for i, (di, si) in enumerate(zip(data_index, start_index)):
                    acs_flat[di].data[:, si : si + template_length] += sign * templates[i]
                return
            main_gpu = self.xp.cuda.runtime.getDevice()
            gpu_per_template = self.gpu_map[data_index]
            try:
                for gpu in np.unique(gpu_per_template):
                    mask = gpu_per_template == gpu
                    rows = np.where(mask)[0]
                    with self.xp.cuda.Device(int(gpu)):
                        for i in rows:
                            di = int(data_index[i])
                            si = int(start_index[i])
                            t_i = templates[i]
                            # stft_tof cross-device hygiene (see _signal_on_device).
                            if isinstance(t_i, self.xp.ndarray) and t_i.device.id != int(gpu):
                                if not t_i.flags["C_CONTIGUOUS"]:
                                    with self.xp.cuda.Device(t_i.device.id):
                                        t_i = self.xp.ascontiguousarray(t_i)
                                t_i = self.xp.asarray(t_i.get())
                            acs_flat[di].data[:, si : si + template_length] += (
                                sign * t_i
                            )
            finally:
                self.xp.cuda.runtime.setDevice(main_gpu)
            return
        else:
            raise TypeError(
                f"templates must be a DomainBase, list of DomainBase, "
                f"DomainBaseArray, or ndarray (legacy). Got {type(templates)}."
            )

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

        # Per-split execution (serial by default; one thread per split with
        # run_threaded=True — distinct ACs per row, so threaded workers
        # never touch the same residual buffer).
        split_to_rows = self._split_rows(data_index)

        if self.gpus is None:

            def _worker(split, rows):
                for i in rows:
                    acs_flat[int(data_index[int(i)])].data.add_signal(
                        item_list[int(i)], sign=sign
                    )

            self._run_per_split(_worker, split_to_rows)
            return

        # Multi-GPU: enter each owning device once per shard. Per-domain
        # partial-overlap handling is delegated to DomainBase.add_signal,
        # which performs an in-place update on acs_flat[di].data._arr
        # (a slice of self.linear_data_arr[split] that lives on this GPU).
        # Templates produced on a different device are first migrated via
        # _signal_on_device (stft_tof cross-device hygiene).
        main_gpu = self.xp.cuda.runtime.getDevice()

        def _worker(split, rows):
            gpu = int(self.gpus[split])
            with self.xp.cuda.Device(gpu):
                for i in rows:
                    signal = self._signal_on_device(item_list[int(i)], gpu)
                    acs_flat[int(data_index[int(i)])].data.add_signal(
                        signal, sign=sign
                    )

        try:
            self._run_per_split(_worker, split_to_rows)
        finally:
            self.xp.cuda.runtime.setDevice(main_gpu)

    def apply_signal_from_params(
        self,
        sign: int,
        params: Any,
        index: Optional[np.ndarray] = None,
        *,
        waveform_kwargs: Optional[dict] = None,
        signal_gen: Any = None,
        signal_gen_resolver: Optional[Callable] = None,
        apply_transform: bool = True,
        domain_error: str = "raise",
    ) -> None:
        """Build each row's template and apply ``sign * template`` to its AC.

        The generation+application analogue of :meth:`signal_operation` for
        params-based templates: each row's template is assembled through the
        targeted container's own ``signal_gen`` machinery
        (:meth:`AnalysisContainer.build_template`) and applied to that
        container's residual ONE ROW AT A TIME — never more than one live
        template per split (the peak-memory guard the source moves rely
        on). Rows are grouped by owning split and run inside that split's
        device context through the shared per-split runner, so shards run
        concurrently when the array is ``run_threaded`` (multi-GPU) and
        serially on CPU / single GPU.

        Args:
            sign: ``+1`` adds to the residual array, ``-1`` subtracts.
            params: Row-major params — an array with leading axis ``N`` or a
                ``Mapping`` of ``{model_name: rows}`` — sliced per row and
                handed to :meth:`AnalysisContainer.build_template`.
            index: 1-D int array mapping row ``i`` to
                ``self.acs.flatten()[index[i]]``; ``None`` means one row per
                AC (``N == num_acs``).
            waveform_kwargs: Forwarded to ``build_template``.
            signal_gen: Static generator override forwarded to
                ``build_template`` (may itself be a callable generator — it
                is passed through untouched).
            signal_gen_resolver: Optional ``resolver(ac) -> override``
                evaluated per targeted container (e.g. a move's
                installed-generator resolution); wins over ``signal_gen``.
            apply_transform: Forwarded to ``build_template``.
            domain_error: ``"raise"`` (default) propagates
                :class:`WaveformDomainError`; ``"skip"`` treats that row's
                template as zero (logged) — the sampling convention where
                the row's likelihood carries the ``-1e300`` sentinel
                elsewhere.
        """
        import gc

        if domain_error not in ("raise", "skip"):
            raise ValueError(
                f"domain_error must be 'raise' or 'skip'; got {domain_error!r}"
            )

        n_rows = self._payload_leading(params)
        if index is None:
            if n_rows != self.num_acs:
                raise ValueError(
                    f"params leading axis ({n_rows}) must equal num_acs "
                    f"({self.num_acs}) when index is None"
                )
            index_arr = np.arange(self.num_acs)
        else:
            index_arr = np.atleast_1d(np.asarray(index, dtype=int))
            if n_rows != len(index_arr):
                raise ValueError(
                    f"params leading axis ({n_rows}) must equal len(index) "
                    f"({len(index_arr)})"
                )

        acs_flat = self.acs.flatten()

        def _slice_row(p, r):
            if isinstance(p, Mapping):
                return {k: v[r] for k, v in p.items()}
            return p[r]

        def _worker(split, rows):
            dev = None if self.gpus is None else int(self.gpus[int(split)])
            with (nullcontext() if dev is None else self.xp.cuda.Device(dev)):
                for r in rows:
                    ac = acs_flat[int(index_arr[int(r)])]
                    try:
                        sig = ac.build_template(
                            _slice_row(params, int(r)),
                            waveform_kwargs=waveform_kwargs,
                            signal_gen=(
                                signal_gen_resolver(ac)
                                if signal_gen_resolver is not None
                                else signal_gen
                            ),
                            apply_transform=apply_transform,
                        )
                    except WaveformDomainError as exc:
                        if domain_error == "raise":
                            raise
                        logger.warning(
                            "apply_signal_from_params: row %d (AC %d) is "
                            "outside the waveform domain; treating its "
                            "template as zero: %s",
                            int(r), int(index_arr[int(r)]), exc,
                        )
                        continue
                    if dev is not None:
                        sig = self._signal_on_device(sig, dev)
                    ac.data.add_signal(sig, sign=sign)
                    del sig
                gc.collect()
                if self.gpus is not None:
                    self.xp.get_default_memory_pool().free_all_blocks()

        split_to_rows = self._split_rows(index_arr)
        if self.gpus is None:
            self._run_per_split(_worker, split_to_rows)
        else:
            main_gpu = self.xp.cuda.runtime.getDevice()
            try:
                self._run_per_split(_worker, split_to_rows)
            finally:
                self.xp.cuda.runtime.setDevice(main_gpu)

    def _signal_on_device(self, signal: DomainBase, gpu: int) -> DomainBase:
        """Return ``signal`` with its array resident on ``gpu``.

        Cross-device hygiene absorbed from the stft_tof branch: templates
        produced by a proposal on another device are first made C-contiguous
        on their *source* device (CuPy cannot copy non-contiguous arrays
        between devices), then routed through host memory -- ``cp.asarray``
        does NOT copy a CuPy array across devices (it returns the same
        object) and direct P2P access between non-adjacent devices is not
        guaranteed. Must be called inside the target ``Device(gpu)`` context.
        """
        arr = signal.arr
        if not isinstance(arr, self.xp.ndarray):
            return signal
        if arr.device.id == gpu:
            return signal
        if not arr.flags["C_CONTIGUOUS"]:
            with self.xp.cuda.Device(arr.device.id):
                arr = self.xp.ascontiguousarray(arr)
        # host-route the cross-device copy (no P2P assumption)
        arr = self.xp.asarray(arr.get())
        settings = signal.settings
        return settings.associated_class(arr, settings)

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

    # ------------------------------------------------------------------
    # Cross-shard gather helpers (for non-GB callers like MBH/PSD)
    # ------------------------------------------------------------------

    def _gather_per_gpu_to_single(
        self,
        per_gpu_list: list,
        target_gpu: Optional[int] = None,
    ):
        """Concatenate per-GPU shards into a single buffer in AC order.

        Used by callers (MBH/PSD recipe steps, etc.) that need a flat
        single-buffer view of the residual or inverse PSD, e.g. to hand
        to a kernel that doesn't speak the multi-shard list contract.

        - Single-shard (``len(per_gpu_list) == 1``): returns the buffer
          directly (no copy); ``target_gpu`` is ignored.
        - CPU (``self.gpus is None``): returns ``np.concatenate(per_gpu_list)``.
        - Multi-shard GPU: copies every shard onto ``target_gpu`` (default:
          ``self.gpus[0]``), then concatenates in AC order so the result is
          indexable by global AC index just like the original
          ``linear_*_arr[0]`` was.

        The concatenation is by ``gpu_splits`` order. To preserve global
        AC order, we sort each shard's entries by their original AC index.
        """
        if len(per_gpu_list) == 1:
            return per_gpu_list[0]
        if self.gpus is None:
            return np.concatenate(per_gpu_list)

        if target_gpu is None:
            target_gpu = self.gpus[0]

        # Per-AC element size = len(buffer) / number_of_acs_on_that_shard.
        sizes_per_ac = [int(buf.size // max(len(self.gpu_splits[i]), 1))
                        for i, buf in enumerate(per_gpu_list)]
        # Sanity: all shards must agree on per-AC size.
        if sizes_per_ac and len(set(sizes_per_ac)) != 1:
            raise RuntimeError(
                f"Per-AC element size mismatch across shards: {sizes_per_ac}. "
                "Cannot gather; shard layout must be uniform."
            )
        per_ac = sizes_per_ac[0] if sizes_per_ac else 0

        main_gpu = self.xp.cuda.runtime.getDevice()
        try:
            with self.xp.cuda.Device(int(target_gpu)):
                gathered = self.xp.zeros(
                    self.acs_total_entries * per_ac, dtype=per_gpu_list[0].dtype
                )
                for i, gpu in enumerate(self.gpus):
                    split = self.gpu_splits[i]
                    if len(split) == 0:
                        continue
                    # Move this shard's buffer onto target_gpu, then scatter
                    # back into AC order in `gathered`.
                    src = self.xp.asarray(per_gpu_list[i])
                    src = src.reshape(len(split), per_ac)
                    for intra_i, ac_i in enumerate(split):
                        gathered[int(ac_i) * per_ac : (int(ac_i) + 1) * per_ac] = (
                            src[intra_i]
                        )
            return gathered
        finally:
            self.xp.cuda.runtime.setDevice(main_gpu)

    def gather_linear_data_arr(self, target_gpu: Optional[int] = None):
        """Return the residual buffer as a single flat array (concatenated across shards).

        For single-GPU runs this returns ``linear_data_arr[0]`` directly (no copy).
        For multi-GPU runs it gathers every shard onto ``target_gpu`` (default
        ``self.gpus[0]``). Use this for legacy callers that expect a single buffer.

        .. warning:: On multi-shard runs the return is a **copy** — mutating
            it does NOT propagate to the shards. Write changes back with
            :meth:`scatter_linear_data_arr`.
        """
        return self._gather_per_gpu_to_single(self.linear_data_arr, target_gpu)

    def gather_linear_psd_arr(self, target_gpu: Optional[int] = None):
        """Return the inverse-PSD buffer as a single flat array (concatenated across shards).

        Same semantics as :meth:`gather_linear_data_arr` for the PSD side,
        including the multi-shard copy caveat (write back with
        :meth:`scatter_linear_psd_arr`).
        """
        return self._gather_per_gpu_to_single(self.linear_psd_arr, target_gpu)

    def _scatter_single_to_per_gpu(self, flat, per_gpu_list: list):
        """Inverse of :meth:`_gather_per_gpu_to_single`: write a flat gathered
        buffer back into the per-shard buffers **in place**.

        Mirrors the gather's three paths exactly:

        - Single-shard: the gather fast path returned the buffer itself, so a
          write-back with that same array is a no-op; a different array is
          copied in.
        - CPU multi-shard: the gather concatenated in shard order — split back
          in shard order.
        - Multi-shard GPU: the gather produced global-AC order — route each
          shard's rows back through host (no P2P), inside the owning device
          context.

        Shard buffers are never reallocated (memory-lifecycle rule), so views
        bound by ``reset_linear_*`` stay valid.
        """
        if len(per_gpu_list) == 1:
            if flat is not per_gpu_list[0]:
                per_gpu_list[0][:] = self.xp.asarray(flat)
            return
        if self.gpus is None:
            offset = 0
            for buf in per_gpu_list:
                buf[:] = flat[offset : offset + buf.size]
                offset += buf.size
            return

        sizes_per_ac = [int(buf.size // max(len(self.gpu_splits[i]), 1))
                        for i, buf in enumerate(per_gpu_list)]
        if sizes_per_ac and len(set(sizes_per_ac)) != 1:
            raise RuntimeError(
                f"Per-AC element size mismatch across shards: {sizes_per_ac}. "
                "Cannot scatter; shard layout must be uniform."
            )
        per_ac = sizes_per_ac[0] if sizes_per_ac else 0

        flat_host = flat.get() if hasattr(flat, "get") else np.asarray(flat)
        flat_host = flat_host.reshape(self.acs_total_entries, per_ac)

        main_gpu = self.xp.cuda.runtime.getDevice()
        try:
            for i, gpu in enumerate(self.gpus):
                split = self.gpu_splits[i]
                if len(split) == 0:
                    continue
                src = flat_host[np.asarray(split, dtype=int)].reshape(-1)
                with self.xp.cuda.Device(int(gpu)):
                    per_gpu_list[i][:] = self.xp.asarray(src)
        finally:
            self.xp.cuda.runtime.setDevice(main_gpu)

    def scatter_linear_data_arr(self, flat):
        """Write a flat gathered residual buffer (from
        :meth:`gather_linear_data_arr`) back into the per-shard buffers."""
        self._scatter_single_to_per_gpu(flat, self.linear_data_arr)

    def scatter_linear_psd_arr(self, flat):
        """Write a flat gathered inverse-PSD buffer (from
        :meth:`gather_linear_psd_arr`) back into the per-shard buffers."""
        self._scatter_single_to_per_gpu(flat, self.linear_psd_arr)

    def gather_data_shaped(self):
        """Return the full ``(num_acs, nchannels, *end_shape)`` residual ndarray.

        Single-GPU: returns ``data_shaped[0]`` directly (a live reshape
        view; no copy). Multi-GPU: gathers every shard's slab into a
        single contiguous ndarray on ``gpus[0]``.

        Use this from legacy callers that expect a single multi-AC array
        (e.g. ``acs.data_shaped[0]`` passed to a kernel that doesn't
        understand the multi-shard list).
        """
        if len(self.linear_data_arr) == 1:
            return self.data_shaped[0]
        return self.data_shaped_view().gather()

    def gather_psd_shaped(self):
        """Same as :meth:`gather_data_shaped` for the inverse-PSD buffer."""
        if len(self.linear_psd_arr) == 1:
            return self.psd_shaped[0]
        return self.psd_shaped_view().gather()

    def data_shaped_view(self, n_bands: Optional[int] = None) -> "BandView":
        """Return a :class:`BandView` over the residual buffer.

        Indexable by global band id (``view[band_inds] = value`` /
        ``view[band_inds]``). Works for single-GPU and multi-GPU runs
        uniformly; on single-GPU the underlying shard is touched directly.
        ``n_bands`` limits the view to the first ``n_bands`` bound slots of
        a fixed-capacity allocation (GB RJ SubBandBuffer); ``None`` keeps
        the full-ACA behavior. See :class:`BandView` for the full op
        contract.
        """
        return BandView(self, kind="data", n_bands=n_bands)

    def psd_shaped_view(self, n_bands: Optional[int] = None) -> "BandView":
        """Return a :class:`BandView` over the inverse-PSD buffer.

        Companion to :meth:`data_shaped_view`. Same semantics.
        """
        return BandView(self, kind="psd", n_bands=n_bands)

    @property
    def data_shaped(self):
        """Per-GPU data buffers reshaped to ``(n_acs_on_gpu, nchannels, *end_shape)``."""
        out = []
        if self.gpus is None:
            for tmp in self.linear_data_arr:
                out.append(tmp.reshape((-1, self.nchannels,) + self.end_shape))
            return out
        main_gpu = self.xp.cuda.runtime.getDevice()
        try:
            for i, tmp in enumerate(self.linear_data_arr):
                with self.xp.cuda.Device(self.gpus[i]):
                    out.append(tmp.reshape((-1, self.nchannels,) + self.end_shape))
        finally:
            self.xp.cuda.runtime.setDevice(main_gpu)
        return out

    @property
    def psd_shaped(self):
        """Per-GPU PSD buffers reshaped to ``(n_acs_on_gpu, *shape_sens, *end_shape)``."""
        out = []
        if self.gpus is None:
            for tmp in self.linear_psd_arr:
                out.append(tmp.reshape((-1,) + self.shape_sens + self.end_shape))
            return out
        main_gpu = self.xp.cuda.runtime.getDevice()
        try:
            for i, tmp in enumerate(self.linear_psd_arr):
                with self.xp.cuda.Device(self.gpus[i]):
                    out.append(tmp.reshape((-1,) + self.shape_sens + self.end_shape))
        finally:
            self.xp.cuda.runtime.setDevice(main_gpu)
        return out
