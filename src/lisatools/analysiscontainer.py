"""High-level analysis containers combining data, sensitivity, and signal generation."""

from __future__ import annotations

import math
import warnings
from abc import ABC
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

        data_res_arr: Deprecated alias of ``data`` (kept for backward compatibility).

    """

    def __init__(
        self,
        data: Union[DomainBase, DataResidualArray, None] = None,
        sens_mat: Union[SensitivityMatrixBase, type, None] = None,
        signal_gen: Optional[SignalGenSpec] = None,
        sens_mat_kwargs: Optional[dict] = None,
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

    def build_template(
        self,
        params: Union[tuple, list, np.ndarray, Mapping[str, Any]],
        waveform_kwargs: Union[dict, Mapping[str, dict], None] = None,
        per_model_per_signal: bool = False,
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

        Returns:
            A :class:`DomainBase` template (or a structured dict when
            ``per_model_per_signal=True`` in multi-model mode).
        """
        waveform_kwargs = waveform_kwargs or {}

        if not self.is_multi_model:
            if isinstance(params, Mapping):
                raise TypeError(
                    "signal_gen is a single callable, but params is a dict. "
                    "Either pass tuple/array params, or configure signal_gen "
                    "as a dict of per-model generators."
                )
            return self._call_single_model(
                self._signal_gen, params,
                waveform_kwargs if isinstance(waveform_kwargs, dict) else {},
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
                wf = per_model_kwargs.get(name, {})
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
            wf = per_model_kwargs.get(name, {})
            per_model = self._call_single_model(fn, p, wf)
            if combined is None:
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

    def add_signal_to_data(self, template, sign: int = +1) -> DomainBase:
        """Add ``template`` to ``self.data`` (in-place, domain-aware).

        ``template`` may be a :class:`DomainBase`, a :class:`DataResidualArray`,
        a tuple/array of parameters, or a multi-model params ``dict``. In the
        latter cases :meth:`build_template` is used to assemble the combined
        template from :attr:`signal_gen`.
        """
        if isinstance(template, (DomainBase, DataResidualArray)):
            tmpl = self._to_data_domain(template)
        else:
            tmpl = self.build_template(template)
        self._data.add_signal(tmpl, sign=sign)
        return self._data

    def subtract_signal_from_data(self, template) -> DomainBase:
        """Subtract ``template`` from ``self.data`` (in-place, see :meth:`add_signal_to_data`)."""
        return self.add_signal_to_data(template, sign=-1)

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
        source_only: bool = False,
        waveform_kwargs: Optional[Union[dict, Mapping[str, dict]]] = None,
        transform_fn: Optional[TransformContainer] = None,
        signal_gen: Optional[SignalGenSpec] = None,
        per_model_per_signal: bool = False,
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
                (leave out noise part).
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
            **kwargs: Forwarded to :func:`lisatools.diagnostic.inner_product`.

        Returns:
            Likelihood / inner-product / SNR value, or a structured dict
            (per_model_per_signal mode).
        """
        # Temporarily swap in a per-call signal_gen if provided.
        prev_gen = self._signal_gen if hasattr(self, "_signal_gen") else None
        if signal_gen is not None:
            self.signal_gen = signal_gen
        try:
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
                template_or_struct = self.build_template(
                    params,
                    waveform_kwargs=waveform_kwargs or {},
                    per_model_per_signal=per_model_per_signal,
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
                template = self.build_template(
                    args_in,
                    waveform_kwargs=waveform_kwargs or {},
                )
        finally:
            if signal_gen is not None:
                if prev_gen is None:
                    del self._signal_gen
                else:
                    self._signal_gen = prev_gen

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
        source_only: bool = False,
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

        if not use_vmap:
            out = np.zeros(x.shape[0])
            for i in range(x.shape[0]):
                input_vals = tuple(x[i]) + tuple(args)
                out[i] = self.calculate_signal_likelihood(
                    *input_vals, source_only=source_only, **kwargs
                )
            return out

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
    ) -> None:

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

        self.gpus = gpus
        if gpus is not None:
            if isinstance(gpus, list):
                if len(gpus) > 1:
                    raise NotImplementedError
                self.xp.cuda.runtime.setDevice(gpus[0])
            elif isinstance(gpus, int):
                self.xp.cuda.runtime.setDevice(gpus)

        ac_tmp = acs.flatten()[0]
        self.shape_sens = shape_sens = ac_tmp.sens_mat.shape[: -len(ac_tmp.sens_mat.data_shape)]

        if isinstance(ac_tmp.sens_mat.basis_settings, domains.WDMSettings):
            self.data_dtype = float
        else:
            self.data_dtype = complex

        self.noise_dtype = float if not complex_psd else complex
            
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
            if gpus is not None:
                self.gpu_map[split] = gpus[i]
            else:
                self.gpu_map[split] = 0
            self.split_map[split] = i
            self.linear_data_arr.append(
                self.xp.zeros(
                    self.data_length * self.nchannels * len(split),
                    dtype=self.data_dtype,
                )
            )
            self.linear_psd_arr.append(
                self.xp.zeros(self.data_length * np.prod(shape_sens) * len(split), dtype=self.noise_dtype)
            )

        self.num_acs = len(acs.flatten())
        self.reset_linear_data_arr()
        self.reset_linear_psd_arr()

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
            self.linear_psd_arr[split][start_index:end_index] = self.xp.asarray(
                ac.sens_mat.invC.flatten()
            )
            ac.sens_mat.invC = self.linear_psd_arr[split][start_index:end_index].reshape(
                self.shape_sens + self.end_shape
            )

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
        """Frequency array of the first analysis container."""
        return self.acs[0].data_res_arr.f_arr

    @property
    def df(self):
        """Frequency spacing inferred from :attr:`f_arr`."""
        return self.f_arr[1] - self.f_arr[0]

    def __len__(self) -> int:
        return len(self.acs)

    def _loop_operation(self, operation: str, **kwargs: Any) -> np.ndarray:
        """Apply ``operation`` to every container and stack the per-container results."""
        for i, ac in enumerate(self.acs.flatten()):
            _tmp = getattr(ac, operation)
            if callable(_tmp):
                _tmp_output = _tmp(**kwargs)
            else:
                # must be property or attribute
                _tmp_output = _tmp

            if i == 0:
                _type = _tmp_output.dtype if hasattr(_tmp_output, "dtype") else type(_tmp_output)
                output = np.zeros(self.acs_total_entries, dtype=_type)

            output[i] = _tmp_output

        return output.reshape(self.acs_shape)

    @property
    def start_freq_ind(self):
        """Per-container ``start_freq_ind`` reshaped to :attr:`acs_shape`."""
        return self._loop_operation("start_freq_ind")

    @property
    def start_freq_layer_inds(self):
        """Per-container WDM ``start_freq_layer_ind`` reshaped to :attr:`acs_shape`.

        Each AC's view is ``Nf_active`` frequency layers wide centred on
        its layer of interest, so this array varies across containers
        (analogous to :attr:`start_freq_ind` in the FD case).
        """
        return self._loop_operation("start_freq_layer_ind")

    @property
    def start_time_layer_inds(self):
        """Per-container WDM ``start_time_layer_ind`` reshaped to :attr:`acs_shape`.

        Every container in the array covers the same active time range,
        so every entry is the same value. Kept as an array for API
        symmetry with :attr:`start_freq_layer_inds`.
        """
        return self._loop_operation("start_time_layer_ind")

    @property
    def layer_df(self):
        """WDM layer frequency spacing of the first container (shared across the array)."""
        return self.acs[0].layer_df

    @property
    def layer_dt(self):
        """WDM layer time spacing of the first container (shared across the array)."""
        return self.acs[0].layer_dt

    def inner_product(self, **kwargs):
        """Per-container :meth:`AnalysisContainer.inner_product` reshaped to :attr:`acs_shape`."""
        return self._loop_operation("inner_product", **kwargs)

    def likelihood(self, **kwargs):
        """Per-container :meth:`AnalysisContainer.likelihood` reshaped to :attr:`acs_shape`."""
        return self._loop_operation("likelihood", **kwargs)

    def snr(self, **kwargs):
        """Per-container :meth:`AnalysisContainer.snr` reshaped to :attr:`acs_shape`."""
        return self._loop_operation("snr", **kwargs)

    def __getitem__(self, index: Any) -> np.ndarray[AnalysisContainer]:
        return self.acs[index]

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
            for i, (di, si) in enumerate(zip(data_index, start_index)):
                self.acs.flatten()[di].data[:, si : si + template_length] += (
                    sign * templates[i]
                )
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
        for i, di in enumerate(data_index):
            signal = item_list[i]
            # Delegate to DomainBase.add_signal, which encapsulates the
            # per-domain partial-overlap handling (STFT/FD/TD/WDM).
            acs_flat[di].data.add_signal(signal, sign=sign)

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
        """Per-GPU data buffers reshaped to ``(n_acs_on_gpu, nchannels, *end_shape)``."""
        out = []
        for i, tmp in enumerate(self.linear_data_arr):
            if self.gpus is not None:
                self.xp.cuda.runtime.setDevice(self.gpus[i])
            out.append(tmp.reshape((-1, self.nchannels,) + self.end_shape))
        return out

    @property
    def psd_shaped(self):
        """Per-GPU PSD buffers reshaped to ``(n_acs_on_gpu, *shape_sens, *end_shape)``."""
        out = []
        for i, tmp in enumerate(self.linear_psd_arr):
            if self.gpus is not None:
                self.xp.cuda.runtime.setDevice(self.gpus[i])
            out.append(tmp.reshape((-1,) + self.shape_sens + self.end_shape))
        return out
