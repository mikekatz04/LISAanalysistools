"""Python wrapper around C++ STFTDomainWrap / FDDomainWrap for batched
likelihood computation of (d|h) and (h|h) inner products."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from concurrent.futures import ThreadPoolExecutor
import numpy as np

from .domains import FDSettings, STFTSettings, WDMSettings
from .utils.parallelbase import LISAToolsParallelModule

if TYPE_CHECKING:
    # The settings classes (``FDSettings`` / ``STFTSettings`` /
    # ``WDMSettings``) are used for the per-domain ``isinstance`` dispatch
    # in the ``*ComputationGroup`` strategy classes below, which each do
    # their own lazy ``from .domains import ...`` inside the method that
    # needs them (see ``STFT``/``FD``/``WDMComputationGroup``).
    import cupy as cp  # typing-only; runtime cupy use lives in call sites

    from .analysiscontainer import AnalysisContainer, AnalysisContainerArray
    from .detector import Orbits
    from .sensitivity import XYZSensitivityBackend
    from .utils.typing import NDArrayLike, ArrayModule

logger = logging.getLogger(__name__)


class DomainKernelStrategy(LISAToolsParallelModule):
    """Wraps C++ DomainWrap for batched likelihood computation on the AnalysisContainerArray data.

    One instance per GPU split.  Holds references to the linearized arrays
    to prevent GC from invalidating the C++ domain's pointers.

    Args:
        acs : AnalysisContainerArray
            The AnalysisContainerArray containing the data and noise arrays.
        split_index : int, optional
            The index of the GPU split to use from the AnalysisContainerArray. Only used if `acs` is provided. Default is 0.
        tdi_type : str, optional
            The TDI type to use for the likelihood computation. Default is "XYZ". Must be a key in the backend's TDITypeDict.
        force_backend : str, optional
            If provided, forces the use of the specified backend. Must be one of the supported backends for the domain computation. Default is 'cpu'.
    """

    def __init__(
        self,
        acs: AnalysisContainerArray,
        split_index: int = 0,
        tdi_type: str = "XYZ",
        force_backend: str = "cpu",
    ):
        super().__init__(force_backend=force_backend)
        self.tdi_type = tdi_type
        self.d_d = None
        self.split_acs = None

        self.extract_from_acs(acs, split_index)
        self.build_cpp_objects()  # create orbits and sensitivity backend on the correct device.

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

        all_acs = acs.acs.flatten()
        split_container_ids = acs.gpu_splits[split_index]
        self.split_acs: list[AnalysisContainer] = [all_acs[i] for i in split_container_ids]
        self.device = acs.gpus[split_index] if acs.gpus is not None else None

        if self.backend.name.split("_")[-1] == "cpu":
            assert (
                self.device == None
            ), "CPU backend specified but device is not None. Please set `device` to None for CPU usage."
        else:
            assert (
                self.device is not None
            ), "GPU backend specified but `device` is None. Please provide a valid `device` for GPU usage."

    def _owned_cpp_array(self, name: str, arr, dtype):
        """Contiguous ``dtype`` view/copy of ``arr`` OWNED by this group.

        The nanobind domain wraps store RAW POINTERS into the arrays they are
        constructed with (``return_pointer_no_check``). If an argument's dtype
        does not match the binding signature (e.g. the float64
        ``linear_psd_arr`` passed to the complex-invC STFT/FD domains),
        nanobind implicitly converts it into a TEMPORARY that is freed as soon
        as the constructor returns -- the C++ domain is left dangling and
        silently reads whatever later heap allocation reuses that block
        (likelihoods corrupt with a common scalar on (d|h) and (h|h)). Owning
        the correctly-typed buffer here pins it for the domain's lifetime.
        When ``arr`` already matches (contiguous, right dtype) this is the
        SAME object, so in-place updates of the source array stay visible.
        """
        key = "_cpp_owned_" + name
        if not hasattr(self, key):
            setattr(self, key, self.xp.ascontiguousarray(arr, dtype=dtype))
        return getattr(self, key)

    @contextmanager
    def group_device_context(self):
        """Context manager to set the device context for this computation group."""
        if self.device is not None:
            with self.xp.cuda.Device(self.device):
                yield
        else:
            yield

    def build_cpp_objects(self):
        """Builds the C++ domain objects for this computation group. Should be called after extracting arrays from the AnalysisContainerArray."""

        ref_analysis_container = self.split_acs[0] if self.split_acs else None
        if ref_analysis_container is None:
            raise ValueError(
                "No AnalysisContainers found in the specified split. Cannot build C++ domain objects."
            )

        sensitivity_backend = ref_analysis_container.sens_mat

        assert hasattr(
            sensitivity_backend, "orbits"
        ), "Sensitivity matrix must have 'orbits' attribute to build C++ domain objects."
        assert hasattr(
            sensitivity_backend.orbits, "kwargs"
        ), "Sensitivity matrix orbits must have 'kwargs' attribute to build C++ domain objects."
        assert hasattr(
            sensitivity_backend, "kwargs"
        ), "Sensitivity matrix must have 'kwargs' attribute to build C++ domain objects."

        sensitivity_backend_kwargs = sensitivity_backend.kwargs.copy()

        with self.group_device_context():
            # Check if the sensitivity matrix array lives on the same device
            # (or CPU equivalent default). Using split_index == 0 ensures
            # the original instance mapped to device 0 is reused on device 0.
            # if self.split_index == 0:
            #     self._orbits = sensitivity_backend.orbits
            #     self._sensitivity_backend = sensitivity_backend
            # else:
            self._orbits = sensitivity_backend.orbits.__class__(*sensitivity_backend.orbits.args, **sensitivity_backend.orbits.kwargs)

            sensitivity_backend_kwargs["orbits"] = self._orbits
            self._sensitivity_backend = sensitivity_backend.__class__(**sensitivity_backend_kwargs)

            #todo figure what to do for the galaxy modulation.

    def __repr__(self):
        split_index = getattr(self, "split_index", None)
        return f"DomainKernelStrategy with split index {split_index} and TDI type {self.tdi_type}"

    @property
    def xp(self) -> ArrayModule:
        return self.backend.xp

    @property
    def orbits(self) -> Orbits:
        if not hasattr(self, "_orbits"):
            raise ValueError("Orbits have not been created yet. Call build_cpp_objects first.")
        return self._orbits

    @property
    def sensitivity_backend(self) -> XYZSensitivityBackend:
        if not hasattr(self, "_sensitivity_backend"):
            raise ValueError(
                "Sensitivity matrix has not been created yet. Call build_cpp_objects first."
            )
        return self._sensitivity_backend

    @property
    def cpp_domain(self):
        if not hasattr(self, "_cpp_domain"):
            raise ValueError("C++ domain has not been created yet. Call _create_cpp_domain first.")
        return self._cpp_domain

    def _create_cpp_domain(self):
        raise NotImplementedError("Subclasses must implement _create_cpp_domain")

    #: dtype of the (d|h) / (h|h) outputs returned by
    #: ``compute_signal_likelihood_terms``. Complex for FD/STFT;
    #: :class:`WDMComputationGroup` overrides with ``float`` (WDM
    #: coefficients and inverse-noise weights are real).
    _likelihood_terms_dtype = complex

    def _prepare_likelihood_arrays(
        self, num_binaries, start_freqs, data_index, noise_index, start_times=None
    ):
        """Allocate output arrays and ensure contiguous input arrays with correct dtypes."""
        d_h_out = self.xp.zeros(num_binaries, dtype=self._likelihood_terms_dtype)
        h_h_out = self.xp.zeros(num_binaries, dtype=self._likelihood_terms_dtype)
        start_freqs = self.xp.ascontiguousarray(start_freqs, dtype=self.xp.float64)
        data_index = self.xp.ascontiguousarray(data_index, dtype=self.xp.int32)
        noise_index = self.xp.ascontiguousarray(noise_index, dtype=self.xp.int32)
        if start_times is not None:
            start_times = self.xp.ascontiguousarray(start_times, dtype=self.xp.float64)
        return d_h_out, h_h_out, start_freqs, data_index, noise_index, start_times

    def compute_d_d_term(self, out: bool = False, **kwargs):
        """
        Compute the :math:`\\langle d | d\\rangle` term for the containers in this split only.

        Args:
            out : bool, optional
                If True, return the computed :math:`\\langle d | d\\rangle` term. Otherwise, store it in the instance variable `self.d_d`. Default is False.
            **kwargs
                Additional keyword arguments to pass to the `inner_product` method of each AnalysisContainer.

        Returns:
            If `out` is True, returns a double array of shape ``(num_data,)`` containing the :math:`\\langle d | d\\rangle` term for each container in the split. Otherwise, returns None and stores the result in `self.d_d`.

        Notes
        -----
        The result ``self.d_d`` has shape ``(num_data,)`` — one value per
        container in the split, indexed by intra-split index.
        """
        if self.split_acs is None:
            raise ValueError(
                "Split ACs are not set. Cannot compute :math:`\\langle d | d\\rangle` term. "
                "Provide an AnalysisContainerArray to extract_from_acs first."
            )

        d_d = self.xp.zeros(self.num_data, dtype=self.xp.float64)
        for i, ac in enumerate(self.split_acs):
            d_d[i] = ac.inner_product(**kwargs)
        self.d_d = d_d

        if out:
            return self.d_d.copy()

    def compute_noise_term(self, out: bool = False, **kwargs):
        """
        Compute :math:`\\log{\\mathcal{L}}_n = -\\sum \\log{\\vec{S}_n}` term for the containers in this split only.

        Args:
            out : bool, optional
                If True, return the computed noise likelihood term. Otherwise, store it in the instance variable `self.noise_term`. Default is False.
            **kwargs
                Additional keyword arguments to pass to the `inner_product` method of each AnalysisContainer.

        Returns:
            If `out` is True, returns a double array of shape ``(num_data,)`` containing the :math:`\\log{\\mathcal{L}}_n` term for each container in the split. Otherwise, returns None and stores the result in `self.noise_term`.

        Notes
        -----
        The result ``self.noise_term`` has shape ``(num_data,)`` — one value per
        container in the split, indexed by intra-split index.
        """
        if self.split_acs is None:
            raise ValueError(
                "Split ACs are not set. Cannot compute :math:`\\log{\\mathcal{L}}_n` term. "
                "Provide an AnalysisContainerArray to extract_from_acs first."
            )

        noise_term = self.xp.zeros(self.num_data, dtype=self.xp.float64)
        for i, ac in enumerate(self.split_acs):
            noise_term[i] = ac.likelihood(source_only=False, noise_only=True, **kwargs)
        self.noise_term = noise_term

        if out:
            return self.noise_term.copy()

    def compute_signal_likelihood_terms(
        self,
        data_index: NDArrayLike,
        noise_index: NDArrayLike,
        template_vals: NDArrayLike,
        start_freqs: NDArrayLike,
        **kwargs,
    ) -> tuple[NDArrayLike, NDArrayLike]:
        """
        Compute the inner products :math:`\\langle d | h\\rangle` and :math:`\\langle h | h\\rangle` for the input set of binaries.

        Args:
            *args: positional arguments
        """
        raise NotImplementedError(
            "The `compute_signal_likelihood_terms` method must be implemented by subclasses"
        )

    def compute_signal_likelihood(
        self,
        data_index: NDArrayLike,
        noise_index: NDArrayLike,
        template_vals: NDArrayLike,
        start_freqs: NDArrayLike,
        start_times: NDArrayLike = None,
        **kwargs,
    ) -> NDArrayLike:
        """
        Compute the log-likelihood for a batch of binaries.

        Args:
            data_index : int array, shape ``(num_binaries,)``
            noise_index : int array, shape ``(num_binaries,)``
            template_vals : complex array
                Shape ``(num_binaries, num_channels, n_t, n_f)`` for STFT or ``(num_binaries, num_channels, n_f)`` for FD.
            start_freqs : double array, shape ``(num_binaries,)``
            start_times : double array, shape ``(num_binaries,)``, optional
                 Only used for STFT. If not provided, defaults to None.
            **kwargs: additional keyword arguments to pass to the `compute_signal_likelihood_terms` method. Kept for future extensibility.

        Returns:
            like_out : double array, shape ``(num_binaries,)``
        """
        if self.d_d is None:
            raise ValueError(
                "d_d has not been computed. Call compute_d_d_term before compute_signal_likelihood."
            )

        d_h_out, h_h_out = self.compute_signal_likelihood_terms(
            data_index=data_index,
            noise_index=noise_index,
            template_vals=template_vals,
            start_freqs=start_freqs,
            start_times=start_times,
            **kwargs,
        )
        d_d_per_binary = self.d_d[data_index]
        like_out = -1.0 / 2.0 * (d_d_per_binary + h_h_out - 2 * d_h_out).real
        return like_out

    def compute_psd_likelihood(
        self,
        data_index: NDArrayLike,
        noise_index: NDArrayLike,
        *args,
        **kwargs,
    ) -> NDArrayLike:
        """
        Compute the log-likelihood for a batch of binaries using the data stored in this split.
        Refer to the :meth:`compute_log_like` of lisatools.sensitivity.XYZSensitivityBackend.
        Args:
            data_index : int array, shape ``(num_binaries,)``
            noise_index : int array, shape ``(num_binaries,)``. Unused but kept for consistency with the signal likelihood method signature.

        """

        return self.sensitivity_backend.compute_log_like(self.data_arr, data_index, *args, **kwargs)


# Back-compat alias: the per-split strategy was renamed from
# ``BaseDomainComputationGroup`` to :class:`DomainKernelStrategy` when the
# array-level coordinator was absorbed into ``AnalysisContainerArray``.
# Downstream code and tests subclass the old name — keep it working.
BaseDomainComputationGroup = DomainKernelStrategy


class STFTComputationGroup(DomainKernelStrategy):
    """Wraps C++ STFTDomainWrap for batched likelihood computation."""

    def __init__(
        self,
        acs: AnalysisContainerArray,
        split_index: int = 0,
        tdi_type: str = "XYZ",
        force_backend: str = "cpu",
        window_alpha: float = 0.0,
        use_midpoint: bool = False,
        linear_envelope: bool = False,
    ):
        """
        Args:
            window_alpha: Tukey-window parameter for the Fresnel transform
                (``0.0`` = rectangular window).
            use_midpoint: If ``True``, the Fresnel transform anchors each
                linear-chirp expansion at the bin midpoint (``t0 + dt/2``)
                instead of the bin start, which is more accurate for signals
                with frequency curvature. The caller must then supply the
                per-window ``(amp, phase0, f0, fdot0)`` evaluated at the bin
                midpoint; the output convention is unchanged.
            linear_envelope: If ``True``, add the analytic linear-envelope
                first-moment correction to each pixel value. The per-channel
                amplitude slope is derived for free from the estimator's
                ``t +- D`` stencil samples, so this models the within-segment
                TDI amplitude drift the const-envelope kernel otherwise freezes.
                Defaults ``False`` (byte-identical to the const-envelope path).
        """
        from .domains import STFTSettings

        if not isinstance(acs.settings, STFTSettings):
            raise ValueError(
                "settings must be an instance of STFTSettings for STFTComputationGroup."
            )
        super().__init__(acs, split_index, tdi_type, force_backend)

        self.window_alpha = window_alpha
        self.use_midpoint = use_midpoint
        self.linear_envelope = linear_envelope

        with self.group_device_context():
            self._create_cpp_domain()

    def __repr__(self):
        return super().__repr__() + f" with STFT settings: {self.settings}"

    @property
    def domain_args(self):
        return [
            self.settings.NT,
            self.settings.NF_active,
            self.num_channels,
            self.settings.t0,
            self.settings.min_freq,
            self.settings.max_freq,
            self.settings.dt,
            self.settings.df,
            # complex128 buffers OWNED by this group: the wrap keeps raw
            # pointers, and a dtype-converted nanobind temporary would dangle
            # (see _owned_cpp_array).
            self._owned_cpp_array("data", self.data_arr, self.xp.complex128),
            self._owned_cpp_array("invC", self.invC_arr, self.xp.complex128),
            self.num_data,
            self.num_noise,
            self.backend.TDITypeDict[self.tdi_type],
        ]

    def _create_cpp_domain(self):
        self._cpp_domain = self.backend.STFTDomainWrap(*self.domain_args)
        self._cpp_fresnel = self.backend.STFTFresnelWrap(
            *self.domain_args[:8],
            window_alpha=self.window_alpha,
            use_midpoint=self.use_midpoint,
            linear_envelope=self.linear_envelope,
        )

    @property
    def cpp_fresnel(self):
        if not hasattr(self, "_cpp_fresnel"):
            raise ValueError("C++ Fresnel object has not been created yet.")
        return self._cpp_fresnel

    def compute_signal_likelihood_terms(
        self,
        data_index: NDArrayLike,
        noise_index: NDArrayLike,
        template_vals: NDArrayLike,
        start_freqs: NDArrayLike,
        start_times: NDArrayLike,
        **kwargs,
    ) -> tuple[NDArrayLike, NDArrayLike]:
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

        d_h_out, h_h_out, start_freqs, data_index, noise_index, start_times = (
            self._prepare_likelihood_arrays(
                num_binaries, start_freqs, data_index, noise_index, start_times
            )
        )

        run_async = kwargs.get("run_async", False)

        self.cpp_domain.compute_likelihood_terms(
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
            run_async,
        )

        return d_h_out, h_h_out


class FDComputationGroup(DomainKernelStrategy):
    """
    Wraps C++ FDDomainWrap for batched likelihood computation.
    """

    def __init__(
        self,
        acs: AnalysisContainerArray,
        split_index: int = 0,
        tdi_type: str = "XYZ",
        force_backend: str = "cpu",
    ):
        from .domains import FDSettings

        if not isinstance(acs.settings, FDSettings):
            raise ValueError("settings must be an instance of FDSettings for FDComputationGroup.")
        super().__init__(acs, split_index, tdi_type, force_backend)

        with self.group_device_context():
            self._create_cpp_domain()

    @property
    def domain_args(self):
        return [
            self.settings.N_active,
            self.num_channels,
            self.settings.min_freq,
            self.settings.max_freq,
            self.settings.df,
            # complex128 buffers OWNED by this group (see _owned_cpp_array).
            self._owned_cpp_array("data", self.data_arr, self.xp.complex128),
            self._owned_cpp_array("invC", self.invC_arr, self.xp.complex128),
            self.num_data,
            self.num_noise,
            self.backend.TDITypeDict[self.tdi_type],
        ]

    def _create_cpp_domain(self):
        self._cpp_domain = self.backend.FDDomainForStftWrap(*self.domain_args)

    def compute_signal_likelihood_terms(
        self,
        data_index: NDArrayLike,
        noise_index: NDArrayLike,
        template_vals: NDArrayLike,
        start_freqs: NDArrayLike,
        **kwargs,
    ) -> tuple[NDArrayLike, NDArrayLike]:
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

        d_h_out, h_h_out, start_freqs, data_index, noise_index, _ = self._prepare_likelihood_arrays(
            num_binaries, start_freqs, data_index, noise_index
        )

        run_async = kwargs.get("run_async", False)

        self.cpp_domain.compute_likelihood_terms(
            d_h_out,
            h_h_out,
            template_vals.ravel(),
            start_freqs,
            num_binaries,
            data_index,
            noise_index,
            num_freqs,
            run_async,
        )

        return d_h_out, h_h_out


class WDMComputationGroup(DomainKernelStrategy):
    """Wraps C++ WDMDomainWrap for batched likelihood computation (WDM).

    WDM counterpart of :class:`STFTComputationGroup` (2026-06 merge
    follow-up). Differences from the STFT group:

    * Templates, data, and inverse-noise weights are **real** doubles, so
      ``(d|h)`` / ``(h|h)`` come back as ``float64`` arrays.
    * The C++ kernel addresses each binary's rectangular template sub-grid
      by integer ``(m, n)`` start indices on the full WDM grid. This wrapper
      converts the base API's physical ``start_freqs`` / ``start_times`` to
      indices (``m = round(f / layer_df)``, ``n = round(t / layer_dt)``;
      times are measured on the same t0-relative axis as
      ``settings.t_arr``) and validates active-band coverage before calling
      into C++ (the CUDA kernel cannot throw).
    * ``tdi_type`` is passed per call to the C++ side (matching the other
      ``WDMDomain`` helpers) rather than stored on the C++ domain object.
    """

    _likelihood_terms_dtype = float

    def __init__(
        self,
        acs: AnalysisContainerArray,
        split_index: int = 0,
        tdi_type: str = "XYZ",
        force_backend: str = "cpu",
    ):
        from .domains import WDMSettings

        if not isinstance(acs.settings, WDMSettings):
            raise ValueError(
                "settings must be an instance of WDMSettings for WDMComputationGroup."
            )
        super().__init__(acs, split_index, tdi_type, force_backend)

        with self.group_device_context():
            self._create_cpp_domain()

    def __repr__(self):
        return super().__repr__() + f" with WDM settings: {self.settings}"

    @property
    def domain_args(self):
        return [
            self.data_arr,
            self.invC_arr,
            self.settings.layer_df,
            self.settings.layer_dt,
            self.settings.Nf,
            self.settings.Nt,
            self.num_channels,
            self.settings.ind_min_t,
            self.settings.ind_max_t,
            self.settings.ind_min_f,
            self.settings.ind_max_f,
            self.num_data,
            self.num_noise,
        ]

    def _create_cpp_domain(self):
        self._cpp_domain = self.backend.WDMDomainWrap(*self.domain_args)

    def compute_signal_likelihood_terms(
        self,
        data_index: NDArrayLike,
        noise_index: NDArrayLike,
        template_vals: NDArrayLike,
        start_freqs: NDArrayLike,
        start_times: NDArrayLike,
        **kwargs,
    ) -> tuple[NDArrayLike, NDArrayLike]:
        """
        Compute (d|h) and (h|h) for a batch of binaries on the WDM grid.

        Args:
            template_vals : double array
                Shape ``(num_binaries, num_channels, n_m, n_n)`` — real WDM
                coefficients of each template sub-grid, frequency-layer (m)
                axis before time-bin (n) axis, matching the data layout.
            data_index : int array, shape ``(num_binaries,)``
            noise_index : int array, shape ``(num_binaries,)``
            start_freqs : double array, shape ``(num_binaries,)``
                Physical frequency of each sub-grid's first layer
                (``m * layer_df``).
            start_times : double array, shape ``(num_binaries,)``
                t0-relative time of each sub-grid's first bin
                (``n * layer_dt``, same axis as ``settings.t_arr``).

        Returns:
            d_h_out : double array, shape ``(num_binaries,)``
            h_h_out : double array, shape ``(num_binaries,)``
        """
        if start_times is None:
            raise ValueError(
                "start_times is required for WDMComputationGroup (t0-relative "
                "time of each template sub-grid's first WDM bin)."
            )

        num_binaries, _, n_m, n_n = template_vals.shape
        s = self.settings

        # physical -> integer WDM grid indices (rounded: template sub-grids
        # must be aligned with the data's WDM grid).
        start_layer_m = self.xp.rint(
            self.xp.asarray(start_freqs, dtype=self.xp.float64) / s.layer_df
        ).astype(self.xp.int32)
        start_time_n = self.xp.rint(
            self.xp.asarray(start_times, dtype=self.xp.float64) / s.layer_dt
        ).astype(self.xp.int32)

        # Active-band validation lives here because the C++ pixel getters
        # index relative to (ind_min_f, ind_min_t) without bounds checks
        # (the CUDA kernel cannot throw).
        if bool((start_layer_m < s.ind_min_f).any()) or bool(
            (start_layer_m + n_m - 1 > s.ind_max_f).any()
        ):
            raise ValueError(
                f"Template frequency layers must lie inside the active band "
                f"[{s.ind_min_f}, {s.ind_max_f}]."
            )
        if bool((start_time_n < s.ind_min_t).any()) or bool(
            (start_time_n + n_n - 1 > s.ind_max_t).any()
        ):
            raise ValueError(
                f"Template time bins must lie inside the active band "
                f"[{s.ind_min_t}, {s.ind_max_t}]."
            )

        d_h_out, h_h_out, _, data_index, noise_index, _ = self._prepare_likelihood_arrays(
            num_binaries, start_freqs, data_index, noise_index
        )

        template_vals = self.xp.ascontiguousarray(template_vals, dtype=self.xp.float64)

        run_async = kwargs.get("run_async", False)

        self.cpp_domain.compute_likelihood_terms(
            d_h_out,
            h_h_out,
            template_vals.ravel(),
            start_layer_m,
            start_time_n,
            num_binaries,
            data_index,
            noise_index,
            n_m,
            n_n,
            self.backend.TDITypeDict[self.tdi_type],
            run_async,
        )

        return d_h_out, h_h_out


class DomainComputationGroupArray:
    """Deprecated thin alias over :class:`AnalysisContainerArray`.

    The multi-split C++ likelihood coordinator was absorbed into
    ``AnalysisContainerArray``, which now owns the per-split strategy
    workspaces (``acs.cpp_splits`` — the STFT/FD/WDM ``*ComputationGroup``
    objects) and the batched orchestration directly. Drive the ACA methods
    instead: ``acs.cpp_template_likelihood`` / ``acs.cpp_signal_likelihood``
    / ``acs.cpp_psd_likelihood`` / ``acs.cpp_splits``.

    This alias is kept only so external settings files that still construct
    ``DomainComputationGroupArray(acs=acs)`` and hand it to the global-fit
    moves keep working — the moves resolve ``dcga.acs`` at their constructor
    boundary. Constructing it directly emits a :class:`DeprecationWarning`;
    the ACA's own :attr:`~AnalysisContainerArray.cpp_likelihood_backend`
    compat handle builds it with ``_internal=True`` and stays quiet.
    """

    def __init__(self, acs, domain_group_kwargs=None, *, _internal=False):
        if not _internal:
            warnings.warn(
                "DomainComputationGroupArray is deprecated: the C++ likelihood "
                "coordinator now lives on AnalysisContainerArray. Pass the ACA "
                "to the global-fit moves and use acs.cpp_template_likelihood / "
                "acs.cpp_splits directly.",
                DeprecationWarning,
                stacklevel=2,
            )
        self.acs = acs
        if domain_group_kwargs is not None:
            # Resets the strategy cache + stores the new kwargs.
            acs.domain_group_kwargs = domain_group_kwargs
        acs._ensure_cpp_splits()

    @property
    def computation_groups(self):
        return self.acs.cpp_splits

    @property
    def gpus(self):
        return self.acs.gpus

    @property
    def num_splits(self):
        return self.acs.num_splits

    @property
    def xp(self):
        return self.acs.xp
