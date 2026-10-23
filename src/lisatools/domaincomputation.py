"""Batched ``(d|h)`` / ``(h|h)`` likelihood groups over the C++ domain wraps, per GPU split.

* :class:`BaseDomainComputationGroup` and its domain children
  :class:`FDComputationGroup` (``FDDomainWrap``), :class:`STFTComputationGroup`
  (``STFTDomainWrap`` + ``STFTFresnelWrap``) and :class:`WDMComputationGroup`
  (``WDMDomainWrap``): one instance per GPU split of an
  :class:`~lisatools.analysiscontainer.AnalysisContainerArray`. Each holds the
  split's flat data / inverse-noise buffers, a device-local copy of the orbits and
  sensitivity backend, and the C++ domain object that scores template batches.
* :class:`DomainComputationGroupArray` (DCGA): one group per split, plus the
  routing tables and device loops that partition a walker batch across GPUs,
  dispatch per-split work (optionally threaded) and gather the results.

Dispatch to the group class is by ``isinstance`` on the ACA's settings
(:attr:`DomainComputationGroupArray.computation_group_class`), never by string.

In the global fit the DCGA is built lazily, once per run, by
:func:`lisatools.globalfit.recipe.get_shared_dcga` and handed to the GB add/remove,
PSD and MBH moves. It is built only on multi-GPU runs whose sensitivity matrix
carries orbits; ``get_shared_dcga`` returns ``None`` otherwise (including the
stock runs' ``CompositeSensitivityBackend``), and the moves then use their own
shard-aware paths.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from concurrent.futures import ThreadPoolExecutor
import jax
import numpy as np

from .domains import FDSettings, STFTSettings, WDMSettings
from .utils.parallelbase import LISAToolsParallelModule

if TYPE_CHECKING:
    # ``domains.py`` imports the computation-group classes from this
    # module at module-top-level, so pulling ``STFTSettings`` /
    # ``FDSettings`` in eagerly here creates a circular import. They're
    # only needed for type hints and an ``isinstance`` check in
    # ``DomainComputationGroupArray.domain_type`` — the latter does its
    # own lazy import.
    import cupy as cp  # typing-only; runtime cupy use lives in call sites

    from .analysiscontainer import AnalysisContainer, AnalysisContainerArray
    from .detector import Orbits
    from .domains import DomainBase, FDSettings, STFTSettings
    from .sensitivity import XYZSensitivityBackend
    from .utils.typing import NDArrayLike, ArrayModule

logger = logging.getLogger(__name__)


class BaseDomainComputationGroup(LISAToolsParallelModule):
    """Wraps C++ DomainWrap for batched likelihood computation on the AnalysisContainerArray data.

    One instance per GPU split.  Holds references to the linearized arrays
    to prevent GC from invalidating the C++ domain's pointers. Construction
    extracts the split (:meth:`extract_from_acs`) and rebuilds the orbits and
    sensitivity backend on this split's device (:meth:`build_cpp_objects`); the
    domain subclasses then create their C++ domain object. The ``(d|d)`` terms
    must be filled (:meth:`compute_d_d_term`) before
    :meth:`compute_signal_likelihood`.

    Args:
        acs: The AnalysisContainerArray containing the data and noise arrays.
            Its reference sensitivity matrix must expose ``orbits`` (with
            ``kwargs``) and ``kwargs``.
        split_index: The index of the GPU split to use. Default ``0``.
        tdi_type: TDI type for the likelihood, a key of the backend's
            ``TDITypeDict`` (``"XYZ"`` default).
        force_backend: Backend for this group. Default ``"cpu"``; a CPU backend
            requires ``acs.gpus is None``, a GPU backend a device for the split.

    Raises:
        AssertionError: If the backend and the split's device disagree, or the
            sensitivity matrix lacks ``orbits`` / ``kwargs``.
        ValueError: If the split holds no containers.

    Attributes:
        data_arr: The split's flat data buffer (``acs.linear_data_arr[split_index]``).
        invC_arr: The split's flat inverse-noise buffer.
        num_channels (int): Number of TDI channels.
        num_data (int): Containers in this split.
        split_acs (list[AnalysisContainer]): The split's containers.
        device (int or None): GPU id of the split, ``None`` on CPU.
        d_d: ``(num_data,)`` ``(d|d)`` values once computed, else ``None``.
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
        """Extract one split's buffers, containers and device from ``acs``.

        Args:
            acs: The AnalysisContainerArray containing the data and noise arrays.
            split_index: The index of the GPU split to use.

        Raises:
            AssertionError: If a CPU backend meets a GPU split or vice versa.
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

    @contextmanager
    def group_device_context(self):
        """Context manager to set the device context for this computation group."""
        if self.device is not None:
            with self.xp.cuda.Device(self.device):
                yield
        else:
            yield

    def build_cpp_objects(self):
        """Rebuild the orbits and sensitivity backend on this group's device.

        Reconstructs ``orbits.__class__(*orbits.args, **orbits.kwargs)`` and the
        sensitivity backend from its ``kwargs`` inside the split's device context.
        Called after :meth:`extract_from_acs`.

        Raises:
            ValueError: If the split holds no containers.
            AssertionError: If the sensitivity matrix lacks ``orbits`` / ``kwargs``.
        """

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
        return f"BaseDomainComputationGroup with split index {split_index} and TDI type {self.tdi_type}"

    @property
    def xp(self) -> ArrayModule:
        """Array module of this group's backend (numpy or cupy)."""
        return self.backend.xp

    @property
    def orbits(self) -> Orbits:
        """Device-local orbits built by :meth:`build_cpp_objects` (``ValueError`` before)."""
        if not hasattr(self, "_orbits"):
            raise ValueError("Orbits have not been created yet. Call build_cpp_objects first.")
        return self._orbits

    @property
    def sensitivity_backend(self) -> XYZSensitivityBackend:
        """Device-local sensitivity backend built by :meth:`build_cpp_objects`."""
        if not hasattr(self, "_sensitivity_backend"):
            raise ValueError(
                "Sensitivity matrix has not been created yet. Call build_cpp_objects first."
            )
        return self._sensitivity_backend

    @property
    def cpp_domain(self):
        """The C++ domain wrap that scores template batches (``ValueError`` before creation)."""
        if not hasattr(self, "_cpp_domain"):
            raise ValueError("C++ domain has not been created yet. Call _create_cpp_domain first.")
        return self._cpp_domain

    def _create_cpp_domain(self):
        """Create ``self._cpp_domain`` (implemented by the domain subclasses)."""
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
        r"""Compute the :math:`\langle d | d\rangle` term for the containers in this split only.

        Calls ``ac.inner_product(**kwargs)`` per container and stores the result in
        :attr:`d_d`, shape ``(num_data,)``, indexed by intra-split index.

        Args:
            out: If ``True``, also return a copy of the result. Default ``False``.
            **kwargs: Forwarded to each container's ``inner_product``.

        Returns:
            ``xp.ndarray`` or None: ``(num_data,)`` float64 copy of :attr:`d_d` when
            ``out`` is ``True``, else ``None``.

        Raises:
            ValueError: If the split's containers are not set.
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
        r"""Compute the noise term :math:`\log{\mathcal{L}}_n` for this split's containers only.

        Calls ``ac.likelihood(source_only=False, noise_only=True, **kwargs)`` per
        container and stores the result in ``self.noise_term``, shape
        ``(num_data,)``, indexed by intra-split index.

        Args:
            out: If ``True``, also return a copy of the result. Default ``False``.
            **kwargs: Forwarded to each container's ``likelihood``.

        Returns:
            ``xp.ndarray`` or None: ``(num_data,)`` float64 copy when ``out`` is
            ``True``, else ``None``.

        Raises:
            ValueError: If the split's containers are not set.
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
        r"""Compute :math:`\langle d | h\rangle` and :math:`\langle h | h\rangle` for templates.

        Implemented by the domain subclasses (their signatures add
        ``start_times`` for STFT / WDM).

        Args:
            data_index: ``(num_binaries,)`` intra-split data slot of each template.
            noise_index: ``(num_binaries,)`` intra-split noise slot of each template.
            template_vals: Template coefficients (layout per subclass).
            start_freqs: ``(num_binaries,)`` first frequency of each template sub-grid (Hz).
            **kwargs: Subclass options (``run_async``).

        Returns:
            tuple: ``(d_h_out, h_h_out)``, each ``(num_binaries,)``.

        Raises:
            NotImplementedError: Always, on the base class.
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
        """Compute the log-likelihood ``-0.5 * Re(d_d + h_h - 2 d_h)`` for a batch of templates.

        ``d_d`` is the stored :attr:`d_d` of each template's ``data_index``.

        Args:
            data_index: ``(num_binaries,)`` int intra-split data slots.
            noise_index: ``(num_binaries,)`` int intra-split noise slots.
            template_vals: ``(num_binaries, num_channels, n_t, n_f)`` complex for STFT,
                ``(num_binaries, num_channels, n_f)`` complex for FD,
                ``(num_binaries, num_channels, n_m, n_n)`` real for WDM.
            start_freqs: ``(num_binaries,)`` first frequency of each sub-grid (Hz).
            start_times: ``(num_binaries,)`` first time of each sub-grid (s); used by
                STFT and required by WDM. Default ``None``.
            **kwargs: Forwarded to :meth:`compute_signal_likelihood_terms`
                (``run_async``).

        Returns:
            ``xp.ndarray``: ``(num_binaries,)`` float log-likelihoods.

        Raises:
            ValueError: If :meth:`compute_d_d_term` has not run.
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
        """Score noise-model proposals against this split's data.

        Calls ``self.sensitivity_backend.compute_log_like(self.data_arr, data_index,
        *args, **kwargs)`` (see
        :meth:`lisatools.sensitivity.XYZSensitivityBackend.compute_log_like`).

        Args:
            data_index: ``(num_binaries,)`` int intra-split data slots.
            noise_index: Unused; kept for signature parity with
                :meth:`compute_signal_likelihood`.
            *args: Forwarded to ``compute_log_like`` (the noise parameters).
            **kwargs: Forwarded to ``compute_log_like``.

        Returns:
            ``xp.ndarray``: Log-likelihood per proposal, as returned by the backend.
        """

        return self.sensitivity_backend.compute_log_like(self.data_arr, data_index, *args, **kwargs)


class STFTComputationGroup(BaseDomainComputationGroup):
    """Wraps C++ STFTDomainWrap for batched likelihood computation.

    Also builds an ``STFTFresnelWrap`` on the same grid (:attr:`cpp_fresnel`).

    Args:
        acs: AnalysisContainerArray whose ``settings`` is an
            :class:`~lisatools.domains.STFTSettings`.
        split_index: GPU split. Default ``0``.
        tdi_type: TDI type key. Default ``"XYZ"``.
        force_backend: Backend. Default ``"cpu"``.
        window_alpha: Window parameter passed to ``STFTFresnelWrap``. Default ``0.0``.

    Raises:
        ValueError: If ``acs.settings`` is not an STFT domain.
    """

    def __init__(
        self,
        acs: AnalysisContainerArray,
        split_index: int = 0,
        tdi_type: str = "XYZ",
        force_backend: str = "cpu",
        window_alpha: float = 0.0,
    ):
        from .domains import STFTSettings

        if not isinstance(acs.settings, STFTSettings):
            raise ValueError(
                "settings must be an instance of STFTSettings for STFTComputationGroup."
            )
        super().__init__(acs, split_index, tdi_type, force_backend)

        self.window_alpha = window_alpha

        with self.group_device_context():
            self._create_cpp_domain()

    def __repr__(self):
        return super().__repr__() + f" with STFT settings: {self.settings}"

    @property
    def domain_args(self):
        """Positional ``STFTDomainWrap`` constructor arguments (grid, buffers, TDI flavor)."""
        return [
            self.settings.NT,
            self.settings.NF_active,
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

    def _create_cpp_domain(self):
        """Create the ``STFTDomainWrap`` and the ``STFTFresnelWrap`` (first 8 domain args)."""
        self._cpp_domain = self.backend.STFTDomainWrap(*self.domain_args)
        self._cpp_fresnel = self.backend.STFTFresnelWrap(*self.domain_args[:8], window_alpha=self.window_alpha)

    @property
    def cpp_fresnel(self):
        """The ``STFTFresnelWrap`` built with the domain (``ValueError`` before creation)."""
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
        """Compute (d|h) and (h|h) for a batch of templates on the STFT grid.

        Args:
            data_index: ``(num_binaries,)`` int intra-split data slots.
            noise_index: ``(num_binaries,)`` int intra-split noise slots.
            template_vals: Complex ``(num_binaries, num_channels, n_t, n_f)`` sub-grids.
            start_freqs: ``(num_binaries,)`` first frequency of each sub-grid (Hz).
            start_times: ``(num_binaries,)`` first segment time of each sub-grid (s).
            **kwargs: ``run_async`` (bool, default ``False``) is passed to the kernel.

        Returns:
            tuple: ``(d_h_out, h_h_out)``, complex ``(num_binaries,)`` arrays.
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


class FDComputationGroup(BaseDomainComputationGroup):
    """Wraps C++ FDDomainWrap for batched likelihood computation.

    Args:
        acs: AnalysisContainerArray whose ``settings`` is an
            :class:`~lisatools.domains.FDSettings`.
        split_index: GPU split. Default ``0``.
        tdi_type: TDI type key. Default ``"XYZ"``.
        force_backend: Backend. Default ``"cpu"``.

    Raises:
        ValueError: If ``acs.settings`` is not an FD domain.
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
        """Positional ``FDDomainWrap`` constructor arguments (band, buffers, TDI flavor)."""
        return [
            self.settings.N_active,
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

    def _create_cpp_domain(self):
        """Create the ``FDDomainWrap``."""
        self._cpp_domain = self.backend.FDDomainWrap(*self.domain_args)

    def compute_signal_likelihood_terms(
        self,
        data_index: NDArrayLike,
        noise_index: NDArrayLike,
        template_vals: NDArrayLike,
        start_freqs: NDArrayLike,
        **kwargs,
    ) -> tuple[NDArrayLike, NDArrayLike]:
        """Compute (d|h) and (h|h) for a batch of templates on the FD grid.

        Args:
            data_index: ``(num_binaries,)`` int intra-split data slots.
            noise_index: ``(num_binaries,)`` int intra-split noise slots.
            template_vals: Complex ``(num_binaries, num_channels, n_f)`` sub-bands.
            start_freqs: ``(num_binaries,)`` first frequency of each sub-band (Hz).
            **kwargs: ``run_async`` (bool, default ``False``) is passed to the kernel;
                other keys (e.g. ``start_times``) are ignored.

        Returns:
            tuple: ``(d_h_out, h_h_out)``, complex ``(num_binaries,)`` arrays.
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


class WDMComputationGroup(BaseDomainComputationGroup):
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

    Args:
        acs: AnalysisContainerArray whose ``settings`` is a
            :class:`~lisatools.domains.WDMSettings`.
        split_index: GPU split. Default ``0``.
        tdi_type: TDI type key. Default ``"XYZ"``.
        force_backend: Backend. Default ``"cpu"``.

    Raises:
        ValueError: If ``acs.settings`` is not a WDM domain.
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
        """Positional ``WDMDomainWrap`` constructor arguments (buffers, grid, active box)."""
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
        """Create the ``WDMDomainWrap``."""
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
        """Compute (d|h) and (h|h) for a batch of templates on the WDM grid.

        Args:
            data_index: ``(num_binaries,)`` int intra-split data slots.
            noise_index: ``(num_binaries,)`` int intra-split noise slots.
            template_vals: Real ``(num_binaries, num_channels, n_m, n_n)`` WDM
                coefficients of each template sub-grid, frequency-layer (m) axis
                before time-bin (n) axis, matching the data layout.
            start_freqs: ``(num_binaries,)`` physical frequency of each sub-grid's
                first layer (``m * layer_df``, rounded to the nearest layer).
            start_times: ``(num_binaries,)`` grid-start-relative time of each
                sub-grid's first bin (``n * layer_dt``, same axis as
                ``settings.t_arr``, rounded to the nearest pixel). Required.
            **kwargs: ``run_async`` (bool, default ``False``) is passed to the kernel.

        Returns:
            tuple: ``(d_h_out, h_h_out)``, float64 ``(num_binaries,)`` arrays.

        Raises:
            ValueError: If ``start_times`` is ``None`` or a sub-grid leaves the active box.
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
    """Manage one domain computation group per GPU split of an AnalysisContainerArray.

    Builds one group per split (class from :attr:`computation_group_class`)
    inside that split's device context, computes every group's ``(d|d)``, and
    precomputes the routing tables ``ac_to_split`` / ``ac_to_intra`` (global
    container id -> owning split / intra-split index). The per-call pattern of the
    moves is :meth:`unpack_indices` -> :meth:`unpack_coords` ->
    :meth:`place_on_device` -> :meth:`compute_signal_likelihood` (or
    :meth:`compute_psd_likelihood`), which scatters the per-split results back
    into flat batch order.

    In the global fit this object is built once per run by
    :func:`lisatools.globalfit.recipe.get_shared_dcga` (multi-GPU runs with an
    orbits-carrying sensitivity matrix only).

    Args:
        acs: The AnalysisContainerArray holding data, noise and the GPU split
            information (``gpu_splits``, ``split_map``, ``gpus``).
        domain_group_kwargs: Extra keyword arguments for every group constructor
            (e.g. ``tdi_type``). Default ``None``.

    Attributes:
        computation_groups (list[BaseDomainComputationGroup]): One per split.
        ac_to_split (np.ndarray): Split owning each global container id.
        ac_to_intra (np.ndarray): Intra-split index of each global container id.

    Example:
        .. code-block:: python

            dcga = DomainComputationGroupArray(acs)
            pos, d_intra, n_intra = dcga.unpack_indices(data_index)
            args = dcga.unpack_coords(pos, (templates, start_freqs, start_times),
                                      keep_tuple=True)
            d_intra, n_intra, args = dcga.place_on_device((d_intra, n_intra, args))
            ll = dcga.compute_signal_likelihood(pos, d_intra, n_intra, args)
    """

    def __init__(self, acs: AnalysisContainerArray, domain_group_kwargs: dict[str, Any] = None):

        self.acs = acs
        
        self.initialize_computation_groups(domain_group_kwargs)

        # Precomputed routing tables. ``ac_to_split`` maps a global AC id
        # (equivalently walker id in the tempered sampler) to the split that
        # owns it; ``ac_to_intra`` maps the same AC id to its intra-split
        # index, which is what ``BaseDomainComputationGroup.compute_signal_likelihood``
        # expects for ``data_index`` / ``noise_index``. Building these once
        # here replaces per-call ``np.where(...)`` lookups on the hot path.
        self.ac_to_split = np.asarray(self.acs.split_map)
        self.ac_to_intra = np.empty(self.acs.acs_total_entries, dtype=np.int32)
        for split_id, ids in enumerate(self.acs.gpu_splits):
            self.ac_to_intra[ids] = np.arange(len(ids), dtype=np.int32)

        self._thread_pool = ThreadPoolExecutor(max_workers=self.num_splits)

    def __del__(self):
        if hasattr(self, "_thread_pool"):
            self._thread_pool.shutdown(wait=False, cancel_futures=True)

    def initialize_computation_groups(self, domain_group_kwargs: dict[str, Any] = None):
        """Build one computation group per GPU split, then compute all ``(d|d)`` terms.

        Each group is built inside its split's device context with
        ``force_backend="cpu"`` when ``acs.gpus`` is ``None``, else ``"gpu"``.

        Args:
            domain_group_kwargs: Extra keyword arguments for every group constructor.
        """

        force_backend = "cpu" if self.acs.gpus is None else "gpu"
        settings = self.acs.settings
        computation_groups: list[BaseDomainComputationGroup] = []
        for split_index in range(self.num_splits):
            with self.device_context(
                self.acs.gpus[split_index] if self.acs.gpus is not None else None
            ):
                group = self.computation_group_class(
                    acs=self.acs,
                    split_index=split_index,
                    force_backend=force_backend,
                    **(domain_group_kwargs or {})
                )
                computation_groups.append(group)
        self.computation_groups = computation_groups

        self.compute_d_d_terms()

    @property
    def xp(self):
        """Array module (numpy or cupy)."""
        return self.acs.xp

    @property
    def gpus(self):
        """list of GPU IDs corresponding to each split, or None if using CPU."""
        return self.acs.gpus

    @property
    def main_gpu(self):
        """GPU ID of the main device, or None if using CPU."""
        return self.gpus[0] if self.gpus is not None else None

    @property
    def num_splits(self):
        """Number of GPU splits (``len(acs.gpu_splits)``)."""
        return len(self.acs.gpu_splits)
    
    @property
    def thread_pool(self):
        """ThreadPoolExecutor for parallel execution across splits"""
        if not hasattr(self, "_thread_pool"):
            self._thread_pool = ThreadPoolExecutor(max_workers=self.num_splits)
            logger.warning("ThreadPoolExecutor not initialized. Initializing with num_splits workers.")
        return self._thread_pool

    @property
    def computation_group_class(self):
        """The :class:`BaseDomainComputationGroup` subclass paired with the
        ACA's settings. Dispatch is by :class:`DomainSettingsBase` child
        class — never a string flag (sprint rule)."""
        if isinstance(self.acs.settings, STFTSettings):
            return STFTComputationGroup
        if isinstance(self.acs.settings, FDSettings):
            return FDComputationGroup
        if isinstance(self.acs.settings, WDMSettings):
            return WDMComputationGroup
        raise NotImplementedError(
            f"Unsupported domain settings type "
            f"{type(self.acs.settings).__name__} for DomainComputationGroupArray."
        )

    @contextmanager
    def device_context(self, device: int = None):
        """Context manager to set the appropriate device for CPU or GPU backends.

        With ``device=None`` (or a CPU run) it enters
        ``jax.default_device(jax.devices("cpu")[0])`` and yields ``"cpu"``; otherwise
        it enters ``xp.cuda.Device(device)`` and yields ``"gpu: <device>"``.

        Args:
            device: GPU device ID to set the context to, or ``None``.
        """

        if device is None or self.gpus is None:
            # CPU context - set context to CPU if using a GPU backend, otherwise do nothing
            with jax.default_device(jax.devices("cpu")[0]):
                yield "cpu"

        else:
            # device_id = self.gpus.index(device)

            # GPU context - set context to the specified GPU device
            # with jax.default_device(jax.devices("gpu")[device_id]):
            with self.xp.cuda.Device(device):
                yield f"gpu: {device}"

    def restore_main_device(self):
        """Free the current device's memory pool and make the main GPU current again.

        Calls ``xp.get_default_memory_pool().free_all_blocks()``, sets the CUDA
        device to :attr:`main_gpu` and points JAX's default device at it. No-op on
        CPU.

        Note:
            Call after multi-device work so later code runs on the main device.
        """
        if self.main_gpu is not None:
            self.xp.get_default_memory_pool().free_all_blocks()
            self.xp.cuda.runtime.setDevice(self.main_gpu)
            jax.config.update(
                "jax_default_device", jax.devices("gpu")[self.gpus.index(self.main_gpu)]
            )

    def synchronize(self):
        """Synchronize every GPU of the run (``deviceSynchronize`` per device); no-op on CPU.

        Note:
            :meth:`compute_signal_likelihood` / :meth:`compute_psd_likelihood` call
            this before gathering results.
        """
        if self.gpus is not None:
            for device in self.gpus:
                with self.device_context(device):
                    self.xp.cuda.runtime.deviceSynchronize()

    def free_gpu_memory(self):
        """Free the CuPy default memory pool on every GPU of the run; no-op on CPU."""
        if self.gpus is not None:
            for device in self.gpus:
                with self.device_context(device):
                    self.xp.get_default_memory_pool().free_all_blocks()

    def _to_host(self, arr: NDArrayLike) -> np.ndarray:
        """Move an array to the host, ensuring it is a numpy ndarray."""
        return arr.get() if hasattr(arr, "get") else arr

    def unpack_indices(
        self,
        data_index: NDArrayLike,
        noise_index: NDArrayLike = None,
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        """Partition a flat ``(data_index, noise_index)`` batch by split.

        Returns three parallel lists of length ``self.num_splits``:

        * ``positions_per_split[s]``: flat-batch positions routed to split ``s``
          (empty arrays for splits with no matching ACs — callers skip
          via ``len(positions) == 0``).
        * ``data_intra_per_split[s]``: intra-split AC ids for those positions,
          ready to be moved to device via ``self.xp.asarray``.
        * ``noise_intra_per_split[s]``: intra-split noise AC ids.

        This is the single owner of the partition — downstream routines
        (waveform generation, likelihood scatter, residual dispatch) share
        its output instead of recomputing ``np.where(ac_to_split == s)``.

        Args:
            data_index: Flat ``(n,)`` global container ids (host or device).
            noise_index: Flat ``(n,)`` global noise ids; ``None`` reuses ``data_index``.

        Returns:
            tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
            ``(positions_per_split, data_intra_per_split, noise_intra_per_split)``,
            all host arrays.
        """
        data_index_cpu = self._to_host(data_index)
        noise_index_cpu = self._to_host(noise_index) if noise_index is not None else data_index_cpu

        split_of_each = self.ac_to_split[data_index_cpu]

        positions_per_split: list[np.ndarray] = []
        data_intra_per_split: list[np.ndarray] = []
        noise_intra_per_split: list[np.ndarray] = []
        for split_id in range(self.num_splits):
            positions = np.where(split_of_each == split_id)[0]
            positions_per_split.append(positions)
            data_intra_per_split.append(self.ac_to_intra[data_index_cpu[positions]])
            noise_intra_per_split.append(self.ac_to_intra[noise_index_cpu[positions]])

        return positions_per_split, data_intra_per_split, noise_intra_per_split

    def unpack_coords(
        self,
        positions_per_split: list[np.ndarray],
        coords: NDArrayLike | tuple[NDArrayLike],
        keep_tuple: bool = False,
    ) -> list[tuple[np.ndarray] | np.ndarray]:
        """Split coordinate arrays by the per-split positions of :meth:`unpack_indices`.

        Each array is brought to the host and indexed with each split's positions
        (rows taken as Fortran-ordered copies).

        Args:
            positions_per_split: One position array per split.
            coords: An array (numpy or cupy) indexed along its first axis, or a tuple
                of such arrays (e.g. one per branch).
            keep_tuple: If ``True``, a single coordinate array is still returned
                wrapped in a 1-tuple per split.

        Returns:
            list: One entry per split: the array, or a tuple of arrays (multiple
            inputs or ``keep_tuple``), or ``()`` for a split with no positions.
        """

        if not isinstance(coords, tuple):
            coords = (coords,)

        coords_host = tuple(self._to_host(coords_here) for coords_here in coords)

        args_per_group: list = []
        for positions in positions_per_split:
            if len(positions) > 0:
                coords_s = [np.asfortranarray(coords_host[i][positions]) for i in range(len(coords_host))]

                args_per_group.append(tuple(coords_s) if len(coords_s) > 1 or keep_tuple else coords_s[0])
            else:
                args_per_group.append(())

        return args_per_group

    def place_on_device(
        self, items: tuple[list[np.ndarray | tuple[np.ndarray]]]
    ) -> tuple[list[NDArrayLike | tuple[NDArrayLike]]]:
        """Copy per-split host arrays onto each split's device.

        Args:
            items: A tuple of per-split lists (e.g. ``(data_intra_per_split,
                coords_per_split)``); each list entry is a numpy array, a tuple of
                numpy arrays, or ``()``.

        Returns:
            tuple: The same structure with every array copied by
            ``xp.asarray(..., copy=True)`` inside its split's device context.

        Raises:
            ValueError: If an entry is neither a numpy array nor a tuple.
        """
        num_items = len(items)

        device_items: list[list] = [[] for _ in range(num_items)]

        for i, device in enumerate(
            self.gpus if self.gpus is not None else [None] * self.num_splits
        ):
            with self.device_context(device):
                for item_id, item_per_split in enumerate(items):
                    entry = item_per_split[i]
                    if isinstance(entry, np.ndarray):
                        device_items[item_id].append(self.xp.asarray(entry, copy=True))
                    elif isinstance(entry, tuple):
                        if len(entry) == 0:
                            device_items[item_id].append(())
                        else:
                            device_items[item_id].append(tuple(self.xp.asarray(arr, copy=True) for arr in entry))
                    else:
                        raise ValueError("Each entry in items must be either a numpy array or a tuple of numpy arrays.")
        return tuple(device_items)

    def _loop_operation(
        self,
        operation: Callable | list[Callable],
        operation_args_per_split: list[tuple] = None,
        operation_kwargs: dict | list[dict] = None,
        aggregate_fn: Callable = None,
        positions_per_split: list[np.ndarray] = None,
        run_threaded: bool = False,
    ) -> list | Any:
        """General loop to perform an operation across splits, with optional result aggregation.

        Args:
            operation: A callable or a list of callables (one per split), each run
                inside its split's device context with that split's args / kwargs.
            operation_args_per_split: Optional list (one per split) of positional-arg
                tuples. The caller places the arguments on the right device (e.g. via
                :meth:`place_on_device`). Default: empty tuples.
            operation_kwargs: Optional dict (shared) or list of dicts (one per split)
                of keyword arguments.
            aggregate_fn: Optional callable applied to the list of per-split results.
            positions_per_split: Optional per-split position arrays; a split with no
                positions is skipped.
            run_threaded: If True, dispatch operations across splits using the
                ThreadPoolExecutor. Default is False.

        Returns:
            A list of results from each split (``None`` for a skipped empty split) if
            aggregate_fn is not provided, or the aggregated result if aggregate_fn is
            provided.

        Raises:
            ValueError: If a per-split list does not have one entry per split.
        """
        if operation_args_per_split is None:
            operation_args_per_split = [()] * self.num_splits
        if operation_kwargs is None:
            operation_kwargs = {}

        if len(operation_args_per_split) != self.num_splits:
            raise ValueError("Length of operation_args_per_split must match the number of splits.")

        if isinstance(operation, list):
            if len(operation) != self.num_splits:
                raise ValueError(
                    "If operation is a list, its length must match the number of splits."
                )
            operations = operation
        else:
            operations = [operation] * self.num_splits

        if isinstance(operation_kwargs, list):
            if len(operation_kwargs) != self.num_splits:
                raise ValueError(
                    "If operation_kwargs is a list, its length must match the number of splits."
                )
            operation_kwargs_per_split = operation_kwargs
        else:
            operation_kwargs_per_split = [operation_kwargs] * self.num_splits

        devices = self.gpus if self.gpus is not None else [None] * self.num_splits

        def _run_split_operation(i, device):
            if positions_per_split is not None and len(positions_per_split[i]) == 0:
                return None
            with self.device_context(device):
                return operations[i](*operation_args_per_split[i], **operation_kwargs_per_split[i])
        
        if run_threaded:
            futures = [self.thread_pool.submit(_run_split_operation, i, device) for i, device in enumerate(devices)]
            outputs = [future.result() for future in futures]
        else:
            outputs = []
            for i, device in enumerate(
                devices
            ):
                out_i = _run_split_operation(i, device)
                outputs.append(out_i)

        if aggregate_fn is not None:
            return aggregate_fn(outputs)
        return outputs

    def compute_d_d_terms(self, out: bool = False, **kwargs):
        """Compute (d|d) terms for all splits and store them in the respective computation groups.

        Args:
            out: If ``True``, return the per-split arrays.
            **kwargs: Forwarded to :meth:`BaseDomainComputationGroup.compute_d_d_term`.

        Returns:
            list or None: One ``(num_data,)`` array per split when ``out`` is ``True``.
        """

        operations = [group.compute_d_d_term for group in self.computation_groups]

        list_out = self._loop_operation(
            operation=operations, operation_kwargs={"out": out, **kwargs}
        )
        if out:
            return list_out

    def compute_noise_terms(self, out: bool = False, **kwargs):
        """Compute noise likelihood terms for all splits and store them in each split's group.

        Args:
            out: If ``True``, return the per-split arrays.
            **kwargs: Forwarded to :meth:`BaseDomainComputationGroup.compute_noise_term`.

        Returns:
            list or None: One ``(num_data,)`` array per split when ``out`` is ``True``.
        """

        operations = [group.compute_noise_term for group in self.computation_groups]

        list_out = self._loop_operation(
            operation=operations, operation_kwargs={"out": out, **kwargs}
        )
        if out:
            return list_out

    def _compute_group_likelihood(
        self,
        positions_per_split: list[np.ndarray],
        data_intra_per_split: list[NDArrayLike],
        noise_intra_per_split: list[NDArrayLike],
        operations: list[Callable],
        likelihood_args_per_split: list[tuple],
        likelihood_kwargs: dict | list[dict] = None, 
        run_threaded: bool = False,
    ):
        """Run one likelihood op per split and scatter the results into flat batch order.

        Each split's op is called as ``op(data_intra, noise_intra, *likelihood_args)``
        (arrays already on the split's device), the devices are synchronized, and
        split results land at their ``positions_per_split`` in a host float64 array
        prefilled with ``-1e300`` (a warning is logged if any slot stays unfilled).

        Returns:
            np.ndarray: ``(n,)`` log-likelihoods in the order of the original batch.
        """
        # todo I am assuming everything has been placed on the correct device. use self.place_on_device if not.

        operation_args_per_split = []
        for split_id in range(self.num_splits):
            likelihood_args = likelihood_args_per_split[split_id]
            args_i = (
                data_intra_per_split[split_id],
                noise_intra_per_split[split_id],
                *(likelihood_args if likelihood_args is not None else ()),
            )
            operation_args_per_split.append(args_i)

        all_logls = self._loop_operation(
            operation=operations, operation_args_per_split=operation_args_per_split, operation_kwargs=likelihood_kwargs, positions_per_split=positions_per_split, run_threaded=run_threaded
        )

        # now synchronize
        self.synchronize()
        n_data = sum(len(p) for p in positions_per_split)
    
        output = np.full(n_data, -1e300, dtype=np.float64)
        for split_id, positions in enumerate(positions_per_split):
            if len(positions) > 0:
                output[positions] = self._to_host(all_logls[split_id])

        if np.any(output == -1e300):
            logger.warning("Some positions were not filled in the output array. This may indicate an issue with the likelihood computation or aggregation.")
        return output

    def compute_psd_likelihood(
        self,
        positions_per_split: list[np.ndarray],
        data_intra_per_split: list[NDArrayLike],
        noise_intra_per_split: list[NDArrayLike],
        likelihood_args_per_split: list[tuple],
        likelihood_kwargs: dict | list[dict] = None,
        run_threaded: bool = False,
    ):
        """Compute PSD likelihood for each split and aggregate results.

        Per split, calls :meth:`BaseDomainComputationGroup.compute_psd_likelihood`
        (see :meth:`_compute_group_likelihood` for the scatter).

        Args:
            positions_per_split: Flat-batch positions per split (:meth:`unpack_indices`).
            data_intra_per_split: Device-resident intra-split data ids per split.
            noise_intra_per_split: Device-resident intra-split noise ids per split.
            likelihood_args_per_split: Per-split tuple of extra positional args.
            likelihood_kwargs: Keyword arguments (one dict, or one per split).
            run_threaded: Dispatch splits on the thread pool. Default ``False``.

        Returns:
            np.ndarray: ``(n,)`` log-likelihoods in batch order.
        """
        operations = [group.compute_psd_likelihood for group in self.computation_groups]
        return self._compute_group_likelihood(
            positions_per_split,
            data_intra_per_split,
            noise_intra_per_split,
            operations,
            likelihood_args_per_split,
            likelihood_kwargs=likelihood_kwargs,
            run_threaded=run_threaded,
        )

    def compute_signal_likelihood(
        self,
        positions_per_split: list[np.ndarray],
        data_intra_per_split: list[NDArrayLike],
        noise_intra_per_split: list[NDArrayLike],
        likelihood_args_per_split: list[tuple],
        likelihood_kwargs: dict | list[dict] = None,
        run_threaded: bool = False,
    ):
        """Compute signal likelihood for each split and aggregate results.

        Per split, calls :meth:`BaseDomainComputationGroup.compute_signal_likelihood`
        with ``likelihood_args_per_split[s] = (template_vals, start_freqs[,
        start_times])``.

        Args:
            positions_per_split: Flat-batch positions per split (:meth:`unpack_indices`).
            data_intra_per_split: Device-resident intra-split data ids per split.
            noise_intra_per_split: Device-resident intra-split noise ids per split.
            likelihood_args_per_split: Per-split tuple of extra positional args.
            likelihood_kwargs: Keyword arguments (one dict, or one per split).
            run_threaded: Dispatch splits on the thread pool. Default ``False``.

        Returns:
            np.ndarray: ``(n,)`` log-likelihoods in batch order.
        """
        operations = [group.compute_signal_likelihood for group in self.computation_groups]
        return self._compute_group_likelihood(
            positions_per_split,
            data_intra_per_split,
            noise_intra_per_split,
            operations,
            likelihood_args_per_split,
            likelihood_kwargs=likelihood_kwargs,
            run_threaded=run_threaded,
        )