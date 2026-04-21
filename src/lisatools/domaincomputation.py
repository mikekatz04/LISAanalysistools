"""Python wrapper around C++ STFTDomainWrap / FDDomainWrap for batched
likelihood computation of (d|h) and (h|h) inner products."""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, List, Tuple

import jax
import numpy as np

from .domains import FDSettings, STFTSettings
from .utils.parallelbase import LISAToolsParallelModule

if TYPE_CHECKING:
    # ``domains.py`` imports the computation-group classes from this
    # module at module-top-level, so pulling ``STFTSettings`` /
    # ``FDSettings`` in eagerly here creates a circular import. They're
    # only needed for type hints and an ``isinstance`` check in
    # ``DomainComputationGroupArray.domain_type`` — the latter does its
    # own lazy import.
    import cupy as cp  # typing-only; runtime cupy use lives in call sites

    from .analysiscontainer import AnalysisContainerArray
    from .domains import FDSettings, STFTSettings, DomainBase

logger = logging.getLogger(__name__)


class BaseDomainComputationGroup(LISAToolsParallelModule):
    """Wraps C++ DomainWrap for batched likelihood computation on the AnalysisContainerArray data.

    One instance per GPU split.  Holds references to the linearized arrays
    to prevent GC from invalidating the C++ domain's pointers.

    Args:
        acs : AnalysisContainerArray, optional
            The AnalysisContainerArray containing the data and noise arrays. If not provided, the necessary arrays and Args: must be provided directly.
        split_index : int, optional
            The index of the GPU split to use from the AnalysisContainerArray. Only used if `acs` is provided. Default is 0.
        data_arr : np.ndarray, optional
            The linearized data array for the current split. Only used if `acs` is not provided.
        invC_arr : np.ndarray, optional
            The linearized inverse noise PSD array for the current split. Only used if `acs` is not provided.
        num_data : int, optional
            The number of data points for the current split. Only used if `acs` is not provided.
        num_noise : int, optional
            The number of noise points for the current split. Only used if `acs` is not provided.
        num_channels : int, optional
            The number of channels for the current split. Only used if `acs` is not provided.
        settings : STFTSettings or FDSettings, optional
            The settings for the domain computation. Must be an instance of STFTSettings for STFTComputationGroup or FDSettings for FDComputationGroup. Only used if `acs` is not provided.
        tdi_type : str, optional
            The TDI type to use for the likelihood computation. Default is "XYZ". Must be a key in the backend's TDITypeDict.
        force_backend : str, optional
            If provided, forces the use of the specified backend. Must be one of the supported backends for the domain computation. Default is 'cpu'.
        device_id : int, optional
            If using a GPU backend and multiple devices are available, specifies the device ID to use. Only used if `force_backend` is a GPU backend. Defaults to None for `cpu` usage.
    """

    def __init__(
        self,
        acs: AnalysisContainerArray = None,
        split_index: int = 0,
        data_arr: np.ndarray | None = None,
        invC_arr: np.ndarray | None = None,
        num_data: int = None,
        num_noise: int = None,
        num_channels: int = None,
        settings: STFTSettings | FDSettings = None,
        tdi_type: str = "XYZ",
        force_backend: str = "cpu",
        device_id: int = None,
    ):
        super().__init__(force_backend=force_backend)
        self.tdi_type = tdi_type
        self.d_d = None
        self.split_acs = None

        if acs is not None:
            self.extract_from_acs(acs, split_index)
        else:
            # ``device_id`` is intentionally NOT in this list — it is
            # legitimately ``None`` on the CPU path. The GPU-backend
            # branch below raises its own assertion for the ``None``
            # case, so we don't need to pre-check it here.
            for param in [
                data_arr,
                invC_arr,
                num_data,
                num_noise,
                num_channels,
                settings,
            ]:
                if param is None:
                    raise ValueError("All Args: must be provided if acs is not given.")
            # Keep references alive so the C++ pointers remain valid. We do not copy to always point to the same memory.
            self.data_arr = data_arr
            self.invC_arr = invC_arr
            self.num_channels = num_channels
            self.num_data = num_data
            self.num_noise = num_noise
            self.settings = settings
            self.device_id = device_id

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

        # Store references to the containers in this split for (d|d) computation.
        # Do NOT wrap them in a new AnalysisContainerArray — that would call
        # reset_linear_data_arr / reset_linear_psd_arr, rebinding each AC's
        # internal ._arr to a new buffer and breaking the C++ pointer contract.
        all_acs = acs.acs.flatten()
        split_container_ids = acs.gpu_splits[split_index]
        self.split_acs = [all_acs[i] for i in split_container_ids]
        self.device_id = acs.gpus[split_index] if acs.gpus is not None else None

        if self.backend.name.split("_")[-1] == "cpu":
            assert (
                self.device_id == None
            ), "CPU backend specified but device_id is not None. Please set device_id to None for CPU usage."
        else:
            assert (
                self.device_id is not None
            ), "GPU backend specified but device_id is None. Please provide a valid device_id for GPU usage."

    def __repr__(self):
        split_index = getattr(self, "split_index", None)
        return f"BaseDomainComputationGroup with split index {split_index} and TDI type {self.tdi_type}"

    @property
    def xp(self):
        return self.backend.xp

    @property
    def cpp_domain(self):
        if not hasattr(self, "_cpp_domain"):
            self._cpp_domain = self._create_cpp_domain()
        return self._cpp_domain

    def _create_cpp_domain(self):
        raise NotImplementedError("Subclasses must implement _create_cpp_domain")

    def _prepare_likelihood_arrays(
        self, num_binaries, start_freqs, data_index, noise_index, start_times=None
    ):
        """Allocate output arrays and ensure contiguous input arrays with correct dtypes."""
        d_h_out = self.xp.zeros(num_binaries, dtype=self.xp.complex128)
        h_h_out = self.xp.zeros(num_binaries, dtype=self.xp.complex128)
        start_freqs = self.xp.ascontiguousarray(start_freqs, dtype=self.xp.float64)
        data_index = self.xp.ascontiguousarray(data_index, dtype=self.xp.int32)
        noise_index = self.xp.ascontiguousarray(noise_index, dtype=self.xp.int32)
        if start_times is not None:
            start_times = self.xp.ascontiguousarray(start_times, dtype=self.xp.float64)
        return d_h_out, h_h_out, start_freqs, data_index, noise_index, start_times

    def compute_d_d_term(self, out: bool=False, **kwargs):
        """
        Compute (d|d) term for the containers in this split only.

        Args:
            out : bool, optional
                If True, return the computed (d|d) term. Otherwise, store it in the instance variable `self.d_d`. Default is False.
            **kwargs
                Additional keyword arguments to pass to the `inner_product` method of each AnalysisContainer.

        Returns:
            If `out` is True, returns a double array of shape ``(num_data,)`` containing the (d|d) term for each container in the split. Otherwise, returns None and stores the result in `self.d_d`.

        Notes
        -----
        The result ``self.d_d`` has shape ``(num_data,)`` — one value per
        container in the split, indexed by intra-split index.
        """
        if self.split_acs is None:
            raise ValueError(
                "Split ACs are not set. Cannot compute (d|d) term. "
                "Provide an AnalysisContainerArray to extract_from_acs first."
            )

        d_d = self.xp.zeros(self.num_data, dtype=self.xp.float64)
        for i, ac in enumerate(self.split_acs):
            d_d[i] = ac.inner_product(**kwargs)
        self.d_d = d_d

        if out:
            return self.d_d.copy()

    def compute_likelihood_terms(
        self,
        data_index: np.ndarray | cp.ndarray,
        noise_index: np.ndarray | cp.ndarray,
        template_vals: np.ndarray | cp.ndarray,
        start_freqs: np.ndarray | cp.ndarray,
        **kwargs,
    ) -> tuple[np.ndarray | cp.ndarray, np.ndarray | cp.ndarray]:
        """
        Compute the inner products :math:`\\langle d | h\\rangle` and :math:`\\langle h | h\\rangle` for the input set of binaries.

        Args:
            *args: positional arguments
        """
        raise NotImplementedError(
            "The `compute_likelihood_terms` method must be implemented by subclasses"
        )

    def compute_likelihood(
        self,
        data_index: np.ndarray | cp.ndarray,
        noise_index: np.ndarray | cp.ndarray,
        template_vals: np.ndarray | cp.ndarray,
        start_freqs: np.ndarray | cp.ndarray,
        start_times: np.ndarray | cp.ndarray = None,
        **kwargs,
    ) -> np.ndarray | cp.ndarray:
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
            **kwargs: additional keyword arguments to pass to the `compute_likelihood_terms` method. Kept for future extensibility.

        Returns:
            like_out : double array, shape ``(num_binaries,)``
        """
        if self.d_d is None:
            raise ValueError(
                "d_d has not been computed. Call compute_d_d_term before compute_likelihood."
            )

        d_h_out, h_h_out = self.compute_likelihood_terms(
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


class STFTComputationGroup(BaseDomainComputationGroup):
    """Wraps C++ STFTDomainWrap for batched likelihood computation."""

    def __init__(self, *args, settings: STFTSettings = None, **kwargs):
        from .domains import STFTSettings

        if settings is None or not isinstance(settings, STFTSettings):
            raise ValueError(
                "settings must be an instance of STFTSettings for STFTComputationGroup."
            )
        super().__init__(*args, settings=settings, **kwargs)

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
            self.data_arr,
            self.invC_arr,
            self.num_data,
            self.num_noise,
            self.backend.TDITypeDict[self.tdi_type],
        ]

    def _create_cpp_domain(self):
        domain = self.backend.STFTDomainWrap(*self.domain_args)
        logger.debug("Initialized STFTDomainWrap with arguments: %s", self.domain_args)
        return domain

    @property
    def cpp_fresnel(self):
        if not hasattr(self, "_cpp_fresnel"):
            self._cpp_fresnel = self.backend.STFTFresnelWrap(*self.domain_args[:8])
        return self._cpp_fresnel

    def compute_likelihood_terms(
        self,
        data_index: np.ndarray | cp.ndarray,
        noise_index: np.ndarray | cp.ndarray,
        template_vals: np.ndarray | cp.ndarray,
        start_freqs: np.ndarray | cp.ndarray,
        start_times: np.ndarray | cp.ndarray,
        **kwargs,
    ) -> tuple[np.ndarray | cp.ndarray, np.ndarray | cp.ndarray]:
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
        )

        return d_h_out, h_h_out
    
class FDComputationGroup(BaseDomainComputationGroup):
    """
    Wraps C++ FDDomainWrap for batched likelihood computation.
    """

    def __init__(self, *args, settings: FDSettings = None, **kwargs):
        from .domains import FDSettings

        if settings is None or not isinstance(settings, FDSettings):
            raise ValueError("settings must be an instance of FDSettings for FDComputationGroup.")
        super().__init__(*args, settings=settings, **kwargs)

    @property
    def domain_args(self):
        return [
            self.settings.N,
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
        domain = self.backend.FDDomainWrap(*self.domain_args)
        logger.debug("Initialized FDDomainWrap with arguments: %s", self.domain_args)
        return domain

    def compute_likelihood_terms(
        self,
        data_index: np.ndarray | cp.ndarray,
        noise_index: np.ndarray | cp.ndarray,
        template_vals: np.ndarray | cp.ndarray,
        start_freqs: np.ndarray | cp.ndarray,
        **kwargs,
    ) -> tuple[np.ndarray | cp.ndarray, np.ndarray | cp.ndarray]:
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

        self.cpp_domain.compute_likelihood_terms(
            d_h_out,
            h_h_out,
            template_vals.ravel(),
            start_freqs,
            num_binaries,
            data_index,
            noise_index,
            num_freqs,
        )

        return d_h_out, h_h_out

class DomainComputationGroupArray:
    """Helper class to manage multiple DomainComputationGroup instances for different splits.

    It can perform computations across different GPU splits and aggregate results as needed.

    Args:
        acs : AnalysisContainerArray
            The AnalysisContainerArray containing the data and noise arrays, as well as the GPU split information. This will be used to initialize the individual DomainComputationGroup instances for each split.
    """

    def __init__(self, acs: AnalysisContainerArray):

        self.acs = acs
        self.initialize_computation_groups()

        # Precomputed routing tables. ``ac_to_split`` maps a global AC id
        # (equivalently walker id in the tempered sampler) to the split that
        # owns it; ``ac_to_intra`` maps the same AC id to its intra-split
        # index, which is what ``BaseDomainComputationGroup.compute_likelihood``
        # expects for ``data_index`` / ``noise_index``. Building these once
        # here replaces per-call ``np.where(...)`` lookups on the hot path.
        self.ac_to_split = np.asarray(self.acs.split_map)
        self.ac_to_intra = np.empty(self.acs.acs_total_entries, dtype=np.int32)
        for split_id, ids in enumerate(self.acs.gpu_splits):
            self.ac_to_intra[ids] = np.arange(len(ids), dtype=np.int32)

    def initialize_computation_groups(self):
        """Initializes a DomainComputationGroup instance for each GPU split in the AnalysisContainerArray."""
        
        force_backend = "cpu" if self.acs.gpus is None else "gpu"
        settings = self.acs.settings
        computation_groups = []
        for split_index in range(self.num_splits):
            group = self.computation_group_class(
                acs=self.acs,
                split_index=split_index,
                settings=settings,
                force_backend=force_backend,
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
        """List of GPU IDs corresponding to each split, or None if using CPU."""
        return self.acs.gpus

    @property
    def main_gpu(self):
        """GPU ID of the main device, or None if using CPU."""
        return self.gpus[0] if self.gpus is not None else None

    @property
    def num_splits(self):
        return len(self.acs.gpu_splits)

    @property
    def domain_type(self):
        """Analysis domain type, either 'STFT' or 'FD', inferred from the settings of the AnalysisContainerArray."""

        if isinstance(self.acs.settings, STFTSettings):
            return "STFT"
        elif isinstance(self.acs.settings, FDSettings):
            return "FD"
        else:
            raise NotImplementedError("Unsupported domain settings type in AnalysisContainerArray.")

    @property
    def computation_group_class(self):
        """Returns the appropriate DomainComputationGroup subclass based on the domain type."""
        if self.domain_type == "STFT":
            return STFTComputationGroup
        elif self.domain_type == "FD":
            return FDComputationGroup
        else:
            raise NotImplementedError("Unsupported domain type for computation group class.")

    @contextmanager
    def device_context(self, device_id: int = None):
        """Context manager to set the appropriate device for CPU or GPU backends.

        Notes:
            Uses ``jax.default_device`` which is safe in serial execution. A
            future threaded execution path must NOT use this context manager:
            ``jax.default_device`` is not reliably thread-safe across JAX
            versions. Worker threads should instead place arrays via
            ``jax.device_put(x, device=jax.devices("gpu")[idx])`` and rely on
            CuPy's TLS-safe ``cupy.cuda.Device`` for kernel dispatch.
        """

        if device_id is None or self.gpus is None:
            # CPU context - set context to CPU if using a GPU backend, otherwise do nothing
            with jax.default_device(jax.devices("cpu")[0]):
                yield "cpu"

        else:
            device = self.gpus[device_id]
            # GPU context - set context to the specified GPU device
            with jax.default_device(jax.devices("gpu")[device_id]):
                with self.xp.cuda.Device(device):
                    yield f"gpu: {device}"

    @contextmanager
    def _threaded_device_context(self, device_id: int | None = None):
        """Thread-safe per-worker device context.

        CuPy-only: ``cupy.cuda.Device`` is TLS, so worker threads can
        hold distinct device contexts concurrently. ``jax.default_device``
        is deliberately NOT set here — it is process-global and races
        across threads. JAX-tracing callables running under this context
        must route their host/cupy inputs via the DLPack bridge
        (``JaxThreadingMixin._to_jax`` in ``utils/jaxbase.py``) so JAX
        arrays are placed on the same CUDA device as the cupy TLS
        allocation.

        Args:
            device_id: Real CUDA device ordinal (e.g. ``group.device_id``).
                ``None`` means CPU — the context is a no-op.
        """
        if device_id is None or self.gpus is None:
            yield "cpu"
            
        else:
            device = self.gpus[device_id]
            with self.xp.cuda.Device(device):
                yield f"gpu: {device}"

    def _dispatch(self, mode: str) -> Callable:
        """Select the per-split dispatcher for the requested mode.

        ``"serial"`` returns :meth:`_loop_operation`; ``"threaded"``
        returns :meth:`_threaded_operation`. Both have the same
        ``(operation, args_per_group, kwargs_per_group, *, aggregate)``
        signature — callers can swap between them by changing this
        single mode string.
        """
        if mode == "serial":
            return self._loop_operation
        if mode == "threaded":
            return self._threaded_operation
        raise ValueError(f"Unknown mode={mode!r}; expected 'serial' or 'threaded'.")

    # ------------------------------------------------------------------ #
    # Threading-safety protocol glue                                     #
    # ------------------------------------------------------------------ #

    def _callable_id(self, callable: Callable) -> str:
        """Stable string key for a callable's role on its owner.

        Uses ``__func__.__qualname__`` for bound methods (distinguishes
        e.g. ``__call__`` from ``get_signals_for_residuals`` on the same
        owner), falling back to the object's ``__qualname__`` for
        callable instances.
        """
        fn = getattr(callable, "__func__", callable)
        return getattr(fn, "__qualname__", repr(fn))

    def _callable_owner(self, callable: Callable):
        """Return the object that owns a callable.

        Bound method → ``__self__``. Callable instance (e.g. a waveform
        object implementing ``__call__``) → the instance itself, so the
        threading-safety protocol can be queried on the object where the
        mixin lives.
        """
        return getattr(callable, "__self__", callable)

    def _threading_safe(
        self,
        callable: Callable,
        args_per_group: list,
        kwargs_per_group: list,
    ) -> bool:
        """Consult the owner's ``supports_threaded`` for every non-empty split.

        Returns ``True`` if every split reports safe, ``False`` if any
        one reports unsafe. Callables whose owner does not implement
        ``supports_threaded`` are treated as default-safe (this keeps
        non-JAX waveforms zero-config). Empty splits are skipped — they
        never trigger a compile.
        """
        owner = self._callable_owner(callable)
        if not hasattr(owner, "supports_threaded"):
            return True
        cid = self._callable_id(callable)
        for i, group in enumerate(self.computation_groups):
            if len(args_per_group[i]) == 0:
                continue
            if not owner.supports_threaded(
                cid, group.device_id, *args_per_group[i], **kwargs_per_group[i]
            ):
                return False
        return True

    def _record_threading_completion(
        self,
        callable: Callable,
        args_per_group: list,
        kwargs_per_group: list,
    ) -> None:
        """Notify the owner's registry that a per-split dispatch completed.

        Called after a successful dispatch (serial or threaded); enables
        the threaded fast path for subsequent calls with the same
        ``(callable_id, device_id, shape)``. Owners that do not
        implement ``record_completion`` are no-ops.
        """
        owner = self._callable_owner(callable)
        if not hasattr(owner, "record_completion"):
            return
        cid = self._callable_id(callable)
        for i, group in enumerate(self.computation_groups):
            if len(args_per_group[i]) == 0:
                continue
            owner.record_completion(
                cid, group.device_id, *args_per_group[i], **kwargs_per_group[i]
            )

    def restore_main_device(self):
        """Restores the main device context after GPU computations, and calls ``xp.get_default_memory_pool().free_all_blocks()`` to clear GPU memory.

        Notes:
            This method should be called after performing computations on multiple GPU devices to ensure that the main device context is restored.
        """
        if self.main_gpu is not None:
            self.xp.get_default_memory_pool().free_all_blocks()
            self.xp.cuda.runtime.setDevice(self.main_gpu)
            jax.default_device(jax.devices("gpu")[0])

    def _to_host_array(self, arr: np.ndarray | cp.ndarray) -> np.ndarray:
        """Move an array to the host, ensuring it is a numpy ndarray."""
        return np.asarray(arr.get() if hasattr(arr, "get") else arr)

    def _unpack_indices(
        self,
        data_index: np.ndarray | cp.ndarray,
        noise_index: np.ndarray | cp.ndarray,
    ) -> Tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
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
        """
        data_index_cpu = self._to_host_array(data_index)
        noise_index_cpu = self._to_host_array(noise_index)

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

    def _unpack_coords(
        self, positions_per_split: list[np.ndarray], coords: np.ndarray | cp.ndarray, kwargs: dict
    ) -> Tuple[list[tuple], list[dict]]:
        """
        Unpack coordinates for each split based on the positions per split.

        Args:
            positions_per_split: List of numpy arrays, where each array contains the positions for a specific split.
            coords: A numpy or cupy array containing the coordinates to be unpacked.
            kwargs: A dictionary of keyword arguments to be passed to the likelihood computation.

        Returns:
            A tuple containing two lists:
                - args_per_group: A list of tuples, where each tuple contains the coordinates for a specific split.
                - kwargs_per_group: A list of dictionaries, where each dictionary contains the keyword arguments for a specific split.
        """

        coords_host = self._to_host_array(coords)

        args_per_group: list = []
        kwargs_per_group: list = []
        for positions in positions_per_split:
            if len(positions) > 0:
                coords_s = coords_host[positions]
                args_per_group.append(tuple(coords_s.T))
                kwargs_per_group.append(dict(kwargs))
            else:
                args_per_group.append(())
                kwargs_per_group.append({})

        return args_per_group, kwargs_per_group

    def compute_d_d_terms(self, out=False, **kwargs) -> list[np.ndarray] | None:
        """Compute (d|d) terms across all computation groups and aggregate results.

        Args:
            out : bool, optional
                If True, return the computed (d|d) terms. Otherwise, store them in each group's `self.d_d`. Default is False.
        Returns:
            If `out` is True, returns a concatenated array of (d|d) terms from all groups. Otherwise, returns None and stores the results in each group's `self.d_d`
        """

        d_d_list = []
        for i, group in enumerate(self.computation_groups):
            with self.device_context(i) as device_info:
                d_d = group.compute_d_d_term(out=out, **kwargs)
            if out:
                d_d_list.append(d_d)

        if out:
            return d_d_list

    def compute_likelihood(
        self,
        data_index: np.ndarray | cp.ndarray,
        noise_index: np.ndarray | cp.ndarray,
        likelihood_args: list[tuple],
        mode: str = "serial",
    ) -> np.ndarray:
        """Dispatch to the per-mode likelihood implementation.

        Calls :meth:`_unpack_indices` once to partition the flat batch, then
        routes the ``(positions, data_intra, noise_intra, likelihood_args)``
        tuple to the requested per-mode path. See
        :meth:`_compute_likelihood_serial` /
        :meth:`_compute_likelihood_threaded` for the concurrency-model
        invariants of each path.

        Args:
            data_index: Global AC ids, shape ``(N,)``.
            noise_index: Global noise AC ids, shape ``(N,)``.
            likelihood_args: List of length ``num_splits`` where
                ``likelihood_args[s]`` is the positional-args tuple forwarded
                to split ``s``'s ``compute_likelihood`` (e.g. ``(vals,
                sfreqs)`` for FD, ``(vals, sfreqs, stimes)`` for STFT). Entries
                for empty splits are not inspected.
            mode: ``"serial"`` or ``"threaded"``.

        Raises:
            ValueError: If ``mode`` is not ``'serial'`` or ``'threaded'``.
        """
        positions, data_intra, noise_intra = self._unpack_indices(data_index, noise_index)
        if mode == "serial":
            return self._compute_likelihood_serial(
                positions, data_intra, noise_intra, likelihood_args
            )
        if mode == "threaded":
            return self._compute_likelihood_threaded(
                positions, data_intra, noise_intra, likelihood_args
            )
        raise ValueError(f"Unknown mode={mode!r}; expected 'serial' or 'threaded'.")

    def _compute_likelihood_serial(
        self,
        positions_per_split: list[np.ndarray],
        data_intra_per_split: list[np.ndarray],
        noise_intra_per_split: list[np.ndarray],
        likelihood_args: list[tuple],
    ) -> np.ndarray:
        """Serial per-split likelihood path — input-order scatter.

        Iterates ``self.computation_groups`` in order, placing each group's
        slice with ``self.device_context(group.device_id)``. Single-threaded,
        so ``jax.default_device`` is safe — no JAX compile race, no
        cross-thread device flip.

        Empty splits (``len(positions_per_split[s]) == 0``) are skipped so
        the pattern generalizes to batches that miss a split entirely.

        Args:
            positions_per_split: Flat-batch positions per split, from
                :meth:`_unpack_indices`.
            data_intra_per_split: Intra-split AC ids per split.
            noise_intra_per_split: Intra-split noise AC ids per split.
            likelihood_args: Per-split positional-args tuple forwarded to
                :meth:`BaseDomainComputationGroup.compute_likelihood`.

        Returns:
            Host ``np.ndarray`` of shape ``(N,)`` with per-binary
            log-likelihoods in the original flat input order.

        Raises:
            ValueError: Propagated from
                :meth:`BaseDomainComputationGroup.compute_likelihood` when
                ``group.d_d`` has not been populated (call
                :meth:`compute_d_d_terms` first).
        """
        n_data = sum(len(p) for p in positions_per_split)
        output = np.empty(n_data, dtype=np.float64)
        for i, group in enumerate(self.computation_groups):
            positions = positions_per_split[i]
            if len(positions) == 0:
                continue
            with self.device_context(i):
                idx_d = self.xp.asarray(data_intra_per_split[i])
                idx_n = self.xp.asarray(noise_intra_per_split[i])
                likes = group.compute_likelihood(idx_d, idx_n, *likelihood_args[i])
                likes_host = self._to_host_array(likes)
            output[positions] = likes_host
        return output

    def _compute_likelihood_threaded(
        self,
        positions_per_split: list[np.ndarray],
        data_intra_per_split: list[np.ndarray],
        noise_intra_per_split: list[np.ndarray],
        likelihood_args: list[tuple],
    ) -> np.ndarray:
        """Threaded per-split likelihood path — input-order scatter.

        One worker thread per computation group. Each worker materializes
        ``idx_d`` / ``idx_n`` on its own device under
        :meth:`_threaded_device_context` so cupy allocations land on the
        right GPU, then runs the C++ likelihood kernel and returns host
        ``np.ndarray``.

        No warmup guard is required here: the likelihood kernels are pure
        cupy / C++ — they hold no process-wide compile lock, so
        concurrent execution on distinct devices is always safe. JAX's
        compile hazard (handled via the threading-safety protocol at
        the :meth:`_generate_waveform` layer) is irrelevant to this
        function.

        Waveform outputs in ``likelihood_args[i]`` are already on split
        ``i``'s device (produced earlier by ``_generate_waveform`` under
        the same TLS context) — no cross-device transfer here.

        Args:
            positions_per_split: Flat-batch positions per split.
            data_intra_per_split: Intra-split AC ids per split.
            noise_intra_per_split: Intra-split noise AC ids per split.
            likelihood_args: Per-split positional-args tuple forwarded to
                :meth:`BaseDomainComputationGroup.compute_likelihood`.

        Returns:
            Host ``np.ndarray`` of shape ``(N,)`` with per-binary
            log-likelihoods in the original flat input order.
        """
        n_data = sum(len(p) for p in positions_per_split)
        output = np.empty(n_data, dtype=np.float64)

        def _eval_split_likelihood(i: int, group):
            positions = positions_per_split[i]
            if len(positions) == 0:
                return None
            idx_d = self.xp.asarray(data_intra_per_split[i])
            idx_n = self.xp.asarray(noise_intra_per_split[i])
            likes = group.compute_likelihood(idx_d, idx_n, *likelihood_args[i])
            return self._to_host_array(likes)

        per_split_out = self._threaded_operation(
            _eval_split_likelihood,
            args_per_group=[(i, g) for i, g in enumerate(self.computation_groups)],
            kwargs_per_group=None,
        )

        for i, likes_host in enumerate(per_split_out):
            if likes_host is None:
                continue
            output[positions_per_split[i]] = likes_host
        return output

    def _generate_waveform(
        self,
        waveform_callable: Callable,
        coords: np.ndarray,
        positions_per_split: list[np.ndarray],
        waveform_gen_kwargs: dict | None = None,
        *,
        mode: str = "serial",
    ) -> list:
        """Run ``waveform_callable`` once per non-empty split under device context.

        Opaque-output contract: whatever the callable returns is threaded
        through unchanged — DCGA does not destructure or branch on the
        content. This lets a single dispatcher serve both the likelihood
        path (``waveform_gen`` → tuples) and the residual path
        (``waveform_gen.get_signals_for_residuals`` → ``DomainBaseArray``).

        Concurrency-safety for ``mode="threaded"``: this is where the
        threading-safety protocol is consulted, not inside
        :meth:`_threaded_operation`. The protocol query happens against
        the *unwrapped* ``waveform_callable`` (not the local ``_call``
        wrapper built below) so bound-method identity and compile-shape
        keys resolve on the real waveform owner. When any non-empty
        split reports unsafe for ``(callable_id, device, shape)``, the
        whole dispatch falls back to :meth:`_loop_operation` silently —
        MCMC shape drift is self-healing this way: the first call with a
        novel N runs serially, seeds the owner's registry via
        :meth:`record_completion`, and subsequent calls with the same N
        run threaded.

        Args:
            waveform_callable: Callable invoked as
                ``waveform_callable(*coords_s.T, **waveform_gen_kwargs)`` on
                each non-empty split.
            coords: ``(N, ndim)`` coord batch. Materialized to host before
                slicing so the per-split partitions can be dispatched
                independently to each group's device.
            positions_per_split: Flat-batch positions per split, as returned
                by :meth:`_unpack_indices`. Empty entries produce a ``None``
                output for that split so callers can uniformly iterate
                ``num_splits`` groups.
            waveform_gen_kwargs: Extra kwargs forwarded to the callable.
            mode: Dispatch mode. ``"serial"`` runs under
                :meth:`_loop_operation` (single-threaded, JAX-safe).
                ``"threaded"`` runs under :meth:`_threaded_operation`
                after the protocol check clears every non-empty split.

        Returns:
            List of length ``num_splits``. ``out[s]`` is the raw return
            value of ``waveform_callable`` for split ``s`` (on that split's
            device), or ``None`` if that split's partition was empty.
        """
        if waveform_gen_kwargs is None:
            waveform_gen_kwargs = {}

        args_per_group, kwargs_per_group = self._unpack_coords(
            positions_per_split, coords, waveform_gen_kwargs
        )

        dispatch_mode = mode
        if mode == "threaded" and not self._threading_safe(
            waveform_callable, args_per_group, kwargs_per_group
        ):
            logger.debug(
                "threading-safety protocol reported unsafe for callable=%s; "
                "falling back to serial waveform generation.",
                self._callable_id(waveform_callable),
            )
            dispatch_mode = "serial"

        dispatch = self._dispatch(dispatch_mode)

        def _call(*flat_coords, **kwargs):
            if not flat_coords or flat_coords[0].shape[0] == 0:
                return None
            return waveform_callable(*flat_coords, **kwargs)

        outputs = dispatch(_call, args_per_group, kwargs_per_group)

        # Record completion on success. Owners without a registry
        # (non-JAX callables) are no-ops. If the dispatch raised, we
        # don't reach this line — the exception propagates and no
        # compile claim is made.
        self._record_threading_completion(
            waveform_callable, args_per_group, kwargs_per_group
        )
        return outputs

    def compute_likelihood_from_coords(
        self,
        waveform_gen: Callable,
        coords: np.ndarray,
        data_index: np.ndarray,
        noise_index: np.ndarray | None = None,
        *,
        waveform_gen_kwargs: dict | None = None,
        mode: str = "serial",
    ) -> np.ndarray:
        """End-to-end coords → per-binary log-likelihoods.

        Partitions the flat batch once via :meth:`_unpack_indices`, runs
        ``waveform_gen`` per-split via :meth:`_generate_waveform` (outputs
        stay on their origin device), and dispatches to the serial /
        threaded likelihood path with the same partition.

        Args:
            waveform_gen: Batched waveform generator. Its return value is
                treated opaquely — likelihood args are forwarded as
                ``*out`` to the group's ``compute_likelihood``.
            coords: ``(N, ndim)`` raw coord batch.
            data_index: ``(N,)`` global AC ids.
            noise_index: ``(N,)`` global noise AC ids. Defaults to
                ``data_index``.
            waveform_gen_kwargs: Extra kwargs forwarded to ``waveform_gen``.
            mode: Dispatch mode. ``"serial"`` by default.

        Returns:
            Host ``np.ndarray`` of shape ``(N,)`` with per-binary
            log-likelihoods in flat input order.
        """
        if noise_index is None:
            noise_index = data_index

        positions, data_intra, noise_intra = self._unpack_indices(data_index, noise_index)
        likelihood_args = self._generate_waveform(
            waveform_gen, coords, positions, waveform_gen_kwargs, mode=mode
        )

        if mode == "serial":
            return self._compute_likelihood_serial(
                positions, data_intra, noise_intra, likelihood_args
            )
        if mode == "threaded":
            return self._compute_likelihood_threaded(
                positions, data_intra, noise_intra, likelihood_args
            )
        raise ValueError(f"Unknown mode={mode!r}; expected 'serial' or 'threaded'.")

    def generate_signals(
        self,
        waveform_gen,
        coords: np.ndarray,
        data_index: np.ndarray = None,
        waveform_gen_kwargs: dict | None = None,
        *,
        mode: str = "serial",
    ) -> List[DomainBase | None]:
        """Produce per-split residual-ready signals on their origin devices.

        Routes ``waveform_gen`` through
        :meth:`_generate_waveform`. Each split's output is a
        ``DomainBaseArray`` of ``FDSignal`` / ``STFTSignal`` on that split's
        GPU — ready to pass to
        :meth:`AnalysisContainerArray.signal_operation` with the matching
        ``positions_per_split[s]`` as ``data_index``. Nothing is ever
        transferred to a single "target" device.

        Args:
            waveform_gen: Waveform object that should accept the per-split coords as
                ``(*coords.T, **kwargs)``.
            coords: ``(N, ndim)`` raw coord batch.
            data_index: ``(N,)`` global AC ids.
            waveform_gen_kwargs: Extra kwargs forwarded to
                ``get_signals_for_residuals``.

        Returns:
            ``(signals_per_split, positions_per_split)`` where
            ``signals_per_split[s]`` is a ``DomainBaseArray`` (or ``None`` if
            empty) and ``positions_per_split[s]`` is the flat-batch index
            array whose element ``k`` maps to
            ``signals_per_split[s][k]``.
        """
        if data_index is None:
            # infer the number of temperatures by the shape of coords
            data_index = np.arange(coords.shape[0], dtype=np.int32) % len(self.acs)

        positions, *_ = self._unpack_indices(data_index, data_index)
        signals_per_split = self._generate_waveform(
            waveform_gen,
            coords,
            positions,
            waveform_gen_kwargs,
            mode=mode,
        )
        # now use the positions to reorder the signals into the original input order, concatenating the per-split outputs as needed. 

        out_signals = [None] * len(data_index)
        for split_id, signals in enumerate(signals_per_split):
            if signals is None:
                continue
            positions_here = positions[split_id]
            for pos, signal in zip(positions_here, signals):
                out_signals[pos] = signal

        return out_signals

    def _execute_op(
        self,
        group: BaseDomainComputationGroup,
        operation: str | Callable,
        args: tuple,
        kwargs: dict,
        device_info: str,
        split_index: int,
    ) -> Any:
        """Helper to invoke either a free callable or a group method / property."""
        if callable(operation):
            op_label = getattr(operation, "__name__", repr(operation))
            logger.debug(
                f"Executing external callable {op_label} on {device_info} for split index {split_index}"
            )
            return operation(*args, **kwargs)
        else:
            target = getattr(group, operation)
            if callable(target):
                logger.debug(f"Executing {operation} on {device_info} for split index {split_index}")
                return target(*args, **kwargs)
            return target

    def _loop_operation(
        self,
        operation: str | Callable,
        args_per_group: list | None = None,
        kwargs_per_group: list | None = None,
        *,
        aggregate: Callable[[list], Any] | None = None,
    ) -> list | Any:
        """Dispatch an operation to each computation group under its device context.

        This is the one primitive the coordinator uses to drive per-split
        work. Callers are responsible for routing their inputs into
        per-group form (see :meth:`_unpack_input_args`); this method only
        handles the device switching and aggregation.

        Args:
            operation: Either a string naming an attribute / method on
                each :class:`BaseDomainComputationGroup`, or a free
                callable. String form: if the attribute is callable it is
                invoked with per-group args/kwargs; otherwise its value is
                returned as-is (useful for property reads). Callable form:
                the callable is invoked directly with per-group
                args/kwargs on every group — useful for running external
                per-device work such as waveform generation.
            args_per_group: List of length ``num_splits`` of positional arg
                tuples, one tuple per group. If ``None``, no positional args
                are passed.
            kwargs_per_group: List of length ``num_splits`` of kwargs dicts,
                one dict per group. If ``None``, no kwargs are passed.
            aggregate: Optional callable applied to the raw list of
                per-group outputs. If omitted the raw list is returned.

        Returns:
            The raw per-group output list, or ``aggregate(outputs)`` if
            ``aggregate`` was provided.
        """

        if args_per_group is None:
            args_per_group = [()] * self.num_splits
        if kwargs_per_group is None:
            kwargs_per_group = [{}] * self.num_splits

        if len(args_per_group) != self.num_splits:
            raise ValueError(
                f"args_per_group length {len(args_per_group)} != num_splits {self.num_splits}"
            )
        if len(kwargs_per_group) != self.num_splits:
            raise ValueError(
                f"kwargs_per_group length {len(kwargs_per_group)} != num_splits {self.num_splits}"
            )

        outputs: list = []
        for i, group in enumerate(self.computation_groups):
            with self.device_context(i) as device_info:
                out_i = self._execute_op(
                    group,
                    operation,
                    args_per_group[i],
                    kwargs_per_group[i],
                    device_info,
                    i,
                )
            outputs.append(out_i)

        return aggregate(outputs) if aggregate is not None else outputs

    def _threaded_operation(
        self,
        operation: str | Callable,
        args_per_group: list | None = None,
        kwargs_per_group: list | None = None,
        *,
        aggregate: Callable[[list], Any] | None = None,
    ) -> list | Any:
        """Threaded twin of :meth:`_loop_operation`.

        Same interface and input/output contract — only the concurrency
        model differs. One worker thread per computation group under
        :meth:`_threaded_device_context` (cupy TLS; never
        ``jax.default_device``). Output ordering is preserved:
        ``outputs[i]`` is group ``i``'s return regardless of join order.
        The first worker exception (lowest split index) is re-raised on
        the coordinator after all threads join; additional exceptions
        are logged at ``DEBUG``.

        No protocol-safety check lives here — this is the generic
        primitive. The "is this JAX-tracing op safe to run concurrently?"
        question is answered one layer up in :meth:`_generate_waveform`,
        which consults the owner's threading-safety protocol and falls
        back to :meth:`_loop_operation` when unsafe.

        Args:
            operation: String (attribute / method name on each group) or
                free callable. Same semantics as
                :meth:`_loop_operation`.
            args_per_group: Per-group positional-args tuples.
            kwargs_per_group: Per-group kwargs dicts.
            aggregate: Optional post-processor over the raw outputs list.

        Returns:
            List of per-group outputs (or ``aggregate(outputs)`` if
            provided), in split-index order.
        """
        if args_per_group is None:
            args_per_group = [()] * self.num_splits
        if kwargs_per_group is None:
            kwargs_per_group = [{}] * self.num_splits
        if len(args_per_group) != self.num_splits:
            raise ValueError(
                f"args_per_group length {len(args_per_group)} != num_splits {self.num_splits}"
            )
        if len(kwargs_per_group) != self.num_splits:
            raise ValueError(
                f"kwargs_per_group length {len(kwargs_per_group)} != num_splits {self.num_splits}"
            )

        outputs: list = [None] * self.num_splits
        errors: list[tuple[int, BaseException]] = []
        err_lock = threading.Lock()

        def _worker(i: int, group):
            try:
                with self._threaded_device_context(i) as device_info:
                    out_i = self._execute_op(
                        group,
                        operation,
                        args_per_group[i],
                        kwargs_per_group[i],
                        device_info,
                        i,
                    )
                outputs[i] = out_i
            except BaseException as exc:
                with err_lock:
                    errors.append((i, exc))

        threads = [
            threading.Thread(target=_worker, args=(i, g), name=f"dcga-split-{i}")
            for i, g in enumerate(self.computation_groups)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        if errors:
            errors.sort(key=lambda ie: ie[0])
            for idx, exc in errors[1:]:
                logger.debug(
                    "additional worker exception on split %d: %r", idx, exc
                )
            raise errors[0][1]

        return aggregate(outputs) if aggregate is not None else outputs

class ComputationRouter:
    """Routes attribute/method access to per-device replicas based on the current TLS device.

    Use this to wrap objects that hold device-resident allocations (e.g.
    waveform generators owning cupy-backed orbits or sensitivity tensors)
    so every thread of a multi-GPU pipeline sees a replica whose internal
    buffers live on the current GPU. The router is a drop-in for a single
    instance in APIs that previously accepted one: ``DCGA`` keeps its
    single ``waveform_gen`` parameter and the router is transparent to it.

    Construction discipline
    -----------------------
    The router does not build replicas. Callers must construct each
    replica under its target GPU's TLS — a pre-built shared instance
    passed into a closure defeats the mechanism. Use :func:`build_replicas`
    for a factory-based helper.

    Method routing is lazy
    ----------------------
    ``__getattr__`` returns a *wrapper function* that re-reads
    :attr:`current_device` on every invocation, not a bound method of the
    replica current at lookup time. This matters because DCGA captures
    the callable once on the main thread and calls it from every worker;
    an eager resolution would pin every worker to the main-thread
    replica and re-introduce the cross-device bug.

    The wrapper exposes ``__func__`` / ``__self__`` / ``__qualname__`` so
    DCGA's :meth:`_callable_id` / :meth:`_callable_owner` resolve to
    stable per-method identities and to the router (not to any specific
    replica). Consequently DCGA's ``_threading_safe`` and
    ``_record_threading_completion`` invoke the router's
    :meth:`supports_threaded` / :meth:`record_completion`, which delegate
    by explicit ``device_id`` to the right replica.

    Non-method attribute access (``router.some_data``) is resolved
    eagerly against ``class_map[current_device]`` — lazy resolution only
    makes sense for callables. Callers that read device-resident
    attributes directly are responsible for ensuring they use them from a
    thread bound to the correct device.

    Limitation: callable dispatcher, not transparent proxy
    ------------------------------------------------------
    This class routes *named attribute access* and *explicit* ``__call__``
    — nothing more. Python's data model looks up implicit protocol
    dunders (``__len__``, ``__iter__``, ``__getitem__``, ``__array__``,
    ``__reduce__``, etc.) on ``type(obj)``, bypassing both the instance
    ``__dict__`` and ``__getattr__``. Therefore ``len(router)``,
    ``router[i]``, ``iter(router)``, ``np.asarray(router)``, and
    pickling will NOT delegate to the underlying replica — they will
    fail or use ``object``'s defaults.

    This is intentional: waveform generators (the primary use case) are
    callable computational objects and do not implement those protocols.
    If you ever need to wrap an object that does, add an explicit
    passthrough dunder on :class:`ComputationRouter` that routes via
    :meth:`_routed_operation` — do not migrate to
    ``__getattribute__`` (too fragile for descriptor protocol) or
    rely on ``__getattr__`` (wrong lookup path for implicit protocols).

    :meth:`__deepcopy__` is explicitly blocked: silently deep-copying a
    router would build per-device replicas whose device affinity depends
    on the current TLS at copy time, which silently shreds the
    per-device construction contract.

    Args:
        class_map: Mapping from device id to the replica for that device.
            Keys must match ``ComputationGroup.device_id``: integer GPU
            indices on the GPU backend, or ``None`` on the CPU backend.
    """
    def __init__(self, class_map: dict[int | None, Any]):
        self.class_map = class_map

        first_class = next(iter(self.class_map.values()))
        self._xp = getattr(first_class, "xp", np)
    
    @property
    def xp(self):
        """The array module (``numpy`` or ``cupy``) of the replicas, for convenience."""
        return self._xp

    def __getattr__(self, name: str) -> Any:
        """Return a call-time-dispatching wrapper for methods; eagerly resolved value for non-method attributes.

        Dunders (``__xxx__``) raise ``AttributeError`` to avoid interfering
        with Python protocols (pickling, copy, ``hasattr`` probes for
        ``__func__``/``__self__``/``__qualname__`` that land on the router
        itself).
        """
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)

        class_map = self.__dict__["class_map"]
        any_replica = next(iter(class_map.values()))
        real_fn = getattr(type(any_replica), name, None)
        if real_fn is None or not callable(real_fn):
            # Not a class-level method — fall back to eager resolution on
            # the current-device replica. Captures reference semantics at
            # lookup time; caller owns any device-affinity concerns.
            return getattr(class_map[self.current_device], name)

        router = self

        def routed(*args, **kwargs):
            return getattr(router.class_map[router.current_device], name)(*args, **kwargs)

        # Identity forwarding so DCGA's callable_id / callable_owner see
        # the replica's method signature and the router as the owner.
        routed.__func__ = real_fn
        routed.__self__ = self
        routed.__qualname__ = getattr(real_fn, "__qualname__", name)
        routed.__name__ = getattr(real_fn, "__name__", name)
        return routed

    @property
    def current_device(self) -> int | None:
        """Device id of the current thread's TLS cupy context.

        Returns ``None`` on the CPU backend to match
        ``ComputationGroup.device_id``'s convention. On GPU, returns the
        cupy TLS device id — DCGA's ``_threaded_device_context`` installs
        the split's ``device_id`` before invoking routed methods, so this
        picks the right replica per worker.
        """
        if self.xp is np:
            return None
        return self.xp.cuda.runtime.getDevice()

    def _routed_operation(self, operation: str, *args, **kwargs) -> Any:
        """
        Run the specified operation on the class corresponding to the current device.

        Args:
            operation: The name of the operation to run.
            *args: Positional arguments to pass to the operation.
            **kwargs: Keyword arguments to pass to the operation.
        Returns:
            The result of the operation.
        """
        # Route execution based on current Thread-Local Storage device
        fn = getattr(self.class_map[self.current_device], operation)
        return fn(*args, **kwargs)

    def __call__(self, *args, **kwargs) -> Any:
        """
        Run the :meth:`__call__` method of the class corresponding to the current device.

        Args:
            *args: Positional arguments to pass to the __call__ method.
            **kwargs: Keyword arguments to pass to the __call__ method.
        Returns:
            The result of the __call__ method.
        """
        return self._routed_operation("__call__", *args, **kwargs)

    def __repr__(self) -> str:
        keys = list(self.class_map.keys())
        replica_types = sorted({type(r).__name__ for r in self.class_map.values()})
        return f"ComputationRouter(devices={keys}, replicas={replica_types})"

    def __deepcopy__(self, memo):
        """Blocked — deep-copying a router silently shreds per-device construction.

        A naive ``deepcopy`` would recursively copy every replica under
        whatever cupy TLS happens to be current at copy time, producing
        a router whose replicas all share one device. If you need a
        fresh router, rebuild it explicitly with :func:`build_replicas`
        and a factory that re-allocates device-resident state.
        """
        raise TypeError(
            "ComputationRouter is not deep-copyable. Per-device replicas "
            "must be constructed under their target GPU's TLS via "
            "build_replicas(gpus, factory); deepcopy cannot preserve that "
            "contract. Rebuild the router explicitly instead."
        )

    # --- Threading-Safety Protocol Delegation ---

    def supports_threaded(self, cid: str, device_id: int | None, *args, **kwargs) -> bool:
        """Route the safety query to the replica for ``device_id``.

        DCGA pre-queries this on the main thread before spawning workers,
        so we dispatch by explicit ``device_id`` (which matches
        ``ComputationGroup.device_id``) rather than by TLS — the main
        thread's TLS is unrelated to any split's target device.
        """
        cls = self.class_map[device_id]
        if hasattr(cls, "supports_threaded"):
            return cls.supports_threaded(cid, device_id, *args, **kwargs)
        return True  # Non-JAX replicas are default-safe.

    def record_completion(self, cid: str, device_id: int | None, *args, **kwargs) -> None:
        """Route the completion registry to the replica for ``device_id``."""
        cls = self.class_map[device_id]
        if hasattr(cls, "record_completion"):
            cls.record_completion(cid, device_id, *args, **kwargs)


def build_replicas(
    gpus: list[int] | None,
    factory: Callable[[], Any],
) -> dict[int | None, Any]:
    """Construct one replica per target GPU under that GPU's cupy TLS.

    Helper for building a ``class_map`` that :class:`ComputationRouter`
    can wrap. Each replica is constructed inside a ``cupy.cuda.Device(g)``
    block so any device-resident attribute allocated inside ``factory``
    (orbits arrays, sensitivity basis, pre-tapered windows, cached TDI
    tensors) lands on GPU ``g``.

    Critical: ``factory`` must not close over a pre-built shared
    device-resident object (e.g. an ``orbits`` already allocated on a
    single GPU). If it does, every replica's closure captures the same
    GPU-pinned instance and the per-device construction is a no-op. The
    argument type is ``Callable[[], Any]`` — not an instance — so the
    factory runs fresh under each GPU's TLS.

    Args:
        gpus: Absolute GPU ids to replicate across, or ``None`` / empty
            list for the CPU path.
        factory: Zero-argument callable that builds a fresh replica.

    Returns:
        Mapping from device id to replica. Keys match
        ``ComputationGroup.device_id``: GPU indices when ``gpus`` is
        non-empty, or ``{None: factory()}`` on the CPU path.
    """
    if not gpus:
        return {None: factory()}
    # Local import keeps the module importable on CPU-only installs.
    import cupy as cp  # type: ignore[import-not-found]
    replicas: dict[int | None, Any] = {}
    for g in gpus:
        with cp.cuda.Device(g):
            replicas[g] = factory()
    return replicas