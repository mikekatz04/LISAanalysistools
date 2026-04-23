"""Python wrapper around C++ STFTDomainWrap / FDDomainWrap for batched
likelihood computation of (d|h) and (h|h) inner products."""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

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
            raise ValueError("C++ domain has not been created yet. Call _create_cpp_domain first.")
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

    def compute_signal_likelihood_terms(
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
            "The `compute_signal_likelihood_terms` method must be implemented by subclasses"
        )

    def compute_signal_likelihood(
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


class STFTComputationGroup(BaseDomainComputationGroup):
    """Wraps C++ STFTDomainWrap for batched likelihood computation."""

    def __init__(self, *args, settings: STFTSettings = None, **kwargs):
        from .domains import STFTSettings

        if settings is None or not isinstance(settings, STFTSettings):
            raise ValueError(
                "settings must be an instance of STFTSettings for STFTComputationGroup."
            )
        super().__init__(*args, settings=settings, **kwargs)

        if self.device_id is not None:
            with self.xp.cuda.Device(self.device_id):
                self._cpp_domain = self._create_cpp_domain()
        else:
            self._cpp_domain = self._create_cpp_domain()

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

    def compute_signal_likelihood_terms(
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

        if self.device_id is not None:
            with self.xp.cuda.Device(self.device_id):
                self._cpp_domain = self._create_cpp_domain()
        else:
            self._cpp_domain = self._create_cpp_domain()

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

    def compute_signal_likelihood_terms(
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
        # index, which is what ``BaseDomainComputationGroup.compute_signal_likelihood``
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
        computation_groups: list[BaseDomainComputationGroup] = []
        for split_index in range(self.num_splits):
            with self.device_context(self.acs.gpus[split_index] if self.acs.gpus is not None else None):
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
        """list of GPU IDs corresponding to each split, or None if using CPU."""
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
    def device_context(self, device: int = None):
        """Context manager to set the appropriate device for CPU or GPU backends.

        Args:
            device: GPU device ID to set the context to. If None, it will set the context to CPU if using a GPU backend, or do nothing if already using a CPU backend.
        """

        if device is None or self.gpus is None:
            # CPU context - set context to CPU if using a GPU backend, otherwise do nothing
            with jax.default_device(jax.devices("cpu")[0]):
                yield "cpu"

        else:
            device_id = self.gpus.index(device)

            # GPU context - set context to the specified GPU device
            with jax.default_device(jax.devices("gpu")[device_id]):
                with self.xp.cuda.Device(device):
                    yield f"gpu: {device}"

    def restore_main_device(self):
        """Restores the main device context after GPU computations, and calls ``xp.get_default_memory_pool().free_all_blocks()`` to clear GPU memory.

        Notes:
            This method should be called after performing computations on multiple GPU devices to ensure that the main device context is restored.
        """
        if self.main_gpu is not None:
            self.xp.get_default_memory_pool().free_all_blocks()
            self.xp.cuda.runtime.setDevice(self.main_gpu)
            jax.default_device(jax.devices("gpu")[self.gpus.index(self.main_gpu)])

    def synchronize(self):
        """Synchronizes all GPU devices to ensure that all computations are complete before proceeding.

        Notes:
            This method should be called after performing computations on multiple GPU devices to ensure that all devices have completed their tasks before moving on to the next steps in the workflow.
        """
        if self.gpus is not None:
            for device in self.gpus:
                with self.device_context(device):
                    self.xp.cuda.runtime.deviceSynchronize()
                    self.xp.get_default_memory_pool().free_all_blocks()

    def _to_host(self, arr: np.ndarray | cp.ndarray) -> np.ndarray:
        """Move an array to the host, ensuring it is a numpy ndarray."""
        return arr.get() if hasattr(arr, "get") else arr

    def _unpack_indices(
        self,
        data_index: np.ndarray | cp.ndarray,
        noise_index: np.ndarray | cp.ndarray,
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
        """
        data_index_cpu = self._to_host(data_index)
        noise_index_cpu = self._to_host(noise_index)

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
        self, 
        positions_per_split: list[np.ndarray], 
        coords: np.ndarray | cp.ndarray | list[np.ndarray | cp.ndarray], 
    ) -> tuple[list[tuple], list[dict]]:
        """
        Unpack coordinates for each split based on the positions per split.

        Args:
            positions_per_split: list of numpy arrays, where each array contains the positions for a specific split.
            coords: A numpy or cupy array containing the coordinates to be unpacked, or a list of such arrays if there are multiple coordinate arrays.

        Returns:
            args_per_group: A list of coordinates for each split, where each element is either a single array (if there is only one coordinate array) or a tuple of arrays (if there are multiple coordinate arrays). The coordinates are indexed according to the positions for each split.
        """

        if not isinstance(coords, list):
            coords = [coords]

        coords_host = [self._to_host(coords_here) for coords_here in coords]

        args_per_group: list = []
        for positions in positions_per_split:
            if len(positions) > 0:
                coords_s = [coords_host[i][positions] for i in range(len(coords_host))]

                args_per_group.append(tuple(coords_s) if len(coords_s) > 1 else coords_s[0])
            else:
                args_per_group.append(())

        return args_per_group

    def _loop_operation(
        self,
        operation: Callable | list[Callable],
        operation_args_per_split: list[tuple] = None,
        operation_kwargs: dict | list[dict] = None,
        aggregate_fn: Callable = None,
    ) -> list | Any:
        """General loop to perform an operation across splits, with optional result aggregation.

        Args:
            operation: A callable or a list of callables (one per split) to execute within each split's device context. Each callable should accept the corresponding args and kwargs for that split.
            operation_args_per_split: Optional list of tuples, where each tuple contains the positional arguments to pass to the operation for the corresponding split. The caller is responsible for ensuring that the arguments are correctly placed on the appropriate device (e.g., via self.xp.asarray). If not provided, defaults to a list of empty tuples.
            operation_kwargs: Optional dictionary or list of dictionaries of keyword arguments to pass to the operation for all splits. 
            aggregate_fn: Optional callable to aggregate the results from all splits after the loop. If not provided, the raw list of results from each split is returned.
        
        Returns:
            A list of results from each split if aggregate_fn is not provided, or the aggregated result if aggregate_fn is provided.
        """
        if operation_args_per_split is None:
            operation_args_per_split = [()] * self.num_splits
        if operation_kwargs is None:
            operation_kwargs = {}
        
        if len(operation_args_per_split) != self.num_splits:
            raise ValueError("Length of operation_args_per_split must match the number of splits.")

        if isinstance(operation, list):
            if len(operation) != self.num_splits:
                raise ValueError("If operation is a list, its length must match the number of splits.")
            operations = operation
        else:
            operations = [operation] * self.num_splits
        
        if isinstance(operation_kwargs, list):
            if len(operation_kwargs) != self.num_splits:
                raise ValueError("If operation_kwargs is a list, its length must match the number of splits.")
            operation_kwargs_per_split = operation_kwargs
        else:
            operation_kwargs_per_split = [operation_kwargs] * self.num_splits

        outputs = []
        for i, device in enumerate(self.gpus if self.gpus is not None else [None] * self.num_splits):
            with self.device_context(device):
                out_i = operations[i](*operation_args_per_split[i], **operation_kwargs_per_split[i])
            outputs.append(out_i)

        if aggregate_fn is not None:
            return aggregate_fn(outputs)
        return outputs

    def compute_d_d_terms(self, out: bool = False, **kwargs):
        """Compute (d|d) terms for all splits and store them in the respective computation groups."""
        
        operations = [group.compute_d_d_term for group in self.computation_groups]

        list_out = self._loop_operation(operation=operations, operation_kwargs={'out': out, **kwargs})
        if out:
            return list_out