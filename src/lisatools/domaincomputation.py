"""Python wrapper around C++ STFTDomainWrap / FDDomainWrap for batched
likelihood computation of (d|h) and (h|h) inner products."""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, List, Tuple

import jax
import numpy as np

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
    from .domains import FDSettings, STFTSettings

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

    def compute_d_d_term(self, out=False, **kwargs):
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

        # JAX-safe concurrency scaffolding. The lock guards first-time compile
        # misses when a future threaded path is enabled; ``_warm_compile_done``
        # is flipped by ``warm_jax_compile`` once every device has compiled.
        self._jax_compile_lock = threading.Lock()
        self._warm_compile_done: bool = False

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

    @property
    def xp(self):
        """Array module (numpy or cupy)."""
        return self.acs.xp

    @property
    def gpus(self):
        """List of GPU IDs corresponding to each split, or None if using CPU."""
        return self.acs.gpus

    @property
    def num_splits(self):
        return len(self.acs.gpu_splits)

    @property
    def domain_type(self):
        """Analysis domain type, either 'STFT' or 'FD', inferred from the settings of the AnalysisContainerArray."""
        from .domains import FDSettings, STFTSettings

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

        if device_id is None:
            # CPU context - set context to CPU if using a GPU backend, otherwise do nothing
            with jax.default_device(jax.devices("cpu")[0]):
                yield "cpu"

        else:
            device = self.gpus[device_id]
            # GPU context - set context to the specified GPU device
            with jax.default_device(jax.devices("gpu")[device_id]):
                with self.xp.cuda.Device(device):
                    yield f"gpu: {device}"

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
        data_index_cpu = np.asarray(data_index.get() if hasattr(data_index, "get") else data_index)
        noise_index_cpu = np.asarray(
            noise_index.get() if hasattr(noise_index, "get") else noise_index
        )
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

        coords_host = np.asarray(coords.get() if hasattr(coords, "get") else coords)

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

    def compute_d_d_terms(self, out=False, **kwargs):
        """Compute (d|d) terms across all computation groups and aggregate results.

        Args:
            out : bool, optional
                If True, return the computed (d|d) terms. Otherwise, store them in each group's `self.d_d`. Default is False.
        Returns:
            If `out` is True, returns a concatenated array of (d|d) terms from all groups. Otherwise, returns None and stores the results in each group's `self.d_d`
        """

        d_d_list = []
        for group in self.computation_groups:
            with self.device_context(group.device_id) as device_info:
                d_d = group.compute_d_d_term(out=out, **kwargs)
            if out:
                d_d_list.append(d_d)

        if out:
            return self.xp.concatenate(d_d_list).flatten()

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
            with self.device_context(group.device_id):
                idx_d = self.xp.asarray(data_intra_per_split[i])
                idx_n = self.xp.asarray(noise_intra_per_split[i])
                likes = group.compute_likelihood(idx_d, idx_n, *likelihood_args[i])
                likes_host = likes.get() if hasattr(likes, "get") else np.asarray(likes)
            output[positions] = likes_host
        return output

    def _compute_likelihood_threaded(
        self,
        positions_per_split: list[np.ndarray],
        data_intra_per_split: list[np.ndarray],
        noise_intra_per_split: list[np.ndarray],
        likelihood_args: list[tuple],
    ) -> np.ndarray:
        """Threaded per-split likelihood path (reserved).

        When implemented, worker threads must place arrays with
        ``jax.device_put(x, device=jax.devices("gpu")[idx])`` — NOT via
        ``self.device_context`` / ``jax.default_device``, which is
        process-global and races across threads. Callers must invoke
        :meth:`warm_jax_compile` once before the first threaded run so
        every device's JAX compile cache is populated serially.

        Raises:
            NotImplementedError: Always, until the threaded path is wired.
        """
        raise NotImplementedError(
            "threaded mode not implemented; use mode='serial'. "
            "Call warm_jax_compile() once before any future threaded run."
        )

    def _generate_waveform(
        self,
        waveform_callable: Callable,
        coords: np.ndarray,
        positions_per_split: list[np.ndarray],
        waveform_gen_kwargs: dict | None = None,
    ) -> list:
        """Run ``waveform_callable`` once per non-empty split under device context.

        Opaque-output contract: whatever the callable returns is threaded
        through unchanged — DCGA does not destructure or branch on the
        content. This lets a single dispatcher serve both the likelihood
        path (``waveform_gen`` → tuples) and the residual path
        (``waveform_gen.get_signals_for_residuals`` → ``DomainBaseArray``).

        Args:
            waveform_callable: Callable invoked as
                ``waveform_callable(*coords_s.T, **waveform_gen_kwargs)`` on
                each non-empty split, under ``self.device_context`` for the
                owning computation group.
            coords: ``(N, ndim)`` coord batch. Materialized to host before
                slicing so the per-split partitions can be dispatched
                independently to each group's device.
            positions_per_split: Flat-batch positions per split, as returned
                by :meth:`_unpack_indices`. Empty entries produce a ``None``
                output for that split so callers can uniformly iterate
                ``num_splits`` groups.
            waveform_gen_kwargs: Extra kwargs forwarded to the callable.

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

        def _call(*flat_coords, **kwargs):
            if not flat_coords or flat_coords[0].shape[0] == 0:
                return None
            return waveform_callable(*flat_coords, **kwargs)

        return self._loop_operation(_call, args_per_group, kwargs_per_group)

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
            waveform_gen, coords, positions, waveform_gen_kwargs
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

    def generate_signals_for_residuals(
        self,
        waveform_gen,
        coords: np.ndarray,
        data_index: np.ndarray,
        *,
        waveform_gen_kwargs: dict | None = None,
    ) -> Tuple[list, list[np.ndarray]]:
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
        positions, *_ = self._unpack_indices(data_index, data_index)
        signals_per_split = self._generate_waveform(
            waveform_gen,
            coords,
            positions,
            waveform_gen_kwargs,
        )
        return signals_per_split, positions

    def warm_jax_compile(
        self,
        waveform_gen: Callable,
        sample_coords_per_split=None,
        *,
        coords=None,
        data_index=None,
        sample_size_per_split: int = 1,
        **waveform_gen_kwargs,
    ) -> None:
        """Pre-warm the JAX compile cache once per device, serially.

        Threaded execution paths contend on the XLA compile lock and first
        compiles on distinct devices can deadlock if triggered concurrently.
        Calling this once at the start of a run (before any future threaded
        likelihood evaluation) makes the compile misses happen in the safe
        serial phase.

        The warmup works by invoking ``waveform_gen`` once per device; the
        MBH path materializes its output to cupy, which is a sync point, so
        by the time each call returns the JIT compile for that device has
        completed. No explicit ``block_until_ready`` traversal is required.

        Two entry paths are supported — supply exactly one:

        * ``sample_coords_per_split``: an explicit list/dict of per-split
          coord batches. Used when the caller already has per-split coords
          (e.g. pre-split warmups in test harnesses).
        * ``coords`` + ``data_index``: a flat batch. DCGA partitions by
          ``ac_to_split[data_index]`` and takes the first
          ``sample_size_per_split`` entries from each split's partition.
          If a split has no entries, the first few global coords are used
          as a stand-in so every device still sees a warmup call.

        Args:
            waveform_gen: Callable invoked as
                ``waveform_gen(*coords.T, **waveform_gen_kwargs)`` — the same
                shape of call the move class uses in its hot path.
            sample_coords_per_split: List of length ``num_splits`` or
                ``dict[int, array]`` keyed by split id. Each entry is a
                small ``(n, ndim)`` coord batch.
            coords: Flat ``(N, ndim)`` coord batch (alternative to
                ``sample_coords_per_split``).
            data_index: Flat ``(N,)`` AC-id batch (required with ``coords``).
            sample_size_per_split: Number of coords to take per split when
                deriving from ``coords`` / ``data_index``. Defaults to 1.
            **waveform_gen_kwargs: Forwarded to ``waveform_gen``.

        Notes:
            Safe to call multiple times — re-invocation is a no-op once
            ``self._warm_compile_done`` is True.
        """
        if self._warm_compile_done:
            logger.debug("warm_jax_compile: already warmed, skipping")
            return

        explicit = sample_coords_per_split is not None
        derived = coords is not None or data_index is not None
        if explicit and derived:
            raise ValueError(
                "warm_jax_compile: pass either sample_coords_per_split OR "
                "(coords, data_index), not both."
            )
        if derived:
            if coords is None or data_index is None:
                raise ValueError("warm_jax_compile: coords and data_index must be given together.")
            sample_coords_per_split = self._derive_warm_samples(
                coords, data_index, sample_size_per_split
            )
        elif not explicit:
            raise ValueError(
                "warm_jax_compile: pass sample_coords_per_split or " "(coords, data_index)."
            )

        if isinstance(sample_coords_per_split, dict):
            coords_iter = [sample_coords_per_split[i] for i in range(self.num_splits)]
        else:
            coords_iter = list(sample_coords_per_split)
        if len(coords_iter) != self.num_splits:
            raise ValueError(
                f"sample_coords_per_split length {len(coords_iter)} "
                f"!= num_splits {self.num_splits}"
            )

        for i, (group, coords_s) in enumerate(zip(self.computation_groups, coords_iter)):
            with self.device_context(group.device_id) as device_info:
                logger.debug(f"warm_jax_compile: warming {device_info} (split {i})")
                coords_arr = np.asarray(coords_s)
                waveform_gen(*coords_arr.T, **waveform_gen_kwargs)

        self._warm_compile_done = True

    def _derive_warm_samples(
        self,
        coords,
        data_index,
        sample_size_per_split: int,
    ) -> dict:
        """Pick per-split warmup coord batches from a flat (coords, data_index).

        For splits whose partition is non-empty, take the first
        ``sample_size_per_split`` entries. For splits with no matching AC
        ids, fall back to the first ``sample_size_per_split`` global coords
        so every device still receives a warmup invocation.
        """
        data_index_cpu = np.asarray(data_index.get() if hasattr(data_index, "get") else data_index)
        coords_host = np.asarray(coords.get() if hasattr(coords, "get") else coords)
        split_of_each = self.ac_to_split[data_index_cpu]
        k = max(1, int(sample_size_per_split))
        fallback = coords_host[:k]

        per_split: dict[int, np.ndarray] = {}
        for split_id in range(self.num_splits):
            positions = np.where(split_of_each == split_id)[0]
            if len(positions) == 0:
                per_split[split_id] = fallback
            else:
                take = positions[: min(k, len(positions))]
                per_split[split_id] = coords_host[take]
        return per_split

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

        op_is_callable = callable(operation)
        op_label = getattr(operation, "__name__", repr(operation)) if op_is_callable else operation

        outputs: list = []
        for i, group in enumerate(self.computation_groups):
            with self.device_context(group.device_id) as device_info:
                if op_is_callable:
                    logger.debug(
                        f"Executing external callable {op_label} on {device_info} for split index {i}"
                    )
                    out_i = operation(*args_per_group[i], **kwargs_per_group[i])
                else:
                    target = getattr(group, operation)
                    if callable(target):
                        logger.debug(f"Executing {operation} on {device_info} for split index {i}")
                        out_i = target(*args_per_group[i], **kwargs_per_group[i])
                    else:
                        out_i = target
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
        """Threaded version of :meth:`_loop_operation`. Reserved for future use."""
        raise NotImplementedError("Threaded operations not yet implemented; use _loop_operation.")
