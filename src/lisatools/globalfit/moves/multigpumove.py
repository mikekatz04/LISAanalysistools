"""
multigpumove.py
=============

Base move to perform likelihood evaluations on multiple devices. 
"""
from __future__ import annotations

from logging import getLogger
import numpy as np
from typing import TYPE_CHECKING

from ...domaincomputation import DomainComputationGroupArray

logger = getLogger(__name__)


class MultiGPUMoveBase:
    def __init__(
        self,
        dcga: DomainComputationGroupArray = None,
        run_async: bool = False,
        run_threaded: bool = False,
        batch_size_per_gpu: int = None,
        *,
        acs=None,
    ):
        # The C++ likelihood coordinator now lives on ``AnalysisContainerArray``
        # (DCGA was absorbed). Accept either an ACA or a (deprecated)
        # ``DomainComputationGroupArray`` shim at the constructor boundary so
        # external settings files that still pass ``dcga=`` keep working, and
        # resolve both to the real ACA. We no longer store the DCGA itself.
        resolved = acs if acs is not None else dcga
        self.acs = resolved.acs if hasattr(resolved, "acs") else resolved
        self._run_async = run_async
        self._run_threaded = run_threaded
        # Cap on the number of waveform+likelihood evaluations a single device
        # runs at once. ``None`` keeps the all-at-once behaviour (every walker
        # on a split evaluated in one call); see :meth:`run_in_gpu_batches`.
        self.batch_size_per_gpu = batch_size_per_gpu

    @property
    def run_async(self) -> bool:
        return self._run_async

    @property
    def run_threaded(self) -> bool:
        return self._run_threaded

    @property
    def xp(self):
        """Return the array library (numpy or cupy) used by the analysis container array."""
        return self.acs.xp

    def iter_gpu_batch_positions(self, data_index):
        """Partition a flat ``data_index`` batch into per-GPU sub-batches.

        Yields position arrays into the flat batch such that each GPU split
        contributes at most :attr:`batch_size_per_gpu` rows per yielded
        sub-batch, so peak per-device memory is bounded by ``batch_size_per_gpu``
        waveforms while all devices stay busy within each sub-batch.

        The walker -> GPU map is *contiguous*
        (``AnalysisContainerArray.gpu_splits`` is built via
        ``np.split(np.arange(total), ...)``), so a flat ``num_gpus * B`` chunk
        would land almost entirely on one device. We therefore chunk *within*
        each split and union the k-th ``B``-sized slice across splits.

        With ``batch_size_per_gpu is None`` a single ``slice(None)`` is yielded,
        i.e. the original all-at-once behaviour with no extra copies.
        """
        if self.batch_size_per_gpu is None:
            yield slice(None)
            return

        B = self.batch_size_per_gpu
        positions_per_split, _, _ = self.acs.unpack_indices(np.asarray(data_index))
        n_chunks = max(
            (int(np.ceil(len(p) / B)) for p in positions_per_split), default=0
        )
        for k in range(n_chunks):
            sub = np.concatenate(
                [p[k * B:(k + 1) * B] for p in positions_per_split if len(p) > k * B]
            )
            if sub.size:
                yield sub

    def run_in_gpu_batches(self, data_index, eval_fn, n_out=None):
        """Evaluate ``eval_fn`` over per-GPU sub-batches and scatter the results.

        Args:
            data_index: flat ``(N,)`` array routing each row to its owning split.
            eval_fn: callable ``eval_fn(positions) -> (len(positions),)`` that
                returns the per-row likelihoods for exactly ``positions``, in
                order. ``positions`` is whatever
                :meth:`iter_gpu_batch_positions` yields (an int array, or
                ``slice(None)`` in the unbatched case).
            n_out: length of the flat output; defaults to ``len(data_index)``.

        Returns:
            Flat ``(n_out,)`` host array of likelihoods.
        """
        n_out = len(data_index) if n_out is None else n_out
        out = np.full(n_out, -1e300, dtype=np.float64)
        for sub in self.iter_gpu_batch_positions(data_index):
            out[sub] = eval_fn(sub)
        return out