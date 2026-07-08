"""EMRI-specific multi-GPU MCMC move (per-source waveform generation).

EMRI waveforms (``few.GenerateEMRIWaveform``) take **scalar floats** and use
numpy internally, so they cannot be driven through the batched, cupy-array
likelihood path of :class:`MultiGPUResidualAddRemoveMove` (which hands the
waveform a per-parameter tuple of cupy arrays for the whole split at once).

:class:`EMRISpecialMove` overrides only the per-chunk likelihood so that, within
each GPU split, sources are generated **one at a time** from numpy-float
coordinates and the resulting single-source templates are concatenated into the
batched array the C++ likelihood expects.  Cross-GPU parallelism is preserved:
each split's per-source loop runs in its own thread (``run_threaded=True``) via
:meth:`~lisatools.analysiscontainer.AnalysisContainerArray._loop_operation`, and
``batch_size_per_gpu`` (from the base move) bounds how many templates are held
per device at once.

Everything else -- the proposal loop, residual add/remove, ``batch_size_per_gpu``
chunking, per-GPU waveform replicas, ``(d|d)`` setup -- is inherited unchanged
from :class:`MultiGPUResidualAddRemoveMove`, and the batched JAX MBH path
(:class:`~lisatools.globalfit.moves.mbhspecialmove.TDMBHSpecialMove`) is not
touched.
"""

from __future__ import annotations

import logging

import numpy as np

from .addremovemove import MultiGPUResidualAddRemoveMove

logger = logging.getLogger(__name__)


class EMRISpecialMove(MultiGPUResidualAddRemoveMove):
    """Multi-GPU add/remove move for per-source (scalar-float) EMRI waveforms.

    Same constructor and behaviour as :class:`MultiGPUResidualAddRemoveMove`.
    The only difference is :meth:`_compute_like_chunk`, which replaces the
    batched, cupy-array waveform call with a per-source loop (per split) and
    re-stacks the single-source templates for the shared C++ likelihood.
    """

    def _compute_like_chunk(
        self, coords_in: np.ndarray, data_index: np.ndarray
    ) -> np.ndarray:
        """Per-source likelihood for one (already-sized) batch of rows.

        Mirrors :meth:`MultiGPUResidualAddRemoveMove._compute_like_chunk` but
        keeps the coordinates as **numpy host floats** and generates each source
        individually (EMRI cannot accept cupy arrays / batched inputs).  The
        per-source templates are concatenated per split into the
        ``(N_split, nchannels, ...)`` array consumed by ``cpp_signal_likelihood``.

        Args:
            coords_in: Transformed coordinates, shape ``(n_sources, ndim)``.
            data_index: Per-source data/AC index, shape ``(n_sources,)``.

        Returns:
            ll: Likelihood per source, shape ``(n_sources,)``.
        """
        # Route walkers to splits; keep coords as numpy host floats -- we do NOT
        # place the *parameters* on the device, because few needs scalar numpy
        # floats (cupy arrays break its numpy-internal frame transforms).
        positions_per_split, data_intra_index_per_split, _ = self.acs.unpack_indices(
            data_index
        )
        coords_per_split = self.acs.unpack_coords(
            positions_per_split, coords_in, keep_tuple=False
        )

        # The routing index must live on each split's device for the C++
        # likelihood kernel (cpp_signal_likelihood -> ascontiguousarray on
        # self.xp); only the *coordinates* stay on the host as numpy floats
        # (for few). The base path does this in prepare_inputs/place_on_device.
        (data_intra_index_per_split,) = self.acs.place_on_device(
            items=(data_intra_index_per_split,)
        )

        # One op method applied to every split; ``split_idx`` (threaded through
        # the args) selects that split's waveform replica / GPU.  _loop_operation
        # enters each split's device context and runs the splits concurrently
        # under ``run_threaded`` -- the per-source loop within a split is serial
        # (few is inherently per-source) but splits overlap across GPUs.
        likelihood_args_per_split = self.acs._loop_operation(
            operation=self._emri_signal_op,
            operation_args_per_split=[
                (coords_per_split[i], i) for i in range(self.acs.num_splits)
            ],
            positions_per_split=positions_per_split,
            run_threaded=self.run_threaded,
        )

        if not self.run_async:
            self.acs.synchronize()

        likelihoods = self.acs.cpp_signal_likelihood(
            positions_per_split=positions_per_split,
            data_intra_per_split=data_intra_index_per_split,
            noise_intra_per_split=data_intra_index_per_split,
            likelihood_args_per_split=likelihood_args_per_split,
            likelihood_kwargs={"run_async": self.run_async},
            run_threaded=self.run_threaded,
        )

        # Drop the (large) template arrays so their blocks return to the cupy
        # pool for reuse by the next chunk. Do NOT free_all_blocks() here: with
        # small per-GPU batches that would tear the pool down after every few
        # waveforms and force fresh (driver-serializing) cudaMallocs each
        # chunk. The pool is emptied once per proposal by the base move.
        del likelihood_args_per_split

        if np.any(~np.isfinite(likelihoods)):
            logger.warning(
                f"Non-finite likelihoods encountered: {likelihoods}. "
                f"This could be a sign of numerical issues."
            )

        return np.where(np.isfinite(likelihoods), likelihoods, -1e300)

    def get_split_inds(self) -> np.ndarray:
        """Half-split the ensemble with per-GPU-balanced membership.

        The walker -> AC-entry map is contiguous per split
        (``AnalysisContainerArray.gpu_splits`` uses ``np.split``), so eryn's
        default full-row shuffle makes each half's per-GPU share
        hypergeometric: the number of sequential ``batch_size_per_gpu`` chunks
        per likelihood call is set by the *largest* split share, and the tail
        chunks drive one GPU while the others idle. Shuffling within each
        split's contiguous block instead keeps every half exactly balanced
        across GPUs (same idea as ``TDMBHSpecialMove.get_split_inds``, but
        indexed by split position so non-contiguous device ids work too).
        """
        all_inds = np.tile(np.arange(self.nwalkers), (self.ntemps, 1))
        inds = all_inds % self.nsplits
        if self.randomize_split:
            if self.acs.gpus is None:
                for row in inds:
                    np.random.shuffle(row)
            else:
                num_splits = len(self.acs.gpus)
                num_per_split = self.nwalkers // num_splits
                for row in inds:
                    for k in range(num_splits):
                        np.random.shuffle(
                            row[k * num_per_split:(k + 1) * num_per_split]
                        )
        return inds

    def _emri_signal_op(self, coords_split: np.ndarray, split_idx: int):
        """Generate split ``split_idx``'s sources one at a time and stack them.

        Runs inside ``_loop_operation``'s device context for ``split_idx`` (so
        every generated array lands on that split's GPU).  Each source is
        produced from **scalar numpy floats** via the split's waveform replica's
        ``waveform_like_method`` (``__call__``), which routes through the
        waveform's single-source path and returns
        ``(signal (1, nch, ...), start_freqs (1,)[, start_times (1,)])``.  The
        per-source outputs are concatenated along axis 0 into the
        ``(template_vals, start_freqs[, start_times])`` tuple consumed by
        :meth:`~lisatools.analysiscontainer.AnalysisContainerArray.cpp_signal_likelihood`.

        Args:
            coords_split: Transformed coordinates for this split's sources,
                shape ``(n_sources_in_split, ndim)`` (numpy host, scalar floats).
            split_idx: Index of the GPU split (selects the waveform replica).

        Returns:
            ``(template_vals, start_freqs)`` for FD, or
            ``(template_vals, start_freqs, start_times)`` for STFT/WDM, with
            ``template_vals`` of shape ``(n_sources_in_split, nchannels, ...)``.
        """
        like = getattr(self.waveform_generators[split_idx], self.waveform_like_method)
        xp = self.waveform_generators[split_idx].xp
        like_kwargs = self.waveform_like_kwargs

        signals, start_freqs, start_times = [], [], []
        for row in coords_split:
            out = like(*row, **like_kwargs)
            signals.append(out[0])
            start_freqs.append(out[1])
            if len(out) > 2:
                start_times.append(out[2])

        template_vals = xp.concatenate(signals, axis=0)
        start_freqs = xp.concatenate(start_freqs)
        if start_times:
            return template_vals, start_freqs, xp.concatenate(start_times)
        return template_vals, start_freqs
