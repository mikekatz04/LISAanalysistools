"""Band-level infrastructure for the GB special moves.

This module owns the sub-band machinery that the GB proposal moves in
:mod:`gbspecialstretch` drive:

* :func:`pack_special_index` / :func:`unpack_special_index` -- the single
  home of the ``(temp * nwalkers + walker) * 1e6 + band`` encoding used to
  identify a (temperature, walker, band) cell of the ensemble.
* :class:`SubBandBuffer` -- the per-cell residual / PSD / template scratch
  buffer. It **is** an :class:`~lisatools.analysiscontainer.AnalysisContainerArray`
  (one :class:`~lisatools.analysiscontainer.AnalysisContainer` per active
  cell) extended with fast GB source injection/removal through a
  :class:`~lisatools.globalfit.moves.gb_likelihood.BandLikelihoodEngine`.
  ``Buffer`` is kept as a back-compat alias.
* :class:`BandSorter` -- flat per-source view of the eryn GB branch
  (coords / inds / temp / walker / leaf / band index arrays) with subset
  and RJ pre-draw machinery.
"""

from __future__ import annotations

import logging
import os
import warnings
from copy import deepcopy
from types import ModuleType
from typing import Optional, Tuple, Union

import numpy as np
import numpy

try:
    import cupy as cp
    import cupy

    gpu_available = True
except ModuleNotFoundError:
    import numpy as cp

    gpu_available = False

from eryn.state import Branch
from eryn.utils import TransformContainer

from ...analysiscontainer import (
    AnalysisContainer,
    AnalysisContainerArray,
    BandView,
    band_gpu_assignment,
)
from ...domains import DomainSettingsBase, FDSettings, WDMSettings
from ...sensitivity import SensitivityMatrixBase
from ...utils.device import device_context
from ...utils.parallelbase import LISAToolsParallelModule
from ...utils.utility import asnumpy, get_array_module

__all__ = [
    "pack_special_index",
    "unpack_special_index",
    "SubBandBuffer",
    "Buffer",
    "BandSorter",
    "BandScheduler",
]

logger = logging.getLogger(__name__)

_to_numpy = asnumpy

# Encoding base for the band part of a special index. Bands live in
# ``[0, _SPECIAL_INDEX_BASE)``; everything above encodes (temp, walker).
_SPECIAL_INDEX_BASE = int(1e6)

# Task-b default WDM-layer leakage half-width for auto-sized per-band slabs.
# A near-monochromatic GB's WDM energy is captured by the chunked-het kernel
# over ``m_floor +/- m_band_half_width`` (default 1 -> 3 layers), which alone
# reaches median mm5 ~1e-9; the canonical mm5 band spans 5 layers. Beyond +/-2
# layers the energy is < ~1e-7 for the recommended Tukey window (alpha
# 0.01-0.05), so 2 neighbor layers each side covers the leakage conservatively.
# ``scripts/diagnostics/check_wdm_band_slab.py`` measures this directly.
_WDM_SLAB_LEAKAGE_LAYERS = 2


def pack_special_index(temp_inds, walker_inds, band_inds, nwalkers: int):
    """Pack ``(temp, walker, band)`` triplets into scalar special indices.

    ``special = (temp * nwalkers + walker) * 1e6 + band``. Works
    elementwise on array inputs of any matching shape.
    """
    return (temp_inds * nwalkers + walker_inds) * _SPECIAL_INDEX_BASE + band_inds


def unpack_special_index(special_band_inds, nwalkers: int) -> tuple:
    """Recover ``(temp, walker, band)`` arrays from packed special indices."""
    # input-driven array module (NOT the module-level cp): this helper runs
    # on numpy inputs during CPU-resolved runs on cupy-installed machines.
    xp = get_array_module(special_band_inds)
    temp_walker_inds = xp.floor(special_band_inds / _SPECIAL_INDEX_BASE).astype(int)
    temp_inds = temp_walker_inds // nwalkers
    walker_inds = temp_walker_inds % nwalkers
    band_inds = (special_band_inds - temp_walker_inds * _SPECIAL_INDEX_BASE).astype(int)
    return (temp_inds, walker_inds, band_inds)


def return_x(x):
    """Identity helper used as a no-op replacement for :func:`copy.deepcopy`."""
    return x


class BandScheduler:
    """Staged loading of (temp, walker, band) cells through the sub-band buffer.

    The proposal loop runs one source-pick per active cell per round. This
    object owns the bookkeeping the loop needs:

    * which cells (packed special indices) currently occupy the buffer's
      ``n_subbands`` slots,
    * how many of each cell's sources have been consumed (a cell is finished
      when every one of its sources has been picked exactly once),
    * which buffer slots to swap out for pending cells when their cell
      finishes (:meth:`advance` returns the ``(inds_fill, new_specials)``
      pair that :meth:`SubBandBuffer` loading consumes).

    Cells are ordered by ascending source count so short cells retire early
    and the buffer stays densely packed with work.
    """

    @property
    def xp(self):
        """Array module derived from a stored flag — never the module itself
        (raw module attributes break deepcopy/pickle of containing graphs)."""
        return cp if self._uses_cupy else np

    def __init__(self, special_band_inds, n_subbands, xp=np):
        # Store a flag, not the module (see the ``xp`` property).
        self._uses_cupy = (getattr(xp, "__name__", "numpy") == "cupy")
        uni, counts = xp.unique(special_band_inds, return_counts=True)
        order = xp.argsort(counts)
        self.cell_specials = uni[order]
        self.cell_counts = counts[order]
        self.cell_run = xp.zeros_like(self.cell_counts)
        self.n_cells = int(len(uni))
        # lookup table: special index -> cell position (cell_specials order)
        self._lookup_order = xp.argsort(self.cell_specials)
        self._specials_sorted = self.cell_specials[self._lookup_order]

        n_slots = min(int(n_subbands), self.n_cells)
        self.slot_cell = xp.arange(n_slots)
        self.slot_active = xp.ones(n_slots, dtype=bool)
        self._next_cell = n_slots

    def _cells_of(self, specials):
        """Map special indices to cell positions."""
        pos = self.xp.searchsorted(self._specials_sorted, specials, side="left")
        return self._lookup_order[pos]

    @property
    def n_slots(self) -> int:
        return len(self.slot_cell)

    @property
    def slot_specials(self):
        """Special index per buffer slot (retired slots keep their last cell)."""
        return self.cell_specials[self.slot_cell]

    @property
    def active_slot_specials(self):
        """Special indices of the slots still doing work."""
        return self.slot_specials[self.slot_active]

    def any_active(self) -> bool:
        return bool(self.xp.any(self.slot_active))

    def record_picks(self, picked_specials) -> None:
        """Count one consumed source for each cell that got a pick."""
        self.cell_run[self._cells_of(picked_specials)] += 1

    def advance(self, frozen_specials=None):
        """Retire finished slots and stage pending cells into them.

        Returns ``(inds_fill, new_specials)``: the buffer slot positions to
        repack and the special indices of the cells to load there. Slots
        with no pending replacement are deactivated.

        ``frozen_specials`` (grouped RJ scheduling, 2026-08-14): cells
        holding a pending in-model source are never retired even when all
        their sources are consumed — a pending source pins its cell's
        buffer slot until the flush runs. Passing the frozen set lets the
        caller stage new cells into the OTHER finished slots so the pool
        keeps accumulating toward a full-width in-model block.
        """
        finished = self.slot_active & (
            self.cell_run[self.slot_cell] >= self.cell_counts[self.slot_cell]
        )
        if frozen_specials is not None and len(frozen_specials):
            finished &= ~self.xp.isin(self.slot_specials, frozen_specials)
        n_finished = int(finished.sum())
        if n_finished == 0:
            return self.xp.zeros(0, dtype=int), self.xp.zeros(0, dtype=int)

        n_pending = self.n_cells - self._next_cell
        n_replace = min(n_finished, n_pending)
        finished_slots = self.xp.arange(self.n_slots)[finished]

        inds_fill = finished_slots[:n_replace]
        new_cells = self._next_cell + self.xp.arange(n_replace)
        self.slot_cell[inds_fill] = new_cells
        self._next_cell += n_replace

        # slots beyond the replacements retire
        self.slot_active[finished_slots[n_replace:]] = False
        return inds_fill, self.cell_specials[new_cells]


class _ShardHolderView:
    """Single-shard holder view over one GPU split of a multi-shard ACA.

    The gbgpu band engines are single-shard by contract: they consume
    ``holder.linear_data_arr[0]`` / ``linear_psd_arr[0]``, index rows by an
    intra-buffer ``data_index``, and cache pointer-bound bindings on the
    holder. This view presents ONE split of a multi-shard
    :class:`~lisatools.analysiscontainer.AnalysisContainerArray` (a
    :class:`SubBandBuffer` or the parent residual ACA) through exactly that
    protocol:

    * ``linear_data_arr`` / ``linear_psd_arr`` are one-element lists holding
      the owning split's live buffer (zero-copy);
    * ``acs_total_entries`` is the split's row count — engine row indices
      are INTRA-shard (:class:`_RoutedBandEngine` translates);
    * ``min_freq_inds`` / ``start_freq_ind`` / ``slab_min_f`` are persistent
      per-shard stores refreshed IN PLACE from the parent
      (:meth:`refresh_row_metadata`) so the engines' pointer-binding contract
      survives cell swaps;
    * everything else (settings, df, xp, ...) delegates to the parent.

    Engine bindings (``_gb_fd_binding``) cache on this object, so a view
    must live exactly as long as its shard buffers: the router stores views
    on the holder itself (``holder._shard_holder_views``), which dies with
    the holder at proposal teardown (memory-lifecycle rule).
    """

    def __init__(self, parent, split_index: int):
        self._parent = parent
        self._split = int(split_index)
        rows = np.asarray(asnumpy(parent.gpu_splits[self._split]), dtype=int)
        self._rows = rows
        self.acs_total_entries = int(rows.shape[0])
        self.device = (
            None if parent.gpus is None else int(parent.gpus[self._split])
        )
        self.gpus = None if parent.gpus is None else [self.device]
        self.gpu_splits = [np.arange(rows.shape[0])]
        self.split_map = np.zeros(rows.shape[0], dtype=int)
        self.gpu_map = (
            np.zeros(rows.shape[0], dtype=int)
            if self.device is None
            else np.full(rows.shape[0], self.device, dtype=int)
        )
        self._min_freq_inds_view = None
        self._start_freq_ind_view = None
        self._slab_min_f_view = None
        self.refresh_row_metadata()

    @property
    def rows(self):
        """Global row ids owned by this shard (ascending)."""
        return self._rows

    @property
    def linear_data_arr(self):
        return [self._parent.linear_data_arr[self._split]]

    @property
    def linear_psd_arr(self):
        return [self._parent.linear_psd_arr[self._split]]

    @property
    def data_shaped(self):
        return [self._parent.data_shaped[self._split]]

    @property
    def psd_shaped(self):
        return [self._parent.psd_shaped[self._split]]

    @property
    def xp(self):
        return self._parent.xp

    @property
    def min_freq_inds(self):
        return self._min_freq_inds_view

    @property
    def start_freq_ind(self):
        return self._start_freq_ind_view

    @property
    def slab_min_f(self):
        """Per-slot narrow-slab layer origins SLICED to this shard's rows.

        MUST be an explicit property: ``__getattr__`` delegation would hand
        back the parent's **global-slot** array while every index the engines
        pass is **intra-shard** -- so a source in shard-1 row 0 would be
        folded against buffer slot 0's slab origin instead of its own. The
        consumers index it exactly that way: the chunked-het kernels via
        ``WDMComputationsBase._slab_kernel_args`` (``get_ll`` / ``swap_ll`` /
        ``fill_global`` / ``get_fstat_ll``) and
        ``GBSignalHetComputations.setup_in_model``
        (``slab_min_f[slots] - ind_min_f``). Mirrors the per-slot kwarg slice
        :meth:`_RoutedBandEngine.fill_template` already applies for the kwarg
        form. ``None`` when the parent carries no slab metadata (narrow slabs
        off, or the parent residual ACA).

        ``band_slab_Nf`` needs no such override: it is a scalar extent shared
        by every slab, hence shard-invariant, and keeps delegating.
        """
        return self._slab_min_f_view

    def refresh_row_metadata(self) -> None:
        """Re-slice per-row metadata from the parent.

        Updates the persistent per-shard ``min_freq_inds`` / ``slab_min_f``
        stores IN PLACE (the FD binding holds a pointer to the former)
        instead of rebinding.
        """
        xp = self._parent.xp
        starts = getattr(self._parent, "min_freq_inds", None)
        if starts is None:
            self._min_freq_inds_view = None
        else:
            vals_host = np.ascontiguousarray(
                np.asarray(asnumpy(starts))[self._rows].astype(np.int32)
            )
            with device_context(xp, self.device):
                if (
                    self._min_freq_inds_view is not None
                    and self._min_freq_inds_view.shape == vals_host.shape
                ):
                    self._min_freq_inds_view[...] = xp.asarray(vals_host)
                else:
                    self._min_freq_inds_view = xp.ascontiguousarray(
                        xp.asarray(vals_host)
                    )
        sfi = getattr(self._parent, "start_freq_ind", None)
        if sfi is None:
            self._start_freq_ind_view = None
        else:
            arr = np.asarray(asnumpy(sfi))
            self._start_freq_ind_view = arr[self._rows] if arr.ndim else arr
        # Narrow per-band slab origins: one value PER BUFFER SLOT, so the
        # shard's view must carry its own rows' values in intra-shard order
        # (see the ``slab_min_f`` property). Refreshed in place like
        # ``min_freq_inds`` so a cell swap on the parent reaches every view.
        slab = getattr(self._parent, "slab_min_f", None)
        if slab is None:
            self._slab_min_f_view = None
        else:
            slab_host = np.ascontiguousarray(
                np.asarray(asnumpy(slab))[self._rows].astype(np.int32)
            )
            with device_context(xp, self.device):
                if (
                    self._slab_min_f_view is not None
                    and self._slab_min_f_view.shape == slab_host.shape
                ):
                    self._slab_min_f_view[...] = xp.asarray(slab_host)
                else:
                    self._slab_min_f_view = xp.ascontiguousarray(
                        xp.asarray(slab_host)
                    )

    def __len__(self) -> int:
        # The engines read ``len(holder)`` as the shard's row (cell) count
        # -> ``num_data``/``num_noise`` for the flat single-shard buffer.
        # MUST be an explicit dunder: ``len()`` resolves ``__len__`` on the
        # TYPE, bypassing ``__getattr__`` delegation, so a parent-forwarded
        # ``__len__`` is never seen. Mirrors AnalysisContainerArray.__len__
        # (== the number of containers on this split).
        return int(self.acs_total_entries)

    def __getattr__(self, name):
        # Guard dunder/underscore probing (deepcopy/pickle safety rule);
        # delegate the public long tail (settings, df, nchannels, ...).
        # NOTE: implicitly-invoked dunders (len(), iter(), ...) resolve on
        # the type and never reach here -- define each one explicitly above.
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._parent, name)


class _FStatRefRowHolder:
    """One-slab wdm_holder replica feeding one device's sig-het F-stat lane.

    ``GBSignalHetComputations.setup_fstat_references`` consumes exactly
    ``linear_data_arr[0]`` / ``linear_psd_arr[0]`` (full-band slab layouts)
    picked by SCALAR ``data_index`` / ``noise_index``, and
    ``get_fstat_ll_wdm`` never touches the holder at all -- so a scoring
    lane needs only the ONE reference walker's residual + inverse-PSD rows,
    copied onto its device once at adapter build (host-routed, no P2P) and
    presented as a single-slab holder scored with ``data_index=0``.
    Private to :meth:`_RoutedBandEngine._sighet_fstat_multidevice`;
    ``device`` / ``gpus`` / ``__len__`` mirror :class:`_ShardHolderView` so
    the router's replica plumbing (``_comp_for``) accepts it as a view.
    Lives only in the returned ``call_fstat`` closure -- it dies with the
    fit, and the copies never reach the settings tree.
    """

    def __init__(self, parent, device, data_row, psd_row):
        self._parent = parent
        self.device = None if device is None else int(device)
        self.gpus = None if self.device is None else [self.device]
        self.acs_total_entries = 1
        self.linear_data_arr = [data_row]
        self.linear_psd_arr = [psd_row]

    @property
    def xp(self):
        return self._parent.xp

    def __len__(self) -> int:
        # The comps read ``len(holder)`` as the slab count (explicit dunder:
        # len() resolves on the type, bypassing ``__getattr__``).
        return 1

    def __getattr__(self, name):
        # Same delegation discipline as _ShardHolderView: guard underscore
        # probing, delegate the public long tail to the parent ACA.
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._parent, name)


class _RoutedBandEngine:
    """Multi-shard router in front of a single-shard band likelihood engine.

    Wraps a :func:`gbgpu.gb_likelihood.make_band_likelihood_engine` product.
    Single-shard holders pass straight through (no overhead). For
    multi-shard holders each call's rows are partitioned by owning split
    (``holder.split_map``), run per shard inside the owning device context
    against a persistent :class:`_ShardHolderView`, and the outputs are
    reassembled full-length on the caller's device. Cross-shard movement is
    host-routed (no P2P), matching the ACA conventions. Per-launch
    host->device upload of wrapper structs (LISA Analysis Tools-wide
    convention) keeps kernel config device-local under each context.

    The GB comps a shard's kernels read (chunk geometry, WDM window,
    ``OrbitsWrap`` / ``TDIConfigWrap`` pointer fields, and under
    ``GB_SIGHET_INMODEL`` the whole heterodyne reference stash) are allocated
    once, on the device current at variant-build time. A shard launching on a
    different device would dereference them across the PCIe link -- a silent
    peer-access tax with P2P, an illegal access without it -- and, for
    sig-het, would share ONE reference stash, ONE slot->reference map and ONE
    ``_in_model`` flag between shards whose slot ids both start at zero. Both
    are closed by per-device replicas: ``engine_factory`` (supplied by the
    two construction sites) rebuilds the whole engine around device-local
    comps for any shard whose device differs from the prototype's, and the
    raw-comp class methods resolve the same replica through
    :meth:`_comp_for`. The prototype's own device reuses the existing engine
    object, so single-shard / primary-shard behaviour is unchanged and
    allocates nothing.
    """

    #: fill_template kwargs holding one value PER BUFFER SLOT — sliced to the
    #: shard's rows so intra-shard indexing stays aligned.
    _PER_SLOT_KWARGS = ("slab_min_f",)

    def __init__(self, engine, engine_factory=None):
        self._engine = engine
        # device -> engine replica, populated lazily on the first multi-shard
        # call that lands on a non-prototype device. The prototype's device
        # maps to ``engine`` ITSELF (never a copy), so ``len(gpus) <= 1``
        # returns the same object and allocates nothing.
        self._engine_factory = engine_factory
        self._engine_by_device = {}

    @property
    def wrapped_engine(self):
        """The underlying single-shard engine."""
        return self._engine

    @property
    def device_engines(self) -> dict:
        """``{device: engine replica}`` built so far (diagnostics/tests)."""
        return self._engine_by_device

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._engine, name)

    # ---------------- per-device comp / engine replicas ----------------

    @staticmethod
    def _comp_build_device(comp):
        """The CUDA device a GB comp's buffers were allocated on, or None.

        ``_build_device`` is recorded by ``WDMComputationsBase.__init__`` and
        ``GBFDComputations.__init__``. The sig-het wrapper records none of its
        own -- ``for_band_engine`` runs in the same device context as its
        chunked delegate -- so it reports the delegate's. The final fallback
        reads residency straight off a known device buffer, which keeps the
        answer meaningful for a comp built before the recording existed.
        """
        dev = getattr(comp, "_build_device", None)
        if dev is None:
            dev = getattr(getattr(comp, "chunked", None), "_build_device", None)
        if dev is None:
            dev = getattr(getattr(comp, "wdm_window", None), "device", None)
            dev = getattr(dev, "id", None)
        return None if dev is None else int(dev)

    @classmethod
    def _engine_comp(cls, engine):
        """The GB comp an engine scores through (WDM or FD), or None."""
        comp = getattr(engine, "gb_comps", None)
        if comp is None:
            comp = getattr(engine, "gb_fd_comp", None)
        return comp

    @classmethod
    def _primary_device(cls, comp, holder):
        """Device whose shard reuses ``comp`` unchanged.

        The comp's OWN build device when it recorded one -- not blindly
        ``gpus[0]``: a comp constructed before the run pinned its main device
        lives on device 0 even when ``gpus=[2, 3]``, and keying on the
        recorded value replicates for every shard (correct) instead of
        handing device-0 pointers to the ``gpus[0]`` shard (wrong).
        """
        dev = None if comp is None else cls._comp_build_device(comp)
        if dev is not None:
            return dev
        gpus = getattr(holder, "gpus", None)
        return None if not gpus else int(gpus[0])

    @classmethod
    def _assert_comp_device(cls, comp, view):
        """Fail loudly when a shard is about to launch on foreign buffers.

        Cheap permanent guard: the moment a new comp-level device buffer is
        added without a matching replica path, this turns "mysteriously slow,
        or an illegal access on a non-P2P node" into a message that names the
        fix. A comp that records nothing (CPU, or an unrecognised comp type)
        is skipped rather than guessed at.
        """
        dev = getattr(view, "device", None)
        if comp is None or dev is None:
            return
        build_dev = cls._comp_build_device(comp)
        if build_dev is not None and build_dev != int(dev):
            raise RuntimeError(
                f"GB comp {type(comp).__name__} holds buffers on device "
                f"{build_dev} but this shard launches on device {dev}: the "
                "kernel would read across devices (a silent P2P tax, or an "
                "illegal access on a node without peer access). A per-device "
                "comp replica is needed -- see _device_local_gb_comp in "
                "lisatools.globalfit.stock.erebor.source_runtime."
            )

    @classmethod
    def _comp_for(cls, comp, holder, view):
        """The device-local replica of ``comp`` for this shard's device."""
        dev = getattr(view, "device", None)
        if comp is None or dev is None:
            return comp
        from ..stock.erebor.source_runtime import _device_local_gb_comp

        out = _device_local_gb_comp(
            comp, holder.xp, int(dev), cls._primary_device(comp, holder)
        )
        cls._assert_comp_device(out, view)
        return out

    def _engine_for(self, holder, view):
        """The likelihood engine this shard must run on.

        The prototype's device (and any holder/engine without the metadata to
        do better) gets ``self._engine`` itself. Every other device gets one
        cached replica built through ``engine_factory`` -- which rebuilds the
        engine around ``_device_local_gb_comp`` comps -- so the shard's
        kernels, its coefficient stash and its ``_in_model`` state are all
        its own.
        """
        dev = getattr(view, "device", None)
        if dev is None or self._engine_factory is None:
            return self._engine
        primary = self._primary_device(self._engine_comp(self._engine), holder)
        if primary is not None and int(dev) == int(primary):
            return self._engine
        engine = self._engine_by_device.get(int(dev))
        if engine is None:
            engine = self._engine_factory(int(dev), primary)
            self._engine_by_device[int(dev)] = engine
        self._assert_comp_device(self._engine_comp(engine), view)
        return engine

    # ---------------- shard bookkeeping ----------------

    @staticmethod
    def _is_multi(holder) -> bool:
        return len(holder.linear_data_arr) > 1

    @staticmethod
    def _shard_views(holder):
        views = getattr(holder, "_shard_holder_views", None)
        n = len(holder.linear_data_arr)
        if views is None or len(views) != n:
            views = [_ShardHolderView(holder, s) for s in range(n)]
            holder._shard_holder_views = views
        else:
            for v in views:
                v.refresh_row_metadata()
        return views

    @staticmethod
    def _partition(holder, data_index, noise_index=None):
        """Per-shard ``(positions, intra_data, intra_noise)`` partition.

        ``positions`` index into the call's row batch; ``intra_*`` are the
        corresponding intra-shard buffer rows. ``data_index`` and
        ``noise_index`` rows must be co-located on the same shard.
        """
        idx = np.asarray(asnumpy(data_index), dtype=int)
        split_map = np.asarray(asnumpy(holder.split_map), dtype=int)
        intra = np.empty(int(holder.acs_total_entries), dtype=int)
        for rows in holder.gpu_splits:
            rr = np.asarray(asnumpy(rows), dtype=int)
            intra[rr] = np.arange(rr.shape[0])
        nidx = None
        if noise_index is not None:
            nidx = np.asarray(asnumpy(noise_index), dtype=int)
            if not np.array_equal(split_map[idx], split_map[nidx]):
                raise ValueError(
                    "data_index and noise_index rows must live on the same "
                    "shard (cross-shard noise rows are unsupported)."
                )
        parts = []
        for s in range(len(holder.linear_data_arr)):
            pos = np.where(split_map[idx] == s)[0]
            parts.append((
                pos,
                intra[idx[pos]],
                None if nidx is None else intra[nidx[pos]],
            ))
        return parts

    @staticmethod
    def _assemble(num, pieces, default, xp):
        """Host-assemble per-shard outputs into one xp array (row order).

        ``pieces`` is ``[(positions, host_values_or_None), ...]``. Returns
        None when every shard produced None (e.g. ``phase_angle`` without
        phase maximisation).
        """
        first = next(
            (np.asarray(v) for _, v in pieces if v is not None), None
        )
        if first is None:
            return None
        out = np.full(
            (num,) + tuple(first.shape[1:]), default, dtype=first.dtype
        )
        for pos, vals in pieces:
            if vals is None or len(pos) == 0:
                continue
            out[pos] = np.asarray(vals)
        return xp.asarray(out)

    @staticmethod
    def _dispatch_shards(holder, items, worker, state_ids=None):
        """Run ``worker(*item)`` once per populated-shard work item.

        THREADED by default (2026-08-13, user ruling): the 2-GPU
        incremental-ll-drift investigation that justified the serial safety
        net CLOSED 2026-08-12 (three root causes fixed, comp replicas
        GPU-verified bit-identical), so shards now run concurrently -- one
        host thread per shard with work, each entering its own device
        context (cupy's current device is thread-local; kernel launches
        release the GIL), mirroring
        ``AnalysisContainerArray._run_per_split``. Callers pre-resolve
        engines/comp replicas and pre-size one result slot per shard BEFORE
        dispatch, so the threaded path shares no mutable state beyond
        disjoint slot writes and reassembly stays deterministic (the
        serial and threaded paths write identical slots).

        ``state_ids``: one hashable per item naming the stateful object the
        item runs on (engine / comp replica). Items sharing one (missing
        engine_factory, two shards on one device, CPU fakes) would race
        that object's output attrs, so any duplicate forces serial.
        ``holder.thread_pool`` is only touched on the threaded branch (the
        real ACA property allocates the pool on first access).
        ``GB_ROUTER_THREADED=0`` restores serial launch-sync-launch
        dispatch (the drift checks / [GB_CELL_LL] reconciles are the
        regression alarms if concurrency ever misbehaves).
        """
        if (len(items) > 1
                and os.environ.get("GB_ROUTER_THREADED", "1") == "1"
                and (state_ids is None or len(set(state_ids)) == len(items))
                and getattr(holder, "thread_pool", None) is not None):
            futures = [holder.thread_pool.submit(worker, *it) for it in items]
            for f in futures:
                f.result()  # re-raise worker exceptions in caller
        else:
            for it in items:
                worker(*it)

    def _mirror_engine_outputs(self):
        """Refresh routed-output attrs from the wrapped engine after a
        passthrough call so stale routed values never shadow them."""
        for name in ("d_h_out", "h_h_out", "phase_angle", "kept_out"):
            if hasattr(self._engine, name):
                setattr(self, name, getattr(self._engine, name))

    # ---------------- routed engine protocol ----------------

    def fill_template(self, holder, params_phys, params_index, N_vals, *,
                      factor, waveform_kwargs, **kwargs):
        if not self._is_multi(holder):
            return self._engine.fill_template(
                holder, params_phys, params_index, N_vals,
                factor=factor, waveform_kwargs=waveform_kwargs, **kwargs)
        xp = holder.xp
        views = self._shard_views(holder)
        parts = self._partition(holder, params_index)
        params_host = asnumpy(params_phys)
        N_host = None if N_vals is None else asnumpy(N_vals)
        slot_kwargs_host = {
            k: np.asarray(asnumpy(kwargs[k]))
            for k in self._PER_SLOT_KWARGS
            if kwargs.get(k) is not None
        }

        def _shard(view, engine, pos, intra):
            kw_s = dict(kwargs)
            with device_context(xp, view.device):
                for k, host_vals in slot_kwargs_host.items():
                    kw_s[k] = xp.asarray(host_vals[view.rows])
                engine.fill_template(
                    view, xp.asarray(params_host[pos]), intra,
                    None if N_host is None else xp.asarray(N_host[pos]),
                    factor=factor, waveform_kwargs=waveform_kwargs, **kw_s)

        items = [
            (view, self._engine_for(holder, view), pos, intra)
            for view, (pos, intra, _) in zip(views, parts)
            if pos.shape[0]
        ]
        self._dispatch_shards(holder, items, _shard,
                              state_ids=[id(it[1]) for it in items])
        # Drift-hunt debug fence (see gbspecialstretch run_proposal): the
        # fills above are enqueued on each shard's own device stream; a
        # caller that immediately reads the filled shard from another
        # device (peer access) is not ordered against them.
        if (os.environ.get("GB_MULTIGPU_SYNC_DEBUG", "0") == "1"
                and getattr(xp, "cuda", None) is not None):
            for view in views:
                if view.device is not None:
                    with device_context(xp, view.device):
                        xp.cuda.runtime.deviceSynchronize()

    def get_ll(self, holder, params_phys, *, data_index, noise_index,
               N_vals, phase_maximize=False, waveform_kwargs, **kwargs):
        if not self._is_multi(holder):
            out = self._engine.get_ll(
                holder, params_phys, data_index=data_index,
                noise_index=noise_index, N_vals=N_vals,
                phase_maximize=phase_maximize,
                waveform_kwargs=waveform_kwargs, **kwargs)
            self._mirror_engine_outputs()
            return out
        xp = holder.xp
        views = self._shard_views(holder)
        parts = self._partition(holder, data_index, noise_index)
        num = int(params_phys.shape[0])
        params_host = asnumpy(params_phys)
        N_host = None if N_vals is None else asnumpy(N_vals)
        items = [
            (si, view, self._engine_for(holder, view), pos, intra,
             intra_noise)
            for si, (view, (pos, intra, intra_noise))
            in enumerate(zip(views, parts))
            if pos.shape[0]
        ]
        # One pre-sized slot per shard (threaded dispatch writes disjoint
        # slots; serial dispatch fills the same slots in the same order).
        slots = {name: [None] * len(views)
                 for name in ("ll", "dh", "hh", "ang", "kept")}

        def _shard(si, view, engine, pos, intra, intra_noise):
            with device_context(xp, view.device):
                ll_s = engine.get_ll(
                    view, xp.asarray(params_host[pos]),
                    data_index=intra,
                    noise_index=intra if intra_noise is None else intra_noise,
                    N_vals=None if N_host is None else xp.asarray(N_host[pos]),
                    phase_maximize=phase_maximize,
                    waveform_kwargs=waveform_kwargs, **kwargs)
                slots["ll"][si] = (pos, asnumpy(ll_s))
                slots["dh"][si] = (pos, asnumpy(engine.d_h_out))
                slots["hh"][si] = (pos, asnumpy(engine.h_h_out))
                ang = getattr(engine, "phase_angle", None)
                slots["ang"][si] = (pos,
                                    None if ang is None else asnumpy(ang))
                kept = getattr(engine, "kept_out", None)
                slots["kept"][si] = (pos,
                                     None if kept is None else asnumpy(kept))

        self._dispatch_shards(holder, items, _shard,
                              state_ids=[id(it[2]) for it in items])
        ll_p, dh_p, hh_p, ang_p, kept_p = (
            [p for p in slots[name] if p is not None]
            for name in ("ll", "dh", "hh", "ang", "kept"))
        ll = self._assemble(num, ll_p, -1e300, xp)
        if ll is None:
            ll = xp.full(num, -1e300)
        self.d_h_out = self._assemble(num, dh_p, 0.0, xp)
        self.h_h_out = self._assemble(num, hh_p, 0.0, xp)
        self.phase_angle = self._assemble(num, ang_p, 0.0, xp)
        kept_arr = self._assemble(num, kept_p, False, xp)
        self.kept_out = (
            xp.ones(num, dtype=bool) if kept_arr is None else kept_arr
        )
        return ll

    def get_swap_ll(self, holder, params_remove_phys, params_add_phys, *,
                    data_index, noise_index, N_vals, phase_maximize=False,
                    waveform_kwargs, **kwargs):
        if not self._is_multi(holder):
            return self._engine.get_swap_ll(
                holder, params_remove_phys, params_add_phys,
                data_index=data_index, noise_index=noise_index,
                N_vals=N_vals, phase_maximize=phase_maximize,
                waveform_kwargs=waveform_kwargs, **kwargs)
        from gbgpu.gb_likelihood import SwapLLResult

        xp = holder.xp
        views = self._shard_views(holder)
        parts = self._partition(holder, data_index, noise_index)
        num = int(params_add_phys.shape[0])
        rem_host = asnumpy(params_remove_phys)
        add_host = asnumpy(params_add_phys)
        N_host = None if N_vals is None else asnumpy(N_vals)
        fields = ("ll_diff", "d_h_add", "d_h_remove", "hh_add",
                  "hh_remove", "hh_cross", "opt_snr_add", "phase_angle",
                  "kept")
        items = [
            (si, view, self._engine_for(holder, view), pos, intra,
             intra_noise)
            for si, (view, (pos, intra, intra_noise))
            in enumerate(zip(views, parts))
            if pos.shape[0]
        ]
        # Pre-sized per-shard slots (see get_ll).
        pieces = {f: [None] * len(views) for f in fields}

        def _shard(si, view, engine, pos, intra, intra_noise):
            with device_context(xp, view.device):
                res = engine.get_swap_ll(
                    view, xp.asarray(rem_host[pos]), xp.asarray(add_host[pos]),
                    data_index=intra,
                    noise_index=intra if intra_noise is None else intra_noise,
                    N_vals=None if N_host is None else xp.asarray(N_host[pos]),
                    phase_maximize=phase_maximize,
                    waveform_kwargs=waveform_kwargs, **kwargs)
                for f in fields:
                    v = getattr(res, f)
                    pieces[f][si] = (pos, None if v is None else asnumpy(v))

        self._dispatch_shards(holder, items, _shard,
                              state_ids=[id(it[2]) for it in items])
        defaults = dict(ll_diff=-1e300, opt_snr_add=0.0, kept=False)
        out = {}
        for f in fields:
            out[f] = self._assemble(
                num, [p for p in pieces[f] if p is not None],
                defaults.get(f, 0.0), xp)
        if out["ll_diff"] is None:
            out["ll_diff"] = xp.full(num, -1e300)
        if out["opt_snr_add"] is None:
            out["opt_snr_add"] = xp.zeros(num)
        if out["kept"] is None:
            out["kept"] = xp.zeros(num, dtype=bool)
        return SwapLLResult(**out)

    def setup_in_model(self, holder, params_phys, data_index, N_vals=None):
        """Route the per-shard in-model reference build (sig-het).

        A truthy return means the comp built heterodyne references: a
        coefficient stash, a slot->reference map and an ``_in_model`` flag,
        all held on the comp and all indexed by INTRA-shard slot ids. Two
        shards therefore need two comps -- their slot ids both start at zero,
        so a shared comp would have the second shard's build silently PATCH
        the first shard's references (the ``_in_model`` flag makes the second
        call take the mid-block patch branch), and every subsequent
        ``get_ll`` would resolve references through the wrong map. With
        ``engine_factory`` supplied each shard resolves to its own engine and
        its own comp, so each takes the fresh-build branch against its own
        residual slabs. Without one, the collision is refused loudly rather
        than computed wrongly.
        """
        if not self._is_multi(holder):
            return self._engine.setup_in_model(
                holder, params_phys, data_index, N_vals=N_vals)
        xp = holder.xp
        views = self._shard_views(holder)
        parts = self._partition(holder, data_index)
        params_host = asnumpy(params_phys)
        N_host = None if N_vals is None else asnumpy(N_vals)
        built_on = set()

        def _shard(view, engine, pos, intra):
            with device_context(xp, view.device):
                ret = engine.setup_in_model(
                    view, xp.asarray(params_host[pos]), intra,
                    N_vals=None if N_host is None else xp.asarray(N_host[pos]))
            if not ret:
                return
            # Collision only exists when two shards share ONE engine, and
            # duplicate state_ids force _dispatch_shards serial -- so this
            # check never races (threaded items all run distinct engines).
            if id(engine) in built_on:
                self.clear_in_model()
                raise NotImplementedError(
                    "sig-het in-model references are per-comp state, but two "
                    "shards resolved to the SAME likelihood engine: the "
                    "second shard's build would patch the first's references "
                    "(intra-shard slot ids collide by construction). Build "
                    "the router with engine_factory= so every shard device "
                    "gets its own comp replica, or run the sig-het in-model "
                    "path on a single GPU."
                )
            built_on.add(id(engine))

        items = [
            (view, self._engine_for(holder, view), pos, intra)
            for view, (pos, intra, _) in zip(views, parts)
            if pos.shape[0]
        ]
        self._dispatch_shards(holder, items, _shard,
                              state_ids=[id(it[1]) for it in items])
        return None

    def clear_in_model(self):
        """Clear the in-model reference on EVERY per-device engine.

        Missed fan-out is a silent bug, not an error: a replica that keeps
        ``_in_model`` set makes the next block's first ``setup_in_model`` on
        that device take the mid-block patch branch against a stale slot map.
        """
        for engine in self._engine_by_device.values():
            engine.clear_in_model()
        return self._engine.clear_in_model()

    def _route_matrix(self, method_name, holder, params_phys, *, data_index,
                      noise_index, N_vals, **kwargs):
        """Shared row-wise routing for matrix-valued outputs (grad/hessian)."""
        if not self._is_multi(holder):
            return getattr(self._engine, method_name)(
                holder, params_phys, data_index=data_index,
                noise_index=noise_index, N_vals=N_vals, **kwargs)
        xp = holder.xp
        views = self._shard_views(holder)
        parts = self._partition(holder, data_index, noise_index)
        num = int(params_phys.shape[0])
        params_host = asnumpy(params_phys)
        N_host = None if N_vals is None else asnumpy(N_vals)
        items = [
            (si, view, self._engine_for(holder, view), pos, intra,
             intra_noise)
            for si, (view, (pos, intra, intra_noise))
            in enumerate(zip(views, parts))
            if pos.shape[0]
        ]
        pieces = [None] * len(views)

        def _shard(si, view, engine, pos, intra, intra_noise):
            method = getattr(engine, method_name)
            with device_context(xp, view.device):
                out_s = method(
                    view, xp.asarray(params_host[pos]),
                    data_index=intra,
                    noise_index=intra if intra_noise is None else intra_noise,
                    N_vals=None if N_host is None else xp.asarray(N_host[pos]),
                    **kwargs)
                pieces[si] = (pos, asnumpy(out_s))

        self._dispatch_shards(holder, items, _shard,
                              state_ids=[id(it[2]) for it in items])
        return self._assemble(num, [p for p in pieces if p is not None],
                              0.0, xp)

    def get_ll_grad(self, holder, params_phys, *, data_index, noise_index,
                    N_vals, **kwargs):
        return self._route_matrix(
            "get_ll_grad", holder, params_phys, data_index=data_index,
            noise_index=noise_index, N_vals=N_vals, **kwargs)

    def hessian(self, holder, params_phys, *, data_index, noise_index,
                N_vals, **kwargs):
        return self._route_matrix(
            "hessian", holder, params_phys, data_index=data_index,
            noise_index=noise_index, N_vals=N_vals, **kwargs)

    @classmethod
    def route_information_matrix(cls, comp, holder, params_phys, *, inds,
                                noise_index, data_index=None, **swap_kwargs):
        """Route ``comp.information_matrix`` per shard.

        The Fisher/information matrix is computed on the RAW GB comp
        (``gb_wdm_comp`` / ``gb_fd_comp``) for the proposal Cholesky, not on
        the wrapped likelihood engine -- so it can't go through the instance
        router and needs its own entry point. Each binary's matrix depends
        only on its walker's PSD (``noise_index``; the data slab is
        irrelevant, per ``information_matrix``), so partition binaries by the
        owning shard of their walker, compute per shard against a persistent
        :class:`_ShardHolderView` inside the owning device context, and
        reassemble the ``(num_bin, nd, nd)`` stack on the caller's device.
        Single-shard holders pass straight through.

        Each shard runs against its own device-local comp replica
        (:meth:`_comp_for`), so the kernel never dereferences another
        device's chunk geometry / window / wrap pointers.
        """
        # ``data_index`` is only meaningful to the sig-het wrapper, where it
        # is the per-source BUFFER SLOT feeding the in-model reference
        # lookup; raw comps have no such parameter. Forward it ONLY when the
        # caller supplied one, and slice it with the rows -- passing it
        # full-length while ``params_host[pos]`` is a subset would misalign
        # every shard but the first.
        _di = {} if data_index is None else {"data_index": data_index}
        if not cls._is_multi(holder):
            return comp.information_matrix(
                params_phys, holder, inds=inds,
                noise_index=noise_index, **_di, **swap_kwargs)
        xp = holder.xp
        views = cls._shard_views(holder)
        # Partition by the owning shard of each source's WALKER: the matrix
        # weights by that walker's PSD. (Historically data_index was assumed
        # irrelevant here and aliased to noise_index -- true for the chunked
        # Fisher, false once the sig-het route consumes a slot index.)
        parts = cls._partition(holder, noise_index, noise_index)
        params_host = np.atleast_2d(asnumpy(params_phys))
        di_host = None if data_index is None else np.atleast_1d(asnumpy(data_index))
        num = int(params_host.shape[0])
        items = [
            (si, view, cls._comp_for(comp, holder, view), pos, intra,
             intra_noise)
            for si, (view, (pos, intra, intra_noise))
            in enumerate(zip(views, parts))
            if pos.shape[0]
        ]
        pieces = [None] * len(views)

        def _shard(si, view, comp_s, pos, intra, intra_noise):
            with device_context(xp, view.device):
                _di_s = ({} if di_host is None
                         else {"data_index": xp.asarray(di_host[pos])})
                out_s = comp_s.information_matrix(
                    xp.asarray(params_host[pos]), view, inds=inds,
                    noise_index=intra if intra_noise is None else intra_noise,
                    **_di_s, **swap_kwargs)
                pieces[si] = (pos, asnumpy(out_s))

        cls._dispatch_shards(holder, items, _shard,
                             state_ids=[id(it[2]) for it in items])
        return cls._assemble(num, [p for p in pieces if p is not None],
                             0.0, xp)

    @classmethod
    def route_fstat_ll(cls, comp, method_name, holder, params_phys, *,
                       data_index, noise_index=None, **kwargs):
        """Route a raw F-stat comp entry per shard.

        ``comp`` is the raw single-shard comp (``GBWDMComputations`` /
        ``GBFDComputations``) and ``method_name`` its F-stat entry
        (``"get_fstat_ll_wdm"`` / ``"get_fstat_ll_fd"``): batched over
        binaries, consuming ``holder.linear_data_arr[0]`` and returning
        ``(N (num_bin, 4), M_upper (num_bin, 10))``. Like
        :meth:`route_information_matrix` it runs on the RAW comp (the
        F-stat basis filters are not an engine op), so it gets its own
        classmethod entry rather than the instance router. Single-shard
        holders pass straight through (no overhead). Multi-shard holders
        are partitioned by the owning shard of each binary's walker
        (``data_index``), computed per shard against a persistent
        :class:`_ShardHolderView` inside the owning device context, and
        ``(N, M)`` are reassembled full-length on the caller's device
        (host-routed, no P2P).

        The comp is taken as an OBJECT (not a bound method) so each shard can
        be dispatched to its own device-local replica
        (:meth:`_comp_for`) -- the in-fit F-stat scores every candidate
        against ONE reference walker, so in practice every row lands on that
        walker's shard, which is exactly the case that reads foreign
        device pointers when the comp is shared.
        """
        if not cls._is_multi(holder):
            return getattr(comp, method_name)(
                params_phys, holder, data_index=data_index,
                noise_index=noise_index, **kwargs)
        if data_index is None:
            raise ValueError(
                "route_fstat_ll requires an explicit data_index on "
                "multi-shard holders (the all-zeros default is only "
                "meaningful for a single-shard buffer).")
        if getattr(holder, "slab_min_f", None) is not None:
            # Scope assertion, kept deliberately. ``_ShardHolderView`` now
            # slices ``slab_min_f`` to its own rows, so a slab holder would
            # in fact be handled correctly here -- but the in-fit F-stat runs
            # on the parent residual ACA by design (it scans one walker's
            # FULL residual, not a per-band slab), and a slab holder reaching
            # this entry point means a caller took a path nobody has
            # validated. Fail loudly rather than silently scan narrow slabs.
            raise NotImplementedError(
                "route_fstat_ll does not support narrow per-band slab "
                "holders; F-stat runs on the parent residual ACA, which "
                "carries no slab metadata.")
        xp = holder.xp
        views = cls._shard_views(holder)
        parts = cls._partition(holder, data_index, noise_index)
        params_host = np.atleast_2d(asnumpy(params_phys))
        num = int(params_host.shape[0])
        items = [
            (si, view, cls._comp_for(comp, holder, view), pos, intra,
             intra_noise)
            for si, (view, (pos, intra, intra_noise))
            in enumerate(zip(views, parts))
            if pos.shape[0]
        ]
        N_pieces = [None] * len(views)
        M_pieces = [None] * len(views)

        def _shard(si, view, comp_s, pos, intra, intra_noise):
            comp_method = getattr(comp_s, method_name)
            with device_context(xp, view.device):
                N_s, M_s = comp_method(
                    xp.asarray(params_host[pos]), view,
                    data_index=intra,
                    noise_index=intra if intra_noise is None else intra_noise,
                    **kwargs)
                N_pieces[si] = (pos, asnumpy(N_s))
                M_pieces[si] = (pos, asnumpy(M_s))

        cls._dispatch_shards(holder, items, _shard,
                             state_ids=[id(it[2]) for it in items])
        return (
            cls._assemble(num, [p for p in N_pieces if p is not None],
                          0.0, xp),
            cls._assemble(num, [p for p in M_pieces if p is not None],
                          0.0, xp),
        )

    @classmethod
    def route_sighet_fstat(cls, comp, holder, *, xp, Tobs, f0_lims_hz,
                           data_index, noise_index=None, **build_kwargs):
        """Shard-route the sig-het shared-reference F-stat scorer.

        ``comp`` is the sig-het wrapper (``GBSignalHetComputations``). Its
        F-stat surface is stateful -- ``setup_fstat_references`` folds the
        reference walker's residual into a stash ON the comp that every
        later ``get_fstat_ll_wdm`` scores through -- so unlike
        :meth:`route_fstat_ll` (one routed call per batch) the whole SCORER
        pins to one shard: the F-stat is single-shard by contract (every
        candidate scores against the ONE reference walker ``data_index``),
        making the partition trivial -- that walker's shard owns every
        call. Setup and score both run against the same module-cached
        device-local comp replica (:meth:`_comp_for`), so the stash the
        lazy reference-block builds write is the stash the score calls
        read, on buffers the shard's kernels can legally dereference.

        Returns the ``call_fstat`` closure from
        :func:`lisatools.sampling.fstat_gridfit.build_sighet_call_fstat`
        built against the reference walker's :class:`_ShardHolderView` with
        its INTRA-shard row index; candidates in / ``(N, M)`` out are
        host-routed across the shard's ``device_context`` (no P2P), so the
        adapter and the sweeps never see the sharding. Single-shard holders
        pass straight through (no overhead, no wrapper).

        With TWO OR MORE run devices (``holder.gpus``),
        ``FSTAT_SIGHET_MULTIDEV=1`` fans the scorer out over ALL of them
        (:meth:`_sighet_fstat_multidevice`): the F-stat is single-shard by
        contract, so the pinned form leaves every other GPU idle for the
        whole comb + stage-B fit. OPT-IN (default 0) until the on-GPU
        parity gate passes; ``FSTAT_SIGHET_MULTIDEV=check`` runs the
        fan-out WITH a pinned single-device shadow scorer and hard-compares
        every batch (the on-cluster bisector for the observed GPU
        divergence). CPU and single-GPU holders never take the fan-out
        path at all.
        """
        from ...sampling.fstat_gridfit import build_sighet_call_fstat

        if not cls._is_multi(holder):
            return build_sighet_call_fstat(
                comp, holder, xp=xp, Tobs=Tobs, f0_lims_hz=f0_lims_hz,
                data_index=data_index, noise_index=noise_index,
                **build_kwargs)
        if data_index is None:
            raise ValueError(
                "route_sighet_fstat requires an explicit data_index on "
                "multi-shard holders (the reference walker's row; the "
                "all-zeros default is only meaningful for a single-shard "
                "buffer).")
        if getattr(holder, "slab_min_f", None) is not None:
            # Same scope assertion as route_fstat_ll: the F-stat scans one
            # walker's FULL residual on the parent ACA, never a per-band
            # slab -- a slab holder here means an unvalidated caller path.
            raise NotImplementedError(
                "route_sighet_fstat does not support narrow per-band slab "
                "holders; F-stat runs on the parent residual ACA, which "
                "carries no slab metadata.")
        views = cls._shard_views(holder)
        parts = cls._partition(
            holder, np.atleast_1d(int(data_index)),
            None if noise_index is None else np.atleast_1d(int(noise_index)))
        view, (_pos, intra, intra_noise) = next(
            (v, p) for v, p in zip(views, parts) if p[0].shape[0])
        gpus = getattr(holder, "gpus", None)
        mode = os.environ.get("FSTAT_SIGHET_MULTIDEV", "0")
        if gpus is not None and len(gpus) >= 2 and mode in ("1", "check"):
            # DEFAULT OFF (2026-08-12): the first on-GPU run of the
            # fan-out produced grids that DIFFER from the single-device
            # path (F_max rel up to 0.81, best-sky sign flips) while the
            # CPU fake/real-comp bit-identity tests pass. The 2026-08-11
            # code audit EXCLUDED the merge bookkeeping (disjoint host row
            # ranges), the transfer ordering (every lane D2H is a blocking
            # .get() behind the wraps' own cudaDeviceSynchronize) and the
            # holder row layouts (both ACA buffers carry exactly len(split)
            # rows; the slices mirror setup_fstat_references' own
            # reshapes); the kernel wraps hold the GIL, so lanes cannot
            # race in C++ either. What CPU cannot reach is the on-GPU
            # scoring of the non-primary lane's comp replica -- mode
            # "check" shadows every batch with the pinned scorer and fails
            # loudly on the first diverging row, localizing it to a lane.
            # Opt-in until that gate passes; the single-device pin is the
            # validated scorer.
            return cls._sighet_fstat_multidevice(
                comp, holder, view, int(intra[0]),
                int(intra[0] if intra_noise is None else intra_noise[0]),
                xp=xp, Tobs=Tobs, f0_lims_hz=f0_lims_hz,
                check=(mode == "check"), **build_kwargs)
        comp_s = cls._comp_for(comp, holder, view)
        inner = build_sighet_call_fstat(
            comp_s, view, xp=xp, Tobs=Tobs, f0_lims_hz=f0_lims_hz,
            data_index=int(intra[0]),
            noise_index=(None if intra_noise is None
                         else int(intra_noise[0])),
            **build_kwargs)

        def call_fstat(params):
            # Candidates in / (N, M) out are host-routed; the scorer --
            # including the lazy reference-block builds it triggers -- runs
            # inside the owning shard's device context.
            params_host = np.atleast_2d(asnumpy(params))
            with device_context(holder.xp, view.device):
                N_s, M_s = inner(xp.asarray(params_host))
                N_host, M_host = asnumpy(N_s), asnumpy(M_s)
            return xp.asarray(N_host), xp.asarray(M_host)

        return call_fstat

    @classmethod
    def _sighet_fstat_multidevice(cls, comp, holder, view, intra_data,
                                  intra_noise, *, xp, Tobs, f0_lims_hz,
                                  check=False, **build_kwargs):
        """All-device fan-out for the sig-het F-stat scorer.

        REPLICATE, don't partition, the resident state: every run device
        gets its own comp replica (module-cached ``_comp_for``) and its own
        :class:`_FStatRefRowHolder` -- the reference walker's residual +
        inverse-PSD rows copied onto that device ONCE here (host-routed, no
        P2P) -- and builds its OWN copy of whichever ~GB reference block its
        rows need (``build_sighet_call_fstat``'s lazy ``_ensure_block``,
        whose per-lane state dict is independent). Reference builds are
        deterministic, so concurrent per-device builds are free
        parallelization, and copying the rows once at adapter build means
        every lane's blocks fold the SAME residual snapshot.

        Each ``call_fstat`` batch is row-split into near-equal CONTIGUOUS
        chunks (the sweeps order rows f0-locally -- node-major comb,
        f0-sorted stage-B boxes -- so contiguous chunks stay
        reference-block coherent per lane). Lanes run concurrently on host
        threads (one per device with rows; cupy's current device is
        thread-local; the fstat scorer + reference-build wraps release the
        GIL via ``nb::call_guard`` in binding_gbgpu -- a GBGPU wheel built
        before that guard still works but serializes the lanes' kernel
        calls), each entering its own device context and returning host
        ``(N, M)``.

        DETERMINISM: results are IDENTICAL to the single-device path --
        every row is scored by exactly ONE lane whose replica performs the
        same arithmetic on the same folded inputs as the pinned scorer
        (replica bit-identity is pinned by
        ``SigHetFStatRouteRealCompTest``), the per-row |df0| hard-assert
        runs unchanged inside every lane, and the merge below is a pure
        permutation into disjoint host row ranges -- no reductions ever
        cross devices.

        ``check=True`` (``FSTAT_SIGHET_MULTIDEV=check``): every batch is
        ALSO scored through the pinned single-device scorer (the exact
        else-branch construction) and hard-compared bit-for-bit. On the
        first divergence it logs per-lane forensics -- which lane's row
        range diverges, on which device, through the prototype comp or a
        replica -- and raises. This is the on-cluster bisector for the
        observed GPU divergence: a mismatch confined to the non-primary
        lane's range convicts that lane's comp replica; a mismatch pattern
        crossing lane boundaries convicts the merge/transfer machinery.
        The pinned shadow may SHARE a comp with the walker-shard lane;
        ``build_sighet_call_fstat``'s stash-identity guard makes the two
        closures rebuild instead of scoring a foreign block.
        """
        from ...sampling.fstat_gridfit import build_sighet_call_fstat

        n_slabs = int(view.acs_total_entries)
        # Slice the walker's rows ON the owning device, host only the rows
        # (asnumpy of the whole shard buffer would round-trip every walker).
        with device_context(holder.xp, view.device):
            data_row_host = np.ascontiguousarray(asnumpy(
                xp.asarray(view.linear_data_arr[0]).reshape(
                    n_slabs, -1)[int(intra_data)]))
            psd_row_host = np.ascontiguousarray(asnumpy(
                xp.asarray(view.linear_psd_arr[0]).reshape(
                    n_slabs, -1)[int(intra_noise)]))

        lanes = []
        lane_comps = []
        for dev in [int(g) for g in holder.gpus]:
            with device_context(holder.xp, dev):
                ref_holder = _FStatRefRowHolder(
                    holder, dev,
                    xp.asarray(data_row_host), xp.asarray(psd_row_host))
            comp_d = cls._comp_for(comp, holder, ref_holder)
            lane_comps.append(comp_d)
            lanes.append((dev, build_sighet_call_fstat(
                comp_d, ref_holder, xp=xp, Tobs=Tobs,
                f0_lims_hz=f0_lims_hz, data_index=0, noise_index=0,
                **build_kwargs)))
        logger.info(
            "[sighet-fstat] multi-device scorer: %d lanes on devices %s%s "
            "(FSTAT_SIGHET_MULTIDEV=0 restores the single-device pin)",
            len(lanes), [dev for dev, _ in lanes],
            " + pinned shadow CHECK" if check else "")

        if check:
            # The exact pinned construction (the route's else branch): the
            # walker-shard replica scoring against the live shard view.
            comp_pin = cls._comp_for(comp, holder, view)
            inner_pin = build_sighet_call_fstat(
                comp_pin, view, xp=xp, Tobs=Tobs, f0_lims_hz=f0_lims_hz,
                data_index=int(intra_data), noise_index=int(intra_noise),
                **build_kwargs)
        else:
            comp_pin = inner_pin = None

        def _check_batch(params_host, N_host, M_host):
            """Shadow-score the batch on the pinned scorer; fail loudly on
            the first diverging row with lane-resolved forensics."""
            with device_context(holder.xp, view.device):
                N_ref, M_ref = inner_pin(xp.asarray(params_host))
                N_ref, M_ref = asnumpy(N_ref), asnumpy(M_ref)
            if np.array_equal(N_host, N_ref) and np.array_equal(M_host, M_ref):
                return
            n = int(params_host.shape[0])
            bounds = (n * np.arange(len(lanes) + 1)) // len(lanes)
            bad = (np.any(N_host != N_ref, axis=1)
                   | np.any(M_host != M_ref, axis=1))
            lines = []
            for i, (dev, _inner) in enumerate(lanes):
                s, e = int(bounds[i]), int(bounds[i + 1])
                nb = int(bad[s:e].sum())
                dN = np.abs(N_host[s:e] - N_ref[s:e])
                dM = np.abs(M_host[s:e] - M_ref[s:e])
                if lane_comps[i] is comp_pin:
                    kind = "PINNED-SHARED"
                elif lane_comps[i] is comp:
                    kind = "prototype"
                else:
                    kind = "replica"
                lines.append(
                    f"lane {i} dev {dev} rows [{s}:{e}) comp={kind} "
                    f"bad {nb}/{e - s} maxdN {dN.max() if dN.size else 0:.3e} "
                    f"maxdM {dM.max() if dM.size else 0:.3e}")
            first = int(np.argmax(bad))
            msg = ("[sighet-fstat] MULTIDEV CHECK FAILED: fan-out != pinned "
                   f"scorer on {int(bad.sum())}/{n} rows (first bad row "
                   f"{first}, f0={params_host[first, 1]:.9e} Hz).\n  "
                   + "\n  ".join(lines))
            logger.error(msg)
            raise RuntimeError(msg)

        def call_fstat(params):
            params_host = np.atleast_2d(asnumpy(params))
            n = int(params_host.shape[0])
            bounds = (n * np.arange(len(lanes) + 1)) // len(lanes)
            N_host = np.zeros((n, 4), dtype=np.float64)
            M_host = np.zeros((n, 10), dtype=np.float64)

            def _score(i):
                s, e = int(bounds[i]), int(bounds[i + 1])
                dev, inner = lanes[i]
                with device_context(holder.xp, dev):
                    N_s, M_s = inner(xp.asarray(params_host[s:e]))
                    # permutation merge into disjoint row ranges (see the
                    # determinism note in the method docstring)
                    N_host[s:e] = asnumpy(N_s)
                    M_host[s:e] = asnumpy(M_s)

            active = [i for i in range(len(lanes))
                      if bounds[i + 1] > bounds[i]]
            # thread_pool only touched with >1 populated lane (the real
            # ACA property allocates the pool on first access)
            pool = (getattr(holder, "thread_pool", None)
                    if len(active) > 1 else None)
            if pool is not None:
                futures = [pool.submit(_score, i) for i in active]
                for f in futures:
                    f.result()  # re-raise lane exceptions in caller
            else:
                for i in active:
                    _score(i)
            if inner_pin is not None:
                _check_batch(params_host, N_host, M_host)
            return xp.asarray(N_host), xp.asarray(M_host)

        return call_fstat


def make_routed_band_engine(basis_settings, *, xp, gb_wdm_comp=None,
                            gb_fd_comp=None, **engine_kwargs):
    """Build the shard-routed band likelihood engine for a holder.

    The prototype engine is exactly the
    :func:`gbgpu.gb_likelihood.make_band_likelihood_engine` product the two
    construction sites (:class:`SubBandBuffer` and the move-level parent-ACA
    engine in ``gbspecialstretch``) built before, so single-GPU behaviour is
    unchanged. The added ``engine_factory`` closure rebuilds the SAME engine
    around per-device comp replicas
    (``source_runtime._device_local_gb_comp``) the first time a shard lands
    on a device other than the comps' own -- giving that shard device-local
    chunk geometry / window / orbit + TDI wraps, and, under
    ``GB_SIGHET_INMODEL``, its own heterodyne reference stash.

    Replicas are module-cached and allocate-once; they deliberately do NOT
    follow the holder's proposal lifetime (rebuilding a ``GBTDIonTheFly`` per
    proposal would be ruinous) and never reach the settings tree.
    """
    from gbgpu.gb_likelihood import make_band_likelihood_engine

    def _engine_factory(device, primary):
        from ..stock.erebor.source_runtime import _device_local_gb_comp

        with device_context(xp, device):
            return make_band_likelihood_engine(
                basis_settings,
                gb_wdm_comp=_device_local_gb_comp(
                    gb_wdm_comp, xp, device, primary),
                gb_fd_comp=_device_local_gb_comp(
                    gb_fd_comp, xp, device, primary),
                **engine_kwargs,
            )

    return _RoutedBandEngine(
        make_band_likelihood_engine(
            basis_settings, gb_wdm_comp=gb_wdm_comp, gb_fd_comp=gb_fd_comp,
            **engine_kwargs),
        engine_factory=_engine_factory,
    )


class SubBandBuffer(AnalysisContainerArray, LISAToolsParallelModule):
    """Per-(temp, walker, band) scratch buffers for the GB special moves.

    One :class:`AnalysisContainer` per active cell, all owned directly by
    this object (it *is* the :class:`AnalysisContainerArray`): the linear
    data buffer holds each cell's residual window, the linear PSD buffer the
    matching inverse-PSD slice. GB sources are written into / removed from
    the residual through the domain-aware likelihood engine
    (:meth:`add_sources_to_band_buffer` / :meth:`remove_sources_from_band_buffer`),
    so the inner MCMC loop avoids reallocating large arrays each iteration.

    The most-used members are:

    - ``special_indices_unique`` / ``special_indices_unique_sort``: lookup
      tables that map a per-source ``special_index`` back into the buffer
      ordering.
    - ``params_interest``: parameters of GBs that participate in the move.

    Sign convention: each cell buffer holds the **residual**
    ``data - sum(templates)``. Removing a source from the model therefore
    means *adding* its template back into the buffer (``factor=+1``);
    adding a source to the model subtracts it (``factor=-1``).
    """

    @property
    def xp(self) -> Union[ModuleType, numpy, cupy]:
        """Active array module (NumPy or CuPy) for this buffer.

        Overrides the ``gpus``-based :class:`AnalysisContainerArray`
        property: the buffer's backend decides the array module, so a
        CUDA-backend run on the current device (``gpus=None``) still
        allocates on the GPU.
        """
        return self.backend.xp

    @property
    def df(self):
        """Frequency spacing used for band-index math.

        FD: the FD bin width. WDM: ``layer_df`` so ``band_edges / df``
        yields WDM layer indices. Overrides the read-only
        :class:`AnalysisContainerArray` property (which derives ``df``
        from ``f_arr``).
        """
        return self._df

    @classmethod
    def supported_backends(cls):
        """List the GPU backend names this buffer supports."""
        return ["lisatools_" + _tmp for _tmp in cls.GPU_RECOMMENDED()]

    def get_index(self, special_inds_test):
        """Map a special-index test value to its position inside the buffer."""
        # Array module from the OPERANDS, never the module-level ``cp``: on
        # a CPU-backend run on a machine where cupy imports (cluster), the
        # module-level ``cp`` is cupy while these are numpy --
        # cupy.searchsorted then raises "Only int or ndarray are supported
        # for a". Same trap as get_buffer's sorter xp (see :2860).
        xp = get_array_module(self.special_indices_unique)
        now_index = (
            self.special_indices_unique_sort[
                xp.searchsorted(
                    self.special_indices_unique[self.special_indices_unique_sort],
                    xp.asarray(special_inds_test),
                    side="right",
                )
                - 1
            ]
        ).astype(xp.int32)
        return now_index

    def __init__(
        self,
        is_rj,
        nwalkers,
        gb,
        band_edges,
        band_N_vals,
        unique_band_combos,
        params_interest,
        num_bands_now,
        nchannels,
        data_length,
        special_indices_unique,
        transform_fn,
        waveform_kwargs,
        df,
        sources_now_map,
        sources_inject_now_map,
        special_band_inds,
        opt_snr_rej_samp_limit=5.0,
        snr_rej_detected=False,
        force_backend="gpu",
        use_template_arr=False,
        basis_settings: Optional[DomainSettingsBase] = None,
        gb_wdm_comp=None,
        gb_fd_comp=None,
        keep_sens_mat: bool = False,
        wdm_band_slab_layers: Optional[int] = None,
        wdm_slab_guard_layers: int = 1,
        *args,
        **kwargs,
    ):
        # (The gb_likelihood import is deferred inside
        # ``make_routed_band_engine`` -- gb_likelihood imports nothing from
        # here, but the module import graph stays acyclic if that changes.)
        self.force_backend = force_backend
        LISAToolsParallelModule.__init__(self, force_backend=force_backend)
        assert self.backend.name.split("_")[-1] == gb.backend.name.split("_")[-1]
        self.gb = gb
        # Domain computation objects (gbgpu.gbcomps.GBWDMComputations /
        # GBFDComputations prototype). The one matching ``basis_settings``
        # is required; the other stays None. The legacy ``gb`` handle is
        # kept only for the info-matrix proposal shaping.
        self.gb_wdm_comp = gb_wdm_comp
        self.gb_fd_comp = gb_fd_comp
        self._df = df
        self.nwalkers = nwalkers
        self.sources_now_map, self.sources_inject_now_map = (
            sources_now_map,
            sources_inject_now_map,
        )
        self.band_edges, self.unique_band_combos = band_edges, unique_band_combos
        self.num_bands = len(self.band_edges) - 1
        self.params_interest = params_interest
        self.num_bands_now, self.nchannels, self.data_length = (
            num_bands_now,
            nchannels,
            data_length,
        )
        # FD store length of one cell window; kept distinct from the ACA's
        # ``data_length`` layout attribute (see :attr:`_fd_store_length`).
        self._fd_store_length_value = data_length
        self.band_N_vals = self.xp.asarray(band_N_vals)
        # TODO: adjust this
        self.edge_buffer = 2000
        self.is_rj = is_rj

        # Frequency-clipped parent domains (FDSettings with min/max_freq)
        # store only bins [ind_min, ind_max]. Every per-cell FD window (and
        # its start index) must live inside that range: clamp the window
        # length here -- BEFORE the ``special_indices_unique`` setter below
        # computes start indices and before the buffers are allocated -- and
        # record the parent bounds for the start-index clamp in the setter.
        # WDM and legacy full-grid FD paths are untouched.
        self._parent_ind_min = None
        self._parent_stored_len = None
        if (
            isinstance(basis_settings, FDSettings)
            and getattr(basis_settings, "ind_min", None) is not None
            and getattr(basis_settings, "ind_max", None) is not None
        ):
            self._parent_ind_min = int(basis_settings.ind_min)
            self._parent_stored_len = (
                int(basis_settings.ind_max) - int(basis_settings.ind_min) + 1
            )
            if self._fd_store_length_value > self._parent_stored_len:
                logger.warning(
                    "FD cell window (%d bins) exceeds the clipped parent domain "
                    "(%d stored bins); clamping each cell window to the full "
                    "stored band.",
                    self._fd_store_length_value,
                    self._parent_stored_len,
                )
                self._fd_store_length_value = self._parent_stored_len

        self.special_indices_unique = special_indices_unique
        self.transform_fn = transform_fn
        # Container-derived roles: phi0's sampled column (phase-max rotation)
        # and whether the container holds per-leaf fills (Eryn per-leaf
        # fill_dict) -- in that case every sampling->physical conversion in
        # this buffer needs the per-row ``leaf_inds``.
        _ib = list(getattr(transform_fn, "input_basis", []) or [])
        self._phi0_col = _ib.index("phi0") if "phi0" in _ib else 3
        self._per_leaf_fill = getattr(transform_fn, "n_leaf_fills", None) is not None
        self.waveform_kwargs = waveform_kwargs
        self.opt_snr_rej_samp_limit = opt_snr_rej_samp_limit
        self.snr_rej_detected = bool(snr_rej_detected)
        self.use_template_arr = use_template_arr
        # Per-band sensitivity storage. Only the inverse covariance ``invC``
        # feeds the WDM / FD likelihood kernels (the ACA repacks
        # ``sens_mat.invC`` into ``linear_psd_arr``; nothing on the band-move
        # path reads the forward ``sens_mat`` array). Default ``False``
        # therefore stores ONLY ``invC`` per band and backs the forward
        # ``sens_mat`` with a zero-storage, shape-correct broadcast view --
        # halving the per-band cross-channel (XYZ 3x3) memory. Set
        # ``keep_sens_mat=True`` to allocate the forward matrix too (e.g. for
        # diagnostics that inspect the per-band covariance directly).
        self.keep_sens_mat = keep_sens_mat
        # Task-b: narrow per-band WDM slabs. Each per-band buffer slab spans a
        # limited number of WDM frequency layers centered on the band --
        # instead of the FULL active band ``Nf_active`` -- cutting the dominant
        # sub-band-buffer memory term by ~Nf_active/slab_Nf. Requires the C++
        # chunked-het kernels built with the task-b per-slab layer origin
        # (default backend); the per-slot origins flow via ``slab_min_f``.
        #   ``wdm_band_slab_layers`` semantics:
        #     None -> OFF (full active band; bit-identical to pre-task-b).
        #     0    -> AUTO: band layer span + 2*(leakage + guard), where
        #             leakage = _WDM_SLAB_LEAKAGE_LAYERS (2) and guard =
        #             ``wdm_slab_guard_layers`` (default 1) -> band_span + 6.
        #     N>0  -> EXPLICIT: exactly N layers (power users).
        # FD path ignores this (already per-band narrow via max_data_store_size).
        self._wdm_band_slab_layers = (
            int(wdm_band_slab_layers) if wdm_band_slab_layers is not None else None
        )
        self._wdm_slab_guard_layers = int(wdm_slab_guard_layers)

        self.tdi_channel_setup = self.waveform_kwargs.get("tdi_channel_setup")
        if self.tdi_channel_setup == "XYZ":
            assert self.nchannels == 3
        else:
            assert "A" in self.tdi_channel_setup and "E" in self.tdi_channel_setup
            logger.warning("using AE(T) channels where we assume ortogonality. This may not be sufficient for realistic orbtis.")

        # Resolve the parent basis-domain settings. Defaults to an FD grid
        # consistent with the legacy Buffer behavior (data_length bins on the
        # parent's df). When invoked via BandSorter.get_buffer, the parent
        # AnalysisContainerArray's settings are forwarded so this buffer can
        # branch on the actual domain (FD vs WDM).
        if basis_settings is None:
            basis_settings = FDSettings(
                N=self.data_length,
                df=float(self.df) if not hasattr(self.df, "item") else self.df.item(),
            )
        self._basis_settings = basis_settings

        # Build the per-cell AnalysisContainers and initialise *ourselves* as
        # the AnalysisContainerArray that owns them. On the WDM path the ACA
        # layout metadata (``data_length = Nf_active * Nt_active``,
        # ``end_shape``) replaces the FD-style ctor value -- the inherited
        # linear-buffer indexing needs the ACA meaning; nothing on the WDM
        # move path consumes the FD-style value.
        ac_list, aca_kwargs = self._build_band_ac_list()
        AnalysisContainerArray.__init__(self, ac_list, **aca_kwargs)
        if self.use_template_arr:
            # Templates mirror the band-buffer layout in a twin ACA so they
            # share the same managed memory region. The per-band sensitivity
            # slot on the template ACA is unused but keeps construction
            # symmetric across the two buffers.
            template_ac_list, template_aca_kwargs = self._build_band_ac_list()
            self._acs_template_buffer = AnalysisContainerArray(
                template_ac_list, **template_aca_kwargs
            )

        # psd_shape is exposed for back-compat with downstream consumers that
        # inspect it; it tracks the shape of the per-band PSD view.
        self.psd_shape = (self.num_bands_now,) + self._per_band_sens_shape

        # Geometry line: the buffer is the dominant per-proposal allocation
        # (cells x per-cell slab); one INFO read diagnoses an OOM-scale
        # configuration (e.g. many bands with slab slicing off).
        _cell_mb = (
            np.prod(self._per_band_data_shape)
            * np.dtype(self._per_band_data_dtype).itemsize / 1e6
        )
        _n_copies = 2 if self.use_template_arr else 1
        _pool = ""
        if self.backend.uses_cupy:
            _mp = cp.get_default_memory_pool()
            _pool = (f"  [GPU pool used {_mp.used_bytes() / 1e9:.2f} / "
                     f"total {_mp.total_bytes() / 1e9:.2f} GB]")
        # Host watermark alongside: SIGKILL-class failures are host-side --
        # this line is the last breadcrumb before a cgroup OOM kill.
        import resource as _resource
        import sys as _sys

        _rss_kb = _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss
        _rss_gb = _rss_kb / (1e9 if _sys.platform == "darwin" else 1e6)
        _pool += f"  [host maxRSS {_rss_gb:.1f} GB]"
        # HONEST total: the data slab alone under-reports by ~4x for XYZ
        # (invC is (nc, nc, slab) per slot -- the term that actually sized
        # the job-183 OOM).
        _invc_mb = (self.nchannels
                    * float(np.prod(self._per_band_data_shape))
                    * np.dtype(self._per_band_data_dtype).itemsize / 1e6)
        logger.info(
            "SubBandBuffer: %d cells x %s per-cell (%s) ~ %.0f MB data "
            "+ ~%.0f MB invC = ~%.1f GB total%s [band_slab_Nf=%s]%s",
            self.num_bands_now, tuple(self._per_band_data_shape),
            np.dtype(self._per_band_data_dtype).name,
            _n_copies * self.num_bands_now * _cell_mb,
            self.num_bands_now * _invc_mb,
            (_n_copies * self.num_bands_now * _cell_mb
             + self.num_bands_now * _invc_mb) / 1e3,
            " (incl. template twin)" if self.use_template_arr else "",
            self.band_slab_Nf, _pool,
        )

        # Build the domain-aware likelihood engine. Dispatch is on
        # ``isinstance(basis_settings, ...)`` -- no string-level mode flag.
        # The engine takes an AnalysisContainerArray at call time, so the
        # buffer's get_swap_ll / get_ll / adjust_sources_in_band_buffer
        # methods don't reach into self.gb (or self.gb_wdm_comp) directly.
        # Persistent per-slot window-start store. Bound BY POINTER into the
        # FD computations clone (FDDomain.start_inds), so it must be updated
        # in place on cell swaps -- never rebound (see the
        # special_indices_unique setter).
        if isinstance(self._basis_settings, FDSettings):
            self._min_freq_inds_store = self.xp.ascontiguousarray(
                self.xp.asarray(self.start_freq_inds), dtype=self.xp.int32
            ).copy()
            if self.use_template_arr:
                # The template twin shares the buffer's per-slot window
                # starts (same array object: in-place updates on cell swaps
                # reach both FD comps clones).
                self._acs_template_buffer.min_freq_inds = self._min_freq_inds_store

        # Routed: multi-shard buffers partition every engine call by owning
        # GPU split (and give each non-prototype device its own comp replica);
        # single-shard buffers pass straight through.
        self._likelihood_engine = make_routed_band_engine(
            self._basis_settings,
            xp=self.xp,
            gb=self.gb,
            gb_fd_comp=self.gb_fd_comp,
            gb_wdm_comp=self.gb_wdm_comp,
            nchannels=self.nchannels,
            tdi_channel_setup=self.tdi_channel_setup,
            df=float(self.df) if not hasattr(self.df, "item") else self.df.item(),
            start_freq_inds=getattr(self, "start_freq_inds", None),
            data_length=self.data_length,
            opt_snr_rej_samp_limit=self.opt_snr_rej_samp_limit,
            snr_rej_detected=self.snr_rej_detected,
        )

        # TODO: fix this 4????
        self.special_band_inds = special_band_inds
        assert special_band_inds.shape[0] == self.params_interest.shape[0]
        self.now_index = self.get_index(special_band_inds)

    # ------------------------------------------------------------------
    # Views into the AnalysisContainerArray-backed scratch buffers
    # ------------------------------------------------------------------

    @property
    def acs_buffer(self) -> AnalysisContainerArray:
        """The :class:`AnalysisContainerArray` backing the per-band residual buffers.

        Post Buffer/ACA merge this *is* the buffer object itself; kept for
        back-compat with callers that used ``buffer.acs_buffer``.
        """
        return self

    # ------------------------------------------------------------------
    # Per-band buffer accessors
    # ------------------------------------------------------------------
    # In multi-GPU mode the buffer holds the bands sharded across GPUs
    # (striped by default; see ``_build_band_ac_list``). The shaped
    # accessors below return a :class:`BandView` that lets callers
    # index by global band number; reads/writes route to the owning
    # shard. In single-GPU mode they return the underlying ndarray
    # view directly (no overhead). The ``*_tmp`` flat accessors stay
    # single-GPU-only -- with multi-shard there is no single flat
    # ndarray, so callers should use the engine path (gb_likelihood
    # passes the list-of-shards through ``buffer_aca.linear_data_arr``
    # / ``linear_psd_arr`` directly).

    def _shaped_or_view(self, acs, kind: str):
        """Return either the single-shard reshape (single-GPU) or a BandView (multi-GPU)."""
        if len(acs.linear_data_arr) == 1:
            return acs.data_shaped[0] if kind == "data" else acs.psd_shaped[0]
        return acs.data_shaped_view() if kind == "data" else acs.psd_shaped_view()

    def _flat_or_raise(self, acs, kind: str):
        if len(acs.linear_data_arr) == 1:
            return (
                acs.linear_data_arr[0] if kind == "data" else acs.linear_psd_arr[0]
            )
        raise RuntimeError(
            f"{kind}_buffer_tmp is only valid in single-GPU mode "
            "(multi-GPU buffers are a list of per-GPU shards). Use the "
            "engine path or BandView accessors instead."
        )

    @property
    def band_buffer_tmp(self):
        """Flat per-GPU residual buffer (1D view; single-GPU only)."""
        return self._flat_or_raise(self, "data")

    @property
    def band_buffer(self):
        """Per-band residual buffer indexable by global band id.

        Single-GPU: returns the ``(num_bands_now, nchannels, data_length)``
        reshape directly. Multi-GPU: returns a :class:`BandView` that
        routes per-band reads/writes through the owning shard.
        """
        return self._shaped_or_view(self, "data")

    @property
    def psd_buffer_tmp(self):
        """Flat per-GPU inverse-PSD buffer (1D view; single-GPU only)."""
        return self._flat_or_raise(self, "psd")

    @property
    def psd_buffer(self):
        """Per-band inverse-PSD buffer indexable by global band id.

        Same single-GPU / multi-GPU behaviour as :attr:`band_buffer`.
        """
        return self._shaped_or_view(self, "psd")

    @property
    def template_buffer_tmp(self):
        """Flat per-GPU template buffer (single-GPU only; ``use_template_arr`` True)."""
        return self._flat_or_raise(self._acs_template_buffer, "data")

    @property
    def template_buffer(self):
        """Per-band template buffer indexable by global band id.

        Same single-GPU / multi-GPU behaviour as :attr:`band_buffer`.
        """
        return self._shaped_or_view(self._acs_template_buffer, "data")

    # ------------------------------------------------------------------
    # Domain-aware allocation helpers
    # ------------------------------------------------------------------

    @property
    def basis_settings(self) -> DomainSettingsBase:
        """Parent basis-domain settings driving per-band buffer geometry."""
        return self._basis_settings

    @property
    def band_slab_Nf(self) -> Optional[int]:
        """WDM-layer extent of a narrow per-band slab (task-b), or ``None``.

        ``None`` on the FD path, or on WDM when ``wdm_band_slab_layers`` is
        ``None`` (full-active-band layout). ``0`` auto-sizes to
        ``band_span + 2*(leakage + guard)``; ``N>0`` uses exactly ``N``. All
        slabs share this constant extent; each has its own origin in
        :attr:`slab_min_f`. Clamped to the parent active band.
        """
        if self._wdm_band_slab_layers is None:
            return None
        if not isinstance(self._basis_settings, WDMSettings):
            return None
        if self._wdm_band_slab_layers > 0:
            slab_Nf = int(self._wdm_band_slab_layers)
        else:
            # Auto: cover the widest band + leakage + guard on each side.
            slab_Nf = self.recommend_band_slab_layers(
                self.band_edges,
                float(self.df) if not hasattr(self.df, "item") else self.df.item(),
                leakage=_WDM_SLAB_LEAKAGE_LAYERS,
                guard=self._wdm_slab_guard_layers,
                xp=self.xp,
            )
        # Never exceed the parent active band.
        return int(min(slab_Nf, self._basis_settings.Nf_active))

    @staticmethod
    def recommend_band_slab_layers(band_edges, layer_df, leakage=_WDM_SLAB_LEAKAGE_LAYERS,
                                   guard=1, xp=np) -> int:
        """Recommended per-band slab extent (WDM layers) for task-b.

        ``max_band_span + 2*(leakage + guard)``: the widest band's layer span
        plus enough neighbor layers on each side to cover the chunked-het
        template spread (``leakage``, ~2 for the recommended Tukey window)
        and a safety ``guard``. This is what ``wdm_band_slab_layers=0``
        (auto) resolves to; ``check_wdm_band_slab.py`` prints it alongside a
        measured leakage estimate.
        """
        edges = xp.asarray(band_edges)
        lo = (edges[:-1] / layer_df).astype(int)
        hi = (edges[1:] / layer_df).astype(int)
        # Layer count of the widest band. Edges are layer-aligned frequency
        # boundaries, so ``hi - lo`` is the exclusive layer span (1 for a
        # 1-layer band); floor to >= 1 for sub-layer bands.
        max_span = max(1, int(xp.max(hi - lo)))
        return int(max_span + 2 * (int(leakage) + int(guard)))

    @property
    def slab_min_f(self):
        """Per-slot start WDM layer of each narrow band slab (task-b), or ``None``.

        Each slot's slab spans ``[slab_min_f[slot], slab_min_f[slot] +
        band_slab_Nf)`` WDM layers, centered on the slot's band and clamped
        into the parent active band ``[ind_min_f, ind_max_f]``. Recomputed
        from the live per-slot band assignment (``unique_band_combos[:, 2]``)
        so it tracks cell swap-outs. Read by the chunked-het kernels (via
        ``fill_global_wdm`` / ``_slab_kernel_args``) as the per-slab layer
        origin. ``None`` when narrow slabs are off.
        """
        slab_Nf = self.band_slab_Nf
        if slab_Nf is None:
            return None
        ldf = float(self.df) if not hasattr(self.df, "item") else self.df.item()
        band_inds = self.unique_band_combos[:, 2]
        # Center the slab on the band (its [lo, hi] layer span), so the
        # m_band_half_width spread on either side of any source in the band
        # stays inside the slab as long as slab_Nf >= band_span + 2*hw.
        lo_layer = (self.band_edges[band_inds] / ldf).astype(self.xp.int32)
        hi_layer = (self.band_edges[band_inds + 1] / ldf).astype(self.xp.int32)
        center = (lo_layer + hi_layer) // 2
        origins = center - slab_Nf // 2
        parent = self._basis_settings
        lo = int(parent.ind_min_f)
        hi = max(lo, int(parent.ind_max_f) + 1 - slab_Nf)
        return self.xp.clip(origins, lo, hi).astype(self.xp.int32)

    @property
    def _per_band_data_shape(self) -> tuple:
        """Shape of a single band's residual buffer (one AC's data_res_arr)."""
        if isinstance(self._basis_settings, FDSettings):
            return (self.nchannels, self._fd_store_length)
        elif isinstance(self._basis_settings, WDMSettings):
            # Task-b: a narrow per-band slab covers ``band_slab_Nf`` layers
            # (centered on the band, origin in ``slab_min_f``) instead of the
            # full active grid; ``band_slab_Nf is None`` keeps the full
            # ``Nf_active`` layout (the WDM kernel then uses the single global
            # [ind_min_f, ind_max_f] origin).
            Nf_use = self._per_band_Nf
            Nt_active = self._basis_settings.Nt_active
            return (self.nchannels, Nf_use, Nt_active)
        else:
            raise NotImplementedError(
                f"Buffer does not support basis domain {type(self._basis_settings).__name__}."
            )

    @property
    def _per_band_Nf(self) -> int:
        """Per-band WDM slab frequency extent: ``band_slab_Nf`` when narrow,
        else the full parent ``Nf_active``."""
        slab_Nf = self.band_slab_Nf
        return int(slab_Nf) if slab_Nf is not None else int(self._basis_settings.Nf_active)

    @property
    def _per_band_sens_shape(self) -> tuple:
        """Shape of a single band's inverse-PSD buffer (one AC's sens_mat.invC)."""
        if isinstance(self._basis_settings, FDSettings):
            if self.tdi_channel_setup == "XYZ":
                return (self.nchannels, self.nchannels, self._fd_store_length)
            return (self.nchannels, self._fd_store_length)
        elif isinstance(self._basis_settings, WDMSettings):
            Nf_use = self._per_band_Nf
            Nt_active = self._basis_settings.Nt_active
            if self.tdi_channel_setup == "XYZ":
                return (self.nchannels, self.nchannels, Nf_use, Nt_active)
            return (self.nchannels, Nf_use, Nt_active)
        else:
            raise NotImplementedError(
                f"Buffer does not support basis domain {type(self._basis_settings).__name__}."
            )

    @property
    def _per_band_data_dtype(self):
        """Element dtype for the per-band residual buffer."""
        if isinstance(self._basis_settings, FDSettings):
            return self.xp.complex128
        elif isinstance(self._basis_settings, WDMSettings):
            return self.xp.float64
        else:
            raise NotImplementedError(
                f"Buffer does not support basis domain {type(self._basis_settings).__name__}."
            )

    @property
    def _per_band_sens_dtype(self):
        """Element dtype for the per-band inverse-PSD buffer.

        FD is REAL: the gb_fd kernels consume the real part of the inverse
        covariance (Hermitian; the imaginary parts cancel in the quadratic
        forms), matching the FDDomain double* layout.
        """
        if isinstance(self._basis_settings, FDSettings):
            return self.xp.float64
        elif isinstance(self._basis_settings, WDMSettings):
            return self.xp.float64
        else:
            raise NotImplementedError(
                f"Buffer does not support basis domain {type(self._basis_settings).__name__}."
            )

    def _build_per_band_basis_settings(self) -> DomainSettingsBase:
        """Construct the per-band domain settings used by each per-band AC.

        Each per-band AC's data domain needs a settings object whose
        ``basis_shape_active`` matches the per-band data shape. For FD this
        is a fresh FDSettings sized to the FD store length. For WDM the
        per-band settings share the parent's active grid.
        """
        if isinstance(self._basis_settings, FDSettings):
            return FDSettings(
                N=self._fd_store_length,
                df=float(self.df) if not hasattr(self.df, "item") else self.df.item(),
                force_backend=self._basis_settings.backend_name.split("_", 1)[1],
            )
        elif isinstance(self._basis_settings, WDMSettings):
            # Per-band WDMSettings: full parent active band by default. Task-b:
            # when narrow slabs are on, the per-band domain only needs the
            # SHAPE (band_slab_Nf, Nt_active) -- the actual per-slot layer
            # origin lives in ``slab_min_f`` (the kernel arg), not here -- so
            # we pin ``ind_max_f`` to make ``Nf_active == band_slab_Nf``.
            parent = self._basis_settings
            # The slab extent must ride IN the constructor args: consumers
            # (WDMDomain.__init__) re-build the settings from args/kwargs, so
            # a post-construction ``ind_max_f`` mutation is silently lost and
            # the per-band domain reverts to the full active band (shape
            # assert). ``ind_max_f = int(max_freq / layer_df)`` (inclusive);
            # the half-layer offset keeps the floor rounding-safe.
            slab_Nf = self.band_slab_Nf
            _ind_max_f = (
                int(parent.ind_min_f) + int(slab_Nf) - 1
                if slab_Nf is not None
                else int(parent.ind_max_f)
            )
            per_band = WDMSettings(
                Nf=parent.Nf,
                Nt=parent.Nt,
                dt=parent.data_dt,
                t0=parent.t0,
                oversample=parent.oversample,
                window=parent.window,
                omega=parent.omega,
                min_freq=parent.ind_min_f * parent.layer_df,
                max_freq=(_ind_max_f + 0.5) * parent.layer_df,
                min_time=parent.ind_min_t * parent.layer_dt,
                max_time=parent.ind_max_t * parent.layer_dt,
            )
            return per_band
        else:
            raise NotImplementedError(
                f"Buffer does not support basis domain {type(self._basis_settings).__name__}."
            )

    def _build_band_ac_list(self) -> tuple:
        """Allocate one :class:`AnalysisContainer` per active cell.

        Returns ``(ac_list, aca_kwargs)`` ready to be fed into
        ``AnalysisContainerArray.__init__`` (either on ``self`` or on the
        template twin). Branches on the parent basis domain: FD buffers are
        complex with ``complex_psd=True`` (XYZ CSD support); WDM buffers are
        real-valued slabs over the full active grid.
        """
        per_band_settings = self._build_per_band_basis_settings()
        data_shape = self._per_band_data_shape
        sens_shape = self._per_band_sens_shape
        data_dtype = self._per_band_data_dtype
        sens_dtype = self._per_band_sens_dtype

        # Forward sensitivity matrix: full per-band allocation only when the
        # caller asks to keep it (``keep_sens_mat=True``). Otherwise back it
        # with a zero-storage broadcast view -- correct ``.shape`` / ``.ndim``
        # (the ACA reads ``sens_mat.shape`` to derive ``shape_sens``; the
        # SensitivityMatrixBase ndarray setter only shape-checks it) but a
        # single element of memory. ``invC`` -- the only array the likelihood
        # kernels consume -- is always a real per-band buffer.
        def _forward_sens():
            if self.keep_sens_mat:
                return cp.zeros(sens_shape, dtype=sens_dtype)
            return cp.broadcast_to(cp.zeros((), dtype=sens_dtype), sens_shape)

        ac_list = []
        for _ in range(self.num_bands_now):
            res_data = cp.zeros(data_shape, dtype=data_dtype)
            data_domain = per_band_settings.associated_class(res_data, per_band_settings)
            sm = SensitivityMatrixBase(per_band_settings, skip_inv_det=True)
            sm.sens_mat = _forward_sens()
            sm.invC = cp.zeros(sens_shape, dtype=sens_dtype)
            sm.channel_shape = sens_shape[: -len(per_band_settings.basis_shape_active)]
            ac_list.append(AnalysisContainer(data_domain, sm))

        gpus_in = getattr(self.gb, "gpus", None) if self.backend.uses_cupy else None
        # Multi-GPU at the GB band-tree level (parallel-resources plan P1):
        # group the per-cell slabs by their BAND id so every cell of a band
        # -- in particular a tempering swap pair (same band, adjacent temps)
        # -- is device-local, while bands round-robin across GPUs in
        # first-appearance order (the GB move activates bands in even/odd
        # parity rotation, so each parity pass still spreads over all
        # devices). Plain slot striping put a row's temps in consecutive
        # slots on ALTERNATING shards, making every tempering swap pair pay
        # a cross-device hop. The per-band accessors (band_buffer /
        # psd_buffer / template_buffer) automatically fall back to a single
        # ndarray view for single-GPU runs and return a BandView
        # (multi-shard router) otherwise -- see the accessor block above.
        gpu_assignment = (
            band_gpu_assignment(
                len(ac_list),
                list(gpus_in),
                group_ids=asnumpy(self.unique_band_combos[:, 2]),
            )
            if gpus_in
            else None
        )
        aca_kwargs = dict(
            gpus=list(gpus_in) if gpus_in else None,
            complex_psd=False,
            gpu_assignment=gpu_assignment,
        )
        return ac_list, aca_kwargs

    def update_special_indices(self, new_special_indices, inds_fill=None):
        if inds_fill is None:
            inds_fill = cp.arange(self.num_bands_now)

        assert inds_fill.shape[0] == new_special_indices.shape[0]
        _tmp_indices = self.special_indices_unique.copy()
        _tmp_indices[inds_fill] = new_special_indices
        self.special_indices_unique = _tmp_indices

    @property
    def special_indices_unique(self):
        return self._special_indices_unique

    @special_indices_unique.setter
    def special_indices_unique(self, special_indices_unique):
        self._special_indices_unique_sort = self.xp.argsort(special_indices_unique)
        self._special_indices_unique = special_indices_unique

        _temp_inds, _walker_inds, _band_inds = self.get_separate_inds_from_special_index(
            special_indices_unique
        )

        self.unique_band_combos = self.xp.array([_temp_inds, _walker_inds, _band_inds]).T

        if self.num_bands == 1:
            tmp_buffer_start_index = (self.band_edges[0] / self.df).astype(
                np.int32
            ) - self.edge_buffer
            if getattr(self, "_parent_ind_min", None) is None:
                # Legacy full-grid parent: the single window must cover the
                # whole band plus both edge buffers. (With a frequency-
                # clipped parent this is un-satisfiable by construction --
                # the clamp below shifts/shrinks the window instead.)
                assert tmp_buffer_start_index + self._fd_store_length >= (
                    (self.band_edges[-1] / self.df).astype(np.int32) + self.edge_buffer
                )
            self.buffer_start_index = self.xp.repeat(
                tmp_buffer_start_index, self.unique_band_combos.shape[0]
            )

        else:
            self.buffer_start_index = (
                self.band_edges[self.unique_band_combos[:, 2] - 1] / self.df
            ).astype(np.int32)
            self.buffer_start_index[self.unique_band_combos[:, 2] == 0] = (
                self.band_edges[0] / self.df
            ).astype(np.int32) - self.edge_buffer
            # Clamp so buffer end never overflows the data range (band_edges[-1])
            max_start = int(self.band_edges[-1] / self.df) - self._fd_store_length
            self.buffer_start_index = np.minimum(self.buffer_start_index, max_start)

        if getattr(self, "_parent_ind_min", None) is not None:
            # Frequency-clipped parent: clamp every window into the stored
            # bin range [ind_min, ind_min + stored_len - window]. Edge cells
            # lose their out-of-domain guard margin (the parent stores
            # nothing there); the engines read placement from
            # ``start_freq_inds`` so a shifted window stays consistent.
            lo = self._parent_ind_min
            hi = max(lo, lo + self._parent_stored_len - self._fd_store_length)
            self.buffer_start_index = self.xp.clip(self.buffer_start_index, lo, hi)

        self.start_freq_inds = self.xp.asarray(self.buffer_start_index.copy().astype(np.int32))
        if hasattr(self, "_min_freq_inds_store"):
            # in-place: the FD comps clone holds a pointer to this array
            self._min_freq_inds_store[:] = self.start_freq_inds

        lower_f_lim = self.band_edges[
            self.unique_band_combos[:, 2]
        ]  #  - self.band_N_vals[self.unique_band_combos[:, 2]] * self.df / 4
        higher_f_lim = self.band_edges[
            self.unique_band_combos[:, 2] + 1
        ]  #  + self.band_N_vals[self.unique_band_combos[:, 2]] * self.df / 4

        # allow to move over band edge when proposing in-model
        if self.is_rj:
            lower_f_lim -= self.band_N_vals[self.unique_band_combos[:, 2]] * self.df / 4
            higher_f_lim += self.band_N_vals[self.unique_band_combos[:, 2]] * self.df / 4
        self.frequency_lims = [lower_f_lim, higher_f_lim]

    @property
    def _fd_store_length(self) -> int:
        """FD frequency-bin count of one cell's residual window.

        This is the FD-specific store size handed in at construction. It is
        deliberately kept separate from the inherited ACA ``data_length``
        (which on the WDM path becomes ``Nf_active * Nt_active`` -- the
        linear-buffer stride the inherited packing methods need).
        """
        return self._fd_store_length_value

    @property
    def min_freq_inds(self):
        """Per-slot minimum-frequency index, unified across domains.

        FD: absolute FD bin where each cell's residual window starts
        (``buffer_start_index``). WDM: the start layer of each cell's slab --
        currently the parent grid's ``ind_min_f`` for every slot, until the
        WDM kernel takes per-band layer offsets. The likelihood engines read
        this off the buffer at every call, so it never goes stale across
        band swap-outs.
        """
        if isinstance(self._basis_settings, WDMSettings):
            return self.xp.full(
                self.num_bands_now, self._basis_settings.ind_min_f, dtype=self.xp.int32
            )
        return self._min_freq_inds_store

    @property
    def special_indices_unique_sort(self):
        return self._special_indices_unique_sort

    @staticmethod
    def _materialize(buf):
        """Return a single ndarray view for ``buf``.

        Single-shard runs already expose ``buf`` as a reshape view of the
        underlying ndarray, so this is a no-op (returns ``buf`` itself).
        Multi-shard runs expose ``buf`` as a :class:`BandView` -- gather it
        to a single ndarray on ``gpus[0]`` so downstream einsum / boolean
        indexing / sum kernels see a contiguous array.
        """
        if isinstance(buf, BandView):
            return buf.gather()
        return buf

    def likelihood(self, source_only: bool = False, noise_only: bool = False) -> float:
        """Band-level log-likelihood over all cells in the buffer.

        Overrides the inherited per-AC ``AnalysisContainerArray.likelihood``
        dispatch: the buffer computes its cell likelihoods directly from the
        shaped residual / PSD views (vectorized over cells).
        """
        assert not (source_only and noise_only)

        # band_buffer / template_buffer / psd_buffer are either ndarrays
        # (single-GPU; in-place mutation rolls back into the underlying
        # buffer) or BandView (multi-GPU; mutating after materialisation
        # PER-SHARD, CHUNKED reduction (2026-08-14). The previous pipeline
        # gathered the FULL band-sharded buffer to gpus[0], copied it for
        # the template subtraction, and let einsum transpose-copy it again:
        # transients scaling with TOTAL slots (~50 GB at 32,768 slots, all
        # on one device) that OOM'd job 180 the moment concurrent lanes
        # made them coexist. Each shard now reduces on its OWNING device in
        # GB_BAND_LL_CHUNK-slot chunks; only the per-band scalars move.
        # Domain-generic inner product: <a|b> = 4 sum(a* invC b) * dc (dc =
        # basis measure; FD df, WDM pixel measure), trailing axes flattened
        # -- identical arithmetic per band, chunking only batches it.
        if noise_only and self.tdi_channel_setup == "XYZ":
            raise NotImplementedError(
                "Noise-only likelihood requires log-determinant over "
                "frequency for XYZ CSD.")

        nc = self.nchannels
        dc = float(self.settings.differential_component)
        chunk = max(1, int(os.environ.get("GB_BAND_LL_CHUNK", "1024")))
        xyz = self.tdi_channel_setup == "XYZ"

        def _reduce(num, psd):
            k = num.shape[0]
            nf = num.reshape(k, nc, -1)
            if xyz:
                pf = psd.reshape(k, nc, nc, -1)
                return (-(1.0 / 2.0) * 4.0 * dc * cp.einsum(
                    "bik,bijk,bjk->b", nf.conj(), pf, nf).real)
            pf = psd.reshape(k, nc, -1)
            return (-(1.0 / 2.0) * 4.0 * dc
                    * cp.sum((nf.conj() * nf) * pf, axis=(1, 2)).real)

        band = self.band_buffer
        psd_b = self.psd_buffer
        tmpl = self.template_buffer if self.use_template_arr else None

        if isinstance(band, BandView):
            aca = band._aca
            nb_tot = int(band.shape[0])
            out_host = np.empty(nb_tot, dtype=np.float64)
            psd_log_acc = 0.0
            uses_dev = getattr(aca, "gpus", None) is not None
            main_dev = cp.cuda.runtime.getDevice() if uses_dev else None
            try:
                for s in range(len(band._shards)):
                    ids = np.asarray(asnumpy(aca.gpu_splits[s]), dtype=int)
                    if uses_dev:
                        cp.cuda.runtime.setDevice(int(aca.gpus[s]))
                    d_sh = band._shards[s]
                    p_sh = psd_b._shards[s]
                    t_sh = tmpl._shards[s] if tmpl is not None else None
                    for c0 in range(0, d_sh.shape[0], chunk):
                        c1 = min(c0 + chunk, d_sh.shape[0])
                        num = (d_sh[c0:c1] - t_sh[c0:c1]
                               if t_sh is not None else d_sh[c0:c1])
                        if not noise_only:
                            out_host[ids[c0:c1]] = asnumpy(
                                _reduce(num, p_sh[c0:c1]))
                        if noise_only or not source_only:
                            pc = p_sh[c0:c1]
                            nz = pc[pc != 0.0]
                            psd_log_acc += float(asnumpy(-cp.sum(cp.log(
                                cp.abs(1 / nz if noise_only else nz)))))
            finally:
                if uses_dev:
                    cp.cuda.runtime.setDevice(main_dev)
            if noise_only:
                return psd_log_acc
            source_term = cp.asarray(out_host)
        else:
            nb_tot = int(band.shape[0])
            source_term = cp.empty(nb_tot, dtype=cp.float64)
            psd_log_acc = 0.0
            for c0 in range(0, nb_tot, chunk):
                c1 = min(c0 + chunk, nb_tot)
                num = (band[c0:c1] - tmpl[c0:c1]
                       if tmpl is not None else band[c0:c1])
                if not noise_only:
                    source_term[c0:c1] = _reduce(num, psd_b[c0:c1])
                if noise_only or not source_only:
                    pc = psd_b[c0:c1]
                    nz = pc[pc != 0.0]
                    psd_log_acc += float(asnumpy(-cp.sum(cp.log(
                        cp.abs(1 / nz if noise_only else nz)))))
            if noise_only:
                return psd_log_acc

        if source_only:
            return source_term

        # Diagonal noise_term fall_back # TODO check if this is sufficient not used currently anyway
        if self.tdi_channel_setup == "XYZ":
            warnings.warn("The current psd ll calculation is not correct for XYZ CSD channel setup.")

        return source_term + psd_log_acc

    # Explicit alias while callers migrate off the ``likelihood`` name (which
    # shadows the inherited per-AC ACA dispatch).
    band_likelihoods = likelihood

    def _to_phys(self, params, leaf_inds=None):
        """Sampling -> physical rows through the transform container.

        ``leaf_inds`` (per-row leaf indices) is required by containers built
        with a per-leaf ``fill_dict`` list (Eryn validates); scalar-fill
        containers ignore it.
        """
        return self.transform_fn.both_transforms(params, xp=cp, leaf_inds=leaf_inds)

    def get_swap_ll(self, params_remove, params_add, data_index, N_vals, phase_maximize=False,
                    leaf_inds=None):
        """Per-proposal swap log-likelihood difference.

        Domain-agnostic: dispatches to ``self._likelihood_engine.get_swap_ll``,
        which is either :class:`FDBandLikelihoodEngine` or
        :class:`WDMBandLikelihoodEngine` depending on the buffer's
        ``basis_settings``. Both engines take the per-band ACA (``self``)
        and the physical params, and return a :class:`SwapLLResult`. The
        rejection-sampling clamp and the phase-maximisation correction live
        here so the engine stays a thin wrapper around the kernel.
        """
        params_remove_phys = self._to_phys(params_remove, leaf_inds=leaf_inds)
        params_add_phys = self._to_phys(params_add, leaf_inds=leaf_inds)

        result = self._likelihood_engine.get_swap_ll(
            self,
            params_remove_phys,
            params_add_phys,
            data_index=data_index,
            noise_index=data_index,
            N_vals=N_vals,
            phase_maximize=phase_maximize,
            waveform_kwargs=self.waveform_kwargs,
        )

        ll_diff = result.ll_diff
        kept = result.kept

        if np.any(~kept):
            logger.info(f"NOT KEEPING: {(~kept).sum()}")

        if phase_maximize and result.phase_angle is not None:
            # Engine returns the per-proposal phase rotation applied during
            # phase-maximisation; subtract it from phi0 so the accepted
            # parameters reflect the maximised draw.
            params_add[kept, self._phi0_col] = (
                params_add[kept, self._phi0_col] - result.phase_angle
            )

        # Rejection sampling on SNR: only applied to *add* proposals (the
        # remove side's opt_snr is meaningless when amp_add is tiny).
        reject = self.xp.zeros(kept.shape[0], dtype=bool)
        _bad_swap = result.opt_snr_add[kept] < self.opt_snr_rej_samp_limit
        if getattr(self, "snr_rej_detected", False) and (
                getattr(result, "d_h_add", None) is not None):
            _det_add = self.xp.asarray(result.d_h_add)[kept].real / self.xp.maximum(
                result.opt_snr_add[kept], 1e-300)
            _bad_swap = _bad_swap | (_det_add < self.opt_snr_rej_samp_limit)
        reject[kept] = _bad_swap & (params_add_phys[kept, 0] > 1e-30)
        ll_diff[reject] = -1e300

        return ll_diff

    def get_ll(self, params, data_index, noise_index, N_vals, phase_maximize=False,
               return_inner_products=False, leaf_inds=None):
        """Per-source log-likelihood against the cell residuals.

        Domain-agnostic dispatch like :meth:`get_swap_ll`. Returns the
        log-likelihood array ``-0.5 * (d_d + h_h - 2 d_h)`` on the engine's
        xp module (``d_d`` per the underlying computation object's
        convention; 0 unless configured). With
        ``return_inner_products=True`` returns ``(ll, d_h, h_h,
        phase_angle)`` instead. The raw inner products also land on
        :attr:`d_h_out` / :attr:`h_h_out`, and :attr:`phase_angle` carries
        the maximising rotation when ``phase_maximize=True``.
        """
        params_phys = self._to_phys(params, leaf_inds=leaf_inds)
        ll = self._likelihood_engine.get_ll(
            self,
            params_phys,
            data_index=data_index,
            noise_index=noise_index,
            N_vals=N_vals,
            phase_maximize=phase_maximize,
            waveform_kwargs=self.waveform_kwargs,
        )
        self.d_h_out = self._likelihood_engine.d_h_out
        self.h_h_out = self._likelihood_engine.h_h_out
        self.phase_angle = self._likelihood_engine.phase_angle
        self.kept_out = getattr(
            self._likelihood_engine, "kept_out",
            self.xp.ones(params.shape[0], dtype=bool),
        )
        if return_inner_products:
            return ll, self.d_h_out, self.h_h_out, self.phase_angle
        return ll

    def setup_in_model_likelihood(self, params, data_index, N_vals=None, leaf_inds=None) -> None:
        """Per-source in-model likelihood setup (once per repeat block).

        Forwards the picked sources' CURRENT sampling-basis params
        (transformed to physical) plus their buffer slots to the engine's
        ``setup_in_model`` hook. Chunked-het / FD engines no-op; a sig-het
        computation builds its heterodyne reference against the
        source-free cell residuals here and holds it constant until
        :meth:`clear_in_model_likelihood`. Call AFTER the sources are
        removed from the residual, BEFORE the reference ll of the repeat
        block is computed.

        Returns the engine hook's value: truthy when a sig-het reference
        is now active (the move uses this to arm its mid-block drift
        refresh), ``None`` from the no-op hooks."""
        params_phys = self._to_phys(params, leaf_inds=leaf_inds)
        return self._likelihood_engine.setup_in_model(
            self, params_phys, data_index, N_vals=N_vals)

    def clear_in_model_likelihood(self) -> None:
        """Deactivate the per-source in-model setup (no-op engines ignore)."""
        self._likelihood_engine.clear_in_model()

    def get_add_ll(self, params, data_index, noise_index, N_vals, phase_maximize=False,
                   leaf_inds=None):
        """Log-likelihood delta of ADDING a source to the model.

        ``ll(r - h) - ll(r) = <r|h> - 0.5 <h|h>`` where ``r`` is the current
        cell residual (which does not contain ``h``). This is a delta, not a
        singular log-likelihood -- the ``d_d`` term cancels. Computed from
        the :attr:`d_h_out` / :attr:`h_h_out` stashed by :meth:`get_ll`;
        :attr:`phase_angle` is available after the call when
        ``phase_maximize=True``. Sources rejected by the engine's bounds
        check (:attr:`kept_out`) come back as ``-1e300``.
        """
        self.get_ll(params, data_index, noise_index, N_vals, phase_maximize=phase_maximize,
                    leaf_inds=leaf_inds)
        delta = self.d_h_out.real - 0.5 * self.h_h_out.real
        delta[~self.kept_out] = -1e300
        return delta

    def get_removal_ll(self, params, data_index, noise_index, N_vals, leaf_inds=None):
        """Log-likelihood delta of REMOVING a source that is in the residual.

        For a residual ``r`` that still *contains* the subtracted template
        ``h`` (i.e. the source is part of the model), the delta of taking it
        out is ``ll(r + h) - ll(r) = -<r|h> - 0.5 <h|h>``. Computed from one
        normal :meth:`get_ll` call by flipping the sign of ``d_h`` -- the
        arithmetic equivalent of evaluating the template with its reference
        phase flipped (``phi -> -phi``, i.e. ``h -> -h``). Delta only; the
        ``d_d`` term cancels. Bounds-rejected sources come back as
        ``-1e300`` (see :attr:`kept_out`).
        """
        self.get_ll(params, data_index, noise_index, N_vals, leaf_inds=leaf_inds)
        delta = -self.d_h_out.real - 0.5 * self.h_h_out.real
        delta[~self.kept_out] = -1e300
        return delta

    def get_replace_ll(self, params_old, params_new, data_index, noise_index,
                       N_vals, phase_maximize=False, leaf_inds=None):
        """Log-likelihood deltas for REPLACING an in-model source (search RJ).

        Self-contained expose -> score -> restore (pure function on the
        residual buffer):

        1. **Expose** the old source: its template is added back into the
           cell residual through :meth:`remove_sources_from_band_buffer`
           (the fixed addremove convention -- removing a source from the
           model ADDS its template to the residual, dev 1928032), so the
           scored residual is ``r' = r + h_old``: the residual as if old
           had never been fit.
        2. **Score** BOTH parameter sets against ``r'`` as add-deltas
           ``<r'|h> - 0.5<h|h>`` in ONE batched :meth:`get_ll` call
           (rows ``[old; new]``), optionally phase-maximized (the engine's
           two-quadrature maximisation; search convention).
        3. **Restore** the touched slots' residual rows from a pre-expose
           snapshot -- bit-exact by construction (a refill with the
           opposite factor would instead round ``(r + h) - h``).

        Row ``i`` of ``params_old`` / ``params_new`` must target the same
        buffer slot ``data_index[i]`` (the old source's cell). Engine
        bounds-rejected rows come back ``-1e300`` (see the ``replace_kept_*``
        attributes).

        Returns:
            ``(delta_old, delta_new, phase_angle_new, delta_old_actual)``:
            add-deltas of the old and new rows vs ``r'``;
            ``phase_angle_new`` is the per-row maximizing rotation of the
            NEW half (``None`` unless ``phase_maximize``);
            ``delta_old_actual`` is the old row's add-delta at its ACTUAL
            phase (equals ``delta_old`` without phase maximisation) -- the
            exact bookkeeping value for an accepted swap's residual-ll
            change, since a rejected/accepted old source is never
            re-phased. Also stashed: :attr:`replace_h_h_old` /
            :attr:`replace_h_h_new` (for SNR clamps) and
            :attr:`replace_kept_old` / :attr:`replace_kept_new`.
        """
        xp = self.xp
        n = params_old.shape[0]

        # (1) snapshot the touched residual rows, then expose the old source.
        slot_rows = xp.unique(xp.asarray(data_index).astype(xp.int64))
        snapshot = self.band_buffer[slot_rows].copy()
        self.remove_sources_from_band_buffer(
            params_old, data_index, N_vals, leaf_inds=leaf_inds
        )
        try:
            # (2) one batched scoring call over [old; new].
            params_cat = xp.concatenate([params_old, params_new], axis=0)
            di_cat = xp.concatenate([data_index, data_index], axis=0)
            ni_cat = xp.concatenate([noise_index, noise_index], axis=0)
            nv_cat = xp.concatenate([N_vals, N_vals], axis=0)
            li_cat = (
                None if leaf_inds is None
                else xp.concatenate([leaf_inds, leaf_inds], axis=0)
            )
            self.get_ll(
                params_cat, di_cat, ni_cat, nv_cat,
                phase_maximize=phase_maximize, leaf_inds=li_cat,
            )
            d_h = self.d_h_out.real.copy()
            h_h = self.h_h_out.real.copy()
            kept = self.kept_out.copy()
            delta = d_h - 0.5 * h_h
            delta[~kept] = -1e300
            phase_angle_new = None
            if phase_maximize and self.phase_angle is not None:
                phase_angle_new = self.phase_angle[n:].copy()
            # Old rows at their ACTUAL phase: the two-quadrature engines
            # stash the un-maximized <r'|h> as ``non_marg_d_h``. Guarded --
            # on a multi-shard route it is not assembled, so fall back to
            # the maximized value (the propose-level ll-drift rebuild then
            # corrects the tracked sum).
            delta_old_actual = delta[:n].copy()
            if phase_maximize:
                _nm = getattr(self._likelihood_engine, "non_marg_d_h", None)
                if _nm is not None and getattr(_nm, "shape", (0,))[0] == 2 * n:
                    delta_old_actual = xp.asarray(_nm).real[:n] - 0.5 * h_h[:n]
                    delta_old_actual[~kept[:n]] = -1e300
        finally:
            # (3) bit-exact restore of the pre-expose residual.
            self.band_buffer[slot_rows] = snapshot

        self.replace_h_h_old = h_h[:n]
        self.replace_h_h_new = h_h[n:]
        self.replace_kept_old = kept[:n]
        self.replace_kept_new = kept[n:]
        return delta[:n], delta[n:], phase_angle_new, delta_old_actual

    def get_ll_grad(self, params, data_index, noise_index, N_vals,
                     *, param_eps=None, chunk=None, leaf_inds=None):
        """Per-source gradient of ``L = <d|h> - 0.5 <h|h>`` w.r.t. params.

        Dispatches to ``self._likelihood_engine.get_ll_grad`` -- only
        the chunked-het backend implements this; the legacy FD path
        raises NotImplementedError. Returns ``(num_proposals, nparams)``
        on the engine's xp module.

        Used by the in-model NUTS / gradient move (the chunked-het
        replacement for the legacy info-matrix Cholesky proposal). The
        buffer must hold the source-of-interest's *clean* residual --
        i.e. ``remove_sources_from_band_buffer`` has been called for
        that source already -- before invoking this.

        The compute backend (C++ central-FD or JAX autograd) is fixed
        on the ``GBWDMComputations`` instance passed in at buffer
        construction time via ``gb_wdm_comp``. Per the sprint-wide
        rule there is no runtime ``backend=`` kwarg; build a JAX-
        backed ``gb_wdm_comp`` if you need the autograd path.
        """
        params_phys = self._to_phys(params, leaf_inds=leaf_inds)
        return self._likelihood_engine.get_ll_grad(
            self,
            params_phys,
            data_index=data_index,
            noise_index=noise_index,
            N_vals=N_vals,
            param_eps=param_eps,
            chunk=chunk,
            waveform_kwargs=self.waveform_kwargs,
        )

    def hessian(self, params, data_index, noise_index, N_vals,
                 *, chunk=None,
                 psd_fix=False, psd_floor_rel=1e-30, leaf_inds=None):
        """Per-source Hessian of ``L = <d|h> - 0.5 <h|h>``.

        Dispatches to ``self._likelihood_engine.hessian``. Returns
        ``(num_proposals, nparams, nparams)``. With ``psd_fix=True``,
        returns ``M = |-H|`` (eigendecompose-then-abs, with a relative
        floor) -- ready to feed to ``NUTSSampler(metric=M)`` as a
        per-leaf mass matrix.

        Same buffer-state precondition as :meth:`get_ll_grad`: the
        active source must have been removed from the band buffer
        before calling.

        Currently only the JAX-backed chunked-het generator
        implements ``hessian_wdm``; the C++ chunked-het backend
        raises until the native Hessian kernel lands. Per the
        sprint-wide rule the backend is fixed on the underlying
        ``gb_wdm_comp`` instance -- no runtime ``backend=`` kwarg.
        """
        params_phys = self._to_phys(params, leaf_inds=leaf_inds)
        return self._likelihood_engine.hessian(
            self,
            params_phys,
            data_index=data_index,
            noise_index=noise_index,
            N_vals=N_vals,
            chunk=chunk,
            psd_fix=psd_fix,
            psd_floor_rel=psd_floor_rel,
            waveform_kwargs=self.waveform_kwargs,
        )

    def reset_residual_buffers(self, inds_fill=None):
        if inds_fill is None:
            inds_fill = cp.arange(self.num_bands_now)
        self.band_buffer[inds_fill] = 0.0

    def reset_psd_buffers(self, inds_fill=None):
        if inds_fill is None:
            inds_fill = cp.arange(self.num_bands_now)
        self.psd_buffer[inds_fill] = 0.0

    def reset_template_buffers(self, inds_fill=None):
        """Zero the template twin for the given slots.

        The twin is filled ADDITIVELY (engine ``fill_template`` with
        ``factor=+1``), so only a fresh allocation starts it at zero. A
        cached-buffer REBIND (propose-scoped cache 2bd76df / cross-proposal
        persist c94d53c) that skips this reset stacks the new cells'
        templates on top of the previous unit's post-swap contents --
        the root cause of the after-tempering incremental-ll drift
        (sign-consistent ~ -<h|h>/2 per affected cold walker) AND of
        biased tempering swap acceptance (paccept scored against the
        contaminated lls). ``self.xp``, not module-level ``cp`` (the
        CPU-run-with-cupy-importable trap).
        """
        if inds_fill is None:
            inds_fill = self.xp.arange(self.num_bands_now)
        self.template_buffer[inds_fill] = 0.0

    def fill_buffer_residual_and_psd_from_acs(
        self, acs: AnalysisContainerArray, inds_fill: Optional[cp.ndarray] = None
    ) -> None:
        # CHUNKED (2026-08-14): each tuple-fancy gather below materialises a
        # rows x slab temporary -- for the XYZ invC that is ~763 KB/slot, so
        # a 16,384-row gather is 12.5 GB in ONE allocation (job-183 OOM on a
        # device already holding the persistent buffers). Bound the
        # transient at GB_FILL_CHUNK slots per pass.
        if inds_fill is None:
            inds_fill = self.xp.arange(self.num_bands_now)
        _chunk = max(1, int(os.environ.get("GB_FILL_CHUNK", "2048")))
        n_fill = int(inds_fill.shape[0])
        if n_fill > _chunk:
            for c0 in range(0, n_fill, _chunk):
                self.fill_buffer_residual_and_psd_from_acs(
                    acs, inds_fill=inds_fill[c0:min(c0 + _chunk, n_fill)])
            return
        # The outer ``acs`` is accessed via tuple-fancy indexing
        # ``data_shaped[0][inds1, inds2, inds3]`` (3-tuple for AET, 5-tuple
        # for XYZ CSD). BandView routes the tuple-fancy index through the
        # owning shard at the right intra-shard band position; on
        # single-shard ACAs the reshape view is touched directly. No
        # outer-buffer materialisation needed.
        # NOTE self.xp, never the module-level ``cp``: on a CPU-backend run
        # on a machine where cupy imports, cp is cupy while the buffer /
        # sorter arrays are numpy (same trap as get_index / the sorter xp).

        outer_data_view = acs.data_shaped_view()
        outer_psd_view = acs.psd_shaped_view()

        inds_get_data = self._get_fill_buffer_ind_map(acs, inds_fill=inds_fill, is_psd=False)

        # load rest of data into buffer (has current sources removed)
        self.reset_residual_buffers(inds_fill=inds_fill)

        # By removing `.flatten()` during indexing, broadcasting gives us the exact shape natively.
        self.band_buffer[inds_fill] += outer_data_view[inds_get_data]
        del inds_get_data

        inds_get_psd = self._get_fill_buffer_ind_map(acs, inds_fill=inds_fill, is_psd=True)
        self.reset_psd_buffers(inds_fill=inds_fill)

        psd_vals = outer_psd_view[inds_get_psd]
        if self.xp.iscomplexobj(psd_vals) and not self.xp.iscomplexobj(
            self.psd_buffer if not isinstance(self.psd_buffer, BandView) else psd_vals
        ):
            # FD buffers store the REAL inverse covariance (gb_fd kernel
            # convention); the parent XYZ CSD invC may be complex.
            psd_vals = psd_vals.real
        self.psd_buffer[inds_fill] = psd_vals
        del inds_get_psd

    def _get_fill_buffer_ind_map(
        self, acs: AnalysisContainerArray, inds_fill: Optional[cp.ndarray] = None, is_psd: bool = False
    ) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:

        if isinstance(self._basis_settings, WDMSettings):
            # WDM fill index map: pick each band's slab out of the parent ACA.
            # Default (full-active) copies the entire (channel, Nf_active,
            # Nt_active) slab. Task-b narrow slabs copy only each slot's own
            # ``band_slab_Nf``-layer window: the parent-local layer offset is
            # ``slab_min_f[slot] - parent.ind_min_f`` (the parent stores active
            # layer ``ind_min_f`` at local index 0), so the layer axis becomes
            # PER-SLOT rather than a shared 0..Nf_active-1 ramp. Data axis
            # position is unique_band_combos[:, 1] (the parent walker index).
            if inds_fill is None:
                inds_fill = self.xp.arange(self.num_bands_now)
            xp = self.xp  # NOT module-level cp (see fill_buffer note)

            Nt_active = self._basis_settings.Nt_active
            slab_Nf = self.band_slab_Nf
            if slab_Nf is None:
                # Full active band: shared layer ramp 0..Nf_active-1.
                Nf_use = self._basis_settings.Nf_active
                layer_lo = xp.zeros(len(inds_fill), dtype=xp.int32)
            else:
                # Narrow: per-slot parent-local layer offset.
                Nf_use = int(slab_Nf)
                layer_lo = (
                    self.slab_min_f[inds_fill] - int(self._basis_settings.ind_min_f)
                ).astype(xp.int32)

            if is_psd and self.tdi_channel_setup == "XYZ":
                # target shape: (len(inds_fill), nchannels, nchannels, Nf_use, Nt_active)
                # The parent WDM ACA's psd_shaped[0] has shape
                # ``(num_walkers, nchan, nchan, Nf_active, Nt_active)`` — one
                # entry per walker with channels as inner axes. Unlike the FD
                # path (which flattens walker*channel into axis 0), here we
                # index axis 0 with the raw walker index and need a full
                # 5-tuple to cover all five axes.
                inds1 = self.unique_band_combos[inds_fill, 1][:, None, None, None, None]
                inds2 = xp.arange(self.nchannels)[None, :, None, None, None]
                inds3 = xp.arange(self.nchannels)[None, None, :, None, None]
                inds4 = (layer_lo[:, None, None, None, None]
                         + xp.arange(Nf_use)[None, None, None, :, None])
                inds5 = xp.arange(Nt_active)[None, None, None, None, :]
                return inds1, inds2, inds3, inds4, inds5

            # target shape: (len(inds_fill), nchannels, Nf_use, Nt_active)
            inds1 = self.unique_band_combos[inds_fill, 1][:, None, None, None]
            inds2 = xp.arange(self.nchannels)[None, :, None, None]
            inds3 = (layer_lo[:, None, None, None]
                     + xp.arange(Nf_use)[None, None, :, None])
            inds4 = xp.arange(Nt_active)[None, None, None, :]
            return inds1, inds2, inds3, inds4
        if not isinstance(self._basis_settings, FDSettings):
            raise NotImplementedError(
                f"Buffer does not support basis domain {type(self._basis_settings).__name__}."
            )

        xp = self.xp  # NOT module-level cp (see fill_buffer note)
        if inds_fill is None:
            inds_fill = xp.arange(self.num_bands_now)

        assert np.all(acs.start_freq_ind[0] == acs.start_freq_ind)
        start_freq_ind = acs.start_freq_ind[0]

        assert np.all((self.buffer_start_index[inds_fill] - start_freq_ind) >= 0), (
            "Buffer start indices fall below the parent data start index."
        )

        assert np.all(
            (self.buffer_start_index[inds_fill] - start_freq_ind + self._fd_store_length)
            <= acs.data_length
        ), f"Buffer indexing exceeds available data length in AnalysisContainerArray. Start indices: {self.buffer_start_index[inds_fill]}, start_freq_ind: {start_freq_ind}, data_length: {self._fd_store_length}, acs data_length: {acs.data_length}"

        start_inds = self.buffer_start_index[inds_fill] - start_freq_ind

        if is_psd and self.tdi_channel_setup == "XYZ":
            # Target output shape:
            # (len(inds_fill), nchannels, nchannels, window_bins).
            # The parent FD ACA's psd_shaped has shape
            # (num_walkers, nchan, nchan, n_bins) -- axis 0 is the raw
            # walker index (mirrors the WDM XYZ 5-tuple below).
            inds1 = self.unique_band_combos[inds_fill, 1][:, None, None, None]
            inds2 = xp.arange(self.nchannels)[None, :, None, None]
            inds3 = xp.arange(self.nchannels)[None, None, :, None]
            inds4 = start_inds[:, None, None, None] + xp.arange(
                self.band_buffer.shape[-1]
            )[None, None, None, :]
            return inds1, inds2, inds3, inds4

        else:
            # Target output shape: (len(inds_fill), self.nchannels, self.band_buffer.shape[-1])
            inds1 = self.unique_band_combos[inds_fill, 1][:, None, None]
            inds2 = xp.arange(self.nchannels)[None, :, None]
            inds3 = start_inds[:, None, None] + xp.arange(self.band_buffer.shape[-1])[None, None, :]

        return inds1, inds2, inds3

    def remove_sources_from_template_buffer(self, *args, **kwargs) -> None:
        self._adjust_via_engine(-1, self._acs_template_buffer, *args, **kwargs)

    def add_sources_to_template_buffer(self, *args, **kwargs) -> None:
        self._adjust_via_engine(+1, self._acs_template_buffer, *args, **kwargs)

    def swap_template_slots(self, slots_a, slots_b) -> None:
        """Exchange the template-buffer contents of slot sets ``a`` and ``b``.

        Used by the tempering stage to swap a temperature pair's per-cell
        templates -- and, called again with the rejected subset, to revert
        the swaps that failed the acceptance draw.
        """
        buf = self.template_buffer
        if isinstance(buf, BandView):
            # Multi-GPU: same-shard pairs (the common case under the
            # band-grouped shard assignment) swap in place on their device
            # with no host hop (parallel-resources plan P1).
            buf.swap_rows(slots_a, slots_b)
            return
        tmp = buf[slots_a].copy()
        buf[slots_a] = buf[slots_b]
        buf[slots_b] = tmp[:]

    def _adjust_via_engine(
        self, factor, target_aca, params, params_index, N_vals, *args,
        leaf_inds=None, **kwargs
    ) -> None:
        """Domain-agnostic dispatch into ``self._likelihood_engine.fill_template``.

        ``factor`` is +1 (write source into the template) or -1 (subtract it).
        ``target_aca`` selects which AnalysisContainerArray to write into
        (the buffer itself for residuals, or the template twin). Both share
        the same per-band geometry, so the engine doesn't need to know which
        one it's filling.
        """
        assert isinstance(factor, int) and (factor == -1 or factor == +1)
        params_phys = self._to_phys(params, leaf_inds=leaf_inds)
        # Task-b: forward this buffer's narrow per-band slab metadata so the
        # template write matches the narrow slab layout. Passed ONLY when this
        # is a narrow WDM buffer (band_slab_Nf set) so the FD engine's
        # fill_template -- which takes no such kwargs -- is untouched.
        _slab_kwargs = {}
        if getattr(self, "band_slab_Nf", None) is not None:
            _slab_kwargs = dict(
                band_slab_Nf=self.band_slab_Nf, slab_min_f=self.slab_min_f
            )
        self._likelihood_engine.fill_template(
            target_aca,
            params_phys,
            params_index,
            N_vals,
            factor=factor,
            waveform_kwargs=self.waveform_kwargs,
            **_slab_kwargs,
        )

    def adjust_sources_in_band_buffer(
        self, factor, input_array, params, params_index, N_vals, *args, **kwargs
    ) -> None:
        """Backwards-compatible shim around :meth:`_adjust_via_engine`.

        Routes ``input_array`` (a flat buffer pointer the legacy code passed
        through) back to whichever ACA owns it. New code should call
        :meth:`_adjust_via_engine` directly.
        """
        if input_array is self.band_buffer_tmp:
            target_aca = self
        elif self.use_template_arr and input_array is self.template_buffer_tmp:
            target_aca = self._acs_template_buffer
        else:
            raise ValueError(
                "adjust_sources_in_band_buffer received an input_array that "
                "is neither the band-residual nor the template buffer."
            )
        self._adjust_via_engine(factor, target_aca, params, params_index, N_vals, *args, **kwargs)

    def remove_sources_from_band_buffer(self, *args, **kwargs) -> None:
        # NOTE: sign is +1 because band_buffer holds the residual
        # (= data - sum(templates)). Removing a source from the model means
        # ADDING it back to the residual, hence factor=+1 here.
        self._adjust_via_engine(+1, self, *args, **kwargs)

    def add_sources_to_band_buffer(self, *args, **kwargs) -> None:
        # See remove_sources_from_band_buffer note; sign is flipped for the
        # residual-tracking band_buffer.
        self._adjust_via_engine(-1, self, *args, **kwargs)

    def get_special_band_index(
        self, temp_inds: np.ndarray, walker_inds: np.ndarray, band_inds: np.ndarray
    ) -> np.ndarray:
        return pack_special_index(temp_inds, walker_inds, band_inds, self.nwalkers)

    def get_separate_inds_from_special_index(self, special_band_inds: np.ndarray) -> tuple:
        return unpack_special_index(special_band_inds, self.nwalkers)


# Back-compat alias: the pre-merge class name.
Buffer = SubBandBuffer


class BandSorter(LISAToolsParallelModule):
    """GPU helper that sorts/ungroups GB samples by frequency band.

    Flattens the eryn GB branch ``(ntemps, nwalkers, nleaves, ndim)`` into
    per-source arrays (``coords`` / ``inds`` / ``temp_inds`` /
    ``walker_inds`` / ``leaf_inds`` / ``band_inds``), assigns each source to
    its frequency band (``searchsorted`` on the *source* frequency -- not the
    domain settings), pre-draws RJ proposals for the ``inds=False`` slots
    when ``rj_prop`` is given, and provides subset / packing machinery for
    the per-band proposal loop and band-temperature swaps.
    """

    @property
    def xp(self) -> Union[ModuleType, numpy, cupy]:
        return self.backend.xp

    @classmethod
    def supported_backends(cls):
        return ["lisatools_" + _tmp for _tmp in cls.GPU_RECOMMENDED()]

    def __init__(
        self,
        gb_branch: Branch,
        band_edges: Optional[np.ndarray] = None,
        band_N_vals: Optional[np.ndarray] = None,
        force_backend: Optional[str] = None,
        transform_fn: Optional[TransformContainer] = None,
        copy: bool = True,
        inds_subset: Optional[np.ndarray] = None,
        inds_main_band_sorter: Optional[np.ndarray] = None,
        gb=None,
        gb_wdm_comp=None,
        gb_fd_comp=None,
        waveform_kwargs={},
        main_band_sorter=None,
        max_data_store_size: int = 6000,
        rj_prop=None,
        keep_all_inds=True,
        wdm_band_slab_layers: Optional[int] = None,
        wdm_slab_guard_layers: int = 1,
        opt_snr_rej_samp_limit: float = 5.0,
        snr_rej_detected: bool = False,
    ):

        LISAToolsParallelModule.__init__(self, force_backend=force_backend)
        self.force_backend = force_backend
        # RJ SNR rejection-sampling clamp (user policy, default 5.0):
        # forwarded into every SubBandBuffer this sorter builds; the copy
        # constructor path below overwrites it with the source sorter's
        # value (attribute copy loop), keeping one source of truth.
        self.opt_snr_rej_samp_limit = float(opt_snr_rej_samp_limit)
        self.snr_rej_detected = bool(snr_rej_detected)

        dc = deepcopy if copy else return_x
        if hasattr(gb_branch, "num_sources"):
            _band_sorter = gb_branch
            self.force_backend = _band_sorter.force_backend
            for key, value in _band_sorter.__dict__.items():
                if key[:2] != "__":
                    if key in [
                        "main_band_sorter",
                        "inds_main_band_sorter",
                        "gb",
                        "gb_wdm_comp",
                        "gb_fd_comp",
                        "rj_prop",
                    ]:
                        continue

                    elif (
                        isinstance(value, self.xp.ndarray)
                        and value.shape[0] == _band_sorter.num_sources
                    ):
                        if inds_subset is None:
                            inds_subset = self.xp.arange(_band_sorter.num_sources)
                        else:
                            assert (
                                isinstance(inds_subset, self.xp.ndarray)
                                and inds_subset.dtype == int
                            )
                            assert inds_subset.max() < (_band_sorter.num_sources)
                        set_value = dc(value[inds_subset])

                    else:
                        set_value = dc(value)

                    setattr(self, key, set_value)

            self.rj_prop = _band_sorter.rj_prop
            self.gb = _band_sorter.gb
            # Forward the computation objects explicitly (skipped in the
            # copy loop so we don't deepcopy GPU-resident objects).
            self.gb_wdm_comp = getattr(_band_sorter, "gb_wdm_comp", None)
            self.gb_fd_comp = getattr(_band_sorter, "gb_fd_comp", None)
            # need to make sure is not mixed up in loop
            self.set_main_band_sorter_info(main_band_sorter, inds_main_band_sorter)
            return

        assert band_edges is not None and band_N_vals is not None
        self.force_backend = force_backend
        self.gb = gb
        # Domain computation objects, forwarded to the buffer in
        # :meth:`get_buffer` so the engine selection matches the parent
        # ACA's basis (WDMSettings -> gb_wdm_comp, FDSettings -> gb_fd_comp).
        self.gb_wdm_comp = gb_wdm_comp
        self.gb_fd_comp = gb_fd_comp
        # Task-b narrow per-band WDM slab extent (None = full active band).
        # Forwarded to each SubBandBuffer built in :meth:`get_buffer`; a plain
        # int/None so the copy-constructor's generic loop carries it through.
        self.wdm_band_slab_layers = wdm_band_slab_layers
        self.wdm_slab_guard_layers = wdm_slab_guard_layers
        self.waveform_kwargs = waveform_kwargs
        self.gb_branch_orig = gb_branch
        self.num_bands = len(band_edges) - 1
        self.band_edges = self.xp.asarray(band_edges)
        self.band_N_vals = self.xp.asarray(band_N_vals)
        self.ntemps, self.nwalkers, self.nleaves_max, self.ndim = gb_branch.shape
        self.orig_inds = self.xp.asarray(gb_branch.inds)
        self.keep_all_inds = keep_all_inds
        self.rj_prop = rj_prop

        if rj_prop is not None:
            if keep_all_inds:
                self.coords = self.xp.asarray(gb_branch.coords.reshape(-1, self.ndim))
                self.inds = self.orig_inds.flatten()
            else:
                self.coords = self.xp.asarray(gb_branch.coords[gb_branch.inds])
                self.inds = self.xp.ones(self.coords.shape[:-1], dtype=bool)

            if self.xp.any(~self.inds):
                # self.xp (run backend), NOT the module-level cp: on a
                # CPU-resolved run on a cupy-installed machine cp is cupy
                # while self.coords is numpy -- mixing them crashes the
                # assignment below (module-cp-vs-force_backend trap).
                new_sources = self.xp.full_like(self.coords[~self.inds], np.nan)
                fix = self.xp.full(new_sources.shape[0], True)
                while self.xp.any(fix):
                    new_sources[fix] = self.xp.asarray(
                        rj_prop.rvs(size=fix.sum().item())
                    )
                    fix = self.xp.any(self.xp.isnan(new_sources), axis=-1)

                self.coords[~self.inds] = new_sources

            proposal_logpdf = self.xp.zeros(self.coords.shape[0])

            batch_here = int(1e6)
            # Batch bounds END at N (eind is exclusive): the old N-1 endpoint
            # silently left the LAST row's proposal logpdf at 0.0, and the
            # [-1] indexing crashed outright on an empty coords array
            # (2026-08-13). np.arange handles N=0 with no special case.
            inds_splitting = np.arange(0, self.coords.shape[0], batch_here)
            if inds_splitting.size == 0 or inds_splitting[-1] != self.coords.shape[0]:
                inds_splitting = np.concatenate(
                    [inds_splitting, np.array([self.coords.shape[0]])]
                )

            for stind, eind in zip(inds_splitting[:-1], inds_splitting[1:]):
                proposal_logpdf[stind:eind] = self.xp.asarray(
                    rj_prop.logpdf(self.coords[stind:eind])
                )
            if self.backend.uses_cupy:
                self.xp.get_default_memory_pool().free_all_blocks()

            if keep_all_inds:
                self.factors = (self.xp.asarray(proposal_logpdf) * -1) * (~self.orig_inds).flatten() + (
                    self.xp.asarray(proposal_logpdf) * +1
                ) * (self.orig_inds).flatten()
                tmp_inds_shaped = self.xp.full_like(self.orig_inds, True)
            else:
                assert self.xp.all(self.inds)
                self.factors = self.xp.asarray(proposal_logpdf) * +1
                tmp_inds_shaped = self.orig_inds.copy()

        else:
            self.coords = self.xp.asarray(gb_branch.coords[gb_branch.inds])
            self.inds = self.xp.ones(self.coords.shape[:-1], dtype=bool)
            self.factors = self.xp.ones_like(self.inds)
            tmp_inds_shaped = self.orig_inds.copy()

        self.has_run_rj = self.xp.zeros_like(self.inds)
        self.num_sources = self.coords.shape[0]
        self.set_main_band_sorter_info(main_band_sorter, inds_main_band_sorter)

        self.max_data_store_size = max_data_store_size
        self.transform_fn = transform_fn

        self.temp_inds = self.xp.repeat(
            self.xp.arange(self.ntemps), self.nwalkers * self.nleaves_max
        ).reshape(self.ntemps, self.nwalkers, self.nleaves_max)[tmp_inds_shaped]
        self.walker_inds = self.xp.tile(
            self.xp.arange(self.nwalkers), (self.ntemps, self.nleaves_max, 1)
        ).transpose((0, 2, 1))[tmp_inds_shaped]
        self.leaf_inds = self.xp.tile(
            self.xp.arange(self.nleaves_max), ((self.ntemps, self.nwalkers, 1))
        )[tmp_inds_shaped]

        # Per-source frequency: the sampled f0 column when the container has
        # one; the per-leaf f0 fill (Eryn per-leaf fill_dict) otherwise --
        # needs ``leaf_inds``, so computed after the label arrays above.
        self.freqs = self._source_freqs_hz()
        self.band_inds = self.xp.searchsorted(band_edges, self.freqs, side="right") - 1
        self.special_band_inds = self.get_special_band_index(
            self.temp_inds, self.walker_inds, self.band_inds
        )

        self.orig_temp_inds = self.temp_inds.copy()
        self.orig_walker_inds = self.walker_inds.copy()
        self.orig_leaf_inds = self.leaf_inds.copy()
        self.orig_special_band_inds = self.special_band_inds.copy()
        self.orig_band_inds = self.band_inds.copy()

    def _source_freqs_hz(self) -> np.ndarray:
        """Per-source frequency in Hz.

        Sampling-basis f0 is in mHz; when f0 is not a sampled column it must
        be a per-leaf transform fill and each source's value is looked up by
        its leaf index (fixed-source branches, e.g. VGBs).
        """
        tf = self.transform_fn
        input_basis = list(getattr(tf, "input_basis", []) or [])
        if "f0" in input_basis:
            return self.coords[:, input_basis.index("f0")] / 1e3
        if tf is None or getattr(tf, "n_leaf_fills", None) is None:
            # legacy layout without a container: f0 at sampling column 1
            return self.coords[:, 1] / 1e3
        fill_keys = list(tf.original_fill_dict[0].keys())
        if "f0" not in fill_keys:
            raise ValueError(
                "BandSorter: 'f0' is neither a sampled column nor a per-leaf "
                "fill key of the transform container."
            )
        f0_fill_mhz = self.xp.asarray(tf.fill_dict["fill_values"])[
            :, fill_keys.index("f0")
        ]
        return f0_fill_mhz[self.leaf_inds] / 1e3

    def set_main_band_sorter_info(self, main_band_sorter, inds_main_band_sorter):
        if main_band_sorter is None:
            self.inds_main_band_sorter = self.xp.arange(self.num_sources)
        else:
            self.inds_main_band_sorter = inds_main_band_sorter

        self.main_band_sorter = main_band_sorter

    @property
    def coords_in(self) -> np.ndarray:
        # leaf_inds is ignored by scalar-fill containers and required by
        # per-leaf-fill ones (Eryn per-leaf fill_dict).
        return self.transform_fn.both_transforms(
            self.coords, xp=self.xp, leaf_inds=self.leaf_inds
        )

    def get_special_band_index(
        self, temp_inds: np.ndarray, walker_inds: np.ndarray, band_inds: np.ndarray
    ) -> np.ndarray:
        return pack_special_index(temp_inds, walker_inds, band_inds, self.nwalkers)

    def get_separate_inds_from_special_index(self, special_band_inds: np.ndarray) -> tuple:
        return unpack_special_index(special_band_inds, self.nwalkers)

    @property
    def special_index_check(self) -> bool:
        return self.xp.all(
            self.special_band_inds
            == self.get_special_band_index(self.temp_inds, self.walker_inds, self.band_inds)
        )

    def exchange_cell_labels(self, specials_a, temp_a, walkers_a,
                             specials_b, temp_b, walkers_b, bands=None) -> None:
        """Pairwise-swap the (temp, walker) labels of the sources in cell
        sets ``a`` and ``b``.

        ``specials_a[k]`` exchanges with ``specials_b[k]``: every source in
        cell ``a_k`` is relabelled to ``(temp_b, walkers_b[k])`` and vice
        versa (band indices are unchanged -- tempering swaps stay within a
        band; pass ``bands`` to assert that). Both membership maps are
        computed BEFORE any mutation so the two directions cannot see each
        other's relabelled sources.
        """
        xp = self.xp

        def _map(specials_from):
            order = xp.argsort(specials_from.flatten())
            keep = xp.isin(self.special_band_inds, specials_from)
            take = order[xp.searchsorted(
                specials_from[order], self.special_band_inds[keep], side="left"
            )]
            return keep, take

        keep_a, take_a = _map(specials_a)   # take_* indexes the OTHER set's rows
        keep_b, take_b = _map(specials_b)

        if bands is not None:
            assert xp.all(self.band_inds[keep_a] == bands[take_a])
            assert xp.all(self.band_inds[keep_b] == bands[take_b])

        self.special_band_inds[keep_a] = specials_b[take_a]
        self.temp_inds[keep_a] = temp_b
        self.walker_inds[keep_a] = walkers_b[take_a]

        self.special_band_inds[keep_b] = specials_a[take_b]
        self.temp_inds[keep_b] = temp_a
        self.walker_inds[keep_b] = walkers_a[take_b]

    @property
    def N_vals(self) -> np.ndarray:
        return self.band_N_vals[self.band_inds]

    @property
    def unique_N(self) -> np.ndarray:
        return self.xp.unique(self.N_vals)

    def get_subset(self, *args, **kwargs):
        subset_inds = self.get_subset_inds(*args, **kwargs)

        if len(subset_inds) == 0:
            return None

        # source information
        subset = BandSorter(
            self,
            inds_subset=subset_inds,
            main_band_sorter=self.main_band_sorter,
            inds_main_band_sorter=self.inds_main_band_sorter[subset_inds],
        )
        # band information
        return subset

    def get_subset_inds(self, *args, **kwargs):
        subset_bool = self.get_subset_bool(*args, **kwargs)
        return self.xp.arange(len(subset_bool))[subset_bool]

    def get_subset_bool(
        self,
        units: Optional[int] = None,
        remainder: Optional[int] = None,
        temp: Optional[int] = None,
        walker: Optional[int] = None,
        leaf: Optional[int] = None,
        band: Optional[int] = None,
        apply_inds: Optional[bool] = False,
        special_band_inds: Optional[int | np.ndarray] = None,
        extra_bool: Optional[np.ndarray] = None,
        full_bool: Optional[np.ndarray] = None,
    ) -> np.ndarray:

        inds_keep = self.xp.ones_like(self.band_inds, dtype=bool)

        if full_bool is None:
            if band is not None:
                assert isinstance(band, int)
                inds_keep &= self.band_inds == band
            elif units is not None or remainder is not None:
                assert units is not None and remainder is not None
                inds_keep &= self.band_inds % units == remainder

            if temp is not None:
                assert isinstance(temp, int)
                inds_keep &= self.temp_inds == temp
            if walker is not None:
                assert isinstance(walker, int)
                inds_keep &= self.walker_inds == walker
            if leaf is not None:
                assert isinstance(leaf, int)
                inds_keep &= self.leaf_inds == leaf

            if extra_bool is not None:
                assert isinstance(extra_bool, self.xp.ndarray)
                assert extra_bool.shape == (self.num_sources,)
                inds_keep &= extra_bool

            if apply_inds:
                inds_keep &= self.inds

            if special_band_inds is not None:
                if isinstance(special_band_inds, int):
                    inds_keep &= self.special_band_inds == special_band_inds

                elif isinstance(special_band_inds, self.xp.ndarray):
                    inds_keep &= self.xp.isin(self.special_band_inds, special_band_inds)

        else:
            assert full_bool.shape[0] == self.num_sources
            inds_keep = full_bool

        return inds_keep

    @property
    def main_band_sorter(self):
        main_band_sorter = self if self._main_band_sorter is None else self._main_band_sorter
        return main_band_sorter

    @main_band_sorter.setter
    def main_band_sorter(self, main_band_sorter):
        self._main_band_sorter = main_band_sorter

    def get_buffer(
        self, acs, special_indices_unique, inds_fill=None, buffer_obj=None, **kwargs
    ) -> SubBandBuffer:

        num_band_preload = len(special_indices_unique)

        # Array module from the SORTER's arrays, not the module-level ``cp``:
        # on a CPU run on a machine where cupy imports (cluster), ``cp`` is
        # cupy while the sorter holds numpy -- cp.isin then dies with
        # ``TypeError: Unsupported type <class 'numpy.ndarray'>`` (and
        # cp.arange/cp.asarray would silently UPLOAD host data instead).
        xp = get_array_module(self.main_band_sorter.special_band_inds)

        # CAN USE main_band_sorter TO GET SOURCES IN BANDS OF INTEREST THAT ARE NOT CURRENTLY OF INTEREST THEMSELVES

        sources_now_map = xp.arange(self.main_band_sorter.special_band_inds.shape[0])[
            xp.isin(self.main_band_sorter.special_band_inds, special_indices_unique)
        ]

        # NOTE: self.main_band_sorter.inds needed to only inject real sources
        # inject sources must include sources that have been turned off in these bands
        sources_inject_now_map = xp.arange(self.main_band_sorter.special_band_inds.shape[0])[
            xp.isin(self.main_band_sorter.special_band_inds, special_indices_unique)
            & self.main_band_sorter.inds
        ]

        # separate out inds
        temp_inds_now, walker_inds_now, band_inds_now = self.get_separate_inds_from_special_index(
            special_indices_unique
        )

        all_unique_band_combos = xp.asarray([temp_inds_now, walker_inds_now, band_inds_now]).T
        num_bands_here_total = all_unique_band_combos.shape[0]
        num_bands_now = special_indices_unique.shape[0]

        points_curr_tmp = self.main_band_sorter.coords[sources_now_map].copy()
        curr_special_band_inds = self.main_band_sorter.special_band_inds[sources_now_map].copy()

        # sort these sources by band
        if inds_fill is None:
            inds_fill = xp.arange(num_band_preload)
            assert buffer_obj is None
            buffer_obj = SubBandBuffer(
                self.rj_prop,
                self.nwalkers,
                self.gb,
                self.band_edges,
                self.band_N_vals,
                all_unique_band_combos,
                points_curr_tmp,
                num_bands_now,
                acs.nchannels,
                self.max_data_store_size,
                special_indices_unique,
                self.transform_fn,
                self.waveform_kwargs,
                # Frequency spacing for the band-index math. FDSettings uses the
                # FD bin resolution ``.df``; WDMSettings uses ``.layer_df`` so the
                # ``band_edges / df`` math yields WDM *layer* indices -- the same
                # quantity the WDM likelihood engine addresses by
                # (WDMBandLikelihoodEngine uses ``basis_settings.layer_df``).
                (acs.settings.df if isinstance(acs.settings, FDSettings)
                 else acs.settings.layer_df),
                sources_now_map,
                sources_inject_now_map,
                self.main_band_sorter.special_band_inds[sources_now_map],
                basis_settings=acs.settings,
                gb_wdm_comp=self.gb_wdm_comp,
                gb_fd_comp=self.gb_fd_comp,
                force_backend=self.force_backend,
                wdm_band_slab_layers=self.wdm_band_slab_layers,
                wdm_slab_guard_layers=self.wdm_slab_guard_layers,
                opt_snr_rej_samp_limit=getattr(
                    self, "opt_snr_rej_samp_limit", 5.0),
                snr_rej_detected=getattr(
                    self, "snr_rej_detected", False),
                **kwargs,
            )

        else:
            assert isinstance(buffer_obj, SubBandBuffer)
            assert inds_fill.max() <= buffer_obj.num_bands_now
            # THIS NEEDS TO HAPPEN before updating data
            buffer_obj.update_special_indices(special_indices_unique, inds_fill=inds_fill)
            if int(inds_fill.shape[0]) == int(buffer_obj.num_bands_now):
                # FULL rebind (cross-unit reuse of a cached buffer: same
                # allocation, new parity unit): refresh the subset-derived
                # source maps so the rebind is exactly construction minus
                # allocation. Partial rebinds (in-round slot rotation) must
                # NOT touch these -- their maps cover only the rotated slots.
                buffer_obj.sources_now_map = sources_now_map
                buffer_obj.sources_inject_now_map = sources_inject_now_map
                buffer_obj.params_interest = points_curr_tmp
                buffer_obj.special_band_inds = curr_special_band_inds
                buffer_obj.now_index = buffer_obj.get_index(curr_special_band_inds)

        buffer_obj.fill_buffer_residual_and_psd_from_acs(acs, inds_fill=inds_fill)
        buffer_obj.parent_acs = acs
        # includes sources in these sub-bands that are no longer getting proposals
        coords_to_inject = self.main_band_sorter.coords[sources_inject_now_map].copy()
        inj_special_indices_now = self.main_band_sorter.special_band_inds[
            sources_inject_now_map
        ].copy()

        inject_index = buffer_obj.get_index(inj_special_indices_now)
        inject_N_vals = self.band_N_vals[
            self.main_band_sorter.band_inds[sources_inject_now_map]
        ].copy()

        assert len(inject_index) == len(coords_to_inject)

        # leaf identity threads the per-leaf transform fills (ignored by
        # scalar-fill containers)
        inject_leaf_inds = self.main_band_sorter.leaf_inds[
            sources_inject_now_map
        ].copy()
        inj_args = (coords_to_inject, inject_index, inject_N_vals)
        if buffer_obj.use_template_arr:
            # The twin fill below is additive: without this reset a cached-
            # buffer rebind inherits the previous bind's (post-swap)
            # templates and every ll scored from the twin is contaminated.
            # No-op cost on a fresh allocation (already zero).
            buffer_obj.reset_template_buffers(inds_fill=inds_fill)
            buffer_obj.add_sources_to_template_buffer(
                *inj_args, leaf_inds=inject_leaf_inds
            )
        else:
            buffer_obj.add_sources_to_band_buffer(
                *inj_args, leaf_inds=inject_leaf_inds
            )

        return buffer_obj

    # ------------------------------------------------------------------
    # Group-stretch friends
    # ------------------------------------------------------------------

    def build_friend_index(self, nfriends: int) -> bool:
        """Build the sorted cold-chain frequency table + per-source friend windows.

        Friends are cold-chain (``temp == 0``, ``inds == True``) sources close
        in frequency: for every source (any temperature) we store the start of
        an ``nfriends``-wide window into the frequency-sorted cold-chain
        coordinate table, centred on the source's own frequency (clamped at
        the table edges). :meth:`draw_friends` then draws one friend uniformly
        from that window per source.

        Returns ``False`` (and clears the table) when there are too few
        cold-chain sources to form a window.
        """
        table = self.build_friend_table(nfriends)
        return self.index_friends(table, nfriends)

    def build_friend_table(self, nfriends: int):
        """The frequency-sorted cold-chain coordinate table, or ``None``.

        Split out of :meth:`build_friend_index` so the table can be built ONCE
        per larger iteration and shared across the GB moves of a cycle, while
        :meth:`index_friends` still runs per proposal (each proposal's sorter
        holds a different set of sources).
        """
        cold = self.inds & (self.temp_inds == 0)
        if int(cold.sum()) < max(2, int(nfriends)):
            return None
        cold_coords = self.coords[cold]
        order = self.xp.argsort(cold_coords[:, 1])
        return cold_coords[order].copy()

    def index_friends(self, friends_coords_sorted, nfriends: int) -> bool:
        """Window every source into ``friends_coords_sorted`` (per proposal)."""
        self.nfriends = int(nfriends)
        if friends_coords_sorted is None or len(friends_coords_sorted) < max(
            2, self.nfriends
        ):
            self.friend_start_inds = None
            return False

        n_cold = len(friends_coords_sorted)
        self.friends_coords_sorted = friends_coords_sorted
        self.friends_freqs_sorted = friends_coords_sorted[:, 1].copy()

        starts = (
            self.xp.searchsorted(self.friends_freqs_sorted, self.coords[:, 1], side="right")
            - self.nfriends // 2
        )
        self.friend_start_inds = self.xp.clip(starts, 0, n_cold - self.nfriends).astype(
            self.xp.int32
        )
        return True

    def draw_friends(self, source_ids):
        """One random friend (coordinate row) per source in ``source_ids``.

        Requires :meth:`build_friend_index` to have been run this proposal.
        """
        starts = self.friend_start_inds[source_ids]
        deviation = self.xp.random.randint(0, self.nfriends, size=len(starts))
        take = self.xp.clip(starts + deviation, 0, len(self.friends_freqs_sorted) - 1)
        return self.friends_coords_sorted[take]

    # ------------------------------------------------------------------
    # Shared info-matrix Cholesky table (group-stretch friends analogue)
    # ------------------------------------------------------------------

    def build_infomat_index(self, infomat_freqs_sorted, infomat_chol_sorted) -> bool:
        """Map every source onto the NEAREST entry of a shared Cholesky table.

        The direct counterpart of :meth:`build_friend_index`: that method
        windows each source into a frequency-sorted table of cold-chain
        COORDINATES, this one points each source at the single closest entry
        of a frequency-sorted table of cold-chain proposal CHOLESKY factors
        (built by the move every N iterations, since it needs the likelihood
        model the sorter has no handle on).  Sources at every temperature and
        walker index into the same cold-chain table, exactly as friends do.

        Returns ``False`` (and clears the index) when the supplied table is
        empty, so the caller can fall back to computing factors directly.
        """
        if infomat_freqs_sorted is None or len(infomat_freqs_sorted) == 0:
            self.infomat_take_inds = None
            return False

        self.infomat_freqs_sorted = infomat_freqs_sorted
        self.infomat_chol_sorted = infomat_chol_sorted
        n_tab = len(infomat_freqs_sorted)

        # searchsorted gives the right-hand neighbour; pick whichever of the
        # bracketing pair is closer in frequency (clamped at the table edges).
        hi = self.xp.searchsorted(infomat_freqs_sorted, self.coords[:, 1], side="left")
        hi = self.xp.clip(hi, 0, n_tab - 1)
        lo = self.xp.clip(hi - 1, 0, n_tab - 1)
        d_hi = self.xp.abs(infomat_freqs_sorted[hi] - self.coords[:, 1])
        d_lo = self.xp.abs(infomat_freqs_sorted[lo] - self.coords[:, 1])
        self.infomat_take_inds = self.xp.where(d_lo <= d_hi, lo, hi).astype(
            self.xp.int32
        )
        return True

    def draw_infomat(self, source_ids):
        """Nearest-in-frequency Cholesky factor per source in ``source_ids``.

        Requires :meth:`build_infomat_index` to have been run this proposal;
        mirrors :meth:`draw_friends`, but the pick is deterministic (nearest)
        rather than a uniform draw from a window.
        """
        take = self.infomat_take_inds[source_ids]
        return self.infomat_chol_sorted[take]

    def get_band_info(self):

        uni_special, uni_special_counts = cp.unique(
            self.special_band_inds[self.inds], return_counts=True
        )
        uni_temp_inds, uni_walker_inds, uni_band_inds = self.get_separate_inds_from_special_index(
            uni_special
        )

        num_bands = len(self.band_edges) - 1
        band_counts = np.zeros((self.ntemps, self.nwalkers, num_bands), dtype=int)
        band_counts[_to_numpy(uni_temp_inds), _to_numpy(uni_walker_inds), _to_numpy(uni_band_inds)] = (
            _to_numpy(uni_special_counts)
        )

        return {"band_counts": band_counts}
