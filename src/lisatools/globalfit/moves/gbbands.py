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
from ...utils.parallelbase import LISAToolsParallelModule
from ...utils.utility import asnumpy

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


def pack_special_index(temp_inds, walker_inds, band_inds, nwalkers: int):
    """Pack ``(temp, walker, band)`` triplets into scalar special indices.

    ``special = (temp * nwalkers + walker) * 1e6 + band``. Works
    elementwise on array inputs of any matching shape.
    """
    return (temp_inds * nwalkers + walker_inds) * _SPECIAL_INDEX_BASE + band_inds


def unpack_special_index(special_band_inds, nwalkers: int) -> tuple:
    """Recover ``(temp, walker, band)`` arrays from packed special indices."""
    temp_walker_inds = cp.floor(special_band_inds / _SPECIAL_INDEX_BASE).astype(int)
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

    def advance(self):
        """Retire finished slots and stage pending cells into them.

        Returns ``(inds_fill, new_specials)``: the buffer slot positions to
        repack and the special indices of the cells to load there. Slots
        with no pending replacement are deactivated.
        """
        finished = self.slot_active & (
            self.cell_run[self.slot_cell] >= self.cell_counts[self.slot_cell]
        )
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
        now_index = (
            self.special_indices_unique_sort[
                cp.searchsorted(
                    self.special_indices_unique[self.special_indices_unique_sort],
                    special_inds_test,
                    side="right",
                )
                - 1
            ]
        ).astype(cp.int32)
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
        force_backend="gpu",
        use_template_arr=False,
        basis_settings: Optional[DomainSettingsBase] = None,
        gb_wdm_comp=None,
        gb_fd_comp=None,
        *args,
        **kwargs,
    ):
        # Deferred import: gb_likelihood imports nothing from here, but keep
        # the module import graph acyclic if that ever changes.
        from gbgpu.gb_likelihood import make_band_likelihood_engine

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
        self.waveform_kwargs = waveform_kwargs
        self.opt_snr_rej_samp_limit = opt_snr_rej_samp_limit
        self.use_template_arr = use_template_arr

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

        self._likelihood_engine = make_band_likelihood_engine(
            self._basis_settings,
            gb=self.gb,
            gb_fd_comp=self.gb_fd_comp,
            gb_wdm_comp=self.gb_wdm_comp,
            nchannels=self.nchannels,
            tdi_channel_setup=self.tdi_channel_setup,
            df=float(self.df) if not hasattr(self.df, "item") else self.df.item(),
            start_freq_inds=getattr(self, "start_freq_inds", None),
            data_length=self.data_length,
            opt_snr_rej_samp_limit=self.opt_snr_rej_samp_limit,
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
    def _per_band_data_shape(self) -> tuple:
        """Shape of a single band's residual buffer (one AC's data_res_arr)."""
        if isinstance(self._basis_settings, FDSettings):
            return (self.nchannels, self._fd_store_length)
        elif isinstance(self._basis_settings, WDMSettings):
            # First-cut: each per-band buffer covers the FULL WDM active grid
            # (Nf_active layers x Nt_active time pixels). The WDM kernel
            # currently uses a single global [ind_min_f, ind_max_f] rather
            # than per-band offsets, so per-band slicing on the layer axis is
            # a follow-on once the kernel takes per-band layer offsets.
            Nf_active = self._basis_settings.Nf_active
            Nt_active = self._basis_settings.Nt_active
            return (self.nchannels, Nf_active, Nt_active)
        else:
            raise NotImplementedError(
                f"Buffer does not support basis domain {type(self._basis_settings).__name__}."
            )

    @property
    def _per_band_sens_shape(self) -> tuple:
        """Shape of a single band's inverse-PSD buffer (one AC's sens_mat.invC)."""
        if isinstance(self._basis_settings, FDSettings):
            if self.tdi_channel_setup == "XYZ":
                return (self.nchannels, self.nchannels, self._fd_store_length)
            return (self.nchannels, self._fd_store_length)
        elif isinstance(self._basis_settings, WDMSettings):
            Nf_active = self._basis_settings.Nf_active
            Nt_active = self._basis_settings.Nt_active
            if self.tdi_channel_setup == "XYZ":
                return (self.nchannels, self.nchannels, Nf_active, Nt_active)
            return (self.nchannels, Nf_active, Nt_active)
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
            # First-cut: per-band WDMSettings matches the parent grid (full
            # WDM active band). A true per-band sliced WDMSettings becomes
            # possible once the WDM kernel takes per-band
            # [ind_min_f, ind_max_f] arrays; until then we share the parent.
            parent = self._basis_settings
            return WDMSettings(
                Nf=parent.Nf,
                Nt=parent.Nt,
                dt=parent.data_dt,
                t0=parent.t0,
                oversample=parent.oversample,
                window=parent.window,
                omega=parent.omega,
                min_freq=parent.ind_min_f * parent.layer_df,
                max_freq=parent.ind_max_f * parent.layer_df,
                min_time=parent.ind_min_t * parent.layer_dt,
                max_time=parent.ind_max_t * parent.layer_dt,
            )
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

        ac_list = []
        for _ in range(self.num_bands_now):
            res_data = cp.zeros(data_shape, dtype=data_dtype)
            data_domain = per_band_settings.associated_class(res_data, per_band_settings)
            sm = SensitivityMatrixBase(per_band_settings, skip_inv_det=True)
            sm.sens_mat = cp.zeros(sens_shape, dtype=sens_dtype)
            sm.invC = cp.zeros(sens_shape, dtype=sens_dtype)
            sm.channel_shape = sens_shape[: -len(per_band_settings.basis_shape_active)]
            ac_list.append(AnalysisContainer(data_domain, sm))

        gpus_in = getattr(self.gb, "gpus", None) if self.backend.uses_cupy else None
        # Multi-GPU at the GB band-tree level: pass the full gpus list and a
        # striped band assignment so consecutive bands land on different
        # GPUs. The BandSorter even/odd within-pass invariant keeps bands in
        # one pass non-overlapping in time-frequency support, so striping is
        # safe. The per-band accessors (band_buffer / psd_buffer /
        # template_buffer) automatically fall back to a single ndarray view
        # for single-GPU runs and return a BandView (multi-shard router)
        # otherwise -- see the accessor block above.
        gpu_assignment = (
            band_gpu_assignment(len(ac_list), list(gpus_in)) if gpus_in else None
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
        # has no effect on the shards). Either way numerator_in needs the
        # explicit ``.copy()`` so the in-place ``-= self.template_buffer``
        # below doesn't corrupt the residual buffer.
        numerator_in = self._materialize(self.band_buffer).copy()
        if self.use_template_arr:
            numerator_in -= self._materialize(self.template_buffer)
        psd_buffer = self._materialize(self.psd_buffer)

        # Domain-generic inner product: <a|b> = 4 sum(a* invC b) * dc where
        # dc is the basis measure (FD: df; WDM: the pixel measure) -- the
        # same convention as lisatools.diagnostic.inner_product. Trailing
        # basis axes (FD: k; WDM: (Nf, Nt)) are flattened.
        nb = numerator_in.shape[0]
        nc = self.nchannels
        num_flat = numerator_in.reshape(nb, nc, -1)
        dc = float(self.settings.differential_component)

        if self.tdi_channel_setup == "XYZ":
            psd_flat = psd_buffer.reshape(nb, nc, nc, -1)
            # b=bands, i/j=channels, k=flattened basis
            source_term = (
                - (1.0 / 2.0) * 4.0 * dc
                * cp.einsum(
                    "bik,bijk,bjk->b", num_flat.conj(), psd_flat, num_flat
                ).real
            )

            if noise_only:
                raise NotImplementedError("Noise-only likelihood requires log=determinant over frequency for XYZ CSD.")

        else:
            psd_flat = psd_buffer.reshape(nb, nc, -1)
            source_term = (
                - (1.0 / 2.0) * 4.0 * dc
                * cp.sum((num_flat.conj() * num_flat) * psd_flat, axis=(1, 2)).real
            )

            if noise_only:
                return -cp.sum(cp.log(cp.abs(1 / psd_buffer[psd_buffer != 0.0])))

        if source_only:
            return source_term

        # Diagonal noise_term fall_back # TODO check if this is sufficient not used currently anyway
        psd_term = -cp.sum(cp.log(cp.abs(psd_buffer[psd_buffer != 0.0])))
        if self.tdi_channel_setup == "XYZ":
            warnings.warn("The current psd ll calculation is not correct for XYZ CSD channel setup.")

        return source_term + psd_term

    # Explicit alias while callers migrate off the ``likelihood`` name (which
    # shadows the inherited per-AC ACA dispatch).
    band_likelihoods = likelihood

    def get_swap_ll(self, params_remove, params_add, data_index, N_vals, phase_maximize=False):
        """Per-proposal swap log-likelihood difference.

        Domain-agnostic: dispatches to ``self._likelihood_engine.get_swap_ll``,
        which is either :class:`FDBandLikelihoodEngine` or
        :class:`WDMBandLikelihoodEngine` depending on the buffer's
        ``basis_settings``. Both engines take the per-band ACA (``self``)
        and the physical params, and return a :class:`SwapLLResult`. The
        rejection-sampling clamp and the phase-maximisation correction live
        here so the engine stays a thin wrapper around the kernel.
        """
        params_remove_phys = self.transform_fn.both_transforms(params_remove, xp=cp)
        params_add_phys = self.transform_fn.both_transforms(params_add, xp=cp)

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
            params_add[kept, 3] = params_add[kept, 3] - result.phase_angle

        # Rejection sampling on SNR: only applied to *add* proposals (the
        # remove side's opt_snr is meaningless when amp_add is tiny).
        reject = self.xp.zeros(kept.shape[0], dtype=bool)
        reject[kept] = (result.opt_snr_add[kept] < self.opt_snr_rej_samp_limit) & (
            params_add_phys[kept, 0] > 1e-30
        )
        ll_diff[reject] = -1e300

        return ll_diff

    def get_ll(self, params, data_index, noise_index, N_vals, phase_maximize=False,
               return_inner_products=False):
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
        params_phys = self.transform_fn.both_transforms(params, xp=cp)
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

    def setup_in_model_likelihood(self, params, data_index, N_vals=None) -> None:
        """Per-source in-model likelihood setup (once per repeat block).

        Forwards the picked sources' CURRENT sampling-basis params
        (transformed to physical) plus their buffer slots to the engine's
        ``setup_in_model`` hook. Chunked-het / FD engines no-op; a sig-het
        computation builds its heterodyne reference against the
        source-free cell residuals here and holds it constant until
        :meth:`clear_in_model_likelihood`. Call AFTER the sources are
        removed from the residual, BEFORE the reference ll of the repeat
        block is computed."""
        params_phys = self.transform_fn.both_transforms(params, xp=cp)
        self._likelihood_engine.setup_in_model(
            self, params_phys, data_index, N_vals=N_vals)

    def clear_in_model_likelihood(self) -> None:
        """Deactivate the per-source in-model setup (no-op engines ignore)."""
        self._likelihood_engine.clear_in_model()

    def get_add_ll(self, params, data_index, noise_index, N_vals, phase_maximize=False):
        """Log-likelihood delta of ADDING a source to the model.

        ``ll(r - h) - ll(r) = <r|h> - 0.5 <h|h>`` where ``r`` is the current
        cell residual (which does not contain ``h``). This is a delta, not a
        singular log-likelihood -- the ``d_d`` term cancels. Computed from
        the :attr:`d_h_out` / :attr:`h_h_out` stashed by :meth:`get_ll`;
        :attr:`phase_angle` is available after the call when
        ``phase_maximize=True``. Sources rejected by the engine's bounds
        check (:attr:`kept_out`) come back as ``-1e300``.
        """
        self.get_ll(params, data_index, noise_index, N_vals, phase_maximize=phase_maximize)
        delta = self.d_h_out.real - 0.5 * self.h_h_out.real
        delta[~self.kept_out] = -1e300
        return delta

    def get_removal_ll(self, params, data_index, noise_index, N_vals):
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
        self.get_ll(params, data_index, noise_index, N_vals)
        delta = -self.d_h_out.real - 0.5 * self.h_h_out.real
        delta[~self.kept_out] = -1e300
        return delta

    def get_ll_grad(self, params, data_index, noise_index, N_vals,
                     *, param_eps=None, chunk=None):
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
        params_phys = self.transform_fn.both_transforms(params, xp=cp)
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
                 psd_fix=False, psd_floor_rel=1e-30):
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
        params_phys = self.transform_fn.both_transforms(params, xp=cp)
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

    def fill_buffer_residual_and_psd_from_acs(
        self, acs: AnalysisContainerArray, inds_fill: Optional[cp.ndarray] = None
    ) -> None:
        # The outer ``acs`` is accessed via tuple-fancy indexing
        # ``data_shaped[0][inds1, inds2, inds3]`` (3-tuple for AET, 5-tuple
        # for XYZ CSD). BandView routes the tuple-fancy index through the
        # owning shard at the right intra-shard band position; on
        # single-shard ACAs the reshape view is touched directly. No
        # outer-buffer materialisation needed.
        if inds_fill is None:
            inds_fill = cp.arange(self.num_bands_now)

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
            # First-cut WDM fill index map. Per-band buffers cover the full
            # WDM active grid, so the index map is the simplest possible: it
            # picks each band's entire (channel, Nf_active, Nt_active) slab
            # out of the parent ACA. The data axis position is taken from
            # unique_band_combos[:, 1] (the parent data index for that band).
            if inds_fill is None:
                inds_fill = cp.arange(self.num_bands_now)

            Nf_active = self._basis_settings.Nf_active
            Nt_active = self._basis_settings.Nt_active

            if is_psd and self.tdi_channel_setup == "XYZ":
                # target shape: (len(inds_fill), nchannels, nchannels, Nf_active, Nt_active)
                # The parent WDM ACA's psd_shaped[0] has shape
                # ``(num_walkers, nchan, nchan, Nf_active, Nt_active)`` — one
                # entry per walker with channels as inner axes. Unlike the FD
                # path (which flattens walker*channel into axis 0), here we
                # index axis 0 with the raw walker index and need a full
                # 5-tuple to cover all five axes.
                inds1 = self.unique_band_combos[inds_fill, 1][:, None, None, None, None]
                inds2 = cp.arange(self.nchannels)[None, :, None, None, None]
                inds3 = cp.arange(self.nchannels)[None, None, :, None, None]
                inds4 = cp.arange(Nf_active)[None, None, None, :, None]
                inds5 = cp.arange(Nt_active)[None, None, None, None, :]
                return inds1, inds2, inds3, inds4, inds5

            # target shape: (len(inds_fill), nchannels, Nf_active, Nt_active)
            inds1 = self.unique_band_combos[inds_fill, 1][:, None, None, None]
            inds2 = cp.arange(self.nchannels)[None, :, None, None]
            inds3 = cp.arange(Nf_active)[None, None, :, None]
            inds4 = cp.arange(Nt_active)[None, None, None, :]
            return inds1, inds2, inds3, inds4
        if not isinstance(self._basis_settings, FDSettings):
            raise NotImplementedError(
                f"Buffer does not support basis domain {type(self._basis_settings).__name__}."
            )

        if inds_fill is None:
            inds_fill = cp.arange(self.num_bands_now)

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
            inds2 = cp.arange(self.nchannels)[None, :, None, None]
            inds3 = cp.arange(self.nchannels)[None, None, :, None]
            inds4 = start_inds[:, None, None, None] + cp.arange(
                self.band_buffer.shape[-1]
            )[None, None, None, :]
            return inds1, inds2, inds3, inds4

        else:
            # Target output shape: (len(inds_fill), self.nchannels, self.band_buffer.shape[-1])
            inds1 = self.unique_band_combos[inds_fill, 1][:, None, None]
            inds2 = cp.arange(self.nchannels)[None, :, None]
            inds3 = start_inds[:, None, None] + cp.arange(self.band_buffer.shape[-1])[None, None, :]

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
        tmp = self.template_buffer[slots_a].copy()
        self.template_buffer[slots_a] = self.template_buffer[slots_b]
        self.template_buffer[slots_b] = tmp[:]

    def _adjust_via_engine(
        self, factor, target_aca, params, params_index, N_vals, *args, **kwargs
    ) -> None:
        """Domain-agnostic dispatch into ``self._likelihood_engine.fill_template``.

        ``factor`` is +1 (write source into the template) or -1 (subtract it).
        ``target_aca`` selects which AnalysisContainerArray to write into
        (the buffer itself for residuals, or the template twin). Both share
        the same per-band geometry, so the engine doesn't need to know which
        one it's filling.
        """
        assert isinstance(factor, int) and (factor == -1 or factor == +1)
        params_phys = self.transform_fn.both_transforms(params, xp=cp)
        self._likelihood_engine.fill_template(
            target_aca,
            params_phys,
            params_index,
            N_vals,
            factor=factor,
            waveform_kwargs=self.waveform_kwargs,
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
    ):

        LISAToolsParallelModule.__init__(self, force_backend=force_backend)
        self.force_backend = force_backend

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
                self.coords = self.xp.asarray(gb_branch.coords.reshape(-1, 8))
                self.inds = self.orig_inds.flatten()
            else:
                self.coords = self.xp.asarray(gb_branch.coords[gb_branch.inds])
                self.inds = self.xp.ones(self.coords.shape[:-1], dtype=bool)

            if self.xp.any(~self.inds):
                new_sources = cp.full_like(self.coords[~self.inds], np.nan)
                fix = cp.full(new_sources.shape[0], True)
                while cp.any(fix):
                    new_sources[fix] = rj_prop.rvs(size=fix.sum().item())
                    fix = cp.any(cp.isnan(new_sources), axis=-1)

                self.coords[~self.inds] = new_sources

            proposal_logpdf = cp.zeros(self.coords.shape[0])

            batch_here = int(1e6)
            inds_splitting = np.arange(0, self.coords.shape[0], batch_here)
            if inds_splitting[-1] != self.coords.shape[0] - 1:
                inds_splitting = np.concatenate(
                    [inds_splitting, np.array([self.coords.shape[0] - 1])]
                )

            for stind, eind in zip(inds_splitting[:-1], inds_splitting[1:]):
                proposal_logpdf[stind:eind] = self.xp.asarray(
                    rj_prop.logpdf(self.coords[stind:eind])
                )
            if self.backend.uses_cupy:
                self.xp.get_default_memory_pool().free_all_blocks()

            if keep_all_inds:
                self.factors = (cp.asarray(proposal_logpdf) * -1) * (~self.orig_inds).flatten() + (
                    cp.asarray(proposal_logpdf) * +1
                ) * (self.orig_inds).flatten()
                tmp_inds_shaped = self.xp.full_like(self.orig_inds, True)
            else:
                assert self.xp.all(self.inds)
                self.factors = cp.asarray(proposal_logpdf) * +1
                tmp_inds_shaped = self.orig_inds.copy()

        else:
            self.coords = self.xp.asarray(gb_branch.coords[gb_branch.inds])
            self.inds = self.xp.ones(self.coords.shape[:-1], dtype=bool)
            self.factors = self.xp.ones_like(self.inds)
            tmp_inds_shaped = self.orig_inds.copy()

        self.has_run_rj = self.xp.zeros_like(self.inds)
        self.num_sources = self.coords.shape[0]
        self.set_main_band_sorter_info(main_band_sorter, inds_main_band_sorter)

        self.freqs = self.coords[:, 1] / 1e3
        self.band_inds = self.xp.searchsorted(band_edges, self.freqs, side="right") - 1
        self.max_data_store_size = max_data_store_size

        self.temp_inds = self.xp.repeat(
            self.xp.arange(self.ntemps), self.nwalkers * self.nleaves_max
        ).reshape(self.ntemps, self.nwalkers, self.nleaves_max)[tmp_inds_shaped]
        self.walker_inds = self.xp.tile(
            self.xp.arange(self.nwalkers), (self.ntemps, self.nleaves_max, 1)
        ).transpose((0, 2, 1))[tmp_inds_shaped]
        self.leaf_inds = self.xp.tile(
            self.xp.arange(self.nleaves_max), ((self.ntemps, self.nwalkers, 1))
        )[tmp_inds_shaped]
        self.special_band_inds = self.get_special_band_index(
            self.temp_inds, self.walker_inds, self.band_inds
        )

        self.orig_temp_inds = self.temp_inds.copy()
        self.orig_walker_inds = self.walker_inds.copy()
        self.orig_leaf_inds = self.leaf_inds.copy()
        self.orig_special_band_inds = self.special_band_inds.copy()
        self.orig_band_inds = self.band_inds.copy()
        self.transform_fn = transform_fn

    def set_main_band_sorter_info(self, main_band_sorter, inds_main_band_sorter):
        if main_band_sorter is None:
            self.inds_main_band_sorter = self.xp.arange(self.num_sources)
        else:
            self.inds_main_band_sorter = inds_main_band_sorter

        self.main_band_sorter = main_band_sorter

    @property
    def coords_in(self) -> np.ndarray:
        return self.transform_fn.both_transforms(self.coords, xp=self.xp)

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

        # CAN USE main_band_sorter TO GET SOURCES IN BANDS OF INTEREST THAT ARE NOT CURRENTLY OF INTEREST THEMSELVES

        sources_now_map = cp.arange(self.main_band_sorter.special_band_inds.shape[0])[
            cp.isin(self.main_band_sorter.special_band_inds, special_indices_unique)
        ]

        # NOTE: self.main_band_sorter.inds needed to only inject real sources
        # inject sources must include sources that have been turned off in these bands
        sources_inject_now_map = cp.arange(self.main_band_sorter.special_band_inds.shape[0])[
            cp.isin(self.main_band_sorter.special_band_inds, special_indices_unique)
            & self.main_band_sorter.inds
        ]

        # separate out inds
        temp_inds_now, walker_inds_now, band_inds_now = self.get_separate_inds_from_special_index(
            special_indices_unique
        )

        all_unique_band_combos = cp.asarray([temp_inds_now, walker_inds_now, band_inds_now]).T
        num_bands_here_total = all_unique_band_combos.shape[0]
        num_bands_now = special_indices_unique.shape[0]

        points_curr_tmp = self.main_band_sorter.coords[sources_now_map].copy()
        curr_special_band_inds = self.main_band_sorter.special_band_inds[sources_now_map].copy()

        # sort these sources by band
        if inds_fill is None:
            inds_fill = cp.arange(num_band_preload)
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
                **kwargs,
            )

        else:
            assert isinstance(buffer_obj, SubBandBuffer)
            assert inds_fill.max() <= buffer_obj.num_bands_now
            # THIS NEEDS TO HAPPEN before updating data
            buffer_obj.update_special_indices(special_indices_unique, inds_fill=inds_fill)

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

        inj_args = (coords_to_inject, inject_index, inject_N_vals)
        if buffer_obj.use_template_arr:
            buffer_obj.add_sources_to_template_buffer(*inj_args)
        else:
            buffer_obj.add_sources_to_band_buffer(*inj_args)

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
        self.nfriends = int(nfriends)
        cold = self.inds & (self.temp_inds == 0)
        n_cold = int(cold.sum())
        if n_cold < max(2, self.nfriends):
            self.friend_start_inds = None
            return False

        cold_coords = self.coords[cold]
        order = self.xp.argsort(cold_coords[:, 1])
        self.friends_coords_sorted = cold_coords[order].copy()
        self.friends_freqs_sorted = self.friends_coords_sorted[:, 1].copy()

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
