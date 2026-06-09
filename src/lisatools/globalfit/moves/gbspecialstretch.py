"""Galactic-binary specialized stretch / RJ moves and supporting infrastructure."""

from __future__ import annotations

import os
import time
import warnings
from copy import deepcopy
from inspect import Attribute
from types import ModuleType
from typing import Optional, Union, Tuple

import numpy as np
import numpy

from eryn.state import BranchSupplemental
from gbgpu.gbgpu import GBGPU
from gbgpu.utils.utility import get_N, get_fdot
from scipy import stats

from ... import sensitivity
from ...detector import sangria
from ...utils.constants import *
from ...analysiscontainer import (
    AnalysisContainer,
    AnalysisContainerArray,
    BandView,
    band_gpu_assignment,
)
# DataResidualArray is deprecated; AnalysisContainer now accepts DomainBase children
# directly (FDSignal / WDMSignal / TDSignal / STFTSignal).
from ...domains import DomainSettingsBase, FDSettings, WDMSettings
from ...sensitivity import SensitivityMatrixBase
from ...utils.parallelbase import LISAToolsParallelModule
from ...utils.utility import asnumpy
from ..galaxyglobal import fit_each_leaf, make_gmm, run_gb_bulk_search
from ._gb_likelihood import (
    BandLikelihoodEngine,
    FDBandLikelihoodEngine,
    SwapLLResult,
    WDMBandLikelihoodEngine,
    make_band_likelihood_engine,
)
from .globalfitmove import GFCombineMove, GlobalFitMove

# -*- coding: utf-8 -*-


try:
    import cupy as cp
    import cupy
    gpu_available = True
except ModuleNotFoundError:
    import numpy as cp

    gpu_available = False

from eryn.moves import GroupStretchMove, Move, StretchMove
from eryn.moves.multipletry import get_mt_computations, logsumexp
from eryn.paraensemble import ParaEnsembleSampler
from eryn.prior import ProbDistContainer, uniform_dist
from eryn.utils import PeriodicContainer
from eryn.utils.utility import groups_from_inds

from ...diagnostic import inner_product
from ...sampling.prior import FullGaussianMixtureModel, GBPriorWrap
from ...utils.utility import get_array_module, get_groups_from_band_structure, searchsorted2d_vec
from ..state import GFState

__all__ = ["GBSpecialStretchMove"]


class _NoOpMempool:
    """CPU stand-in for ``cupy.get_default_memory_pool()`` — calls become no-ops."""

    def free_all_blocks(self):
        return


# ``_to_numpy`` was the file-local cupy/numpy-agnostic ``.get()`` helper.
# Use the central :func:`lisatools.utils.utility.asnumpy` instead.
_to_numpy = asnumpy


def gb_search_func(comm, curr, main_rank, class_extra_gpus, class_ranks_list):
    """Worker entry point for ranks dedicated to the GB bulk search.

    # TODO/DOCS: full handshake protocol with ``main_rank``; the body is the
    canonical reference. Worker ranks pin themselves to a GPU, receive
    band-split assignments, run :func:`run_gb_bulk_search`, and report
    back.

    Args:
        comm: MPI communicator.
        curr: Global-fit info object.
        main_rank: Rank that orchestrates the search.
        class_extra_gpus: GPU indices owned by this move class.
        class_ranks_list: Ranks owned by this move class.
    """
    assert comm is not None

    # get current rank and get index into class_ranks_list
    print(f"INSIDE GB search, RANK: {comm.Get_rank()}")
    rank = comm.Get_rank()
    rank_index = class_ranks_list.index(rank)
    if rank_index == 0:
        comm_info = {"process_ranks_for_fit": class_ranks_list}
        print("waiting to send process ranks")
        comm.send(comm_info, dest=main_rank, tag=232342)
        print("sent process ranks")

    fit_each_leaf(rank, curr, main_rank, comm)


# def gb_search_func(comm, curr, main_rank, class_extra_gpus, class_ranks_list):
#     assert comm is not None

#     # get current rank and get index into class_ranks_list
#     print(f"INSIDE GB search, RANK: {comm.Get_rank()}")
#     rank = comm.Get_rank()
#     rank_index = class_ranks_list.index(rank)
#     gather_rank = class_ranks_list[0]
#     if rank_index == 0:
#         split_remainder = 1  # will fix this setup in the future
#         num_search = 2
#         gpu = class_extra_gpus[0]
#         comm_info = {"process_ranks_for_fit": class_ranks_list[1:]}
#         # run search here
#         run_gb_bulk_search(gpu, curr, comm, comm_info, main_rank, num_search, split_remainder)
#         pass

#     else:
#         # run GMM fit here
#         fit_each_leaf(rank, curr, gather_rank, comm)
#         pass


def fit_gmm(samples, comm, comm_info):
    """Fit a Gaussian mixture model to per-leaf GB chain samples.

    # TODO/DOCS: cross-rank protocol; mirrors the helper in
    :mod:`lisatools.globalfit.galaxyglobal` but is invoked by worker ranks
    of the GB special move set.
    """

    if len(samples) == 0:
        return None

    keep = np.arange(8)  # array([0, 1, 2, 4, 6, 7])

    if samples.ndim == 4:
        num_keep, num_samp, nwalkers_keep, ndim = samples.shape

        args = []
        for band in range(num_keep):
            args.append(samples[band].reshape(-1, ndim)[:, keep])

    elif samples.ndim == 2:
        max_groups = samples[:, 0].astype(int).max()

        args = []
        for group in np.unique(samples[:, 0].astype(int)):
            keep_samp = samples[:, 0].astype(int) == group
            if keep_samp.sum() > 0:
                if np.any(np.isnan(samples[keep_samp, 3:])) or np.any(
                    np.isinf(samples[keep_samp, 3:])
                ):
                    breakpoint()
                args.append(samples[keep_samp, 3:][:, keep])

    else:
        raise ValueError

    # for debugging
    # args = args[:1000]

    batch = 10000
    breaks = np.arange(0, len(args) + batch, batch)
    print("BREAKS", breaks)
    if len(breaks) == 1:
        breakpoint()
    process_ranks_for_fit = comm_info["process_ranks_for_fit"]
    gmm_info_all = []
    for i in range(len(breaks) - 1):
        start = breaks[i]
        end = breaks[i + 1]
        args_tmp = args[start:end]
        gmm_info = [None for tmp in args_tmp]
        gmm_complete = np.zeros(len(gmm_info), dtype=bool)

        # OPPOSITE
        # send_tags = comm_info["rec_tags"]
        # rec_tags = comm_info["send_tags"]
        outer_iteration = 0
        current_send_arg_index = 0
        current_status = [False for _ in process_ranks_for_fit]

        while np.any(~gmm_complete):
            time.sleep(0.1)
            if current_send_arg_index >= len(args_tmp) and np.all(~np.asarray(current_status)):
                current_send_arg_index = 0

            outer_iteration += 1
            if outer_iteration % 500 == 0:
                print(
                    f"ITERATION: {outer_iteration}, need:",
                    np.sum(~gmm_complete),
                    current_status,
                )

            for proc_i, proc_rank in enumerate(process_ranks_for_fit):
                # time.sleep(0.6)
                if current_status[proc_i]:
                    rec_tag = int(str(proc_rank) + "4545")
                    check_output = comm.irecv(source=proc_rank)

                    if not check_output.get_status():
                        check_output.cancel()
                    else:
                        # first two give some delay for the processor that messes up
                        try:
                            output_info = check_output.wait()
                        except (
                            pickle.UnpicklingError,
                            UnicodeDecodeError,
                            ValueError,
                            OverflowError,
                        ) as e:
                            current_status[proc_i] = False
                            print("BAD error on return")
                            continue
                        if "BAD" in output_info:
                            current_status[proc_i] = False
                            print("BAD", output_info["BAD"])
                            continue
                        # print(output_info)

                        arg_index = output_info["arg"]
                        rank_recv = output_info["rank"]
                        output_list = output_info["output"]

                        gmm_info[arg_index] = output_list
                        gmm_complete[arg_index] = True
                        current_status[proc_i] = False

                        if gmm_complete.sum() + 25 > len(args):
                            print(proc_i, current_status)

                if not current_status[proc_i]:
                    while (
                        current_send_arg_index < len(args_tmp)
                        and gmm_complete[current_send_arg_index]
                    ):
                        current_send_arg_index += 1

                    if current_send_arg_index < len(args_tmp):
                        send_info = {
                            "samples": args_tmp[current_send_arg_index],
                            "arg": current_send_arg_index,
                        }
                        # print("sending", process_ranks_for_fit[index_add])
                        send_tag = int(str(proc_rank) + "67676")
                        comm.send(send_info, dest=proc_rank, tag=send_tag)
                        current_status[proc_i] = True

                        current_send_arg_index += 1

        gmm_info_all.append(gmm_info)

    weights = [tmp[0] for tmp in gmm_info]
    means = [tmp[1] for tmp in gmm_info]
    covs = [tmp[2] for tmp in gmm_info]
    invcovs = [tmp[3] for tmp in gmm_info]
    dets = [tmp[4] for tmp in gmm_info]
    mins = [tmp[5] for tmp in gmm_info]
    maxs = [tmp[6] for tmp in gmm_info]

    output = [weights, means, covs, invcovs, dets, mins, maxs]

    return output


def fit_each_leaf(rank, curr, gather_rank, comm):
    """Worker-side helper that fits one GMM per assigned GB leaf."""

    run_process = True

    while run_process:
        try:
            check = comm.recv(source=gather_rank)
        except (
            pickle.UnpicklingError,
            UnicodeDecodeError,
            ValueError,
            OverflowError,
        ) as e:
            # print("BAD BAD ", rank)
            comm.send({"BAD": "receiving issue"}, dest=gather_rank, tag=send_tag)
            continue

        if isinstance(check, str):
            if check == "end":
                run_process = False
            continue

        assert isinstance(check, dict)

        try:
            arg_index = check["arg"]

            # print("INSIDE", rank, arg_index)
            samples = check["samples"]
        except KeyError:
            comm.send({"BAD": "KeyError"}, dest=gather_rank, tag=send_tag)
            continue

        assert isinstance(samples, np.ndarray)

        gmm = GMMFit(samples)
        output_list = [
            gmm.keep_mix.weights_,
            gmm.keep_mix.means_,
            gmm.keep_mix.covariances_,
            np.array(
                [
                    np.linalg.inv(gmm.keep_mix.covariances_[i])
                    for i in range(len(gmm.keep_mix.weights_))
                ]
            ),
            np.array(
                [
                    np.linalg.det(gmm.keep_mix.covariances_[i])
                    for i in range(len(gmm.keep_mix.weights_))
                ]
            ),
            gmm.sample_mins,
            gmm.sample_maxs,
        ]
        comm.send({"output": output_list, "rank": rank, "arg": arg_index}, dest=gather_rank)
    return


def gb_refit_func(comm, curr, main_rank, class_extra_gpus, class_ranks_list):
    """Worker entry point for ranks dedicated to GMM-based GB refits.

    # TODO/DOCS: full protocol; coordinates the per-leaf GMM refits used to
    refresh ``rj_proposal_distribution`` on the GB RJ moves.
    """
    assert comm is not None

    # get current rank and get index into class_ranks_list
    print(f"INSIDE GB refit, RANK: {comm.Get_rank()}")
    rank = comm.Get_rank()
    rank_index = class_ranks_list.index(rank)
    gather_rank = class_ranks_list[0]
    if rank_index == 0:
        split_remainder = 0  # will fix this setup in the future
        num_search = 2
        gpu = class_extra_gpus[0]
        comm_info = {"process_ranks_for_fit": class_ranks_list[1:]}
        # run search here
        # run_gb_bulk_search(gpu, curr, comm, comm_info, main_rank, num_search, split_remainder)
        pass

    else:
        # run GMM fit here
        fit_each_leaf(rank, curr, gather_rank, comm)
        pass


from dataclasses import dataclass

from eryn.state import Branch
from eryn.utils import TransformContainer


class Buffer(LISAToolsParallelModule):
    """GPU-resident scratch buffers used by the GB special moves.

    Allocates and reuses the per-band working memory (sources currently
    in band, indices into the data array, etc.) so the inner MCMC loop
    avoids reallocating large arrays each iteration.

    # TODO/DOCS: detailed semantics of every buffer field — the body is
    the canonical reference. The most-used members are:

    - ``special_indices_unique`` / ``special_indices_unique_sort``: lookup
      tables that map a per-source ``special_index`` back into the buffer
      ordering.
    - ``params_interest``: parameters of GBs that participate in the move.
    """

    @property
    def xp(self) -> Union[ModuleType, numpy , cupy]:
        """Active array module (NumPy or CuPy) for this buffer."""
        return self.backend.xp

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
        *args,
        **kwargs,
    ):
        self.force_backend = force_backend
        LISAToolsParallelModule.__init__(self, force_backend=force_backend)
        assert self.backend.name.split("_")[-1] == gb.backend.name.split("_")[-1]
        self.gb = gb
        # WDM-domain likelihood object (a gbgpu.gbcomps.GBWDMComputations).
        # Required when ``basis_settings`` is a WDMSettings, ignored otherwise.
        self.gb_wdm_comp = gb_wdm_comp
        self.df = df
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
        self.band_N_vals = self.xp.asarray(band_N_vals)
        # TODO: adjust this
        self.edge_buffer = 2000
        self.is_rj = is_rj

        self.special_indices_unique = special_indices_unique
        self.transform_fn = transform_fn
        self.waveform_kwargs = waveform_kwargs
        self.opt_snr_rej_samp_limit = opt_snr_rej_samp_limit
        self.use_template_arr = use_template_arr
        # load data into buffer for these bands
        # 3 is number of sub-bands to store
        
        self.tdi_channel_setup = self.waveform_kwargs.get("tdi_channel_setup")
        if self.tdi_channel_setup == "XYZ":
            assert self.nchannels == 3
        else:
            assert "A" in self.tdi_channel_setup and "E" in self.tdi_channel_setup
            print("WARNING: using AE(T) channels where we assume ortogonality. This may not be sufficient for realistic orbtis.")

        # Resolve the parent basis-domain settings. Defaults to an FD grid
        # consistent with the legacy Buffer behavior (data_length bins on the
        # parent's df). When invoked via BandSorter.get_buffer, the parent
        # AnalysisContainerArray's settings are forwarded so this Buffer can
        # branch on the actual domain (FD vs WDM).
        if basis_settings is None:
            basis_settings = FDSettings(
                N=self.data_length,
                df=float(self.df) if not hasattr(self.df, "item") else self.df.item(),
            )
        self._basis_settings = basis_settings

        # Build the per-band AnalysisContainerArrays. The actual shape, dtype,
        # and per-band domain depend on basis_settings; see _build_band_aca().
        self._acs_buffer = self._build_band_aca()
        if self.use_template_arr:
            # Templates mirror the band-buffer layout in a second ACA so they
            # share the same managed memory region. The per-band sensitivity
            # slot on the template ACA is unused but keeps construction
            # symmetric across the two buffers.
            self._acs_template_buffer = self._build_band_aca()

        # psd_shape is exposed for back-compat with downstream consumers that
        # inspect it; it tracks the shape of the per-band PSD view.
        self.psd_shape = (self.num_bands_now,) + self._per_band_sens_shape

        # Build the domain-aware likelihood engine. Dispatch is on
        # ``isinstance(basis_settings, ...)`` -- no string-level mode flag.
        # The engine takes an AnalysisContainerArray at call time, so the
        # Buffer's get_swap_ll / get_ll / adjust_sources_in_band_buffer
        # methods don't reach into self.gb (or self.gb_wdm_comp) directly.
        self._likelihood_engine = make_band_likelihood_engine(
            self._basis_settings,
            gb=self.gb,
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
        """Internal :class:`AnalysisContainerArray` backing the per-band residual buffers."""
        return self._acs_buffer

    # ------------------------------------------------------------------
    # Per-band buffer accessors
    # ------------------------------------------------------------------
    # In multi-GPU mode the buffer ACA holds the bands sharded across
    # GPUs (striped by default; see ``_build_fd_band_aca``). The shaped
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
        return self._flat_or_raise(self._acs_buffer, "data")

    @property
    def band_buffer(self):
        """Per-band residual buffer indexable by global band id.

        Single-GPU: returns the ``(num_bands_now, nchannels, data_length)``
        reshape directly. Multi-GPU: returns a :class:`BandView` that
        routes per-band reads/writes through the owning shard.
        """
        return self._shaped_or_view(self._acs_buffer, "data")

    @property
    def psd_buffer_tmp(self):
        """Flat per-GPU inverse-PSD buffer (1D view; single-GPU only)."""
        return self._flat_or_raise(self._acs_buffer, "psd")

    @property
    def psd_buffer(self):
        """Per-band inverse-PSD buffer indexable by global band id.

        Same single-GPU / multi-GPU behaviour as :attr:`band_buffer`.
        """
        return self._shaped_or_view(self._acs_buffer, "psd")

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
            return (self.nchannels, self.data_length)
        elif isinstance(self._basis_settings, WDMSettings):
            # First-cut: each per-band buffer covers the FULL WDM active grid
            # (Nf_active layers x Nt_active time pixels). The lisa-on-gpu WDM
            # kernel currently uses a single global [ind_min_f, ind_max_f]
            # rather than per-band offsets, so per-band slicing on the layer
            # axis is a follow-on once the kernel takes per-band layer
            # offsets. data_length is unused on the WDM path.
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
                return (self.nchannels, self.nchannels, self.data_length)
            return (self.nchannels, self.data_length)
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
        """Element dtype for the per-band inverse-PSD buffer."""
        if isinstance(self._basis_settings, FDSettings):
            return self.xp.complex128
        elif isinstance(self._basis_settings, WDMSettings):
            return self.xp.float64
        else:
            raise NotImplementedError(
                f"Buffer does not support basis domain {type(self._basis_settings).__name__}."
            )

    def _build_per_band_basis_settings(self) -> DomainSettingsBase:
        """Construct the per-band domain settings used by each per-band AC.

        Each per-band AC's :class:`DataResidualArray` needs a domain-settings
        object whose ``basis_shape_active`` matches the per-band data shape.
        For FD this is a fresh FDSettings sized to ``data_length``. For WDM
        the per-band geometry depends on the GB WDM waveform spec and is not
        yet implemented; see :meth:`_build_wdm_band_aca`.
        """
        if isinstance(self._basis_settings, FDSettings):
            return FDSettings(
                N=self.data_length,
                df=float(self.df) if not hasattr(self.df, "item") else self.df.item(),
                force_backend=self._basis_settings.backend_name.split("_", 1)[1],
            )
        elif isinstance(self._basis_settings, WDMSettings):
            # First-cut: per-band WDMSettings matches the parent grid (full
            # WDM active band). A true per-band sliced WDMSettings becomes
            # possible once the lisa-on-gpu WDM kernel takes per-band
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

    def _build_band_aca(self) -> AnalysisContainerArray:
        """Allocate one :class:`AnalysisContainer` per band, wrapped in an ACA.

        Branches on the parent basis domain. The FD path is the active code
        path used by the GB special moves today. The WDM path is intentionally
        left as a NotImplementedError so that the failure surfaces at
        construction time once a WDM basis is supplied — the GB WDM template
        generator must land first.
        """
        if isinstance(self._basis_settings, FDSettings):
            return self._build_fd_band_aca()
        elif isinstance(self._basis_settings, WDMSettings):
            return self._build_wdm_band_aca()
        else:
            raise NotImplementedError(
                f"Buffer does not support basis domain {type(self._basis_settings).__name__}."
            )

    def _build_fd_band_aca(self) -> AnalysisContainerArray:
        """FD path: one AC per band, each holding an FD residual buffer of
        shape ``(nchannels, data_length)`` and a complex inverse-PSD."""
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
        # safe. The per-band Buffer accessors (band_buffer / psd_buffer /
        # template_buffer) automatically fall back to a single ndarray view
        # for single-GPU runs and return a BandView (multi-shard router)
        # otherwise -- see the accessor block in the Buffer class.
        gpu_assignment = (
            band_gpu_assignment(len(ac_list), list(gpus_in)) if gpus_in else None
        )
        return AnalysisContainerArray(
            ac_list,
            gpus=list(gpus_in) if gpus_in else None,
            complex_psd=True,
            gpu_assignment=gpu_assignment,
        )

    def _build_wdm_band_aca(self) -> AnalysisContainerArray:
        """WDM path: one AC per band, each holding a real-valued WDM buffer.

        Mirrors :meth:`_build_fd_band_aca`. Per-band data shape is
        ``(nchannels, Nf_active, Nt_active)``; per-band PSD shape is the same
        (or ``(nchannels, nchannels, Nf_active, Nt_active)`` for XYZ, which
        carries the full inverse covariance the WDM kernel's
        ``get_pixel_noise_value_cross_channel`` consumes).

        First-cut: each per-band buffer covers the full WDM active grid. The
        lisa-on-gpu WDM kernel currently uses a single ``[ind_min_f, ind_max_f]``
        from the WDM lookup table rather than per-band offsets, so a true
        per-band layer slicing optimisation is a follow-on once that kernel
        takes per-band offsets.
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
        # Same multi-GPU semantics as _build_fd_band_aca; see the docstring
        # there. BandView wraps the multi-shard reshape views so per-band
        # reads/writes by global band id keep working transparently.
        gpu_assignment = (
            band_gpu_assignment(len(ac_list), list(gpus_in)) if gpus_in else None
        )
        return AnalysisContainerArray(
            ac_list,
            gpus=list(gpus_in) if gpus_in else None,
            complex_psd=False,
            gpu_assignment=gpu_assignment,
        )

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
            assert tmp_buffer_start_index + self.data_length >= (
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
            # self.buffer_start_index[self.unique_band_combos[:, 2] == self.num_bands - 1] = (self.band_edges[-1] / self.df).astype(np.int32) - self.edge_buffer

        self.start_freq_inds = self.xp.asarray(self.buffer_start_index.copy().astype(np.int32))

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

        if self.tdi_channel_setup == "XYZ":
            # using einstein summation: b=bands, i=channel 1, j=channel 2, k=frequency
            source_term = (
                - (1.0 / 2.0) * 4.0 * self.df
                * cp.einsum(
                    "bik,bijk,bjk->b", numerator_in.conj(), psd_buffer, numerator_in
                ).real
            )

            if noise_only:
                raise NotImplementedError("Noise-only likelihood requires log=determinant over frequency for XYZ CSD.")

        else:
            source_term = (
                - (1.0 / 2.0) * 4.0 * self.df
                * cp.sum((numerator_in.conj() * numerator_in) * psd_buffer, axis=(1, 2)).real
            )

            if noise_only:
                return -cp.sum(cp.log(cp.abs(1 / psd_buffer[psd_buffer != 0.0])))

        if source_only:
            return source_term

        # Diagonal noise_term fall_back # TODO check if this is sufficient not used currently anyway
        psd_term = -cp.sum(cp.log(cp.abs(psd_buffer[psd_buffer != 0.0])))
        if self.tdi_channel_setup == "XYZ":
            warnings.warn("The current psd ll calculation is not correct for XYZ CSD channel setup.")

        # cp.get_default_memory_pool().free_all_blocks()

        return source_term + psd_term
    

    def get_swap_ll(self, params_remove, params_add, data_index, N_vals, phase_maximize=False):
        """Per-proposal swap log-likelihood difference.

        Domain-agnostic: dispatches to ``self._likelihood_engine.get_swap_ll``,
        which is either :class:`FDBandLikelihoodEngine` or
        :class:`WDMBandLikelihoodEngine` depending on the Buffer's
        ``basis_settings``. Both engines take the per-band ACA
        (:attr:`acs_buffer`) and the physical params, and return a
        :class:`SwapLLResult`. The rejection-sampling clamp and the
        phase-maximisation correction live here so the engine stays a thin
        wrapper around the kernel.
        """
        params_remove_phys = self.transform_fn.both_transforms(params_remove, xp=cp)
        params_add_phys = self.transform_fn.both_transforms(params_add, xp=cp)

        result = self._likelihood_engine.get_swap_ll(
            self.acs_buffer,
            params_remove_phys,
            params_add_phys,
            data_index=data_index,
            noise_index=data_index,
            N_vals=N_vals,
            phase_marginalize=phase_maximize,
            waveform_kwargs=self.waveform_kwargs,
        )

        ll_diff = result.ll_diff
        kept = result.kept

        if np.any(~kept):
            print(f"NOT KEEPING: {(~kept).sum()}")

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

    def get_ll(self, params, data_index, noise_index, N_vals):
        """Per-source log-likelihood = -0.5 * (h_h - 2 d_h).

        Domain-agnostic dispatch like :meth:`get_swap_ll`. Returns the
        ``(d_h, h_h)`` inner products on the engine's xp module so callers
        can compute their preferred likelihood form.
        """
        params_phys = self.transform_fn.both_transforms(params, xp=cp)
        return self._likelihood_engine.get_ll(
            self.acs_buffer,
            params_phys,
            data_index=data_index,
            noise_index=noise_index,
            N_vals=N_vals,
            waveform_kwargs=self.waveform_kwargs,
        )

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
        on the ``GBWDMComputations`` instance passed in at Buffer
        construction time via ``gb_wdm_comp``. Per the sprint-wide
        rule there is no runtime ``backend=`` kwarg; build a JAX-
        backed ``gb_wdm_comp`` if you need the autograd path.
        """
        params_phys = self.transform_fn.both_transforms(params, xp=cp)
        return self._likelihood_engine.get_ll_grad(
            self.acs_buffer,
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
            self.acs_buffer,
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

    # def fill_buffer_residual_from_acs(self, acs):
    #     inds_get = self._get_fill_buffer_ind_map(acs)
    #     self.reset_residual_buffers()
    #     self.band_buffer[:self.num_bands_now] += rest_of_data[:]

    # def fill_buffer_psd_from_acs(self, acs):
    #     inds_get = self._get_fill_buffer_ind_map(acs)
    #     self.reset_psd_buffers()
    #     self.psd_buffer[:self.num_bands_now] = acs.psd_shaped[0][inds_get].reshape((self.num_bands_now,) + self.band_buffer.shape[1:])

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

        self.psd_buffer[inds_fill] = outer_psd_view[inds_get_psd]
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

        try:
            assert np.all((self.buffer_start_index[inds_fill] - start_freq_ind) >= 0)
        except AssertionError:
            breakpoint()

        assert np.all(
            (self.buffer_start_index[inds_fill] - start_freq_ind + self.data_length)
            <= acs.data_length
        )

        start_inds = self.buffer_start_index[inds_fill] - start_freq_ind

        if is_psd and self.tdi_channel_setup == "XYZ":
            # Target output shape: (len(inds_fill), self.nchannels, self.nchannels, self.band_buffer.shape[-1])
            inds1 = (
                self.unique_band_combos[inds_fill, 1][:, None, None, None]
                * self.nchannels
                + cp.arange(self.nchannels)[None, :, None, None]
            )
            inds2 = cp.arange(self.nchannels)[None, None, :, None]
            inds3 = start_inds[:, None, None, None] + cp.arange(self.band_buffer.shape[-1])[None, None, None, :]

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

    def _adjust_via_engine(
        self, factor, target_aca, params, params_index, N_vals, *args, **kwargs
    ) -> None:
        """Domain-agnostic dispatch into ``self._likelihood_engine.fill_template``.

        ``factor`` is +1 (write source into the template) or -1 (subtract it).
        ``target_aca`` selects which AnalysisContainerArray to write into
        (band-residual ACA or template ACA). Both share the same per-band
        geometry, so the engine doesn't need to know which one it's filling.
        """
        assert isinstance(factor, int) and (factor == -1 or factor == +1)
        params_phys = self.transform_fn.both_transforms(params, xp=cp)
        try:
            self._likelihood_engine.fill_template(
                target_aca,
                params_phys,
                params_index,
                N_vals,
                factor=factor,
                waveform_kwargs=self.waveform_kwargs,
            )
        except AssertionError:
            breakpoint()

    def adjust_sources_in_band_buffer(
        self, factor, input_array, params, params_index, N_vals, *args, **kwargs
    ) -> None:
        """Backwards-compatible shim around :meth:`_adjust_via_engine`.

        Routes ``input_array`` (a flat buffer pointer the legacy code passed
        through) back to whichever ACA owns it. New code should call
        :meth:`_adjust_via_engine` directly.
        """
        if input_array is self.band_buffer_tmp:
            target_aca = self._acs_buffer
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
        self._adjust_via_engine(+1, self._acs_buffer, *args, **kwargs)

    def add_sources_to_band_buffer(self, *args, **kwargs) -> None:
        # See remove_sources_from_band_buffer note; sign is flipped for the
        # residual-tracking band_buffer.
        self._adjust_via_engine(-1, self._acs_buffer, *args, **kwargs)

    def get_special_band_index(
        self, temp_inds: np.ndarray, walker_inds: np.ndarray, band_inds: np.ndarray
    ) -> np.ndarray:
        special_indices = (temp_inds * self.nwalkers + walker_inds) * int(1e6) + band_inds
        return special_indices

    def get_separate_inds_from_special_index(self, special_band_inds: np.ndarray) -> tuple:
        temp_walker_inds_now = cp.floor(special_band_inds / 1e6).astype(int)
        temp_inds_now = temp_walker_inds_now // self.nwalkers
        walker_inds_now = temp_walker_inds_now % self.nwalkers
        band_inds_now = (special_band_inds - temp_walker_inds_now * int(1e6)).astype(int)
        return (temp_inds_now, walker_inds_now, band_inds_now)


def return_x(x):
    """Identity helper used as a no-op replacement for :func:`copy.deepcopy`."""
    return x


class BandSorter(LISAToolsParallelModule):
    """GPU helper that sorts/ungroups GB samples by frequency band.

    # TODO/DOCS: detailed semantics. Used by :class:`GBSpecialBase` to keep
    track of which sources fall into which band and to map data-array
    indices accordingly so band-temperature swaps and per-band proposals
    operate on the correct subset.
    """

    @property
    def xp(self) -> Union[ModuleType, numpy , cupy]:
        return self.backend.xp

    @classmethod
    def supported_backends(cls):
        return ["lisatools_" + _tmp for _tmp in cls.GPU_RECOMMENDED()]

    def __init__(
        self,
        gb_branch: Branch,
        band_edges: Optional[np.ndarray] = None,
        band_N_vals: Optional[np.ndarray] = None,
        force_backend: bool = None,
        transform_fn: Optional[TransformContainer] = None,
        copy: bool = True,
        inds_subset: Optional[np.ndarray] = None,
        inds_main_band_sorter: Optional[np.ndarray] = None,
        gb=None,
        gb_wdm_comp=None,
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
                        if len(inds_subset) == 0:
                            breakpoint()
                        set_value = dc(value)

                    setattr(self, key, set_value)

            self.rj_prop = _band_sorter.rj_prop
            self.gb = _band_sorter.gb
            # Forward the WDM computation object explicitly (skipped in the
            # copy loop so we don't deepcopy a GPU-resident object).
            self.gb_wdm_comp = getattr(_band_sorter, "gb_wdm_comp", None)
            # need to make sure is not mixed up in loop
            self.set_main_band_sorter_info(main_band_sorter, inds_main_band_sorter)
            return

        assert band_edges is not None and band_N_vals is not None
        self.force_backend = force_backend
        self.gb = gb
        # Optional WDM-domain likelihood object. Forwarded to Buffer in
        # :meth:`get_buffer` so the engine selection lands on
        # :class:`WDMBandLikelihoodEngine` when the parent ACA carries a
        # WDMSettings basis. Ignored on the FD path.
        self.gb_wdm_comp = gb_wdm_comp
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

            # if self.name == "rj_prior":
            # proposal_logpdf = self.rj_proposal_distribution["gb"].logpdf(
            #     points_curr[gb_inds]
            # )
            # else:
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
            # self.factors[self.coords[:, 1] / 1e3 < self.band_edges[0]] = -np.inf

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
        special_indices = (temp_inds * self.nwalkers + walker_inds) * int(1e6) + band_inds
        return special_indices

    def get_separate_inds_from_special_index(self, special_band_inds: np.ndarray) -> tuple:
        temp_walker_inds_now = cp.floor(special_band_inds / 1e6).astype(int)
        temp_inds_now = temp_walker_inds_now // self.nwalkers
        walker_inds_now = temp_walker_inds_now % self.nwalkers
        band_inds_now = (special_band_inds - temp_walker_inds_now * int(1e6)).astype(int)
        return (temp_inds_now, walker_inds_now, band_inds_now)

    @property
    def special_index_check(self) -> bool:
        return self.xp.all(
            self.special_band_inds
            == self.get_special_band_index(self.temp_inds, self.walker_inds, self.band_inds)
        )

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

            # TODO: what to do about this
            # inds_keep &= (self.band_inds < len(self.band_edges) - 2)
            # inds_keep &= (self.band_inds > 1)

            if temp is not None:
                assert isinstance(temp, int)
                inds_keep &= self.temp_inds == temp
            if walker is not None:
                assert isinstance(walker, int)
                inds_keep &= self.walker_inds == walker
            if leaf is not None:
                assert isinstance(temp, int)
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
    ) -> Buffer:

        num_band_preload = len(special_indices_unique)

        # CAN USE main_band_sorter TO GET SOURCES IN BANDS OF INTEREST THAT ARE NOT CURRENTLY OF INTEREST THEMSELVES

        # TODO: check the end of this line, is this covered ??
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
            buffer_obj = Buffer(
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
                acs.df,
                sources_now_map,
                sources_inject_now_map,
                self.main_band_sorter.special_band_inds[sources_now_map],
                basis_settings=acs.settings,
                gb_wdm_comp=self.gb_wdm_comp,
                force_backend=self.force_backend,
                **kwargs,
            )

        else:
            assert isinstance(buffer_obj, Buffer)
            assert inds_fill.max() <= buffer_obj.num_bands_now
            # THIS NEEDS TO HAPPEN before updating data
            buffer_obj.update_special_indices(special_indices_unique, inds_fill=inds_fill)

        buffer_obj.fill_buffer_residual_and_psd_from_acs(acs, inds_fill=inds_fill)
        buffer_obj.acs = acs
        # includes sources in these sub-bands that are no longer getting proposals
        coords_to_inject = self.main_band_sorter.coords[sources_inject_now_map].copy()
        inj_special_indices_now = self.main_band_sorter.special_band_inds[
            sources_inject_now_map
        ].copy()

        inject_index = buffer_obj.get_index(inj_special_indices_now)
        inject_N_vals = self.band_N_vals[
            self.main_band_sorter.band_inds[sources_inject_now_map]
        ].copy()

        if len(inject_index) != len(coords_to_inject):
            breakpoint()

        inj_args = (coords_to_inject, inject_index, inject_N_vals)
        if buffer_obj.use_template_arr:
            buffer_obj.add_sources_to_template_buffer(*inj_args)
        else:
            buffer_obj.add_sources_to_band_buffer(*inj_args)

        return buffer_obj

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


# MHMove needs to be to the left here to overwrite GBBruteRejectionRJ RJ proposal method
class GBSpecialBase(GlobalFitMove, GroupStretchMove, Move, LISAToolsParallelModule):
    """Base class for GB-specific stretch / reversible-jump moves.

    Combines :class:`GlobalFitMove`, :class:`eryn.moves.GroupStretchMove`,
    :class:`Move`, and :class:`LISAToolsParallelModule` so each GB move can
    use try-force rejection, optional phase maximization, and GPU-resident
    band-aware buffers (:class:`Buffer`, :class:`BandSorter`).

    # TODO/DOCS: full argument list — many constructor kwargs are passed
    through unmodified to ``GroupStretchMove``. Intended use is via the
    concrete subclasses :class:`GBSpecialStretchMove`,
    :class:`GBSpecialRJPriorMove`, :class:`GBSpecialRJSearchMove`,
    :class:`GBSpecialRJSerialSearchMCMC`, and
    :class:`GBSpecialRJRefitMove`.

    Args:
        gb: :class:`gbgpu.GBGPU` instance.
        priors: :class:`ProbDistContainer` for in-model GB parameters.
        start_freq_ind: Inclusive starting index into the global ``f_arr``
            (FD path); ignored on the WDM path.
        data_length: Length of the data array per channel (FD path); the
            WDM path sizes its per-band buffers from
            ``WDMSettings.Nf_active`` / ``Nt_active`` instead.
        mgh: Shared :class:`AnalysisContainerArray`. The basis-domain
            settings on ``mgh.settings`` drive every domain-dependent
            choice (FD vs WDM); ``df`` / ``f_arr`` are derived from it
            rather than being passed separately.
        band_edges: Frequency-band edges.
        band_N_vals: Per-band waveform sample counts.
        gpu_priors: Branch-keyed GPU-resident priors.
        waveform_kwargs: Forwarded to ``gb`` waveform calls.
        parameter_transforms: :class:`TransformContainer` for GBs.
        snr_lim: Optional SNR cut.
        rj_proposal_distribution: Distribution used to draw RJ proposals.
        is_rj_prop: Marks this move as a reversible-jump proposal.
        num_repeat_proposals: Inner repeat count per call.
        name: Move name (used for logging and bookkeeping).
        use_prior_removal: If ``True``, draw RJ proposals from the prior.
        phase_maximize: If ``True``, marginalize over phase in the
            likelihood.
        ranks_needed / gpus: MPI / GPU resource requests.
        num_band_preload: Number of bands preloaded per call.
        run_swaps: Whether to run band-temperature swaps.
        max_data_store_size: Cap on the per-iteration data store size.
        force_backend: Optional backend override.
        gb_wdm_comp: Optional :class:`gbgpu.gbcomps.GBWDMComputations`
            instance. Required when ``mgh.settings`` is a
            :class:`~lisatools.domains.WDMSettings`; ignored otherwise.
    """

    @property
    def xp(self) -> Union[ModuleType, numpy , cupy]:
        """Active array module (NumPy or CuPy) for this move."""
        return self.backend.xp

    def __init__(
        self,
        gb: GBGPU,
        priors,
        start_freq_ind,
        data_length,
        mgh,
        band_edges,
        band_N_vals,
        gpu_priors,
        *args,
        waveform_kwargs={},
        parameter_transforms=None,
        snr_lim=1e-10,
        rj_proposal_distribution=None,
        is_rj_prop=False,
        num_repeat_proposals=1,
        name=None,
        use_prior_removal=False,
        phase_maximize=False,
        ranks_needed=0,
        gpus=[],
        num_band_preload=20000,
        run_swaps=True,
        # TODO: make this adjustable?
        max_data_store_size=6000,
        force_backend=None,
        gb_wdm_comp=None,
        **kwargs,
    ):
        # return_gpu is a kwarg for the stretch move
        LISAToolsParallelModule.__init__(self, force_backend=force_backend)
        GlobalFitMove.__init__(self, name=name)
        GroupStretchMove.__init__(self, *args, return_gpu=True, **kwargs)

        self.force_backend = force_backend
        self.ranks_needed = ranks_needed
        self.gpus = gpus
        self.gpu_priors = gpu_priors
        self.num_repeat_proposals = num_repeat_proposals
        self.num_band_preload = num_band_preload
        self.band_preload_size = self.max_data_store_size = max_data_store_size
        self.use_prior_removal = use_prior_removal
        self.has_setup_group = False

        # for key in priors:
        #     if not isinstance(priors[key], ProbDistContainer) and not isinstance(priors[key], GBPriorWrap):
        #         raise ValueError(
        #             "Priors need to be eryn.priors.ProbDistContainer object."
        #         )

        self.priors = priors
        self.gb = gb
        # Optional WDM-domain likelihood object. Constructed once by the
        # user (typically a gbgpu.gbcomps.GBWDMComputations) and threaded
        # through to BandSorter -> Buffer -> WDMBandLikelihoodEngine when
        # the analysis container's DomainSettingsBase is a WDMSettings.
        # Stays None on the FD path; the FD engine path doesn't touch it.
        self.gb_wdm_comp = gb_wdm_comp
        self.stop_here = True
        self.run_swaps = run_swaps

        # args = [priors["gb"].priors_in[(0, 1)].rho_star]
        # args += [priors["gb"].priors_in[(0, 1)].frequency_prior.min_val, priors["gb"].priors_in[(0, 1)].frequency_prior.max_val]
        # for i in range(2, 8):
        #     args += [priors["gb"].priors_in[i].min_val, priors["gb"].priors_in[i].max_val]

        # self.gpu_cuda_priors = self.gb.pyPriorPackage(*tuple(args))
        # self.gpu_cuda_wrap = self.gb.pyPeriodicPackage(2 * np.pi, np.pi, 2 * np.pi)

        # use gpu from template generator
        # self.force_backend = gb.force_backend
        if self.backend.uses_cupy:
            self.mempool = self.xp.get_default_memory_pool()
        else:
            self.mempool = _NoOpMempool()

        self.band_edges = band_edges
        self.num_bands = len(band_edges) - 1
        self.start_freq_ind = start_freq_ind
        self.data_length = data_length
        self.waveform_kwargs = waveform_kwargs
        self.parameter_transforms = parameter_transforms
        self.mgh = mgh

        # Derive ``self.fd`` and ``self.df`` from the parent
        # AnalysisContainerArray's basis settings. The band-index math in
        # this move uses ``df = 1 / Tobs`` consistently across FD and WDM
        # (FD: equals ``acs.df``; WDM: ``acs.df == layer_df`` differs, so
        # we recompute). ``self.fd`` is only meaningful in the FD path.
        if isinstance(mgh.settings, FDSettings):
            self.fd = mgh.f_arr.copy()
            self.df = float(self.fd[1] - self.fd[0])
        elif isinstance(mgh.settings, WDMSettings):
            self.fd = None
            self.df = 1.0 / mgh.settings.Tobs
        else:
            raise NotImplementedError(
                f"GBSpecialBase does not support basis domain "
                f"{type(mgh.settings).__name__}."
            )
        self.phase_maximize = phase_maximize

        self.snr_lim = snr_lim

        self.band_edges = self.xp.asarray(self.band_edges)

        self.rj_proposal_distribution = rj_proposal_distribution
        self.is_rj_prop = is_rj_prop or (self.rj_proposal_distribution is not None)

        # if self.is_rj_prop:
        #     if (self.num_repeat_proposals != 1):
        #         print("Adjusting repeat proposals to 1 for RJ.")

        #     self.num_repeat_proposals = 1

        # setup N vals for bands
        self.band_N_vals = self.xp.asarray(band_N_vals)
        
        self.num_proposals = 0
        

    def setup(self, model, branches):
        return

    @classmethod
    def supported_backends(cls):
        return ["lisatools_" + _tmp for _tmp in cls.GPU_RECOMMENDED()]

    def setup_gb_friends(self, band_sorter):
        st = time.perf_counter()
        coords = band_sorter.coords
        inds = band_sorter.inds
        temp_index = band_sorter.temp_inds

        # supps = branch.branch_supplemental
        ntemps = self.ntemps
        nwalkers = self.nwalkers
        all_remaining_freqs = coords[inds & (temp_index == 0)][:, 1]

        all_remaining_cords = coords[inds & (temp_index == 0)]

        num_remaining = len(all_remaining_freqs)

        all_temp_fs = self.xp.asarray(coords[inds][:, 1])

        # TODO: improve this?
        self.inds_freqs_sorted = self.xp.asarray(np.argsort(all_remaining_freqs))
        self.freqs_sorted = self.xp.asarray(np.sort(all_remaining_freqs))
        self.all_coords_sorted = self.xp.asarray(all_remaining_cords)[self.inds_freqs_sorted]

        left_inds, right_inds = self.find_friends_init(all_temp_fs)

        start_inds = asnumpy(left_inds.copy())

        start_inds_all = -np.ones_like(inds, dtype=np.int32)
        start_inds_all[inds] = start_inds.astype(np.int32)

        band_sorter.friend_start_inds = start_inds_all

        # if "friend_start_inds" not in supps:
        #     supps.add_objects({"friend_start_inds": start_inds_all})
        # else:
        #     supps[:] = {"friend_start_inds": start_inds_all}

        et = time.perf_counter()
        self.mempool.free_all_blocks()

        self.has_setup_group = True
        # print("SETUP:", et - st)
        # start_inds_freq_out = np.zeros((ntemps, nwalkers, nleaves_max), dtype=int)
        # freqs_sorted_here = self.freqs_sorted.get()
        # freqs_remaining_here = all_remaining_freqs

        # start_ind_best = np.zeros_like(freqs_remaining_here, dtype=int)

        # best_index = (
        #     np.searchsorted(freqs_sorted_here, freqs_remaining_here, side="right") - 1
        # )
        # best_index[best_index < self.nfriends] = self.nfriends
        # best_index[best_index >= len(freqs_sorted_here) - self.nfriends] = (
        #     len(freqs_sorted_here) - self.nfriends
        # )
        # check_inds = (
        #     best_index[:, None]
        #     + np.tile(np.arange(2 * self.nfriends), (best_index.shape[0], 1))
        #     - self.nfriends
        # )

        # check_freqs = freqs_sorted_here[check_inds]
        # breakpoint()

        # # batch_count = 1000
        # # split_inds = np.arange(batch_count, freqs_remaining_here.shape[0], batch_count)

        # # splits_remain = np.split(freqs_remaining_here, split_inds)
        # # splits_check = np.split(check_freqs, split_inds)

        # # out = []
        # # for i, (split_r, split_c) in enumerate(zip(splits_remain, splits_check)):
        # #     out.append(np.abs(split_r[:, None] - split_c))
        # #     print(i)

        # # freq_distance = np.asarray(out)

        # freq_distance = np.abs(freqs_remaining_here[:, None] - check_freqs)
        # breakpoint()

        # keep_min_inds = np.argsort(freq_distance, axis=-1)[:, : self.nfriends].min(
        #     axis=-1
        # )
        # start_inds_freq = check_inds[(np.arange(len(check_inds)), keep_min_inds)]

        # start_inds_freq_out[inds] = start_inds_freq

        # start_inds_freq_out[~inds] = -1

        # if "friend_start_inds" not in supps:
        #     supps.add_objects({"friend_start_inds": start_inds_freq_out})
        # else:
        #     supps[:] = {"friend_start_inds": start_inds_freq_out}

        # self.all_friends_start_inds_sorted = self.xp.asarray(
        #     start_inds_freq_out[inds][self.inds_freqs_sorted.get()]
        # )

    def find_friends_init(self, all_temp_fs):

        total_binaries = all_temp_fs.shape[0]
        still_going = cp.ones(total_binaries, dtype=bool)
        inds_zero = cp.searchsorted(self.freqs_sorted, all_temp_fs, side="right") - 1
        left_inds = inds_zero - int(self.nfriends / 2)
        right_inds = inds_zero + int(self.nfriends / 2) - 1

        # do right first here
        right_inds[left_inds < 0] = self.nfriends - 1
        left_inds[left_inds < 0] = 0

        # do left first here
        left_inds[right_inds > len(self.freqs_sorted) - 1] = len(self.freqs_sorted) - self.nfriends
        right_inds[right_inds > len(self.freqs_sorted) - 1] = len(self.freqs_sorted) - 1

        assert np.all(right_inds - left_inds == self.nfriends - 1)

        assert (
            not np.any(right_inds < 0)
            and not np.any(right_inds > len(self.freqs_sorted) - 1)
            and not np.any(left_inds < 0)
            and not np.any(left_inds > len(self.freqs_sorted) - 1)
        )

        jjj = 0
        while np.any(still_going):
            distance_left = np.abs(
                all_temp_fs[still_going] - self.freqs_sorted[left_inds[still_going]]
            )
            distance_right = np.abs(
                all_temp_fs[still_going] - self.freqs_sorted[right_inds[still_going]]
            )

            check_move_right = distance_right <= distance_left
            check_left_inds = left_inds[still_going][check_move_right] + 1
            check_right_inds = right_inds[still_going][check_move_right] + 1

            new_distance_right = np.abs(
                all_temp_fs[still_going][check_move_right] - self.freqs_sorted[check_right_inds]
            )

            change_inds = cp.arange(len(all_temp_fs))[still_going][check_move_right][
                (new_distance_right < distance_left[check_move_right])
                & (check_right_inds < len(self.freqs_sorted))
            ]

            left_inds[change_inds] += 1
            right_inds[change_inds] += 1

            stop_inds_right_1 = cp.arange(len(all_temp_fs))[still_going][check_move_right][
                (check_right_inds >= len(self.freqs_sorted))
            ]

            # last part is just for up here, below it will remove if it is still equal
            stop_inds_right_2 = cp.arange(len(all_temp_fs))[still_going][check_move_right][
                (new_distance_right >= distance_left[check_move_right])
                & (check_right_inds < len(self.freqs_sorted))
                & (distance_right[check_move_right] != distance_left[check_move_right])
            ]
            stop_inds_right = cp.concatenate([stop_inds_right_1, stop_inds_right_2])
            assert np.all(still_going[stop_inds_right])

            # equal to should only be left over if it was equal above and moving right did not help
            check_move_left = distance_left <= distance_right
            check_left_inds = left_inds[still_going][check_move_left] - 1
            check_right_inds = right_inds[still_going][check_move_left] - 1

            new_distance_left = np.abs(
                all_temp_fs[still_going][check_move_left] - self.freqs_sorted[check_left_inds]
            )

            change_inds = cp.arange(len(all_temp_fs))[still_going][check_move_left][
                (new_distance_left < distance_right[check_move_left]) & (check_left_inds >= 0)
            ]

            left_inds[change_inds] -= 1
            right_inds[change_inds] -= 1

            stop_inds_left_1 = cp.arange(len(all_temp_fs))[still_going][check_move_left][
                (check_left_inds < 0)
            ]
            stop_inds_left_2 = cp.arange(len(all_temp_fs))[still_going][check_move_left][
                (new_distance_left >= distance_right[check_move_left]) & (check_left_inds >= 0)
            ]
            stop_inds_left = cp.concatenate([stop_inds_left_1, stop_inds_left_2])

            stop_inds = cp.concatenate([stop_inds_right, stop_inds_left])
            still_going[stop_inds] = False
            # print(jjj, still_going.sum())
            if jjj >= self.nfriends:
                breakpoint()
            jjj += 1

        return left_inds, right_inds

    def fix_friends(self, band_sorter, new_inds):

        assert self.xp.all(band_sorter.inds[new_inds])
        all_temp_fs = self.xp.asarray(band_sorter.coords[new_inds][:, 1])

        self.find_friends_init(all_temp_fs)

        start_inds = asnumpy(left_inds.copy())
        # TODO: remove .get()?
        band_sorter.friend_start_inds[new_inds] = start_inds_all

    def find_friends(self, name, gb_points_to_move, s_inds=None, branch_supps=None):
        if s_inds is None:  #  or branch_supps is None:
            raise ValueError

        inds_points_to_move = self.xp.asarray(s_inds.flatten())

        half_friends = int(self.nfriends / 2)

        gb_points_for_move = gb_points_to_move.reshape(-1, 8).copy()

        if not hasattr(self, "ntemps"):
            self.ntemps = 1

        # TODO: update how this is done
        inds_start_freq_to_move = (
            self.friend_start_inds_now
        )  # self.xp.asarray(branch_supps[:]["friend_start_inds"].flatten())
        assert inds_points_to_move.sum().item() == inds_start_freq_to_move.shape[0]

        deviation = self.xp.random.randint(0, self.nfriends, size=len(inds_start_freq_to_move))

        inds_keep_friends = inds_start_freq_to_move + deviation

        inds_keep_friends[inds_keep_friends < 0] = 0
        inds_keep_friends[inds_keep_friends >= len(self.all_coords_sorted)] = (
            len(self.all_coords_sorted) - 1
        )

        gb_points_for_move[inds_points_to_move] = self.all_coords_sorted[inds_keep_friends]
        return gb_points_for_move[None, :, None, :]

    def new_find_friends(self, name, inds_in):
        inds_start_freq_to_move = self.current_friends_start_inds[tuple(inds_in)]

        deviation = self.xp.random.randint(0, self.nfriends, size=len(inds_start_freq_to_move))

        inds_keep_friends = inds_start_freq_to_move + deviation

        inds_keep_friends[inds_keep_friends < 0] = 0
        inds_keep_friends[inds_keep_friends >= len(self.all_coords_sorted)] = (
            len(self.all_coords_sorted) - 1
        )

        gb_points_for_move = self.all_coords_sorted[inds_keep_friends]

        return gb_points_for_move

    def adjust_sources_in_residual_buffer(
        self, factor, model, band_sorter: BandSorter, *args, **kwargs
    ) -> None:
        """Add or remove sources from the main residual buffer.

        Domain-agnostic: uses ``self._likelihood_engine.fill_template`` to
        work with both FD and WDM domains. The ``factor`` controls whether
        sources are added (``-1``, reducing the residual) or removed (``+1``,
        restoring them to the residual).
        """
        assert isinstance(factor, int) and (factor == -1 or factor == +1)

        subset = band_sorter.get_subset(*args, **kwargs)

        if subset is None or subset.inds.sum().item() == 0:
            return

        params_in = subset.coords_in[subset.inds]
        walkers_in = subset.walker_inds[subset.inds].astype(self.xp.int32)
        N_vals_in = subset.N_vals[subset.inds]

        # FD-specific bounds checks (only meaningful when basis is FD).
        # WDM bounds are checked internally by the engine's layer-indexing.
        if isinstance(self._basis_settings, FDSettings):
            if np.any((params_in[:, 1] / self.df).astype(int) - self.waveform_kwargs["start_freq_ind"] + (N_vals_in / 2)  >  model.analysis_container_arr.data_length):
                breakpoint()
            if np.any((params_in[:, 1] / self.df).astype(int) - self.waveform_kwargs["start_freq_ind"] - (N_vals_in / 2) < 0):
                breakpoint()

        # Debug snapshots (kept for back-compat with existing diagnostic paths).
        ac_data_arr_in = model.analysis_container_arr.linear_data_arr.copy()
        ll_before_update = model.analysis_container_arr.likelihood().copy()

        # Transform to physical units and dispatch via the engine.
        params_phys = self.transform_fn.both_transforms(params_in, xp=cp)
        self._likelihood_engine.fill_template(
            model.analysis_container_arr,
            params_phys,
            walkers_in,
            N_vals_in,
            factor=factor,
            waveform_kwargs=self.waveform_kwargs,
        )

    def remove_cold_chain_sources_from_residual(self, *args, **kwargs) -> None:
        kwargs["temp"] = 0
        kwargs["apply_inds"] = True
        self.remove_sources_from_residual(*args, **kwargs)

    def remove_sources_from_residual(self, *args, **kwargs) -> None:
        self.adjust_sources_in_residual_buffer(+1, *args, **kwargs)

    def add_cold_chain_sources_to_residual(self, *args, **kwargs) -> None:
        kwargs["temp"] = 0
        kwargs["apply_inds"] = True
        self.add_sources_to_residual(*args, **kwargs)

    def add_sources_to_residual(self, *args, **kwargs) -> None:
        self.adjust_sources_in_residual_buffer(-1, *args, **kwargs)

    def run_proposal(self, model, state, band_sorter, band_temps):
        source_prop_counter = cp.zeros(band_sorter.coords.shape[0], dtype=int)

        ll_change_log = cp.zeros((self.ntemps, self.nwalkers, self.num_bands))
        total_keep = 0
        units = 2 if not self.is_rj_prop else 2
        if self.num_bands == 1:
            units = 1

        # global_ll_tracker = model.analysis_container_arr.likelihood().copy()
        # accumulated_local_diffs = cp.zeros_like(global_ll_tracker)
        # walker_accept_counts = cp.zeros(self.nwalkers, dtype=int)
        
        # random start to rotation around
        start_unit = model.random.randint(units)
        # NOTE: ``fixed_coords_for_info_mat`` / ``has_fixed_coords`` removed
        # along with the info-matrix Cholesky cache. The chunked-het NUTS
        # path recomputes the per-source Hessian metric each propose() call
        # for only the sources picked for update, so no cross-iteration
        # caching is needed.
        for tmp in range(units):
            # continue
            remainder = (start_unit + tmp) % units
            if self.num_bands == 1:
                remainder = 0

            # add back in all sources in the cold-chain
            # residual from this group
            # llbef1 = model.analysis_container_arr.likelihood(source_only=True)
            self.remove_cold_chain_sources_from_residual(
                model, band_sorter, units=units, remainder=remainder
            )
            # llbef2 = model.analysis_container_arr.likelihood(source_only=True)

            # keep1 = (
            #     (band_indices % units == remainder)
            #     & (band_indices < len(self.band_edges) - 2)
            #     & (band_indices > 1)
            #     & (self.band_N_vals[band_indices] < 1024)  # TESTING
            # )
            # TODO: check issue at ~23.5 mHz, removing for now. Really just removing high edge band

            apply_inds = not self.is_rj_prop
            extra_bool = (
                (band_sorter.band_inds < self.num_bands - 1) & (band_sorter.band_inds > 0)
            ) if self.num_bands > 1 else None       
            subset_of_interest = band_sorter.get_subset(
                units=units,
                remainder=remainder,
                apply_inds=apply_inds,
                extra_bool=extra_bool,
            )
            if subset_of_interest is None:
                continue

            if np.any(subset_of_interest.band_inds == 0):
                breakpoint()
            # start all false, then highlight sources of interest
            # sources_of_interest = self.xp.zeros_like(source_prop_counter, dtype=bool)
            # sources_of_interest[subset_of_interest.inds_main_band_sorter] = True
            # # remove this
            # sources_of_interest[(band_sorter.band_inds >= 463)] = False

            iteration_num = 0

            # with open("tmp.dat", "w") as fp:
            #     tmp = f"{iteration_num}, {sources_of_interest.sum()}\n"
            #     fp.write(tmp)
            #     print(tmp)

            special_indices_unique, special_indices_index, special_indices_count = cp.unique(
                subset_of_interest.special_band_inds,
                return_index=True,
                return_counts=True,
            )
            sort = self.xp.argsort(special_indices_count)  # [::-1]
            special_indices_unique[:] = special_indices_unique[sort]
            special_indices_index[:] = special_indices_index[sort]
            special_indices_count[:] = special_indices_count[sort]
            run_count = self.xp.zeros_like(special_indices_count)
            still_to_run = self.xp.ones_like(special_indices_count, dtype=bool)
            currently_running_special_inds = -self.xp.ones(self.num_band_preload, dtype=int)
            start_index_buf = 0
            _inds_now_tmp = self.xp.arange(special_indices_count.shape[0])

            # MAKE THIS INTO A GENERATOR
            inds_now = _inds_now_tmp[still_to_run][: self.num_band_preload]
            special_indices_unique_now = special_indices_unique[inds_now]

            special_indices_index_now = special_indices_index[inds_now]
            special_indices_count_now = special_indices_count[inds_now]
            currently_running_special_inds = special_indices_unique_now.copy()
            switch_now = self.xp.zeros(currently_running_special_inds.shape[0], dtype=bool)

            # run_it bands in buffer still needed
            run_it = self.xp.ones(currently_running_special_inds.shape[0], dtype=bool)
            buffer_obj = subset_of_interest.get_buffer(
                model.analysis_container_arr, special_indices_unique_now
            )

            accepted_out = self.xp.zeros((self.ntemps, self.nwalkers, self.num_bands), dtype=int)
            current_ind_start = currently_running_special_inds.shape[0]
            # TODO: move sources of interest inside? I do not think so right now
            init_band = False
            while self.xp.any(still_to_run):
                st_1 = time.perf_counter()

                if self.xp.any(switch_now):
                    # breakpoint()
                    num_new_sub_bands = switch_now.sum().item()
                    currend_end_ind = current_ind_start + num_new_sub_bands
                    if currend_end_ind > len(special_indices_unique):
                        currend_end_ind = len(special_indices_unique)
                        num_new_sub_bands = currend_end_ind - current_ind_start

                    if num_new_sub_bands > 0:
                        inds_fill = self.xp.arange(switch_now.shape[0])[switch_now][
                            :num_new_sub_bands
                        ]

                        inds_now[inds_fill] = current_ind_start + self.xp.arange(num_new_sub_bands)
                        special_indices_unique_now = special_indices_unique[
                            current_ind_start:currend_end_ind
                        ]
                        special_indices_index_now = special_indices_index[
                            current_ind_start:currend_end_ind
                        ]
                        special_indices_count_now = special_indices_count[
                            current_ind_start:currend_end_ind
                        ]
                        currently_running_special_inds[inds_fill] = special_indices_unique_now

                        subset_of_interest.get_buffer(
                            model.analysis_container_arr,
                            special_indices_unique_now,
                            inds_fill=inds_fill,
                            buffer_obj=buffer_obj,
                        )
                        current_ind_start = currend_end_ind
                    else:
                        assert num_new_sub_bands == 0
                        run_it = run_count[inds_now] < special_indices_count[inds_now]

                # print("run")
                prev_inds_now = inds_now.copy()[run_it]
                assert self.xp.all(
                    buffer_obj.special_indices_unique == currently_running_special_inds
                )
                source_map_now = band_sorter.get_subset_inds(
                    special_band_inds=buffer_obj.special_indices_unique[run_it]
                )
                coords_now = band_sorter.coords[source_map_now]
                special_band_inds_now = band_sorter.special_band_inds[source_map_now]

                # randomly permute rj order
                # randomly orders every time
                # some have been run already because
                # either need to trick this to order those that have been run to the beginning
                # cannot remove them because need them for in model moves
                permute_inds = self.xp.random.permutation(
                    self.xp.arange(special_band_inds_now.shape[0])
                )
                special_band_inds_now = special_band_inds_now[permute_inds]
                coords_now[:] = coords_now[permute_inds]
                source_map_now[:] = source_map_now[permute_inds]

                # TO FAKE THE ORDERING
                # we subtract 1/2 for any with inds==False to trick the ordering
                _special_band_inds_now = special_band_inds_now.copy().astype(float)
                _has_run_from_sorter = band_sorter.has_run_rj[source_map_now]
                _special_band_inds_now[_has_run_from_sorter] -= 1.0 / 2.0

                sort2 = self.xp.argsort(_special_band_inds_now)
                special_band_inds_now[:] = special_band_inds_now[sort2]
                coords_now[:] = coords_now[sort2]
                source_map_now[:] = source_map_now[sort2]
                _special_band_inds_now[:] = _special_band_inds_now[sort2]
                _has_run_from_sorter[:] = _has_run_from_sorter[sort2]

                sort3 = self.xp.argsort(currently_running_special_inds[run_it])

                _uni_special, _uni_special_index, _uni_special_counts = self.xp.unique(
                    special_band_inds_now, return_index=True, return_counts=True
                )
                # need to arange these like the currently_running_special_inds

                uni_special = -self.xp.ones_like(currently_running_special_inds)
                uni_special_index = -self.xp.ones_like(currently_running_special_inds)
                uni_special_counts = -self.xp.ones_like(currently_running_special_inds)

                inds_fill_uni_special = self.xp.arange(currently_running_special_inds.shape[0])[
                    run_it
                ]
                uni_special[inds_fill_uni_special] = _uni_special[self.xp.argsort(sort3)]
                uni_special_index[inds_fill_uni_special] = _uni_special_index[
                    self.xp.argsort(sort3)
                ]
                uni_special_counts[inds_fill_uni_special] = _uni_special_counts[
                    self.xp.argsort(sort3)
                ]

                # TODO: CHECK BAND TEMPS IN ACCEPT
                if self.use_prior_removal:
                    if self.xp.any((~band_sorter.inds[source_map_now]) & ~_has_run_from_sorter):
                        breakpoint()
                try:
                    assert (currently_running_special_inds == uni_special)[run_it].all()
                except AssertionError:
                    breakpoint()

                current_rj_counter = self.xp.zeros_like(uni_special_index)
                try:
                    current_rj_counter[:] = run_count[inds_now]  # removed [inds_now[run_it]]
                except ValueError:
                    breakpoint()
                max_counts = uni_special_counts.max().item()
                num_proposals_here = (
                    300  # self.num_repeat_proposals  #  if not self.is_rj_prop else max_counts
                )

                # NUTS path: no per-source Cholesky cache anymore. Each
                # in-model propose() step recomputes the Hessian for only
                # the sources picked for update against the *clean* per-band
                # buffer (h_curr already removed). See Chunk 4 wiring below.
                been_picked_for_rj_update = self.xp.zeros_like(source_map_now, dtype=bool)
                have_not_run_in_model = True
                previous_inds = band_sorter.inds.copy()
                for move_i in range(num_proposals_here):
                    is_rj_now = bool(np.random.choice([0, 1], p=[0.97, 0.03]))

                    if band_sorter.inds[source_map_now].sum() == 0:
                        is_rj_now = True

                    # if not self.is_rj_prop:
                    if not is_rj_now:
                        # print("Running Fisher matrix calcualtion while it is not yet setup for Mojito")
                        _new_source_map_here_in_model = self.xp.arange(source_map_now.shape[0])[
                            band_sorter.inds[source_map_now]
                        ]

                        (
                            uni_special_in_model,
                            uni_special_index_in_model,
                            uni_special_counts_in_model,
                        ) = self.xp.unique(
                            special_band_inds_now[band_sorter.inds[source_map_now]],
                            return_index=True,
                            return_counts=True,
                        )

                        choice_fraction = cp.random.rand(len(uni_special_in_model))
                        try:
                            sources_picked_for_update = _new_source_map_here_in_model[
                                uni_special_index_in_model
                                + cp.floor(choice_fraction * uni_special_counts_in_model).astype(
                                    int
                                )
                            ]
                        except ValueError:
                            breakpoint()
                        run_now_tmp = self.xp.isin(
                            currently_running_special_inds,
                            band_sorter.special_band_inds[
                                source_map_now[sources_picked_for_update]
                            ],
                        )
                        assert self.xp.all(
                            band_sorter.inds[source_map_now[sources_picked_for_update]]
                        )

                        # === info-matrix path REMOVED (Chunk 3) ===
                        # The Fisher / Cholesky cache that produced the
                        # in-model Gaussian proposal has been excised. Its
                        # replacement -- per-source NUTS over the band buffer
                        # with the source temporarily subtracted -- lives in
                        # the proposal block below (Chunk 4). No caching is
                        # done across propose() calls; the Hessian metric is
                        # rebuilt for each picked source every call.

                    # st_1 = time.perf_counter()
                    else:
                        run_now_tmp = (current_rj_counter < uni_special_counts) & run_it
                        sources_picked_for_update = (uni_special_index + current_rj_counter)[
                            run_now_tmp
                        ]
                        inds_buffer_running_now = self.xp.arange(run_now_tmp.shape[0])[run_now_tmp]
                        if self.use_prior_removal:
                            if self.xp.any(
                                ~band_sorter.inds[source_map_now[sources_picked_for_update]]
                            ):
                                breakpoint()
                        current_rj_counter[run_now_tmp] += 1

                    inds_to_update = source_map_now[sources_picked_for_update].copy()

                    if is_rj_now:
                        band_sorter.has_run_rj[inds_to_update] = True

                    if not is_rj_now:  # self.is_rj_prop:
                        assert self.xp.all(band_sorter.inds[inds_to_update])

                    params_to_update = coords_now[sources_picked_for_update].copy()
                    special_band_inds_to_update = special_band_inds_now[
                        sources_picked_for_update
                    ].copy()

                    # make sure periodic parameters are wrapped
                    params_to_update[:] = self.periodic.wrap(
                        {"gb": params_to_update[:, None, :]}, xp=self.xp
                    )["gb"][:, 0]

                    data_index_to_update = buffer_obj.get_index(special_band_inds_to_update)
                    # map is back to full band and coords
                    map_to_update = (
                        band_sorter.temp_inds[inds_to_update],
                        band_sorter.walker_inds[inds_to_update],
                        band_sorter.band_inds[inds_to_update],
                    )
                    map_to_update_cpu = (
                        asnumpy(band_sorter.temp_inds[inds_to_update]),
                        asnumpy(band_sorter.walker_inds[inds_to_update]),
                        asnumpy(band_sorter.band_inds[inds_to_update]),
                    )
                    if self.xp.any(params_to_update[:, 0] < -100.0):
                        breakpoint()
                    if not is_rj_now:  # self.is_rj_prop:
                        old_coords = params_to_update.copy()
                        # custom group stretch
                        # TODO: work into main group stretch somehow
                        if False:
                            params_into_proposal = params_to_update[None, :, None, :]

                            self.friend_start_inds_now = band_sorter.friend_start_inds[
                                inds_to_update
                            ]
                            # branch_supps_into_proposal = BranchSupplemental({"friend_start_inds": friends_into_proposal}, base_shape=friends_into_proposal.shape)
                            inds_into_proposal = self.xp.ones(
                                params_into_proposal.shape[:-1], dtype=bool
                            )

                            # TODO: check detailed balance
                            q, update_factors = self.get_proposal(
                                {"gb": params_into_proposal},
                                model.random,
                                s_inds_all={"gb": inds_into_proposal},
                                cp=self.xp,
                                return_gpu=True,
                            )  # , branch_supps=branch_supps_into_proposal)
                            new_coords = q["gb"][0, :, 0, :]

                        else:
                            # === Chunk 4: buffer-flip + per-leaf NUTS step ===
                            # Workflow per source picked for in-model update:
                            #   1) subtract h_curr from the band buffer so the
                            #      residual is "clean" (d - h_other) for that
                            #      band and the Hessian / leapfrog navigate L
                            #      on a self-consistent buffer.
                            #   2) compute the per-leaf Hessian metric (jax
                            #      autograd; psd_fix=True takes |-H| as the
                            #      mass matrix).
                            #   3) run one NUTS iteration -- closures supply
                            #      the per-leaf tempered log-posterior and
                            #      gradient (curr_beta * L; the prior is
                            #      gated downstream by curr_logp).
                            #   4) leave the buffer flipped; the outer M-H
                            #      accept logic decides which template
                            #      (old_coords on reject, new_coords on
                            #      accept) gets added back to the buffer in
                            #      the post-flip block.
                            from eryn.moves.nuts import NUTSSampler
                            nuts_subtract_index = data_index_to_update.copy()
                            nuts_N_vals = self.band_N_vals[
                                band_sorter.band_inds[inds_to_update]
                            ].copy()
                            buffer_obj.remove_sources_from_band_buffer(
                                old_coords, nuts_subtract_index, nuts_N_vals,
                            )
                            # Backend (JAX autograd) is fixed on the
                            # Buffer's gb_wdm_comp at construction; no
                            # runtime backend= kwarg per the sprint rule.
                            M_metric = buffer_obj.hessian(
                                old_coords, nuts_subtract_index, nuts_subtract_index,
                                nuts_N_vals, psd_fix=True,
                            )
                            M_metric = self.xp.asarray(M_metric)

                            # Per-leaf inverse temperature (one entry per
                            # picked source). Closes over via curr_beta_nuts.
                            curr_beta_nuts = band_temps[map_to_update[2], map_to_update[0]]

                            def _log_post_fn(x_batch,
                                              _bidx=nuts_subtract_index,
                                              _Nv=nuts_N_vals,
                                              _beta=curr_beta_nuts):
                                d_h, h_h = buffer_obj.get_ll(
                                    self.xp.asarray(x_batch),
                                    _bidx, _bidx, _Nv,
                                )
                                ll = self.xp.asarray(d_h) - 0.5 * self.xp.asarray(h_h)
                                return _beta * ll

                            def _grad_log_post_fn(x_batch,
                                                   _bidx=nuts_subtract_index,
                                                   _Nv=nuts_N_vals,
                                                   _beta=curr_beta_nuts):
                                g = buffer_obj.get_ll_grad(
                                    self.xp.asarray(x_batch),
                                    _bidx, _bidx, _Nv,
                                )
                                return _beta[:, None] * self.xp.asarray(g)

                            nuts = NUTSSampler(
                                grad_log_posterior_fn=_grad_log_post_fn,
                                log_posterior_fn=_log_post_fn,
                                ndim=8,
                                metric=M_metric,
                                step_size=float(getattr(self, "nuts_step_size", 0.1)),
                                max_tree_depth=int(getattr(self, "nuts_max_tree_depth", 4)),
                                adapt_step_size=False,
                            )
                            nuts_out = nuts.step(old_coords)
                            new_coords = (nuts_out[0] if isinstance(nuts_out, tuple)
                                          else nuts_out)
                            # NUTS leapfrog preserves detailed balance, so
                            # no log-Jacobian correction is needed at the
                            # outer M-H gate.
                            update_factors = self.xp.zeros(new_coords.shape[0])
                            # Restore the buffer to its pre-flip state by
                            # adding h_curr (old_coords) back. The
                            # downstream accept-path then runs its own
                            # remove(old)/add(new) cycle for accepted
                            # sources from a clean baseline, identical to
                            # how the legacy Cholesky proposal left it.
                            # ll_diff=0 in the swap_ll branch ensures the
                            # outer M-H gate is prior-only for in-model.
                            buffer_obj.add_sources_to_band_buffer(
                                old_coords, nuts_subtract_index, nuts_N_vals,
                            )

                        new_coords[:] = self.periodic.wrap(
                            {"gb": new_coords[:, None, :]}, xp=self.xp
                        )["gb"][:, 0]

                        prev_logp = cp.asarray(
                            self.gpu_priors["gb"].logpdf(params_to_update)
                        )  # , psds=self.mgh.psd_shaped[0][0], walker_inds=curr_index)
                        curr_logp = cp.asarray(
                            self.gpu_priors["gb"].logpdf(new_coords)
                        )  # , psds=self.mgh.psd_shaped[0][0], walker_inds=curr_index)

                    else:
                        old_coords = params_to_update.copy()
                        new_coords = params_to_update.copy()
                        logp_tmp = cp.asarray(self.gpu_priors["gb"].logpdf(old_coords))
                        
                        # if self.xp.any(self.xp.isinf(logp_tmp[run_now_tmp])):
                        #     breakpoint()
                            
                        prev_logp = cp.zeros_like(logp_tmp)
                        curr_logp = cp.zeros_like(logp_tmp)

                        inds = band_sorter.inds[inds_to_update].copy()
                        update_factors = band_sorter.factors[inds_to_update].copy()
                        # prevent unecessar

                        old_coords[~inds, 0] = np.log(1e-80)
                        new_coords[inds, 0] = np.log(1e-80)
                        # wrap in case
                        new_coords[:] = self.periodic.wrap(
                            {"gb": new_coords[:, None, :]}, xp=self.xp
                        )["gb"][:, 0]

                        prev_logp[inds] = logp_tmp[inds]
                        curr_logp[~inds] = logp_tmp[~inds]

                    # if cp.any(cp.isinf(prev_logp)):  # [run_now_tmp]
                    #     breakpoint()
                    # inputs into swap proposal
                    # guard on the edges with too-large frequency proposals out of band that would not be physical
                    if is_rj_now and self.xp.any(
                        ~band_sorter.inds[inds_to_update]
                        & (
                            (
                                new_coords[:, 1] / 1e3
                                < buffer_obj.frequency_lims[0][data_index_to_update]
                            )
                            | (
                                new_coords[:, 1] / 1e3
                                > buffer_obj.frequency_lims[1][data_index_to_update]
                            )
                        )
                    ):
                        breakpoint()

                    # if not is_rj_now and self.xp.any((old_coords[:, 1] / 1e3 < buffer_obj.frequency_lims[0][data_index_to_update]) | (old_coords[:, 1] / 1e3 > buffer_obj.frequency_lims[1][data_index_to_update])):
                    #     breakpoint()
                    # if self.xp.any(~run_now_tmp):
                    #     breakpoint()

                    if is_rj_now:
                        curr_logp[
                            (
                                new_coords[:, 1] / 1e3
                                < buffer_obj.frequency_lims[0][data_index_to_update]
                            )
                            | (
                                new_coords[:, 1] / 1e3
                                > buffer_obj.frequency_lims[1][data_index_to_update]
                            )
                        ] = -np.inf

                    # TODO: 2 vs 4?
                    else:
                        curr_logp[
                            (
                                cp.abs(old_coords[:, 1] / 1e3 - new_coords[:, 1] / 1e3) / self.df
                            ).astype(int)
                            > (self.band_N_vals[band_sorter.band_inds[inds_to_update]] / 4).astype(
                                int
                            )
                        ] = -np.inf

                    # outside wavelength / 4 of band
                    curr_logp[
                        (
                            cp.abs(new_coords[:, 1] / 1e3 / self.df).astype(int)
                            < (buffer_obj.frequency_lims[0][data_index_to_update] / self.df).astype(
                                int
                            )
                            - (self.band_N_vals[band_sorter.band_inds[inds_to_update]] / 4).astype(
                                int
                            )
                        )
                    ] = -np.inf
                    curr_logp[
                        (
                            cp.abs(new_coords[:, 1] / 1e3 / self.df).astype(int)
                            > (buffer_obj.frequency_lims[1][data_index_to_update] / self.df).astype(
                                int
                            )
                            + (self.band_N_vals[band_sorter.band_inds[inds_to_update]] / 4).astype(
                                int
                            )
                        )
                    ] = -np.inf

                    # remove any from log like comp when finished running for that band
                    # curr_logp[~run_now_tmp] = -np.inf
                    ll_diff = cp.full_like(prev_logp, -1e300)
                    opt_snr = cp.full_like(prev_logp, 0.0)
                    keep2 = ~cp.isinf(curr_logp)
                    # et_1 = time.perf_counter()
                    # print("2nd:", et_1 - st_1)

                    # st_1 = time.perf_counter()
                    params_remove = old_coords[keep2].copy()
                    params_add = new_coords[keep2].copy()

                    # data indexes align with the buffers (1 per buffer except for inf priors)
                    data_index = data_index_to_update[keep2].astype(np.int32)
                    swap_N_vals = self.band_N_vals[
                        band_sorter.band_inds[inds_to_update[keep2]]
                    ].copy()

                    # CANNOT COPY PARAMETER ARRAYS, IN PLACE ADJUSTMENT IF PHASE MAXIMIZING
                    if is_rj_now:
                        # RJ swap (add-one / remove-one): canonical swap_ll
                        # on the unflipped buffer. Phase-maximisation is
                        # allowed here -- it's the analytic phi0 marginal
                        # the legacy RJ path has always used.
                        ll_diff[keep2] = buffer_obj.get_swap_ll(
                            params_remove,
                            params_add,
                            data_index,
                            swap_N_vals,
                            phase_maximize=self.phase_maximize,
                        )
                        # in case there is phase marginalization, need to adjust in new_coords
                        if self.phase_maximize:
                            new_coords[keep2] = params_add[:]
                    else:
                        # In-model NUTS path: leapfrog already integrated
                        # the tempered likelihood, so the outer M-H gate
                        # is on prior support only. Setting ll_diff=0
                        # collapses ``curr_beta * ll_diff + (curr_logp -
                        # prev_logp)`` down to the prior delta.
                        ll_diff[keep2] = 0.0

                    curr_beta = band_temps[map_to_update[2], map_to_update[0]]
                    # print("change priors?, need to adjust here")

                    delta_logP = curr_beta * ll_diff + (curr_logp - prev_logp)
                    lnpdiff = delta_logP + update_factors.squeeze()
                    accept = lnpdiff >= cp.log(cp.random.rand(*lnpdiff.shape))

                    if is_rj_now and self.use_prior_removal:
                        if self.xp.any(~(band_sorter.inds[inds_to_update][accept])):
                            breakpoint()
                    # if self.is_rj_prop:
                    #     _band_count = self.xp.asarray(band_sorter.get_band_info()["band_counts"][map_to_update_cpu])
                    #     # TODO: remove this for PE part
                    #     accept[(_band_count >= (self.time + 1)) & (~band_sorter.inds[inds_to_update])] = False

                    # if self.xp.any(special_band_inds_now == 65):
                    #     breakpoint()
                    # need to copy to old array before changing in place
                    old_params_to_update = params_to_update.copy()
                    if self.xp.any(params_to_update[:, 0] < -100.0):
                        breakpoint()
                    # if rj prop, then the parameters do not change, just inds
                    if is_rj_now:  # self.is_rj_prop:
                        # adjust phase in case of phase maximization
                        # NEEDED in search to work properly
                        # index 3 is phi0, all other parameters are the same
                        params_to_update[accept, 3] = new_coords[accept, 3]

                    else:
                        params_to_update[accept] = new_coords[accept]

                    coords_now[sources_picked_for_update] = params_to_update[:]

                    if self.xp.any(params_to_update[:, 0] < -100.0):
                        breakpoint()

                    if cp.any(accept):
                        inds_update_accept = inds_to_update[accept]

                        ll_accept = ll_diff[accept]
                        if is_rj_now:  # self.is_rj_prop:
                            # update inds
                            band_sorter.inds[inds_update_accept] = ~band_sorter.inds[
                                inds_update_accept
                            ]
                            # NOTE: the fixed_coords_for_info_mat / has_fixed_coords
                            # bookkeeping that used to update here has been removed
                            # along with the info-matrix Cholesky cache (Chunk 3).

                        temp_inds_accept = band_sorter.temp_inds[inds_update_accept]
                        walker_inds_accept = band_sorter.walker_inds[inds_update_accept]
                        band_inds_accept = band_sorter.band_inds[inds_update_accept]
                        ll_change_log[
                            temp_inds_accept, walker_inds_accept, band_inds_accept
                        ] += ll_accept

                        accepted_out[temp_inds_accept, walker_inds_accept, band_inds_accept] += 1
                        
                        # for t_idx, w_idx, ll_change in zip(temp_inds_accept, walker_inds_accept, ll_accept):
                        #     if t_idx == 0:  # Count acceptances for the cold chain
                        #         accumulated_local_diffs[w_idx] += ll_change
                        #         walker_accept_counts[w_idx] += 1
                                
                        # switch accepted waveform
                        old_coords_for_change = old_coords[accept].copy()
                        new_coords_for_change = new_coords[accept].copy()

                        old_change_index = data_index_to_update[accept].copy().astype(np.int32)
                        new_change_index = old_change_index.copy()

                        old_change_N_vals = self.band_N_vals[
                            band_sorter.band_inds[inds_update_accept]
                        ].copy()
                        new_change_N_vals = old_change_N_vals.copy()

                        # TODO: should we combine this to make faster
                        ll_before = buffer_obj.likelihood(source_only=True)
                        buffer_obj.remove_sources_from_band_buffer(
                            old_coords_for_change, old_change_index, old_change_N_vals
                        )
                        ll_mid = buffer_obj.likelihood(source_only=True)
                        buffer_obj.add_sources_to_band_buffer(
                            new_coords_for_change, new_change_index, new_change_N_vals
                        )
                        ll_after = buffer_obj.likelihood(source_only=True)

                        ll_check = np.zeros_like(ll_after)
                        ll_check[data_index_to_update[accept]] = ll_accept

                        # if not np.allclose(ll_check, ll_after - ll_before):
                        #     breakpoint()
                        # if move_i % 25 == 0:
                        #     try:
                        #         if 1e-4 < np.abs((ll_after - ll_before)[data_index_to_update] - ll_diff * accept).max():
                        #             breakpoint()
                        #     except ValueError:
                        #         breakpoint()

                    # print(iteration_num, move_i)
                    self.mempool.free_all_blocks()
                    previous_inds = band_sorter.inds.copy()

                    source_prop_counter[inds_to_update] += 1
                    # with open("tmp.dat", "a") as fp:
                    #     tmp = f"move {move_i}: {iteration_num}, {sources_of_interest.sum()}"
                    #     fp.write(tmp + "\n")
                    #     print(tmp)
                    # will recalculate prior anyways so leaving that out

                    # change WAVEFORMS THAT HAVE BEEN ACCEPTED
                # I THINK THIS SHOULD BE OK WITHOUT COUNTING IN MODEL
                # RJ COUNT IS PROPORTIONAL TO NUMBER OF SOURCES IN THE BAND,
                # SO IT WILL ALSO ACCOUNT FOR NUM_REPEAT_PROPOSALS FOR IN-MODEL
                run_count[inds_now] = current_rj_counter

                # if not self.is_rj_prop:
                #     # should be subset for in model
                #     switch_now[:] = run_count[inds_now] >= special_indices_count[inds_now]
                #     still_to_run = run_count < special_indices_count
                # else:
                #     # should all for RJ
                switch_now[:] = run_count[inds_now] >= special_indices_count[inds_now]
                still_to_run = run_count < special_indices_count

                band_sorter.coords[source_map_now] = coords_now[:]

                # inds change is taken care of inplace
                iteration_num += 1
                # with open("tmp.dat", "a") as fp:
                #     tmp = f"{iteration_num}, {sources_of_interest.sum()}"
                #     fp.write(tmp + "\n")
                #     print(tmp)
                self.mempool.free_all_blocks()
                # update prop counter
                print(f"For {self.name}, we still have to run, {still_to_run.sum()}")
            # add back in all sources in the cold-chain
            # residual from this group
            # llaf1 = model.analysis_container_arr.likelihood()

            self.add_cold_chain_sources_to_residual(
                model, band_sorter, units=units, remainder=remainder
            )
            # final_global_ll = model.analysis_container_arr.likelihood().copy()
            # true_global_diffs = final_global_ll - global_ll_tracker
            
            # print("\n" + "="*65)
            # print("WALKER DRIFT ANALYSIS (COLD CHAIN)")
            # print(f"{'Walker':<8} | {'Accepted':<8} | {'Local C++ Sum':<15} | {'Global Diff':<15} | {'Error':<15}")
            # print("-" * 65)
            
            # for w in range(self.nwalkers):
            #     loc = accumulated_local_diffs[w].item()
            #     glob = true_global_diffs[w].item()
            #     err = abs(glob - loc)
            #     acc = walker_accept_counts[w].item()
            #     print(f"{w:<8} | {acc:<8} | {loc:<15.4f} | {glob:<15.4f} | {err:<15.4f}")
            # print("="*65)
            
            # llaf2 = model.analysis_container_arr.likelihood(source_only=True)
            # breakpoint()
            # ll_change_sum = ll_change_log.sum(axis=-1)
            # check_in = state.log_like[0] + ll_change_sum[0].get()    
                
            if self.backend.uses_cupy:
                self.xp.cuda.runtime.deviceSynchronize()


        return ll_change_log

    def run_tempering(self, model, state, band_sorter, band_temps):
        ll_change_log_temp = cp.zeros((self.ntemps, self.nwalkers, self.num_bands))

        band_swaps_accepted = cp.zeros((len(self.band_edges) - 1, self.ntemps - 1), dtype=int)
        band_swaps_proposed = cp.zeros((len(self.band_edges) - 1, self.ntemps - 1), dtype=int)
        current_band_counts = cp.zeros((len(self.band_edges) - 1, self.ntemps), dtype=int)

        # start_band_sorter = deepcopy(band_sorter)
        units = 2
        tmp_start = np.random.randint(units)
        for tmp in range(units):
            remainder = (tmp_start + tmp) % units
            start = remainder
            if start == 0:
                # this is because we start at band 1
                odd = True
                bool_remainder = 1

            else:
                odd = False
                bool_remainder = 0

            ll_before2 = model.analysis_container_arr.likelihood()
            self.remove_cold_chain_sources_from_residual(
                model,
                band_sorter,
                extra_bool=(band_sorter.band_inds % 2 == bool_remainder),
            )
            ll_after2 = model.analysis_container_arr.likelihood()

            if self.num_bands == 1:
                num_bands_tempered = 1
                band_index_arr = cp.arange(1)
            else:
                num_bands_tempered = self.num_bands - 2
                band_index_arr = cp.arange(1, self.num_bands -1)
                
            num_bands_unit = np.arange(num_bands_tempered)[start::2].shape[0]

            walkers_permuted = (
                cp.asarray(
                    [
                        cp.random.permutation(cp.arange(self.nwalkers))
                        for _ in range(self.ntemps * num_bands_tempered)
                    ]
                )
                .reshape(num_bands_tempered, self.ntemps, self.nwalkers)
                .transpose(0, 2, 1)[start::2]
            )
            temp_index = (
                cp.repeat(cp.arange(self.ntemps), num_bands_tempered * self.nwalkers)
                .reshape(self.ntemps, num_bands_tempered, self.nwalkers)
                .transpose(1, 2, 0)[start::2]
            )
            band_index = (
                cp.repeat(band_index_arr, self.ntemps * self.nwalkers)
                .reshape(num_bands_tempered, self.ntemps, self.nwalkers)
                .transpose(0, 2, 1)[start::2]
            )
            special_index = band_sorter.get_special_band_index(
                temp_index, walkers_permuted, band_index
            )

            num_bands_preload_temp = 200
            num_bands_run = 0
            while num_bands_run < self.nwalkers * num_bands_unit:
                start_ind = num_bands_run
                end_ind = start_ind + num_bands_preload_temp

                band_inds_now = band_index.reshape(-1, self.ntemps)[start_ind:end_ind].copy()
                temp_inds_now = temp_index.reshape(-1, self.ntemps)[start_ind:end_ind].copy()
                walker_inds_now = walkers_permuted.reshape(-1, self.ntemps)[
                    start_ind:end_ind
                ].copy()
                special_inds_now = special_index.reshape(-1, self.ntemps)[start_ind:end_ind].copy()
                num_bands_now = band_inds_now.shape[0]
                special_inds_now_flat = special_inds_now.flatten()

                # need to include inds
                # now_bool_full = cp.isin(band_sorter.special_band_inds, special_inds_now_flat)  # & band_sorter.inds
                # if not cp.any(now_bool_full):
                #     num_bands_run += num_bands_preload_temp
                #     # print("num bands", num_bands_run)
                #     continue

                # _, special_inds_index = cp.unique(band_sorter.special_band_inds[now_bool_full], return_index=True)
                buffer_obj = band_sorter.get_buffer(
                    model.analysis_container_arr,
                    special_inds_now_flat,
                    use_template_arr=True,
                )

                current_lls = buffer_obj.likelihood(source_only=True).reshape(-1, self.ntemps)
                band_combo_map = buffer_obj.unique_band_combos.reshape(-1, self.ntemps, 3)
                current_lls_orig = current_lls.copy()
                # TODO: CHECK LIKELIHOODS/
                for t in range(self.ntemps)[1:][::-1]:
                    st = time.perf_counter()
                    i1 = t
                    i2 = t - 1

                    buffer_i1 = cp.arange(buffer_obj.num_bands_now)[i1 :: self.ntemps]
                    buffer_i2 = cp.arange(buffer_obj.num_bands_now)[i2 :: self.ntemps]

                    # IMPORTANT: MAPPING IMPLICITLY UNDERSTANDS WHERE THINGS WILL BE
                    tmp_buffer = buffer_obj.template_buffer[buffer_i1].copy()
                    buffer_obj.template_buffer[buffer_i1] = buffer_obj.template_buffer[buffer_i2]
                    buffer_obj.template_buffer[buffer_i2] = tmp_buffer[:]

                    # TODO: add indices because not every likelihood is needed
                    new_lls = buffer_obj.likelihood(source_only=True).reshape(-1, self.ntemps)[
                        :, i2 : i1 + 1
                    ]
                    old_lls = current_lls[:, i2 : i1 + 1]

                    beta1 = band_temps[(band_inds_now[:, 0], i1)]
                    beta2 = band_temps[(band_inds_now[:, 0], i2)]

                    paccept = beta1 * (new_lls[:, 1] - old_lls[:, 1]) + beta2 * (
                        new_lls[:, 0] - old_lls[:, 0]
                    ) # ! this is changed because it think this was wrong, below is the previous paccept (comparing with paccept in paper, it should now be good)
                    # paccept = bi * (band_here_i1->swapped_like - band_here_i->current_like) + bi1 * (band_here_i->swapped_like - band_here_i1->current_like);

                    raccept = cp.log(cp.random.uniform(size=paccept.shape))
                    sel = paccept > raccept

                    current_lls[sel, i2 : i1 + 1] = new_lls[sel]
                    
                    # reverse not accepted ones
                    buffer_i1_reject = buffer_i1[~sel]
                    buffer_i2_reject = buffer_i2[~sel]

                    tmp_i1 = buffer_obj.template_buffer[buffer_i1_reject].copy()
                    buffer_obj.template_buffer[buffer_i1_reject] = buffer_obj.template_buffer[
                        buffer_i2_reject
                    ]
                    buffer_obj.template_buffer[buffer_i2_reject] = tmp_i1[:]

                    band_swaps_accepted[band_inds_now[:, 0], i2] += sel.astype(int)
                    band_swaps_proposed[band_inds_now[:, 0], i2] += 1

                    band_inds_exchange_i1 = band_inds_now[sel, i1]
                    walker_inds_exchange_i1 = walker_inds_now[sel, i1]
                    band_inds_exchange_i2 = band_inds_now[sel, i2]
                    walker_inds_exchange_i2 = walker_inds_now[sel, i2]

                    special_ind_test_1 = band_sorter.get_special_band_index(
                        i1, walker_inds_exchange_i1, band_inds_exchange_i1
                    )
                    special_ind_test_2 = band_sorter.get_special_band_index(
                        i2, walker_inds_exchange_i2, band_inds_exchange_i2
                    )

                    # temp_indices[fix_1] = i2
                    # temp_indices[fix_2] = i1

                    ind_sort_1 = cp.argsort(special_ind_test_1.flatten())
                    ind_keep_1 = cp.isin(band_sorter.special_band_inds, special_ind_test_1)
                    sorted_map_1 = cp.searchsorted(
                        special_ind_test_1[ind_sort_1],
                        band_sorter.special_band_inds[ind_keep_1],
                        side="left",
                    )
                    # inds_now_1 = band_sorter.inds[ind_keep_1][sorted_map_1]

                    ind_sort_2 = cp.argsort(special_ind_test_2.flatten())
                    ind_keep_2 = cp.isin(band_sorter.special_band_inds, special_ind_test_2)
                    sorted_map_2 = cp.searchsorted(
                        special_ind_test_2[ind_sort_2],
                        band_sorter.special_band_inds[ind_keep_2],
                        side="left",
                    )
                    # inds_now_2 = band_sorter.inds[ind_keep_2][sorted_map_2]

                    band_sorter.special_band_inds[ind_keep_1] = special_ind_test_2[
                        ind_sort_1[sorted_map_1]
                    ]
                    band_sorter.temp_inds[ind_keep_1] = i2
                    band_sorter.walker_inds[ind_keep_1] = walker_inds_exchange_i2[
                        ind_sort_1[sorted_map_1]
                    ]
                    # do not need to change band index but check it
                    assert cp.all(
                        band_sorter.band_inds[ind_keep_1]
                        == band_inds_exchange_i2[ind_sort_1[sorted_map_1]]
                    )

                    band_sorter.special_band_inds[ind_keep_2] = special_ind_test_1[
                        ind_sort_2[sorted_map_2]
                    ]
                    band_sorter.temp_inds[ind_keep_2] = i1
                    band_sorter.walker_inds[ind_keep_2] = walker_inds_exchange_i1[
                        ind_sort_2[sorted_map_2]
                    ]

                    et = time.perf_counter()
                    # print(et - st, t, num_bands_run, self.nwalkers * num_bands_unit)

                diffs = current_lls - current_lls_orig
                # TODO: this should be = not += (?)
                ll_change_log_temp[
                    (
                        buffer_obj.unique_band_combos[:, 0],
                        buffer_obj.unique_band_combos[:, 1],
                        buffer_obj.unique_band_combos[:, 2],
                    )
                ] = diffs.flatten()
                num_bands_run += num_bands_preload_temp

            ll_before3 = model.analysis_container_arr.likelihood()
            self.add_cold_chain_sources_to_residual(
                model,
                band_sorter,
                extra_bool=(band_sorter.band_inds % 2 == bool_remainder),
            )
            ll_after3 = model.analysis_container_arr.likelihood()

        # adapt if desired
        print("change adaptation")
        if self.time > 0:
            ratios = (band_swaps_accepted / band_swaps_proposed).T
            betas0 = band_temps.copy().T
            betas1 = betas0.copy()

            # Modulate temperature adjustments with a hyperbolic decay.
            decay = self.temperature_control.adaptation_lag / (
                self.time + self.temperature_control.adaptation_lag
            )
            kappa = decay / self.temperature_control.adaptation_time

            # Construct temperature adjustments.
            dSs = kappa * (ratios[:-1] - ratios[1:])

            # Compute new ladder (hottest and coldest chains don't move).
            deltaTs = cp.diff(1 / betas1[:-1], axis=0)

            deltaTs *= cp.exp(dSs)
            betas1[1:-1] = 1 / (cp.cumsum(deltaTs, axis=0) + 1 / betas1[0])

            # Don't mutate the ladder here; let the client code do that.
            dbetas = betas1 - betas0

            band_temps += self.xp.asarray(dbetas.T)

        print("NEED TO FIX ANALYSIS CONTAINER extra factor")
        ll_change_sum_temp = ll_change_log_temp.sum(axis=-1)

        return ll_change_sum_temp, band_swaps_accepted, band_swaps_proposed

    def propose(self, model, state):
        """Use the move to generate a proposal and compute the acceptance

        Args:
            model (:class:`eryn.model.Model`): Carrier of sampler information.
            state (:class:`GFState`): Current state of the sampler.

        Returns:
            :class:`GFState`: GFState of sampler after proposal is complete.

        """

        st_all = time.perf_counter()

        if self.backend.uses_cupy:
            self.xp.cuda.runtime.setDevice(model.analysis_container_arr.gpus[0])
        # nchannels = model.analysis_container_arr.nchannels
        # data_length = model.analysis_container_arr.data_length
        # Refresh fd/df from the live ACA in case the parent attached a new
        # one between construction and propose-time. Same FD-vs-WDM
        # dispatch as in __init__.
        acs_settings = model.analysis_container_arr.settings
        if isinstance(acs_settings, FDSettings):
            self.fd = model.analysis_container_arr.f_arr.copy()
            self.df = float(model.analysis_container_arr.df)
        elif isinstance(acs_settings, WDMSettings):
            self.fd = None
            self.df = 1.0 / acs_settings.Tobs
        self.current_state = state
        np.random.seed(10)
        # print("start stretch")

        # Check that the dimensions are compatible.
        ntemps, nwalkers, nleaves_max, ndim = state.branches_coords["gb"].shape

        if not self.is_rj_prop and not np.any(state.branches["gb"].inds):
            return state, np.zeros((ntemps, nwalkers), dtype=bool)

        self.nwalkers = nwalkers
        self.ntemps = ntemps

        # Run any move-specific setup.
        self.setup(model, state.branches)
        self.num_proposals += 1
        
        # after setup is still no dist, return
        if self.rj_proposal_distribution is None:
            return state, np.zeros((ntemps, nwalkers), dtype=bool)

        new_state = GFState(state, copy=True)
        assert new_state.log_like is not None

        band_temps = cp.asarray(state.sub_states["gb"].band_info["band_temps"].copy())

        if self.is_rj_prop:
            orig_store = new_state.log_like[0].copy()

        gb_coords = cp.asarray(new_state.branches["gb"].coords)

        self.mempool.free_all_blocks()

        waveform_kwargs_now = self.waveform_kwargs.copy()
        if "N" in waveform_kwargs_now:
            waveform_kwargs_now.pop("N")
        # waveform_kwargs_now["start_freq_ind"] = self.start_freq_ind

        # if self.is_rj_prop:
        #     print("START:", new_state.log_like[0])
        # log_like_tmp = self.xp.asarray(new_state.log_like)
        # log_prior_tmp = self.xp.asarray(new_state.log_prior)

        # self.mempool.free_all_blocks()

        # gb_inds = self.xp.asarray(new_state.branches["gb"].inds)
        # gb_inds_orig = gb_inds.copy()

        # data = model.analysis_container_arr.linear_data_arr
        # psd = model.analysis_container_arr.linear_psd_arr

        # do unique for band size as separator between asynchronous kernel launches
        # band_indices = self.xp.asarray(new_state.branches["gb"].branch_supplemental.holder["band_inds"])
        # band_indices = (
        #     self.xp.searchsorted(
        #         self.band_edges,
        #         cp.asarray(new_state.branches["gb"].coords[:, :, :, 1]).flatten() / 1e3,
        #         side="right",
        #     ).reshape(new_state.branches["gb"].coords[:, :, :, 1].shape)
        #     - 1
        # )

        # N_vals_in = self.xp.asarray(new_state.branches["gb"].branch_supplemental.holder["N_vals"])
        # points_curr = self.xp.asarray(new_state.branches["gb"].coords)
        # points_curr_orig = points_curr.copy()
        # N_vals_in_orig = N_vals_in.copy()
        # band_indices_orig = band_indices.copy()

        rj_prop = None if not self.is_rj_prop else self.rj_proposal_distribution["gb"]

        # make sure all periodic parameters have been put into their range
        new_state.branches["gb"].coords[:] = self.periodic.wrap(
            {"gb": new_state.branches["gb"].coords[:].reshape(ntemps * nwalkers, nleaves_max, ndim)}
        )["gb"].reshape(ntemps, nwalkers, nleaves_max, ndim)

        print("is this okay for rj? I do not think so, check with below use of gb_inds_in")
        if self.use_prior_removal:  # TODO: make this stronger?
            keep_all_inds = False
        else:
            keep_all_inds = True

        band_sorter = BandSorter(
            new_state.branches["gb"],
            self.band_edges,
            self.band_N_vals,
            force_backend=self.force_backend,
            transform_fn=self.parameter_transforms,
            max_data_store_size=self.max_data_store_size,
            gb=self.gb,
            gb_wdm_comp=self.gb_wdm_comp,
            waveform_kwargs=self.waveform_kwargs,
            rj_prop=rj_prop,
            keep_all_inds=keep_all_inds,
        )

        do_synchronize = False
        device = self.xp.cuda.runtime.getDevice() if self.backend.uses_cupy else -1

        # get non-gb contribution
        self.remove_cold_chain_sources_from_residual(model, band_sorter, apply_inds=True)
        # Multi-GPU: snapshot every per-GPU shard of linear_data_arr inside its
        # owning device context so the copies live on the right device. Restored
        # in check_ll_inject() symmetrically.
        self.reset_non_gb_linear_data_arr = self._snapshot_linear_data_arr(
            model.analysis_container_arr
        )
        self.add_cold_chain_sources_to_residual(model, band_sorter, apply_inds=True)
        ll_after = model.analysis_container_arr.likelihood(
            source_only=False
        )  #  - cp.sum(cp.log(cp.asarray(psd[:2])), axis=(0, 2))).get()

        print(np.abs(new_state.log_like - ll_after).max())        
        # store_max_diff = np.abs(new_state.log_like[0] - ll_after).max()
        start_diffs = np.abs(new_state.log_like[0] - ll_after)

        check = ll_after - new_state.log_like[0] - start_diffs
        
        print(f"Start check: {start_diffs=}, {check=}")
        if not np.abs(check).max() < 1e-4:
            # assert np.abs(check).max() < 1.0
            new_state.log_like[0] = self.check_ll_inject(model, band_sorter)
            #? update start diffs
            start_diffs = np.abs(new_state.log_like[0] - ll_after)

        # print("CHECKING 0:", store_max_diff, self.is_rj_prop)
        # self.check_ll_inject(new_state, verbose=True)
        # assert np.all(start_diffs < 2.0)
        per_walker_band_proposals = cp.zeros((ntemps, nwalkers, self.num_bands), dtype=int)
        per_walker_band_accepted = cp.zeros((ntemps, nwalkers, self.num_bands), dtype=int)
        
        # TODO: make sure band temps transfers out
        st_prop = time.perf_counter()
        ll_change_log = self.run_proposal(model, new_state, band_sorter, band_temps)
        et_prop = time.perf_counter()
        print(self.name, "reg prop:", et_prop - st_prop)
        
        print("NEED TO FIX ANALYSIS CONTAINER extra factor")
        ll_change_sum = ll_change_log.sum(axis=-1)
        new_state.log_like[0] += _to_numpy(ll_change_sum[0])

        ll_after = model.analysis_container_arr.likelihood()
        check = ll_after - new_state.log_like[0] - start_diffs

        print(f"After proposal check: {start_diffs=}, {check=}")
        if not np.abs(check).max() < 1e-4:
            assert np.abs(check).max() < 1.0
            new_state.log_like[0] = self.check_ll_inject(model, band_sorter)
        # breakpoint()

        # TEMPERING
        self.temperature_control.swaps_accepted = np.zeros(ntemps - 1)
        self.temperature_control.swaps_proposed = np.zeros(ntemps - 1)

        # TODO: move this and check if it is needed
        # self.nchannels = model.analysis_container_arr.nchannels

        band_swaps_accepted = cp.zeros((len(self.band_edges) - 1, self.ntemps - 1), dtype=int)
        band_swaps_proposed = cp.zeros((len(self.band_edges) - 1, self.ntemps - 1), dtype=int)

        if (
            self.temperature_control is not None
            and self.time % 1 == 0
            and self.ntemps > 1
            and self.is_rj_prop
            and self.run_swaps
            # and False
        ):
            st_temp = time.perf_counter()
            ll_before1 = model.analysis_container_arr.likelihood()

            ll_change_sum_temp, band_swaps_accepted, band_swaps_proposed = self.run_tempering(
                model, new_state, band_sorter, band_temps
            )

            new_state.log_like[0] += _to_numpy(ll_change_sum_temp[0])

            ll_after = model.analysis_container_arr.likelihood()
            check = ll_after - new_state.log_like[0] - start_diffs

            print(f"After tempering check: {start_diffs=}, {check=}")
            if not np.abs(check).max() < 1e-4:
                assert np.abs(check).max() < 1.0
                new_state.log_like[0] = self.check_ll_inject(model, band_sorter)

            self.mempool.free_all_blocks()
            et_temp = time.perf_counter()
            print(self.name, "tempering duration is", et_temp - st_temp, "seconds")
            
        print("make sure this works for rj")
        special_indices_finish = (
            band_sorter.temp_inds[band_sorter.inds] * nwalkers
            + band_sorter.walker_inds[band_sorter.inds]
        ) * int(1e6) + band_sorter.coords[band_sorter.inds, 1]
        special_inds_temp_walker = (
            band_sorter.temp_inds[band_sorter.inds] * nwalkers
            + band_sorter.walker_inds[band_sorter.inds]
        )
        sorted_inds = cp.argsort(special_indices_finish)

        uni, uni_inds, uni_inverse, uni_counts = cp.unique(
            special_inds_temp_walker[sorted_inds],
            return_index=True,
            return_counts=True,
            return_inverse=True,
        )

        leaf_inds_new_tmp = cp.arange(special_indices_finish.shape[0]) - uni_inds[uni_inverse]
        leaf_inds_new = cp.zeros_like(leaf_inds_new_tmp)
        leaf_inds_new[sorted_inds] = leaf_inds_new_tmp

        print("NEED TO PROPERLY MOVE SUPPLEMENTAL INFO BASED ON OLD LEAVES.")
        inds_new = (
            _to_numpy(band_sorter.temp_inds[band_sorter.inds]),
            _to_numpy(band_sorter.walker_inds[band_sorter.inds]),
            _to_numpy(leaf_inds_new),
        )
        inds_old = (
            _to_numpy(band_sorter.orig_temp_inds[band_sorter.inds]),
            _to_numpy(band_sorter.orig_walker_inds[band_sorter.inds]),
            _to_numpy(band_sorter.orig_leaf_inds[band_sorter.inds]),
        )
        new_state.branches["gb"].coords[inds_new] = _to_numpy(band_sorter.coords[band_sorter.inds])
        new_state.branches["gb"].inds[:] = False
        # turn on all the ones that are there
        new_state.branches["gb"].inds[inds_new] = True

        # new_state.branches["gb"].branch_supplemental[inds_new] = state.branches["gb"].branch_supplemental[inds_old]
        et_all = time.perf_counter()
        print(self.name, et_all - st_all)

        # TODO: need to redo the acceptance fraction
        # get accepted fraction
        # # if not self.is_rj_prop:
        # #     accepted_check_tmp = np.zeros_like(
        # #         state.branches_inds["gb"], dtype=bool
        # #     )
        # #     accepted_check_tmp[state.branches_inds["gb"]] = np.all(
        # #         np.abs(
        # #             new_state.branches_coords["gb"][
        # #                 state.branches_inds["gb"]
        # #             ]
        # #             - state.branches_coords["gb"][state.branches_inds["gb"]]
        # #         )
        # #         > 0.0,
        # #         axis=-1,
        # #     )
        # #     proposed = gb_inds.get()
        # #     accepted_check = accepted_check_tmp.sum(
        # #         axis=(1, 2)
        # #     ) / proposed.sum(axis=(1, 2))
        # # else:
        # #     accepted_check_tmp = (
        # #         new_state.branches_inds["gb"] == (~state.branches_inds["gb"])
        # #     )

        # #     proposed = gb_inds.get()
        # #     accepted_check = accepted_check_tmp.sum(axis=(1, 2)) / proposed.sum(axis=(1, 2))

        # # manually tell temperatures how real overall acceptance fraction is
        # number_of_walkers_for_accepted = np.floor(nwalkers * accepted_check).astype(int)

        # accepted_inds = np.tile(np.arange(nwalkers), (ntemps, 1))

        # accepted = np.zeros((ntemps, nwalkers), dtype=bool)
        # accepted[accepted_inds < number_of_walkers_for_accepted[:, None]] = True

        # tmp1 = np.all(
        #     np.abs(
        #         new_state.branches_coords["gb"]
        #         - state.branches_coords["gb"]
        #     )
        #     > 0.0,
        #     axis=-1,
        # ).sum(axis=(2,))
        # tmp2 = new_state.branches_inds["gb"].sum(axis=(2,))

        # # add to move-specific accepted information
        # self.accepted += tmp1
        # if isinstance(self.num_proposals, int):
        #     self.num_proposals = tmp2
        # else:
        #     self.num_proposals += tmp2

        new_inds = cp.asarray(new_state.branches_inds["gb"])
        del band_sorter
        self.mempool.free_all_blocks()
        new_band_sorter = BandSorter(
            new_state.branches["gb"],
            self.band_edges,
            self.band_N_vals,
            force_backend=self.force_backend,
            transform_fn=self.parameter_transforms,
            max_data_store_size=self.max_data_store_size,
            gb=self.gb,
            gb_wdm_comp=self.gb_wdm_comp,
            waveform_kwargs=self.waveform_kwargs,
        )

        # in-model inds will not change
        tmp_freqs_find_bands = cp.asarray(new_state.branches_coords["gb"][:, :, :, 1])

        # calculate current band counts
        band_here = (
            cp.searchsorted(self.band_edges, tmp_freqs_find_bands.flatten() / 1e3, side="right") - 1
        ).reshape(tmp_freqs_find_bands.shape)

        group_temp_finder = [
            cp.repeat(cp.arange(ntemps), nwalkers * nleaves_max).reshape(
                ntemps, nwalkers, nleaves_max
            ),
            cp.tile(cp.arange(nwalkers), (ntemps, nleaves_max, 1)).transpose((0, 2, 1)),
            cp.tile(cp.arange(nleaves_max), ((ntemps, nwalkers, 1))),
        ]

        # TEMPERING
        self.temperature_control.swaps_accepted = np.zeros(ntemps - 1)
        self.temperature_control.swaps_proposed = np.zeros(ntemps - 1)

        self.mempool.free_all_blocks()

        self.time += 1
        # self.xp.cuda.runtime.deviceSynchronize()

        band_info = new_band_sorter.get_band_info()

        new_state.sub_states["gb"].update_band_information(
            _to_numpy(band_temps),
            _to_numpy(per_walker_band_proposals.sum(axis=1).T),
            _to_numpy(per_walker_band_accepted.sum(axis=1).T),
            _to_numpy(band_swaps_proposed),
            _to_numpy(band_swaps_accepted),
            band_info["band_counts"],
            self.is_rj_prop,
        )
        # TODO: check rj numbers

        # new_state.log_like[:] = self.check_ll_inject(new_state)

        self.mempool.free_all_blocks()
        new_state.log_like[:] = self.check_ll_inject(model, new_band_sorter)
        # if self.is_rj_prop:
        #     pass  # print(self.name, "2nd count check:", new_state.branches["gb"].inds.sum(axis=-1).mean(axis=-1), "\nll:", new_state.log_like[0] - orig_store, new_state.log_like[0])

        # new_state.log_prior[:] = model.compute_log_prior_fn(new_state.branches_coords, inds=new_state.branches_inds, supps=new_state.supplemental)
        accepted = np.zeros((ntemps, nwalkers), dtype=bool)
        
        return new_state, accepted

    def check_ll_inject(self, model, band_sorter, verbose=False):
        # breakpoint()
        init_like = model.analysis_container_arr.likelihood()
        model.analysis_container_arr.zero_out_data_arr()
        # Restore the non-GB residual snapshot per-shard inside each owning
        # device context. Matches the snapshot loop in propose().
        self._restore_linear_data_arr(
            model.analysis_container_arr, self.reset_non_gb_linear_data_arr
        )
        self.add_cold_chain_sources_to_residual(model, band_sorter, apply_inds=True)
        final_like = model.analysis_container_arr.likelihood()
        return final_like

    @staticmethod
    def _snapshot_linear_data_arr(aca):
        """Per-GPU shard copy of ``aca.linear_data_arr``. Returns a list of
        device-local buffers (one per entry in ``aca.linear_data_arr``).
        """
        if aca.gpus is None:
            return [b.copy() for b in aca.linear_data_arr]
        main_gpu = cp.cuda.runtime.getDevice()
        try:
            out = []
            for i, gpu in enumerate(aca.gpus):
                with cp.cuda.Device(int(gpu)):
                    out.append(aca.linear_data_arr[i].copy())
            return out
        finally:
            cp.cuda.runtime.setDevice(main_gpu)

    @staticmethod
    def _restore_linear_data_arr(aca, snapshot):
        """In-place restore of every shard from the matching snapshot entry."""
        if aca.gpus is None:
            for buf, snap in zip(aca.linear_data_arr, snapshot):
                buf[:] = snap[:]
            return
        main_gpu = cp.cuda.runtime.getDevice()
        try:
            for i, gpu in enumerate(aca.gpus):
                with cp.cuda.Device(int(gpu)):
                    aca.linear_data_arr[i][:] = snapshot[i][:]
        finally:
            cp.cuda.runtime.setDevice(main_gpu)
            
    @property
    def ranks_needed(self):
        if not hasattr(self, "_ranks_needed"):
            raise ValueError("Need to set ranks needed for this class.")

        return self._ranks_needed

    @ranks_needed.setter
    def ranks_needed(self, ranks_needed):
        assert isinstance(ranks_needed, int)
        self._ranks_needed = ranks_needed


class GBSpecialStretchMove(GBSpecialBase):
    """In-model GB stretch move with band-aware group proposals.

    On each call updates the per-leaf group structure (when due) and runs
    a stretch proposal restricted to GBs that share the same frequency
    band, so band-temperature swaps can interact correctly.
    """

    def setup(self, model, branches):
        for i, (name, branch) in enumerate(branches.items()):
            if name != "gb":
                continue

            if branch.inds[0].sum() >= self.nfriends and (
                (self.time % self.n_iter_update == 0) or (not self.has_setup_group)
            ):  # not self.is_rj_prop and
                self.setup_gbs(model, branch)

            # update any shifted start inds due to tempering (need to do this every non-rj move)
            """if not self.is_rj_prop:
                # fix the ones that have been added in RJ
                fix = (
                    branch.branch_supplemental.holder["friend_start_inds"][:] == -1
                ) & branch.inds

                if np.any(fix):
                    new_freqs = cp.asarray(branch.coords[fix][:, 1])
                    # TODO: is there a better way of doing this?

                    # fill information into friend finder for new binaries
                    branch.branch_supplemental.holder["friend_start_inds"][fix] = (
                        (
                            cp.searchsorted(self.freqs_sorted, new_freqs, side="right")
                            - 1
                        )
                        * (
                            (new_freqs > self.freqs_sorted[0])
                            & (new_freqs < self.freqs_sorted[-1])
                        )
                        + 0 * (new_freqs < self.freqs_sorted[0])
                        + (len(self.freqs_sorted) - 1)
                        * (new_freqs > self.freqs_sorted[-1])
                    ).get()

                # make sure current start inds reflect alive binaries
                self.current_friends_start_inds = self.xp.asarray(
                    branch.branch_supplemental.holder["friend_start_inds"][:]
                )
            """

            self.mempool.free_all_blocks()


class GBSpecialRJPriorMove(GBSpecialBase):
    """Reversible-jump GB move that draws proposals from the prior distribution."""
    pass


def para_log_like(
    x,
    gb,
    acs,
    walker_max,
    transform_fn,
    phase_maximize,
    waveform_kwargs,
    fstat=True,
    return_snr=False,
):
    """Vectorized GB log-likelihood used by serial-search and refit moves.

    Args:
        x: GB parameter rows (untransformed).
        gb: :class:`gbgpu.GBGPU` instance.
        acs: :class:`AnalysisContainerArray`.
        walker_max: Index of the walker whose data the proposals are scored
            against.
        transform_fn: :class:`TransformContainer` for GB parameters.
        phase_maximize: If ``True``, marginalize over phase.
        waveform_kwargs: Forwarded to ``gb.get_fstat_ll`` / ``gb.get_ll``.
        fstat: If ``True``, use the F-statistic likelihood (``get_fstat_ll``)
            and overwrite the amplitude / phase / iota / polarization
            entries of ``x`` with their maximized values.
        return_snr: If ``True`` and ``fstat`` is ``False``, also return the
            optimal SNR per row.

    Returns:
        Per-row log-likelihood (or ``(ll, snr)`` tuple).
    """
    xp = gb.backend.xp

    x_tmp = transform_fn.both_transforms(x, xp=xp)
    # need to get just f, fdot, fddot, lam, beta
    data_index = xp.full(x.shape[0], walker_max, dtype=xp.int32)
    if fstat:
        x_in = x_tmp[:, xp.array([1, 2, 3, 7, 8])]
        # TODO: fix for N>256?
        ll = gb.get_fstat_ll(
            x_in,
            acs.linear_data_arr,
            acs.linear_psd_arr,
            data_index=data_index,
            noise_index=data_index,
            data_length=acs.data_length,
            data_splits=np.array([gb.gpus[0]]),
            phase_marginalize=phase_maximize,
            return_cupy=True,
            N=512,  # 1024 is too much shared memory I think
            **waveform_kwargs,
        )

        x[:, 0] = np.log(gb.A_max)
        x[:, 3] = gb.phi0_max % (2 * np.pi)
        x[:, 4] = np.cos(gb.iota_max % (np.pi))
        x[:, 5] = gb.psi_max % (np.pi)

    else:
        # breakpoint()
        x_in = x_tmp[:]
        ll = gb.get_ll(
            x_in,
            acs.linear_data_arr,
            acs.linear_psd_arr,
            data_index=data_index,
            noise_index=data_index,
            data_length=acs.data_length,
            data_splits=np.array([gb.gpus[0]]),
            phase_marginalize=phase_maximize,
            return_cupy=True,
            # N=256,
            **waveform_kwargs,
        )
        # breakpoint()

        # params_remove_in = x_in.copy()
        # params_add_in = x_in.copy()

        # params_remove_in[:, 0] *= 1e-50
        # breakpoint()
        # ll_diff_2 = gb.swap_likelihood_difference(
        #     params_remove_in,
        #     params_add_in,
        #     acs.linear_data_arr,
        #     acs.linear_psd_arr,
        #     # start_freq_ind=self.xp.asarray(self.acs.start_freq_ind).astype(np.int32),
        #     data_index=data_index,
        #     noise_index=data_index,
        #     # N=N_vals,
        #     data_length=acs.data_length,
        #     data_splits=np.array([gb.gpus[0]]),
        #     phase_marginalize=phase_maximize,
        #     return_cupy=True,
        #     N=256,
        #     **waveform_kwargs,
        # )
        # breakpoint()

        if phase_maximize:
            x[:, 3] = (x[:, 3] - xp.angle(xp.asarray(gb.non_marg_d_h))) % (2 * np.pi)

        if return_snr:
            opt_snr = gb.h_h.real ** (1 / 2)
            return (ll, opt_snr)

    return ll


class PriorTransformFn:
    """Transform between unit-cube prior coordinates and GB :math:`(f, \\dot f)`.

    Used by the serial-search move to draw uniform-in-band proposals while
    keeping the rest of the GB prior unchanged.

    Args:
        f_min: Minimum frequency (Hz).
        f_max: Maximum frequency (Hz).
        fdot_min: Minimum frequency derivative.
        fdot_max: Maximum frequency derivative.
    """

    def __init__(self, f_min: float, f_max: float, fdot_min: float, fdot_max: float):
        self.f_min, self.f_max, self.fdot_min, self.fdot_max = (
            f_min,
            f_max,
            fdot_min,
            fdot_max,
        )

    def adjust_logp(self, logp, groups_running):
        """Add the (uniform) ``f`` and ``fdot`` log-density to ``logp``."""
        xp = get_array_module(self.f_min)

        if groups_running is None:
            groups_running = xp.arange(len(self.f_min))

        f_min_here = self.f_min[groups_running]
        f_max_here = self.f_max[groups_running]
        f_logpdf = np.log(1.0 / (f_max_here - f_min_here))

        fdot_min_here = self.fdot_min[groups_running]
        fdot_max_here = self.fdot_max[groups_running]
        fdot_logpdf = np.log(1.0 / (fdot_max_here - fdot_min_here))

        logp[:] += f_logpdf[:, None, None]
        logp[:] += fdot_logpdf[:, None, None]

        return logp

    def transform_to_prior_basis(self, coords, groups_running):
        """Map ``f`` / ``fdot`` columns of ``coords`` to the unit-cube basis."""
        xp = get_array_module(self.f_min)

        if groups_running is None:
            groups_running = xp.arange(len(self.f_min))

        f_min_here = self.f_min[groups_running]
        f_max_here = self.f_max[groups_running]
        try:
            coords[:, :, :, 1] = (coords[:, :, :, 1] - f_min_here[:, None, None]) / (
                f_max_here[:, None, None] - f_min_here[:, None, None]
            )
        except:
            breakpoint()

        fdot_min_here = self.fdot_min[groups_running]
        fdot_max_here = self.fdot_max[groups_running]
        coords[:, :, :, 2] = (coords[:, :, :, 2] - fdot_min_here[:, None, None]) / (
            fdot_max_here[:, None, None] - fdot_min_here[:, None, None]
        )

        return

    def transform_from_prior_basis(self, coords, groups_running):
        """Map ``f`` / ``fdot`` columns of ``coords`` from unit cube back to physical."""
        if groups_running is None:
            groups_running = xp.arange(len(self.f_min))

        assert groups_running.shape[0] == coords.shape[0]
        f_min_here = self.f_min[groups_running]
        f_max_here = self.f_max[groups_running]
        coords[:, :, :, 1] = (
            coords[:, :, :, 1] * (f_max_here[:, None, None] - f_min_here[:, None, None])
        ) + f_min_here[:, None, None]

        fdot_min_here = self.fdot_min[groups_running]
        fdot_max_here = self.fdot_max[groups_running]
        coords[:, :, :, 2] = (
            coords[:, :, :, 2] * (fdot_max_here[:, None, None] - fdot_min_here[:, None, None])
        ) + fdot_min_here[:, None, None]

        return


class BayesGMMFit:
    """Variational Bayesian GMM fit to per-leaf GB samples (sklearn ``BayesianGaussianMixture``).

    Stores the per-feature min/max so samples can be transformed in/out of
    a ``[-1, 1]`` GMM basis.

    Args:
        samples_in: 2D NumPy array of GB samples to fit.
    """

    def __init__(self, samples_in):

        assert isinstance(samples_in, np.ndarray)

        run = True
        min_bic = np.inf
        self.sample_mins = sample_mins = samples_in.min(axis=0)
        self.sample_maxs = sample_maxs = samples_in.max(axis=0)

        samples = self.transform_to_gmm_basis(samples_in)

        mixture = BayesianGaussianMixture(
            weight_concentration_prior_type="dirichlet_distribution",
            n_components=60,
            # reg_covar=0,
            # init_params="random",
            max_iter=5000,
            # mean_precision_prior=0.8,
            # random_state=random_state,
        )
        mixture.fit(samples)

        self.keep_mix = mixture

    def transform_to_gmm_basis(self, samples):
        """Map samples from physical to ``[-1, 1]`` GMM basis."""
        return (
            (samples - self.sample_mins[None, :])
            / (self.sample_maxs[None, :] - self.sample_mins[None, :])
        ) * 2 - 1

    def transform_from_gmm_basis(self, samples):
        """Map samples from ``[-1, 1]`` GMM basis back to physical."""
        return (samples + 1.0) / 2.0 * (
            self.sample_maxs[None, :] - self.sample_mins[None, :]
        ) + self.sample_mins[None, :]


from sklearn.mixture import GaussianMixture


class GMMFit:
    """Plain GMM fit to per-leaf GB samples (sklearn ``GaussianMixture``).

    Args:
        samples_in: 2D NumPy array of GB samples to fit.
    """

    def __init__(self, samples_in):

        assert isinstance(samples_in, np.ndarray)

        run = True
        min_bic = np.inf
        self.sample_mins = sample_mins = samples_in.min(axis=0)
        self.sample_maxs = sample_maxs = samples_in.max(axis=0)

        samples = self.transform_to_gmm_basis(samples_in)

        mixture = GaussianMixture(n_components=30, verbose=False, verbose_interval=2)

        mixture.fit(samples)

        # bad = False
        # for n_components in range(1, 31)[-1:]:
        #     if not run:
        #         continue
        #     #fit_gaussian_mixture_model(n_components, samples)
        #     #breakpoint()
        #     try:
        #         mixture = GaussianMixture(n_components=n_components, verbose=False, verbose_interval=2)

        #         mixture.fit(samples)
        #         test_bic = mixture.bic(samples)
        #     except ValueError:
        #         # print("ValueError", samples)
        #         run = False
        #         bad = True
        #         continue
        #     # print(n_components, test_bic)
        #     if test_bic < min_bic:
        #         min_bic = test_bic
        #         keep_mix = mixture
        #         keep_components = n_components

        #     else:
        #         run = False

        #         # print(leaf, n_components - 1, et - st)

        #     """if keep_components >= 9:
        #         new_samples = keep_mix.sample(n_samples=100000)[0]
        #         old_samples = samples
        #         fig = corner.corner(old_samples, hist_kwargs=dict(density=True, color="r"), color="r", plot_datapoints=False, plot_density=False)
        #         corner.corner(new_samples, hist_kwargs=dict(density=True, color="b"), color="b", plot_datapoints=False, plot_contours=True, plot_density=False, fig=fig)
        #         fig.savefig("mix_check.png")
        #         plt.close()
        #         breakpoint()"""

        # if bad:
        #     print("BAD")
        # if keep_components >= 19:
        #     print(keep_components)
        # # output_list = [keep_mix.weights_, keep_mix.means_, keep_mix.covariances_, np.array([np.linalg.inv(keep_mix.covariances_[i]) for i in range(len(keep_mix.weights_))]), np.array([np.linalg.det(keep_mix.covariances_[i]) for i in range(len(keep_mix.weights_))]), sample_mins, sample_maxs]

        self.keep_mix = mixture

    def transform_to_gmm_basis(self, samples):
        """Map samples from physical to ``[-1, 1]`` GMM basis."""
        return (
            (samples - self.sample_mins[None, :])
            / (self.sample_maxs[None, :] - self.sample_mins[None, :])
        ) * 2 - 1

    def transform_from_gmm_basis(self, samples):
        """Map samples from ``[-1, 1]`` GMM basis back to physical."""
        return (samples + 1.0) / 2.0 * (
            self.sample_maxs[None, :] - self.sample_mins[None, :]
        ) + self.sample_mins[None, :]


def gather_gmms(gmms):
    """Pack a list of GMM fits into the dict format expected by :func:`make_gmm`."""
    weights = []
    means = []
    covs = []
    inv_covs = []
    dets = []
    sample_mins = []
    sample_maxs = []

    for gmm in gmms:
        weights.append(gmm.keep_mix.weights_)
        means.append(gmm.keep_mix.means_)
        covs.append(gmm.keep_mix.covariances_)
        inv_covs.append(
            np.array(
                [
                    np.linalg.inv(gmm.keep_mix.covariances_[i])
                    for i in range(len(gmm.keep_mix.weights_))
                ]
            )
        )
        dets.append(
            np.array(
                [
                    np.linalg.det(gmm.keep_mix.covariances_[i])
                    for i in range(len(gmm.keep_mix.weights_))
                ]
            )
        )
        sample_mins.append(gmm.sample_mins)
        sample_maxs.append(gmm.sample_maxs)

    return (weights, means, covs, inv_covs, dets, sample_mins, sample_maxs)


from lisatools.sampling.gmm import vec_fit_gmm_min_bic

class GBSpecialRJSerialSearchMCMC(GBSpecialBase):
    """Reversible-jump GB move that runs a serial F-statistic MCMC search per band.

    Each band proposes one new GB at a time using a parallel ensemble
    sampler driven by :func:`para_log_like`, with proposals drawn from a
    band-restricted prior via :class:`PriorTransformFn`.
    """
    comm_info = None

    def get_rank_function(self):
        return gb_search_func

    def setup(self, model, branches):
        # FOR FAST TESTING/DEBUGGING
        # import pickle
        # with open("gmm_tmp.pickle", "rb") as fp:
        #     full_gmm = pickle.load(fp)

        # rj_dist = ProbDistContainer(
        #     {
        #         (r"$\log A$", r"$f_0$", r"$\dot{f}$", r"$\cos\iota$", r"$\lambda$", r"$\sin\beta$"): full_gmm,
        #         r"$\phi_0$": uniform_dist(0.0, 2 * np.pi),
        #         r"$\psi$": uniform_dist(0.0, np.pi),
        #     },
        #     use_cupy=True,
        # )
        # rj_dist.reset_key_order([r"$\log A$", r"$f_0$", r"$\dot{f}$", r"$\phi_0$", r"$\cos\iota$", r"$\psi$", r"$\lambda$", r"$\sin\beta$"])
        # return
        # run paraensemble MCMC.
        max_logl_walker = np.argmax(model.analysis_container_arr.likelihood()).item()
        self.gb.d_d = 0.0  # model.analysis_container_arr.inner_product()[max_logl_walker]
        ndim = branches["gb"].ndim
        nwalkers = 30  # TODO: adjustable
        ntemps = 24  # TODO: adjustable
        shutoff_band_iteration = 2
        priors_global = self.priors if not self.backend.uses_cuda else self.gpu_priors            
        
        if self.num_bands == 1:
            f0_max = self.band_edges[1:]
            f0_min = self.band_edges[:-1]
        else:
            f0_max = self.band_edges[2:-1]
            f0_min = self.band_edges[1:-2]
        
        # logic to shutoff bands #? Think about how this should change when we change SNR_thresh
        if self.num_proposals >= shutoff_band_iteration:
            bands_to_shutoff = np.all(~self.found_source_in_band[-shutoff_band_iteration:, :], axis=0)

            if np.all(bands_to_shutoff):
                print(f"No sources found across all bands for {shutoff_band_iteration} iterations, reverting to priors")
                self.rj_proposal_distribution = priors_global
                return
            
            else:
                shutoff_mask = ~bands_to_shutoff
                f0_max = f0_max[shutoff_mask]
                f0_min = f0_min[shutoff_mask]
                ngroups = np.sum(shutoff_mask)
            
        else:
            ngroups = max(1, self.num_bands - 2)
            assert f0_max.shape[0] == ngroups
            assert f0_min.shape[0] == ngroups
            bands_to_shutoff = None
    
        print(f"The current number of active bands is {ngroups}")

        # TODO: make this adjustable to match settings
        m_chirp_lims = [0.001, 1.2]
        fdot_max = get_fdot(f0_max, Mc=m_chirp_lims[-1])
        fdot_min = -fdot_max

        priors_in = deepcopy(priors_global)["gb"].priors_in
        priors_in[r"$f_0$"] = uniform_dist(0.0, 1.0, use_cupy=self.backend.uses_cupy)
        priors_in[r"$\dot{f}$"] = uniform_dist(0.0, 1.0, use_cupy=self.backend.uses_cupy)
        priors = {
            "gb": ProbDistContainer(priors_in, return_gpu=True, use_cupy=self.backend.uses_cupy)
        }
        # print(priors["gb"].key_order)
        # catalogue = gather_catalogue(
        #     "/sps/lisaf/crondeel/secret_sauce/GBgpu_sampler/data/Catalogue_Mojito_lite_wdwd_gbgpu_params_full.npy",
        #     self.band_edges[1:].min(),
        #     self.band_edges[:-1].max()  
        # )
        # truths_in_band = get_true_source_for_bands(self.band_edges[1:-1], catalogue, output_shape=(ngroups, ntemps, nwalkers, 9))
        # truths_in_band = self.xp.delete(truths_in_band, 3, axis=-1)
        # noised_truths = noise_parameters(truths_in_band, 0.00001)
        
        start_params = priors["gb"].rvs(size=(ngroups, ntemps, nwalkers))
        prior_transform_fn = PriorTransformFn(f0_min * 1e3, f0_max * 1e3, fdot_min, fdot_max)
        prior_transform_fn.transform_from_prior_basis(start_params, self.xp.arange(ngroups))
        
        # start_params[noised_truths[..., 0] < -1] = noised_truths[noised_truths[..., 0] < -1]

        print("phase maximizing here right now (?)")
        ll_args = (
            self.gb,
            model.analysis_container_arr,
            max_logl_walker,
            self.parameter_transforms,
            True,  # self.phase_maximize,
            self.waveform_kwargs,
        )

        ll_args_2 = (
            self.gb,
            model.analysis_container_arr,
            max_logl_walker,
            self.parameter_transforms,
            False,  # self.phase_maximize,
            self.waveform_kwargs,
        )

        # test_ll = para_log_like(
        #     test_params,
        #     *ll_args
        # )
        gibbs_sampling_setup = np.ones(8, dtype=bool)
        gibbs_sampling_setup[np.array([0, 3, 4, 5])] = False
        para_sampler = ParaEnsembleSampler(
            ndim,
            nwalkers,
            ngroups,
            para_log_like,
            priors,
            tempering_kwargs=dict(ntemps=ntemps, Tmax=np.inf),
            args=ll_args,
            # kwargs: dict = {},
            gpu=self.gb.gpus[0],
            periodic=self.periodic,
            # backend: ParaBackend = None,  # add ParaHDFBackend
            # update_fn: Callable = None,
            # update_iterations=-1,
            # stopping_fn: Callable = None,
            # stopping_iterations: int=-1,
            prior_transform_fn=prior_transform_fn,
            name="gb",
            gibbs_sampling_setup=gibbs_sampling_setup,
            # provide_supplemental=False,
        )

        from eryn.state import ParaState

        state = ParaState({"gb": start_params}, groups_running=self.xp.ones(ngroups, dtype=bool))
        state.log_prior = para_sampler.compute_log_prior(state.branches_coords)
        state.log_like = para_sampler.compute_log_like(state.branches_coords, logp=state.log_prior)

        nsteps = 500
        para_sampler.run_mcmc(state, nsteps, burn=500, progress=True)

        samples = self.xp.asarray(para_sampler.get_chain()[:, :, 0])
        check_ll = para_sampler.get_log_like()[:, :, 0]
        sample_ll = asnumpy(
            para_log_like(samples.reshape(-1, 8), *ll_args).reshape(samples.shape[:-1])
        )

        check_real_ll_phase_maximized = asnumpy(
            para_log_like(samples.reshape(-1, 8), *ll_args, fstat=False)
            .reshape(samples.shape[:-1])
        )
        check_real_ll, opt_snr = para_log_like(
            samples.reshape(-1, 8), *ll_args_2, fstat=False, return_snr=True
        )
        check_real_ll = asnumpy(check_real_ll.reshape(samples.shape[:-1]))
        opt_snr = asnumpy(opt_snr.reshape(samples.shape[:-1]))

        # np.save("opt_snr_from_ldc_parasampler_check.npy", opt_snr)
        # np.save("samples_from_ldc_parasampler_check.npy", samples)
        
        # TODO: make cut adjustable
        groups_running_now = opt_snr.min(axis=(0, 2)) > 8.0
        
        if self.num_proposals == 0:
            self.found_source_in_band = groups_running_now
        else:
            if bands_to_shutoff is None:
                self.found_source_in_band = np.vstack([self.found_source_in_band, groups_running_now])
            else:
                shutoff_temp = np.zeros(self.found_source_in_band.shape[1], dtype=bool)
                shutoff_temp[~bands_to_shutoff] = groups_running_now
                self.found_source_in_band = np.vstack([self.found_source_in_band, shutoff_temp])
        
        print(f"Found a source in {groups_running_now.sum()} out of {groups_running_now.shape[0]} active bands")
        if not np.any(groups_running_now):
            print("Did not find any new sources.")
            return

        start_params_2 = np.tile(samples[-1][groups_running_now, None], (1, ntemps, 1, 1))

        gibbs_sampling_setup_2 = np.ones(8, dtype=bool)
        gibbs_sampling_setup_2[np.array([3])] = False
        prior_transform_fn_2 = PriorTransformFn(
            f0_min[groups_running_now] * 1e3,
            f0_max[groups_running_now] * 1e3,
            fdot_min[groups_running_now],
            fdot_max[groups_running_now],
        )
        ngroups_2 = groups_running_now.sum().item()
        # prior_transform_fn_2.transform_from_prior_basis(start_params_2, self.xp.arange(ngroups_2))

        para_sampler_2 = ParaEnsembleSampler(
            ndim,
            nwalkers,
            ngroups_2,
            para_log_like,
            priors,
            tempering_kwargs=dict(ntemps=ntemps, Tmax=np.inf),
            args=ll_args,
            kwargs=dict(fstat=False),
            gpu=self.gb.gpus[0],
            periodic=self.periodic,
            # backend: ParaBackend = None,  # add ParaHDFBackend
            # update_fn: Callable = None,
            # update_iterations=-1,
            # stopping_fn: Callable = None,
            # stopping_iterations: int=-1,
            prior_transform_fn=prior_transform_fn_2,
            name="gb",
            gibbs_sampling_setup=gibbs_sampling_setup_2,
            # provide_supplemental=False,
        )

        new_state = ParaState(
            {"gb": start_params_2}, groups_running=self.xp.ones(ngroups_2, dtype=bool)
        )
        new_state.log_prior = para_sampler_2.compute_log_prior(new_state.branches_coords)
        new_state.log_like = para_sampler_2.compute_log_like(
            new_state.branches_coords, logp=new_state.log_prior
        )

        if np.any(np.isinf(new_state.log_prior)):
            breakpoint()

        nsteps = 500
        para_sampler_2.run_mcmc(new_state, nsteps, burn=500, progress=True)

        samples_2 = self.xp.asarray(para_sampler_2.get_chain()[:, :, 0])
        check_ll_2 = para_sampler_2.get_log_like()[:, :, 0]

        check_real_ll_phase_maximized_2 = asnumpy(
            para_log_like(samples_2.reshape(-1, 8), *ll_args, fstat=False)
            .reshape(samples_2.shape[:-1])
        )
        check_real_ll_2 = asnumpy(
            para_log_like(samples_2.reshape(-1, 8), *ll_args_2, fstat=False)
            .reshape(samples_2.shape[:-1])
        )

        # TODO: add removal of bands that consistently dont find things
        samples_2 = samples_2.transpose(1, 0, 2, 3)
        # np.save("/sps/lisaf/crondeel/secret_sauce/GBgpu_sampler/data/output_dir_ldc_highf_test/diagnostics/samples_ldc_run_3.npy", samples_2)

        st = time.perf_counter()
        samples_2_tmp = samples_2.reshape(samples_2.shape[0], -1, samples_2.shape[-1])[
            :, :, np.array([0, 1, 2, 4, 6, 7])
        ]
        full_gmm = vec_fit_gmm_min_bic(
            samples_2_tmp,
            min_comp=1,
            max_comp=30,
            n_samp_bic_test=5000,
            gpu=self.xp.cuda.runtime.getDevice(),
            verbose=False,
        )
        # import pickle
        # with open("gmm_tmp.pickle", "wb") as fp:
        #     pickle.dump(full_gmm, fp, pickle.HIGHEST_PROTOCOL)

        et = time.perf_counter()
        print(f"GPU GMM FIT: {et - st}")

        rj_dist = ProbDistContainer(
            {
                (r"$\log A$", r"$f_0$", r"$\dot{f}$", r"$\cos\iota$", r"$\lambda$", r"$\sin\beta$"): full_gmm,
                r"$\phi_0$": uniform_dist(0.0, 2 * np.pi),
                r"$\psi$": uniform_dist(0.0, np.pi),
            },
            use_cupy=True,
        )
        rj_dist.reset_key_order([r"$\log A$", r"$f_0$", r"$\dot{f}$", r"$\phi_0$", r"$\cos\iota$", r"$\psi$", r"$\lambda$", r"$\sin\beta$"])
        # if self.ranks_needed == 0:
        #     gmms = [GMMFit(samples_2[i].get().reshape(-1, 8)) for i in range(samples_2.shape[0])[:10]]
        #     gmm_info = gather_gmms(gmms)

        # else:
        #     if self.comm_info is None:
        #         # this only happens the first time through
        #         self.comm_info = self.comm.recv(tag=232342)

        #     gmm_info = fit_gmm(samples_2.get(), self.comm, self.comm_info)

        # full_gmm = FullGaussianMixtureModel(*gmm_info, use_cupy=self.use_gpu)
        # breakpoint()

        # gen_samp = self.xp.asarray(rj_dist.rvs(1000))
        # gen_ll, gen_opt_snr = para_log_like(gen_samp, *ll_args, fstat=False, return_snr=True)
        # print(gen_ll, self.gb.d_h / gen_opt_snr, gen_opt_snr)
        # breakpoint()
        self.rj_proposal_distribution = {"gb": rj_dist}
  
    

class GBSpecialRJSearchMove(GBSpecialBase): #? only needed for mutli GPU usage
    """Reversible-jump GB search move that delegates work to extra GPU/MPI ranks.

    # TODO/DOCS: full multi-GPU coordination protocol with the worker
    function :func:`gb_search_func` — used when sufficient ranks are
    available so the bulk search runs concurrently with PE.
    """
    def get_rank_function(self):
        return gb_search_func

    def setup(self, model, branches):
        self.interact_with_search()
        super(GBSpecialRJSearchMove, self).setup(branches) # should be serial move?

    def interact_with_search(self):
        search_rank = self.ranks[0]

        search_ch = self.comm.irecv(source=search_rank)
        if search_ch.get_status():
            search_req = search_ch.wait()

            if "receive" in search_req and search_req["receive"]:
                search_dict = self.comm.recv(source=search_rank)
                self.rj_proposal_distribution["gb"] = make_gmm(self.gb, search_dict["search"])

            if "send" in search_req and search_req["send"]:
                # get random instance of residual, psd, lisasens
                # TODO: decide about random versus max ll
                random_ind = np.random.randint(self.nwalkers)

                data = [
                    asnumpy(self.mgh.data_shaped[0][0][random_ind]),
                    asnumpy(self.mgh.data_shaped[1][0][random_ind]),
                ]
                psd = [
                    asnumpy(self.mgh.psd_shaped[0][0][random_ind]),
                    asnumpy(self.mgh.psd_shaped[1][0][random_ind]),
                ]
                lisasens = [
                    asnumpy(self.mgh.psd_shaped[0][0][random_ind]),
                    asnumpy(self.mgh.lisasens_shaped[1][0][random_ind]),
                ]

                output_data = dict(data=data, psd=psd, lisasens=lisasens)
                self.comm.send(output_data, dest=search_rank)

        else:
            search_ch.cancel()

        print("CHECK INSIDE PROP")


from lisatools.globalfit.gathergalaxy import gather_gb_samples
from lisatools.globalfit.hdfbackend import GBHDFBackend, GFHDFBackend, MBHHDFBackend
from lisatools.globalfit.state import GBState


class GBSpecialRJRefitMove(GBSpecialBase):
    """Reversible-jump GB move that uses GMM-refitted proposals.

    Loads per-leaf GMM proposals (refit by :func:`gb_refit_func`) and uses
    them as the RJ proposal distribution. This is typically alternated
    with :class:`GBSpecialRJPriorMove` to sharpen accepted GB candidates.
    """
    def __init__(self, *args, fp=None, **kwargs):
        assert fp is not None and isinstance(fp, str)
        assert os.path.exists(fp)
        self.fp = fp
        GBSpecialBase.__init__(self, *args, **kwargs)

    def setup(self, model, branches):
        # FOR FAST TESTING/DEBUGGING
        # import pickle
        # with open("gmm_tmp.pickle", "rb") as fp:
        #     full_gmm = pickle.load(fp)

        # rj_dist = ProbDistContainer(
        #     {
        #         (r"$\log A$", r"$f_0$", r"$\dot{f}$", r"$\cos\iota$", r"$\lambda$", r"$\sin\beta$"): full_gmm,
        #         r"$\phi_0$": uniform_dist(0.0, 2 * np.pi),
        #         r"$\psi$": uniform_dist(0.0, np.pi),
        #     },
        #     use_cupy=True,
        # )
        # rj_dist.key_order = [r"$\log A$", r"$f_0$", r"$\dot{f}$", r"$\phi_0$", r"$\cos\iota$", r"$\psi$", r"$\lambda$", r"$\sin\beta$"]
        # self.rj_proposal_distribution = {"gb": rj_dist}
        # return
        # run paraensemble MCMC.

        max_logl_walker = np.argmax(model.analysis_container_arr.likelihood()).item()
        self.gb.d_d = 0.0  # model.analysis_container_arr.inner_product()[max_logl_walker]
        reader = GFHDFBackend(
            self.fp, sub_state_bases={"gb": GBState}, sub_backend={"gb": GBHDFBackend}
        )

        st = time.perf_counter()
        sens_mat = model.analysis_container_arr[max_logl_walker].sens_mat
        samples_keep = 5
        if reader.iteration < 2 * samples_keep:
            return

        num_compare_samples = 1
        nwalkers = 30
        gpu = self.xp.cuda.runtime.getDevice() if self.backend.uses_cupy else -1
        # ``gather_gb_samples`` builds FD waveforms for the GMM refit;
        # the WDM equivalent has not been wired in yet.
        if not isinstance(model.analysis_container_arr.settings, FDSettings):
            raise NotImplementedError(
                "GBSpecialRJRefitMove currently requires the FD basis "
                "(GMM refit fits FD waveforms)."
            )
        fd = model.analysis_container_arr.f_arr.copy()
        groups = gather_gb_samples(
            fd,
            self.parameter_transforms,
            self.gb,
            self.waveform_kwargs.copy(),
            self.band_edges,
            self.band_N_vals,
            reader,
            sens_mat,
            gpu,
            num_compare_samples=num_compare_samples,
            samples_keep=samples_keep,
            thin_by=1,
        )

        num_in_groups = np.asarray([len(tmp) for tmp in groups])
        keep = num_in_groups > nwalkers * samples_keep / 2

        max_num_source = max([tmp.shape[0] for tmp in groups])
        samples = np.full((len(groups), max_num_source, groups[0].shape[-1]), np.nan)
        for i, group in enumerate(groups):
            samples[i, : len(group)] = group

        samples_fin = samples[keep]
        num_in_groups_fin = num_in_groups[keep]
        cp.cuda.runtime.setDevice(gpu)
        output_info = []
        step = 5
        steps = np.arange(num_in_groups_fin.min(), num_in_groups_fin.max(), step)
        if steps[-1] < num_in_groups_fin.max().item():
            steps = np.concatenate([steps, np.array([num_in_groups_fin.max().item()])])

        weights_all = []
        means_all = []
        covs_all = []
        invcovs_all = []
        dets_all = []
        mins_all = []
        maxs_all = []
        for start, end in zip(steps[:-1], steps[1:]):
            here = (num_in_groups_fin >= start) & (num_in_groups_fin < end)
            # this randomly throughs away ~step amount of samples to make gmm work
            samples_here = samples_fin[here][:, :start, np.array([0, 1, 2, 4, 6, 7])].copy()
            weights, means, covs, invcovs, dets, mins, maxs = vec_fit_gmm_min_bic(
                cp.asarray(samples_here),
                min_comp=1,
                max_comp=30,
                n_samp_bic_test=5000,
                gpu=gpu,
                verbose=False,
                return_components=True,
            )
            weights_all += weights
            means_all += means
            covs_all += covs
            invcovs_all += invcovs
            dets_all += dets
            mins_all += mins
            maxs_all += maxs
            print(start, end)

        full_gmm = FullGaussianMixtureModel(
            weights_all,
            means_all,
            covs_all,
            invcovs_all,
            dets_all,
            mins_all,
            maxs_all,
            use_cupy=True,
        )

        print("time:", time.perf_counter() - st)
        rj_dist = ProbDistContainer(
            {
                (r"$\log A$", r"$f_0$", r"$\dot{f}$", r"$\cos\iota$", r"$\lambda$", r"$\sin\beta$"): full_gmm,
                r"$\phi_0$": uniform_dist(0.0, 2 * np.pi),
                r"$\psi$": uniform_dist(0.0, np.pi),
            },
            use_cupy=True,
        )
        rj_dist.key_order = [r"$\log A$", r"$f_0$", r"$\dot{f}$", r"$\phi_0$", r"$\cos\iota$", r"$\psi$", r"$\lambda$", r"$\sin\beta$"]
        # if self.ranks_needed == 0:
        #     gmms = [GMMFit(samples_2[i].get().reshape(-1, 8)) for i in range(samples_2.shape[0])[:10]]
        #     gmm_info = gather_gmms(gmms)

        # else:
        #     if self.comm_info is None:
        #         # this only happens the first time through
        #         self.comm_info = self.comm.recv(tag=232342)

        #     gmm_info = fit_gmm(samples_2.get(), self.comm, self.comm_info)

        # full_gmm = FullGaussianMixtureModel(*gmm_info, use_cupy=self.use_gpu)
        # breakpoint()

        # gen_samp = self.xp.asarray(rj_dist.rvs(1000))
        # gen_ll, gen_opt_snr = para_log_like(gen_samp, *ll_args, fstat=False, return_snr=True)
        # print(gen_ll, self.gb.d_h / gen_opt_snr, gen_opt_snr)
        # breakpoint()

        self.rj_proposal_distribution = {"gb": rj_dist}


def get_param_limits(array): # can be used for debugging of coordinate values
    """Return per-column min/max of ``array`` (debug helper for GB coordinates)."""
    num_params = array.shape[-1]
    
    if num_params == 8:
        param_labels = [r"$\log A$", r"$f_0$", r"$\dot{f}$", r"$\phi_0$", r"$\cos\iota$", r"$\psi$", r"$\lambda$", r"$\sin\beta$"]
    if num_params == 9:
        param_labels = [r"$\log A$", r"$f_0$", r"$\dot{f}$", r"$\ddot{f}$", r"$\phi_0$" r"$\cos\iota$", r"$\psi$", r"$\lambda$", r"$\sin\beta$"]
    
    for i, param_label in enumerate(param_labels):
        param_values = array[..., i]
        min_array_i = param_values.min()
        max_array_i = param_values.max()
        print(f"For parameter {param_label}, the minimun value is {min_array_i}, the maximum value is {max_array_i}")
        print(f"The mean of {param_label} is {cp.mean(param_values)} with a std of {cp.std(param_values)}")