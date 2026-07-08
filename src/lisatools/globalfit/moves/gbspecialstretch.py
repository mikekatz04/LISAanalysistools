"""Galactic-binary specialized stretch / RJ moves and supporting infrastructure."""

from __future__ import annotations

import os
import time
import logging
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
from gbgpu.gb_likelihood import (
    BandLikelihoodEngine,
    FDBandLikelihoodEngine,
    SwapLLResult,
    WDMBandLikelihoodEngine,
    make_band_likelihood_engine,
)
from .globalfitmove import GFCombineMove, GlobalFitMove
from ..priors.gbpriors import get_fdot_mojito

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

logger = logging.getLogger(__name__)

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
    logger.info(f"INSIDE GB search, RANK: {comm.Get_rank()}")
    rank = comm.Get_rank()
    rank_index = class_ranks_list.index(rank)
    if rank_index == 0:
        comm_info = {"process_ranks_for_fit": class_ranks_list}
        logger.info("waiting to send process ranks")
        comm.send(comm_info, dest=main_rank, tag=232342)
        logger.info("sent process ranks")

    fit_each_leaf(rank, curr, main_rank, comm)

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
    logger.info("BREAKS", breaks)
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
                logger.info(
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
                            logger.warning("BAD error on return")
                            continue
                        if "BAD" in output_info:
                            current_status[proc_i] = False
                            logger.warning("BAD", output_info["BAD"])
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
    logger.info(f"INSIDE GB refit, RANK: {comm.Get_rank()}")
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


# Band-level infrastructure now lives in gbbands.py; re-exported here so
# existing ``from ...gbspecialstretch import Buffer / BandSorter`` imports
# keep working.
from .gbbands import (
    BandScheduler,
    BandSorter,
    Buffer,
    SubBandBuffer,
    pack_special_index,
    return_x,
    unpack_special_index,
)


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
        acs: :class:`AnalysisContainerArray` used for SETUP ONLY: the
            basis-domain settings on ``acs.settings`` drive every
            domain-dependent choice (FD vs WDM) and the initial parent
            binding. It is not stored -- at run time everything reads
            from / fills into the ACA that arrives with the model in
            :meth:`propose` (re-binding if it changed).
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
            instance. Required when ``acs.settings`` is a
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
        acs,
        band_edges,
        band_N_vals,
        gpu_priors,
        *args,
        waveform_kwargs={},
        parameter_transforms: Optional[TransformContainer] = None,
        snr_lim=1e-10,
        rj_proposal_distribution=None,
        is_rj_prop=False,
        num_repeat_proposals=100,
        name=None,
        use_prior_removal=False,
        phase_maximize=False,
        ranks_needed=0,
        gpus=[],
        num_band_preload=20000,
        run_swaps=True,
        max_data_store_size=6000,
        force_backend=None,
        gb_wdm_comp=None,
        gb_fd_comp=None,
        orbits=None,
        tdi_config=None,
        t_ref=0.0,
        search_kwargs=None,
        stretch_probability=0.5,
        band_units=2,
        jump_factor=0.005,
        debug=False,
        debug_plot_dir="./gf_output/gb_debug/",
        debug_plot_walker=0,
        debug_plot_band=None,
        **kwargs,
    ):
        # return_gpu is a kwarg for the stretch move
        LISAToolsParallelModule.__init__(self, force_backend=force_backend)
        GlobalFitMove.__init__(self, name=name)
        Move.__init__(self, *args, return_gpu=True, **kwargs)

        # Stretch-move state. GroupStretchMove.__init__ is not chained (the
        # MRO would double-init Move), so the pieces its get_proposal /
        # get_new_points read are set explicitly here.
        self.a = float(kwargs.get("a", 2.0))
        self.nfriends = int(kwargs.get("nfriends", 32))
        self.return_gpu = True
        self.use_gpu = self.backend.uses_cupy

        self.force_backend = force_backend
        self.ranks_needed = ranks_needed
        self.gpus = gpus
        self.gpu_priors = gpu_priors
        self.num_repeat_proposals = num_repeat_proposals
        # ``n_subbands`` is the user-facing alias for the number of
        # (temp, walker, band) cells held in the sub-band buffer at once.
        if kwargs.get("n_subbands") is not None:
            num_band_preload = int(kwargs["n_subbands"])
        self.num_band_preload = self.n_subbands = num_band_preload
        self.band_preload_size = self.max_data_store_size = max_data_store_size
        self.use_prior_removal = use_prior_removal
        self.has_setup_group = False

        # GB-sampler verification instrumentation. When ``debug`` is on, the
        # ``_debug_*`` hooks below run band residual round-trip / get_ll
        # consistency checks and dump begin/middle/end band plots at the real
        # operation sites in ``run_proposal``. Off by default -> the hooks
        # early-return, so the production path is untouched.
        self.debug = bool(debug)
        self.debug_plot_dir = debug_plot_dir
        # Plot selection: only the chosen (walker, band) cell is plotted --
        # ONE figure per plotted step with a panel per temperature -- instead
        # of a PNG for every (temp, walker, band) the proposal touches.
        # ``debug_plot_band=None`` resolves to the central band at plot time.
        self.debug_plot_walker = int(debug_plot_walker)
        self.debug_plot_band = (None if debug_plot_band is None
                                else int(debug_plot_band))
        self._dbg_plot_counter = 0

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

        if self.backend.uses_cupy:
            self.mempool = self.xp.get_default_memory_pool()
        else:
            # TODO: add a NoOpMempool setup to the backend itself
            self.mempool = _NoOpMempool()

        self.band_edges = band_edges
        self.num_bands = len(band_edges) - 1
        self.start_freq_ind = start_freq_ind
        self.data_length = data_length
        self.waveform_kwargs = waveform_kwargs
        self.parameter_transforms = parameter_transforms
        # NOTE: the constructor ACA (``acs``) is used for SETUP ONLY (domain
        # dispatch, shapes, initial parent binding). It is NOT stored: every
        # run-time fill / likelihood targets the ACA that arrives with the
        # model in :meth:`propose`, which re-binds the parent engine whenever
        # that ACA differs from the one currently bound.

        self._configure_domain(acs)
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
        self.search_kwargs = search_kwargs
        # In-model proposal mix: probability of drawing the band-aware group
        # stretch instead of the info-matrix Cholesky jump per repeat round.
        # The default in_model_proposal() is the overridable hook consuming
        # this; subclass it for other proposal components.
        self.stretch_probability = float(stretch_probability)
        # Band parity stride: 2 = odds/evens (minimum separation); larger
        # values give wider separation between simultaneously-updated bands.
        self.band_units = max(1, int(band_units))
        # Info-matrix jump scale (Gaussian draw through the Cholesky factor).
        self.jump_factor = float(jump_factor)
        self._fdot_scale = 1e-16

        # Parent binding: config-only FD comp + move-level engine. A user-
        # supplied gb_fd_comp is honored as-is; otherwise one is built from
        # the ACA's FDSettings -- data holders are passed to it at call
        # time, so ACA changes only rewire the engine (see _bind_parent_acs,
        # re-invoked from propose()).
        self.transform_fn = self.parameter_transforms
        self._gb_fd_comp_user_supplied = gb_fd_comp is not None
        self.gb_fd_comp = gb_fd_comp
        self._proposal_orbits = orbits
        self._proposal_tdi_config = tdi_config
        self._t_ref = float(t_ref)
        self._bind_parent_acs(acs)


    def _configure_domain(self, acs) -> None:
        """Derive ``self.fd`` / ``self.df`` / ``self._basis_settings`` from an ACA.

        The band-index math uses ``df = 1 / Tobs`` consistently across FD
        and WDM (FD: equals ``acs.df``; WDM: ``acs.df == layer_df`` differs,
        so we recompute). ``self.fd`` is only meaningful in the FD path.
        """

        # TODO: make this more generic
        if isinstance(acs.settings, FDSettings):
            self.fd = acs.f_arr.copy()
            self.df = float(self.fd[1] - self.fd[0])
        elif isinstance(acs.settings, WDMSettings):
            self.fd = None
            self.df = 1.0 / acs.settings.Tobs
        else:
            raise NotImplementedError(
                f"GBSpecialBase does not support basis domain "
                f"{type(acs.settings).__name__}."
            )
        self._basis_settings = acs.settings

    def _bind_parent_acs(self, acs) -> None:
        """(Re)bind the parent-level engine to ``acs``.

        Every run-time fill / likelihood targets the ACA that arrives with
        the model in :meth:`propose`; the constructor ACA only provides the
        initial binding. Post-2026-07 the FD comp is CONFIG-ONLY
        (``GBFDComputations(fd_settings, t_ref, ...)``) -- data holders are
        passed at get_ll/fill_global time, so an ACA change only refreshes
        the engine wiring here, never any C-side data pointers.
        """
        token = (id(acs), id(acs.linear_data_arr[0]))
        if getattr(self, "_parent_acs_token", None) == token:
            return

        # The parent is itself a window of the global rfft grid starting at
        # ``start_freq_ind``; the engine's bounds mask uses this when the
        # holder itself can't resolve per-row starts.
        self._parent_start_inds = self.xp.full(
            int(acs.acs_total_entries), int(self.start_freq_ind),
            dtype=self.xp.int32,
        )

        if isinstance(acs.settings, FDSettings):
            if self._gb_fd_comp_user_supplied:
                if getattr(self, "_parent_acs_token", None) is not None:
                    logger.warning(
                        f"{self.name}: parent ACA changed but gb_fd_comp was "
                        "user-supplied; keeping the user's comp."
                    )
            elif (
                self.gb_fd_comp is None
                # Rebuild if the grid config drifted (band re-slice etc.);
                # cheap, no data pointers involved.
                or self.gb_fd_comp.df != float(self._basis_settings.df)
                or self.gb_fd_comp.ind_min != int(self._basis_settings.ind_min)
                or self.gb_fd_comp.ind_max != int(self._basis_settings.ind_max)
            ):
                # TODO: check this a little further
                from gbgpu.gbcomps import GBFDComputations

                if self._proposal_tdi_config is None:
                    raise ValueError(
                        "FD-basis GB moves need tdi_config= (and orbits=) to "
                        "build the GBFDComputations comp, or pass "
                        "gb_fd_comp= directly."
                    )
                orbits_in = (
                    self._proposal_orbits
                    if self._proposal_orbits is not None
                    else getattr(self.gb, "orbits", None)
                )
                self.gb_fd_comp = GBFDComputations(
                    self._basis_settings,
                    self._t_ref,
                    N_sparse=int(self.band_N_vals.max()),
                    orbits=orbits_in, tdi_config=self._proposal_tdi_config,
                    force_backend=self.force_backend,
                    d_d=0.0,
                    tdi_type=self.waveform_kwargs.get(
                        "tdi_channel_setup", "XYZ"),
                    nchannels=acs.nchannels,
                )

        # Move-level engine for parent-residual fills (cold-chain
        # open/close). Same domain dispatch as the sub-band buffer's engine;
        # the parent ACA has no per-slot ``min_freq_inds``, so the FD engine
        # falls back to ``start_freq_inds`` (one shared window start per
        # walker row).
        self._likelihood_engine = make_band_likelihood_engine(
            self._basis_settings,
            gb=self.gb,
            gb_fd_comp=self.gb_fd_comp,
            gb_wdm_comp=self.gb_wdm_comp,
            nchannels=acs.nchannels,
            tdi_channel_setup=self.waveform_kwargs.get("tdi_channel_setup"),
            df=self.df,
            start_freq_inds=self._parent_start_inds,
            data_length=acs.data_length,
        )
        self._parent_acs_token = token

    def setup(self, model, branches):
        return

    @classmethod
    def supported_backends(cls):
        return ["lisatools_" + _tmp for _tmp in cls.GPU_RECOMMENDED()]

    def find_friends(self, name, s, s_inds=None, branch_supps=None):
        """Complement points for the band-aware group stretch.

        The friends were drawn (one per proposed source) by
        :meth:`BandSorter.draw_friends` immediately before the
        ``GroupStretchMove.get_proposal`` call in :meth:`in_model_proposal`;
        this override just reshapes them to match ``s``
        ``(1, n_sources, 1, ndim)``.
        """
        friends = self._friends_for_stretch
        assert friends.shape[0] == s.shape[1]
        return friends[None, :, None, :]

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
            f_bin = (params_in[:, 1] / self.df).astype(int) - int(self.start_freq_ind)
            assert not np.any(
                f_bin + (N_vals_in / 2) > model.analysis_container_arr.data_length
            ), "cold-chain source window exceeds the parent data range"
            assert not np.any(f_bin - (N_vals_in / 2) < 0), (
                "cold-chain source window falls below the parent data range"
            )

        # ``coords_in`` is already in physical units; dispatch via the engine.
        self._likelihood_engine.fill_template(
            model.analysis_container_arr,
            params_in,
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

    # ================= GB-sampler verification (debug mode) =================
    # Each hook does ONE piece at the site where that operation happens in
    # run_proposal; all early-return unless ``self.debug`` is set (so the
    # production path is untouched). Together they realize the residual
    # round-trip the sampler relies on: load the neighbour cold-chain residual
    # -> the band buffer holds that walker's signals -> remove the source being
    # proposed -> get_ll -> add it back -> ll, verifying the single-source
    # get_ll is consistent with both the swap_ll the sampler scores and the
    # full residual change, plus begin/middle/end band plots.

    def _debug_cold_chain_residual_loaded(self, model, remainder) -> None:
        """At the cold-chain residual load site: log the neighbour cold-chain
        residual baseline ll for this band unit ("load neighbouring residual")."""
        if not self.debug:
            return
        try:
            ll_np = _to_numpy(model.analysis_container_arr.likelihood())
            logger.info(
                "[GB_DEBUG %s] cold-chain residual loaded (remainder=%s): "
                "sum ll = %.6e over %d walker(s)",
                self.name, remainder, float(np.sum(ll_np.real)), ll_np.size,
            )
        except Exception as e:  # debug-only: never break the sampler
            logger.warning("[GB_DEBUG %s] cold-chain snapshot skipped: %r", self.name, e)

    def _debug_verify_rj_step(self, buffer_obj, params, alive, slots, N_vals,
                              delta_ll, keep, picked, round_i, scheduler) -> None:
        """At the RJ scoring site: independently re-verify the birth/death
        deltas through get_add_ll / get_removal_ll, check the removal
        identity ``<r+h|h> = <r|h> + <h|h>`` by an actual residual
        round-trip, and plot the band on the first rounds."""
        if not self.debug:
            return
        try:
            xp = self.xp
            k = _to_numpy(keep).astype(bool)
            if not k.any():
                return
            births = k & ~_to_numpy(alive)
            deaths = k & _to_numpy(alive)

            if births.any():
                b = xp.asarray(np.where(births)[0])
                check = buffer_obj.get_add_ll(params[b], slots[b], slots[b], N_vals[b])
                lhs = delta_ll[b]
                fin = xp.isfinite(check) & (lhs > -1e290)
                if bool(fin.any()):
                    relmax = float(xp.max(xp.abs(lhs[fin] - check[fin])
                                          / xp.maximum(xp.abs(check[fin]), 1.0)))
                    logger.info(
                        "[GB_DEBUG %s] RJ birth delta vs get_add_ll: max rel %.3e",
                        self.name, relmax,
                    )
                d_h_b = buffer_obj.d_h_out.copy()
                h_h_b = buffer_obj.h_h_out.copy()
                self._debug_residual_round_trip(
                    buffer_obj, params[b], slots[b], N_vals[b], d_h_b, h_h_b
                )

            if deaths.any():
                d = xp.asarray(np.where(deaths)[0])
                check = buffer_obj.get_removal_ll(params[d], slots[d], slots[d], N_vals[d])
                lhs = delta_ll[d]
                fin = xp.isfinite(check) & (lhs > -1e290)
                if bool(fin.any()):
                    relmax = float(xp.max(xp.abs(lhs[fin] - check[fin])
                                          / xp.maximum(xp.abs(check[fin]), 1.0)))
                    logger.info(
                        "[GB_DEBUG %s] RJ death delta vs get_removal_ll: max rel %.3e",
                        self.name, relmax,
                    )
                # Removal identity: restore the template (factor +1), then
                # <r+h|h> must equal <r|h> + <h|h>. Restored in ``finally``.
                d_h_1 = buffer_obj.d_h_out.copy()
                h_h_1 = buffer_obj.h_h_out.copy()
                eng = buffer_obj._likelihood_engine
                params_phys = self.transform_fn.both_transforms(params[d], xp=cp)
                di = slots[d].astype(xp.int32)
                eng.fill_template(buffer_obj, params_phys, di, N_vals[d],
                                  factor=+1, waveform_kwargs=self.waveform_kwargs)
                try:
                    buffer_obj.get_ll(params[d], slots[d], slots[d], N_vals[d])
                    expected = (d_h_1 + h_h_1).real
                    relmax = float(xp.max(xp.abs(buffer_obj.d_h_out.real - expected)
                                          / xp.maximum(xp.abs(expected), 1.0)))
                    logger.info(
                        "[GB_DEBUG %s] removal identity <r+h|h>=<r|h>+<h|h>: max rel %.3e",
                        self.name, relmax,
                    )
                finally:
                    eng.fill_template(buffer_obj, params_phys, di, N_vals[d],
                                      factor=-1, waveform_kwargs=self.waveform_kwargs)

            if round_i in (0, 1):
                map_cpu = (
                    _to_numpy(picked["temp_inds"]),
                    _to_numpy(picked["walker_inds"]),
                    _to_numpy(picked["band_inds"]),
                )
                self._debug_plot_band(buffer_obj, params, slots, N_vals,
                                      delta_ll, map_cpu, keep, round_i,
                                      stage="rj")
        except Exception as e:  # debug-only: never break the sampler
            logger.warning("[GB_DEBUG %s] verify_rj_step skipped: %r", self.name, e)

    def _debug_verify_in_model(self, buffer_obj, curr, new, slots, N_vals,
                               delta_ll, keep, map_cpu, move_i) -> None:
        """At the in-model repeat site: on the first repeat run the residual
        round-trip on the current source (the buffer holds the source-free
        residual) and plot the band at begin/middle/end of the repeats."""
        if not self.debug:
            return
        try:
            if move_i == 0:
                buffer_obj.get_ll(curr, slots, slots, N_vals)
                d_h_c = buffer_obj.d_h_out.copy()
                h_h_c = buffer_obj.h_h_out.copy()
                self._debug_residual_round_trip(
                    buffer_obj, curr, slots, N_vals, d_h_c, h_h_c
                )
            if move_i in (0, self.num_repeat_proposals // 2,
                          max(self.num_repeat_proposals - 1, 0)):
                self._debug_plot_band(buffer_obj, new, slots, N_vals, delta_ll,
                                      map_cpu, keep, move_i, stage="in-model")
        except Exception as e:  # debug-only: never break the sampler
            logger.warning("[GB_DEBUG %s] verify_in_model skipped: %r", self.name, e)

    def _debug_residual_round_trip(self, buffer_obj, params_add, data_index,
                                   swap_N_vals, d_h_a, h_h_a) -> None:
        """Add the proposed template to the band residual (factor=-1), confirm
        the get_ll shift equals -<h|h>, then remove it (factor=+1) so the buffer
        is restored exactly. Restoration is in a ``finally`` so the live sampler
        state is never left perturbed."""
        if not self.debug:
            return
        xp = self.xp
        eng = buffer_obj._likelihood_engine
        params_phys = self.transform_fn.both_transforms(params_add, xp=cp)
        di = data_index.astype(xp.int32)
        eng.fill_template(buffer_obj.acs_buffer, params_phys, di, swap_N_vals,
                          factor=-1, waveform_kwargs=self.waveform_kwargs)
        try:
            buffer_obj.get_ll(params_add, data_index, data_index, swap_N_vals)
            d_h2 = xp.asarray(buffer_obj.d_h_out).real
            # residual r' = r - h_add  =>  <r'|h_add> = d_h_a - h_h_a
            expected = (xp.asarray(d_h_a) - xp.asarray(h_h_a)).real
            finite = xp.isfinite(d_h2) & xp.isfinite(expected)
            if bool(xp.any(finite)):
                diff = xp.abs(d_h2[finite] - expected[finite])
                scale = xp.maximum(xp.abs(expected[finite]), 1.0)
                relmax = float(xp.max(diff / scale))
                logger.info(
                    "[GB_DEBUG %s] residual add/remove round-trip: max rel diff = %.3e",
                    self.name, relmax,
                )
        finally:
            eng.fill_template(buffer_obj.acs_buffer, params_phys, di, swap_N_vals,
                              factor=+1, waveform_kwargs=self.waveform_kwargs)

    def _debug_seq_select(self, buffer_obj, t_i, w_i, b_i, slots, curr):
        """Pick the entry of this repeat batch to trace with the 3x3
        sequence figures: the chosen (walker, band) cell at its coldest
        temperature present, once per sampler step. Returns None when the
        cell is absent, tracing is off, or it already ran this step."""
        if not self.debug or getattr(self, "_dbg_seq_done", True):
            return None
        try:
            sel_w = self.debug_plot_walker
            sel_b = (self.debug_plot_band if self.debug_plot_band is not None
                     else (len(self.band_edges) - 1) // 2)
            w_np = _to_numpy(w_i); b_np = _to_numpy(b_i); t_np = _to_numpy(t_i)
            match = np.where((w_np == sel_w) & (b_np == sel_b))[0]
            if match.size == 0:
                return None
            idx = int(match[np.argmin(t_np[match])])
            self._dbg_seq_done = True
            f0_old = float(_to_numpy(
                self.transform_fn.both_transforms(curr[idx:idx + 1], xp=cp)[0, 1]))
            return dict(
                idx=idx,
                slot=int(_to_numpy(slots)[idx]),
                temp=int(t_np[idx]), walker=sel_w, band=sel_b,
                f0_old=f0_old, f0_new=f0_old,
                snaps={},
            )
        except Exception as e:
            logger.warning("[GB_DEBUG %s] seq select skipped: %r", self.name, e)
            return None

    def _debug_slab_snapshot(self, buffer_obj, slot):
        """Copy one cell's residual slab as (nchannels, Nf_active, Nt_active)."""
        bs = self._basis_settings
        Nf_a = int(getattr(bs, "Nf_active", None) or bs.Nf)
        Nt_a = int(getattr(bs, "Nt_active", None) or bs.Nt)
        nc = buffer_obj.nchannels
        return _to_numpy(buffer_obj.band_buffer[slot]).copy().reshape(nc, Nf_a, Nt_a)

    def _debug_band_source_only_ll(self, buffer_obj, arr, slot, band):
        """Source-only ll ``-1/2 <a|a>`` of ``arr`` (nc, Nf_a, Nt_a), sliced
        to the WDM layers whose centers lie in ``band``'s edge interval
        (same slicing as :meth:`_debug_log_band_null`)."""
        bs = self._basis_settings
        layer_df = float(bs.layer_df)
        ind_min_f = int(bs.ind_min_f)
        Nf_a = arr.shape[1]
        be = _to_numpy(self.band_edges)
        k0 = max(int(np.ceil(be[band] / layer_df - 1e-9)) - ind_min_f, 0)
        k1 = min(int(np.floor(be[band + 1] / layer_df + 1e-9)) + 1 - ind_min_f,
                 Nf_a)
        dc = float(buffer_obj.settings.differential_component)
        nc = buffer_obj.nchannels
        r = arr[:, k0:k1]
        psd_np = _to_numpy(buffer_obj._materialize(buffer_obj.psd_buffer)[slot])
        if buffer_obj.tdi_channel_setup == "XYZ":
            ic = psd_np.reshape(nc, nc, Nf_a, -1)[:, :, k0:k1]
            return -0.5 * 4.0 * dc * float(
                np.einsum("ifk,ijfk,jfk->", r, ic.real, r))
        ic = psd_np.reshape(nc, Nf_a, -1)[:, k0:k1]
        return -0.5 * 4.0 * dc * float(np.sum(r * ic.real * r))

    def _debug_plot_band_sequence(self, buffer_obj, seq) -> None:
        """Four 3x3 figures (rows = X/Y/Z channels; columns = |template| /
        |data| / |buffer residual|) at the four buffer moments of one
        in-model repeat block on the traced source:

            1 before_removal   (source modeled: residual ~ null)
            2 after_removal    (source UN-modeled: signal IN the residual)
            3 before_addback   (must equal 2 -- repeats never touch the buffer)
            4 after_addback    (final template re-subtracted: signal OUT)

        The third column is the RAW buffer state at that moment -- the
        signal visibly enters at 2 and leaves at 4. Each figure's suptitle
        carries the band's SOURCE-ONLY ll of that state (-1/2 <r|r> over
        the band's layers), so the in/out shows up numerically too.

        Buffer sign convention: the band buffer holds the RESIDUAL
        (data - templates); ``remove_sources_from_band_buffer`` UN-models
        the source (residual += template) and
        ``add_sources_to_band_buffer`` re-subtracts it. Templates come
        from snapshot differences (old = snap2 - snap1; new =
        snap3 - snap4); the data column is the with-source state of each
        pair (constant by construction -- the moving part is column 3).
        """
        if not self.debug:
            return
        try:
            import os as _os
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            s = seq["snaps"]
            need = ("before_removal", "after_removal",
                    "before_addback", "after_addback")
            if any(k not in s for k in need):
                return
            bs = self._basis_settings
            layer_df = float(bs.layer_df)
            ind_min_f = int(bs.ind_min_f)

            T_old = s["after_removal"] - s["before_removal"]
            T_new = s["before_addback"] - s["after_addback"]
            lls = {k: self._debug_band_source_only_ll(
                       buffer_obj, s[k], seq["slot"], seq["band"])
                   for k in need}
            figures = [
                # (tag, template, data, ACTUAL buffer state, f0)
                ("1_before_removal", T_old, s["after_removal"],
                 s["before_removal"], seq["f0_old"]),
                ("2_after_removal", T_old, s["after_removal"],
                 s["after_removal"], seq["f0_old"]),
                ("3_before_addback", T_new, s["before_addback"],
                 s["before_addback"], seq["f0_new"]),
                ("4_after_addback", T_new, s["before_addback"],
                 s["after_addback"], seq["f0_new"]),
            ]

            nc = T_old.shape[0]
            ch_names = ["X", "Y", "Z"][:nc]
            local = int(round(seq["f0_old"] / layer_df)) - ind_min_f
            # 5-layer (mm5-style) span: tight enough that neighboring
            # galaxy sources outside the band (e.g. 19.668 mHz, 5 layers
            # below the 20.38 mHz source) stay out of the figures.
            lo = max(local - 2, 0)
            hi = min(local + 3, T_old.shape[1])
            ylo = (ind_min_f + lo - 0.5) * layer_df * 1e3
            yhi = (ind_min_f + hi - 0.5) * layer_df * 1e3

            _os.makedirs(self.debug_plot_dir, exist_ok=True)
            # ONE color scale per channel row, shared by every column of
            # every figure in the sequence: the panels are directly
            # comparable, so the signal entering/leaving the buffer shows
            # up as brightness (a near-null residual stays dark instead of
            # being autoscaled up to look like signal).
            vmax_row = [
                max(float(np.abs(arr[row, lo:hi]).max())
                    for _tag, _T, _D, _R, _f0 in figures
                    for arr in (_T, _D, _R))
                for row in range(nc)
            ]
            for tag, T, D, R, f0 in figures:
                ll_state = lls[tag[2:]]
                fig, axes = plt.subplots(
                    nc, 3, figsize=(13.5, 3.2 * nc), squeeze=False,
                    sharex=True, sharey=True,
                )
                for row in range(nc):
                    for col, (name, arr) in enumerate(
                            [("template", T), ("data", D),
                             ("buffer residual", R)]):
                        ax = axes[row][col]
                        im = ax.imshow(
                            np.abs(arr[row, lo:hi]), aspect="auto",
                            origin="lower",
                            extent=[0, arr.shape[2], ylo, yhi],
                            vmin=0.0, vmax=vmax_row[row],
                        )
                        ax.axhline(f0 * 1e3, color="r", ls="--", lw=1.0)
                        if row == 0:
                            ax.set_title(f"|{name}|", fontsize=11)
                        if col == 0:
                            ax.set_ylabel(f"{ch_names[row]}\nfrequency [mHz]",
                                          fontsize=10)
                        if row == nc - 1:
                            ax.set_xlabel("WDM time pixel", fontsize=10)
                        ax.tick_params(labelsize=8)
                        cbar = fig.colorbar(im, ax=ax)
                        cbar.ax.tick_params(labelsize=7)
                fig.suptitle(
                    f"GB in-model sequence {tag.replace('_', ' ')} — "
                    f"band {seq['band']} | walker {seq['walker']} | "
                    f"T{seq['temp']} | f0 = {f0 * 1e3:.4f} mHz\n"
                    f"band SOURCE-ONLY ll of buffer residual = "
                    f"{ll_state:.4e}",
                    fontsize=13,
                )
                fname = _os.path.join(
                    self.debug_plot_dir,
                    f"gb_debug_seq{tag}_band{seq['band']}_w{seq['walker']}"
                    f"_t{seq['temp']}_{self._dbg_plot_counter:04d}.png",
                )
                fig.savefig(fname, dpi=120, bbox_inches="tight")
                plt.close(fig)
                self._dbg_plot_counter += 1
                logger.info("[GB_DEBUG %s] saved sequence plot -> %s",
                            self.name, fname)
        except Exception as e:
            logger.warning("[GB_DEBUG %s] sequence plots skipped: %r",
                           self.name, e)

    def _debug_log_band_null(self, buffer_obj) -> None:
        """Log the CHOSEN band's source-only residual log-likelihood per
        temperature, once per sampler step (right after the first buffer
        load, i.e. central-band sources subtracted, before any proposal).

        This is the direct null check: with noiseless data and the cold
        chain at (or near) the injection, the chosen band's
        ``-1/2 <r|r>`` restricted to the band's WDM layers must be ~0 for
        T0. The full-buffer ``likelihood(source_only=True)`` cannot show
        this on the WDM path -- each cell slab spans the whole active band,
        so the unsubtracted rest of the galaxy dominates; this slices out
        exactly the band's layers.
        """
        if not self.debug or getattr(self, "_dbg_null_logged", True):
            return
        try:
            bs = self._basis_settings
            if not hasattr(bs, "layer_df"):
                return  # WDM-only diagnostic for now
            sel_w = self.debug_plot_walker
            sel_b = (self.debug_plot_band if self.debug_plot_band is not None
                     else (len(self.band_edges) - 1) // 2)

            combos = _to_numpy(buffer_obj.unique_band_combos)
            rows = [i for i in range(combos.shape[0])
                    if int(combos[i, 1]) == sel_w and int(combos[i, 2]) == sel_b]
            if not rows:
                return
            self._dbg_null_logged = True

            layer_df = float(bs.layer_df)
            ind_min_f = int(bs.ind_min_f)
            Nf_a = int(getattr(bs, "Nf_active", None) or bs.Nf)
            Nt_a = int(getattr(bs, "Nt_active", None) or bs.Nt)
            be = _to_numpy(self.band_edges)
            # WDM layers are CENTERED on m*layer_df while band edges are at
            # m*layer_df, so the band interval straddles two layers: include
            # every layer whose center lies in [edge_b, edge_b+1] (both
            # boundary layers), so the carrier layer is always covered.
            k0 = max(int(np.ceil(be[sel_b] / layer_df - 1e-9)) - ind_min_f, 0)
            k1 = min(int(np.floor(be[sel_b + 1] / layer_df + 1e-9)) + 1 - ind_min_f,
                     Nf_a)
            dc = float(buffer_obj.settings.differential_component)
            nc = buffer_obj.nchannels

            band_np = _to_numpy(buffer_obj._materialize(buffer_obj.band_buffer))
            psd_np = _to_numpy(buffer_obj._materialize(buffer_obj.psd_buffer))
            msgs = []
            for i in sorted(rows, key=lambda i: int(combos[i, 0])):
                t = int(combos[i, 0])
                r = band_np[i].reshape(nc, Nf_a, Nt_a)[:, k0:k1]
                if buffer_obj.tdi_channel_setup == "XYZ":
                    ic = psd_np[i].reshape(nc, nc, Nf_a, Nt_a)[:, :, k0:k1]
                    ll = -0.5 * 4.0 * dc * float(
                        np.einsum("ifk,ijfk,jfk->", r, ic.real, r))
                else:
                    ic = psd_np[i].reshape(nc, Nf_a, Nt_a)[:, k0:k1]
                    ll = -0.5 * 4.0 * dc * float(np.sum(r * ic.real * r))
                msgs.append(f"T{t}: {ll:.6e}")
            logger.info(
                "[GB_DEBUG %s] sub-band SOURCE-ONLY residual ll "
                "(band %d, walker %d, layers %d:%d): %s "
                "(cold chain at injection should be ~0)",
                self.name, sel_b, sel_w, ind_min_f + k0, ind_min_f + k1,
                "; ".join(msgs),
            )
        except Exception as e:
            logger.warning("[GB_DEBUG %s] band-null log skipped: %r", self.name, e)

    def _debug_plot_band(self, buffer_obj, params_add, data_index, swap_N_vals,
                         ll_diff_kept, map_to_update_cpu, keep2, move_i,
                         stage: str = "in-model") -> None:
        """Save ONE WDM time-frequency figure for the CHOSEN (walker, band)
        cell, with a panel per temperature present in this proposal batch.

        Only the cell selected by ``debug_plot_walker`` / ``debug_plot_band``
        (default: walker 0 / the central band) is plotted -- not every
        (temp, walker, band) the proposal touches -- so a debug run produces
        a small, readable progression instead of hundreds of PNGs.

        ``stage`` labels the proposal context in the title and file name:
        ``"rj"`` (birth/death step) or ``"in-model"`` (repeat block).
        """
        if not self.debug:
            return
        try:
            import os as _os
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            xp = self.xp
            bs = self._basis_settings
            layer_df = float(bs.layer_df)
            ind_min_f = int(bs.ind_min_f)

            sel_w = self.debug_plot_walker
            sel_b = (self.debug_plot_band if self.debug_plot_band is not None
                     else (len(self.band_edges) - 1) // 2)

            # ``params_add`` / ``data_index`` / ``ll_diff_kept`` are aligned
            # with the KEPT subset; map_to_update_cpu indexes the full batch,
            # bridged by ``orig``.
            orig = np.where(_to_numpy(keep2).astype(bool))[0]
            if orig.size == 0:
                return
            temps_all, walkers_all, bands_all = map_to_update_cpu
            pos = [i for i, j in enumerate(orig)
                   if int(walkers_all[j]) == sel_w and int(bands_all[j]) == sel_b]
            if not pos:
                return  # chosen cell not in this batch
            # One figure per stage per sampler step: the set is reset at the
            # top of run_proposal, so later picks of the same cell within
            # this step do not save additional figures.
            plotted = getattr(self, "_dbg_plotted_stages", None)
            if plotted is not None:
                if stage in plotted:
                    return
                plotted.add(stage)
            pos.sort(key=lambda i: int(temps_all[orig[i]]))

            params_phys = self.transform_fn.both_transforms(params_add, xp=cp)
            di_np = _to_numpy(data_index)
            ll_np = _to_numpy(xp.asarray(ll_diff_kept))

            # band_buffer is per-slot (num_bands_now, nchannels, data_length)
            # with the WDM tile flattened; recover (Nf_active, Nt_active).
            Nf_a = int(getattr(bs, "Nf_active", None) or bs.Nf)
            Nt_a = int(getattr(bs, "Nt_active", None) or bs.Nt)

            n = len(pos)
            ncols = min(n, 4)
            nrows = (n + ncols - 1) // ncols
            fig, axes = plt.subplots(
                nrows, ncols, figsize=(5.6 * ncols, 4.4 * nrows),
                squeeze=False, sharey=True,
            )
            for ax in axes.flat[n:]:
                ax.set_visible(False)

            for panel, i0 in enumerate(pos):
                ax = axes.flat[panel]
                temp = int(temps_all[orig[i0]])
                slab = int(di_np[i0])
                f0 = float(_to_numpy(params_phys[i0, 1]))
                local = int(round(f0 / layer_df)) - ind_min_f
                ll_val = float(ll_np[i0])
                tile = np.abs(
                    _to_numpy(buffer_obj.band_buffer[slab][0])
                ).reshape(Nf_a, Nt_a)
                # 5-layer (mm5-style) span, matching the sequence figures.
                lo = max(local - 2, 0)
                hi = min(local + 3, tile.shape[0])
                sub = tile[lo:hi]
                # WDM layer m is CENTERED on m*layer_df (span (m +- 1/2)*df):
                # the y-extent runs from the bottom edge of layer lo to the
                # top edge of layer hi-1, i.e. offset -1/2 layer relative to
                # the raw indices. (Without the -0.5 every row displayed a
                # half-layer too high and sources looked misaligned.)
                im = ax.imshow(
                    sub, aspect="auto", origin="lower",
                    extent=[0, sub.shape[1],
                            (ind_min_f + lo - 0.5) * layer_df * 1e3,
                            (ind_min_f + hi - 0.5) * layer_df * 1e3],
                )
                ax.axhline(f0 * 1e3, color="r", ls="--", lw=1.2,
                           label=f"f0 = {f0 * 1e3:.4f} mHz")
                # RJ forbidden proposals carry a -1e300 sentinel, not a
                # likelihood -- label them instead of printing the sentinel.
                ll_txt = ("forbidden proposal" if ll_val < -1e290
                          else f"$\\Delta$logL = {ll_val:.3e}")
                ax.set_title(f"T{temp}  {ll_txt}", fontsize=11)
                ax.set_xlabel("WDM time pixel (X)", fontsize=10)
                if panel % ncols == 0:
                    ax.set_ylabel("frequency [mHz]", fontsize=10)
                ax.tick_params(labelsize=9)
                ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
                cbar = fig.colorbar(im, ax=ax)
                cbar.ax.tick_params(labelsize=8)

            fig.suptitle(
                f"GB {stage} proposal — |WDM residual| around the source | "
                f"band {sel_b} | walker {sel_w} | repeat {move_i} | "
                f"all temperatures",
                fontsize=13,
            )
            _os.makedirs(self.debug_plot_dir, exist_ok=True)
            fname = _os.path.join(
                self.debug_plot_dir,
                f"gb_debug_{stage.replace('-', '')}_band{sel_b}_w{sel_w}"
                f"_move{move_i}_{self._dbg_plot_counter:04d}.png",
            )
            # bbox_inches="tight" guarantees the suptitle, axis labels, and
            # colorbar labels are all inside the saved figure.
            fig.savefig(fname, dpi=130, bbox_inches="tight")
            plt.close(fig)
            self._dbg_plot_counter += 1
            logger.info("[GB_DEBUG %s] saved band plot -> %s", self.name, fname)
        except Exception as e:
            logger.warning("[GB_DEBUG %s] band plot skipped: %r", self.name, e)

    def run_proposal(self, model, state, band_sorter, band_temps):
        """One full pass of per-band proposals.

        For each band-parity unit: *open* the parity class (cold-chain
        templates restored into the parent residual so every central-band
        window holds coordinate-independent raw data), load the
        (temp, walker, band) cells into the sub-band buffer, then repeatedly
        pick one not-yet-visited source per active cell (random order,
        without replacement) and run the RJ step plus the in-model repeats
        on it. Finished cells are swapped out for pending ones. *Closing*
        the parity class re-subtracts the cold-chain templates with the
        possibly-updated coordinates -- that is what propagates accepted
        cold-chain changes into the parent residual for the next unit and
        for the tempering stage.

        Returns ``(ll_change_log, prop_counts, acc_counts)``; the count
        arrays have shape ``(2, ntemps, nwalkers, num_bands)`` with row 0 =
        RJ proposals and row 1 = in-model proposals.
        """
        ll_change_log = cp.zeros((self.ntemps, self.nwalkers, self.num_bands))
        prop_counts = cp.zeros((2, self.ntemps, self.nwalkers, self.num_bands), dtype=int)
        acc_counts = cp.zeros_like(prop_counts)

        # One debug figure per STAGE per run_proposal call (i.e. per sampler
        # step): _debug_plot_band consumes this set. The band-null log fires
        # once per step too.
        self._dbg_plotted_stages = set()
        self._dbg_null_logged = False
        self._dbg_seq_done = False

        units = self.band_units if self.num_bands > 1 else 1
        start_unit = model.random.randint(units)

        for unit_i in range(units):
            remainder = (start_unit + unit_i) % units

            if self.debug:
                _dbg_ll_unit_start = _to_numpy(
                    model.analysis_container_arr.likelihood()
                ).copy()
                _dbg_change_start = _to_numpy(ll_change_log[0].sum(axis=-1)).copy()

            # Open this parity class in the parent residual.
            self.remove_cold_chain_sources_from_residual(
                model, band_sorter, units=units, remainder=remainder
            )
            self._debug_cold_chain_residual_loaded(model, remainder)

            apply_inds = not self.is_rj_prop
            
            extra_bool = (
                (band_sorter.band_inds < self.num_bands - 1) & (band_sorter.band_inds > 0)
            ) if self.num_bands > 1 else None

            subset = band_sorter.get_subset(
                units=units,
                remainder=remainder,
                apply_inds=apply_inds,
                extra_bool=extra_bool,
            )
            if subset is not None:
                self._run_band_unit(
                    model, band_sorter, subset, band_temps,
                    ll_change_log, prop_counts, acc_counts,
                )

            # Close: re-subtract with (possibly updated) cold-chain coords.
            self.add_cold_chain_sources_to_residual(
                model, band_sorter, units=units, remainder=remainder
            )

            if self.debug:
                _dbg_ll_unit_end = _to_numpy(
                    model.analysis_container_arr.likelihood()
                )
                _dbg_change_end = _to_numpy(ll_change_log[0].sum(axis=-1))
                _direct = _dbg_ll_unit_end - _dbg_ll_unit_start
                _tracked = _dbg_change_end - _dbg_change_start
                logger.info(
                    "[GB_DEBUG %s] unit %d (remainder %d) parent-ll reconcile: "
                    "direct per-walker %s vs tracked %s (max abs diff %.3e)",
                    self.name, unit_i, remainder,
                    np.array2string(_direct, precision=3),
                    np.array2string(_tracked, precision=3),
                    float(np.abs(_direct - _tracked).max()),
                )

            if self.backend.uses_cupy:
                self.xp.cuda.runtime.deviceSynchronize()
            self.mempool.free_all_blocks()

        return ll_change_log, prop_counts, acc_counts

    def _run_band_unit(self, model, band_sorter, subset, band_temps,
                       ll_change_log, prop_counts, acc_counts):
        """Drive one parity unit's cells through the sub-band buffer."""
        scheduler = BandScheduler(
            subset.special_band_inds, self.num_band_preload, xp=self.xp
        )
        buffer_obj = subset.get_buffer(
            model.analysis_container_arr, scheduler.slot_specials.copy()
        )
        self._debug_log_band_null(buffer_obj)

        # Pick eligibility lives on the MAIN sorter: only sources inside this
        # unit's subset are candidates (for in-model moves the subset already
        # applied ``inds``; for RJ it includes the freshly-drawn dead ones).
        eligible = self.xp.zeros(band_sorter.num_sources, dtype=bool)
        eligible[subset.inds_main_band_sorter] = True

        round_i = 0
        while scheduler.any_active():
            picked = self._pick_sources(band_sorter, buffer_obj, scheduler, eligible)
            if picked is None:
                break

            if self.is_rj_prop:
                self._run_rj_step(
                    model, band_sorter, buffer_obj, band_temps, picked,
                    ll_change_log, prop_counts, acc_counts, round_i, scheduler,
                )

            self._run_in_model_repeats(
                model, band_sorter, buffer_obj, band_temps, picked,
                ll_change_log, prop_counts, acc_counts,
            )

            scheduler.record_picks(picked["specials"])
            inds_fill, new_specials = scheduler.advance()
            if len(inds_fill):
                subset.get_buffer(
                    model.analysis_container_arr, new_specials,
                    inds_fill=inds_fill, buffer_obj=buffer_obj,
                )
                self._debug_log_band_null(buffer_obj)
            round_i += 1
            self.mempool.free_all_blocks()

        logger.info(
            f"{self.name}: band unit complete after {round_i} pick rounds "
            f"({scheduler.n_cells} cells)."
        )

    def _pick_sources(self, band_sorter, buffer_obj, scheduler, eligible):
        """One not-yet-visited source per active cell, without replacement.

        Vectorized on ``self.xp``: candidates are gathered through the
        special-index maps, randomly ranked within each cell, and the first
        per cell wins. ``band_sorter.has_run_rj`` marks consumed sources for
        the remainder of this proposal, so every source is visited exactly
        once per pass.
        """
        xp = self.xp
        cand = (
            eligible
            & (~band_sorter.has_run_rj)
            & band_sorter.get_subset_bool(
                special_band_inds=scheduler.active_slot_specials
            )
        )
        cand_ids = xp.arange(band_sorter.num_sources)[cand]
        if len(cand_ids) == 0:
            return None

        # Random rank within each cell: specials are integers spaced >= 1,
        # so adding U[0, 0.5) keeps the cell blocks intact while randomizing
        # the within-cell order (robust to non-stable argsort).
        specials = band_sorter.special_band_inds[cand_ids]
        key = specials.astype(xp.float64) + xp.random.rand(len(specials)) * 0.5
        order = xp.argsort(key)
        ids_sorted = cand_ids[order]
        _, first = xp.unique(specials[order], return_index=True)
        ids = ids_sorted[first]

        band_sorter.has_run_rj[ids] = True

        specials_picked = band_sorter.special_band_inds[ids]
        band_inds = band_sorter.band_inds[ids]
        return {
            "ids": ids,
            "specials": specials_picked,
            "slot_index": buffer_obj.get_index(specials_picked).astype(xp.int32),
            "temp_inds": band_sorter.temp_inds[ids],
            "walker_inds": band_sorter.walker_inds[ids],
            "band_inds": band_inds,
            "N_vals": band_sorter.band_N_vals[band_inds].copy(),
        }

    def _run_rj_step(self, model, band_sorter, buffer_obj, band_temps, picked,
                     ll_change_log, prop_counts, acc_counts, round_i, scheduler):
        """Birth/death proposal for each picked source (vectorized over cells).

        Births (``inds == False``; coordinates pre-drawn from the RJ proposal
        distribution in the BandSorter) score the add delta
        ``<r|h> - 0.5<h|h>``; deaths (``inds == True``) score the removal
        delta ``-<r|h> - 0.5<h|h>`` (see
        :meth:`SubBandBuffer.get_removal_ll`). Detailed-balance factors are
        the pre-computed ±logpdf of the RJ proposal distribution. On accept,
        ``inds`` flips and the cell residual is updated through
        ``fill_template`` with the appropriate sign.
        """
        xp = self.xp
        ids = picked["ids"]
        slots = picked["slot_index"]
        N_vals = picked["N_vals"]
        alive = band_sorter.inds[ids].copy()   # True -> death proposal

        params = band_sorter.coords[ids].copy()
        params[:] = self.periodic.wrap({"gb": params[:, None, :]}, xp=xp)["gb"][:, 0]

        logp = cp.asarray(self.gpu_priors["gb"].logpdf(params))
        prev_logp = cp.zeros_like(logp)
        curr_logp = cp.zeros_like(logp)
        prev_logp[alive] = logp[alive]
        curr_logp[~alive] = logp[~alive]

        # Births outside this cell's frequency window are unphysical.
        f_hz = params[:, 1] / 1e3
        out_of_band = (
            (f_hz < buffer_obj.frequency_lims[0][slots])
            | (f_hz > buffer_obj.frequency_lims[1][slots])
        )
        curr_logp[(~alive) & out_of_band] = -np.inf

        delta_ll = cp.full_like(logp, -1e300)
        d_h = cp.zeros_like(logp)
        h_h = cp.zeros_like(logp)
        keep = ~cp.isinf(curr_logp)

        if bool(keep.any()):
            k_ids = xp.arange(len(ids))[keep]
            birth_k = k_ids[~alive[keep]]
            death_k = k_ids[alive[keep]]

            def _eval(rows, phase_maximize):
                buffer_obj.get_ll(
                    params[rows], slots[rows], slots[rows], N_vals[rows],
                    phase_maximize=phase_maximize,
                )
                # use d_h and h_h to determine birth and death logls by trick with 
                # phase -> -phase
                d_h[rows] = buffer_obj.d_h_out.real
                h_h[rows] = buffer_obj.h_h_out.real
                bad_rows = rows[~buffer_obj.kept_out]
                return bad_rows

            oob_rows = xp.zeros(0, dtype=int)
            if self.phase_maximize and len(birth_k):
                # Maximise the birth phase; deaths keep the true phase.
                oob_rows = _eval(birth_k, True)
                if buffer_obj.phase_angle is not None:
                    params[birth_k, 3] = params[birth_k, 3] - buffer_obj.phase_angle
                if len(death_k):
                    oob_rows = xp.concatenate([oob_rows, _eval(death_k, False)])
            else:
                oob_rows = _eval(k_ids, False)

            delta_all = xp.where(alive, -d_h - 0.5 * h_h, d_h - 0.5 * h_h)
            delta_ll[keep] = delta_all[keep]
            delta_ll[oob_rows] = -1e300

            # SNR rejection-sampling clamp on births.
            opt_snr = xp.sqrt(xp.maximum(h_h, 0.0))
            reject = (~alive) & keep & (opt_snr < buffer_obj.opt_snr_rej_samp_limit)
            delta_ll[reject] = -1e300

            self._debug_verify_rj_step(
                buffer_obj, params, alive, slots, N_vals, delta_ll, keep,
                picked, round_i, scheduler,
            )

        beta = band_temps[picked["band_inds"], picked["temp_inds"]]
        factors = band_sorter.factors[ids]
        lnpdiff = beta * delta_ll + (curr_logp - prev_logp) + factors
        accept = lnpdiff >= cp.log(cp.random.rand(*lnpdiff.shape))

        # Coordinates outside the prior can only be accepted at beta == 0;
        # everything else is a bug -> warn and reject.
        bad_mask = (delta_ll <= -1e299) | (curr_logp <= -1e229)
        bad_accepts = accept & bad_mask
        if bool(xp.any(bad_accepts)):
            if bool(xp.any(beta[bad_accepts] != 0.0)) and not (
                "fstat" in self.name or "refit" in self.name
            ):
                logger.warning(
                    f"{self.name}: accepted an out-of-prior RJ coordinate at beta > 0."
                )
            accept[bad_accepts] = False

        t_i, w_i, b_i = picked["temp_inds"], picked["walker_inds"], picked["band_inds"]
        prop_counts[0][t_i, w_i, b_i] += 1

        if os.environ.get("GB_RJ_TRACE"):
            # Trace every cold-chain DEATH proposal (accepted or not) plus
            # every accepted cold-chain move. The death delta is
            # -<r|h> - 0.5<h|h>: for a well-fit bright source it must sit
            # near -0.5*SNR^2 and essentially never be accepted at beta=1,
            # so a cold-chain leaf loss means either h_h came back ~0 from
            # the kernel (template dropped: band-edge layer gating /
            # sub-band slab window / stale device index arrays) or the
            # accept bookkeeping raced the residual update. d_h/h_h at
            # proposal time distinguish the two -- compare GPU vs CPU runs
            # of the same seed/config.
            for _k in range(len(ids)):
                if t_i[_k] != 0:
                    continue
                _acc = bool(accept[_k])
                if not (_acc or bool(alive[_k])):
                    continue
                logger.warning(
                    "RJTRACE %s t=%d w=%d b=%d slot=%d f0=%.9e mHz N=%d "
                    "d_h=%.6e h_h=%.6e delta=%.6e beta=%.3e lnp=%.6e "
                    "factors=%.4e curr_lp=%.4e prev_lp=%.4e accept=%d",
                    "DEATH" if alive[_k] else "BIRTH",
                    int(t_i[_k]), int(w_i[_k]), int(b_i[_k]), int(slots[_k]),
                    float(params[_k, 1]), int(N_vals[_k]),
                    float(d_h[_k]), float(h_h[_k]), float(delta_ll[_k]),
                    float(beta[_k]), float(lnpdiff[_k]), float(factors[_k]),
                    float(curr_logp[_k]), float(prev_logp[_k]), int(_acc),
                )

        if bool(accept.any()):
            acc_ids = ids[accept]
            band_sorter.inds[acc_ids] = ~band_sorter.inds[acc_ids]
            # Phase-maximised births carry the rotated phi0 forward.
            band_sorter.coords[acc_ids] = self.periodic.wrap(
                {"gb": params[accept][:, None, :]}, xp=xp
            )["gb"][:, 0]

            ll_change_log[t_i[accept], w_i[accept], b_i[accept]] += delta_ll[accept]
            acc_counts[0][t_i[accept], w_i[accept], b_i[accept]] += 1

            birth_acc = accept & (~alive)
            death_acc = accept & alive
            if bool(birth_acc.any()):
                buffer_obj.add_sources_to_band_buffer(
                    band_sorter.coords[ids[birth_acc]],
                    slots[birth_acc], N_vals[birth_acc],
                )
            if bool(death_acc.any()):
                buffer_obj.remove_sources_from_band_buffer(
                    band_sorter.coords[ids[death_acc]],
                    slots[death_acc], N_vals[death_acc],
                )

    def _compute_proposal_cholesky(self, model, band_sorter, ids):
        """Batched Cholesky of the inverse Fisher matrix for ``ids``.

        Domain-symmetric through the fast computation objects:
        FD -> :meth:`GBFDComputations.information_matrix`,
        WDM -> :meth:`GBWDMComputations.information_matrix` (both against the
        parent inverse-covariance rows keyed by walker; the legacy
        SharedMemory ``gb.information_matrix`` path is retired).

        The Fisher comes back in PHYSICAL parameter space; it is mapped to
        the sampling basis with the (numerical, per-source diagonal)
        Jacobian of the transform container, conditioned by the fdot
        rescale, inverted, and factorized.

        TODO(known GBs): the fdot conditioning column (sampling index 2) and
        the 8->9 test_inds layout are still GB-specific; revisit with the
        known-GB branch.
        """
        xp = self.xp
        coords = band_sorter.coords[ids]
        n_src, ndim = coords.shape
        params_phys = self.transform_fn.both_transforms(coords, xp=cp)
        _test_inds = np.asarray(self.parameter_transforms.fill_dict["test_inds"])
        walker_inds = band_sorter.walker_inds[ids].astype(xp.int32)

        if isinstance(self._basis_settings, FDSettings):
            info_phys = self.gb_fd_comp.information_matrix(
                params_phys, model.analysis_container_arr,
                inds=_test_inds, noise_index=walker_inds,
            )
        else:
            info_phys = self.gb_wdm_comp.information_matrix(
                params_phys, model.analysis_container_arr,
                inds=_test_inds, noise_index=walker_inds,
            )

        # Conditioning scales for the sampling basis (fdot spans ~1e-13 in
        # sampled units; without the rescale the Fisher inversion is
        # ill-conditioned). The proposal draws in the rescaled coordinates
        # y = x / s and maps back with * s (see in_model_proposal).
        s = xp.ones(ndim)
        s[2] = self._fdot_scale
        self._proposal_param_scales = s

        # Numerical diagonal Jacobian d(phys[test_inds[i]]) / d(y_i) through
        # the transform container -- generic in the container's transforms.
        J = xp.zeros((n_src, ndim))
        for i in range(ndim):
            h = 1e-6 * xp.maximum(xp.abs(coords[:, i]), 1e-3)
            up = coords.copy()
            dn = coords.copy()
            up[:, i] += h
            dn[:, i] -= h
            dphys = (
                self.transform_fn.both_transforms(up, xp=cp)[:, _test_inds[i]]
                - self.transform_fn.both_transforms(dn, xp=cp)[:, _test_inds[i]]
            )
            J[:, i] = dphys / (2.0 * h) * s[i]

        info_y = info_phys * J[:, :, None] * J[:, None, :]

        self.mempool.free_all_blocks()
        # Robust inverse-Fisher factor: near-zero-SNR (prior-drawn) sources
        # give (numerically) singular Fishers. Eigendecompose and clamp the
        # spectrum to a relative floor; B = V diag(lambda^-1/2) satisfies
        # B B^T = inv(info) and is all the Gaussian proposal needs (the
        # proposal shape only -- M-H corrects).
        evals, evecs = xp.linalg.eigh(info_y)
        floor = 1e-10 * xp.maximum(
            xp.abs(evals).max(axis=-1, keepdims=True), 1e-300
        )
        evals = xp.maximum(xp.abs(evals), floor)
        return evecs / xp.sqrt(evals)[:, None, :]

    def in_model_proposal(self, coords, chol, band_sorter, source_ids, model):
        """Default in-model proposal: group-stretch / info-matrix mix.

        Overridable hook: subclasses provide other proposal components with
        the same ``(new_coords, factors)`` contract (``factors`` are the
        detailed-balance log-factors, zero for symmetric proposals).

        Group stretch is only allowed once the move has completed at least
        one full pass (``self.time >= 1``) and the cold-chain friend table
        exists; it is then drawn with probability ``stretch_probability``
        per repeat round (the info-matrix Cholesky jump otherwise).
        """
        xp = self.xp
        use_stretch = (
            self.stretch_probability > 0.0
            and self.time >= 1
            and getattr(band_sorter, "friend_start_inds", None) is not None
            and bool(np.random.rand() < self.stretch_probability)
        )

        if use_stretch:
            # Friends drawn per source; Eryn's GroupStretchMove supplies the
            # stretch math + (ndim-1)*log(zz) factors through find_friends.
            self._friends_for_stretch = band_sorter.draw_friends(source_ids)
            q, factors = self.get_proposal(
                {"gb": coords[None, :, None, :]},
                model.random,
                s_inds_all={"gb": xp.ones((1, coords.shape[0], 1), dtype=bool)},
            )
            new_coords = q["gb"][0, :, 0, :]
            factors = factors.reshape(-1)
        else:
            # Gaussian jump through the Fisher Cholesky (drawn in the
            # conditioned coordinates y = x / s; mapped back with * s).
            _rand = xp.random.randn(*coords.shape)
            dy = xp.einsum("...ij,...j->...i", chol, _rand)
            new_coords = coords + self.jump_factor * dy * self._proposal_param_scales[None, :]
            factors = xp.zeros(coords.shape[0])   # symmetric draw

        return new_coords, factors

    def _run_in_model_repeats(self, model, band_sorter, buffer_obj, band_temps,
                              picked, ll_change_log, prop_counts, acc_counts):
        """``num_repeat_proposals`` in-model rounds on the picked live sources.

        The picked source is first taken OUT of its cell residual, so every
        repeat scores through a plain ``get_add_ll`` against the
        source-free residual (the buffer is not touched between repeats --
        only the tracked coordinates and counters move). After the repeats,
        the final coordinates are written back into the residual and into
        the BandSorter.
        """
        xp = self.xp
        # Read ``inds`` from the MAIN sorter AFTER the RJ step: _run_rj_step
        # flips band_sorter.inds on accepted births/deaths, so this mask
        # includes freshly-born sources (they get the repeat block) and drops
        # freshly-killed ones (their template is already out of the residual).
        alive = band_sorter.inds[picked["ids"]]
        if not bool(alive.any()):
            return

        ids = picked["ids"][alive]
        slots = picked["slot_index"][alive]
        N_vals = picked["N_vals"][alive]
        t_i = picked["temp_inds"][alive]
        w_i = picked["walker_inds"][alive]
        b_i = picked["band_inds"][alive]
        beta = band_temps[b_i, t_i]

        curr = band_sorter.coords[ids].copy()
        curr[:] = self.periodic.wrap({"gb": curr[:, None, :]}, xp=xp)["gb"][:, 0]

        # Debug 3x3 sequence figures (channels x template/data/residual) at
        # the four buffer moments of this repeat block, for the chosen
        # (walker, band) cell only, once per sampler step.
        seq = self._debug_seq_select(buffer_obj, t_i, w_i, b_i, slots, curr)
        if seq is not None:
            seq["snaps"]["before_removal"] = self._debug_slab_snapshot(
                buffer_obj, seq["slot"])

        # Take the source out of the cell residual for the whole repeat block.
        buffer_obj.remove_sources_from_band_buffer(curr, slots, N_vals)

        if seq is not None:
            seq["snaps"]["after_removal"] = self._debug_slab_snapshot(
                buffer_obj, seq["slot"])

        chol = self._compute_proposal_cholesky(model, band_sorter, ids)
        ll_ref = buffer_obj.get_add_ll(curr, slots, slots, N_vals)
        curr_prior = cp.asarray(self.gpu_priors["gb"].logpdf(curr))

        n4 = (N_vals / 4).astype(int)
        lo_bin = (buffer_obj.frequency_lims[0][slots] / self.df).astype(int)
        hi_bin = (buffer_obj.frequency_lims[1][slots] / self.df).astype(int)

        for move_i in range(self.num_repeat_proposals):
            new, factors = self.in_model_proposal(curr, chol, band_sorter, ids, model)
            new[:] = self.periodic.wrap({"gb": new[:, None, :]}, xp=xp)["gb"][:, 0]

            new_logp = cp.asarray(self.gpu_priors["gb"].logpdf(new))
            # In-model steps stay within +- N/4 bins of the current source
            # and inside the band window (widened by N/4).
            new_bin = cp.abs(new[:, 1] / 1e3 / self.df).astype(int)
            new_logp[
                (cp.abs(new[:, 1] / 1e3 - curr[:, 1] / 1e3) / self.df).astype(int) > n4
            ] = -np.inf
            new_logp[new_bin < lo_bin - n4] = -np.inf
            new_logp[new_bin > hi_bin + n4] = -np.inf

            keep = ~cp.isinf(new_logp)
            new_ll = cp.full(len(ids), -1e300)
            if bool(keep.any()):
                new_ll[keep] = buffer_obj.get_add_ll(
                    new[keep], slots[keep], slots[keep], N_vals[keep],
                    phase_maximize=self.phase_maximize,
                )
                if self.phase_maximize and buffer_obj.phase_angle is not None:
                    new[keep, 3] = new[keep, 3] - buffer_obj.phase_angle
                    new[keep] = self.periodic.wrap(
                        {"gb": new[keep][:, None, :]}, xp=xp
                    )["gb"][:, 0]

            delta_ll = new_ll - ll_ref
            lnpdiff = beta * delta_ll + (new_logp - curr_prior) + factors
            accept = lnpdiff >= cp.log(cp.random.rand(*lnpdiff.shape))

            bad_mask = (new_ll <= -1e299) | (new_logp <= -1e229)
            bad_accepts = accept & bad_mask
            if bool(xp.any(bad_accepts)):
                if bool(xp.any(beta[bad_accepts] != 0.0)):
                    logger.warning(
                        f"{self.name}: accepted an out-of-prior in-model "
                        "coordinate at beta > 0."
                    )
                accept[bad_accepts] = False

            prop_counts[1][t_i, w_i, b_i] += 1
            if bool(accept.any()):
                curr[accept] = new[accept]
                ll_ref[accept] = new_ll[accept]
                curr_prior[accept] = new_logp[accept]
                ll_change_log[t_i[accept], w_i[accept], b_i[accept]] += delta_ll[accept]
                acc_counts[1][t_i[accept], w_i[accept], b_i[accept]] += 1

            self._debug_verify_in_model(
                buffer_obj, curr, new, slots, N_vals, delta_ll, keep,
                (asnumpy(t_i), asnumpy(w_i), asnumpy(b_i)), move_i,
            )

        # Final coordinates back into the residual and the sorter.
        band_sorter.coords[ids] = curr
        if seq is not None:
            seq["snaps"]["before_addback"] = self._debug_slab_snapshot(
                buffer_obj, seq["slot"])
        buffer_obj.add_sources_to_band_buffer(curr, slots, N_vals)
        if seq is not None:
            seq["snaps"]["after_addback"] = self._debug_slab_snapshot(
                buffer_obj, seq["slot"])
            seq["f0_new"] = float(_to_numpy(
                self.transform_fn.both_transforms(
                    curr[seq["idx"]:seq["idx"] + 1], xp=cp)[0, 1]))
            self._debug_plot_band_sequence(buffer_obj, seq)

    def _tempering_swap_grid(self, band_sorter, start):
        """Permuted (band, walker, temp) cell grid for one tempering parity.

        Interior bands only (the edge bands host no swaps), every
        temperature, and an independent random walker permutation per
        (band, temp) -- adjacent temperature columns of a grid row are the
        cells whose templates may exchange. Only the ``start``-parity
        interior bands are kept.

        Returns ``(band_index, temp_index, walkers_permuted, special_index,
        num_bands_unit)``; the first four are shaped
        ``(bands_this_parity, nwalkers, ntemps)``.
        """
        if self.num_bands == 1:
            num_bands_tempered = 1
            band_index_arr = cp.arange(1)
        else:
            num_bands_tempered = self.num_bands - 2
            band_index_arr = cp.arange(1, self.num_bands - 1)

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
        return band_index, temp_index, walkers_permuted, special_index, num_bands_unit

    def _adapt_band_temps(self, band_temps, band_swaps_accepted, band_swaps_proposed):
        """Per-band temperature-ladder adaptation, in place on ``band_temps``.

        Hyperbolic-decay adjustment of the inverse-temperature ladder from
        the just-collected swap acceptance ratios (hottest and coldest
        chains pinned) -- the standard eryn/ptemcee adaptation applied
        band-by-band. No-op on the first proposal (``self.time == 0``).

        TODO: change temperature adaptation.
        """
        if self.time <= 0:
            return
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

        dbetas = betas1 - betas0
        band_temps += self.xp.asarray(dbetas.T)

    def run_tempering(self, model, state, band_sorter, band_temps):
        ll_change_log_temp = cp.zeros((self.ntemps, self.nwalkers, self.num_bands))

        band_swaps_accepted = cp.zeros((len(self.band_edges) - 1, self.ntemps - 1), dtype=int)
        band_swaps_proposed = cp.zeros((len(self.band_edges) - 1, self.ntemps - 1), dtype=int)

        units = 2
        tmp_start = np.random.randint(units)
        for tmp in range(units):
            remainder = (tmp_start + tmp) % units
            start = remainder
            # start == 0 pairs with bool_remainder 1 because tempering
            # begins at band 1 (the interior bands).
            bool_remainder = 1 if start == 0 else 0

            self.remove_cold_chain_sources_from_residual(
                model,
                band_sorter,
                extra_bool=(band_sorter.band_inds % 2 == bool_remainder),
            )

            (band_index, temp_index, walkers_permuted, special_index,
             num_bands_unit) = self._tempering_swap_grid(band_sorter, start)

            num_bands_preload_temp = 200
            num_bands_run = 0
            while num_bands_run < self.nwalkers * num_bands_unit:
                start_ind = num_bands_run
                end_ind = start_ind + num_bands_preload_temp

                band_inds_now = band_index.reshape(-1, self.ntemps)[start_ind:end_ind].copy()
                walker_inds_now = walkers_permuted.reshape(-1, self.ntemps)[
                    start_ind:end_ind
                ].copy()
                special_inds_now = special_index.reshape(-1, self.ntemps)[start_ind:end_ind].copy()
                special_inds_now_flat = special_inds_now.flatten()

                buffer_obj = band_sorter.get_buffer(
                    model.analysis_container_arr,
                    special_inds_now_flat,
                    use_template_arr=True,
                )

                current_lls = buffer_obj.band_likelihoods(source_only=True).reshape(-1, self.ntemps)
                current_lls_orig = current_lls.copy()
                for t in range(self.ntemps)[1:][::-1]:
                    i1 = t
                    i2 = t - 1

                    # Buffer slots interleave temperatures: column t of a
                    # grid row is slot (row * ntemps + t).
                    buffer_i1 = cp.arange(buffer_obj.num_bands_now)[i1 :: self.ntemps]
                    buffer_i2 = cp.arange(buffer_obj.num_bands_now)[i2 :: self.ntemps]

                    buffer_obj.swap_template_slots(buffer_i1, buffer_i2)

                    # TODO: add indices because not every likelihood is needed
                    # TODO: C-side vectorized temperature-pair swap kernel
                    # (batch the template exchange + per-cell likelihoods).
                    new_lls = buffer_obj.band_likelihoods(source_only=True).reshape(-1, self.ntemps)[
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

                    # Reverse the swaps that were not accepted.
                    buffer_obj.swap_template_slots(buffer_i1[~sel], buffer_i2[~sel])

                    band_swaps_accepted[band_inds_now[:, 0], i2] += sel.astype(int)
                    band_swaps_proposed[band_inds_now[:, 0], i2] += 1

                    # Accepted cells trade their (temp, walker) labels in the
                    # sorter so the sources follow their templates.
                    specials_i1 = band_sorter.get_special_band_index(
                        i1, walker_inds_now[sel, i1], band_inds_now[sel, i1]
                    )
                    specials_i2 = band_sorter.get_special_band_index(
                        i2, walker_inds_now[sel, i2], band_inds_now[sel, i2]
                    )
                    band_sorter.exchange_cell_labels(
                        specials_i1, i1, walker_inds_now[sel, i1],
                        specials_i2, i2, walker_inds_now[sel, i2],
                        bands=band_inds_now[sel, i2],
                    )

                diffs = current_lls - current_lls_orig
                # ``=`` (not ``+=``): each (temp, walker, band) cell is
                # visited exactly once per tempering pass.
                ll_change_log_temp[
                    (
                        buffer_obj.unique_band_combos[:, 0],
                        buffer_obj.unique_band_combos[:, 1],
                        buffer_obj.unique_band_combos[:, 2],
                    )
                ] = diffs.flatten()
                num_bands_run += num_bands_preload_temp

            # ll_before3 = model.analysis_container_arr.likelihood()
            self.add_cold_chain_sources_to_residual(
                model,
                band_sorter,
                extra_bool=(band_sorter.band_inds % 2 == bool_remainder),
            )
            # ll_after3 = model.analysis_container_arr.likelihood()

        self._adapt_band_temps(band_temps, band_swaps_accepted, band_swaps_proposed)

        # TODO Ask michael what this is about print("NEED TO FIX ANALYSIS CONTAINER extra factor")
        ll_change_sum_temp = ll_change_log_temp.sum(axis=-1)

        return ll_change_sum_temp, band_swaps_accepted, band_swaps_proposed

    def _write_back_state(self, new_state, band_sorter) -> None:
        """Repack the sorter's live sources into ``new_state.branches['gb']``.

        Leaves are re-indexed densely per (temp, walker) in frequency order:
        live sources are ranked by the composite key
        ``(temp * nwalkers + walker) * 1e6 + f0`` and numbered ``0..n-1``
        within each (temp, walker) block. ``inds`` is rebuilt from scratch
        (all ``False``, then ``True`` at the repacked leaves), so RJ
        births/deaths and tempering walker reassignments all land here.

        TODO: NEED TO PROPERLY MOVE SUPPLEMENTAL INFO BASED ON OLD LEAVES
        (``inds_old`` below is the source-side index for that move).
        """
        alive = band_sorter.inds
        special_indices_finish = (
            band_sorter.temp_inds[alive] * self.nwalkers
            + band_sorter.walker_inds[alive]
        ) * int(1e6) + band_sorter.coords[alive, 1]
        special_inds_temp_walker = (
            band_sorter.temp_inds[alive] * self.nwalkers
            + band_sorter.walker_inds[alive]
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

        inds_new = (
            _to_numpy(band_sorter.temp_inds[alive]),
            _to_numpy(band_sorter.walker_inds[alive]),
            _to_numpy(leaf_inds_new),
        )
        inds_old = (
            _to_numpy(band_sorter.orig_temp_inds[alive]),
            _to_numpy(band_sorter.orig_walker_inds[alive]),
            _to_numpy(band_sorter.orig_leaf_inds[alive]),
        )
        new_state.branches["gb"].coords[inds_new] = _to_numpy(band_sorter.coords[alive])
        new_state.branches["gb"].inds[:] = False
        # turn on all the ones that are there
        new_state.branches["gb"].inds[inds_new] = True
        # new_state.branches["gb"].branch_supplemental[inds_new] = state.branches["gb"].branch_supplemental[inds_old]

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
        # Run-time source of truth is the ACA that arrives with the model:
        # refresh the domain quantities and re-bind the parent engine (FD
        # prototype comp + move-level engine) if this ACA differs from the
        # one currently bound. All fills / likelihoods below go through
        # model.analysis_container_arr.
        self._configure_domain(model.analysis_container_arr)
        self._bind_parent_acs(model.analysis_container_arr)
        self.current_state = state
        # np.random.seed(10)
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

        # An RJ move without a proposal distribution (e.g. search/refit
        # variants whose setup() has not produced one yet) cannot run. Pure
        # in-model moves don't need one.
        if self.is_rj_prop and self.rj_proposal_distribution is None:
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

        rj_prop = None if not self.is_rj_prop else self.rj_proposal_distribution["gb"]

        # make sure all periodic parameters have been put into their range
        new_state.branches["gb"].coords[:] = self.periodic.wrap(
            {"gb": new_state.branches["gb"].coords[:].reshape(ntemps * nwalkers, nleaves_max, ndim)}
        )["gb"].reshape(ntemps, nwalkers, nleaves_max, ndim)

        # TODO Ask Michael about this print("is this okay for rj? I do not think so, check with below use of gb_inds_in")
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
            gb_fd_comp=self.gb_fd_comp,
            waveform_kwargs=self.waveform_kwargs,
            rj_prop=rj_prop,
            keep_all_inds=keep_all_inds,
        )

        # Cold-chain friend table for the group-stretch half of the in-model
        # mix (rebuilt every proposal; cheap sort of the cold-chain f0s).
        self._infomat_wdm_logged = False
        if self.stretch_probability > 0.0:
            band_sorter.build_friend_index(self.nfriends)

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

        # print(np.abs(new_state.log_like - ll_after).max())        
        # store_max_diff = np.abs(new_state.log_like[0] - ll_after).max()
        start_diffs = np.abs(new_state.log_like[0] - ll_after)

        check = ll_after - new_state.log_like[0] - start_diffs

        logger.debug(f"Start check: {start_diffs=}, {check=}")
        if not np.abs(check).max() < 1e-4:
            # assert np.abs(check).max() < 1.0
            new_state.log_like[0] = self.check_ll_inject(model, band_sorter)
            #? update start diffs
            start_diffs = np.abs(new_state.log_like[0] - ll_after)

        # print("CHECKING 0:", store_max_diff, self.is_rj_prop)
        # self.check_ll_inject(new_state, verbose=True)
        # assert np.all(start_diffs < 2.0)
        num_active_leaves = new_state.branches["gb"].inds[0].sum(axis=-1) # cold chain only
        logger.info(f"Number of active leaves before proposal: {num_active_leaves}")
        # TODO: make sure band temps transfers out
        st_prop = time.perf_counter()
        ll_change_log, prop_counts, acc_counts = self.run_proposal(
            model, new_state, band_sorter, band_temps
        )
        et_prop = time.perf_counter()
        # Diagnostic: per-temperature alive source counts after run_proposal
        _alive_per_temp_post_prop = [
            int(band_sorter.inds[band_sorter.temp_inds == _t].sum()) for _t in range(ntemps)
        ]
        logger.info(f"Alive sources per temp after run_proposal: {_alive_per_temp_post_prop}")
        logger.info(f"Runtime of {self.name} proposal is {round(et_prop - st_prop,3)} seconds.")

        # TODO ask michael about this print("NEED TO FIX ANALYSIS CONTAINER extra factor")
        ll_change_sum = ll_change_log.sum(axis=-1)
        new_state.log_like[0] += _to_numpy(ll_change_sum[0])

        ll_after = model.analysis_container_arr.likelihood()
        check = ll_after - new_state.log_like[0] - start_diffs

        logger.debug(f"After proposal check: {start_diffs=}, {check=}")
        drift = float(np.abs(check).max())
        if drift >= 1e-4:
            # Incremental per-accept bookkeeping drifted from the true
            # residual likelihood (the narrow-band inner product reads a few
            # layers beyond the central band, whose context differs between
            # the cell buffer and the closed parent). The rebuild below is
            # exact, so the sampler stays correct; the warning tracks how
            # large the incremental drift got.
            logger.warning(
                f"{self.name}: incremental ll drift {drift:.3e} after "
                "proposal; rebuilding log_like from the residual."
            )
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

            logger.debug(f"After tempering check: {start_diffs=}, {check=}")
            drift = float(np.abs(check).max())
            if drift >= 1e-4:
                logger.warning(
                    f"{self.name}: incremental ll drift {drift:.3e} after "
                    "tempering; rebuilding log_like from the residual."
                )
                new_state.log_like[0] = self.check_ll_inject(model, band_sorter)

            self.mempool.free_all_blocks()
            et_temp = time.perf_counter()
            logger.info(f"Runtime of {self.name} tempering is {round(et_temp - st_temp,3)} seconds.")
            # Diagnostic: per-temperature alive source counts after run_tempering
            # _alive_per_temp_post_temp = [
            #     int(band_sorter.inds[band_sorter.temp_inds == _t].sum()) for _t in range(ntemps)
            # ]
            # logger.info(f"Alive sources per temp after run_tempering: {_alive_per_temp_post_temp}")

        # TODO ask michael about this print("make sure this works for rj")
        self._write_back_state(new_state, band_sorter)

        et_all = time.perf_counter()
        logger.info(f"Full runtime of {self.name} is {round(et_all - st_all, 3)} seconds.")
        num_active_leaves = new_state.branches["gb"].inds[0].sum(axis=-1)
        logger.info(f"Number of active leaves in cold chain after proposal: {num_active_leaves}")

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
            gb_fd_comp=self.gb_fd_comp,
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

        # prop/acc counts: row 0 = RJ, row 1 = in-model; band_info wants
        # (num_bands, ntemps) summed over walkers. The two families are
        # recorded separately (one propose produces both kinds).
        sub = new_state.sub_states["gb"]
        sub.band_info["band_temps"][:] = _to_numpy(band_temps)
        sub.band_info["band_num_binaries"][:] = band_info["band_counts"]
        sub.accumulate_proposals(
            _to_numpy(prop_counts[0].sum(axis=1).T),
            _to_numpy(acc_counts[0].sum(axis=1).T),
            is_rj=True,
        )
        sub.accumulate_proposals(
            _to_numpy(prop_counts[1].sum(axis=1).T),
            _to_numpy(acc_counts[1].sum(axis=1).T),
            is_rj=False,
        )
        sub.accumulate_swaps(
            _to_numpy(band_swaps_proposed), _to_numpy(band_swaps_accepted)
        )
        # TODO: check rj numbers

        # new_state.log_like[:] = self.check_ll_inject(new_state)

        self.mempool.free_all_blocks()
        new_state.log_like[:] = self.check_ll_inject(model, new_band_sorter)
        # if self.is_rj_prop:
        #     pass  # print(self.name, "2nd count check:", new_state.branches["gb"].inds.sum(axis=-1).mean(axis=-1), "\nll:", new_state.log_like[0] - orig_store, new_state.log_like[0])

        # new_state.log_prior[:] = model.compute_log_prior_fn(new_state.branches_coords, inds=new_state.branches_inds, supps=new_state.supplemental)
        accepted = np.zeros((ntemps, nwalkers), dtype=bool)

        num_active_sources = new_state.branches["gb"].inds.sum(axis=-1)[0]
        logger.info(f"Current number of active sources in cold chain is {num_active_sources}")

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
    """In-model GB move with the band-aware group-stretch / info-matrix mix.

    All machinery lives in :class:`GBSpecialBase`; the cold-chain friend
    table for the group stretch is rebuilt at the top of every ``propose``
    call (see ``build_friend_index``), so no per-iteration setup is needed
    here.
    """

    pass


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
    # need to get just f, fdot, fddot, alpha, delta
    data_index = xp.full(x.shape[0], walker_max, dtype=xp.int32)
    if fstat:
        x_in = x_tmp[:, xp.array([1, 2, 3, 7, 8])]
        # breakpoint()
        # TODO: fix for N>256?
        ll = gb.get_fstat_ll(
            x_in,
            acs.linear_data_arr,
            acs.linear_psd_arr,
            data_index=data_index,
            noise_index=data_index,
            data_length=acs.end_shape[0],
            data_splits=np.array([gb.gpus[0]]),
            phase_marginalize=phase_maximize,
            return_cupy=True,
            N=512,  
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
            data_length=acs.end_shape[0],
            data_splits=np.array([gb.gpus[0]]),
            phase_marginalize=phase_maximize,
            return_cupy=True,
            # N=512,
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
        assert isinstance(self.search_kwargs, dict)
        nwalkers: int = self.search_kwargs["nwalkers"]
        ntemps: int = self.search_kwargs["ntemps"]
        shutoff_band_iteration: int = self.search_kwargs["shutoff_band_iteration"]
        shutoff_frequency_threshold: float = self.search_kwargs["shutoff_frequency_threshold"]
        burn_1: int = self.search_kwargs["burn_1"]
        nsteps_1: int = self.search_kwargs["nsteps_1"]
        snr_threshold: float = self.search_kwargs["snr_threshold"]
        burn_2: int = self.search_kwargs["burn_2"]
        nsteps_2: int = self.search_kwargs["nsteps_2"]

        # FOR FAST TESTING/DEBUGGING
        # import pickle
        # with open("gmm_tmp.pickle", "rb") as fp:
        #     full_gmm = pickle.load(fp)

        # rj_dist = ProbDistContainer(
        #     {
        #         ("A", "f0", "fdot", "cos_iota", "alpha", "sin_delta"): full_gmm,
        #         "phi0": uniform_dist(0.0, 2 * np.pi),
        #         "psi": uniform_dist(0.0, np.pi),
        #     },
        #     use_cupy=True,
        # )
        # rj_dist.reset_key_order(["A", "f0", "fdot", "phi0", "cos_iota", "psi", "alpha", "sin_delta"])
        # return

        # run paraensemble MCMC.
        max_logl_walker = np.argmax(model.analysis_container_arr.likelihood()).item()
        self.gb.d_d = model.analysis_container_arr.inner_product()[max_logl_walker] # 0.0
        ndim = branches["gb"].ndim
        priors_global = self.priors if not self.backend.uses_cuda else self.gpu_priors            

        if self.num_bands == 1:
            f0_max = self.band_edges[1:]
            f0_min = self.band_edges[:-1]
        else:
            f0_max = self.band_edges[2:-1]
            f0_min = self.band_edges[1:-2]

        # logic to shutoff bands #? Think about how this should change when we change SNR_thresh and with changing noise
        if self.num_proposals >= shutoff_band_iteration:
            bands_to_shutoff = np.all(~self.found_source_in_band[-shutoff_band_iteration:, :], axis=0)

            if shutoff_frequency_threshold is not None:
                min_freqs = getattr(f0_min, "get")() if hasattr(f0_min, "get") else f0_min
                freq_mask = min_freqs >= shutoff_frequency_threshold
                bands_to_shutoff = bands_to_shutoff & freq_mask

            if np.all(bands_to_shutoff):
                logger.info(f"No sources found across all bands for {shutoff_band_iteration} iterations, reverting to priors")
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

        logger.info(f"The current number of active bands is {ngroups}")

        fdot_max = get_fdot_mojito(f0_max, sign="+")
        fdot_min = get_fdot_mojito(f0_max, sign="-")

        priors_in = deepcopy(priors_global)["gb"].priors_in
        priors_in["f0"] = uniform_dist(0.0, 1.0, use_cupy=self.backend.uses_cupy)
        priors_in["fdot"] = uniform_dist(0.0, 1.0, use_cupy=self.backend.uses_cupy)
        priors = {
            "gb": ProbDistContainer(priors_in, return_gpu=True, use_cupy=self.backend.uses_cupy)
        }
        start_params = priors["gb"].rvs(size=(ngroups, ntemps, nwalkers))
        prior_transform_fn = PriorTransformFn(f0_min * 1e3, f0_max * 1e3, fdot_min, fdot_max)
        prior_transform_fn.transform_from_prior_basis(start_params, self.xp.arange(ngroups))

        #? print("phase maximizing here right now (?)")
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
            self.phase_maximize, # False, #
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

        para_sampler.run_mcmc(state, nsteps_1, burn=burn_1, progress=True)

        samples = self.xp.asarray(para_sampler.get_chain()[:, :, 0])
        # Diagnostics disabled (stft_tof): computed-but-unused and each costs
        # a full likelihood sweep over all samples.
        # check_ll = para_sampler.get_log_like()[:, :, 0]
        # sample_ll = asnumpy(
        #     para_log_like(samples.reshape(-1, 8), *ll_args).reshape(samples.shape[:-1])
        # )

        # check_real_ll_phase_maximized = asnumpy(
        #     para_log_like(samples.reshape(-1, 8), *ll_args, fstat=False)
        #     .reshape(samples.shape[:-1])
        # )
        check_real_ll, opt_snr = para_log_like(
            samples.reshape(-1, 8), *ll_args_2, fstat=False, return_snr=True
        )
        check_real_ll = asnumpy(check_real_ll.reshape(samples.shape[:-1]))
        opt_snr = asnumpy(opt_snr.reshape(samples.shape[:-1]))

        # np.save("opt_snr_from_ldc_parasampler_check.npy", opt_snr)
        # np.save("samples_from_ldc_parasampler_check.npy", samples)

        # TODO: make cut adjustable
        groups_running_now = opt_snr.min(axis=(0, 2)) > snr_threshold

        if self.num_proposals == 0:
            self.found_source_in_band = groups_running_now
        else:
            if bands_to_shutoff is None:
                self.found_source_in_band = np.vstack([self.found_source_in_band, groups_running_now])
            else:
                shutoff_temp = np.zeros(self.found_source_in_band.shape[1], dtype=bool)
                shutoff_temp[~bands_to_shutoff] = groups_running_now
                self.found_source_in_band = np.vstack([self.found_source_in_band, shutoff_temp])

        logger.info(f"Found a source in {groups_running_now.sum()} out of {groups_running_now.shape[0]} active bands")
        if not np.any(groups_running_now):
            logger.info("Did not find any new sources.")
            return

        start_params_2 = np.tile(samples[-1][groups_running_now, None], (1, ntemps, 1, 1))
        # Maybe not start from maximized values?
        gibbs_sampling_setup_2 = np.ones(8, dtype=bool)
        if ll_args_2[4]: # phase_maximization
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
            args=ll_args_2,
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

        para_sampler_2.run_mcmc(new_state, nsteps_2, burn=burn_2, progress=True)

        samples_2 = self.xp.asarray(para_sampler_2.get_chain()[:, :, 0])
        # check_ll_2 = para_sampler_2.get_log_like()[:, :, 0]

        # Diagnostics disabled (stft_tof): computed-but-unused likelihood sweeps.
        # check_real_ll_phase_maximized_2 = asnumpy(
        #     para_log_like(samples_2.reshape(-1, 8), *ll_args, fstat=False)
        #     .reshape(samples_2.shape[:-1])
        # )
        # check_real_ll_2 = asnumpy(
        #     para_log_like(samples_2.reshape(-1, 8), *ll_args_2, fstat=False)
        #     .reshape(samples_2.shape[:-1])
        # )

        samples_2 = samples_2.transpose(1, 0, 2, 3)
        # np.save("/workspace/rrondeel/erebor/testing/highf_gb/search2_samples_check.npy", samples_2)

        st = time.perf_counter()
        samples_2_tmp = samples_2.reshape(samples_2.shape[0], -1, samples_2.shape[-1])[
            :, :, np.array([0, 1, 2, 4, 6, 7])
        ]

        if self.xp.isnan(samples_2_tmp).any() or self.xp.isinf(samples_2_tmp).any():
            logger.warning(
                f"samples_2_tmp contains NaN or Inf before GMM fitting. \
                NaN count: {self.xp.isnan(samples_2_tmp).sum()}. \
                Inf count: {self.xp.isinf(samples_2_tmp).sum()}. \
                Skipping search..."
            )
            return
            # breakpoint()
            # raise ValueError(
            #     f"samples_2_tmp contains NaN or Inf before GMM fitting. "
            #     f"NaN count: {self.xp.isnan(samples_2_tmp).sum()}, "
            #     f"Inf count: {self.xp.isinf(samples_2_tmp).sum()}"
            # )

        ranges = samples_2_tmp.max(axis=1) - samples_2_tmp.min(axis=1)  # (n_groups, n_features)
        degenerate = (ranges == 0)
        if degenerate.any():
            bad_groups, bad_feats = self.xp.where(degenerate)
            logger.warning(
                f"Degenerate features (zero range) in groups {bad_groups} \
                for features {bad_feats}. transform_to_gmm_basis will produce NaN. \
                Skipping search..."
            )
            return
            # breakpoint()
            # raise ValueError(
            #     f"Degenerate features (zero range) in groups {bad_groups} "
            #     f"for features {bad_feats}. transform_to_gmm_basis will produce NaN."
            # )

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
        logger.info(f"Runtime of GPU GMM FIT: {round(et - st,3)} seconds")

        rj_dist = ProbDistContainer(
            {
                ("A", "f0", "fdot", "cos_iota", "alpha", "sin_delta"): full_gmm,
                "phi0": uniform_dist(0.0, 2 * np.pi),
                "psi": uniform_dist(0.0, np.pi),
            },
            use_cupy=True,
        )
        rj_dist.reset_key_order(["A", "f0", "fdot", "phi0", "cos_iota", "psi", "alpha", "sin_delta"])
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
                # DEPRECATED (2026-07 rework): this legacy MPI hand-off read
                # from a stored ``self.mgh`` with a two-shard layout and a
                # ``lisasens_shaped`` attribute that no longer exist. When
                # the multi-GPU search path is revived, read the residual /
                # psd from the model ACA at propose time instead.
                raise NotImplementedError(
                    "GBSpecialRJSearchMove residual hand-off needs re-wiring "
                    "to the model's AnalysisContainerArray (legacy self.mgh "
                    "path removed in the 2026-07 rework)."
                )

        else:
            search_ch.cancel()

        # TODO print("CHECK INSIDE PROP")


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
        samples_keep = self.search_kwargs["refit_start_iteration"]
        nwalkers = self.search_kwargs["nwalkers"]
        num_compare_samples = 1
        # FOR FAST TESTING/DEBUGGING
        # import pickle
        # with open("gmm_tmp.pickle", "rb") as fp:
        #     full_gmm = pickle.load(fp)

        # rj_dist = ProbDistContainer(
        #     {
        #         ("A", "f0", "fdot", "cos_iota", "alpha", "sin_delta"): full_gmm,
        #         "phi0": uniform_dist(0.0, 2 * np.pi),
        #         "psi": uniform_dist(0.0, np.pi),
        #     },
        #     use_cupy=True,
        # )
        # rj_dist.key_order = ["A", "f0", "fdot", "phi0", "cos_iota", "psi", "alpha", "sin_delta"]
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
        if reader.iteration < 2 * samples_keep:
            logger.info("Not enough samples to perform refitting, reverting to priors.")
            self.rj_proposal_distribution = {"gb": self.priors if not self.backend.uses_cuda else self.gpu_priors}
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

        logger.info(
            f"Groups passing sample count filter: {keep.sum()} / {len(keep)}. "
            f"num_in_groups: {num_in_groups}"
        )

        if not keep.any():
            logger.warning(
                f"No groups have enough samples (threshold={nwalkers * samples_keep / 2:.0f}). "
                f"Max samples in any group: {num_in_groups.max()}. "
                f"Reverting to priors."
            )
            self.rj_proposal_distribution = {
                "gb": self.priors if not self.backend.uses_cuda else self.gpu_priors
            }
            return

        max_num_source = max([tmp.shape[0] for tmp in groups])
        samples = np.full((len(groups), max_num_source, groups[0].shape[-1]), np.nan)
        for i, group in enumerate(groups):
            samples[i, : len(group)] = group

        samples_fin = samples[keep]
        num_in_groups_fin = num_in_groups[keep]

        if len(num_in_groups_fin) == 0 or num_in_groups_fin.min() == num_in_groups_fin.max():
            logger.warning(
                f"Cannot construct step range from num_in_groups_fin={num_in_groups_fin}. "
                f"Reverting to priors..."
            )
            self.rj_proposal_distribution = {
                "gb": self.priors if not self.backend.uses_cuda else self.gpu_priors
            }
            return

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

            if np.isnan(samples_here).any():
                nan_groups = np.where(np.isnan(samples_here).any(axis=(1, 2)))[0]
                logger.warning(
                    f"NaN padding leaked into samples_here at start={start}. \
                    Affected groups (local indices): {nan_groups}. \
                    num_in_groups for those groups: {num_in_groups_fin[here][nan_groups]} \
                    Skipping Refit..."
                )
                return
                # raise ValueError(
                #     f"NaN padding leaked into samples_here at start={start}. "
                #     f"Affected groups (local indices): {nan_groups}. "
                #     f"num_in_groups for those groups: {num_in_groups_fin[here][nan_groups]}"
                # )

            ranges = samples_here.max(axis=1) - samples_here.min(axis=1)
            if (ranges == 0).any():
                bad = np.where((ranges == 0))
                logger.warning(
                    f"Degenerate features at start={start}: groups={bad[0]}, features={bad[1]} \
                    Skipping Refit..."
                )
                # raise ValueError(
                #     f"Degenerate features at start={start}: groups={bad[0]}, features={bad[1]}"
                # )

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
            # logger.info(start, end)

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

        logger.info(f"Runtime GMM Refit: {round(time.perf_counter() - st)}")
        rj_dist = ProbDistContainer(
            {
                ("A", "f0", "fdot", "cos_iota", "alpha", "sin_delta"): full_gmm,
                "phi0": uniform_dist(0.0, 2 * np.pi),
                "psi": uniform_dist(0.0, np.pi),
            },
            use_cupy=True,
        )
        rj_dist.key_order = ["A", "f0", "fdot", "phi0", "cos_iota", "psi", "alpha", "sin_delta"]
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
        param_labels = ["A", "f0", "fdot", "phi0", "cos_iota", "psi", "alpha", "sin_delta"]
    elif num_params == 9:
        param_labels = ["A", "f0", "fdot", "fddot", "phi0", "cos_iota", "psi", "alpha", "sin_delta"]
    else:
        param_labels = num_params * [""]
    for i, param_label in enumerate(param_labels):
        param_values = array[..., i]
        min_array_i = param_values.min()
        max_array_i = param_values.max()
        print(f"For parameter {param_label}, the minimun value is {min_array_i}, the maximum value is {max_array_i}")