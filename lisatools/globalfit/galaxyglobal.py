
from copy import deepcopy
import cupy as xp
import numpy as np
import time
import pickle
import shutil
from mpi4py import MPI

from gbgpu.gbgpu import GBGPU


from lisatools.sampling.moves.gbspecialstretch import GBSpecialStretchMove
from gbgpu.utils.utility import get_fdot

mempool = xp.get_default_memory_pool()

from lisatools.utils.multigpudataholder import MultiGPUDataHolder
from eryn.moves import CombineMove
from lisatools.sampling.prior import FullGaussianMixtureModel, GBPriorWrap
from eryn.moves.tempering import make_ladder
from eryn.state import BranchSupplimental
from eryn.prior import ProbDistContainer, uniform_dist
from eryn.ensemble import EnsembleSampler
from lisatools.sampling.prior import SNRPrior, AmplitudeFromSNR

import subprocess

import warnings

warnings.filterwarnings("ignore")

stop_here = True

from eryn.moves import Move
from .state import State as GBState
from .hdfbackend import HDFBackend as GBHDFBackend


from copy import deepcopy
import numpy as np
from scipy import stats
import warnings
import time
from gbgpu.utils.utility import get_N
import pickle
from tqdm import tqdm
import os

mempool = xp.get_default_memory_pool()

from ..utils.multigpudataholder import MultiGPUDataHolder
from .gathergalaxy import gather_gb_samples

from sklearn.mixture import GaussianMixture

try:
    import cupy as xp

    gpu_available = True
except ModuleNotFoundError:
    import numpy as xp

    gpu_available = False

from lisatools.utils.utility import searchsorted2d_vec, get_groups_from_band_structure
from eryn.moves import StretchMove
from eryn.moves import TemperatureControl
from eryn.prior import ProbDistContainer
from eryn.utils.utility import groups_from_inds
from eryn.utils import PeriodicContainer

from eryn.moves import GroupStretchMove

from lisatools.diagnostic import inner_product
from eryn.state import State



class PlaceHolder(Move):
    def __init__(self, *args, **kwargs):
        super(PlaceHolder, self).__init__(*args, **kwargs)

    def propose(self, model, state):
        accepted = np.zeros(state.log_like.shape)
        self.temperature_control.swaps_accepted = np.zeros(
            self.temperature_control.ntemps - 1
        )
        return state, accepted


class BasicResidualMGHLikelihood:
    def __init__(self, mgh):
        self.mgh = mgh

    def __call__(self, *args, supps=None, **kwargs):
        ll_temp = self.mgh.get_ll()
        overall_inds = supps["overal_inds"]

        return ll_temp[overall_inds]


def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a, idx, axis=axis)
    

from eryn.utils.updates import Update


class UpdateNewResiduals(Update):
    def __init__(
        self, mgh, gb, comm, head_rank, gpu_priors, last_prior_vals, verbose=False
    ):
        self.mgh = mgh
        self.comm = comm
        self.head_rank = head_rank
        self.verbose = verbose
        self.gb = gb
        self.gpu_priors = gpu_priors
        self.last_prior_vals = last_prior_vals

    def __call__(self, iter, last_sample, sampler):
        
        if self.verbose:
            print("Sending gb update to head process.")

        update_dict = {
            "cc_params": last_sample.branches["gb"].coords[0, :, :].copy(),
            "cc_inds": last_sample.branches["gb"].inds[0, :, :].copy(),
            "cc_ll": last_sample.log_like[0].copy(),
            "cc_lp": last_sample.log_prior[0].copy(),
            "last_state": last_sample
        }
        self.comm.send({"send": True, "receive": True}, dest=self.head_rank, tag=50)

        self.comm.send({"gb_update": update_dict}, dest=self.head_rank, tag=58)

        if self.verbose:
            print("Requesting updated data from head process.")

        new_info = self.comm.recv(source=self.head_rank, tag=51)
        
        if self.verbose:
            print("Received new data from head process.")

        # get new gmm information

        new_search_gmm = new_info.gb_info["search_gmm_info"]
        if new_search_gmm is not None:
            gen_dist_search = make_gmm(self.gb, new_search_gmm)
        else:
            gen_dist_search = self.gpu_priors["gb"]

        new_refit_gmm = new_info.gb_info["refit_gmm_info"]
        if new_refit_gmm is not None:
            gen_dist_refit = make_gmm(self.gb, new_refit_gmm)
        elif new_search_gmm is not None:
            gen_dist_refit = make_gmm(self.gb, new_search_gmm)
        else:
            gen_dist_refit = self.gpu_priors["gb"]

        # sub out the proposal distributions
        for move in sampler.rj_moves:
            if move.name == "rj_refit_gmm":
                move.rj_proposal_distribution = {"gb": gen_dist_refit}
            
            elif move.name == "rj_search_gmm":
                move.rj_proposal_distribution = {"gb": gen_dist_search}

        if self.verbose:
            print("Finished GMM reset.")
            print("Generating new base data.")

        ntemps_pe = last_sample.log_like.shape[0]
        nwalkers_pe = last_sample.log_like.shape[1]
        nleaves_max = last_sample.branches["gb"].shape[2]

        # get old data off gpu
        old_data = self.mgh.get()

        ntemps, nwalkers, nleaves_max = last_sample.branches["gb"].inds.shape
        walker_inds_old = np.repeat(np.arange(nwalkers)[:, None], ntemps * nleaves_max, axis=-1).reshape(nwalkers, ntemps, nleaves_max).transpose(1, 0, 2)
        
        per_source_lp_old = np.zeros_like(last_sample.branches["gb"].inds, dtype=np.float64)
        walker_inds_old = np.repeat(np.arange(nwalkers_pe)[:, None], ntemps_pe * nleaves_max, axis=-1).reshape(nwalkers_pe, ntemps_pe, nleaves_max).transpose(1, 0, 2)[last_sample.branches["gb"].inds]
        
        per_source_lp_old[last_sample.branches["gb"].inds] = self.gpu_priors["gb"].logpdf(last_sample.branches["gb"].coords[last_sample.branches["gb"].inds], psds=self.mgh.lisasens_shaped[0][0], walker_inds=walker_inds_old).get()
        
        lp_gbs_old = per_source_lp_old.sum(axis=-1)

        generated_info = new_info.get_data_psd(include_gbs=False, n_gen_in=nwalkers_pe, include_lisasens=True, return_prior_val=True, fix_val_in_gen=["gb"])  # , include_ll=True, include_source_only_ll=True)
        data = generated_info["data"]
        psd = generated_info["psd"]
        lisasens = generated_info["lisasens"]
        
        prev_logl_check = self.mgh.get_ll(include_psd_info=True)

        # needs to leave out gbs
        self.mgh.sub_in_data_and_psd(data, psd, lisasens)

        if self.verbose:
            print("Finished subbing in new data.")

        xp.get_default_memory_pool().free_all_blocks()

        # add priors from other parts of global fit
        prev_logp = last_sample.log_prior[0] + self.last_prior_vals
        prev_logl = last_sample.log_like[0].copy()

        ntemps, nwalkers, nleaves_max = last_sample.branches["gb"].inds.shape
        walker_inds = np.repeat(np.arange(nwalkers)[:, None], ntemps * nleaves_max, axis=-1).reshape(nwalkers, ntemps, nleaves_max).transpose(1, 0, 2)
        
        per_source_lp = np.zeros_like(last_sample.branches["gb"].inds, dtype=np.float64)
        walker_inds = np.repeat(np.arange(nwalkers_pe)[:, None], ntemps_pe * nleaves_max, axis=-1).reshape(nwalkers_pe, ntemps_pe, nleaves_max).transpose(1, 0, 2)[last_sample.branches["gb"].inds]
        
        per_source_lp[last_sample.branches["gb"].inds] = self.gpu_priors["gb"].logpdf(last_sample.branches["gb"].coords[last_sample.branches["gb"].inds], psds=lisasens[0], walker_inds=walker_inds).get()
        
        new_lp_gbs = per_source_lp.sum(axis=-1)
        # add priors from other parts of global fit
        logp = new_lp_gbs[0] + generated_info["psd_prior_vals"] + generated_info["mbh_prior_vals"]
        logl = self.mgh.get_ll(include_psd_info=True)

        # all in the cold chain (beta = 1)
        prev_logP = prev_logl + prev_logp
        logP = logl + logp

        # factors = 0.0 TODO: fix detailed balance here?
        accept = (logP - prev_logP) > np.log(sampler.get_model().random.rand(*logP.shape))
        # TODO: this was not right in the end. Need to think about more. 
        # so adding this:
        accept[:] = True
        
        last_sample.log_like[0, accept] = logl[accept]
        last_sample.log_prior[:, accept] = new_lp_gbs[:, accept]
        self.last_prior_vals[accept] = (generated_info["psd_prior_vals"] + generated_info["mbh_prior_vals"])[accept]

        keys_pass = [f"channel{i + 1}_data" for i in range(2)]

        # start with new
        new_data = self.mgh.get()
        keep_data = self.mgh.get()
        for key in old_data:
            if key in keys_pass:
                continue

            assert len(keep_data[key]) == 1
            try:
                keep_data[key][0][~accept, :] = old_data[key][0][~accept, :]
            except IndexError:
                breakpoint()

        out_data = [keep_data[f"channel{i + 1}_base_data"][0].copy() for i in range(2)]
        out_psd = [keep_data[f"channel{i + 1}_psd"][0].copy() for i in range(2)]
        out_lisasens = [keep_data[f"channel{i + 1}_lisasens"][0].copy() for i in range(2)]
        
        # needs to leave out gbs
        self.mgh.sub_in_data_and_psd(out_data, out_psd, out_lisasens)

        print("ACCEPTED:", np.arange(len(accept))[accept], np.abs(last_sample.log_like[0] - self.mgh.get_ll(include_psd_info=True)))
        return

def make_gmm(gb, gmm_info_in):
    gmm_info = FullGaussianMixtureModel(gb, *gmm_info_in, use_cupy=True)
    
    probs_in = {
        (0, 1, 2, 4, 6, 7): gmm_info,
        3: uniform_dist(0.0, 2 * np.pi, use_cupy=True),
        5: uniform_dist(0.0, np.pi, use_cupy=True)
    }
    gen_dist = ProbDistContainer(probs_in, use_cupy=True)
    return gen_dist


def run_gb_pe(gpu, comm, head_rank, save_plot_rank):

    gpus_pe = [gpu]
    gpus = gpus_pe

    xp.cuda.runtime.setDevice(gpus[0])

    gb = GBGPU(use_gpu=True)
    # from lisatools.sampling.stopping import SearchConvergeStopping2
    gf_information = comm.recv(source=head_rank, tag=255)

    gb_info = gf_information.gb_info
    band_edges = gb_info["band_edges"]
    
    num_sub_bands = len(band_edges)

    nwalkers_pe = gb_info["pe_info"]["nwalkers"]
    ntemps_pe = gb_info["pe_info"]["ntemps"]
    betas = gb_info["pe_info"]["betas"]

    last_sample = gb_info["last_state"]

    adjust_temps = False
    if hasattr(last_sample, "band_info"):
        band_info_check = deepcopy(last_sample.band_info)
        adjust_temps = True
        del last_sample.band_info
    
    if not hasattr(last_sample, "band_info"):
        band_temps = np.tile(np.asarray(betas), (len(band_edges) - 1, 1))
        last_sample.initialize_band_information(nwalkers_pe, ntemps_pe, band_edges, band_temps)
        if adjust_temps:
            last_sample.band_info["band_temps"][:] = band_info_check["band_temps"][0, :]

    import time
    st = time.perf_counter()
    generated_info = gf_information.get_data_psd(include_gbs=False, include_ll=True, include_source_only_ll=True, n_gen_in=nwalkers_pe, return_prior_val=True, fix_val_in_gen=["gb"])
    et = time.perf_counter()

    print("Read in", et - st)
    data = generated_info["data"]
    psd = generated_info["psd"]
    lisasens = generated_info["lisasens"]

    df = gf_information.general_info["df"]

    A_going_in = np.repeat(data[0], 2, axis=0).reshape(nwalkers_pe, 2, gf_information.general_info["data_length"]).transpose(1, 0, 2)
    E_going_in = np.repeat(data[1], 2, axis=0).reshape(nwalkers_pe, 2, gf_information.general_info["data_length"]).transpose(1, 0, 2)

    A_psd_in = np.repeat(psd[0], 1, axis=0).reshape(nwalkers_pe, 1, gf_information.general_info["data_length"]).transpose(1, 0, 2)
    E_psd_in = np.repeat(psd[1], 1, axis=0).reshape(nwalkers_pe, 1, gf_information.general_info["data_length"]).transpose(1, 0, 2)
    
    A_lisasens_in = np.repeat(lisasens[0], 1, axis=0).reshape(nwalkers_pe, 1, gf_information.general_info["data_length"]).transpose(1, 0, 2)
    E_lisasens_in = np.repeat(lisasens[1], 1, axis=0).reshape(nwalkers_pe, 1, gf_information.general_info["data_length"]).transpose(1, 0, 2)
    
    mgh = MultiGPUDataHolder(
        gpus,
        A_going_in,
        E_going_in,
        A_going_in, # store as base
        E_going_in, # store as base
        A_psd_in,
        E_psd_in,
        A_lisasens_in,
        E_lisasens_in,
        df,
        base_injections=[gf_information.general_info["A_inj"], gf_information.general_info["E_inj"]],
        base_psd=None,  # [psd.copy(), psd.copy()]
    )
    ll_c = mgh.get_ll()

    mempool.free_all_blocks()

    # setup data streams to add to and subtract from
    supps_shape_in = (ntemps_pe, nwalkers_pe)

    gb.gpus = mgh.gpus

    priors = {"gb": gb_info["priors"]}
    nleaves_max = last_sample.branches["gb"].shape[2]
    transform_fn = gb_info["transform"]

    band_mean_f = (band_edges[1:] + band_edges[:-1]) / 2
    
    waveform_kwargs = gb_info["pe_info"]["pe_waveform_kwargs"].copy()
    if "N" in waveform_kwargs:
        waveform_kwargs.pop("N")

    from gbgpu.utils.utility import get_N
    band_N_vals = xp.asarray(get_N(np.full_like(band_mean_f, 1e-30), band_mean_f, waveform_kwargs["T"], waveform_kwargs["oversample"]))

    if last_sample.branches["gb"].inds[0].sum() > 0:
        
        # from lisatools.sampling.prior import SNRPrior, AmplitudeFromSNR
        # L = 2.5e9
        # amp_transform = AmplitudeFromSNR(L, gf_information.general_info['Tobs'], fd=gf_information.general_info["fd"])

        # walker_inds = np.repeat(np.arange(nwalkers_pe)[:, None], ntemps_pe * nleaves_max, axis=-1).reshape(nwalkers_pe, ntemps_pe, nleaves_max).transpose(1, 0, 2)[last_sample.branches["gb"].inds]
        
        # coords_fix = last_sample.branches["gb"].coords[last_sample.branches["gb"].inds]
        # coords_fix[:, 0], _ = amp_transform(coords_fix[:, 0], coords_fix[:, 1] / 1e3, psds=psd[0], walker_inds=walker_inds)
        
        # last_sample.branches["gb"].coords[last_sample.branches["gb"].inds, 0] = coords_fix[:, 0]

        coords_out_gb = last_sample.branches["gb"].coords[0,
            last_sample.branches["gb"].inds[0]
        ]

        walker_inds = np.repeat(np.arange(nwalkers_pe)[:, None], nleaves_max, axis=-1)[last_sample.branches["gb"].inds[0]]
        
        check = priors["gb"].logpdf(coords_out_gb, psds=lisasens[0], walker_inds=walker_inds)

        if np.any(np.isinf(check)):
            raise ValueError("Starting priors are inf.")

        coords_out_gb[:, 3] = coords_out_gb[:, 3] % (2 * np.pi)
        coords_out_gb[:, 5] = coords_out_gb[:, 5] % (1 * np.pi)
        coords_out_gb[:, 6] = coords_out_gb[:, 6] % (2 * np.pi)
        
        coords_in_in = transform_fn.both_transforms(coords_out_gb)

        band_inds = np.searchsorted(band_edges, coords_in_in[:, 1], side="right") - 1

        walker_vals = np.tile(
            np.arange(nwalkers_pe), (nleaves_max, 1)
        ).transpose((1, 0))[last_sample.branches["gb"].inds[0]]

        data_index_1 = ((band_inds % 2) + 0) * nwalkers_pe + walker_vals

        data_index = xp.asarray(data_index_1).astype(
            xp.int32
        )

        # goes in as -h
        factors = -xp.ones_like(data_index, dtype=xp.float64)

        
        N_vals = band_N_vals[band_inds]

        print("before global template")
        # TODO: add test to make sure that the genertor send in the general information matches this one
        gb.generate_global_template(
            coords_in_in,
            data_index,
            mgh.data_list,
            data_length=mgh.data_length,
            factors=factors,
            data_splits=mgh.gpu_splits,
            N=N_vals,
            **waveform_kwargs,
        )
        print("after global template")
        del data_index
        del factors
        mempool.free_all_blocks()

    ll = np.tile(mgh.get_ll(include_psd_info=True), (ntemps_pe, 1))

    state_mix = GBState(
        last_sample.branches_coords,
        inds=last_sample.branches_inds,
        log_like=ll,
        # supplimental=supps,
        betas=last_sample.betas,
        band_info=last_sample.band_info
    )

    band_inds_in = np.zeros((ntemps_pe, nwalkers_pe, nleaves_max), dtype=int)
    N_vals_in = np.zeros((ntemps_pe, nwalkers_pe, nleaves_max), dtype=int)

    if state_mix.branches["gb"].inds.sum() > 0:
        f_in = state_mix.branches["gb"].coords[state_mix.branches["gb"].inds][:, 1] / 1e3
        band_inds_in[state_mix.branches["gb"].inds] = np.searchsorted(band_edges, f_in, side="right") - 1
        N_vals_in[state_mix.branches["gb"].inds] = band_N_vals.get()[band_inds_in[state_mix.branches["gb"].inds]]

    branch_supp_base_shape = (ntemps_pe, nwalkers_pe, nleaves_max)
    state_mix.branches["gb"].branch_supplimental = BranchSupplimental(
        {"N_vals": N_vals_in, "band_inds": band_inds_in}, base_shape=branch_supp_base_shape, copy=True
    )

    gpu_priors_in = deepcopy(priors["gb"].priors_in)
    for key, item in gpu_priors_in.items():
        item.use_cupy = True

    gpu_priors = {"gb": GBPriorWrap(gb_info["pe_info"]["ndim"], ProbDistContainer(gpu_priors_in, use_cupy=True))}

    gb_kwargs = dict(
        waveform_kwargs=waveform_kwargs,
        parameter_transforms=transform_fn,
        provide_betas=True,
        skip_supp_names_update=["group_move_points"],
        random_seed=gf_information.general_info["random_seed"],
        use_gpu=True,
        nfriends=nwalkers_pe,
        phase_maximize=gb_info["pe_info"]["in_model_phase_maximize"],
        **gb_info["pe_info"]["group_proposal_kwargs"]
    )

    fd = gf_information.general_info["fd"].copy()
    
    gb_args = (
        gb,
        priors,
        gf_information.general_info["start_freq_ind"],
        mgh.data_length,
        mgh,
        np.asarray(fd),
        band_edges,
        gpu_priors,
    )

    gb_move = GBSpecialStretchMove(
        *gb_args,
        **gb_kwargs,
    )

    # add the other
    gb_move.gb.gpus = gpus
    moves_in_model = [gb_move]

    rj_moves_in = []

    rj_moves_in_frac = []

    gb_kwargs_rj = dict(
        waveform_kwargs=waveform_kwargs,
        parameter_transforms=transform_fn,
        search=False,
        provide_betas=True,
        skip_supp_names_update=["group_move_points"],
        random_seed=gf_information.general_info["random_seed"],
        use_gpu=True,
        rj_proposal_distribution=gpu_priors,
        name="rj_prior",
        use_prior_removal=gb_info["pe_info"]["use_prior_removal"],
        nfriends=nwalkers_pe,
        phase_maximize=gb_info["pe_info"]["rj_phase_maximize"],
        **gb_info["pe_info"]["group_proposal_kwargs"]  # needed for it to work
    )

    gb_args_rj = (
        gb,
        priors,
        gf_information.general_info["start_freq_ind"],
        mgh.data_length,
        mgh,
        np.asarray(fd),
        band_edges,
        gpu_priors,
    )

    rj_move_prior = GBSpecialStretchMove(
        *gb_args_rj,
        **gb_kwargs_rj,
    )

    rj_moves_in.append(rj_move_prior)
    rj_moves_in_frac.append(gb_info["pe_info"]["rj_prior_fraction"])

    if state_mix.branches["gb"].inds.sum() == 0:
        # wait until we have a search distribution
        waiting = True
        while waiting:
            time.sleep(20.0)
            comm.send({"send": True, "no_binaries": True}, dest=head_rank, tag=50)
            new_info = comm.recv(source=head_rank, tag=51)
            # print("CHECKING:", new_info.gb_info["search_gmm_info"])
            if new_info.gb_info["search_gmm_info"] is not None:
                waiting = False
                gb_info = new_info.gb_info

    current_gmm_search_info = gb_info["search_gmm_info"]
    if current_gmm_search_info is not None:
        gen_dist_search = make_gmm(gb, current_gmm_search_info)
    else:
        gen_dist_search = gpu_priors["gb"]

    gb_kwargs_rj_2 = gb_kwargs_rj.copy()
    gb_kwargs_rj_2["rj_proposal_distribution"] = {"gb": gen_dist_search}
    gb_kwargs_rj_2["name"] = "rj_search_gmm"

    rj_move_search = GBSpecialStretchMove(
        *gb_args_rj,
        **gb_kwargs_rj_2,
    )

    rj_moves_in.append(rj_move_search)
    rj_moves_in_frac.append(gb_info["pe_info"]["rj_search_fraction"])

    current_gmm_refit_info = gb_info["search_gmm_info"]

    if current_gmm_refit_info is not None:
        gen_dist_refit = make_gmm(gb, current_gmm_refit_info)
    elif current_gmm_search_info is not None:
        gen_dist_refit = make_gmm(gb, current_gmm_search_info)
    else:
        gen_dist_refit = gpu_priors["gb"]
    
    gb_kwargs_rj_3 = gb_kwargs_rj.copy()

    gb_kwargs_rj_3["rj_proposal_distribution"] = {"gb": gen_dist_refit}
    gb_kwargs_rj_3["name"] = "rj_refit_gmm"

    rj_move_refit = GBSpecialStretchMove(
        *gb_args_rj,
        **gb_kwargs_rj_3,
    )

    rj_moves_in.append(rj_move_refit)
    # TODO: check for this in updates
    rj_moves_in_frac.append(gb_info["pe_info"]["rj_refit_fraction"])

    total_frac = sum(rj_moves_in_frac)

    rj_moves = [(rj_move_i, rj_move_frac_i / total_frac) for rj_move_i, rj_move_frac_i in zip(rj_moves_in, rj_moves_in_frac)]

    for rj_move in rj_moves:
        rj_move[0].gb.gpus = gpus

    like_mix = BasicResidualMGHLikelihood(mgh)
    branch_names = ["gb"]

    update = UpdateNewResiduals(
        mgh, gb, comm, head_rank, gpu_priors, generated_info["psd_prior_vals"] + generated_info["mbh_prior_vals"], verbose=False
    )

    ndims = {"gb": gb_info["pe_info"]["ndim"]}
    nleaves_max = {"gb": gb_info["pe_info"]["nleaves_max"]}
        
    moves = moves_in_model + rj_moves
    backend = GBHDFBackend(
        gf_information.general_info["file_information"]["fp_gb_pe"],
        compression="gzip",
        compression_opts=9,
        comm=comm,
        save_plot_rank=save_plot_rank
    )
    
    if not backend.initialized:
        backend.reset(
            nwalkers_pe,
            ndims,
            nleaves_max=nleaves_max,
            ntemps=ntemps_pe,
            branch_names=branch_names,
            nbranches=len(branch_names),
            rj=True,
            moves=None,
            num_bands=len(band_edges) - 1,
            band_edges=band_edges
        )

    # state_mix.random_state = np.random.get_state()
    # state_mix.log_prior = np.zeros_like(state_mix.log_like)
    # backend.save_step(state_mix, np.zeros((ntemps_pe, nwalkers_pe), dtype=int), rj_accepted=np.zeros((ntemps_pe, nwalkers_pe), dtype=int))
    stopping_fn = gb_info["pe_info"]["stopping_function"]

    if hasattr(stopping_fn, "add_comm"):
        stopping_fn.add_comm(comm)

    if stopping_fn.stop_fn is not None and hasattr(stopping_fn.stop_fn, "add_mgh"):
        stopping_fn.stop_fn.add_mgh(mgh)

    stopping_iterations = gb_info["pe_info"]["stopping_iterations"]
    thin_by = gb_info["pe_info"]["thin_by"]

    sampler_mix = EnsembleSampler(
        nwalkers_pe,
        ndims,  # assumes ndim_max
        like_mix,
        priors,
        tempering_kwargs={"betas": betas, **gb_info["pe_info"]["other_tempering_kwargs"]},
        nbranches=len(branch_names),
        nleaves_max=nleaves_max,
        moves=moves_in_model,
        rj_moves=rj_moves,
        kwargs=None,  # {"start_freq_ind": start_freq_ind, **waveform_kwargs},
        backend=backend,
        vectorize=True,
        periodic={"gb": gb_info["periodic"]},  # TODO: add periodic to proposals
        branch_names=gb_info["pe_info"]["branch_names"],
        update_fn=update,  # stop_converge_mix,
        update_iterations=gb_info["pe_info"]["update_iterations"],
        provide_groups=True,
        provide_supplimental=True,
        track_moves=False,
        stopping_fn=stopping_fn,
        stopping_iterations=stopping_iterations,
    )

    nsteps_mix = gb_info["pe_info"]["nsteps"]

    # with open(current_save_state_file, "wb") as fp:
    #     pickle.dump(state_mix, fp, pickle.HIGHEST_PROTOCOL)

    print("Starting mix ll best:", state_mix.log_like.max(axis=-1))
    mempool.free_all_blocks()
    
    out = sampler_mix.run_mcmc(
        state_mix, nsteps_mix, store=True, progress=gb_info["pe_info"]["progress"], thin_by=gb_info["pe_info"]["thin_by"]
    )
    print("ending mix ll best:", out.log_like.max(axis=-1))

    # communicate end of run to head process
    comm.send({"finish_run": True}, dest=backend.save_plot_rank, tag=90)
    comm.send({"finish_run": True}, dest=head_rank, tag=50)

    del mgh
    
    mempool.free_all_blocks()
    return
    

def shuffle_along_axis(a, axis, xp=None):
    if xp is None:
        xp = np
    idx = xp.random.rand(*a.shape).argsort(axis=axis)
    return xp.take_along_axis(a,idx,axis=axis)

def fit_gmm(samples, comm, comm_info):

    if len(samples) == 0:
        return None

    keep = np.array([0, 1, 2, 4, 6, 7])

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
                if np.any(np.isnan(samples[keep_samp, 3:])) or np.any(np.isinf(samples[keep_samp, 3:])):
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
                print(f"ITERATION: {outer_iteration}, need:", np.sum(~gmm_complete), current_status)

            for proc_i, proc_rank in enumerate(process_ranks_for_fit):
                # time.sleep(0.6)
                if current_status[proc_i]:
                    rec_tag = int(str(proc_rank) + "4545")
                    check_output = comm.irecv(source=proc_rank, tag=rec_tag)

                    if not check_output.get_status():
                        check_output.cancel()
                    else:
                        # first two give some delay for the processor that messes up
                        try:
                            output_info = check_output.wait()
                        except (pickle.UnpicklingError, UnicodeDecodeError, ValueError, OverflowError) as e:
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
                    while current_send_arg_index < len(args_tmp) and gmm_complete[current_send_arg_index]:
                        current_send_arg_index += 1

                    if current_send_arg_index < len(args_tmp):
                        send_info = {"samples": args_tmp[current_send_arg_index], "arg": current_send_arg_index}
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


def fit_each_leaf(rank, gather_rank, rec_tag, send_tag, comm):

    run_process = True

    rec_tag = int(str(rank) + "67676")
    send_tag = int(str(rank) + "4545")
    while run_process:
        try:
            check = comm.recv(source=gather_rank, tag=rec_tag)
        except (pickle.UnpicklingError, UnicodeDecodeError, ValueError, OverflowError) as e:
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
            comm.send({"BAD":  "KeyError"}, dest=gather_rank, tag=send_tag)
            continue

        assert isinstance(samples, np.ndarray)

        run = True
        min_bic = np.inf
        sample_mins = samples.min(axis=0)
        sample_maxs = samples.max(axis=0)
        samples[:] = ((samples - sample_mins) / (sample_maxs - sample_mins)) * 2 - 1
        bad = False
        for n_components in range(1, 20):
            if not run:
                continue
            #fit_gaussian_mixture_model(n_components, samples)
            #breakpoint()
            try:
                mixture = GaussianMixture(n_components=n_components, verbose=False, verbose_interval=2)

                mixture.fit(samples)
                test_bic = mixture.bic(samples)
            except ValueError:
                # print("ValueError", samples)
                run = False
                bad = True
                continue
            # print(n_components, test_bic)
            if test_bic < min_bic:
                min_bic = test_bic
                keep_mix = mixture
                keep_components = n_components
                
            else:
                run = False

                # print(leaf, n_components - 1, et - st)
            
            """if keep_components >= 9:
                new_samples = keep_mix.sample(n_samples=100000)[0]
                old_samples = samples
                fig = corner.corner(old_samples, hist_kwargs=dict(density=True, color="r"), color="r", plot_datapoints=False, plot_density=False)
                corner.corner(new_samples, hist_kwargs=dict(density=True, color="b"), color="b", plot_datapoints=False, plot_contours=True, plot_density=False, fig=fig)
                fig.savefig("mix_check.png")
                plt.close()
                breakpoint()"""

        if bad:
            comm.send({"BAD": "ValueError"}, dest=gather_rank, tag=send_tag)
            continue
        if keep_components >= 19:
            print(keep_components)
        output_list = [keep_mix.weights_, keep_mix.means_, keep_mix.covariances_, np.array([np.linalg.inv(keep_mix.covariances_[i]) for i in range(len(keep_mix.weights_))]), np.array([np.linalg.det(keep_mix.covariances_[i]) for i in range(len(keep_mix.weights_))]), sample_mins, sample_maxs]
        comm.send({"output": output_list, "rank": rank, "arg": arg_index}, dest=gather_rank, tag=send_tag)
    return

def run_iterative_subtraction_mcmc(current_info, gpu, ndim, nwalkers, ntemps, band_inds_running, priors_good, f0_maxs, f0_mins, fdot_maxs, fdot_mins, data_in, psd_in, lisasens_in, comm, comm_info):
    
    xp.cuda.runtime.setDevice(gpu)
    gb = GBGPU(use_gpu=True)
    temperature_control = TemperatureControl(ndim, nwalkers, ntemps=ntemps)

    # TODO: clean this up
    gb_info = current_info.gb_info
    df = current_info.general_info["df"]

    num_max_proposals = 100000
    convergence_iter_count = 500

    periodic = {"gb": gb_info["periodic"]}
    move_proposal = StretchMove(periodic=PeriodicContainer(periodic), temperature_control=temperature_control, return_gpu=True, use_gpu=True)
    
    band_inds_here = xp.where(xp.asarray(band_inds_running))[0]

    new_points = priors_good.rvs(size=(ntemps, nwalkers, len(band_inds_here)))  # , psds=lisasens_in[0][None, :])

    fix = xp.any(xp.isinf(new_points), axis=-1) | xp.any(xp.isnan(new_points), axis=-1)
    while xp.any(fix):
        tmp = priors_good.rvs(size=int((fix.flatten() == True).sum()))
        new_points[fix == True] = tmp
        fix = xp.any(xp.isinf(new_points), axis=-1) | xp.any(xp.isnan(new_points), axis=-1)

    # TODO: fix fs stuff
    prev_logp = priors_good.logpdf(new_points.reshape(-1, ndim)).reshape(new_points.shape[:-1])
    assert not xp.any(xp.isinf(prev_logp))
    new_points_with_fs = new_points.copy()

    L = 2.5e9
    amp_transform = AmplitudeFromSNR(L, current_info.general_info['Tobs'], fd=current_info.general_info["fd"], sens_fn="lisasens", use_cupy=True)

    original_snr_params = new_points_with_fs[:, :, :, 0].copy()
    new_points_with_fs[:, :, :, 1] = (f0_maxs[band_inds_here] - f0_mins[band_inds_here]) * new_points_with_fs[:, :, :, 1] + f0_mins[band_inds_here]
    new_points_with_fs[:, :, :, 2] = (fdot_maxs[band_inds_here] - fdot_mins[band_inds_here]) * new_points_with_fs[:, :, :, 2] + fdot_mins[band_inds_here]
    new_points_with_fs[:, :, :, 0] = amp_transform(new_points_with_fs[:, :, :, 0].flatten(), new_points_with_fs[:, :, :, 1].flatten() / 1e3, psds=lisasens_in[0][None, :])[0].reshape(new_points_with_fs.shape[:-1])

    lp_factors = np.log(original_snr_params / new_points_with_fs[:, :, :, 0])
    prev_logp += lp_factors
    transform_fn = gb_info["transform"]

    new_points_in = transform_fn.both_transforms(new_points_with_fs.reshape(-1, ndim), xp=xp).reshape(new_points_with_fs.shape[:-1] + (ndim + 1,)).reshape(-1, ndim + 1)
    inner_product = 4 * df * (xp.sum(data_in[0].conj() * data_in[0] / psd_in[0]) + xp.sum(data_in[1].conj() * data_in[1] / psd_in[1])).real
    ll = (-1/2 * inner_product - xp.sum(xp.log(xp.asarray(psd_in)))).item()
    gb.d_d = inner_product

    start_ll = -1/2 * inner_product
    print(ll)

    waveform_kwargs = gb_info["waveform_kwargs"].copy()
    if "N" in waveform_kwargs:
        waveform_kwargs.pop("N")

    prev_logl = xp.asarray(gb.get_ll(new_points_in, data_in, psd_in, phase_marginalize=True, **waveform_kwargs).reshape(prev_logp.shape))

    if xp.any(xp.isnan(prev_logl)):
        breakpoint()

    old_points = new_points.copy()
    
    best_logl = prev_logl.max(axis=(0, 1))
    best_logl_ind = prev_logl.reshape(ntemps * nwalkers, len(band_inds_here)).argmax(axis=0)
    
    best_logl_coords = old_points.reshape(ntemps * nwalkers, len(band_inds_here), ndim)[(best_logl_ind, xp.arange(len(band_inds_here)))]

    start_best_logl = best_logl.copy()

    temp_guide = xp.repeat(xp.arange(ntemps)[:, None], nwalkers * len(band_inds_here), axis=-1).reshape(ntemps, nwalkers, len(band_inds_here))
    walker_guide = xp.repeat(xp.arange(nwalkers)[:, None], ntemps * len(band_inds_here), axis=-1).reshape(nwalkers, ntemps, len(band_inds_here)).transpose(1, 0, 2)
    band_guide = xp.repeat(xp.arange(len(band_inds_here))[None, :], ntemps * nwalkers, axis=0).reshape(ntemps, nwalkers, len(band_inds_here))

    still_going_here = xp.ones(len(band_inds_here), dtype=bool)
    num_proposals_per = np.zeros_like(still_going_here, dtype=int)
    iter_count = np.zeros_like(still_going_here, dtype=int)
    betas = xp.repeat(xp.asarray(temperature_control.betas[:, None].copy()), len(band_inds_here), axis=-1)

    run_number = 0
    for prop_i in range(num_max_proposals):  # tqdm(range(num_max_proposals)):
        # st = time.perf_counter()
        num_still_going_here = still_going_here.sum().item()
        inds_split = np.arange(nwalkers)
    
        np.random.shuffle(inds_split)
        
        for split in range(2):
            inds_here = np.arange(nwalkers)[inds_split % 2 == split]
            inds_not_here = np.delete(np.arange(nwalkers), inds_here)

            inds_here = xp.asarray(inds_here)
            inds_not_here = xp.asarray(inds_not_here)

            s_in = old_points[:, inds_here][:, :, still_going_here].transpose(0, 2, 1, 3).reshape((ntemps * num_still_going_here, int(nwalkers/2), 1, -1))
            c_in = [old_points[:, inds_not_here][:, :, still_going_here].transpose(0, 2, 1, 3).reshape((ntemps * num_still_going_here, int(nwalkers/2), 1, -1))]

            temps_here = temp_guide[:, inds_here][:, :, still_going_here]
            walkers_here = walker_guide[:, inds_here][:, :, still_going_here]
            bands_here = band_guide[:, inds_here][:, :, still_going_here]

            new_points_dict, factors = move_proposal.get_proposal({"gb": s_in}, {"gb": c_in}, xp.random)
            new_points = new_points_dict["gb"].reshape(ntemps, num_still_going_here, int(nwalkers/2), -1).transpose(0, 2, 1, 3)
            logp = priors_good.logpdf(new_points.reshape(-1, ndim)).reshape(new_points.shape[:-1])

            # TODO: make sure factors are reshaped properly
            factors = factors.reshape(logp.shape)
            keep_logp = ~xp.isinf(logp)

            new_points_with_fs = new_points.copy()

            original_snr_params = new_points_with_fs[:, :, :, 0].copy()
            new_points_with_fs[:, :, :, 1] = (f0_maxs[None, None, band_inds_here[still_going_here]] - f0_mins[None, None, band_inds_here[still_going_here]]) * new_points_with_fs[:, :, :, 1] + f0_mins[None, None, band_inds_here[still_going_here]]
            new_points_with_fs[:, :, :, 2] = (fdot_maxs[None, None, band_inds_here[still_going_here]] - fdot_mins[None, None, band_inds_here[still_going_here]]) * new_points_with_fs[:, :, :, 2] + fdot_mins[None, None, band_inds_here[still_going_here]]
            new_points_with_fs[:, :, :, 0] = amp_transform(new_points_with_fs[:, :, :, 0].flatten(), new_points_with_fs[:, :, :, 1].flatten() / 1e3, psds=lisasens_in[0][None, :])[0].reshape(new_points_with_fs.shape[:-1])

            lp_factors = np.log(original_snr_params / new_points_with_fs[:, :, :, 0])
            logp += lp_factors

            new_points_with_fs_keep = new_points_with_fs[keep_logp]
            new_points_in = transform_fn.both_transforms(new_points_with_fs_keep, xp=xp)

            logl = xp.full_like(logp, -1e300)

            logl[keep_logp] = xp.asarray(gb.get_ll(new_points_in, data_in, psd_in, phase_marginalize=True, **waveform_kwargs))

            # fix any nans that may come up
            logl[xp.isnan(logl)] = -1e300

            xp.cuda.runtime.deviceSynchronize()
            
            prev_logl_here = prev_logl[:, inds_here][:, :, still_going_here]
            prev_logp_here = prev_logp[:, inds_here][:, :, still_going_here]
            
            prev_logP_here = betas[:, still_going_here][:, None, :] * prev_logl_here + prev_logp_here

            logP = betas[:, still_going_here][:, None, :] * logl + logp

            lnpdiff = factors + logP - prev_logP_here
            keep = lnpdiff > xp.asarray(xp.log(xp.random.rand(*logP.shape)))

            prev_logp[temps_here[keep], walkers_here[keep], bands_here[keep]] = logp[keep]
            prev_logl[temps_here[keep], walkers_here[keep], bands_here[keep]] = logl[keep]
            old_points[temps_here[keep], walkers_here[keep], bands_here[keep]] = new_points[keep]
            
        original_snr_params = new_points_with_fs[:, :, :, 0].copy()

        # prepare information on how many swaps are accepted this time
        swaps_accepted = xp.zeros((ntemps - 1, num_still_going_here), dtype=int)
        swaps_proposed = xp.full_like(swaps_accepted, nwalkers)

        # iterate from highest to lowest temperatures
        for i in range(ntemps - 1, 0, -1):
            # get both temperature rungs
            bi = betas[i, still_going_here]
            bi1 = betas[i - 1, still_going_here]

            # difference in inverse temps
            dbeta = bi1 - bi

            # permute the indices for the walkers in each temperature to randomize swap positions
            iperm = shuffle_along_axis(xp.tile(xp.arange(nwalkers), (num_still_going_here, 1)), -1)
            i1perm = shuffle_along_axis(xp.tile(xp.arange(nwalkers), (num_still_going_here, 1)), -1)
            
            # random draw that produces log of the acceptance fraction
            raccept = xp.log(xp.random.uniform(size=(num_still_going_here, nwalkers)))
            
            # log of the detailed balance fraction
            walker_swap_i = iperm.flatten()
            walker_swap_i1 = i1perm.flatten()

            temp_swap_i = np.full_like(walker_swap_i, i)
            temp_swap_i1 = np.full_like(walker_swap_i1, i - 1)
            band_swap = xp.repeat(xp.arange(len(still_going_here))[still_going_here], nwalkers)

            paccept = dbeta[:, None] * (
                prev_logl[(temp_swap_i, walker_swap_i, band_swap)].reshape(num_still_going_here, nwalkers)
                - prev_logl[(temp_swap_i1, walker_swap_i1, band_swap)].reshape(num_still_going_here, nwalkers)
            )

            # How many swaps were accepted
            sel = paccept > raccept
            swaps_accepted[i - 1] = xp.sum(sel, axis=-1)

            temp_swap_i_keep = temp_swap_i[sel.flatten()]
            walker_swap_i_keep = walker_swap_i[sel.flatten()]
            band_swap_keep = band_swap[sel.flatten()]

            temp_swap_i1_keep = temp_swap_i1[sel.flatten()]
            walker_swap_i1_keep = walker_swap_i1[sel.flatten()]

            coords_tmp_i = old_points[(temp_swap_i_keep, walker_swap_i_keep, band_swap_keep)].copy()
            logl_tmp_i = prev_logl[(temp_swap_i_keep, walker_swap_i_keep, band_swap_keep)].copy()
            logp_tmp_i = prev_logp[(temp_swap_i_keep, walker_swap_i_keep, band_swap_keep)].copy()

            old_points[(temp_swap_i_keep, walker_swap_i_keep, band_swap_keep)] = old_points[(temp_swap_i1_keep, walker_swap_i1_keep, band_swap_keep)]
            prev_logl[(temp_swap_i_keep, walker_swap_i_keep, band_swap_keep)] = prev_logl[(temp_swap_i1_keep, walker_swap_i1_keep, band_swap_keep)]
            prev_logp[(temp_swap_i_keep, walker_swap_i_keep, band_swap_keep)] = prev_logp[(temp_swap_i1_keep, walker_swap_i1_keep, band_swap_keep)]

            old_points[(temp_swap_i1_keep, walker_swap_i1_keep, band_swap_keep)] = coords_tmp_i
            prev_logl[(temp_swap_i1_keep, walker_swap_i1_keep, band_swap_keep)] = logl_tmp_i
            prev_logp[(temp_swap_i1_keep, walker_swap_i1_keep, band_swap_keep)] = logp_tmp_i
        # print(prev_logl.max(axis=(1, 2)))
        
        # print(time.perf_counter() - st)
        ratios = swaps_accepted / swaps_proposed
        # adjust temps 
        betas0 = betas[:, still_going_here].copy()
        betas1 = betas[:, still_going_here].copy()

        # Modulate temperature adjustments with a hyperbolic decay.
        decay = temperature_control.adaptation_lag / (prop_i + temperature_control.adaptation_lag)
        kappa = decay / temperature_control.adaptation_time

        # Construct temperature adjustments.
        dSs = kappa * (ratios[:-1] - ratios[1:])

        # Compute new ladder (hottest and coldest chains don't move).
        deltaTs = xp.diff(1 / betas1[:-1], axis=0)
        deltaTs *= xp.exp(dSs)
        betas1[1:-1] = 1 / (np.cumsum(deltaTs, axis=0) + 1 / betas1[0])

        dbetas = betas1 - betas0
        betas[:, still_going_here] += dbetas

        new_best_logl = prev_logl.max(axis=(0, 1))

        improvement = (new_best_logl - best_logl > 0.01)

        # print(new_best_logl - best_logl, best_logl)
        best_logl[improvement] = new_best_logl[improvement]

        best_logl_ind = prev_logl.reshape(ntemps * nwalkers, len(band_inds_here)).argmax(axis=0)[improvement]
        best_logl_coords[improvement] = old_points.reshape(ntemps * nwalkers, len(band_inds_here), ndim)[(best_logl_ind, xp.arange(len(band_inds_here))[improvement])]

        best_binaries_coords_with_fs = best_logl_coords.copy()

        best_binaries_coords_with_fs[:, 1] = (f0_maxs[band_inds_here] - f0_mins[band_inds_here]) * best_binaries_coords_with_fs[:, 1] + f0_mins[band_inds_here]
        best_binaries_coords_with_fs[:, 2] = (fdot_maxs[band_inds_here] - fdot_mins[band_inds_here]) * best_binaries_coords_with_fs[:, 2] + fdot_mins[band_inds_here]
        best_binaries_coords_with_fs[:, 0] = amp_transform(best_binaries_coords_with_fs[:, 0].flatten(), best_binaries_coords_with_fs[:, 1].flatten() / 1e3, psds=lisasens_in[0][None, :])[0].reshape(best_binaries_coords_with_fs.shape[:-1])

        best_logl_points_in = transform_fn.both_transforms(best_binaries_coords_with_fs, xp=xp)

        best_logl_check = xp.asarray(gb.get_ll(best_logl_points_in, data_in, psd_in, phase_marginalize=True, **waveform_kwargs))

        if prop_i > convergence_iter_count:
            iter_count[improvement] = 0
            iter_count[~improvement] += 1

        num_proposals_per[still_going_here] += 1
        still_going_here[iter_count >= convergence_iter_count] = False
        
        if prop_i % convergence_iter_count == 0:
            print(f"Proposal {prop_i}, Still going:", still_going_here.sum().item())  # , still_going_here[825], np.sort(prev_logl[0, :, 825] - start_ll))
        if run_number == 2:
            iter_count[:] = 0
            collect_sample_check_iter += 1
            if collect_sample_check_iter % thin_by == 0:
                coords_with_fs = old_points.transpose(2, 0, 1, 3)[still_going_here, 0, :].copy()
                coords_with_fs[:, :, 1] = (f0_maxs[band_inds_here[still_going_here]] - f0_mins[band_inds_here][still_going_here])[:, None] * coords_with_fs[:, :, 1] + f0_mins[band_inds_here[still_going_here]][:, None]
                coords_with_fs[:, :, 2] = (fdot_maxs[band_inds_here[still_going_here]] - fdot_mins[band_inds_here][still_going_here])[:, None] * coords_with_fs[:, :, 2] + fdot_mins[band_inds_here[still_going_here]][:, None]
                coords_with_fs[:, :, 0] = amp_transform(coords_with_fs[:, :, 0].flatten(), coords_with_fs[:, :, 1].flatten() / 1e3, psds=lisasens_in[0][None, :])[0].reshape(coords_with_fs.shape[:-1])
    
                samples_store[:, collect_sample_iter] = coords_with_fs.get()
                collect_sample_iter += 1
                print(collect_sample_iter, num_samples_store)
                if collect_sample_iter == num_samples_store:
                    still_going_here[:] = False

        if still_going_here.sum().item() == 0:
            if run_number < 2:
                betas = xp.repeat(xp.asarray(temperature_control.betas[:, None].copy()), len(band_inds_here), axis=-1)
                
                old_points_old = old_points.copy()
                old_points[:] = best_logl_coords[None, None, :]

                gen_points = old_points.transpose(2, 0, 1, 3).reshape(best_logl_coords.shape[0], -1, ndim).copy()
                iter_count[:] = 0
                still_going_here[:] = True

                factor = 1e-5
                cov = xp.ones(ndim) * 1e-3
                cov[1] = 1e-8

                still_going_start_like = xp.ones(best_logl_coords.shape[0], dtype=bool)
                starting_points = np.zeros((best_logl_coords.shape[0], nwalkers * ntemps, ndim))

                iter_check = 0
                max_iter = 10000
                while np.any(still_going_start_like) and iter_check < max_iter:
                    num_still_going_start_like = still_going_start_like.sum().item()
                    
                    start_like = np.zeros((num_still_going_start_like, nwalkers * ntemps))
                
                    logp = np.full_like(start_like, -np.inf)
                    tmp = xp.zeros((num_still_going_start_like, ntemps * nwalkers, ndim))
                    fix = xp.ones((num_still_going_start_like, ntemps * nwalkers), dtype=bool)
                    while xp.any(fix):
                        tmp[fix] = (gen_points[still_going_start_like, :] * (1. + factor * cov * xp.random.randn(num_still_going_start_like, nwalkers * ntemps, ndim)))[fix]

                        tmp[:, :, 3] = tmp[:, :, 3] % (2 * np.pi)
                        tmp[:, :, 5] = tmp[:, :, 5] % (np.pi)
                        tmp[:, :, 6] = tmp[:, :, 6] % (2 * np.pi)
                        logp = priors_good.logpdf(tmp.reshape(-1, ndim)).reshape(tmp.shape[:-1])

                        fix = xp.isinf(logp)
                        if xp.all(fix):
                            factor /= 10.0

                    new_points_with_fs = tmp.copy()

                    original_snr_params = new_points_with_fs[:, :, 0].copy()
                    new_points_with_fs[:, :, 1] = (f0_maxs[None, band_inds_here[still_going_start_like]] - f0_mins[None, band_inds_here[still_going_start_like]]).T * new_points_with_fs[:, :, 1] + f0_mins[None, band_inds_here[still_going_start_like]].T
                    new_points_with_fs[:, :, 2] = (fdot_maxs[None, band_inds_here[still_going_start_like]] - fdot_mins[None, band_inds_here[still_going_start_like]]).T * new_points_with_fs[:, :, 2] + fdot_mins[None, band_inds_here[still_going_start_like]].T
                    new_points_with_fs[:, :, 0] = amp_transform(new_points_with_fs[:, :, 0].flatten(), new_points_with_fs[:, :, 1].flatten() / 1e3, psds=lisasens_in[0][None, :])[0].reshape(new_points_with_fs.shape[:-1])
 
                    lp_factors = np.log(original_snr_params / new_points_with_fs[:, :, 0])
                    logp += lp_factors
                    new_points_in = transform_fn.both_transforms(new_points_with_fs.reshape(-1, ndim), xp=xp)

                    start_like = xp.asarray(gb.get_ll(new_points_in, data_in, psd_in, phase_marginalize=True, **waveform_kwargs)).reshape(new_points_with_fs.shape[:-1])

                    old_points[:, :, still_going_start_like, :] = tmp.transpose(1, 0, 2).reshape(ntemps, nwalkers, -1, ndim)
                    prev_logl[:, :, still_going_start_like] = start_like.T.reshape(ntemps, nwalkers, -1)
                    prev_logp[:, :, still_going_start_like] = logp.T.reshape(ntemps, nwalkers, -1)
                    # fix any nans that may come up
                    start_like[xp.isnan(start_like)] = -1e300
                    
                    update = xp.arange(still_going_start_like.shape[0])[still_going_start_like][xp.std(start_like, axis=-1) > 15.0]
                    still_going_start_like[update] = False 

                    iter_check += 1
                    factor *= 1.5
                    
                    # if still_going_start_like[400]:
   
                    #     ind_check = np.where(np.arange(still_going_start_like.shape[0])[still_going_start_like] == 400)[0]
                    #     print(iter_check, still_going_start_like.sum(), start_like[ind_check].max(axis=-1), start_like[ind_check].min(axis=-1), start_like[ind_check].max(axis=-1) - start_like[ind_check].min(axis=-1), xp.std(start_like, axis=-1)[ind_check])

                # breakpoint()
                if run_number == 1:
                    best_binaries_coords_with_fs = best_logl_coords.copy()

                    best_binaries_coords_with_fs[:, 1] = (f0_maxs[band_inds_here] - f0_mins[band_inds_here]) * best_binaries_coords_with_fs[:, 1] + f0_mins[band_inds_here]
                    best_binaries_coords_with_fs[:, 2] = (fdot_maxs[band_inds_here] - fdot_mins[band_inds_here]) * best_binaries_coords_with_fs[:, 2] + fdot_mins[band_inds_here]
                    best_binaries_coords_with_fs[:, 0] = amp_transform(best_binaries_coords_with_fs[:, 0].flatten(), best_binaries_coords_with_fs[:, 1].flatten() / 1e3, psds=lisasens_in[0][None, :])[0].reshape(best_binaries_coords_with_fs.shape[:-1])
            
                    best_logl_points_in = transform_fn.both_transforms(best_binaries_coords_with_fs, xp=xp)

                    best_logl_check = xp.asarray(gb.get_ll(best_logl_points_in, data_in, psd_in, phase_marginalize=True, **waveform_kwargs))

                    if not xp.allclose(best_logl, best_logl_check):
                        breakpoint()

                    snr_lim = gb_info["search_info"]["snr_lim"]
                    keep_binaries = gb.d_h / xp.sqrt(gb.h_h.real) > snr_lim

                    print(f"SNR lim: {snr_lim}")
                    
                    still_going_here = keep_binaries.copy()

                    num_new_binaries = keep_binaries.sum().item()
                    print(f"num new search: {num_new_binaries}")

                    thin_by = 25
                    num_samples_store = 30
                    samples_store = np.zeros((still_going_here.sum().item(), num_samples_store, nwalkers, ndim))
                    collect_sample_iter = 0
                    collect_sample_check_iter = 0

                    # # TODO: add in based on sensitivity changing
                    # # band_inds_running[band_inds_here[~keep_binaries].get()] = False
                    # keep_coords = best_binaries_coords_with_fs[keep_binaries].get()

                    # # adjust the phase from marginalization
                    # phase_change = np.angle(gb.non_marg_d_h)[keep_binaries.get()]
                    # keep_coords[:, 3] -= phase_change
                    # # best_logl_points_in[keep_binaries, 4] -= xp.asarray(phase_change)

                    # # check if there are sources near band edges that are overlapping
                    # assert np.all(keep_coords[:, 1] == np.sort(keep_coords[:, 1]))
                    # f_found = keep_coords[:, 1] / 1e3
                    # N = get_N(np.full_like(f_found, 1e-30), f_found, Tobs=waveform_kwargs["T"], oversample=waveform_kwargs["oversample"])
                    # inds_check = np.where((np.diff(f_found) / df).astype(int) < N[:-1])[0]

                    # params_add = keep_coords[inds_check]
                    # params_remove = keep_coords[inds_check + 1]
                    # N_check = N[inds_check]

                    # params_add_in = transform_fn.both_transforms(params_add)
                    # params_remove_in = transform_fn.both_transforms(params_remove)

                    # waveform_kwargs_tmp = waveform_kwargs.copy()
                    # if "N" in waveform_kwargs_tmp:
                    #     waveform_kwargs_tmp.pop("N")
                    # waveform_kwargs_tmp["use_c_implementation"] = False

                    # gb.swap_likelihood_difference(params_add_in, params_remove_in, data_in, psd_in, N=256, **waveform_kwargs_tmp)

                    # likelihood_difference = -1/2 * (gb.add_add + gb.remove_remove - 2 * gb.add_remove).real.get()
                    # overlap = (gb.add_remove.real / np.sqrt(gb.add_add.real * gb.remove_remove.real)).get()

                    # fix = np.where((likelihood_difference > -100.0) | (overlap > 0.4))

                    # if np.any(fix):
                    #     params_comp_add = params_add[fix]
                    #     params_comp_remove = params_remove[fix]

                    #     # not actually in the data yet, just using swap for quick likelihood comp
                    #     snr_add = (gb.d_h_add.real[fix] / gb.add_add.real[fix] ** (1/2)).get()
                    #     snr_remove = (gb.d_h_remove.real[fix] / gb.remove_remove.real[fix] ** (1/2)).get()

                    #     inds_add = inds_check[fix]
                    #     inds_remove = inds_add + 1

                    #     inds_delete = (inds_add) * (snr_add < snr_remove) + (inds_remove) * (snr_remove < snr_add)
                    #     keep_coords = np.delete(keep_coords, inds_delete, axis=0)
                        

                run_number += 1

            else:
                break

    return fit_gmm(samples_store, comm, comm_info)

    # import pickle
    # with open(f"new_4_gmm_info.pickle", "wb") as fp:
    #     pickle.dump(output, fp, pickle.HIGHEST_PROTOCOL)
    
    
    
    # nwalkers_pe = nwalkers
    # ntemps_pe = 1
    
    # factor = 1e-5
    # cov = np.ones(ndim) * 1e-3
    # cov[1] = 1e-8

    # still_going_start_like = np.ones(keep_coords.shape[0], dtype=bool)
    # starting_points = np.zeros((keep_coords.shape[0], nwalkers_pe * ntemps_pe, ndim))
    # iter_check = 0
    # max_iter = 10000
    # while np.any(still_going_start_like):
    #     num_still_going_start_like = still_going_start_like.sum().item()
        
    #     start_like = np.zeros((num_still_going_start_like, nwalkers_pe * ntemps_pe))
    
    #     logp = np.full_like(start_like, -np.inf)
    #     tmp = np.zeros((num_still_going_start_like, ntemps_pe * nwalkers_pe, ndim))
    #     fix = np.ones((num_still_going_start_like, ntemps_pe * nwalkers_pe), dtype=bool)
    #     while np.any(fix):
    #         tmp[fix] = (keep_coords[still_going_start_like, None, :] * (1. + factor * cov * np.random.randn(num_still_going_start_like, nwalkers_pe * ntemps_pe, ndim)))[fix]

    #         tmp[:, :, 3] = tmp[:, :, 3] % (2 * np.pi)
    #         tmp[:, :, 5] = tmp[:, :, 5] % (np.pi)
    #         tmp[:, :, 6] = tmp[:, :, 6] % (2 * np.pi)
    #         logp = priors["gb"].logpdf(tmp.reshape(-1, ndim)).reshape(tmp.shape[:-1])

    #         fix = np.isinf(logp)
    #         if np.all(fix):
    #             breakpoint()

    #     tmp_in = transform_fn.both_transforms(tmp.reshape(-1, ndim))
    #     start_like = gb.get_ll(tmp_in, data_in, psd_in, phase_marginalize=False, **waveform_kwargs).reshape(tmp.shape[:-1])

    #     starting_points[still_going_start_like] = tmp
        
    #     update = np.arange(still_going_start_like.shape[0])[still_going_start_like][np.std(start_like, axis=-1) > 5.0]
    #     still_going_start_like[update] = False 

    #     iter_check += 1
    #     factor *= 1.5

    #     # print(np.std(start_like))

    #     if iter_check > max_iter:
    #         raise ValueError("Unable to find starting parameters.")

    # starting_points = starting_points.reshape((keep_coords.shape[0], ntemps_pe, nwalkers_pe, ndim)).transpose(1, 2, 0, 3)

    # if binaries_found is None:
    #     binaries_found = keep_coords
    # else:
    #     binaries_found = np.concatenate([binaries_found, keep_coords], axis=0)

    # binaries_found = binaries_found[np.argsort(binaries_found[:, 1])]

    # num_binaries_found_this_iteration = keep_binaries.sum().item()

    # np.save("starting_points_last_batch", starting_points)
    # return starting_points


def refit_gmm(current_info, gpu, comm, comm_info, gb_reader, data, psd, number_samples_keep):
    print("GATHER")
    samples_gathered = gather_gb_samples(current_info, gb_reader, psd, gpu, samples_keep=number_samples_keep, thin_by=20)
    
    return fit_gmm(samples_gathered, comm, comm_info)

def run_gb_bulk_search(gpu, comm, comm_info, head_rank, num_search, split_remainder):
    gpus = [gpu]
    xp.cuda.runtime.setDevice(gpus[0])

    ntemps = 10
    nwalkers = 100
    ndim = 8

    stop = True

    rank = comm.Get_rank()
    tag = int(str(2929) + str(rank))
    print("CHECK yep", rank, tag, head_rank)
    gf_information = comm.recv(source=head_rank, tag=tag)
    print("CHECK n", rank, tag, head_rank)
    gb_info = gf_information.gb_info
    band_edges = gb_info["band_edges"]

    band_inds_running = np.zeros_like(band_edges[:-1], dtype=bool)

    # split_remainder of 0 is refit
    # split_remainder >= 1 is search
    if split_remainder > 0:
        search_split_remainder = split_remainder - 1
        num_search_for_search = num_search - 1
        assert search_split_remainder >= 0
        assert num_search_for_search > 0 and search_split_remainder < num_search_for_search

        band_inds_running[np.arange(len(band_inds_running)) % num_search_for_search == search_split_remainder] = True

    print(rank, f"FIGURE ", num_search, split_remainder, band_inds_running.sum(), band_inds_running.shape[0])
    
    priors_here = deepcopy(gb_info["priors"].priors_in)
    priors_here.pop((0, 1))
    priors_here[0] = SNRPrior(gb_info["pe_info"]["rho_star"], use_cupy=True) 
    priors_here[1] = uniform_dist(0.0, 1.0, use_cupy=True) 
    priors_here[2] = uniform_dist(0.0, 1.0, use_cupy=True) 

    priors_good = ProbDistContainer(priors_here, use_cupy=True)
    
    m_chirp_lims = gb_info["search_info"]["m_chirp_lims"]
    
    # now update for negative fdots
    fdot_max_vals = xp.asarray(get_fdot(band_edges[1:], Mc=np.full(band_edges.shape[0] - 1, m_chirp_lims[1])))
    fdot_mins = -fdot_max_vals
    fdot_maxs = fdot_max_vals
    f0_mins = xp.asarray(band_edges[:-1] * 1e3)
    f0_maxs = xp.asarray(band_edges[1:] * 1e3)

    # do not run the last band
    band_inds_running[-1] = False

    # stopping function
    stopping_function_here = deepcopy(gb_info["search_info"]["stopping_function"])

    run_counter = 0
    print("start run")
    run = True
    while run:
        try:
            comm.send({"send": True}, dest=head_rank, tag=20)
            print("waiting for data")
            incoming_data = comm.recv(source=head_rank, tag=27)
            print("received data")

            if "cc_ll" not in incoming_data.psd_info or "cc_ll" not in incoming_data.mbh_info or "cc_ll" not in incoming_data.gb_info:
                time.sleep(20.0)
                print("Do not have maximum likelihood for all pieces for search. Waiting and then will try again.")
                continue

            generated_info = incoming_data.get_data_psd(only_max_ll=True, return_prior_val=True, fix_val_in_gen=["gb"]) 
            # generated_info_0 = generate_class(incoming_data, only_max_ll=True, include_mbhs=False, include_gbs=False, include_ll=True, include_source_only_ll=True)
            # generated_info_1 = generate_class(incoming_data, only_max_ll=True, include_mbhs=True, include_ll=True, include_source_only_ll=True)
            # generated_info_2 = generate_class(incoming_data, only_max_ll=True, include_mbhs=True, include_gbs=True, include_ll=True, include_source_only_ll=True)
            # plt.loglog(2 * np.sqrt(df) * np.abs(generated_info_0["data"][0]))
            # plt.loglog(2 * np.sqrt(df) * np.abs(generated_info_1["data"][0]))
            # plt.savefig("check1.png")
            # breakpoint()

            data_cpu = generated_info["data"]
            psd_cpu = generated_info["psd"]
            lisasens_cpu = generated_info["lisasens"]

            data = [xp.asarray(tmp) for tmp in data_cpu]
            psd = [xp.asarray(tmp) for tmp in psd_cpu]
            lisasens = [xp.asarray(tmp) for tmp in lisasens_cpu]

            # max ll combination of psd and mbhs and gbs
            
            # run refit
            if split_remainder == 0:
                if os.path.exists(incoming_data.gb_info["reader"].filename) and incoming_data.gb_info["reader"].iteration > incoming_data.gb_info["pe_info"]["start_resample_iter"] and (run_counter % incoming_data.gb_info["pe_info"]["iter_count_per_resample"]) == 0:
                    gmm_samples_refit = refit_gmm(incoming_data, gpu, comm, comm_info, incoming_data.gb_info["reader"], data, psd, 100)

                else:
                    gmm_samples_refit = None

                send_out_dict = {"sample_refit": gmm_samples_refit}

            else:
                gmm_mcmc_search_info = run_iterative_subtraction_mcmc(incoming_data, gpu, ndim, nwalkers, ntemps, band_inds_running, priors_good, f0_maxs, f0_mins, fdot_maxs, fdot_mins, data, psd, lisasens, comm, comm_info)
                send_out_dict = {"search": gmm_mcmc_search_info}

            print("AFTER SEARCH", comm.Get_rank())
            comm.send({"receive": True}, dest=head_rank, tag=20)
            comm.send(send_out_dict, dest=head_rank, tag=29)
            print("SENT AFTER SEARCH", comm.Get_rank())
            if not hasattr(stopping_function_here, "comm") and hasattr(stopping_function_here, "add_comm"):
                stopping_function_here.add_comm(comm)

            print("load comm if needed AFTER SEARCH", comm.Get_rank())
            
            stop = stopping_function_here(incoming_data)
            print("after stop function AFTER SEARCH", comm.Get_rank())
            
            run_counter += 1
            if stop:
                break
        
        except BlockingIOError as e:
            print("bulk", e)
            time.sleep(20.0)  
            # refit GMM


            # evens_odds = 0
            # if not "starting_points_last_batch.npy" in os.listdir():
                
            # else:
            #     starting_points = np.load("starting_points_last_batch.npy")
            #     starting_points = np.full((1, 100, 0, 0), np.array([]))

            # num_binaries_found_this_iteration = starting_points.shape[2]
            # num_binaries_total += num_binaries_found_this_iteration

            # # starting_points = None
            # # num_binaries_found_this_iteration = 0
            # print(iter_i, f"Number of bands running: {band_inds_running.sum().item()}, found {num_binaries_found_this_iteration} binaries. Total binaries: {num_binaries_total}")

            # data_in, psd_in = run_gb_mixing(iter_i, gpus, fp_gb_mixing, num_binaries_found_this_iteration, starting_points)

    # communicate end of run to head process
    for rank in comm_info["process_ranks_for_fit"]:
        rec_tag = int(str(rank) + "67676")
        comm.send("end", dest=rank, tag=rec_tag)

    comm.send({"finish_run": True}, dest=head_rank, tag=20)

    try:
        for i in range(len(data)):
            data[i] = None
            psd[i] = None
            lisasens[i] = None

    # have not stored this info yet
    except NameError:
        pass

    xp.get_default_memory_pool().free_all_blocks()
    return


if __name__ == "__main__":
    import argparse

    """parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=int,
                        help='which gpu', required=True)


    args = parser.parse_args()"""

    output = run_gb_pe(3)
