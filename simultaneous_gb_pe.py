
from copy import deepcopy
import cupy as xp
import numpy as np
import time
import pickle
import shutil
from mpi4py import MPI

from gbgpu.gbgpu import GBGPU


from lisatools.sampling.moves.gbspecialstretch import GBSpecialStretchMove

mempool = xp.get_default_memory_pool()

from lisatools.utils.multigpudataholder import MultiGPUDataHolder
from eryn.moves import CombineMove
from lisatools.sampling.prior import FullGaussianMixtureModel
from eryn.moves.tempering import make_ladder
from eryn.state import BranchSupplimental
from eryn.prior import ProbDistContainer, uniform_dist
from eryn.ensemble import EnsembleSampler

import subprocess

import warnings

warnings.filterwarnings("ignore")

stop_here = True

from eryn.moves import Move
from lisatools.globalfit.state import State
from lisatools.globalfit.hdfbackend import HDFBackend


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
        self, mgh, gb, comm, head_rank, gpu_priors, verbose=False
    ):
        self.mgh = mgh
        self.comm = comm
        self.head_rank = head_rank
        self.verbose = verbose
        self.gb = gb
        self.gpu_priors = gpu_priors

    def __call__(self, iter, last_sample, sampler):
        
        if self.verbose:
            print("Sending gb update to head process.")

        update_dict = {
            "cc_params": last_sample.branches["gb_fixed"].coords[0, :, :].copy(),
            "cc_inds": last_sample.branches["gb_fixed"].inds[0, :, :].copy(),
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
            gen_dist_search = self.gpu_priors["gb_fixed"]

        new_refit_gmm = new_info.gb_info["refit_gmm_info"]
        if new_refit_gmm is not None:
            gen_dist_refit = make_gmm(self.gb, new_refit_gmm)
        elif new_search_gmm is not None:
            gen_dist_refit = make_gmm(self.gb, new_search_gmm)
        else:
            gen_dist_refit = self.gpu_priors["gb_fixed"]

        # sub out the proposal distributions
        for move in sampler.rj_moves:
            if move.name == "rj_refit_gmm":
                move.rj_proposal_distribution = {"gb_fixed": gen_dist_refit}
            
            elif move.name == "rj_search_gmm":
                move.rj_proposal_distribution = {"gb_fixed": gen_dist_search}

        if self.verbose:
            print("Finished GMM reset.")
            print("Generating new base data.")

        nwalkers_pe = last_sample.log_like.shape[1]
        generated_info = new_info.get_data_psd(n_gen_in=nwalkers_pe)
        data = generated_info["data"]
        psd = generated_info["psd"]
        
        self.mgh.sub_in_data_and_psd(data, psd)

        if self.verbose:
            print("Finished subbing in new data.")

        xp.get_default_memory_pool().free_all_blocks()
        new_ll = self.mgh.get_ll(include_psd_info=True)
        last_sample.log_like[0, :] = new_ll[:]
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


def run_gb_pe(gpu, comm, head_rank):

    gb = GBGPU(use_gpu=True)
    gpus_pe = [gpu]
    gpus = gpus_pe
    # from lisatools.sampling.stopping import SearchConvergeStopping2

    gf_information = comm.recv(source=head_rank, tag=255)

    gb_info = gf_information.gb_info
    band_edges = gb_info["band_edges"]
    
    num_sub_bands = len(band_edges)

    xp.cuda.runtime.setDevice(gpus[0])

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
            last_sample.band_info["band_temps"][:] = band_info_check["band_temps"][:]

    import time
    st = time.perf_counter()
    generated_info = gf_information.get_data_psd(include_gbs=False, include_ll=True, include_source_only_ll=True, n_gen_in=nwalkers_pe)
    et = time.perf_counter()

    print("Read in", et - st)

    data = generated_info["data"]
    psd = generated_info["psd"]

    df = gf_information.general_info["df"]

    A_going_in = np.repeat(data[0], 2, axis=0).reshape(nwalkers_pe, 2, gf_information.general_info["data_length"]).transpose(1, 0, 2)
    E_going_in = np.repeat(data[1], 2, axis=0).reshape(nwalkers_pe, 2, gf_information.general_info["data_length"]).transpose(1, 0, 2)

    A_psd_in = np.repeat(psd[0], 1, axis=0).reshape(nwalkers_pe, 1, gf_information.general_info["data_length"]).transpose(1, 0, 2)
    E_psd_in = np.repeat(psd[1], 1, axis=0).reshape(nwalkers_pe, 1, gf_information.general_info["data_length"]).transpose(1, 0, 2)
    
    mgh = MultiGPUDataHolder(
        gpus,
        A_going_in,
        E_going_in,
        A_going_in, # store as base
        E_going_in, # store as base
        A_psd_in,
        E_psd_in,
        df,
        base_injections=[gf_information.general_info["A_inj"], gf_information.general_info["E_inj"]],
        base_psd=None,  # [psd.copy(), psd.copy()]
    )
    ll_c = mgh.get_ll()

    mempool.free_all_blocks()

    # setup data streams to add to and subtract from
    supps_shape_in = (ntemps_pe, nwalkers_pe)

    gb.gpus = mgh.gpus

    priors = {"gb_fixed": gb_info["priors"]}
    nleaves_max = last_sample.branches["gb_fixed"].shape[2]
    transform_fn = gb_info["transform"]

    band_mean_f = (band_edges[1:] + band_edges[:-1]) / 2
    
    waveform_kwargs = gb_info["pe_info"]["pe_waveform_kwargs"].copy()
    if "N" in waveform_kwargs:
        waveform_kwargs.pop("N")

    from gbgpu.utils.utility import get_N
    band_N_vals = xp.asarray(get_N(np.full_like(band_mean_f, 1e-30), band_mean_f, waveform_kwargs["T"], waveform_kwargs["oversample"]))


    if last_sample.branches["gb_fixed"].inds[0].sum() > 0:
        coords_out_gb_fixed = last_sample.branches["gb_fixed"].coords[0,
            last_sample.branches["gb_fixed"].inds[0]
        ]
        
        check = priors["gb_fixed"].logpdf(coords_out_gb_fixed)

        if np.any(np.isinf(check)):
            raise ValueError("Starting priors are inf.")

        coords_out_gb_fixed[:, 3] = coords_out_gb_fixed[:, 3] % (2 * np.pi)
        coords_out_gb_fixed[:, 5] = coords_out_gb_fixed[:, 5] % (1 * np.pi)
        coords_out_gb_fixed[:, 6] = coords_out_gb_fixed[:, 6] % (2 * np.pi)
        
        coords_in_in = transform_fn.both_transforms(coords_out_gb_fixed)

        band_inds = np.searchsorted(band_edges, coords_in_in[:, 1], side="right") - 1

        walker_vals = np.tile(
            np.arange(nwalkers_pe), (nleaves_max, 1)
        ).transpose((1, 0))[last_sample.branches["gb_fixed"].inds[0]]

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
            batch_size=1000,
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

    state_mix = State(
        last_sample.branches_coords,
        inds=last_sample.branches_inds,
        log_like=ll,
        # supplimental=supps,
        betas=last_sample.betas,
        band_info=last_sample.band_info
    )

    band_inds_in = np.zeros((ntemps_pe, nwalkers_pe, nleaves_max), dtype=int)
    N_vals_in = np.zeros((ntemps_pe, nwalkers_pe, nleaves_max), dtype=int)

    if state_mix.branches["gb_fixed"].inds.sum() > 0:
        f_in = state_mix.branches["gb_fixed"].coords[state_mix.branches["gb_fixed"].inds][:, 1] / 1e3
        band_inds_in[state_mix.branches["gb_fixed"].inds] = np.searchsorted(band_edges, f_in, side="right") - 1
        N_vals_in[state_mix.branches["gb_fixed"].inds] = band_N_vals.get()[band_inds_in[state_mix.branches["gb_fixed"].inds]]

    branch_supp_base_shape = (ntemps_pe, nwalkers_pe, nleaves_max)
    state_mix.branches["gb_fixed"].branch_supplimental = BranchSupplimental(
        {"N_vals": N_vals_in, "band_inds": band_inds_in}, base_shape=branch_supp_base_shape, copy=True
    )

    gpu_priors_in = deepcopy(priors["gb_fixed"].priors_in)
    for key, item in gpu_priors_in.items():
        item.use_cupy = True

    gpu_priors = {"gb_fixed": ProbDistContainer(gpu_priors_in, use_cupy=True)}

    gb_kwargs = dict(
        waveform_kwargs=waveform_kwargs,
        parameter_transforms=transform_fn,
        provide_betas=True,
        skip_supp_names_update=["group_move_points"],
        random_seed=gf_information.general_info["random_seed"],
        nfriends=nwalkers_pe,
        n_iter_update=1,
        live_dangerously=True,
        # rj_proposal_distribution=gpu_priors,
        a=1.75,
        use_gpu=True,
        num_repeat_proposals=30
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

    gb_fixed_move = GBSpecialStretchMove(
        *gb_args,
        **gb_kwargs,
    )

    # add the other
    gb_fixed_move.gb.gpus = gpus
    moves_in_model = [gb_fixed_move]

    rj_moves_in = []

    rj_moves_in_frac = []

    gb_kwargs_rj = dict(
        waveform_kwargs=waveform_kwargs,
        parameter_transforms=transform_fn,
        search=False,
        provide_betas=True,
        skip_supp_names_update=["group_move_points"],
        random_seed=gf_information.general_info["random_seed"],
        nfriends=nwalkers_pe,
        rj_proposal_distribution=gpu_priors,
        a=1.7,
        use_gpu=True,
        name="rj_prior"
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
    rj_moves_in_frac.append(0.2)

    if state_mix.branches["gb_fixed"].inds.sum() == 0:
        # wait until we have a search distribution
        waiting = True
        while waiting:
            time.sleep(20.0)
            comm.send({"send": True}, dest=head_rank, tag=50)
            new_info = comm.recv(source=head_rank, tag=51)
            # print("CHECKING:", new_info.gb_info["search_gmm_info"])
            if new_info.gb_info["search_gmm_info"] is not None:
                waiting = False
                gb_info = new_info.gb_info

    current_gmm_search_info = gb_info["search_gmm_info"]
    if current_gmm_search_info is not None:
        gen_dist_search = make_gmm(gb, current_gmm_search_info)
    else:
        gen_dist_search = gpu_priors["gb_fixed"]

    gb_kwargs_rj_2 = gb_kwargs_rj.copy()
    gb_kwargs_rj_2["rj_proposal_distribution"] = {"gb_fixed": gen_dist_search}
    gb_kwargs_rj_2["name"] = "rj_search_gmm"

    rj_move_search = GBSpecialStretchMove(
        *gb_args_rj,
        **gb_kwargs_rj_2,
    )

    rj_moves_in.append(rj_move_search)
    rj_moves_in_frac.append(0.8)

    current_gmm_refit_info = gb_info["search_gmm_info"]

    if current_gmm_refit_info is not None:
        gen_dist_refit = make_gmm(gb, current_gmm_refit_info)
    elif current_gmm_search_info is not None:
        gen_dist_refit = make_gmm(gb, current_gmm_search_info)
    else:
        gen_dist_refit = gpu_priors["gb_fixed"]
    
    gb_kwargs_rj_3 = gb_kwargs_rj.copy()

    gb_kwargs_rj_3["rj_proposal_distribution"] = {"gb_fixed": gen_dist_refit}
    gb_kwargs_rj_3["name"] = "rj_refit_gmm"

    rj_move_refit = GBSpecialStretchMove(
        *gb_args_rj,
        **gb_kwargs_rj_3,
    )

    rj_moves_in.append(rj_move_refit)
    # TODO: check for this in updates
    if current_gmm_refit_info is not None:
        rj_moves_in_frac.append(0.2)
    else:
        rj_moves_in_frac.append(0.0)

    total_frac = sum(rj_moves_in_frac)

    rj_moves = [(rj_move_i, rj_move_frac_i / total_frac) for rj_move_i, rj_move_frac_i in zip(rj_moves_in, rj_moves_in_frac)]

    for rj_move in rj_moves:
        rj_move[0].gb.gpus = gpus

    like_mix = BasicResidualMGHLikelihood(mgh)
    branch_names = ["gb_fixed"]

    update = UpdateNewResiduals(
        mgh, gb, comm, head_rank, gpu_priors, verbose=True
    )

    ndims = {"gb_fixed": gb_info["pe_info"]["ndim"]}
    nleaves_max = {"gb_fixed": gb_info["pe_info"]["nleaves_max"]}
        
    moves = moves_in_model + rj_moves
    backend = HDFBackend(
        gf_information.general_info["file_information"]["fp_gb_pe"],
        compression="gzip",
        compression_opts=9,
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

    sampler_mix = EnsembleSampler(
        nwalkers_pe,
        ndims,  # assumes ndim_max
        like_mix,
        priors,
        tempering_kwargs={"betas": betas, "adaptation_time": 2, "permute": True},
        nbranches=len(branch_names),
        nleaves_max=nleaves_max,
        moves=moves_in_model,
        rj_moves=rj_moves,
        kwargs=None,  # {"start_freq_ind": start_freq_ind, **waveform_kwargs},
        backend=backend,
        vectorize=True,
        periodic={"gb_fixed": gb_info["periodic"]},  # TODO: add periodic to proposals
        branch_names=gb_info["pe_info"]["branch_names"],
        update_fn=update,  # stop_converge_mix,
        update_iterations=1,
        provide_groups=True,
        provide_supplimental=True,
        num_repeats_in_model=1,
        track_moves=False
    )

    nsteps_mix = 1000

    # with open(current_save_state_file, "wb") as fp:
    #     pickle.dump(state_mix, fp, pickle.HIGHEST_PROTOCOL)

    print("Starting mix ll best:", state_mix.log_like.max(axis=-1))
    mempool.free_all_blocks()
    
    # exit()
    out = sampler_mix.run_mcmc(
        state_mix, nsteps_mix, progress=False, thin_by=20, store=True
    )
    print("ending mix ll best:", out.log_like.max(axis=-1))
    

if __name__ == "__main__":
    import argparse

    """parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=int,
                        help='which gpu', required=True)


    args = parser.parse_args()"""

    output = run_gb_pe(3)
