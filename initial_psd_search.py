from multiprocessing.sharedctypes import Value
import cupy as xp
import time
import pickle
import shutil

# from lisatools.sampling.moves.gbspecialgroupstretch import GBSpecialGroupStretchMove

mempool = xp.get_default_memory_pool()

from full_band_global_fit_settings import *
from single_mcmc_run import run_single_band_search
from lisatools.utils.multigpudataholder import MultiGPUDataHolder
from eryn.moves import CombineMove
from lisatools.sampling.moves.specialforegroundmove import GBForegroundSpecialMove

import subprocess

import warnings
warnings.filterwarnings("ignore")

stop_here = True


from eryn.moves import Move
class PlaceHolder(Move):
    def __init__(self, *args, **kwargs):
        super(PlaceHolder, self).__init__(*args, **kwargs)

    def propose(self, model, state):
        accepted = np.zeros(state.log_like.shape)
        self.temperature_control.swaps_accepted = np.zeros(self.temperature_control.ntemps - 1)
        return state, accepted

class BasicResidualMGHLikelihood:
    def __init__(self, mgh):
        self.mgh = mgh
    def __call__(self, *args, supps=None, **kwargs):
        ll_temp = self.mgh.get_ll()
        overall_inds = supps["overal_inds"]
        
        return ll_temp[overall_inds]

data = [
    np.asarray(A_inj),
    np.asarray(E_inj),
]

def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a,idx,axis=axis)

# TODO: fix initial setup for mix where it backs up the likelihood

class PointGeneratorSNR:
    def __init__(self, generate_snr_ladder):
        self.generate_snr_ladder = generate_snr_ladder

    def __call__(self, size=(1,)):
        new_points = self.generate_snr_ladder.rvs(size=size)
        logpdf = self.generate_snr_ladder.logpdf(new_points)
        return (new_points, logpdf)

point_generator = PointGeneratorSNR(generate_snr_ladder)


def log_like(x, freqs, data, supps=None, **sens_kwargs):
    # if supps is None:
    #    raise ValueError("Must provide supps to identify the data streams.")

    # wi = supps["walker_inds"]
    psd_pars = x[0]
    galfor_pars = x[1]
    A_data = data[0]
    E_data = data[1]
    data_index_all = xp.zeros(psd_pars.shape[0], dtype=xp.int32)
    ll = xp.zeros(psd_pars.shape[0]) 
    A_Soms_d_in_all = xp.asarray(psd_pars[:, 0])
    A_Sa_a_in_all = xp.asarray(psd_pars[:, 1])
    E_Soms_d_in_all = xp.asarray(psd_pars[:, 2])
    E_Sa_a_in_all = xp.asarray(psd_pars[:, 3])
    Amp_all = xp.asarray(galfor_pars[:, 0])
    alpha_all = xp.asarray(galfor_pars[:, 1])
    sl1_all = xp.asarray(galfor_pars[:, 2])
    kn_all = xp.asarray(galfor_pars[:, 3])
    sl2_all = xp.asarray(galfor_pars[:, 4])
    num_data = 1
    num_psds = psd_pars.shape[0]

    gb.psd_likelihood(ll, freqs, A_data, E_data, data_index_all,  A_Soms_d_in_all,  A_Sa_a_in_all,  E_Soms_d_in_all,  E_Sa_a_in_all, 
                     Amp_all,  alpha_all,  sl1_all,  kn_all, sl2_all, df, data_length, num_data, num_psds)
    
    # # galfor_pars = None
    # ll2 = xp.zeros_like(ll)
    # for i, (psd_pars_i, galfor_pars_i) in enumerate(zip(psd_pars, galfor_pars)):
    #     psd = [
    #         get_sensitivity(freqs, model=psd_pars_i[:2], foreground_params=galfor_pars_i, **sens_kwargs),
    #         get_sensitivity(freqs, model=psd_pars_i[2:], foreground_params=galfor_pars_i, **sens_kwargs)
    #     ]
    #     psd[0][0] = psd[0][1]
    #     psd[1][0] = psd[1][1]

    #     # inner_product = 4 * df * (xp.sum(data[0][wi].conj() * data[0][wi] / psd[0]) + xp.sum(data[1][wi].conj() * data[1][wi] / psd[1])).real
    #     inner_product = 4 * df * (xp.sum(data[0].conj() * data[0] / psd[0]) + xp.sum(data[1].conj() * data[1] / psd[1])).real
    #     ll2[i] = -1/2 * inner_product - xp.sum(xp.log(xp.asarray(psd)))
    # assert np.allclose(ll.get(), ll2.get())
    return ll.get()

from eryn.utils.stopping import Stopping
class SearchConvergeStopping(Stopping):
    """Stopping function based on a convergence to a maximunm Likelihood.

    Stopping checks are only performed every ``thin_by`` iterations.
    Therefore, the iterations of stopping checks are really every
    ``sampler iterations * thin_by``.  

    All arguments are stored as attributes.

    Args:
        n_iters (int, optional): Number of iterative stopping checks that need to pass
            in order to stop the sampler. (default: ``30``)
        diff (float, optional): Change in the Likelihood needed to fail the stopping check. In other words,
            if the new maximum Likelihood is more than ``diff`` greater than the old, all iterative checks 
            reset. (default: 0.1). 
        start_iteration (int, optional): Iteration of sampler to start checking to stop. (default: 0)
        verbose (bool, optional): If ``True``, print information. (default: ``False``)

    Attributes:
        iters_consecutive (int): Number of consecutive passes of the stopping check.
        past_like_best (float): Previous best Likelihood. The initial value is ``-np.inf``.
    
    """

    def __init__(self, n_iters=30, diff=0.1, start_iteration=0, verbose=False):

        # store all the relevant information
        self.n_iters = n_iters

        self.diff = diff
        self.verbose = verbose
        self.start_iteration = start_iteration

        # initialize important info
        self.iters_consecutive = 0
        self.past_like_best = -np.inf

    def __call__(self, iter, sample, sampler):

        # if we have not reached the start iteration return
        if iter < self.start_iteration:
            return False

        # get best Likelihood so far
        like_best = sampler.get_log_like(discard=self.start_iteration).max()

        # compare to last
        # if it is less than diff change it passes
        if np.abs(like_best - self.past_like_best) < self.diff:
            self.iters_consecutive += 1

        else:
            # if it fails reset iters consecutive
            self.iters_consecutive = 0

            # store new best
            self.past_like_best = like_best

        # print information
        if self.verbose:
            print(
                f"\nITERS CONSECUTIVE: {self.iters_consecutive}",
                f"Previous best LL: {self.past_like_best}",
                f"Current best LL: {like_best}\n",
            )

        if self.iters_consecutive >= self.n_iters:
            # if we have passes the number of iters necessary, return True and reset
            self.iters_consecutive = 0
            return True

        else:
            return False

from eryn.utils.updates import Update


class UpdateNewResiduals(Update):
    def __init__(self, fd, fp_gb, fp_mbh, fp_psd, nwalkers):
        self.fp_gb = fp_gb
        self.fp_psd = fp_psd
        self.nwalkers = nwalkers
        self.fd = xp.asarray(fd)
        self.fp_mbh = fp_mbh

    def __call__(self, iter, last_sample, sampler):

        A_going_in = np.asarray(A_inj).copy()
        E_going_in = np.asarray(E_inj).copy()
        imported = False
        while not imported:
            try:
                mbh_inj = np.load("best_logl_mbhs_from_psd_run.npy")
                imported = True
            except ValueError:
                time.sleep(1)
        A_mbh_going_in = mbh_inj[0]
        E_mbh_going_in = mbh_inj[1]

        imported = False
        while not imported:
            try:
                gbs_inj = np.load("best_logl_gbs_from_psd_run.npy")
                imported = True
            except ValueError:
                time.sleep(1)
        A_gb_going_in = gbs_inj[0]
        E_gb_going_in = gbs_inj[1]

        if "check0.png" in os.listdir():
            shutil.copy("check0.png", "check1.png")
        plt.loglog(fd, 2 * df * np.abs(A_going_in) ** 2)
        

        A_going_in[:] -= A_mbh_going_in
        E_going_in[:] -= E_mbh_going_in

        A_going_in[:] -= A_gb_going_in
        E_going_in[:] -= E_gb_going_in

        plt.loglog(fd, 2 * df * np.abs(A_going_in) ** 2)

        before = last_sample.log_like.max()
        sampler.log_like_fn.args[1][0][:] = xp.asarray(A_going_in)
        sampler.log_like_fn.args[1][1][:] = xp.asarray(E_going_in)
        last_sample.log_prior = sampler.compute_log_prior(last_sample.branches_coords)
        last_sample.log_like = sampler.compute_log_like(last_sample.branches_coords, logp=last_sample.log_prior)[0]
        after = last_sample.log_like.max()

        print(before, after, after - before)

        ind_best = np.where(last_sample.log_like[0] == last_sample.log_like[0].max())[0][0]
        psd_pars = last_sample.branches_coords["psd"][0, ind_best, 0]
        galfor_pars = last_sample.branches_coords["galfor"][0, ind_best, 0]
        psds_out_A = get_sensitivity(self.fd, sens_fn="noisepsd_AE", model=psd_pars[:2], foreground_params=galfor_pars, xp=xp)
        psds_out_E = get_sensitivity(self.fd, sens_fn="noisepsd_AE", model=psd_pars[2:], foreground_params=galfor_pars, xp=xp)
        plt.loglog(fd, psds_out_A.get())
        plt.savefig("check0.png")
        plt.close()
        np.save("best_logl_psd_from_psd_run", np.asarray([psds_out_A.get(), psds_out_E.get()]))
        del psds_out_A, psds_out_E

        xp.get_default_memory_pool().free_all_blocks()

def run_psd_search(A_going_in, E_going_in, gpu):

    """while fp_gb + ".npy" not in os.listdir():
        print(f"{fp_gb + '.npy'} not in current directory so far...")
        time.sleep(20)"""

    branch_names = ["psd", "galfor"]

    gpus_pe = [gpu]
    gpus = gpus_pe
    # from lisatools.sampling.stopping import SearchConvergeStopping2

    waveform_kwargs["start_freq_ind"] = start_freq_ind

    # for testing
    # search_f_bin_lims = np.arange(f0_lims_in[0], f0_lims_in[1], 2 * 128 * df)

    num_sub_bands = len(search_f_bin_lims)

    xp.cuda.runtime.setDevice(gpus[0])

    nwalkers_pe = 100
    ntemps_pe = 10

    num_binaries_needed_to_mix = 1
    num_binaries_current = 0
    
    if fp_psd_search_initial in os.listdir():
        reader = HDFBackend(fp_psd_search_initial)
        last_sample = reader.get_last_sample()

        coords = {key: last_sample.branches_coords[key] for key in branch_names}
        inds = {key: last_sample.branches_inds[key] for key in branch_names}

        last_sample = State(coords, inds=inds, log_like=last_sample.log_like, log_prior=last_sample.log_prior)
        
    elif False: # current_save_state_file_psd in os.listdir():
        """print("LOADING PSD save state")
        with open(current_save_state_file_psd, "rb") as fp_out:
            last_sample = pickle.load(fp_out)

        coords = {key: last_sample.branches_coords[key] for key in branch_names}
        inds = {key: last_sample.branches_inds[key] for key in branch_names}

        last_sample = State(coords, inds=inds, log_like=last_sample.log_like, log_prior=last_sample.log_prior)
        """
    else:
        coords = {}
        inds = {}

        coords["psd"] = priors["psd"].rvs(size=(ntemps_pe, nwalkers_pe))
        inds["psd"] = np.ones((ntemps_pe, nwalkers_pe, 1), dtype=bool)
        coords["galfor"] = priors["galfor"].rvs(size=(ntemps_pe, nwalkers_pe))
        inds["galfor"] = np.ones((ntemps_pe, nwalkers_pe, 1), dtype=bool)

        last_sample = State(coords, inds=inds)
    
    data = [xp.asarray(A_going_in), xp.asarray(E_going_in)]

    psd_params = last_sample.branches["psd"].coords.reshape(-1, last_sample.branches["psd"].shape[-1])

    foreground_params = last_sample.branches["galfor"].coords.reshape(-1, last_sample.branches["galfor"].shape[-1])

    """walker_vals = np.tile(np.arange(nwalkers_pe), (ntemps_pe, 1))

    supps_base_shape = (ntemps_pe, nwalkers_pe)
    supps = BranchSupplimental({"walker_inds": walker_vals}, base_shape=supps_base_shape, copy=True)
    """
    state_mix = State(last_sample.branches_coords, inds=last_sample.branches_inds) # , supplimental=supps)

    from eryn.moves.tempering import make_ladder
    betas = make_ladder(4, ntemps=ntemps_pe)

    sens_kwargs = dict(sens_fn="noisepsd_AE")

    if fp_psd_search_initial in os.listdir():
        shutil.copy(fp_psd_search_initial, "back_current_" + fp_psd_search_initial)
        os.remove(fp_psd_search_initial)

    update = UpdateNewResiduals(fd, fp_gb + ".npy", fp_mbh + ".npy", fp_psd, nwalkers_pe)
    stop = SearchConvergeStopping(n_iters=100, diff=0.01, start_iteration=0, verbose=True)
    sampler_mix = EnsembleSampler(
        nwalkers_pe,
        [4, 5],  # assumes ndim_max
        log_like,
        priors,
        tempering_kwargs={"betas": betas, "permute": True},  # , "skip_swap_supp_names": ["walker_inds"]},
        nbranches=len(branch_names),
        nleaves_max=[1, 1],
        kwargs=sens_kwargs,  # {"start_freq_ind": start_freq_ind, **waveform_kwargs},
        args=(xp.asarray(fd), data),
        backend=fp_psd_search_initial,
        vectorize=True,
        periodic=periodic,  # TODO: add periodic to proposals
        branch_names=branch_names,
        stopping_fn=stop,
        stopping_iterations=1,
        # update_fn=update,  # sttop_converge_mix,
        # update_iterations=-11,
        provide_groups=False,
        provide_supplimental=False,  # True,
        # TODO: add convergence?
    )

    lp = sampler_mix.compute_log_prior(state_mix.branches_coords, inds=state_mix.branches_inds)
    ll = sampler_mix.compute_log_like(state_mix.branches_coords, inds=state_mix.branches_inds, logp=lp)[0] # , supps=state_mix.supplimental

    state_mix.log_like = ll
    state_mix.log_prior = lp

    # equlibrating likelihood check: -4293090.6483655665,
    nsteps_mix = 10000

    print("Starting mix ll best:", state_mix.log_like.max(axis=-1))
    mempool.free_all_blocks()

    out = sampler_mix.run_mcmc(state_mix, nsteps_mix, progress=True, thin_by=25, store=True)
    print("ending mix ll best:", out.log_like.max(axis=-1))

    inds_max = np.where(sampler_mix.get_log_like() == sampler_mix.get_log_like().max())

    psd_vals = sampler_mix.get_chain()["psd"][inds_max][0, 0]
    gf_vals = sampler_mix.get_chain()["galfor"][inds_max][0, 0]
    
    best_A = get_sensitivity(fd, sens_fn="noisepsd_AE", model=psd_vals[:2], foreground_params=gf_vals)
    best_E = get_sensitivity(fd, sens_fn="noisepsd_AE", model=psd_vals[2:], foreground_params=gf_vals)
    
    np.save("best_logl_psd_from_psd_run", np.asarray([best_A, best_E]))
    return (best_A, best_E)

if __name__ == "__main__":
    import argparse
    """parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=int,
                        help='which gpu', required=True)

    args = parser.parse_args()"""


    imported = False

    """A_going_in = np.zeros((nwalkers_pe, A_inj.shape[0]), dtype=complex)
    E_going_in = np.zeros((nwalkers_pe, E_inj.shape[0]), dtype=complex)

    A_going_in[:] = np.asarray(A_inj)
    E_going_in[:] = np.asarray(E_inj)

    
    """

    A_going_in = np.asarray(A_inj).copy()
    E_going_in = np.asarray(E_inj).copy()

    # Apsd, Epsd = np.load("best_logl_psd_from_psd_run.npy")
    """mbh_inj = np.load("best_logl_mbhs_from_psd_run.npy")

    while "best_logl_gbs_from_psd_run.npy" not in os.listdir():  # 
        print(f"best_logl_gbs_from_psd_run.npy not in current directory so far...")
        time.sleep(20)

    gbs_tmp = np.load("best_logl_gbs_from_psd_run.npy")

    A_going_in[:] -= gbs_tmp[0]
    E_going_in[:] -= gbs_tmp[1]

    A_mbh_going_in = mbh_inj[0]
    E_mbh_going_in = mbh_inj[1]

    A_going_in[:] -= A_mbh_going_in
    E_going_in[:] -= E_mbh_going_in"""
    
    output = run_psd_search(A_going_in, E_going_in, 7)
                