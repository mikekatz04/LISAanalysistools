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
    if supps is None:
        raise ValueError("Must provide supps to identify the data streams.")

    wi = supps["walker_inds"]

    if isinstance(x, list):
        psd_pars = x[0]
        galfor_pars = x[1]
    else:
        psd_pars = x
        galfor_pars = None

    psd = [
        get_sensitivity(freqs, model=psd_pars[:2], foreground_params=galfor_pars, **sens_kwargs),
        get_sensitivity(freqs, model=psd_pars[2:], foreground_params=galfor_pars, **sens_kwargs)
    ]
    psd[0][0] = psd[0][1]
    psd[1][0] = psd[1][1]

    inner_product = 4 * df * (xp.sum(data[0][wi].conj() * data[0][wi] / psd[0]) + xp.sum(data[1][wi].conj() * data[1][wi] / psd[1])).real
    ll = -1/2 * inner_product - xp.sum(xp.log(xp.asarray(psd)))
    return ll.get()


from eryn.utils.updates import Update


class UpdateNewResiduals(Update):
    def __init__(self, fd, fp_gb, fp_mbh, fp_psd, nwalkers, last_mbhs, include_foreground=True, include_gbs=True):
        self.fp_gb = fp_gb
        self.fp_psd = fp_psd
        self.nwalkers = nwalkers
        self.fd = xp.asarray(fd)
        self.fp_mbh = fp_mbh
        self.include_foreground = include_foreground
        self.include_gbs = include_gbs
        self.last_mbhs = last_mbhs

    def __call__(self, iter, last_sample, sampler):
        
        # read out best residual and psd
        if not self.include_gbs:
            assert not self.include_foreground
            best = np.where(last_sample.log_like == last_sample.log_like.max())
            mbh_index = best[1][0]
            np.save("best_logl_mbhs_from_psd_run", self.last_mbhs[mbh_index])
            
            psd_par = last_sample.branches_coords["psd"][best].squeeze()
            A_psd_best = get_sensitivity(self.fd, sens_fn="noisepsd_AE", model=psd_par[:2], foreground_params=None, xp=xp).get()
            E_psd_best = get_sensitivity(self.fd, sens_fn="noisepsd_AE", model=psd_par[2:], foreground_params=None, xp=xp).get()
            np.save("best_logl_psd_from_psd_run", np.array([A_psd_best, E_psd_best]))
        
        A_going_in = np.zeros((self.nwalkers, A_inj.shape[0]), dtype=complex)
        E_going_in = np.zeros((self.nwalkers, E_inj.shape[0]), dtype=complex)

        A_going_in[:] = np.asarray(A_inj)
        E_going_in[:] = np.asarray(E_inj)
    
        if self.include_gbs:
            imported = False
            while not imported:
                try:
                    data_in = np.load(self.fp_gb)
                    imported = True
                except ValueError:
                    time.sleep(1)

            A_going_in[:] -= data_in[:, 0]
            E_going_in[:] -= data_in[:, 1]

        imported = False
        while not imported:
            try:
                mbh_inj = np.load(self.fp_mbh)
                imported = True
            except ValueError:
                time.sleep(1)
        
        self.last_mbhs = mbh_inj

        A_mbh_going_in = np.zeros((self.nwalkers, A_inj.shape[0]), dtype=complex)
        E_mbh_going_in = np.zeros((self.nwalkers, E_inj.shape[0]), dtype=complex)

        A_mbh_going_in[:] = mbh_inj[:, 0][None, :]
        E_mbh_going_in[:] = mbh_inj[:, 1][None, :]

        A_going_in[:] -= A_mbh_going_in
        E_going_in[:] -= E_mbh_going_in

        sampler.log_like_fn.args[1][0][:] = xp.asarray(A_going_in)
        sampler.log_like_fn.args[1][1][:] = xp.asarray(E_going_in)

        lp = sampler.compute_log_prior(last_sample.branches_coords, inds=last_sample.branches_inds)
        ll = sampler.compute_log_like(last_sample.branches_coords, inds=last_sample.branches_inds, supps=last_sample.supplimental, logp=lp)[0]

        last_sample.log_like = ll
        last_sample.log_prior = lp

        num = self.nwalkers
        psd_pars = last_sample.branches_coords["psd"][0].reshape(-1, 4)
        
        if self.include_foreground:
            galfor_pars = last_sample.branches_coords["galfor"][0].reshape(-1, 5)
        else:
            galfor_pars = [None for _ in psd_pars]

        psds_out = xp.zeros((num, 2, len(self.fd)))
        for i, (psd_par, galfor_par) in enumerate(zip(psd_pars, galfor_pars)):
            psds_out[i, 0] = get_sensitivity(self.fd, sens_fn="noisepsd_AE", model=psd_par[:2], foreground_params=galfor_par, xp=xp)
            psds_out[i, 1] = get_sensitivity(self.fd, sens_fn="noisepsd_AE", model=psd_par[2:], foreground_params=galfor_par, xp=xp)

            psds_out[i, 0, 0] = psds_out[i, 0, 1]
            psds_out[i, 1, 0] = psds_out[i, 1, 1]

        np.save(self.fp_psd, psds_out.get())
        del psds_out
        xp.get_default_memory_pool().free_all_blocks()


def run_psd_pe(gpu):
    search = True
    include_gbs = True
    include_gb_foreground = True

    fp_gb_here = fp_gb if not search else fp_gb_template_search

    if include_gbs:
        while fp_gb_here + ".npy" not in os.listdir():
            print(f"{fp_gb_here + '.npy'} not in current directory so far...")
            time.sleep(20)

    if include_gb_foreground:
        branch_names = ["psd", "galfor"]
    else:
        branch_names = ["psd"]

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

    A_going_in = np.zeros((nwalkers_pe, A_inj.shape[0]), dtype=complex)
    E_going_in = np.zeros((nwalkers_pe, E_inj.shape[0]), dtype=complex)

    A_going_in[:] = np.asarray(A_inj)
    E_going_in[:] = np.asarray(E_inj)

    fp_mbh_here = fp_mbh if not search else fp_mbh_template_search
    if include_gbs:
        data_in = np.load(fp_gb_here + ".npy")

        A_going_in[:] -= data_in[:, 0]
        E_going_in[:] -= data_in[:, 1]

    mbh_inj = np.load(fp_mbh_here + ".npy")

    A_mbh_going_in = np.zeros((nwalkers_pe, A_inj.shape[0]), dtype=complex)
    E_mbh_going_in = np.zeros((nwalkers_pe, E_inj.shape[0]), dtype=complex)

    A_mbh_going_in[:] = mbh_inj[:, 0]
    E_mbh_going_in[:] = mbh_inj[:, 1]

    A_going_in[:] -= A_mbh_going_in
    E_going_in[:] -= E_mbh_going_in

    data = [xp.asarray(A_going_in), xp.asarray(E_going_in)]

    walker_vals = np.tile(np.arange(nwalkers_pe), (ntemps_pe, 1))

    supps_base_shape = (ntemps_pe, nwalkers_pe)
    supps = BranchSupplimental({"walker_inds": walker_vals}, base_shape=supps_base_shape, copy=True)

    fp_psd_pe_here = fp_psd_pe if not search else fp_psd_search
    
    sens_kwargs = dict(sens_fn="noisepsd_AE")
 
    if fp_psd_pe_here in os.listdir():
        reader = HDFBackend(fp_psd_pe_here)
        last_sample = reader.get_last_sample()

        coords = {key: last_sample.branches_coords[key] for key in last_sample.branches_coords}
        inds = {key: last_sample.branches_inds[key] for key in last_sample.branches_coords}

        if "galfor" not in coords:
            if include_gb_foreground:
                coords["galfor"] = priors["galfor"].rvs(size=(ntemps_pe, nwalkers_pe))
                inds["galfor"] = np.ones((ntemps_pe, nwalkers_pe, 1), dtype=bool)
                os.remove(fp_psd_pe_here)

        coords = {key: coords[key] for key in branch_names}
        inds = {key: inds[key] for key in branch_names}

        last_sample = State(coords, inds=inds, log_like=last_sample.log_like, log_prior=last_sample.log_prior)
        
    elif False:  # current_save_state_file_psd in os.listdir():
        print("LOADING PSD save state")
        with open(current_save_state_file_psd, "rb") as fp_out:
            last_sample = pickle.load(fp_out)

        coords = {key: last_sample.branches_coords[key] for key in branch_names}
        inds = {key: last_sample.branches_inds[key] for key in branch_names}

        last_sample = State(coords, inds=inds, log_like=last_sample.log_like, log_prior=last_sample.log_prior)

    else:
        coords = {}
        inds = {}

        psd_reader = HDFBackend(fp_psd_search_initial)
        best_psd_ind = psd_reader.get_log_like().flatten().argmax()
        best_psd_params = psd_reader.get_chain()["psd"].reshape(-1, 4)[best_psd_ind]

        cov = np.ones(len(best_psd_params))
        factor = 1e-6

        fd_gpu = xp.asarray(fd)
        iter_check = 0
        max_iter = 1000
        start_like = np.zeros(ntemps_pe * nwalkers_pe)
        while np.std(start_like) < 20.0:
            
            logp = np.full_like(start_like, -np.inf)
            tmp = np.zeros((ntemps_pe * nwalkers_pe, 4))
            fix = np.ones((ntemps_pe * nwalkers_pe), dtype=bool)
            while np.any(fix):
                tmp[fix] = (best_psd_params[None, :] * (1. + factor * cov * np.random.randn(nwalkers_pe * ntemps_pe, 4)))[fix]

                logp = priors["psd"].logpdf(tmp)

                fix = np.isinf(logp)
                if np.all(fix):
                    breakpoint()

            start_like = np.asarray([
                log_like(tmp[i], fd_gpu, data, supps={"walker_inds": supps[:]["walker_inds"].flatten()[i]}, **sens_kwargs)
            for i in range(len(tmp))])

            iter_check += 1
            factor *= 1.5

            print(np.std(start_like))

            if iter_check > max_iter:
                raise ValueError("Unable to find starting parameters.")

        mempool.free_all_blocks()
        coords["psd"] = tmp.reshape(ntemps_pe, nwalkers_pe, 4)
        inds["psd"] = np.ones((ntemps_pe, nwalkers_pe, 1), dtype=bool)

        if include_gb_foreground:
            coords["galfor"] = priors["galfor"].rvs(size=(ntemps_pe, nwalkers_pe))
            inds["galfor"] = np.ones((ntemps_pe, nwalkers_pe, 1), dtype=bool)

        last_sample = State(coords, inds=inds)

    state_mix = State(last_sample.branches_coords, inds=last_sample.branches_inds, supplimental=supps)

    from eryn.moves.tempering import make_ladder
    betas = make_ladder(9, ntemps=ntemps_pe)

    fp_psd_here = fp_psd if not search else fp_psd_residual_search
    update = UpdateNewResiduals(fd, fp_gb_here + ".npy", fp_mbh_here + ".npy", fp_psd_here, nwalkers_pe, mbh_inj, include_foreground=include_gb_foreground, include_gbs=include_gbs)

    ndims_in = [4, 5] if include_gb_foreground else [4]
    nleaves_max_in = [1, 1] if include_gb_foreground else [1]

    sampler_mix = EnsembleSampler(
        nwalkers_pe,
        ndims_in,  # assumes ndim_max
        log_like,
        priors,
        tempering_kwargs={"betas": betas, "permute": False, "skip_swap_supp_names": ["walker_inds"]},
        nbranches=len(branch_names),
        nleaves_max=nleaves_max_in,
        kwargs=sens_kwargs,  # {"start_freq_ind": start_freq_ind, **waveform_kwargs},
        args=(xp.asarray(fd), data),
        backend=fp_psd_pe_here,
        vectorize=False,
        periodic=periodic,  # TODO: add periodic to proposals
        branch_names=branch_names,
        update_fn=update,  # sttop_converge_mix,
        update_iterations=4,
        provide_groups=False,
        provide_supplimental=True,
    )

    lp = sampler_mix.compute_log_prior(state_mix.branches_coords, inds=state_mix.branches_inds)
    ll = sampler_mix.compute_log_like(state_mix.branches_coords, inds=state_mix.branches_inds, supps=state_mix.supplimental, logp=lp)[0]

    state_mix.log_like = ll
    state_mix.log_prior = lp

    # equlibrating likelihood check: -4293090.6483655665,
    nsteps_mix = 10000

    print("Starting mix ll best:", state_mix.log_like.max(axis=-1))
    mempool.free_all_blocks()
    
    out = sampler_mix.run_mcmc(state_mix, nsteps_mix, progress=True, thin_by=25, store=True)
    print("ending mix ll best:", out.log_like.max(axis=-1))

if __name__ == "__main__":
    import argparse
    """parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=int,
                        help='which gpu', required=True)

    args = parser.parse_args()"""

    output = run_psd_pe(2)
                