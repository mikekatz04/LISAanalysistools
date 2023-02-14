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
    psd_pars = x[0]
    galfor_pars = x[1]

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
    def __init__(self, fd, fp_gb, fp_psd, nwalkers):
        self.fp_gb = fp_gb
        self.fp_psd = fp_psd
        self.nwalkers = nwalkers
        self.fd = xp.asarray(fd)

    def __call__(self, iter, last_sample, sampler):

        imported = False
        while not imported:
            try:
                data_in = np.load(self.fp_gb)
                imported = True
            except ValueError:
                time.sleep(1)

        sampler.log_like_fn.args[1][0][:] = xp.asarray(data_in[:, 0])
        sampler.log_like_fn.args[1][1][:] = xp.asarray(data_in[:, 1])

        lp = sampler.compute_log_prior(last_sample.branches_coords, inds=last_sample.branches_inds)
        ll = sampler.compute_log_like(last_sample.branches_coords, inds=last_sample.branches_inds, supps=last_sample.supplimental, logp=lp)[0]

        last_sample.log_like = ll
        last_sample.log_prior = lp

        num = self.nwalkers
        psd_pars = last_sample.branches_coords["psd"][0].reshape(-1, 4)
        galfor_pars = last_sample.branches_coords["galfor"][0].reshape(-1, 5)

        psds_out = xp.zeros((num, 2, len(self.fd)))
        for i, (psd_par, galfor_par) in enumerate(zip(psd_pars, galfor_pars)):
            psds_out[i, 0] = get_sensitivity(self.fd, sens_fn="noisepsd_AE", model=psd_par[:2], foreground_params=galfor_par, xp=xp)
            psds_out[i, 1] = get_sensitivity(self.fd, sens_fn="noisepsd_AE", model=psd_par[2:], foreground_params=galfor_par, xp=xp)

            psds_out[i, 0, 0] = psds_out[i, 0, 1]
            psds_out[i, 1, 0] = psds_out[i, 1, 1]

        np.save(self.fp_psd, psds_out.get())
        del psds_out
        xp.get_default_memory_pool().free_all_blocks()


def run_psd_pe(gpu, fp_gb):

    while fp_gb not in os.listdir():
        print(f"{fp_gb} not in current directory so far...")
        time.sleep(20)

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
    
    if fp_psd_pe in os.listdir():
        reader = HDFBackend(fp_psd_pe)
        last_sample = reader.get_last_sample()

        coords = {key: last_sample.branches_coords[key] for key in branch_names}
        inds = {key: last_sample.branches_inds[key] for key in branch_names}

        last_sample = State(coords, inds=inds, log_like=last_sample.log_like, log_prior=last_sample.log_prior)
        
    elif current_save_state_file_psd in os.listdir():
        with open(current_save_state_file, "rb") as fp_out:
            last_sample = pickle.load(fp_out)

        coords = {key: last_sample.branches_coords[key] for key in branch_names}
        inds = {key: last_sample.branches_inds[key] for key in branch_names}

        last_sample = State(coords, inds=inds, log_like=last_sample.log_like, log_prior=last_sample.log_prior)

    else:
        coords = {}
        inds = {}

        coords["psd"] = priors["psd"].rvs(size=(ntemps_pe, nwalkers_pe))
        inds["psd"] = np.ones((ntemps_pe, nwalkers_pe, 1), dtype=bool)
        coords["galfor"] = priors["galfor"].rvs(size=(ntemps_pe, nwalkers_pe))
        inds["galfor"] = np.ones((ntemps_pe, nwalkers_pe, 1), dtype=bool)

        last_sample = State(coords, inds=inds)

    A_going_in = np.zeros((nwalkers_pe, A_inj.shape[0]), dtype=complex)
    E_going_in = np.zeros((nwalkers_pe, E_inj.shape[0]), dtype=complex)

    data_in = np.load(fp_gb)
    A_going_in[:] = data_in[:, 0]
    E_going_in[:] = data_in[:, 1]

    data = [xp.asarray(A_going_in), xp.asarray(E_going_in)]

    psd_params = last_sample.branches["psd"].coords.reshape(-1, last_sample.branches["psd"].shape[-1])

    foreground_params = last_sample.branches["galfor"].coords.reshape(-1, last_sample.branches["galfor"].shape[-1])

    walker_vals = np.tile(np.arange(nwalkers_pe), (ntemps_pe, 1))

    supps_base_shape = (ntemps_pe, nwalkers_pe)
    supps = BranchSupplimental({"walker_inds": walker_vals}, base_shape=supps_base_shape, copy=True)

    state_mix = State(last_sample.branches_coords, inds=last_sample.branches_inds, supplimental=supps)

    from eryn.moves.tempering import make_ladder
    betas = make_ladder(7, ntemps=ntemps_pe)

    sens_kwargs = dict(sens_fn="noisepsd_AE")

    update = UpdateNewResiduals(fd, fp_gb, fp_psd, nwalkers_pe)

    sampler_mix = EnsembleSampler(
        nwalkers_pe,
        [4, 5],  # assumes ndim_max
        log_like,
        priors,
        tempering_kwargs={"betas": betas, "permute": False, "skip_swap_supp_names": ["walker_inds"]},
        nbranches=len(branch_names),
        nleaves_max=[1, 1],
        kwargs=sens_kwargs,  # {"start_freq_ind": start_freq_ind, **waveform_kwargs},
        args=(xp.asarray(fd), data),
        backend=fp_psd_pe,
        vectorize=False,
        periodic=periodic,  # TODO: add periodic to proposals
        branch_names=branch_names,
        update_fn=update,  # stop_converge_mix,
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
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=int,
                        help='which gpu', required=True)

    parser.add_argument('--gb-file', '-gf', type=str,
                        help='which psd file', required=True)

    args = parser.parse_args()

    output = run_psd_pe(args.gpu, args.gb_file)
                