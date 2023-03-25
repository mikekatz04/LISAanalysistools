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
from lisatools.sampling.moves.mbhspecialmove import MBHSpecialMove

from bbhx.waveformbuild import BBHWaveformFD
from bbhx.waveforms.phenomhm import PhenomHMAmpPhase
from bbhx.response.fastfdresponse import LISATDIResponse
from bbhx.likelihood import Likelihood as MBHLikelihood
from bbhx.likelihood import HeterodynedLikelihood
from bbhx.utils.constants import *
from bbhx.utils.transform import *

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
    np.asarray(A_inj).copy(),
    np.asarray(E_inj).copy(),
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



from eryn.utils.updates import Update


class UpdateNewResiduals(Update):
    def __init__(self, mbh_move, last_gb_residuals, fp_mbh, fp_psd, fp_gb, psd_shape, include_gbs=True, search=False):
        self.mbh_move = mbh_move
        self.fp_mbh = fp_mbh
        self.fp_psd = fp_psd
        self.psd_shape = psd_shape
        self.fp_gb = fp_gb
        self.last_gb_residuals = last_gb_residuals
        self.include_gbs = include_gbs
        self.search = search

    def __call__(self, iter, last_sample, sampler):

        imported = False
        if self.search:
            while not imported:
                try:
                    psd_best_search = np.load("best_logl_psd_from_psd_run.npy")
                    imported = True
                except ValueError:
                    time.sleep(1)

            psd_best_search[:, 0] = psd_best_search[:, 1]

            psd_tmp = xp.zeros_like(self.mbh_move.psd[:2])

            psd_tmp[0, :] = xp.asarray(psd_best_search[0])
            psd_tmp[1, :] = xp.asarray(psd_best_search[1])
        else:
            while not imported:
                try:
                    psds = np.load(self.fp_psd)
                    imported = True
                except ValueError:
                    time.sleep(1)

            psd_tmp = xp.asarray(psds).transpose(1, 0, 2)

        self.mbh_move.psd[:2] = psd_tmp
        del psd_tmp
        xp.get_default_memory_pool().free_all_blocks()

        if self.include_gbs:
            gb_removal = xp.asarray(self.last_gb_residuals).transpose(1, 0, 2)
            self.mbh_move.data_residuals[:2] += gb_removal
            del gb_removal
            xp.get_default_memory_pool().free_all_blocks()
            if self.search:
                gbs_in = np.zeros_like(self.last_gb_residuals)
                imported = False
                while not imported:
                    try:
                        gbs_tmp = np.load("best_logl_gbs_from_psd_run.npy")
                        imported = True
                    except ValueError:
                        time.sleep(1)

                gbs_in[:] = gbs_tmp[None, :, :]

            else:
                imported = False
                while not imported:
                    try:
                        gbs_in = np.load(self.fp_gb)
                        imported = True
                    except ValueError:
                        time.sleep(1)

            self.last_gb_residuals[:] = gbs_in[:]

            gb_add = xp.asarray(gbs_in)
            self.mbh_move.data_residuals[:2] -= gb_add.transpose(1, 0, 2)
            del gb_add

        xp.get_default_memory_pool().free_all_blocks()

        ntemps, nwalkers, nleaves_max, ndim = last_sample.branches["mbh"].shape

        ind_best = np.where(last_sample.log_like[0] == last_sample.log_like[0].max())[0][0]
        # TODO: change for adding T
        waveforms = xp.zeros_like(self.mbh_move.data_residuals[:2])

        waveforms_best = xp.zeros_like(self.mbh_move.data_residuals[:2, 0])
        for leaf in range(nleaves_max):
            coords = last_sample.branches["mbh"].coords[0, :, leaf].reshape(-1, ndim)
            coords_in = self.mbh_move.transform_fn.both_transforms(coords)
            waveform_tmp = self.mbh_move.waveform_gen(*coords_in.T, fill=True, freqs=self.mbh_move.fd, **self.mbh_move.mbh_kwargs).transpose(1, 0, 2)
            waveforms_best[:] += waveform_tmp[:2, ind_best]
            waveforms[:] += waveform_tmp[:2]

        np.save("best_logl_mbhs_from_psd_run", waveforms_best)
        np.save(self.fp_mbh, waveforms.transpose(1, 0, 2).get())
        del waveforms, waveform_tmp
        xp.get_default_memory_pool().free_all_blocks()


def run_mbh_pe(gpu):

    include_gbs = True
    search = True
    fp_psd_here = fp_psd if not search else fp_psd_residual_search
    fp_mbh_here = fp_mbh if not search else fp_mbh_template_search
    fp_mbh_pe_here = fp_mbh_pe if not search else fp_mbh_search
    fp_gb_here = fp_gb if not search else fp_gb_template_search
    
    if include_gbs:
        while "best_logl_gbs_from_psd_run.npy" not in os.listdir():  # 
            print(f"{fp_gb_here + '.npy'} not in current directory so far...")
            time.sleep(20)

    branch_names = ["mbh"]

    gpus_pe = [gpu]
    gpus = gpus_pe
    # from lisatools.sampling.stopping import SearchConvergeStopping2
    # for testing
    # search_f_bin_lims = np.arange(f0_lims_in[0], f0_lims_in[1], 2 * 128 * df)

    num_sub_bands = len(search_f_bin_lims)

    xp.cuda.runtime.setDevice(gpus[0])

    nwalkers_pe = 100
    ntemps_pe = 10

    A_going_in = np.zeros((nwalkers_pe, A_inj.shape[0]), dtype=complex)
    E_going_in = np.zeros((nwalkers_pe, E_inj.shape[0]), dtype=complex)

    A_going_in[:] = np.asarray(A_inj).copy()
    E_going_in[:] = np.asarray(E_inj).copy()

    if include_gbs:
        if search:
            gbs_in = np.zeros((nwalkers_pe, 2, A_inj.shape[0]), dtype=complex)
            gbs_tmp = np.load("best_logl_gbs_from_psd_run.npy")     
            gbs_in[:] = gbs_tmp[None, :, :]

        else:
            gbs_in = np.load(fp_gb_here + ".npy")

        A_going_in = np.zeros((nwalkers_pe, A_inj.shape[0]), dtype=complex)
        E_going_in = np.zeros((nwalkers_pe, E_inj.shape[0]), dtype=complex)

        A_going_in -= gbs_in[:, 0]
        E_going_in -= gbs_in[:, 1]

    else:
        gbs_in = None

    A_psd_in = np.zeros((nwalkers_pe, A_inj.shape[0]), dtype=np.float64)
    E_psd_in = np.zeros((nwalkers_pe, E_inj.shape[0]), dtype=np.float64)

    if search:
        psd_best_search = np.load("best_logl_psd_from_psd_run.npy")
        psd_best_search[:, 0] = psd_best_search[:, 1]

        A_psd_in[:] = np.asarray(psd_best_search[0])
        E_psd_in[:] = np.asarray(psd_best_search[1])
    else:
        psds = np.load(fp_psd_here + ".npy" )
        psds[:, :, 0] = psds[:, :, 1]
        A_psd_in[:] = psds[:, 0] # A
        E_psd_in[:] = psds[:, 1]  # E

    data_fin = xp.asarray([A_going_in, E_going_in, np.zeros_like(E_going_in)])
    psds_fin = xp.asarray([A_psd_in, E_psd_in, np.full_like(E_psd_in, 1e10)])

    start_ll_check = (-1/2 * 4 * df * xp.sum(data_fin.conj() * data_fin / psds_fin, axis=(0, 2)) - xp.sum(xp.log(xp.asarray(psds_fin)), axis=(0, 2))).get().max()

    # templates_fin = xp.zeros_like(data_fin)
    walker_vals = np.tile(np.arange(nwalkers_pe), (ntemps_pe, 1))

    from eryn.moves.tempering import make_ladder
    betas = make_ladder(11, ntemps=ntemps_pe)

    sens_kwargs = dict(sens_fn="noisepsd_AE")

    like_mix = BasicResidualMGHLikelihood(None)

    wave_gen = BBHWaveformFD(
        amp_phase_kwargs=dict(run_phenomd=True),
        response_kwargs=dict(TDItag="AET"),
        use_gpu=use_gpu
    )

    # TODO: fix this?
    wave_gen.d_d = 0.0

    # for transforms
    fill_dict = {
        "ndim_full": 12,
        "fill_values": np.array([0.0]),
        "fill_inds": np.array([6]),
    }

    # priors
    priors = {
        "mbh": ProbDistContainer(
            {
                0: uniform_dist(np.log(1e5), np.log(1e7)),
                1: uniform_dist(0.01, 0.999999999),
                2: uniform_dist(-0.99999999, +0.99999999),
                3: uniform_dist(-0.99999999, +0.99999999),
                4: uniform_dist(0.01, 1000.0),
                5: uniform_dist(0.0, 2 * np.pi),
                6: uniform_dist(-1.0 + 1e-6, 1.0 - 1e-6),
                7: uniform_dist(0.0, 2 * np.pi),
                8: uniform_dist(-1.0 + 1e-6, 1.0 - 1e-6),
                9: uniform_dist(0.0, np.pi),
                10: uniform_dist(0.0, Tobs + 3600.0),
            }
        ) 
    }

    waveform_kwargs = {
        "modes": [(2,2)],
        "length": 1024,
    }

    # transforms from pe to waveform generation
    parameter_transforms = {
        0: np.exp,
        4: lambda x: x * PC_SI * 1e9,  # Gpc
        7: np.arccos,
        9: np.arcsin,
        (0, 1): mT_q,
        (11, 8, 9, 10): LISA_to_SSB,
    }

    transform_fn = TransformContainer(
        parameter_transforms=parameter_transforms,
        fill_dict=fill_dict,
    )

    # sampler treats periodic variables by wrapping them properly
    periodic = {
        "mbh": {5: 2 * np.pi, 7: np.pi, 8: np.pi}
    }
    
    fd_gpu = xp.asarray(fd)
    found_mbh_points = np.load("mbh_injection_points_after_initial_search.npy")
    
    nleaves_mbh = found_mbh_points.shape[0]
    start_points = np.zeros((ntemps_pe * nwalkers_pe, found_mbh_points.shape[0], 11))
    ndim = 11

    print(f"NUMBER of MBHS: {found_mbh_points.shape[0]}")

    if fp_mbh_pe_here in os.listdir():
        reader = HDFBackend(fp_mbh_pe_here)
        start_state = reader.get_last_sample()

        for inj_i, mbh_inj in enumerate(found_mbh_points):
            tmp = start_state.branches["mbh"].coords[0, :, inj_i]
            injection_in = transform_fn.both_transforms(tmp, return_transpose=True)

            injection_in_keep = injection_in[:, :nwalkers_pe]
            data_channels_AET = wave_gen(*injection_in_keep, freqs=fd_gpu,
                    modes=[(2,2)], direct=False, fill=True, squeeze=True, length=1024
                )
            data_fin[:2] -= data_channels_AET.transpose((1, 0, 2))[:2]
      
    else:
        start_ll_check = (-1/2 * 4 * df * xp.sum(data_fin.conj() * data_fin / psds_fin, axis=(0, 2)) - xp.sum(xp.log(xp.asarray(psds_fin)), axis=(0, 2))).get().max()
        old_ll = start_ll_check
        print(start_ll_check, start_ll_check - old_ll)
        for inj_i, mbh_inj in enumerate(found_mbh_points):
            if "start_points_mbh.npy" not in os.listdir():
                # generate starting points
                factor = 1e-6
                cov = np.ones(ndim) * 1e-3
                cov[0] = 1e-5
                cov[-1] = 1e-5

                start_like = np.zeros((nwalkers_pe * ntemps_pe))
                iter_check = 0
                max_iter = 1000
                while np.std(start_like) < 5.0:
                    
                    logp = np.full_like(start_like, -np.inf)
                    tmp = np.zeros((nwalkers_pe * ntemps_pe, ndim))
                    fix = np.ones((nwalkers_pe * ntemps_pe,), dtype=bool)
                    while np.any(fix):
                        tmp[fix] = (mbh_inj[None, :] * (1. + factor * cov * np.random.randn(nwalkers_pe * ntemps_pe, ndim)))[fix]

                        tmp[:, 5] = tmp[:, 5] % (2 * np.pi)
                        tmp[:, 7] = tmp[:, 7] % (2 * np.pi)
                        tmp[:, 9] = tmp[:, 9] % (1 * np.pi)
                        logp = priors["mbh"].logpdf(tmp)

                        fix = np.isinf(logp)
                        if np.all(fix):
                            breakpoint()
                    
                    injection_in = transform_fn.both_transforms(tmp, return_transpose=True)
                    data_index = np.tile(np.arange(nwalkers_pe), (ntemps_pe, 1)).flatten().astype(np.int32)
                    noise_index = np.tile(np.arange(nwalkers_pe), (ntemps_pe, 1)).flatten().astype(np.int32)

                    data_index = np.zeros((ntemps_pe * nwalkers_pe), dtype=np.int32)
                    noise_index = np.zeros((ntemps_pe * nwalkers_pe), dtype=np.int32)

                    if iter_check == 0:
                        base_like = (-1/2 * 4 * df * xp.sum(data_fin.conj() * data_fin / psds_fin, axis=(0, 2)) - xp.sum(xp.log(xp.asarray(psds_fin)), axis=(0, 2))).get()
                        xp.get_default_memory_pool().free_all_blocks()
                        wave_gen.d_d = -2 * base_like[0]

                    tmp_like = wave_gen.get_direct_ll(fd_gpu, data_fin.flatten(), psds_fin.flatten(), df, *injection_in, noise_index=noise_index, data_index=data_index, **waveform_kwargs)

                    # ensures spread is based on mbhs added
                    start_like = tmp_like.real

                    iter_check += 1
                    factor *= 1.5

                    # print(np.std(start_like))

                    if iter_check > max_iter:
                        raise ValueError("Unable to find starting parameters.")

            else:
                tmp = np.load("start_points_mbh.npy")[:, inj_i]
                injection_in = transform_fn.both_transforms(tmp, return_transpose=True)

            # print(inj_i, (wave_gen.h_h.real / wave_gen.d_h.real).min(), (wave_gen.h_h.real / wave_gen.d_h.real).max())
            # breakpoint()
            injection_in_keep = injection_in[:, :nwalkers_pe]
            data_channels_AET = wave_gen(*injection_in_keep, freqs=fd_gpu,
                    modes=[(2,2)], direct=False, fill=True, squeeze=True, length=1024
                )

            data_fin[:2] -= data_channels_AET.transpose((1, 0, 2))[:2]
            # templates_fin[:2] += data_channels_AET.transpose((1, 0, 2))[:2]
            start_points[:, inj_i] = tmp
            start_ll_check = (-1/2 * 4 * df * xp.sum(data_fin.conj() * data_fin / psds_fin, axis=(0, 2)) - xp.sum(xp.log(xp.asarray(psds_fin)), axis=(0, 2))).get()
            print(start_ll_check.max(), start_ll_check.max() - old_ll)
            old_ll = start_ll_check.max()
            best = start_ll_check.argmax()
        #np.save("best_check_mbh", templates_fin[:, best].get())
        np.save("start_points_mbh", start_points)

        start_ll_cold = (-1/2 * 4 * df * xp.sum(data_fin.conj() * data_fin / psds_fin, axis=(0, 2)) - xp.sum(xp.log(xp.asarray(psds_fin)), axis=(0, 2))).get()

        start_ll = np.zeros((ntemps_pe, nwalkers_pe))
        start_ll[0] = start_ll_cold
        
        start_lp = priors["mbh"].logpdf(start_points.reshape(-1, ndim)).reshape(ntemps_pe, nwalkers_pe, nleaves_mbh).sum(axis=-1)
        start_lp[1:] = 0.0

        start_state = State({"mbh": start_points.reshape(ntemps_pe, nwalkers_pe, nleaves_mbh, ndim)}, log_like=start_ll, log_prior=start_lp)

    xp.get_default_memory_pool().free_all_blocks()
    num_repeats = 10

    # TODO: start ll needs to be done carefully

    from eryn.moves import StretchMove
    from lisatools.sampling.moves.skymodehop import SkyMove

    inner_moves = [
        (SkyMove(which="both"), 0.00),
        (SkyMove(which="long"), 0.00),
        (SkyMove(which="lat"), 0.08),
        (StretchMove(), 0.92)
    ]
    move = MBHSpecialMove(wave_gen, fd_gpu, data_fin, psds_fin, num_repeats, transform_fn, priors, waveform_kwargs, inner_moves, df)

    update = UpdateNewResiduals(move, gbs_in, fp_mbh_here + ".npy", fp_psd_here + ".npy", fp_gb_here + ".npy", A_psd_in.shape, include_gbs=include_gbs, search=search)

    # key permute is False
    sampler_mix = EnsembleSampler(
        nwalkers_pe,
        [11],  # assumes ndim_max
        like_mix,
        priors,
        moves=move,
        tempering_kwargs={"betas": betas, "permute": False},
        nbranches=len(branch_names),
        nleaves_max=[nleaves_mbh],
        nleaves_min=[nleaves_mbh],
        backend=fp_mbh_pe_here,
        vectorize=True,
        periodic=periodic,  # TODO: add periodic to proposals
        branch_names=branch_names,
        update_fn=update,  # stop_converge_mix,
        update_iterations=1,
        provide_groups=False,
        provide_supplimental=False,
    )

    # equlibrating likelihood check: -4293090.6483655665,
    nsteps_mix = 10000
    mempool.free_all_blocks()
    out = sampler_mix.run_mcmc(start_state, nsteps_mix, progress=True, thin_by=1, store=True)
    print("ending mix ll best:", out.log_like.max(axis=-1))

if __name__ == "__main__":
    import argparse
    """parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=int,
                        help='which gpu', required=True)

    args = parser.parse_args()"""

    output = run_mbh_pe(5)
                