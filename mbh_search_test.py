import numpy as np
from eryn.state import State
from eryn.ensemble import EnsembleSampler
from eryn.prior import ProbDistContainer, uniform_dist
from lisatools.glitch import tdi_glitch_XYZ1
import corner
from lisatools.utils.utility import AET

from bbhx.waveformbuild import BBHWaveformFD
from bbhx.waveforms.phenomhm import PhenomHMAmpPhase
from bbhx.response.fastfdresponse import LISATDIResponse
from bbhx.likelihood import Likelihood as MBHLikelihood
from bbhx.likelihood import HeterodynedLikelihood
from bbhx.utils.constants import *
from bbhx.utils.transform import *

from eryn.moves import StretchMove
from lisatools.sampling.likelihood import Likelihood
from lisatools.sampling.moves.skymodehop import SkyMove
from lisatools.sampling.stopping import SearchConvergeStopping

from lisatools.sensitivity import get_sensitivity
from lisatools.sampling.utility import HeterodynedUpdate

from eryn.utils import TransformContainer

from full_band_global_fit_settings import *
fd1 = fd.copy()
np.random.seed(111222)

try:
    import cupy as xp
    # set GPU device
    xp.cuda.runtime.setDevice(7)
    gpu_available = True

except (ImportError, ModuleNotFoundError) as e:
    import numpy as xp
    gpu_available = False

# whether you are using 
use_gpu = True

if use_gpu is False:
    xp = np

if use_gpu and not gpu_available:
    raise ValueError("Requesting gpu with no GPU available or cupy issue.")

stop1 = SearchConvergeStopping(n_iters=50, diff=0.01, verbose=True)
def stop(iter, sample, sampler):
    print("LL MAX:", sampler.get_log_like().max())
    temp = stop1(iter, sample, sampler)
    return temp

# function call
def run_mbh_search():

    t_lims = np.linspace(0.0, YEAR, 26)

    for t_i in range(len(t_lims) - 1)[3:]:
        start_t = t_lims[t_i]
        end_t = t_lims[t_i + 1]
        keep_t = (t >= start_t) & (t < end_t)
        t_here = t[keep_t]
        X_here = X[keep_t]
        Y_here = Y[keep_t]
        Z_here = Z[keep_t]

        Tobs = dt * len(t_here)

        # fucking dt 
        Xf, Yf, Zf = (np.fft.rfft(X_here) * dt, np.fft.rfft(Y_here) * dt, np.fft.rfft(Z_here) * dt)
        Af, Ef, Tf = AET(Xf, Yf, Zf)

        A_inj, E_inj = Af, Ef
        fd = np.fft.rfftfreq(len(X_here), dt)
        df = fd[1] - fd[0]
        data_length = len(fd)

        fp = f"test_mbh_search_{start_t:.4e}_{end_t:.4e}.h5"

        ntemps = 10
        nwalkers = 100

        mbh_kwargs = {
            "modes": [(2,2)],
            "length": 1024,
            "t_obs_start": start_t,
            "t_obs_end": end_t,
            "shift_t_limits": True
        }

        # wave generating class
        wave_gen = BBHWaveformFD(
            amp_phase_kwargs=dict(run_phenomd=True, initial_t_val=start_t),
            response_kwargs=dict(TDItag="AET"),
            use_gpu=use_gpu
        )

        # for transforms
        fill_dict = {
            "ndim_full": 12,
            "fill_values": np.array([0.0]),
            "fill_inds": np.array([6]),
        }

        try:
            del priors
        except NameError:
            pass

        # priors
        priors = {
            "mbh": ProbDistContainer(
                {
                    0: uniform_dist(np.log(5e4), np.log(5e7)),
                    1: uniform_dist(0.05, 0.99),
                    2: uniform_dist(-0.99, 0.99),
                    3: uniform_dist(-0.99, 0.99),
                    4: uniform_dist(0.01, 1000.0),
                    5: uniform_dist(0.0, 2 * np.pi),
                    6: uniform_dist(-1.0 + 1e-6, 1.0 - 1e-6),
                    7: uniform_dist(0.0, 2 * np.pi),
                    8: uniform_dist(-1.0 + 1e-6, 1.0 - 1e-6),
                    9: uniform_dist(0.0, np.pi),
                    10: uniform_dist(start_t + 1000.0, end_t + 4 * 7 * 24 * 3600), # 4 weeks after end of window
                }
            ) 
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

        psd_reader = HDFBackend(fp_psd_search)
        best_psd_ind = psd_reader.get_log_like().flatten().argmax()
        best_psd_params = psd_reader.get_chain()["psd"].reshape(-1, 4)[best_psd_ind]

        A_psd = get_sensitivity(fd, sens_fn="noisepsd_AE", model=best_psd_params[:2])
        E_psd = get_sensitivity(fd, sens_fn="noisepsd_AE", model=best_psd_params[2:])

        A_psd[0] = A_psd[1]
        E_psd[0] = E_psd[1]

        """# A_psd_in[:] = np.asarray(psd)
        # E_psd_in[:] = np.asarray(psd)
        fp_psd = "last_psd_cold_chain_info.npy"
        psds = np.load(fp_psd)
        psds[:, :, 0] = psds[:, :, 1]
        
        A_psd_in = psds[0, 0]
        E_psd_in = psds[0, 1]

        psd = xp.asarray([A_psd_in, E_psd_in, np.full_like(A_psd_in, 1e10)])

        fp_gb = "last_gb_cold_chain_info.npy"
        data_in = np.load(fp_gb)

        A_going_in = A_inj.copy()
        E_going_in = E_inj.copy()

        A_going_in -= data_in[0, 0]
        E_going_in -= data_in[0, 1]

        fd = xp.asarray(fd1)
        if "mbh_found_points.npy" in os.listdir():
            found_mbh_points = np.load("mbh_found_points.npy")
            for mbh_injection_params in found_mbh_points:
                injection_in = transform_fn.both_transforms(mbh_injection_params[None, :], return_transpose=True)
            
                # get XYZ
                data_channels_AET = wave_gen(*injection_in, freqs=fd,
                    modes=[(2,2)], direct=False, fill=True, squeeze=True, length=1024
                )[0]

                A_going_in -= data_channels_AET[0].get()
                E_going_in -= data_channels_AET[1].get()"""

        
        try:
            del like_mbh
            del like
            del sampler
            del data_channels
            del psd
        except NameError:
            pass

        data_channels = xp.asarray([A_inj, E_inj, np.zeros_like(E_inj)])

        psd = xp.asarray([A_psd, E_psd, np.full_like(A_psd, 1e10)])
        
        xp.get_default_memory_pool().free_all_blocks()
        # initialize Likelihood
        like_mbh = MBHLikelihood(
            wave_gen, xp.asarray(fd), data_channels, psd, use_gpu=use_gpu
        )

        # this is a parent likelihood class that manages the parameter transforms
        like = Likelihood(
            like_mbh,
            3,
            dt=None,
            df=None,
            f_arr=fd,
            parameter_transforms={"mbh": transform_fn},
            use_gpu=use_gpu,
            vectorized=True,
            transpose_params=True,
            subset=100,
        )

        ndim = 11

        if fp not in os.listdir():

            # generate starting points
            start_params = priors["mbh"].rvs(size=(nwalkers * ntemps,))

            # get_ll and not __call__ to work with lisatools
            start_like = like(start_params, **mbh_kwargs)
            start_prior = priors["mbh"].logpdf(start_params)

            # start state
            start_state = State(
                {"mbh": start_params.reshape(ntemps, nwalkers, 1, ndim)}, 
                log_like=start_like.reshape(ntemps, nwalkers), 
                log_prior=start_prior.reshape(ntemps, nwalkers)
            )

        else:
            raise NotImplementedError
            reader = HDFBackend(fp)
            start_state = reader.get_last_sample()

        # MCMC moves (move, percentage of draws)
        moves = [
            (SkyMove(which="both"), 0.02),
            (SkyMove(which="long"), 0.01),
            (SkyMove(which="lat"), 0.01),
            (StretchMove(), 0.96)
        ]

        # prepare sampler
        sampler = EnsembleSampler(
            nwalkers,
            [ndim],  # assumes ndim_max
            like,
            priors,
            tempering_kwargs={"ntemps": ntemps, "Tmax": np.inf},
            moves=moves,
            kwargs=mbh_kwargs,
            backend=fp,
            vectorize=True,
            periodic=periodic,  # TODO: add periodic to proposals
            stopping_fn=stop,
            stopping_iterations=1,
            branch_names=["mbh"],
        )

        # TODO: check about using injection as reference when the glitch is added
        # may need to add the heterodyning updater

        nsteps = 10000
        out = sampler.run_mcmc(start_state, nsteps, progress=True, thin_by=50, burn=0)
        breakpoint()

        mbh_injection_params = out.branches_coords["mbh"][0, np.argmax(out.log_like[0]), 0]

        # generate starting points
        factor = 1e-5
        cov = np.ones(ndim) * 1e-3
        cov[0] = 1e-5
        cov[-1] = 1e-5

        start_like = np.zeros((nwalkers * ntemps))
        
        iter_check = 0
        max_iter = 1000
        while np.std(start_like) < 5.0:
            
            logp = np.full_like(start_like, -np.inf)
            tmp = np.zeros((ntemps * nwalkers, ndim))
            fix = np.ones((ntemps * nwalkers), dtype=bool)
            while np.any(fix):
                tmp[fix] = (mbh_injection_params[None, :] * (1. + factor * cov * np.random.randn(nwalkers * ntemps, ndim)))[fix]

                tmp[:, 5] = tmp[:, 5] % (2 * np.pi)
                tmp[:, 7] = tmp[:, 7] % (2 * np.pi)
                tmp[:, 9] = tmp[:, 9] % (1 * np.pi)
                logp = priors["mbh"].logpdf(tmp)

                fix = np.isinf(logp)
                if np.all(fix):
                    breakpoint()

            start_like = like(tmp, **mbh_kwargs)

            iter_check += 1
            factor *= 1.5

            print(np.std(start_like))

            if iter_check > max_iter:
                raise ValueError("Unable to find starting parameters.")

        start_params = tmp
        start_like = like(start_params, **mbh_kwargs)
        start_prior = priors["mbh"].logpdf(start_params)

        # start state
        start_state = State(
            {"mbh": start_params.reshape(ntemps, nwalkers, 1, ndim)}, 
            log_like=start_like.reshape(ntemps, nwalkers), 
            log_prior=start_prior.reshape(ntemps, nwalkers)
        )

        nsteps = 10000
        print("start second part of run")
        out = sampler.run_mcmc(start_state, nsteps, progress=True, thin_by=50, burn=0)

        mbh_best = out.branches_coords["mbh"][0, np.argmax(out.log_like[0]), 0]

        print("found:", mbh_best)
        if "mbh_found_points.npy" in os.listdir():
            found_mbh_points = np.load("mbh_found_points.npy")
            found_mbh_points = np.concatenate([found_mbh_points, np.array([mbh_best])], axis=0)

        else:
            found_mbh_points = np.array([mbh_best])

        np.save("mbh_found_points", found_mbh_points)
        os.remove(fp)

        tmp_check = np.load("mbh_found_points.npy")
        if tmp_check.shape[0] >= 15:
            break

    return

if __name__ == "__main__":
    # set parameters
    
    

    #print("start", fp)
    run_mbh_search()
    #print("end", fp)
        # frequencies to interpolate to
        
        