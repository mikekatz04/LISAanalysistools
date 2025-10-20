import h5py
import numpy as np
import shutil
import matplotlib.pyplot as plt
try:
    import cupy as cp
except (ModuleNotFoundError, ImportError) as e:
    import numpy as cp

from eryn.moves.tempering import TemperatureControl, make_ladder

from lisatools.detector import EqualArmlengthOrbits
from eryn.state import BranchSupplemental

from bbhx.utils.transform import *

from lisatools.utils.utility import AET

from eryn.prior import uniform_dist
from eryn.utils import TransformContainer
from eryn.prior import ProbDistContainer
from eryn.utils import PeriodicContainer
    
from eryn.moves import StretchMove
from lisatools.sampling.moves.skymodehop import SkyMove
from global_fit_input.global_fit_settings import dtrend
from lisatools.utils.utility import tukey

from lisatools.globalfit.mbhsearch import search_likelihood_wrap

from eryn.ensemble import EnsembleSampler
from lisatools.sensitivity import AET1SensitivityMatrix
from lisatools.detector import sangria
from lisatools.datacontainer import DataResidualArray
from lisatools.analysiscontainer import AnalysisContainer


def mbh_dist_trans(x):
    return x * PC_SI * 1e9  # Gpc


def ll_wrap(x, ll_het, transform_fn):
    x_in = transform_fn.both_transforms(x)

    ll = ll_het.get_ll(x_in.T, phase_marginalize=False)
    return ll



def search_likelihood_wrap(x, wave_gen, initial_t_vals, end_t_vals, d_d_vals, t_ref_lims, transform_fn, like_args, mbh_kwargs):
    x_in = transform_fn.both_transforms(x)

    data_index = noise_index = (np.searchsorted(t_ref_lims, x[:, -1], side="right") - 1).astype(np.int32)
    # wave_gen.amp
    
    wave_gen.amp_phase_gen.initial_t_val = initial_t_vals[data_index][:, None]
    t_obs_start = initial_t_vals[data_index] / YRSID_SI
    t_obs_end = end_t_vals[data_index] / YRSID_SI
    wave_gen.d_d = cp.asarray(d_d_vals[data_index])
    fd, all_data, psd, df = like_args
    ll = wave_gen.get_direct_ll(fd, all_data, psd, df, *x_in.T, noise_index=noise_index, data_index=data_index, t_obs_start=t_obs_start, t_obs_end=t_obs_end, **mbh_kwargs).real.get()
    
    return ll


class MBHWrap:
    def __init__(self, signal_gen):
        self.signal_gen = signal_gen

    def __call__(self, *args, **kwargs):
        return self.signal_gen(*args, **kwargs)[0]

def update_fn(i, last_sample, sampler):
    print("max logl:", last_sample.log_like.max()) 
    last_sample.branches_coords["mbh"][-1] = last_sample.branches_coords["mbh"][0]
    last_sample.log_like[-1] = last_sample.log_like[0]
    last_sample.log_prior[-1] = last_sample.log_prior[0]

if __name__ == "__main__":

    gpus = [0]
    cp.cuda.runtime.setDevice(gpus[0])
     # transforms from pe to waveform generation
    parameter_transforms_mbh = {
        0: np.exp,
        4: mbh_dist_trans,
        7: np.arccos,
        9: np.arcsin,
        (0, 1): mT_q,
        (11, 8, 9, 10): LISA_to_SSB,
    }

    # for transforms
    fill_dict_mbh = {
        "ndim_full": 12,
        "fill_values": np.array([0.0]),
        "fill_inds": np.array([6]),
    }

    transform_fn_mbh = TransformContainer(
        parameter_transforms=parameter_transforms_mbh,
        fill_dict=fill_dict_mbh,
    )
    ldc_source_file = "LDC2_sangria_training_v2.h5"
    with h5py.File(ldc_source_file, "r") as f:
        mbh_params = f["sky"]["mbhb"]["cat"][:]
        # tXYZ = f["obs"]["tdi"][:]

    ind_inj = 0
    lnmT = (np.log(mbh_params[ind_inj]["Mass1"] + mbh_params[ind_inj]["Mass2"])).item()
    q = (mbh_params[ind_inj]["Mass2"] / mbh_params[ind_inj]["Mass1"]).item()
    a1 = 0.823982938423 # mbh_params[ind_inj]["Spin1"].item()
    a2 = 0.768234287  # mbh_params[ind_inj]["Spin2"].item()
    dist = mbh_params[ind_inj]["Distance"].item() / 1e3
    phi_ref = mbh_params[ind_inj]["PhaseAtCoalescence"].item()
    cos_inc = np.cos(1.098234090)
    lam_ssb = mbh_params[ind_inj]["EclipticLongitude"].item()
    sin_beta_ssb = np.sin(mbh_params[ind_inj]["EclipticLatitude"]).item()
    psi_ssb = 0.7293487209
    t_ref_ssb = mbh_params[ind_inj]["CoalescenceTime"].item()
    t_ref_L, lam_L, beta_L, psi_L = SSB_to_LISA(t_ref_ssb, lam_ssb, np.arcsin(sin_beta_ssb), psi_ssb)
    sin_beta_L = np.sin(beta_L)
    
    psi_L %= (np.pi)
    lam_L %= (2 * np.pi)
    phi_ref %= (2 * np.pi)

    # injection_params = np.array([lnmT, q, a1, a2, dist, phi_ref, cos_inc, lam_L, sin_beta_L, psi_L, t_ref_L])
    # inj_params_in = transform_fn_mbh.both_transforms(injection_params[None, :])[0]

    with h5py.File(ldc_source_file, "r") as f:
        tXYZ = f["obs"]["tdi"][:]

        # remove sources
        # for source in ["mbhb"]:  # , "dgb", "igb"]:  # "vgb" ,
        #     change_arr = f["sky"][source]["tdi"][:]
        #     for change in ["X", "Y", "Z"]:
        #         tXYZ[change] -= change_arr[change]

        # tXYZ = f["sky"]["mbhb"]["tdi"][:]
        # tXYZ["X"] += f["sky"]["dgb"]["tdi"][:]["X"]
        # tXYZ["Y"] += f["sky"]["dgb"]["tdi"][:]["Y"]
        # tXYZ["Y"] += f["sky"]["dgb"]["tdi"][:]["Z"]

    t1, X1, Y1, Z1 = (
        tXYZ["t"].squeeze(),
        tXYZ["X"].squeeze(),
        tXYZ["Y"].squeeze(),
        tXYZ["Z"].squeeze(),
    )

    dt = t1[1] - t1[0]

    _Tobs = 3.0 * YRSID_SI / 12.0  # 1 month
    Nobs = int(_Tobs / dt)  # len(t)
    Tobs = Nobs * dt

    t1 = t1[:Nobs]
    X1 = X1[:Nobs]
    Y1 = Y1[:Nobs]
    Z1 = Z1[:Nobs]
    
    # TODO: @nikos what do you think about the window needed here. For this case at 1 year, I do not think it matters. But for other stuff.
    # the time domain waveforms like emris right now will apply this as well
    X1 = dtrend(t1, X1.copy())
    Y1 = dtrend(t1, Y1.copy())
    Z1 = dtrend(t1, Z1.copy())

    A, E, T = AET(X1, Y1, Z1)

    ndays = int(t1.shape[0] * dt / (24.0 * 3600.0))
    delta = int(12.0 * 3600.0 / dt)  # int(15 * 24 * 3600.0 / dt)  # 
    # TODO: need to fill in days on edges of windows
    t_start_ind = int((1.5 * YRSID_SI / 12.0 - 24.0 * 3600.0) / dt)  # 0
    t_lims = np.arange(t_start_ind, (t1.shape[0] - 1), delta) * dt
    # t_lims = np.array([0.0, 3600.0, Tobs - 3600.0, Tobs])
    print(f"Not searching for {t_lims[-1]} to {t1[-1]}")

    # take out found sources
    # fp_old = f"check_new_searching_rebirth_4th_month.h5"
    fp = f"check_2_searching_rebirth_th_month.h5"
    from eryn.backends import HDFBackend
    
    from bbhx.waveformbuild import BBHWaveformFD

    gpu_orbits = EqualArmlengthOrbits(use_gpu=True)
    
    # waveform kwargs
    initialize_kwargs_mbh = dict(
        amp_phase_kwargs=dict(run_phenomd=True),
        response_kwargs=dict(TDItag="AET", orbits=gpu_orbits),
        use_gpu=True
    )

    wave_gen = BBHWaveformFD(**initialize_kwargs_mbh)

    # for MBH waveform class initialization
    waveform_kwargs_mbh = dict(
        modes=[(2,2)],
        length=1024,
    )

    _fd = cp.asarray(np.fft.rfftfreq(Nobs, dt))
    tmp_kwargs = dict(freqs=_fd, fill=True, direct=True, **waveform_kwargs_mbh)

    reader = HDFBackend(fp_old)
    best = np.where(reader.get_log_like() == reader.get_log_like().max())
    mbh_params_best = reader.get_chain()["mbh"][best].squeeze()
    mbh_params_best[5] = 4.961183137438699
    mbh_params_best_in = transform_fn_mbh.both_transforms(mbh_params_best[None, :])
    
    AET_found = wave_gen(*mbh_params_best_in.T, t_obs_start=0.0, t_obs_end=Tobs / YRSID_SI, **tmp_kwargs)
    _Af, _Ef, _Tf = np.fft.rfft(A) * dt, np.fft.rfft(E) * dt, np.fft.rfft(T) * dt
    _Af -= AET_found[0, 0].get()
    _Ef -= AET_found[0, 1].get()
    _Tf -= AET_found[0, 2].get()
    A_old = A.copy()
    A, E, T = np.fft.irfft(_Af / dt), np.fft.irfft(_Ef / dt), np.fft.irfft(_Tf / dt)

    data = []
    acs_tmp = []
    initial_t_vals = []
    # assert ind_start > 0
    # omits first day and last day
    t_ref_lims = t_lims[1:-1]
    num_t_ref_bins = len(t_ref_lims) - 1
    
    end_t_vals = []
    for i, t_i in enumerate(range(0, t_lims.shape[0] - 3)):
        start_t = t_lims[t_i]
        end_t = t_lims[t_i + 3]
        # start_t = 0.0
        keep_t = (t1 >= start_t) & (t1 < end_t)
        print(i, keep_t.sum().item())
        if keep_t.sum().item() == 0:
            continue
        tukey_alpha = 0.03
        tukey_here = tukey(keep_t.sum().item(), tukey_alpha)
        t_here = t1[keep_t]
        A_here = A[keep_t] * tukey_here
        E_here = E[keep_t] * tukey_here
        T_here = T[keep_t] * tukey_here

        # plt.plot(X_here)
        # plt.savefig("check0.png")
        # plt.close()
        # fucking dt
       
        
        Af, Ef, Tf = (
            np.fft.rfft(A_here) * dt,
            np.fft.rfft(E_here) * dt,
            np.fft.rfft(T_here) * dt,
        )
        
        initial_t_vals.append(start_t)
        end_t_vals.append(end_t)
        if i == 0:
            length_check = len(Af)
            fd = np.fft.rfftfreq(len(A_here), dt)
            df = fd[1] - fd[0]
            Tobs = dt * len(t_here)
        else:
            assert length_check == len(Af)  

        data_res_arr = DataResidualArray([Af, Ef, Tf], f_arr=fd)
        sens_mat = AET1SensitivityMatrix(fd, model=sangria)  # , stochastic_params=(YRSID_SI,))
        analysis = AnalysisContainer(data_res_arr, sens_mat)  # , signal_gen=MBHWrap(wave_gen))
        acs_tmp.append(analysis)
        print(start_t, t_ref_lims[i], end_t, t_ref_lims[i + 1], analysis.inner_product())
    
    fd = cp.asarray(fd)
    # all_data_cpu = np.asarray(data).transpose(1, 0, 2)
    data_length = len(fd)
    initial_t_vals = np.asarray(initial_t_vals)
    end_t_vals = np.asarray(end_t_vals)

    from lisatools.analysiscontainer import AnalysisContainerArray
    acs = AnalysisContainerArray(acs_tmp, gpus=gpus)
    fd = cp.asarray(fd)
    data_length = len(fd)
    initial_t_vals = np.asarray(initial_t_vals)
    end_t_vals = np.asarray(end_t_vals)
    
    # check = analysis.calculate_signal_likelihood(*inj_params_in, source_only=True, waveform_kwargs=tmp_kwargs)
    
    from bbhx.likelihood import HeterodynedLikelihood, NewHeterodynedLikelihood, Likelihood
    
    # priors
    from eryn.prior import ProbDistContainer
    priors_mbh = {"mbh": ProbDistContainer({
        0: uniform_dist(np.log(1e4), np.log(1e8)),  # 
        1: uniform_dist(0.01, 0.999999999),
        2: uniform_dist(-0.99999999, +0.99999999),
        3: uniform_dist(-0.99999999, +0.99999999),
        4: uniform_dist(0.01, 1000.0),
        5: uniform_dist(0.0, 2 * np.pi),
        6: uniform_dist(-1.0 + 1e-6, 1.0 - 1e-6),
        7: uniform_dist(0.0, 2 * np.pi),
        8: uniform_dist(-1.0 + 1e-6, 1.0 - 1e-6),
        9: uniform_dist(0.0, np.pi),
        10: uniform_dist(t_ref_lims[0], t_ref_lims[-1]), #0.0, t_ref_L + 48 * 3600.0),
    })}

    # ll_het = NewHeterodynedLikelihood(
    #     wave_gen,
    #     fd,
    #     AET[None, :],
    #     sens_mat[:][None, :],
    #     inj_params_in[None, :],
    #     256,
    #     # data_index=data_index,
    #     # noise_index=noise_index,
    #     gpu=cp.cuda.runtime.getDevice(),  # self.use_gpu,
    # )

    # sampler treats periodic variables by wrapping them properly
    periodic_mbh = PeriodicContainer({
        "mbh": {5: 2 * np.pi, 7: 2 * np.pi, 9: np.pi}
    })

    inner_moves = [
        (SkyMove(which="both"), 0.02),
        (SkyMove(which="long"), 0.05),
        (SkyMove(which="lat"), 0.05),
        (StretchMove(), 0.88)
    ]

    nwalkers = 100
    ntemps = 12
    temp_kwargs = dict(ntemps=ntemps, Tmax=np.inf)

    # psd_cpu = np.asarray([np.tile(A_psd, (all_data_cpu.shape[1], 1)), np.tile(E_psd, (all_data_cpu.shape[1], 1)), np.tile(np.full_like(A_psd, 1e10), (all_data_cpu.shape[1], 1))])

    d_d_vals = np.zeros(len(acs))  # 4 * df * np.sum(np.asarray([(all_data_cpu[i].conj() * all_data_cpu[i]) / psd_cpu[i] for i in range(all_data_cpu.shape[0])]), axis=(0, 2))

    # d_d_vals[:] = acs.inner_product()
    
    full_kwargs = waveform_kwargs_mbh.copy()
    full_kwargs["phase_marginalize"] = True
    full_kwargs["length"] = 1024

    like_args = (wave_gen, initial_t_vals, end_t_vals, d_d_vals, t_ref_lims, transform_fn_mbh, (cp.asarray(fd), acs.linear_data_arr[0], 1. / acs.linear_psd_arr[0], df), full_kwargs)
    # # check3 = search_likelihood_wrap(start_points, *like_args)
    
    from lisatools.sampling.stopping import SearchConvergeStopping

    stop_fn = SearchConvergeStopping(n_iters=30, diff=1.0, verbose=True, start_iteration=0)
    sampler = EnsembleSampler(
        nwalkers,
        {"mbh": 11}, 
        search_likelihood_wrap,  ## ll_wrap, #
        priors_mbh,
        tempering_kwargs=temp_kwargs,
        args=like_args,  # (ll_het, transform_fn_mbh), # 
        vectorize=True,
        periodic=periodic_mbh, 
        backend=fp,
        moves=inner_moves,
        branch_names=["mbh"],
        update_fn=update_fn,
        update_iterations=200, 
        stopping_fn=stop_fn,
        stopping_iterations=1
    )

    start_points = priors_mbh["mbh"].rvs(size=(ntemps * nwalkers,))
    # start_like4 = search_likelihood_wrap(start_points, *like_args)
    # wave_gen.amp_phase_gen.initial_t_val = 0.0
    # check2 = analysis.calculate_signal_likelihood(*start_points[0], phase_maximize=True, transform_fn=transform_fn_mbh, source_only=True, waveform_kwargs=tmp_kwargs, sum_instead_of_trapz=True)
    # # check_inner = analysis.calculate_signal_inner_product(*start_points[0], transform_fn=transform_fn_mbh, source_only=True, waveform_kwargs=tmp_kwargs, sum_instead_of_trapz=False)
    # check_snr = analysis.calculate_signal_snr(*start_points[0], transform_fn=transform_fn_mbh, source_only=True, waveform_kwargs=tmp_kwargs, sum_instead_of_trapz=True)
    # breakpoint()
    # from eryn.backends import HDFBackend
    # reader = HDFBackend("check_bbhx_search_het.h5")
    # samples = reader.get_chain(temp_index=0)["mbh"][5000:, :, 0].reshape(-1, 11)
    # # fig = corner.corner(chain, plot_datapoints=False, plot_density=False, levels=(1 - np.exp(-0.5 * np.array([1, 2, 3]) ** 2)))
    # # fig.savefig("check1.png")
    
    # checkit = ll_wrap(samples[-10:], ll_het, transform_fn_mbh)
    # breakpoint()
    start_points = start_points.reshape(ntemps, nwalkers, 1, 11)
    # start_points = priors_mbh["mbh"].rvs(size=(ntemps, nwalkers, 1))
    from eryn.state import State
    start_state = State({"mbh": start_points})

    nsteps = 500
    print("start like", acs.likelihood(source_only=True).max())
    # check_params = np.load("check_params.npy")[None, :]
    # start_like4 = search_likelihood_wrap(mbh_params_best[None, :], *like_args)
    # breakpoint()
    sampler.run_mcmc(start_state, nsteps, thin_by=50, progress=True)
    
    print("End like:", sampler.get_log_like().max())
    breakpoint()
    # chain = sampler.get_chain()["mbh"][:, 0].reshape(-1, 11)
    # import corner
    # fig = corner.corner(chain, plot_datapoints=False, plot_density=False, levels=(1 - np.exp(-0.5 * np.array([1, 2, 3]) ** 2)))
    # fig.savefig(fp[:-3] + ".png")
    # breakpoint()
    
    
