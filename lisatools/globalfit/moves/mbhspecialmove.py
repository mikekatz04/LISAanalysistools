import numpy as np
import cupy as cp
from copy import deepcopy
import os

from eryn.moves import RedBlueMove, StretchMove
# from eryn.state import State
from ..state import GFState
from ...sampling.moves.skymodehop import SkyMove
from bbhx.likelihood import NewHeterodynedLikelihood
from tqdm import tqdm
from .globalfitmove import GlobalFitMove
from .addremovemove import ResidualAddOneRemoveOneMove
from ...utils.utility import tukey
from ...sampling.stopping import SearchConvergeStopping
from ...utils.constants import *

from eryn.ensemble import EnsembleSampler
from ...sensitivity import AET1SensitivityMatrix
from ...detector import sangria
from ...datacontainer import DataResidualArray
from ...analysiscontainer import AnalysisContainer


def update_fn(i, last_sample, sampler):
    print("max logl:", last_sample.log_like.max()) 
    last_sample.branches_coords["mbh"][-1] = last_sample.branches_coords["mbh"][0]
    last_sample.log_like[-1] = last_sample.log_like[0]
    last_sample.log_prior[-1] = last_sample.log_prior[0]


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


class MBHSpecialMove(ResidualAddOneRemoveOneMove, GlobalFitMove, RedBlueMove):
    def __init__(self, *args, run_search=False, **kwargs):
        
        RedBlueMove.__init__(self, **kwargs)
        ResidualAddOneRemoveOneMove.__init__(self, *args, **kwargs)
        self.run_search = run_search
        self.finished_search = False
        
    def setup(self, model, state):

        if not self.run_search:
            return 

        acs_all = model.analysis_container_arr 
        while True:
            # TODO: adjust these moves?
            _moves = [(move_i, weight_i) for move_i, weight_i in zip(self.moves, self.move_weights)]
            max_logl_walker = np.argmax(acs_all.likelihood()).item()
            
            data = acs_all.data_shaped[0][max_logl_walker].copy()
            psd = acs_all.psd_shaped[0][max_logl_walker].copy()

            fd = cp.asarray(acs_all.f_arr)
            df = acs_all.df
            dt = acs_all[0].data_res_arr.dt

            ntemps = 10
            nwalkers = 50

            # take inverse FFT 
            At, Et = np.fft.irfft(data[0].get() / dt), np.fft.irfft(data[1].get() / dt)
            full_length = len(At)
            # TODO: do anything with incoming tukey window?
            

            start_params = self.priors["mbh"].rvs(size=(ntemps, nwalkers, 1))
            # ll = wave_gen.get_direct_ll(fd, data, psd, df, *x_in.T, **self.waveform_like_kwargs).real.get()

            _Tobs = At.shape[0] * dt  # 1 month
            Nobs = int(_Tobs / dt)  # len(t)
            Tobs = Nobs * dt

            t1 = np.arange(Nobs) * dt
            At = At[:Nobs]
            Et = Et[:Nobs]

            data = []
            acs_tmp = []
            initial_t_vals = []
            # assert ind_start > 0
            # omits first day and last day
            delta = int(12.0 * 3600.0 / dt)  # int(15 * 24 * 3600.0 / dt)  # 
            # TODO: need to fill in days on edges of windows
            t_start_ind = int((0.25 * YRSID_SI / 12.0 - 24.0 * 3600.0) / dt)  # 0
            if t_start_ind > len(At) - 2:
                breakpoint()
                raise ValueError

            t_lims = np.arange(t_start_ind, (t1.shape[0] - 1), delta) * dt
            # t_lims = np.array([0.0, 3600.0, Tobs - 3600.0, Tobs])
            print(f"Not searching for {t_lims[-1]} to {t1[-1]}")

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
                A_here = At[keep_t] * tukey_here
                E_here = Et[keep_t] * tukey_here

                # plt.plot(X_here)
                # plt.savefig("check0.png")
                # plt.close()
                # fucking dt
            
                # Af, Ef, Tf = (
                #     np.fft.rfft(A_here) * dt,
                #     np.fft.rfft(E_here) * dt,
                #     np.fft.rfft(T_here) * dt,
                # )

                Af, Ef = (
                    np.fft.rfft(A_here) * dt,
                    np.fft.rfft(E_here) * dt,
                )
                
                initial_t_vals.append(start_t)
                end_t_vals.append(end_t)
                if i == 0:
                    length_check = len(Af)
                    fd_short = np.fft.rfftfreq(len(A_here), dt)
                    df_short = fd_short[1] - fd_short[0]
                    Tobs_short = dt * len(t_here)
                else:
                    assert length_check == len(Af)  

                data_res_arr = DataResidualArray([Af, Ef, np.zeros_like(Af)], f_arr=fd_short)
                sens_mat = AET1SensitivityMatrix(fd_short, model=sangria)  # , stochastic_params=(YRSID_SI,))
                sens_mat[2] = 1e10
                analysis = AnalysisContainer(data_res_arr, sens_mat)  # , signal_gen=MBHWrap(wave_gen))
                acs_tmp.append(analysis)
                print(start_t, t_ref_lims[i], end_t, t_ref_lims[i + 1], analysis.inner_product())
            
            fd_short = cp.asarray(fd_short)
            # all_data_cpu = np.asarray(data).transpose(1, 0, 2)
            # data_length = len(fd_short)
            initial_t_vals = np.asarray(initial_t_vals)
            end_t_vals = np.asarray(end_t_vals)

            from lisatools.analysiscontainer import AnalysisContainerArray
            acs = AnalysisContainerArray(acs_tmp, gpus=[cp.cuda.runtime.getDevice()])
            fd_short = cp.asarray(fd_short)
            data_length_short = len(fd_short)
            initial_t_vals = np.asarray(initial_t_vals)
            end_t_vals = np.asarray(end_t_vals)
            
            # check = analysis.calculate_signal_likelihood(*inj_params_in, source_only=True, waveform_kwargs=tmp_kwargs)
            
            from bbhx.likelihood import HeterodynedLikelihood, NewHeterodynedLikelihood, Likelihood
            
            # priors
            from eryn.prior import ProbDistContainer, uniform_dist
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

            from eryn.utils import PeriodicContainer
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
            full_kwargs = self.waveform_like_kwargs.copy()
            full_kwargs["phase_marginalize"] = True
            full_kwargs["length"] = 1024

            like_args = (self.waveform_gen, initial_t_vals, end_t_vals, d_d_vals, t_ref_lims, self.transform_fn, (cp.asarray(fd_short), acs.linear_data_arr[0], 1. / acs.linear_psd_arr[0], df), full_kwargs)
            # # check3 = search_likelihood_wrap(start_points, *like_args)
            
            _fp = "mbh_search_tmp_file.h5"
            fp = "global_fit_output/" + _fp
            if os.path.exists(fp):
                os.remove(fp)
            stop_fn = SearchConvergeStopping(n_iters=1, diff=1.0, verbose=True, start_iteration=0)
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
            final_state = sampler.run_mcmc(start_state, nsteps, thin_by=50, progress=True)
            print("End like:", sampler.get_log_like().max())

            full_kwargs["phase_marginalize"] = True
            coords_post_like = final_state.branches["mbh"].coords[0, :, 0]
            new_ll = search_likelihood_wrap(coords_post_like, *like_args)

            opt_snr = self.waveform_gen.h_h.copy() ** (1/2)
            det_snr = self.waveform_gen.d_h / opt_snr
            phase_change = np.angle(self.waveform_gen.non_marg_d_h)
            full_kwargs["phase_marginalize"] = False
            _coords_check = coords_post_like.copy()
            # adjust phase due to phase marginalizations
            _coords_check[:, 5] = _coords_check[:, 5] + 1./2. * phase_change.get()
            _ll_check = search_likelihood_wrap(_coords_check, *like_args)
            # TODO: make option
            self.snr_det_lim = 20.0
            keep_it = np.any((opt_snr > self.snr_det_lim) & (det_snr > self.snr_det_lim))
            if True:  # keep_it:
                # get current highest leaf in MBH store
                # this will return first False value for leaves
                next_leaf = state.branches["mbh"].inds[0, 0].argmin()
                state.branches['mbh'].inds[:, :, next_leaf] = True
                state.branches['mbh'].coords[:, :, next_leaf] = coords_post_like[:state.branches['mbh'].coords.shape[1]][None, :, :]
                
                # remove the cold chain parameters from their respective residual. 
                AET_remove_params = state.branches['mbh'].coords[0, :, next_leaf].copy()
                AET_remove_params_in = self.transform_fn.both_transforms(AET_remove_params)
                self.waveform_gen.amp_phase_gen.initial_t_val = 0.0
                AET_remove = self.waveform_gen(*AET_remove_params_in.T, t_obs_start=0.0, t_obs_end=full_length * dt / YRSID_SI, compress=True, direct=False, fill=True, freqs=self.xp.asarray(acs_all.f_arr), **self.waveform_gen_kwargs)
                AE_remove = AET_remove[:, :2]
                acs_all.add_signal_to_residual(AE_remove)

            else:
                self.finished_search = True
                return

    def propose(self, model, state):
        __doc__ = ResidualAddOneRemoveOneMove.propose.__doc__
        assert np.all(state.branches["mbh"].nleaves[0,0] == state.branches["mbh"].nleaves)
        if self.finished_search and state.branches["mbh"].nleaves[0,0] == 0:
            print("No MBHs in sampler. Skipping proposal.")
            ntemps, nwalkers = state.branches["mbh"].shape[:2]
            _accepted = np.zeros((ntemps, nwalkers), dtype=int)
            return state, _accepted
        
        return ResidualAddOneRemoveOneMove.propose(self, model, state)

    def setup_likelihood_here(self, coords):
        # TODO: should we try to pick specifically based on max ll for MBHs rather than data as a whole
        start_likelihood = self.acs.likelihood()
        keep_het = start_likelihood.argmax()

        data_index = cp.arange(self.nwalkers, dtype=np.int32)
        noise_index = cp.arange(self.nwalkers, dtype=np.int32)
        het_coords = np.tile(coords[keep_het], (self.nwalkers, 1))

        # self.waveform_like_kwargs = dict(
        #     **self.waveform_like_kwargs,
        #     constants_index=data_index
        # )

        self.like_fn = NewHeterodynedLikelihood(
            self.waveform_gen,
            self.fd,
            self.acs.data_shaped[0],
            self.acs.psd_shaped[0],
            het_coords,
            256,
            data_index=data_index,
            noise_index=noise_index,
            gpu=cp.cuda.runtime.getDevice(),  # self.use_gpu,
        )
        data_index = walker_inds_base.astype(np.int32)
        noise_index = walker_inds_base.astype(np.int32)

        # set d_d term in the likelihood
        self.like_fn.d_d = self.like_fn(
            removal_coords_in, 
            **self.waveform_like_kwargs
        ) + self.acs.likelihood(noise_only=True)[data_index]

        cp.get_default_memory_pool().free_all_blocks()
        
    def compute_like(self, new_points_in, data_index):
        assert data_index is not None
        logl = like_het.get_ll(
            new_points_in, 
            constants_index=data_index,
        )
                    
        return logl

    def get_waveform_here(self, coords):
        cp.get_default_memory_pool().free_all_blocks()
        waveforms = cp.zeros((coords.shape[0], self.acs.nchannels, self.acs.data_length), dtype=complex)
        
        for i in range(coords.shape[0]):
            waveforms[i] = self.waveform_gen(*coords[i], **self.waveform_gen_kwargs)
        
        return waveforms

    def replace_residuals(self, old_state, new_state):
        fd = cp.asarray(self.acs.fd)
        old_contrib = [None, None]
        new_contrib = [None, None]
        for leaf in range(old_state.branches["mbh"].shape[-2]):
            removal_coords = old_state.branches["mbh"].coords[0, :, leaf]
            removal_coords_in = self.transform_fn.both_transforms(removal_coords)
            removal_waveforms = self.waveform_gen(*removal_coords_in.T, fill=True, freqs=fd, **self.mbh_kwargs).transpose(1, 0, 2)
            
            add_coords = new_state.branches["mbh"].coords[0, :, leaf]
            add_coords_in = self.transform_fn.both_transforms(add_coords)
            add_waveforms = self.waveform_gen(*add_coords_in.T, fill=True, freqs=fd, **self.mbh_kwargs).transpose(1, 0, 2)

            if leaf == 0:
                old_contrib[0] = removal_waveforms[0]
                old_contrib[1] = removal_waveforms[1]
                new_contrib[0] = add_waveforms[0]
                new_contrib[1] = add_waveforms[1]
            else:
                old_contrib[0] += removal_waveforms[0]
                old_contrib[1] += removal_waveforms[1]
                new_contrib[0] += add_waveforms[0]
                new_contrib[1] += add_waveforms[1]
            
        self.acs.swap_out_in_base_data(old_contrib, new_contrib)
        cp.get_default_memory_pool().free_all_blocks()

