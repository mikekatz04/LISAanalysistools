from full_band_global_fit_settings import *
import argparse 
import pickle
from eryn.moves import StretchMove
from lisatools.sampling.stopping import SearchConvergeStopping
from cupy.cuda.runtime import setDevice
import time

from gbgpu.utils.utility import get_N

import warnings
warnings.filterwarnings("ignore")

np.random.seed(82398)

def run_single_band_search(process_i, gpu_i, sub_band_i, nwalkers_prep, ntemps_prep, sub_band_lower_f0, sub_band_upper_f0, generating_priors, Tobs, oversample, waveform_kwargs, data_minus_templates):

    setDevice(gpu_i)
    
    # nwalkers_prep = 500
    num_total = int(1e7)
    num_per = int(1e7)
    num_rounds = num_total // num_per

    data_minus_templates = xp.asarray(np.load(current_residuals_file_iterative_search))
    # print(f"begin - process: {process_i}, sub band: {sub_band_i}")

    like_prep = Likelihood(gb, 2, f_arr=fd, parameter_transforms={"gb": transform_fn}, fill_data_noise=True, vectorized=True, transpose_params=False, use_gpu=True)
    # print(f"like - process: {process_i}, sub band: {sub_band_i}")
    
    like_prep.inject_signal(
        data_stream=list(data_minus_templates.get()),
        noise_fn=get_sensitivity,
        noise_args=[[]],
        noise_kwargs=[{"sens_fn": "noisepsd_AE", "model": "sangria"} for _ in range(like_prep.num_channels)],
        add_noise=False,
    )
    # print(f"inject - process: {process_i}, sub band: {sub_band_i}")
    
    noise_in = like_prep.psd
    noise_in[0][0] = 1e100
    noise_in[1][0] = 1e100
    data_temp = like_prep.injection_channels

    middle_f0 = (sub_band_lower_f0 + sub_band_upper_f0) / 2.
    new_dist = uniform_dist(sub_band_lower_f0 * 1e3, sub_band_upper_f0 * 1e3)
    generating_priors["gb"].priors_in[1] = new_dist
    generating_priors["gb"].priors[1][1] = new_dist
   
    fdot_lims = [get_fdot(lim, Mc=m_chirp_lims[i]) for i, lim in enumerate([sub_band_lower_f0, sub_band_upper_f0])]

    new_dist_2 = uniform_dist(*fdot_lims)
    generating_priors["gb"].priors_in[2] = new_dist_2
    generating_priors["gb"].priors[2][1] = new_dist_2

    N_sub = get_N(1e-23, middle_f0, Tobs, oversample=oversample).item()
    waveform_kwargs_sub = waveform_kwargs.copy()
    waveform_kwargs_sub["N"] = N_sub
    
    gb.d_d = df * 4 * xp.sum(xp.asarray(data_temp).conj() * xp.asarray(data_temp) / xp.asarray(noise_in)[None, :])

    # print(f"others - process: {process_i}, sub band: {sub_band_i}")
    
    import time
    st = time.perf_counter()
    out_ll = []
    out_snr = []
    out_params = []
    # print("start gen")
    for i in range(num_rounds):
        params = generating_priors["gb"].rvs(size=(num_per, 1)).reshape(num_per, -1)
        
        params_in = transform_fn.both_transforms(params, return_transpose=True)
        # print(f"generated - process: {process_i}, sub band: {sub_band_i}")
    
        phase_maximized_ll = gb.get_ll(
            params_in.T,
            data_temp,
            noise_in,
            phase_marginalize=True,
            **waveform_kwargs_sub,
        )

        # print(f"max ll - process: {process_i}, sub band: {sub_band_i}")
    
        phase_maximized_snr = (xp.abs(gb.d_h) / xp.sqrt(gb.h_h.real)).real.copy()
        phase_change = np.angle(gb.non_marg_d_h)

        try:
            phase_maximized_snr = phase_maximized_snr.get()
            phase_change = phase_change.get()

        except AttributeError:
            pass

        params[:, 3] -= phase_change
        params[:, 3] %= (2 * np.pi)

        inds_keep = ~np.any(np.isnan(params), axis=-1)
        out_ll.append(phase_maximized_ll[inds_keep])  # temp_ll[inds_keep]
        out_snr.append(phase_maximized_snr[inds_keep])
        out_params.append(params[inds_keep])

            
        #if (i + 1) % 100:
        #    # print(i + 1, num_rounds)

    # print(f"sort - process: {process_i}, sub band: {sub_band_i}")
    

    out_ll = np.concatenate(out_ll)
    out_snr = np.concatenate(out_snr)
    out_params = np.concatenate(out_params, axis=0)

    inds_sort_snr = np.argsort(out_ll)[::-1]
    out_ll = out_ll[inds_sort_snr]
    out_snr = out_snr[inds_sort_snr]
    out_params = out_params[inds_sort_snr]

    # TODO: this might need to be adjusted when adding noise
    if out_snr[0] < 1.0 or out_params[0, 0] < 1.0:
        return dict(
        sub_band_i=sub_band_i,
        new_coords=out_params[0],
        det_snr=0.0,
        opt_snr=0.0,
        starting_coords=None,
        starting_ll_from_sampler=None,
    )

    et = time.perf_counter()
    # print("end gen", et - st)

    factor = 1e-5
    cov = np.ones(8) * 1e-3
    cov[1] = 1e-7
    start_like = np.zeros((nwalkers_prep * ntemps_prep,))
    iter_check = 0
    run_flag = True
    while np.std(start_like) < 5.0 and run_flag:
        
        logp = np.full_like(start_like, -np.inf)
        tmp = np.zeros((ntemps_prep * nwalkers_prep, ndim))
        fix = np.ones((ntemps_prep * nwalkers_prep), dtype=bool)
        tried_all_fix = False
        while np.any(fix):
            tmp[fix] = (out_params[0][None, :] * (1. + factor * cov * np.random.randn(nwalkers_prep * ntemps_prep, 8)))[fix]

            tmp[:, 3] = tmp[:, 3] % (2 * np.pi)
            tmp[:, 5] = tmp[:, 5] % (np.pi)
            tmp[:, 6] = tmp[:, 6] % (2 * np.pi)
            logp = priors["gb_fixed"].logpdf(tmp)
            fix = np.isinf(logp)
            if np.all(fix):
                if tried_all_fix is False:
                    tried_all_fix = True
                    continue
                else:
                    print("BAD PRIOR", sub_band_i)
                    return dict(
                        sub_band_i=sub_band_i,
                        new_coords=out_params[0],
                        det_snr=0.0,
                        opt_snr=0.0,
                        starting_coords=None,
                        starting_ll_from_sampler=None,
                    )

        tmp_in = transform_fn.both_transforms(tmp, return_transpose=True)

        start_like = gb.get_ll(
            tmp_in.T,
            data_temp,
            noise_in,
            phase_marginalize=True,
            **waveform_kwargs_sub,
        )  # - np.sum([np.log(psd), np.log(psd)])
        iter_check += 1

        #print(np.std(start_like))
        factor *= 1.5

    start_coords = tmp.reshape(ntemps_prep, nwalkers_prep, 1, -1).copy()
    
    start_ll = start_like.reshape(ntemps_prep, nwalkers_prep) # out_ll[:ntemps_prep * nwalkers_prep].reshape(ntemps_prep, nwalkers_prep)
    """
    start_coords = out_params[:ntemps_prep * nwalkers_prep].reshape(ntemps_prep, nwalkers_prep, 1, -1).copy()
    
    start_ll = out_ll[:ntemps_prep * nwalkers_prep].reshape(ntemps_prep, nwalkers_prep)
    """

    start_snr = out_snr[:ntemps_prep * nwalkers_prep].reshape(ntemps_prep, nwalkers_prep)

    prep_state = State({"gb": start_coords}, log_prob=start_ll)
    
    verbose = False  # True if process_i == 2 else False
    stop_converge = SearchConvergeStopping(n_iters=30, diff=0.1, verbose=verbose)
    prep_moves = [StretchMove()]
    # print(f"before sampler - process: {process_i}, sub band: {sub_band_i}")
    sampler_prep = EnsembleSampler(
        nwalkers_prep,
        [ndim],  # assumes ndim_max
        like_prep,
        generating_priors,
        tempering_kwargs={"ntemps": ntemps_prep, "Tmax": np.inf},
        nbranches=1,
        nleaves_max=[1],
        moves=prep_moves,
        kwargs={"start_freq_ind": start_freq_ind, "phase_marginalize": True, **waveform_kwargs_sub},
        backend=None,
        vectorize=True,
        periodic=periodic,  # TODO: add periodic to proposals
        branch_names=["gb"],
        stopping_fn=stop_converge,
        stopping_iterations=1
    )

    # TODO: probably useful to put together a class for this operation if possible that has a "restarter" type method
    # things like cutting down number of points to where we know things are ok
    # waiting for mixing to produce high likelihood residual

    nsteps_prep = 50000
    progress = False  # True if process_i == 2 else False
    # print(f"start - process: {process_i}, sub band: {sub_band_i}, gpu: {gpu_i}")
    out = sampler_prep.run_mcmc(prep_state, nsteps_prep, burn=100, progress=progress, thin_by=100)
    # print(f"end - process: {process_i}, sub band: {sub_band_i}, gpu: {gpu_i}")
    lp = sampler_prep.get_log_prob()
    coords = sampler_prep.get_chain()["gb"]
    keep = np.where(lp == lp.max())
    keep_coords = coords[keep].squeeze()
    if keep_coords.ndim == 2:
        keep_coords = keep_coords[0]

    keep_coords_in = transform_fn.both_transforms(np.array([keep_coords]))
    
    # TODO: fix issue in GBGPU that is issue with not copying data for ll
    ll_temp = gb.get_ll(keep_coords_in, data_temp, noise_in, **waveform_kwargs_sub)

    keep_coords[3] -= np.angle(gb.d_h)[0]    

    keep_coords_in = transform_fn.both_transforms(np.array([keep_coords]))

    ll_check = gb.get_ll(keep_coords_in, data_temp, noise_in, **waveform_kwargs_sub)

    det_snr = (gb.d_h.real[0] / np.sqrt(gb.h_h.real[0])).item()
    det_marg_snr = (np.abs(gb.d_h)[0] / np.sqrt(gb.h_h.real[0])).item()

    det_snr_finding = det_marg_snr
    opt_snr =  (np.sqrt(gb.h_h.real[0])).item()
    # print(f"finish - process: {process_i}, sub band: {sub_band_i}")

    # read out sampler coordinates
    inds = np.argsort(out.log_prob.flatten())[::-1]

    num_coords_out = 500
    starting_coords = out.branches["gb"].coords.reshape(-1, 8)[inds[:num_coords_out]]
    starting_ll_from_sampler = np.sort(out.log_prob.flatten())[::-1][:num_coords_out]

    starting_coords_in = transform_fn.both_transforms(starting_coords)
    ll_check = gb.get_ll(starting_coords_in, data_temp, noise_in, phase_marginalize=True, **waveform_kwargs_sub)

    phase_change = np.angle(gb.non_marg_d_h)

    starting_coords[:, 3] -= phase_change

    starting_coords[:, 3] = starting_coords[:, 3] % (2 * np.pi)
    starting_coords_in = transform_fn.both_transforms(starting_coords)

    ll_check2 = gb.get_ll(starting_coords_in, data_temp, noise_in, phase_marginalize=False, **waveform_kwargs_sub)

    assert np.allclose(ll_check.real, ll_check2.real)

    return dict(
        sub_band_i=sub_band_i,
        new_coords=keep_coords,
        det_snr=det_marg_snr, 
        opt_snr=opt_snr,
        starting_coords=starting_coords,
        starting_ll_from_sampler=starting_ll_from_sampler,
    )

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    #parser.add_argument('--gpu', type=int,
    #                    help='an integer identifier', required=True)

    parser.add_argument('--process-id', '-pi', type=int,
                        help='an integer identifier', required=True)

    parser.add_argument('--dir', type=str,
                        help='directory to save temporary files', required=True)

    args = parser.parse_args()

    while True:
        fps_in = os.listdir(args.dir + f"/process_{args.process_id}")
        time.sleep(0.1)
        if len(fps_in) == 1:
            fp_in = args.dir + f"/process_{args.process_id}/" + fps_in[0]

            if fp_in[-15:] == "transfer.pickle":
                fp_out = fp_in[:-16] + ".pickle"
                time.sleep(0.1)
                with open(fp_in, "rb") as fp_tmp:
                    info_in = pickle.load(fp_tmp)
            
                output = run_single_band_search(*info_in)
                xp.cuda.runtime.setDevice(xp.cuda.runtime.getDevice())
                mempool = xp.get_default_memory_pool()
                mempool.free_all_blocks()

                time.sleep(0.1)
                with open(fp_out, "wb") as fp_tmp:
                    pickle.dump(output, fp_tmp, protocol=pickle.HIGHEST_PROTOCOL)

                os.remove(fp_in)

        elif len(fps_in) > 1:
            raise ValueError("More files than allowed in temporary directory.")
 