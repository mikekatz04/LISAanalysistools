from global_fit_settings import *
import cupy as xp
from cupy.cuda.runtime import setDevice


data = [
    xp.asarray(A_inj),
    xp.asarray(E_inj),
]

# TODO: fix initial setup for mix where it backs up the likelihood
gb_args = (
    gb,
    priors,
    int(1e3),
    start_freq_ind,
    data_length,
    data,
    psd_in,
    xp.asarray(fd),
    # [nleaves_max, 1],
    # [min_k, 1],
)
class PointGeneratorSNR:
    def __init__(self, generate_snr_ladder):
        self.generate_snr_ladder = generate_snr_ladder

    def __call__(self, size=(1,)):
        new_points = self.generate_snr_ladder.rvs(size=size)
        logpdf = self.generate_snr_ladder.logpdf(new_points)
        return (new_points, logpdf)

point_generator = PointGeneratorSNR(generate_snr_ladder)

gb_kwargs = dict(
    waveform_kwargs=waveform_kwargs,
    parameter_transforms=transform_fn,
    search=False,
    search_samples=None,  # out_params,
    search_snrs=None,  # out_snr,
    search_snr_lim=None, # snr_lim,  # 50.0,
    psd_func=flat_psd_function,
    noise_kwargs=dict(xp=xp),
    provide_betas=True,
    point_generator_func=point_generator,
    batch_size=-1,
    skip_supp_names=["group_move_points"]
)

from lisatools.sampling.stopping import SearchConvergeStopping
stop_converge = SearchConvergeStopping(n_iters=30, diff=0.01, verbose=False)
stop_converge_mix = SearchConvergeStopping(n_iters=10, diff=0.01, verbose=True)

generating_priors = deepcopy(priors)

waveform_kwargs["start_freq_ind"] = start_freq_ind

snr_lim = 5.0
dbin = 2 * N

f_width = f0_lims[1] - f0_lims[0]
num_f_bins = int(np.ceil(f_width / df))
num_sub_bands = num_f_bins // dbin + int((num_f_bins % dbin) != 0)
lim_inds = np.concatenate([np.array([0]), np.arange(dbin, num_f_bins, dbin), np.array([num_f_bins - 1])])
search_f_bin_lims = f0_lims[0] + df * np.arange(num_f_bins)[lim_inds]
search_f_bin_lims[-1] = f0_lims[-1]
run_mix_first = False
# TODO: adjust the starting points each iteration to avoid issues at gaps
print(f"num_sub_bands: {num_sub_bands}")
if fp_mix_final not in os.listdir():
    
    data_minus_templates = xp.asarray([A_inj, E_inj])[None, :, :]

    if current_start_points_file not in os.listdir():
        current_start_points = []  
        current_snrs_search = []
        gb.d_d = df * 4 * xp.sum((data_minus_templates.conj() * data_minus_templates) / xp.asarray(psd), axis=(1,2))

    else:
        current_start_points = np.load(current_start_points_file)
        current_start_points_in = transform_fn.both_transforms(current_start_points)
        current_snrs_search = list(np.load(current_start_points_snr_file))

        groups = xp.zeros(len(current_start_points), dtype=np.int32)

        #  -1 is to do -(-d + h) = d - h  
        data_minus_templates *= -1.
        gb.generate_global_template(current_start_points_in, groups, data_minus_templates, batch_size=1000, **waveform_kwargs)
        data_minus_templates *= -1.

        current_start_points = list(current_start_points)

    data_minus_templates = data_minus_templates.squeeze()

    nwalkers_prep = 500
    ntemps_prep = 10

    from eryn.moves import StretchMove
    prep_moves = [StretchMove()]

    max_iter = 1000
    snr_break = 0.0
    num_bin_pause = 100
    like_prep = Likelihood(gb, 2, f_arr=fd, parameter_transforms={"gb": transform_fn}, fill_data_noise=True, vectorized=True, transpose_params=True, use_gpu=use_gpu)
    num_binaries_needed_to_mix = 1
    num_binaries_current = 0
    max_iter = 1000
    for iter_i in range(max_iter):
        try:
            del like_prep.injection_channels
            del like_prep.psd
        except AttributeError:
            pass
        try:
            del sampler_prep
            del prep_state
        except NameError:
            pass
        like_prep.inject_signal(
            data_stream=list(data_minus_templates.get()),
            noise_fn=flat_psd_function,
            noise_args=[[base_psd_val]],
            noise_kwargs={"xp": np},
            add_noise=False,
        )

        data_temp = like_prep.injection_channels
        noise_in = like_prep.psd

        gb.d_d = df * 4 * xp.sum(data_minus_templates.conj() * data_minus_templates / xp.asarray(psd)[None, :])
        print(-1/2 * gb.d_d, "ll going in")
        tmp_found_this_iter = 0
        num_total = int(5e6)
        num_per = int(5e6)
        num_rounds = num_total // num_per
        start_bin_inds = np.arange(len(search_f_bin_lims) - 1)
        if (iter_i > 0 and run_mix_first) or not run_mix_first:
            for jj in range(2):
                # odd or even
                start_bin_inds_here = start_bin_inds[jj::2]
                end_bin_inds_here = start_bin_inds_here + 1
                lower_f0 = search_f_bin_lims[start_bin_inds_here]
                upper_f0 = search_f_bin_lims[end_bin_inds_here]

                for sub_band_i, (sub_band_lower_f0, sub_band_upper_f0) in enumerate(zip(lower_f0, upper_f0)):
                    new_dist = uniform_dist(sub_band_lower_f0 * 1e3, sub_band_upper_f0 * 1e3)
                    generating_priors["gb"].priors_in[1] = new_dist
                    generating_priors["gb"].priors[1][1] = new_dist
                    
                    out_ll = []
                    out_snr = []
                    out_params = []
                    for i in range(num_rounds):
                        params = generating_priors["gb"].rvs(size=(num_per, 1)).reshape(num_per, -1)

                        params_in = transform_fn.both_transforms(params, return_transpose=True)

                        phase_maximized_ll = gb.get_ll(
                            params_in,
                            data_temp,
                            noise_in,
                            phase_marginalize=True,
                            **waveform_kwargs,
                        )
                        
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
                        #    print(i + 1, num_rounds)

                    out_ll = np.concatenate(out_ll)
                    out_snr = np.concatenate(out_snr)
                    out_params = np.concatenate(out_params, axis=0)

                    inds_sort_snr = np.argsort(out_ll)[::-1]
                    out_ll = out_ll[inds_sort_snr]
                    out_snr = out_snr[inds_sort_snr]
                    out_params = out_params[inds_sort_snr]

                    start_coords = out_params[:ntemps_prep * nwalkers_prep].reshape(ntemps_prep, nwalkers_prep, 1, -1)
                    start_ll = out_ll[:ntemps_prep * nwalkers_prep].reshape(ntemps_prep, nwalkers_prep)
                    start_snr = out_snr[:ntemps_prep * nwalkers_prep].reshape(ntemps_prep, nwalkers_prep)

                    prep_state = State({"gb": start_coords}, log_prob=start_ll)

                    sampler_prep = EnsembleSampler(
                        nwalkers_prep,
                        [ndim],  # assumes ndim_max
                        like_prep,
                        generating_priors,
                        tempering_kwargs={"ntemps": ntemps_prep, "Tmax": np.inf},
                        nbranches=1,
                        nleaves_max=[1],
                        moves=prep_moves,
                        kwargs={"start_freq_ind": start_freq_ind, **waveform_kwargs},
                        backend=None,
                        vectorize=True,
                        periodic=periodic,  # TODO: add periodic to proposals
                        branch_names=["gb"],
                        stopping_fn=stop_converge,
                        stopping_iterations=50
                    )

                    nsteps_prep = 50000
                    out = sampler_prep.run_mcmc(prep_state, nsteps_prep, burn=100, progress=True, thin_by=1)
                    lp = sampler_prep.get_log_prob()
                    coords = sampler_prep.get_chain()["gb"]
                    keep = np.where(lp == lp.max())
                    keep_coords = coords[keep].squeeze()
                    if keep_coords.ndim == 2:
                        keep_coords = keep_coords[0]

                    current_start_points.append(keep_coords)

                    keep_coords_in = transform_fn.both_transforms(np.array([keep_coords]))
                    
                    # TODO: fix issue in GBGPU that is issue with not copying data for ll
                    try:
                        ll_temp = gb.get_ll(keep_coords_in.T, data_temp, noise_in, **waveform_kwargs)
                    except ValueError:
                        breakpoint()

                    det_snr = (gb.d_h.real[0] / np.sqrt(gb.h_h.real[0])).item()
                    det_snr_finding = det_snr
                    opt_snr =  (np.sqrt(gb.h_h.real[0])).item()

                    if opt_snr < snr_lim:
                        print("found source to low in optimal SNR: ", det_snr, opt_snr, f"sub band: {sub_band_i + 1} out of {len(lower_f0)}")
                        continue
                    current_snrs_search.append([det_snr, opt_snr])

                    print("found source: ", det_snr, opt_snr, f"sub band: {sub_band_i + 1} out of {len(lower_f0)}")

                    num_binaries_current += 1

                    groups = xp.zeros(1, dtype=np.int32)
                    #  -1 is to do -(-d + h) = d - h  
                    data_minus_templates *= -1.
                    gb.generate_global_template(keep_coords_in, groups, data_minus_templates[None, :, :], batch_size=1000, **waveform_kwargs)
                    data_minus_templates *= -1.

                    ll_check = -1/2 * df * 4 * xp.sum(data_minus_templates.conj() * data_minus_templates / xp.asarray(psd)[None, :])

                    gb.d_d = df * 4 * xp.sum(data_minus_templates.conj() * data_minus_templates / xp.asarray(psd)[None, :])

            if num_binaries_current == 0:
                print("END")
                breakpoint()
        
            # remove the old mixing file if we are adding binaries
            if fp_mix in os.listdir():
                os.remove(fp_mix)

        np.save(current_start_points_snr_file, np.asarray(current_snrs_search))
        np.save(current_start_points_file, np.asarray(current_start_points))

        current_start_points = np.load(current_start_points_file)

        nleaves_max_fix = len(current_start_points)
        coords_out = np.zeros((ntemps, nwalkers, nleaves_max_fix, ndim))
        data_minus_templates_mix = xp.zeros((ntemps * nwalkers, 2, A_inj.shape[0]), dtype=complex)
        data_minus_templates_mix[:, 0] = xp.asarray(A_inj)
        data_minus_templates_mix[:, 1] = xp.asarray(E_inj)

        gb.d_d = df * 4 * xp.sum(data_minus_templates_mix.conj() * data_minus_templates_mix / xp.asarray(psd)[None, :], axis=(1, 2))
        
        # setup data streams to add to and subtract from
        supps_shape_in = xp.asarray(data_minus_templates_mix).shape

        reader_mix = HDFBackend(fp_mix)
        if fp_mix in os.listdir():
            state_mix = get_last_gb_state(data_minus_templates_mix, reader_mix, df, psd, supps_base_shape)

        else:
            for j, params in enumerate(current_start_points):
                factor = 1e-5
                cov = np.ones(8) * 1e-3
                cov[1] = 1e-7

                start_like = np.zeros((nwalkers * ntemps))
                data_index = xp.asarray((np.tile(np.arange(nwalkers), (ntemps, 1)) + nwalkers * np.repeat(np.arange(ntemps), nwalkers).reshape(ntemps, nwalkers)).astype(np.int32).flatten())
                iter_check = 0
                run_flag = True
                while np.std(start_like) < 1.0 and run_flag:
                    
                    logp = np.full_like(start_like, -np.inf)
                    tmp = np.zeros((ntemps * nwalkers, ndim))
                    fix = np.ones((ntemps * nwalkers), dtype=bool)
                    while np.any(fix):
                        tmp[fix] = (params[None, :] * (1. + factor * cov * np.random.randn(nwalkers * ntemps, 8)))[fix]
                        logp = priors["gb_fixed"].logpdf(tmp)
                        fix = np.isinf(logp)
                        if np.all(fix):
                            breakpoint()

                    tmp_in = transform_fn.both_transforms(tmp, return_transpose=True)
                    
                    start_like = gb.get_ll(
                        tmp_in,
                        data_minus_templates_mix.transpose(1, 0, 2).copy(),
                        [xp.asarray(psd), xp.asarray(psd)],
                        phase_marginalize=False,
                        data_index=data_index,
                        **waveform_kwargs,
                    )  # - np.sum([np.log(psd), np.log(psd)])
                    iter_check += 1
                
                    if j > 0:
                        start_like -= -1/2 * gb.d_d.get().real
                    #print(np.std(start_like))
                    factor *= 1.5

                    if factor > 10.0:
                        run_flag = False

                #  -1 is to do -(-d + h) = d - h  
                data_minus_templates_mix *= -1.
                gb.generate_global_template(tmp_in.T, data_index, data_minus_templates_mix, batch_size=1000, **waveform_kwargs)
                data_minus_templates_mix *= -1.

                gb.d_d = df * 4 * xp.sum(data_minus_templates_mix.conj() * data_minus_templates_mix / xp.asarray(psd)[None, None, :], axis=(1, 2))
            
                coords_out[:, :, j] = tmp.reshape(ntemps, nwalkers, 8)
                print(j)

            ll = -1/2 * df * 4 * xp.sum(data_minus_templates_mix.conj() * data_minus_templates_mix / xp.asarray(psd)[None, None, :], axis=(1, 2)).real

            data_minus_templates_mix = data_minus_templates_mix.reshape(ntemps, nwalkers, 2, -1).copy()
            
            supps = BranchSupplimental({"data_minus_template": data_minus_templates_mix}, obj_contained_shape=supps_base_shape, copy=True)

            state_mix = State({"gb_fixed": coords_out}, log_prob=ll.reshape(ntemps, nwalkers).get(), supplimental=supps)
            check = priors["gb_fixed"].logpdf(coords_out.reshape(-1, 8))

            if np.any(np.isinf(check)):
                breakpoint()

        moves_mix = GBSpecialStretchMove(
            *gb_args,
            **gb_kwargs,
        )

        like_mix = GlobalLikelihood(
            None,
            2,
            f_arr=fd,
            parameter_transforms=transform_fn,
            fill_templates=True,
            vectorized=True,
            use_gpu=use_gpu,
            adjust_psd=False
        )

        like_mix.inject_signal(
            data_stream=[A_inj.copy(), E_inj].copy(),
            noise_fn=flat_psd_function,
            noise_args=[(base_psd_val,)],
            #noise_kwargs={"xp": np},
            add_noise=False,
        )

        sampler_mix = EnsembleSampler(
                nwalkers,
                [ndim],  # assumes ndim_max
                like_mix,
                priors,
                tempering_kwargs={"ntemps": ntemps},
                nbranches=1,
                nleaves_max=[nleaves_max_fix],
                moves=moves_mix,
                kwargs=None,  # {"start_freq_ind": start_freq_ind, **waveform_kwargs},
                backend=reader_mix,
                vectorize=True,
                periodic=periodic,  # TODO: add periodic to proposals
                branch_names=["gb_fixed"],
                stopping_fn=stop_converge_mix,
                stopping_iterations=1,
                provide_groups=True,
                provide_supplimental=True,
            )

        nsteps_mix = 200
        print("Starting mix ll best:", state_mix.log_prob.max())
        out = sampler_mix.run_mcmc(state_mix, nsteps_mix, progress=True, thin_by=50)
        max_ind = np.where(out.log_prob == out.log_prob.max())

        data_minus_templates = out.supplimental.holder["data_minus_template"][max_ind].squeeze().copy()
        current_start_points = list(out.branches["gb_fixed"].coords[max_ind][out.branches["gb_fixed"].inds[max_ind]])
        
        current_start_points_in = transform_fn.both_transforms(np.asarray(current_start_points))

        gb.d_d = df * 4 * xp.sum(data_minus_templates.conj() * data_minus_templates / xp.asarray(psd)[None, :])

        _ = gb.get_ll(current_start_points_in.T, data_minus_templates, [xp.asarray(psd), xp.asarray(psd)], **waveform_kwargs)

        det_snr = (gb.d_h.real / np.sqrt(gb.h_h.real)).get()
        opt_snr =  (np.sqrt(gb.h_h.real)).get()

        current_snrs_search = list(np.array([det_snr, opt_snr]).T)
        np.save(current_start_points_snr_file, np.asarray(current_snrs_search))
        np.save(current_start_points_file, np.asarray(current_start_points))
        num_binaries_current = 0

        if fp_mix in os.listdir():
            os.remove(fp_mix)
        
        print("DONE MIXING","num sources:", len(current_start_points), "opt snrs:", np.sort(opt_snr))
        #if det_snr_finding < snr_break or len(current_snrs_search) > num_bin_pause:
        #    break

copyfile(fp_mix, fp_mix_final)
            
        
