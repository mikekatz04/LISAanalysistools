from global_fit_settings import *
import cupy as xp
from cupy.cuda.runtime import setDevice


# generate initial search information
num_total = int(1e6)
num_per = int(1e6)
num_rounds = num_total // num_per

data = [
    xp.asarray(A_inj),
    xp.asarray(E_inj),
]


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

moves_mix = GBSpecialStretchMove(
            *gb_args,
            **gb_kwargs,
)


from lisatools.sampling.stopping import SearchConvergeStopping
stop_converge = SearchConvergeStopping(n_iters=10, diff=0.01, verbose=True)

import time

#st = time.perf_counter()

# TODO: add periodic reeval of base LL

snr_lim_iterate = 60.0
prev_snr_lim_iterate = snr_lim_iterate
fp_old2 = "new_mixing_check_12.h5"

if fp_old2 in os.listdir():
    snr_lim_iterate = float(fp_old2.split("_")[-1].split(".")[0]) - 1.0
    if snr_lim_iterate >= 20.0:
        prev_snr_lim_iterate += 5.0
    else:
        prev_snr_lim_iterate += 1.0

    reader = HDFBackend(fp_old2)
    data_here_first = xp.tile(xp.asarray([xp.asarray(A_inj),xp.asarray(E_inj)]), (1 * nwalkers, 1, 1))
else:
    reader = HDFBackend(fp_mix_final)
    data_here_first = xp.tile(xp.asarray([xp.asarray(A_inj),xp.asarray(E_inj)]), (ntemps * nwalkers, 1, 1))

num_per_data = num_per // nwalkers
data_index = xp.asarray(np.repeat(np.arange(nwalkers), num_per_data), dtype=np.int32)
out_coords = []

state = get_last_gb_state(data_here_first, reader, df, psd, supps_base_shape)

ind_keep = np.where(state.log_prob == state.log_prob.max())
ind_keep = (ind_keep[0] * 100 + ind_keep[1]).item()
current_coords = state.branches["gb_fixed"].coords[0]
# current_coords = None
stop_check = True
#data_here = xp.tile(data_here_first.copy()[ind_keep], (data_here_first.shape[0], 1, 1))
data_here = data_here_first.copy()[:100]

gb.d_d = df * 4 * xp.sum((data_here.conj() * data_here) / xp.asarray(psd), axis=(1,2))
run_repeats = 30
for jj in range(1000):
    run_flag = True
    num_bad = 0
    while run_flag:
        out_ll = []
        out_snr = []
        out_params = []
        waveform_kwargs_in = waveform_kwargs.copy()
        for i in range(num_rounds):
            gb.d_d = df * 4 * xp.sum((data_here.conj() * data_here) / xp.asarray(psd), axis=(1,2))
            params = generate_snr_ladder.rvs(size=num_per)
            params_in = transform_fn.both_transforms(params, return_transpose=True)

            data_here_tmp = [data_here[:, 0].copy(), data_here[:, 1].copy()]
            back_d_d = gb.d_d.copy()
            gb.d_d = xp.repeat(gb.d_d, num_per_data)

            if "start_freq_ind" in waveform_kwargs_in:
                waveform_kwargs_in.pop("start_freq_ind")

            ll_tmp = gb.get_ll(
                params_in,
                data_here_tmp,
                [xp.asarray(psd), xp.asarray(psd)],
                phase_marginalize=True,
                start_freq_ind=start_freq_ind,
                data_index=data_index,
                **waveform_kwargs_in,
            )  #  - np.sum([np.log(psd), np.log(psd)])
            
            params_in[0] *= snr_lim_iterate / xp.sqrt(gb.h_h.real).get()
            params[:, 0] = amp_transform.forward(params_in[0], params_in[1])[0]

            phase_maximized_ll = gb.get_ll(
                params_in,
                data_here_tmp,
                [xp.asarray(psd), xp.asarray(psd)],
                phase_marginalize=True,
                start_freq_ind=start_freq_ind,
                data_index=data_index,
                **waveform_kwargs_in,
            )  #  - np.sum([np.log(psd), np.log(psd)])

            gb.d_d = back_d_d.copy()

            phase_maximized_snr = (xp.abs(gb.d_h) / xp.sqrt(gb.h_h.real)).real.copy()
            phase_change = np.angle(gb.non_marg_d_h)

            try:
                phase_maximized_snr = phase_maximized_snr.get()
                phase_change = phase_change.get()

            except AttributeError:
                pass

            params[:, 3] -= phase_change
            params_in[4] -= phase_change

            params[:, 3] %= (2 * np.pi)
            params_in[4] %= (2 * np.pi)

            inds_keep = np.where((phase_maximized_snr > 0.8 * snr_lim_iterate) & (xp.sqrt(gb.h_h.real).get() > snr_lim_iterate) & (phase_maximized_ll == phase_maximized_ll.max()))
            out_ll.append(phase_maximized_ll[inds_keep])  # temp_ll[inds_keep]
            out_snr.append(phase_maximized_snr[inds_keep])
            out_params.append(params[inds_keep])

            #if (i + 1) % 100:
            #    print(i + 1, num_rounds)

        #et = time.perf_counter()

        #print(f"time: {et - st}")
        out_ll = np.concatenate(out_ll)
        out_snr = np.concatenate(out_snr)
        out_params = np.concatenate(out_params, axis=0)

        if len(out_ll) == 0:
            num_bad += 1
            #print("BAD", snr_lim_iterate)
            if num_bad >= run_repeats:
                run_flag = False
            continue

        inds_sort_ll = np.argsort(out_ll)[::-1]
        out_ll = out_ll[inds_sort_ll]
        out_snr = out_snr[inds_sort_ll]
        out_params = out_params[inds_sort_ll]

        #np.save(
        #    "check_pars",
        #    np.concatenate([out_params, np.array([out_snr.real, out_ll]).T], axis=1),
        #)

        best_start_point = out_params[0]
        best_start_ll = out_ll[0]
        best_start_snr = out_snr[0]

        # make sure it is found in all walkers at this snr setup
        check_best = np.tile(best_start_point, (100, 1))
        check_best_in = transform_fn.both_transforms(check_best, return_transpose=True)

        data_index2 = xp.arange(100, dtype=np.int32)
        phase_maximized_ll2 = gb.get_ll(
            check_best_in,
            data_here_tmp,
            [xp.asarray(psd), xp.asarray(psd)],
            phase_marginalize=True,
            start_freq_ind=start_freq_ind,
            data_index=data_index2,
            **waveform_kwargs_in,
        )  #  - np.sum([np.log(psd), np.log(psd)])

        phase_maximized_snr2 = (xp.abs(gb.d_h) / xp.sqrt(gb.h_h.real)).real.copy()
        phase_change2 = np.angle(gb.non_marg_d_h)

        try:
            phase_maximized_snr2 = phase_maximized_snr2.get()
            phase_change2 = phase_change2.get()

        except AttributeError:
            pass

        inds_keep2 = np.where((phase_maximized_snr2 > 0.8 * snr_lim_iterate) & (xp.sqrt(gb.h_h.real).get() > snr_lim_iterate))
        if inds_keep2[0].shape[0] < 100:
            # reset because it almost found one
            if num_bad >= run_repeats:
                run_flag = False
            continue

        # reset num_bad
        num_bad = 0

        print("GOOD", snr_lim_iterate, len(out_coords) + 1)
        
        #coords_out = np.zeros((1, nwalkers, nleaves_max, ndim))

        factor = 1e-7
        cov = np.ones(8) * 1e-3
        cov[1] = 1e-7

        start_like = np.zeros((nwalkers,))
        data_index_find_start = xp.arange(nwalkers, dtype=np.int32)

        # gb.d_d = back_d_d.copy()
        while np.std(start_like) < 5.0:
            tmp = best_start_point[None, :] * (1. + factor * cov * np.random.randn(nwalkers, 8))

            tmp_in = transform_fn.both_transforms(tmp, return_transpose=True)
            
            start_like = gb.get_ll(
                tmp_in,
                data_here_tmp,
                [xp.asarray(psd), xp.asarray(psd)],
                data_index=data_index_find_start,
                phase_marginalize=True,
                start_freq_ind=start_freq_ind,
                **waveform_kwargs_in,
            )

            phase_maximized_snr3 = (xp.abs(gb.d_h) / xp.sqrt(gb.h_h.real)).real.copy()
            phase_change3 = np.angle(gb.non_marg_d_h)

            try:
                phase_maximized_snr3 = phase_maximized_snr3.get()
                phase_change3 = phase_change3.get()

            except AttributeError:
                pass

            tmp[:, 3] -= phase_change3
            tmp_in[4] -= phase_change3

            tmp[:, 3] %= (2 * np.pi)
            tmp_in[4] %= (2 * np.pi)

            factor *= 2
            
        out_coords.append(tmp)
        data_here *= -1.

        gb.generate_global_template(tmp_in.T, data_index_find_start, data_here, start_freq_ind=start_freq_ind, batch_size=1000, **waveform_kwargs_in)
        data_here *= -1.

        gb.d_d = 4 * df * xp.sum((data_here.conj() * data_here) / xp.asarray(psd), axis=(1,2)).real 

        assert np.allclose(start_like, -1/2 * gb.d_d.get())

    print("done", snr_lim_iterate)
    run_mix = False
    if snr_lim_iterate > 20.0:
        if prev_snr_lim_iterate - snr_lim_iterate >= 5.0:
            prev_snr_lim_iterate = snr_lim_iterate
            run_mix = True
    elif snr_lim_iterate >= 5.0:
        if prev_snr_lim_iterate - snr_lim_iterate >= 1.0:
            prev_snr_lim_iterate = snr_lim_iterate
            run_mix = True

    else:
        break

    if len(out_coords) == 0:
        run_mix = False

    if run_mix:
        try:
            del sampler_mix
            del like_mix
            
        except NameError:
            pass

        

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

        reader = f"new_mixing_check_{snr_lim_iterate:.2g}.h5"

        coords_to_add = np.asarray(out_coords)
        num_new = coords_to_add.shape[0]
        if current_coords is not None:
            nleaves_max_fix_old = current_coords.shape[1] 
            
            nleaves_max_fix = nleaves_max_fix_old + num_new

            new_coords_arr = np.zeros((nwalkers, nleaves_max_fix, ndim))
            new_coords_arr[:, :nleaves_max_fix_old] = current_coords
            new_coords_arr[:, nleaves_max_fix_old:] = coords_to_add.transpose((1, 0, 2))

            old_coords = current_coords.copy()
            

        else:
            nleaves_max_fix_old = 0
            nleaves_max_fix = nleaves_max_fix_old + num_new
            new_coords_arr = coords_to_add.transpose((1, 0, 2))
        
        current_coords = new_coords_arr
        start_ll_here = -1/2 * gb.d_d.get().real

        supp = BranchSupplimental({"data_minus_template": data_here.reshape(1, nwalkers, 2, -1)}, obj_contained_shape=(1, nwalkers), copy=True)
        state_mix = State({"gb_fixed": current_coords[None, :, :, :]}, log_prob=start_ll_here[None, :], supplimental=supp)

        out_coords = []

        sampler_mix = EnsembleSampler(
            nwalkers,
            [ndim],  # assumes ndim_max
            like_mix,
            priors,
            tempering_kwargs={"betas": np.array([1.])},
            nbranches=1,
            nleaves_max=[nleaves_max_fix],
            moves=moves_mix,
            kwargs=None,  # {"start_freq_ind": start_freq_ind, **waveform_kwargs},
            backend=reader,
            vectorize=True,
            periodic=periodic,  # TODO: add periodic to proposals
            branch_names=["gb_fixed"],
            stopping_fn=stop_converge,
            stopping_iterations=1,
            provide_groups=True,
            provide_supplimental=True,
        )

        nsteps_mix = 2000
        print("Starting mix ll best:", state_mix.log_prob.max(), current_coords.shape)
        out = sampler_mix.run_mcmc(state_mix, nsteps_mix, progress=True, thin_by=50)
        current_coords = out.branches_coords["gb_fixed"][0]

        ind_keep = np.where(state.log_prob == state.log_prob.max())
        ind_keep = (ind_keep[0] * 100 + ind_keep[1]).item()
        data_here = out.supplimental.holder["data_minus_template"][0].copy()
        print("Mix complete", current_coords.shape)

    snr_lim_iterate -= 1.0

breakpoint()
inds = out_ll < -1/2 * d_h_d_h
plt.scatter(out_params[inds, 1], out_params[inds, 0], c="C0", s=15, label="Binary not found")
plt.scatter(out_params[~inds, 1], out_params[~inds, 0], c="C1", s=40, label="Potential binary")
plt.axhline(injection_params[0, -1], c="C2", label="True missing binary parameters")
plt.axvline(injection_params[1, -1], c="C2")
plt.xlabel("Frequency (Hz)", fontsize=16)
plt.ylabel(r"$\log{A}$", fontsize=16)

#plt.axhline(injection_params[0, -2], c="C3", lw=3)
#plt.axvline(injection_params[1, -2], c="C3", lw=3)
plt.legend(loc="upper left")
plt.show()
