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


max_snr_generate = 25.0
generate_dists[0] = uniform_dist(1.0, max_snr_generate)
generate_snr_ladder = PriorContainer(generate_dists)
point_generator = PointGeneratorSNR(generate_snr_ladder)

gb_kwargs = dict(
    waveform_kwargs=waveform_kwargs,
    parameter_transforms=transform_fn,
    search=False,
    search_samples=None,  # out_params,
    search_snrs=None,  # out_snr,
    search_snr_lim=None,  # 50.0,
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
stop_converge = SearchConvergeStopping(n_iters=10000, diff=5.0, verbose=True)

start_snr_for_fixed_lim = 10.0
start_snr_for_adjust_lim = 8.0

data_minus_templates = xp.zeros((ntemps_pe * nwalkers, 2, A_inj.shape[0]), dtype=complex)
data_minus_templates[:, 0] = xp.asarray(A_inj)
data_minus_templates[:, 1] = xp.asarray(E_inj)
waveform_kwargs["start_freq_ind"] = start_freq_ind

if fp_old in os.listdir(folder):
    reader = HDFBackend(fp_old)

    state = get_last_gb_state(data_minus_templates, reader, df, psd, supps_base_shape, fix_temp_initial_ind=3, fix_temp_inds=[4, 5, 6, 7])

    # TODO: update with noies
    psd_here = np.asarray([flat_psd_function(fd, *state.branches_coords["noise_params"].reshape(-1, 1).T) for _ in range(2)]).transpose((1, 0, 2))
                    
    noise_ll = -np.sum(np.log(psd_here), axis=(1, 2)).reshape(ntemps_pe, nwalkers)

    nleaves_max_fixed = state.branches["gb_fixed"].nleaves.max()

    branch_supps_in = {
        "group_move_points": np.zeros((ntemps_pe, nwalkers, nleaves_max, nfriends, ndim))
    }

    obj_contained_shape = (ntemps_pe, nwalkers, nleaves_max)

    branch_supps = BranchSupplimental(branch_supps_in, obj_contained_shape=obj_contained_shape, copy=True)
    #inds_checking = np.arange(np.prod(obj_contained_shape)).reshape(obj_contained_shape)[initial_state.branches_inds["gb"]]

    #branch_supps_in_noise = {
    #    "group_move_points": np.zeros((ntemps_pe, nwalkers, 1, nfriends, n_noise_params))
    #}
    obj_contained_shape_noise = (ntemps_pe, nwalkers, 1)

    # branch_supps_noise = BranchSupplimental(branch_supps_in_noise, obj_contained_shape=obj_contained_shape_noise, copy=True)


    # setup data streams to add to and subtract from
    supps_shape_in = xp.asarray(data).shape

    data_minus_template = state.supplimental.holder["data_minus_template"]

    state.branches["gb"].branch_supplimental = branch_supps
    state.branches["gb_fixed"].branch_supplimental = None
    state.branches["noise_params"].branch_supplimental = None
    # state.branches["noise_params"].branch_supplimental = branch_supps_noise
    state.supplimental = BranchSupplimental({"data_minus_template": data_minus_template}, obj_contained_shape=obj_contained_shape, copy=True)

    state.log_prob += noise_ll
    for name, coords in state.branches_coords.items():
        coords[np.isnan(coords)] = 0.0

    """
    upsample = 1  # int(nwalkers / 20)
    state.branches["gb"].coords[np.isnan(state.branches_coords["gb"])] = 0.0
    new_coords = np.tile(state.branches_coords["gb"], (1, upsample, 1, 1))
    new_inds = np.tile(state.branches_inds["gb"], (1, upsample, 1))
    new_log_prob = np.tile(state.log_prob, (1, upsample))
    new_log_prior = np.tile(state.log_prior,d (1, upsample))
    state = State({"gb": new_coords}, inds={"gb": new_inds}, log_prob=new_log_prob, log_prior=new_log_prior)
    """

else:

    gb.d_d = df * 4 * xp.sum(data_minus_templates.conj() * data_minus_templates / xp.asarray(psd)[None, None, :], axis=(1, 2))

    current_start_points = np.load(current_start_points_file)
    current_start_snrs = np.load(current_start_points_snr_file)[:, 1]

    # TODO: add observed snr to this?
    
    inds_snr_sort = np.argsort(current_start_snrs)
    current_start_snrs = current_start_snrs[inds_snr_sort]
    current_start_points = current_start_points[inds_snr_sort]
    
    keep_fixed = np.arange(len(current_start_snrs))[current_start_snrs > start_snr_for_fixed_lim]
    keep_adjust = np.arange(len(current_start_snrs))[(current_start_snrs > start_snr_for_adjust_lim) & (current_start_snrs <= start_snr_for_fixed_lim)]
    nleaves_max_fixed = len(keep_fixed)
    nleaves_max_adjust = len(keep_adjust)

    print(f"Fixed: {nleaves_max_fixed}, Adjust: {nleaves_max_adjust}")

    assert nleaves_max > nleaves_max_adjust
    coords_adjust = np.zeros((ntemps_pe, nwalkers, nleaves_max, ndim))
    inds_adjust = np.zeros((ntemps_pe, nwalkers, nleaves_max)).astype(bool)

    coords_out_fixed = np.zeros((ntemps_pe, nwalkers, nleaves_max_fixed, ndim))
    inds_out_fixed = np.ones((ntemps_pe, nwalkers, nleaves_max_fixed)).astype(bool)
    fixed_ind = 0
    adjust_ind = 0
    for j, params in enumerate(current_start_points):
        if j not in keep_adjust and j not in keep_fixed:
            continue
        factor = 1e-5
        cov = np.ones(8) * 1e-3
        cov[1] = 1e-7

        start_like = np.zeros((nwalkers * ntemps_pe))
        data_index = xp.asarray((np.tile(np.arange(nwalkers), (ntemps_pe, 1)) + nwalkers * np.repeat(np.arange(ntemps_pe), nwalkers).reshape(ntemps_pe, nwalkers)).astype(np.int32).flatten())
        while np.std(start_like) < 0.5:
            logp = np.full_like(start_like, -np.inf)
            tmp = np.zeros((ntemps_pe * nwalkers, ndim))
            fix = np.ones((ntemps_pe * nwalkers), dtype=bool)
            while np.any(fix):
                tmp[fix] = (params[None, :] * (1. + factor * cov * np.random.randn(nwalkers * ntemps_pe, 8)))[fix]
                logp = priors["gb_fixed"].logpdf(tmp)
                fix = np.isinf(logp)

            tmp_in = transform_fn.both_transforms(tmp, return_transpose=True)
            
            start_like = gb.get_ll(
                tmp_in,
                data_minus_templates.transpose(1, 0, 2).copy(),
                [xp.asarray(psd), xp.asarray(psd)],
                phase_marginalize=False,
                data_index=data_index,
                **waveform_kwargs,
            )  # - np.sum([np.log(psd), np.log(psd)])
            
            old_val = -1/2 * gb.d_d.get().real.copy()
            if j > 0:
                start_like -= -1/2 * gb.d_d.get().real
            #print(np.std(start_like))
            factor *= 1.5


        #  -1 is to do -(-d + h) = d - h  
        data_minus_templates *= -1.
        gb.generate_global_template(tmp_in.T, data_index, data_minus_templates, batch_size=1000, **waveform_kwargs)
        data_minus_templates *= -1.

        gb.d_d = df * 4 * xp.sum(data_minus_templates.conj() * data_minus_templates / xp.asarray(psd)[None, None, :], axis=(1, 2))
    
        if j in keep_fixed:
            coords_out_fixed[:, :, fixed_ind] = tmp.reshape(ntemps_pe, nwalkers, 8)
            fixed_ind += 1
        elif j in keep_adjust:
            coords_adjust[:, :, adjust_ind] = tmp.reshape(ntemps_pe, nwalkers, 8)
            inds_adjust[:, :, adjust_ind] = True
            adjust_ind += 1
        else:
            raise ValueError

        if j > 0:
            assert np.allclose(-1/2 * gb.d_d.get(), start_like + old_val)
        if (j + 1) % 100 == 0:
            print(j+1)
    
    n_noise_params = 1
    noise_start_factor = 0.0 * 1e-1
    coords_start_noise = np.full((ntemps_pe, nwalkers, 1, n_noise_params), base_psd_val) * (1 + noise_start_factor * np.random.randn(ntemps_pe, nwalkers, 1, n_noise_params))
    inds_noise = np.full((ntemps_pe, nwalkers, 1), True)

    # TODO: update with noies
    psd_here = np.asarray([flat_psd_function(fd, *coords_start_noise.reshape(-1, 1).T) for _ in range(2)]).transpose((1, 0, 2))

    signal_ll = (- 1/2 * df * 4 * xp.sum(data_minus_templates.conj() * data_minus_templates / xp.asarray(psd)[None, None, :], axis=(1, 2))).get().reshape(ntemps_pe, nwalkers).real

    noise_ll = -np.sum(np.log(psd_here), axis=(1, 2)).reshape(ntemps_pe, nwalkers).real
    state = state = State(
        {"gb": coords_adjust, "gb_fixed": coords_out_fixed, "noise_params": coords_start_noise}, 
        inds=dict(gb=inds_adjust, gb_fixed=inds_out_fixed, noise_params=inds_noise),
        log_prob=signal_ll + noise_ll,
    )

    branch_supps_in = {
        "group_move_points": np.zeros((ntemps_pe, nwalkers, nleaves_max, nfriends, ndim))
    }

    obj_contained_shape = (ntemps_pe, nwalkers, nleaves_max)

    branch_supps = BranchSupplimental(branch_supps_in, obj_contained_shape=obj_contained_shape, copy=True)
    #inds_checking = np.arange(np.prod(obj_contained_shape)).reshape(obj_contained_shape)[initial_state.branches_inds["gb"]]

    #branch_supps_in_noise = {
    #    "group_move_points": np.zeros((ntemps_pe, nwalkers, 1, nfriends, n_noise_params))
    #}
    obj_contained_shape_noise = (ntemps_pe, nwalkers, 1)

    # branch_supps_noise = BranchSupplimental(branch_supps_in_noise, obj_contained_shape=obj_contained_shape_noise, copy=True)


    # setup data streams to add to and subtract from
    supps_shape_in = xp.asarray(data).shape

    state.branches["gb"].branch_supplimental = branch_supps
    state.branches["gb_fixed"].branch_supplimental = None
    state.branches["noise_params"].branch_supplimental = None
    # state.branches["noise_params"].branch_supplimental = branch_supps_noise

    state.supplimental = BranchSupplimental({"data_minus_template": data_minus_templates.reshape(ntemps_pe, nwalkers, 2, -1).copy()}, obj_contained_shape=obj_contained_shape, copy=True)

    # add inds_keep
    #state.branches_supplimental["gb"].add_objects({"inds_keep": state.branches_inds["gb"]})

    #build_waves(state.branches_coords, inds=state.branches_inds, supps=state.supplimental, branch_supps=state.branches_supplimental, inds_keep=state.branches_inds)

    # remove inds_keep
    #state.branches_supplimental["gb"].remove_objects("inds_keep")

from lisatools.sampling.moves.gbfreqjump import GBFreqJump
from lisatools.sampling.moves.gbspecialgroupstretch import GBSpecialGroupStretchMove

factor = 1e-3

gb_args += ([nleaves_max, nleaves_max_fixed, 1], [min_k, nleaves_max_fixed, 1])

# TODO: add capability to ensemble sampler to track max log L found for every sample, not just current or those stored in the backend. may make it converge faster
# TODO: set to fix_change=+1 during search at least at higher SNRs
# TODO: brute force SNR annealing, where we use the fast likelihood to sample 100 million points and if it does not find anything above that SNR, then we move on. Otherwise, we sparn those in existence and do in model steps until they converge in the Likelihood.
bf = GBMutlipleTryRJ(
    *gb_args,
    **gb_kwargs,
)

from eryn.moves import CombineMove

rj_moves = bf
# rj_moves = PlaceHolder()
if isinstance(rj_moves, PlaceHolder):
    rj_moves.min_k = [0, 0, 0]
    rj_moves.max_k = [1, 1, 1]

df_freq_jump = 1 / YEAR
# TODO: adjust these settings over time during search

#

moves_in = [
    (GBSpecialStretchMove(  # GBGroupStretchMove(
        *gb_args,
        live_dangerously=True, 
        a=2.0, 
        random_seed=random_seed,
        use_gpu=True,
        **gb_kwargs,
        ), 0.5),  # 0.1666666667),
    (GBSpecialGroupStretchMove(
        gb_args,
        gb_kwargs,
        nfriends=nfriends,
        live_dangerously=True, 
        a=2.0, 
        gibbs_sampling_leaves_per=1, 
        n_iter_update=30,
        random_seed=random_seed,
        use_gpu=True,
        skip_supp_names=["group_move_points"]
        ), 0.5)
]

num_rj_moves_repeat = 5
moves = CombineMove(moves_in)
rj_moves = CombineMove([bf for _ in range(num_rj_moves_repeat)])
# rj_moves = PlaceHolder()
#moves = moves[:1]
# moves = PlaceHolder()
"""
import warnings
class HackyTemperatureInfo:
    def __init__(self, ntemps_pe, nwalkers, nsteps):
        self.ntemps_pe, self.nwalkers, self.nsteps = ntemps_pe, nwalkers, nsteps
        self.rj_acceptance_fraction_over_time = np.zeros((self.nsteps, self.ntemps_pe, self.nwalkers))
        self.acceptance_fraction_over_time = np.zeros((self.nsteps, self.ntemps_pe, self.nwalkers))
        self.iteration = 0

    def __call__(self, i, last_result, sampler):
        if self.iteration < self.acceptance_fraction_over_time.shape[0]:
            self.acceptance_fraction_over_time[self.iteration] = sampler.acceptance_fraction.copy()
            self.rj_acceptance_fraction_over_time[self.iteration] = sampler.rj_acceptance_fraction.copy()
            self.iteration += 1
        else:
            warnings.warn("Not adding any more acceptance fraction information because max iterations has been met.")

update_fn = HackyTemperatureInfo(ntemps_pe, nwalkers, 20)
"""

like = GlobalLikelihood(
    None,
    2,
    f_arr=fd,
    parameter_transforms=transform_fn,
    fill_templates=True,
    vectorized=True,
    use_gpu=use_gpu,
    adjust_psd=True
)

like.inject_signal(
    data_stream=[A_inj.copy(), E_inj].copy(),
    noise_fn=flat_psd_function,
    noise_args=[base_psd_val],
    noise_kwargs={"xp": xp},
    add_noise=False,
)

d_d = 4.0 * df * xp.sum(xp.asarray([(temp.conj() * temp) / xp.asarray(psd) for temp in like.injection_channels]))

#### MUST DO THIS #####???
params_test = injection_params.T.copy()[:, np.array([0, 1, 2, 4, 5, 6, 7, 8])]

#check = like(
#    params_test,
#    np.zeros(len(params_test), dtype=np.int32),
#    kwargs_list=[waveform_kwargs],
#    data_length=data_length,
#    start_freq_ind=start_freq_ind,
#)


backend = HDFBackend(fp)

import tqdm

from eryn.moves.tempering import make_ladder

betas = make_ladder(int(ndim * (nleaves_max_fixed + nleaves_max / 2)), ntemps=ntemps_pe)
# betas = np.array([1.0])
sampler = EnsembleSampler(
    nwalkers,
    [ndim, ndim, 1],  # assumes ndim_max
    like,
    priors,
    provide_groups=True,  # TODO: improve this
    provide_supplimental=True,
    tempering_kwargs={"betas": betas},
    nbranches=len(branch_names),
    nleaves_max=[nleaves_max, nleaves_max_fixed, 1],
    moves=moves,
    rj_moves=rj_moves,
    kwargs=dict(
        kwargs_list=[waveform_kwargs, waveform_kwargs, None],
        data_length=data_length,
        start_freq_ind=start_freq_ind,
    ),
    backend=backend,
    vectorize=True,
    plot_iterations=-1,
    periodic=periodic,  # TODO: add periodic to proposals
    branch_names=branch_names,
    verbose=False,
    update_fn=None,
    update_iterations=-1,
    stopping_fn=stop_converge,
    stopping_iterations=10
)

lp = sampler.compute_log_prior(state.branches_coords, inds=state.branches_inds)
state.log_prior = lp

"""# TODO: set this up for check
import time
num = 10
st = time.perf_counter()

for _ in range(num):
    ll = sampler.compute_log_prob(state.branches_coords, inds=state.branches_inds, supps=state.supplimental, logp=lp, branch_supps=state.branches_supplimental)[0]
et = time.perf_counter()
print("timing:", (et - st)/num)
# state.log_prob = ll
"""

#state, accepted = bf.propose(sampler.get_model(), state)
nsteps = 10000
print(noise_ll)
print(fp, (state.log_prob - noise_ll)[0])
# TODO: when resuming make sure betas resume as well. 

out = sampler.run_mcmc(state, nsteps, burn=0, progress=True, thin_by=20)
print(out.log_prob)
breakpoint()
