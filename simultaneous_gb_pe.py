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
from lisatools.sampling.prior import FullGaussianMixtureModel
from eryn.moves.tempering import make_ladder

import subprocess

import warnings

warnings.filterwarnings("ignore")

stop_here = True

from eryn.moves import Move
from lisatools.globalfit.state import State
from lisatools.globalfit.hdfbackend import HDFBackend

band_edges = search_f_bin_lims

# for _ in range(2):
#     band_edges = np.sort(np.concatenate([band_edges, (band_edges[:-1] + band_edges[1:]) / 2]))

class PlaceHolder(Move):
    def __init__(self, *args, **kwargs):
        super(PlaceHolder, self).__init__(*args, **kwargs)

    def propose(self, model, state):
        accepted = np.zeros(state.log_like.shape)
        self.temperature_control.swaps_accepted = np.zeros(
            self.temperature_control.ntemps - 1
        )
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
    return np.take_along_axis(a, idx, axis=axis)


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
    def __init__(
        self, initial_mbhs, mgh, fp_psd, fp_mbh, fp_gb, psd_shape, waveform_kwargs
    ):
        self.mgh = mgh
        self.fp_psd = fp_psd
        self.fp_mbh = fp_mbh
        self.psd_shape = psd_shape
        self.fp_gb = fp_gb
        self.waveform_kwargs = waveform_kwargs
        self.last_mbh_template = initial_mbhs
        self.output_residual_number = 0

    def __call__(self, iter, last_sample, sampler):

        with open(current_save_state_file, "wb") as fp:
            pickle.dump(last_sample, fp, pickle.HIGHEST_PROTOCOL)

        return
        """residual_file = "residuals_for_flows"
        psd_file = "psds_for_flows"
        freq_file = "freq_for_flows"

        np.save(residual_file, np.asarray([self.mgh.data_shaped[0][0][:100].get(), self.mgh.data_shaped[1][0][:100].get()]).transpose(1, 0, 2))
        np.save(psd_file, np.asarray([self.mgh.psd_shaped[0][0][:100].get(), self.mgh.psd_shaped[1][0][:100].get()]).transpose(1, 0, 2))
        np.save(freq_file, self.mgh.fd)
        breakpoint()"""

        nwalkers_pe = self.psd_shape[1]

        self.output_residual_number += 1
        A_psd_in = np.zeros(self.psd_shape, dtype=np.float64)
        E_psd_in = np.zeros(self.psd_shape, dtype=np.float64)
        xp.get_default_memory_pool().free_all_blocks()
        imported = False
        while not imported:
            try:
                psds = np.load(self.fp_psd)
                imported = True
            except ValueError:
                time.sleep(1)

        psds[:, :, 0] = psds[:, :, 1]
        A_psd_in[:] = psds[:, 0][None, :nwalkers_pe]  # A
        E_psd_in[:] = psds[:, 1][None, :nwalkers_pe]  # E
        xp.get_default_memory_pool().free_all_blocks()
        self.mgh.set_psd_from_arrays(
            A_psd_in.reshape(-1, self.psd_shape[-1]),
            E_psd_in.reshape(-1, self.psd_shape[-1]),
            overall_inds=self.mgh.map,
        )
        xp.get_default_memory_pool().free_all_blocks()
        # ll_bef = self.mgh.get_ll(include_psd_info=True)
        A_mbh_remove = self.last_mbh_template[0]
        E_mbh_remove = self.last_mbh_template[1]
        self.mgh.add_templates_from_arrays_to_residuals(
            -1 * A_mbh_remove.reshape(-1, self.psd_shape[-1]),
            -1 * E_mbh_remove.reshape(-1, self.psd_shape[-1]),
            overall_inds=self.mgh.map,
        )
        # ll_af = self.mgh.get_ll(include_psd_info=True)
        imported = False
        while not imported:
            try:
                mbh_inj = np.load(fp_mbh + ".npy")
                imported = True
            except ValueError:
                time.sleep(1)

        A_mbh_going_in = np.zeros_like(self.last_mbh_template[0])
        E_mbh_going_in = np.zeros_like(self.last_mbh_template[1])

        A_mbh_going_in[:] = mbh_inj[:, 0][None, :nwalkers_pe]  # A
        E_mbh_going_in[:] = mbh_inj[:, 1][None, :nwalkers_pe]  # A
        xp.get_default_memory_pool().free_all_blocks()
        # TODO: need to check that everything is aligned
        ll_bef = self.mgh.get_ll(include_psd_info=True)
        self.mgh.add_templates_from_arrays_to_residuals(
            A_mbh_going_in.reshape(-1, self.psd_shape[-1]),
            E_mbh_going_in.reshape(-1, self.psd_shape[-1]),
            overall_inds=self.mgh.map,
        )
        # ll_af = self.mgh.get_ll(include_psd_info=True)
        
        self.last_mbh_template = [A_mbh_going_in, E_mbh_going_in]

        ll = self.mgh.get_ll(include_psd_info=True)

        last_sample.log_like = ll.flatten()[self.mgh.map].reshape(ll.shape)
        xp.get_default_memory_pool().free_all_blocks()
        """tmp = self.mgh.data_shaped

        if len(tmp[0]) > 1:
            raise ValueError

        nwalkers = self.psd_shape[1]
        A_out = tmp[0][0][self.mgh.map[:nwalkers]].get()
        E_out = tmp[1][0][self.mgh.map[:nwalkers]].get()
        """
        coords_out_gb_fixed = last_sample.branches_coords["gb_fixed"][0]
        coords_in = transform_fn.both_transforms(
            coords_out_gb_fixed[last_sample.branches["gb_fixed"].inds[0]]
        )
        ntemps_pe, nwalkers_pe, nleaves_max, ndim = last_sample.branches[
            "gb_fixed"
        ].shape
        xp.get_default_memory_pool().free_all_blocks()
        walker_vals = np.repeat(np.arange(nwalkers_pe)[:, None], nleaves_max, axis=-1)

        data_index = xp.asarray(
            walker_vals[last_sample.branches["gb_fixed"].inds[0]]
        ).astype(xp.int32)
        xp.get_default_memory_pool().free_all_blocks()
        # NEEDS TO BE +1
        factors = +xp.ones_like(data_index, dtype=xp.float64)
        xp.get_default_memory_pool().free_all_blocks()
        templates_out = [
            [xp.zeros((nwalkers_pe, self.psd_shape[-1]), dtype=complex).flatten()],
            [xp.zeros((nwalkers_pe, self.psd_shape[-1]), dtype=complex).flatten()],
        ]
        A_out, E_out = templates_out[0][0], templates_out[1][0]
        data_splits = [np.arange(nwalkers_pe)]

        main_gpu = xp.cuda.runtime.getDevice()

        gb.generate_global_template(
            coords_in,
            data_index,
            templates_out,
            data_length=data_length,
            factors=factors,
            data_splits=data_splits,
            **self.waveform_kwargs,
        )

        xp.cuda.runtime.setDevice(main_gpu)
        np.save(
            self.fp_gb,
            xp.array([A_out.reshape(nwalkers_pe, -1), E_out.reshape(nwalkers_pe, -1)])
            .transpose((1, 0, 2))
            .get(),
        )
        xp.get_default_memory_pool().free_all_blocks()


def run_gb_pe(gpu):
    while fp_psd + ".npy" not in os.listdir():
        print(f"{fp_psd + '.npy'} not in current directory so far...")
        time.sleep(20)

    gpus_pe = [gpu]
    gpus = gpus_pe
    # from lisatools.sampling.stopping import SearchConvergeStopping2

    waveform_kwargs["start_freq_ind"] = start_freq_ind

    # for testing
    # band_edges = np.arange(f0_lims_in[0], f0_lims_in[1], 2 * 128 * df)

    num_sub_bands = len(band_edges)

    xp.cuda.runtime.setDevice(gpus[0])

    nwalkers_pe = 18

    snrs_ladder = np.array([1., 1.5, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 75.0, 100.0, 5e2])
    ntemps_pe = 24  # len(snrs_ladder)
    # betas =  1 / snrs_ladder ** 2  # make_ladder(ndim * 10, Tmax=5e6, ntemps=ntemps_pe)
    betas = 1 / 1.2 ** np.arange(ntemps_pe)
    betas[-1] = 0.0001

    num_binaries_needed_to_mix = 1
    num_binaries_current = 0

    if fp_pe in os.listdir():
        reader = HDFBackend(fp_pe)
        last_sample = reader.get_last_sample()
        nleaves_max_fix = last_sample.branches["gb_fixed"].coords.shape[
            -2
        ]  # will all be the same

        nleaves_max_fix_new = nleaves_max_fix

    elif current_save_state_file in os.listdir():
        with open(current_save_state_file, "rb") as fp_out:
            last_sample = pickle.load(fp_out)

        coords = last_sample.branches_coords
        inds = last_sample.branches_inds
        last_sample = State(
            coords,
            inds=inds,
            log_like=last_sample.log_like,
            log_prior=last_sample.log_prior,
            band_info=last_sample.band_info
        )

        nleaves_max_fix = last_sample.branches["gb_fixed"].coords.shape[
            -2
        ]  # will all be the same

        nleaves_max_fix_new = 15000  # nleaves_max_fix

    elif "final_search_mixing_copy_4.pickle" in os.listdir():
        with open("final_search_mixing_copy_4.pickle", "rb") as fp_search_state:
            end_search_state = pickle.load(fp_search_state)

        # TODO: check this!!!
        end_search_gb_coords = end_search_state.branches["gb_fixed"].coords[:, :nwalkers_pe]
        end_search_gb_inds = end_search_state.branches["gb_fixed"].inds[:, :nwalkers_pe]

        test_coords = end_search_gb_coords[0, 0]
        test_inds = end_search_gb_inds[0, 0]
        for num in range(6):
            
            # test_coords = test_coords[test_coords[:, 1] > 15.0]
            # order = np.arange(test_coords.shape[0])  # np.argsort(test_coords[:, 1])
            # test_coords = test_coords[order]

            test_template = xp.asarray([[A_inj, E_inj]])
            last_test_template = test_template.copy()
            last_h_h_test = (-1/2 * 4 * df * xp.sum(test_template.conj() * test_template / xp.asarray(psd_in))).real
            for i, test_coord in enumerate(test_coords):
                if not test_inds[i]:
                    continue
                test_coord_in = transform_fn.both_transforms(np.array([test_coord]))
                waveform_kwargs_test = waveform_kwargs.copy()
                waveform_kwargs_test['use_c_implementation'] = False

                gb.generate_global_template(
                    test_coord_in,
                    xp.zeros(test_coord_in.shape[0], dtype=xp.int32),
                    test_template,
                    factors=-xp.ones(test_coord_in.shape[0]),
                    **waveform_kwargs,
                )

                h_h_test = -1/2 * 4 * df * xp.sum(test_template.conj() * test_template / xp.asarray(psd_in))
                diff_temp = -(test_template - last_test_template)
                h_h_new_test = 4 * df * xp.sum(diff_temp.conj() * diff_temp / xp.asarray(psd_in))
                ll_diff = h_h_test.real - last_h_h_test
                if ll_diff < num * 20.0:
                    test_inds[i] = False
                    test_template[:] = last_test_template[:]
                else:
                    last_h_h_test = h_h_test.real
                    last_test_template[:] = test_template[:]
            print(test_inds.sum())
        
        end_search_gb_inds[:, :, :] = test_inds[:]

        walker_factor = nwalkers_pe / end_search_gb_coords.shape[1]
        # assert float(walker_factor) == float(nwalkers_pe) / float(end_search_gb_coords.shape[1])
        start_coords = {
            "gb_fixed": np.repeat(end_search_gb_coords, repeats=int(ntemps_pe * walker_factor), axis=0).reshape((ntemps_pe, nwalkers_pe,) + end_search_gb_coords.shape[2:])
        }
        start_inds = {
            "gb_fixed": np.repeat(end_search_gb_inds, repeats=int(ntemps_pe * walker_factor), axis=0).reshape((ntemps_pe, nwalkers_pe,) + end_search_gb_coords.shape[2:3])
        }
        # breakpoint()
        start_like = np.repeat(end_search_state.log_like[:, :nwalkers_pe], repeats=int(ntemps_pe * walker_factor), axis=0).reshape((ntemps_pe, nwalkers_pe,))
        start_prior = np.repeat(end_search_state.log_prior[:, :nwalkers_pe], repeats=int(ntemps_pe * walker_factor), axis=0).reshape((ntemps_pe, nwalkers_pe,))

        last_sample = State(
            start_coords,
            inds=start_inds,
            log_like=start_like,
            log_prior=start_prior,
        )

        nleaves_max_fix_new = 15000
        nleaves_max_fix = end_search_gb_inds.shape[-1]

    else:
        raise FileNotFoundError(
            current_save_state_file + " and " + fp_pe + " not in current directory"
        )

    coords_new = np.zeros((ntemps_pe, nwalkers_pe, nleaves_max_fix_new, ndim))
    coords_new[:, :, :nleaves_max_fix, :] = last_sample.branches["gb_fixed"].coords[:]

    inds_new = np.zeros((ntemps_pe, nwalkers_pe, nleaves_max_fix_new), dtype=bool)
    inds_new[:, :, :nleaves_max_fix] = last_sample.branches["gb_fixed"].inds[:]

    new_sample = State(
        {"gb_fixed": coords_new},
        inds={"gb_fixed": inds_new},
        log_like=last_sample.log_like,
        log_prior=last_sample.log_prior,
        band_info=last_sample.band_info
    )

    if not hasattr(new_sample, "band_info"):
        band_temps = np.tile(np.asarray(betas), (len(band_edges) - 1, 1))
        new_sample.initialize_band_information(nwalkers_pe, ntemps_pe, band_edges, band_temps)

    A_going_in = np.zeros((2, nwalkers_pe, A_inj.shape[0]), dtype=complex)
    E_going_in = np.zeros((2, nwalkers_pe, E_inj.shape[0]), dtype=complex)
    A_going_in[:] = np.asarray(A_inj)
    E_going_in[:] = np.asarray(E_inj)

    mbh_inj = np.load(fp_mbh + ".npy")

    A_mbh_going_in = np.zeros((2, nwalkers_pe, A_inj.shape[0]), dtype=complex)
    E_mbh_going_in = np.zeros((2, nwalkers_pe, E_inj.shape[0]), dtype=complex)

    A_mbh_going_in[:] = np.repeat(mbh_inj[:nwalkers_pe, 0], 2, axis=0).reshape(nwalkers_pe, 2, A_inj.shape[0]).transpose(1, 0, 2)
    E_mbh_going_in[:] = np.repeat(mbh_inj[:nwalkers_pe, 1], 2, axis=0).reshape(nwalkers_pe, 2, A_inj.shape[0]).transpose(1, 0, 2)
    
    # walker_factor = int(nwalkers_pe / mbh_inj[:, 0].shape[0])
    # if walker_factor > 1:
    #     assert float(walker_factor) == float(nwalkers_pe) / float(mbh_inj[:, 0].shape[0])
            
    #     A_mbh_going_in[:] = np.repeat(mbh_inj[:, 0][None, :], repeats=ntemps_pe * walker_factor, axis=0).reshape(ntemps_pe, nwalkers_pe, mbh_inj[:, 0].shape[1])
    #     E_mbh_going_in[:] = np.repeat(mbh_inj[:, 1][None, :], repeats=ntemps_pe * walker_factor, axis=0).reshape(ntemps_pe, nwalkers_pe, mbh_inj[:, 1].shape[1])

    A_going_in[:] -= A_mbh_going_in
    E_going_in[:] -= E_mbh_going_in

    A_psd_in = np.zeros((1, nwalkers_pe, A_inj.shape[0]), dtype=np.float64)
    E_psd_in = np.zeros((1, nwalkers_pe, E_inj.shape[0]), dtype=np.float64)

    # A_psd_in[:] = np.asarray(psd)
    # E_psd_in[:] = np.asarray(psd)
    psds = np.load(fp_psd + ".npy")
    psds[:, :, 0] = psds[:, :, 1]

    A_psd_in[:] = np.repeat(psds[:nwalkers_pe, 0], 1, axis=0).reshape(nwalkers_pe, 1, A_inj.shape[0]).transpose(1, 0, 2)
    E_psd_in[:] = np.repeat(psds[:nwalkers_pe, 1], 1, axis=0).reshape(nwalkers_pe, 1, A_inj.shape[0]).transpose(1, 0, 2)
    
    # if walker_factor > 1:
    #     A_psd_in[:] = np.repeat(psds[:, 0][None, :], repeats=ntemps_pe * walker_factor, axis=0).reshape(ntemps_pe, nwalkers_pe, psds[:, 0].shape[1])  # A
    #     E_psd_in[:] = np.repeat(psds[:, 1][None, :], repeats=ntemps_pe * walker_factor, axis=0).reshape(ntemps_pe, nwalkers_pe, psds[:, 1].shape[1])  # E

    """try:
        del mgh
        del sampler_mix
        del supps
        del state_mix
        main_gpu = xp.cuda.runtime.getDevice()
        mempool.free_all_blocks()
        for gpu in gpus:
            with xp.cuda.device.Device(gpu):
                xp.cuda.runtime.deviceSynchronize()
                mempool_tmp = xp.get_default_memory_pool()
                mempool_tmp.free_all_blocks()
                xp.cuda.runtime.deviceSynchronize()
        
        # back to main GPU
        xp.cuda.runtime.setDevice(main_gpu)
        xp.cuda.runtime.deviceSynchronize()
        
    except NameError:
        pass
    """

    mgh = MultiGPUDataHolder(
        gpus,
        A_going_in,
        E_going_in,
        A_going_in, # store as base
        E_going_in, # store as base
        A_psd_in,
        E_psd_in,
        df,
        base_injections=[A_inj, E_inj],
        base_psd=None,  # [psd.copy(), psd.copy()]
    )

    # psd_params = new_sample.branches["psd"].coords.reshape(-1, new_sample.branches["psd"].shape[-1])

    # foreground_params = new_sample.branches["galfor"].coords.reshape(-1, new_sample.branches["galfor"].shape[-1])

    # mgh.set_psd_vals(psd_params, foreground_params=foreground_params)

    gb.d_d = xp.asarray(mgh.get_inner_product().flatten())
    check = mgh.get_psd_term()

    mempool.free_all_blocks()

    # setup data streams to add to and subtract from
    supps_shape_in = (ntemps_pe, nwalkers_pe)

    gb.gpus = mgh.gpus

    """waveform_kwargs_in = {'dt': 5.0, 'T': 31457280.0, 'use_c_implementation': True, 'oversample': 4, 'start_freq_ind': 0}
    data_index = xp.asarray(np.load("data_index.npy")).astype(xp.int32)
    noise_index = data_index.copy()
    prior_generated_points_in = np.load("prior_gen_check.npy")
    N_temp = xp.asarray(np.load("N_temp.npy"))

    inds = xp.where(xp.asarray(prior_generated_points_in)[:, 1] > 1e-5)
    inds_cpu = [tmp.get() for tmp in list(inds)]
    gb.d_d = xp.asarray(np.load("gb_d_d.npy"))[inds]
    gb.get_ll(prior_generated_points_in[inds_cpu], mgh.data_list, mgh.psd_list, data_index=data_index[inds], noise_index=noise_index[inds], phase_marginalize=False, data_length=data_length,  data_splits=mgh.gpu_splits,  N=N_temp[inds], **waveform_kwargs_in).shape
    breakpoint()"""

    coords_out_gb_fixed = new_sample.branches["gb_fixed"].coords[0,
        new_sample.branches["gb_fixed"].inds[0]
    ]

    check = priors["gb_fixed"].logpdf(coords_out_gb_fixed)

    if np.any(np.isinf(check)):
        breakpoint()

    coords_out_gb_fixed[:, 3] = coords_out_gb_fixed[:, 3] % (2 * np.pi)
    coords_out_gb_fixed[:, 5] = coords_out_gb_fixed[:, 5] % (1 * np.pi)
    coords_out_gb_fixed[:, 6] = coords_out_gb_fixed[:, 6] % (2 * np.pi)
    # mgh = MultiGPUDataHolder(gpus, data_minus_templates_mix[0].get().reshape(ntemps_pe, nwalkers_pe, -1), data_minus_templates_mix[1].get().reshape(ntemps_pe, nwalkers_pe, -1), A_psd_in, E_psd_in, df)

    # mgh = MultiGPUDataHolder(gpus, A_going_in, E_going_in, A_psd_in, E_psd_in, df)

    coords_in_in = transform_fn.both_transforms(coords_out_gb_fixed)

    band_inds = np.searchsorted(band_edges, coords_in_in[:, 1], side="right") - 1

    odd_even_vals = np.repeat(
        np.arange(ntemps_pe)[:, None], nleaves_max_fix_new * nwalkers_pe, axis=-1
    ).reshape(ntemps_pe, nwalkers_pe, nleaves_max_fix_new)

    walker_vals = np.tile(
        np.arange(nwalkers_pe), (nleaves_max_fix_new, 1)
    ).transpose((1, 0))[new_sample.branches["gb_fixed"].inds[0]]

    data_index_1 = ((band_inds % 2) + 0) * nwalkers_pe + walker_vals

    data_index = xp.asarray(data_index_1).astype(
        xp.int32
    )

    # goes in as -h
    factors = -xp.ones_like(data_index, dtype=xp.float64)

    """gb.d_d = 0.0  # gb.d_d[data_index]

    ll = gb.get_ll(
        coords_in_in,
        mgh.data_list,
        mgh.psd_list,
        data_index=data_index,
        noise_index=data_index.copy(),
        phase_marginalize=False,
        data_length=data_length,
        data_splits=mgh.gpu_splits,
        return_cupy=True,
        **waveform_kwargs,
    )

    breakpoint()"""

    gb.generate_global_template(
        coords_in_in,
        data_index,
        mgh.data_list,
        batch_size=1000,
        data_length=data_length,
        factors=factors,
        data_splits=mgh.gpu_splits,
        **waveform_kwargs,
    )

    del data_index
    del factors
    mempool.free_all_blocks()

    ll = np.tile(mgh.get_ll(include_psd_info=True), (ntemps_pe, 1))

    state_mix = State(
        new_sample.branches_coords,
        inds=new_sample.branches_inds,
        log_like=ll,
        # supplimental=supps,
        betas=new_sample.betas,
        band_info=new_sample.band_info
    )

    state_mix.band_info = new_sample.band_info

    from gbgpu.utils.utility import get_N

    for name in ["gb_fixed"]:
        ntemps_pe, nwalkers_pe, nleaves_max_here, _ = state_mix.branches[name].shape
        if nleaves_max_here == 0:
            continue
        points_start = state_mix.branches[name].coords[state_mix.branches[name].inds]
        N_vals = np.zeros((ntemps_pe, nwalkers_pe, nleaves_max_here), dtype=int)
        points_start_transform = transform_fn.both_transforms(points_start)
        amp_start = points_start_transform[:, 0]
        f0_start = points_start_transform[:, 1]

        N_temp = get_N(
            amp_start, f0_start, waveform_kwargs["T"], waveform_kwargs["oversample"]
        )

        N_vals[state_mix.branches[name].inds] = N_temp
        branch_supp_base_shape = (ntemps_pe, nwalkers_pe, nleaves_max_here)

        # get band information
        band_inds = np.zeros((ntemps_pe, nwalkers_pe, nleaves_max_here), dtype=int)
        
        band_temp = np.searchsorted(band_edges, f0_start, side="right") - 1

        band_inds[state_mix.branches[name].inds] = band_temp

        state_mix.branches[name].branch_supplimental = BranchSupplimental(
            {"N_vals": N_vals, "band_inds": band_inds}, base_shape=branch_supp_base_shape, copy=True
        )

    gpu_priors_in = deepcopy(priors["gb"].priors_in)
    for key, item in gpu_priors_in.items():
        item.use_cupy = True

    gpu_priors = {"gb_fixed": ProbDistContainer(gpu_priors_in, use_cupy=True)}

    gb_kwargs = dict(
        waveform_kwargs=waveform_kwargs,
        parameter_transforms=transform_fn,
        search=False,
        provide_betas=True,
        skip_supp_names_update=["group_move_points"],
        random_seed=random_seed,
        nfriends=nwalkers,
        n_iter_update=30,
        # rj_proposal_distribution=gpu_priors,
        a=1.75,
        use_gpu=True,
    )

    gb_args = (
        gb,
        priors,
        start_freq_ind,
        data_length,
        mgh,
        np.asarray(fd),
        band_edges,
        gpu_priors,
    )

    gb_fixed_move = GBSpecialStretchMove(
        *gb_args,
        **gb_kwargs,
    )

    # add the other

    gb_fixed_move.gb.gpus = gpus

    """point_generator_func_tmp = deepcopy(priors["gb"].priors_in)

    for key, item in point_generator_func_tmp.items():
        item.use_cupy = True

    point_generator_func_tmp[1] = uniform_dist(0.0, 1.0, use_cupy=True)
    point_generator_func_tmp[2] = uniform_dist(0.0, 1.0, use_cupy=True)

    point_generator_func = ProbDistContainer(point_generator_func_tmp, use_cupy=True)

    gb_kwargs_rj = dict(
        waveform_kwargs=waveform_kwargs,
        parameter_transforms=transform_fn,
        search=False,
        provide_betas=True,
        # skip_supp_names_update=["group_move_points"],
        random_seed=10,
        nfriends=nwalkers,
        n_iter_update=20, 
        use_gpu=True
    )

    gb_args_rj = (
        gb,
        priors,
        start_freq_ind,
        data_length,
        mgh,
        np.asarray(fd),
        band_edges,
        gpu_priors
    )

    rj_moves = GBMutlipleTryRJ(
        gb_args_rj,
        gb_kwargs_rj,
        m_chirp_lims,
        [nleaves_max_fix_new],
        [0],
        num_try=int(3e1),
        gibbs_sampling_setup=["gb_fixed"],
        point_generator_func=point_generator_func,
        fix_change=None,
        prevent_swaps=True
    )"""

    with open("gmm_info.pickle", "rb") as fp:
        gmm_info = pickle.load(fp)


    gmm_all = FullGaussianMixtureModel(gb, *gmm_info, use_cupy=True)

    probs_in = {
        (0, 1, 2, 4, 6, 7): gmm_all,
        3: uniform_dist(0.0, 2 * np.pi, use_cupy=True),
        5: uniform_dist(0.0, np.pi, use_cupy=True)
    }
    gen_dist = ProbDistContainer(probs_in, use_cupy=True)

    tmp = gen_dist.rvs(size=1000)

    tmp[::20, 1] = 500.0
    tmp_logpdf = gen_dist.logpdf(tmp)
    
    gb_kwargs_rj = dict(
        waveform_kwargs=waveform_kwargs,
        parameter_transforms=transform_fn,
        search=False,
        provide_betas=True,
        skip_supp_names_update=["group_move_points"],
        random_seed=random_seed,
        nfriends=nwalkers,
        rj_proposal_distribution={"gb_fixed": gen_dist},  # gpu_priors,
        a=1.7,
        use_gpu=True,
    )

    gb_args_rj = (
        gb,
        priors,
        start_freq_ind,
        data_length,
        mgh,
        np.asarray(fd),
        band_edges,
        gpu_priors,
    )

    rj_move_1 = GBSpecialStretchMove(
        *gb_args_rj,
        **gb_kwargs_rj,
    )

    gb_kwargs_rj_2 = dict(
        waveform_kwargs=waveform_kwargs,
        parameter_transforms=transform_fn,
        search=False,
        provide_betas=True,
        skip_supp_names_update=["group_move_points"],
        random_seed=random_seed,
        nfriends=nwalkers,
        rj_proposal_distribution=gpu_priors,
        a=1.7,
        use_gpu=True,
    )

    rj_move_2 = GBSpecialStretchMove(
        *gb_args_rj,
        **gb_kwargs_rj_2,
    )
    rj_moves = [(rj_move_1, 0.2), (rj_move_2, 0.8)]

    for rj_move in rj_moves:
        rj_move[0].gb.gpus = gpus

    moves_in_model = [gb_fixed_move]

    like_mix = BasicResidualMGHLikelihood(mgh)
    branch_names = ["gb_fixed"]

    # TODO: get betas out of state object from old run if there
    if not hasattr(state_mix, "betas") or state_mix.betas is None:
        # betas = make_ladder(20 * 8, Tmax=5000.0, ntemps=ntemps_pe)
        betas = betas  #  1 / snrs_ladder ** 2

    else:
        betas = state_mix.betas

    state_mix.betas = betas
    update = UpdateNewResiduals(
        [A_mbh_going_in, E_mbh_going_in],
        mgh,
        fp_psd + ".npy",
        fp_mbh + ".npy",
        fp_gb,
        (ntemps_pe, nwalkers_pe, len(fd)),
        waveform_kwargs,
    )

    ndims = {"gb_fixed": ndim}
    nleaves_max = {"gb_fixed": nleaves_max_fix_new}
        
    moves = moves_in_model + rj_moves
    backend = HDFBackend(
        fp_pe,
        compression="gzip",
        compression_opts=9,
    )
    
    if not backend.initialized:
        backend.reset(
            nwalkers_pe,
            ndims,
            nleaves_max=nleaves_max,
            ntemps=ntemps_pe,
            branch_names=branch_names,
            nbranches=len(branch_names),
            rj=True,
            moves=None,
            num_bands=len(band_edges) - 1,
            band_edges=band_edges
        )

    sampler_mix = EnsembleSampler(
        nwalkers_pe,
        ndims,  # assumes ndim_max
        like_mix,
        priors,
        tempering_kwargs={"betas": betas, "adaptation_time": 2, "permute": True},
        nbranches=len(branch_names),
        nleaves_max=nleaves_max,
        moves=moves_in_model,
        rj_moves=rj_moves,
        kwargs=None,  # {"start_freq_ind": start_freq_ind, **waveform_kwargs},
        backend=backend,
        vectorize=True,
        periodic=periodic,  # TODO: add periodic to proposals
        branch_names=branch_names,
        update_fn=update,  # stop_converge_mix,
        update_iterations=1,
        provide_groups=True,
        provide_supplimental=True,
        num_repeats_in_model=1,
        track_moves=False
    )

    # equlibrating likelihood check: -4293090.6483655665,

    """out = state_mix
    old_number = nleaves_max_fix_new
    starting_new = True
    print("starting new", nleaves_max_fix_new)

    if ((current_iteration + 1) % 40 == 0 or starting_new) and current_iteration != 0:
        out.branches["gb_fixed"].coords[-1] = out.branches["gb_fixed"].coords[0]
        out.branches["gb_fixed"].inds[-1] = out.branches["gb_fixed"].inds[0]
        out.branches["gb_fixed"].branch_supplimental[-1] = out.branches["gb_fixed"].branch_supplimental[0]
        # state_mix.supplimental[-1] = state_mix.supplimental[0]
        out.log_like[-1] = out.log_like[0]
        try:
            out.log_prior[-1] = out.log_prior[0]
            out.betas[-1] = out.betas[0]
        except TypeError:
            pass

        inds_cold = out.supplimental.holder["overall_inds"][0]
        inds_hot = out.supplimental.holder["overall_inds"][-1]
        main_gpu = xp.cuda.runtime.getDevice()
        for gpu_i, split_now in enumerate(mgh.gpu_splits):
            gpu = mgh.gpus[gpu_i]
            with xp.cuda.device.Device(gpu):
                xp.cuda.runtime.deviceSynchronize()
                for overall_ind_hot, overall_ind_cold in zip(inds_hot, inds_cold):
                    if overall_ind_hot in split_now:
                        ind_keep_hot = np.where(split_now == overall_ind_hot)[0].item()
                        for gpu_i_cold, split_cold in enumerate(mgh.gpu_splits):
                            if overall_ind_cold in split_cold:
                                ind_keep_cold = np.where(split_cold == overall_ind_cold)[0].item()
                                for jjj in range(len(mgh.data_list)):
                                    sliceit_hot = slice(ind_keep_hot * data_length, (ind_keep_hot + 1) * data_length)
                                    sliceit_cold = slice(ind_keep_cold * data_length, (ind_keep_cold + 1) * data_length)
                                    mgh.data_list[jjj][gpu_i][sliceit_hot] = mgh.data_list[jjj][gpu_i_cold][sliceit_cold]

                xp.cuda.runtime.deviceSynchronize()
        xp.cuda.runtime.setDevice(main_gpu)
    """
    nsteps_mix = 1000
    with open(current_save_state_file, "wb") as fp:
        pickle.dump(state_mix, fp, pickle.HIGHEST_PROTOCOL)

    print("Starting mix ll best:", state_mix.log_like.max(axis=-1))
    mempool.free_all_blocks()
    out = sampler_mix.run_mcmc(
        state_mix, nsteps_mix, progress=True, thin_by=25, store=True
    )
    print("ending mix ll best:", out.log_like.max(axis=-1))

    breakpoint()

    # needs to be max from mgh or need to map it
    lp_mgh = mgh.get_ll(use_cpu=True)
    max_ind = np.where(lp_mgh == lp_mgh.max())

    max_temp = max_ind[0][0]
    max_walker = max_ind[1][0]
    overall_ind_best = max_temp * nwalkers_pe + max_walker

    data_minus_templates = []
    for gpu_i, split_now in enumerate(mgh.gpu_splits):
        if overall_ind_best in split_now:
            ind_keep = np.where(split_now == overall_ind_best)[0].item()
            for tmp_data in mgh.data_list:
                with xp.cuda.device.Device(gpus[0]):
                    data_minus_templates.append(
                        tmp_data[gpu_i].reshape(len(split_now), -1)[ind_keep].copy()
                    )

    xp.cuda.runtime.setDevice(gpus[0])

    data_minus_templates = xp.asarray(data_minus_templates)

    np.save(current_residuals_file_iterative_search, data_minus_templates.get())

    save_state = State(
        out.branches_coords,
        inds=out.branches_inds,
        log_like=out.log_like,
        log_prior=out.log_prior,
    )

    if current_save_state_file in os.listdir():
        shutil.copyfile(current_save_state_file, "old_" + current_save_state_file)

    with open(current_save_state_file, "wb") as fp_out:
        pickle.dump(save_state, fp_out, protocol=pickle.HIGHEST_PROTOCOL)

    """# current_start_points = list(out.branches["gb_fixed"].coords[max_ind][out.branches["gb_fixed"].inds[max_ind]])

    # current_found_coords_for_starting_mix = list(current_found_coords_for_starting_mix)
    
    # current_start_points_in = transform_fn.both_transforms(np.asarray(current_start_points))

    gb.d_d = mgh.get_inner_product(use_cpu=True)[real_temp_best, real_walker_best].item()

    _ = gb.get_ll(current_start_points_in, mgh.data_list, mgh.psd_list, data_length=mgh.data_length, data_splits=mgh.gpu_splits, **waveform_kwargs)

    
    # TODO: should check if phase marginalize
    det_snr = (np.abs(gb.d_h) / np.sqrt(gb.h_h.real)).get()
    opt_snr =  (np.sqrt(gb.h_h.real)).get()

    phase_angle = np.angle(gb.d_h)

    try:
        phase_angle = phase_angle.get()
    except AttributeError:
        pass

    tmp_current_start_points = np.asarray(current_start_points)

    tmp_current_start_points[:, 3] -= phase_angle
    current_start_points_in[:, 4] -= phase_angle

    current_start_points = list(tmp_current_start_points)

    current_snrs_search = list(np.array([det_snr, opt_snr]).T)
    # np.save(current_start_points_snr_file, np.asarray(current_snrs_search))
    # np.save(current_start_points_file, np.asarray(current_start_points))"""

    num_binaries_current = 0

    # )
    # if det_snr_finding < snr_break or len(current_snrs_search) > num_bin_pause:
    #    break
    current_iteration += 1


if __name__ == "__main__":
    import argparse

    """parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=int,
                        help='which gpu', required=True)


    args = parser.parse_args()"""

    output = run_gb_pe(4)
