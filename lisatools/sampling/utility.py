from multiprocessing.sharedctypes import Value
import os

import numpy as np

try:
    import cupy as xp

except (ImportError, ModuleNotFoundError) as e:
    pass

from eryn.state import State, BranchSupplimental
from eryn.utils.utility import groups_from_inds


class DetermineGBGroups:
    def __init__(self, gb_wave_generator, transform_fn=None, waveform_kwargs={}):
        self.gb_wave_generator = gb_wave_generator
        self.xp = self.gb_wave_generator.xp
        self.transform_fn = transform_fn
        self.waveform_kwargs = waveform_kwargs
            
    def __call__(self, last_sample, name_here, check_temp=0, input_groups=None, input_groups_inds=None, fix_group_count=False, mismatch_lim=0.2, double_check_lim=0.2, start_term="random", waveform_kwargs={}, index_within_group="random"):
        # TODO: mess with mismatch lim setting
        # TODO: some time of mismatch annealing may be useful
        if isinstance(last_sample, State):
            state = last_sample
            coords = state.branches_coords[name_here][check_temp]
            inds = state.branches_inds[name_here][check_temp]
        elif isinstance(last_sample, dict):
            coords = last_sample[name_here][check_temp]["coords"]
            inds = last_sample[name_here][check_temp]["inds"]

        waveform_kwargs = {**self.waveform_kwargs, **waveform_kwargs}
        
        # get coordinates and inds of the temperature you are considering.
        
        nwalkers, nleaves_max, ndim = coords.shape
        if input_groups is None:

            # figure our which walker to start with 
            if start_term == "max":
                start_walker_ind = inds[check_temp].sum(axis=-1).argmax()
            elif start_term == "first":
                start_walker_ind = 0
            elif start_term == "random":
                start_walker_ind = np.random.randint(0, nwalkers)
            else:
                raise ValueError("start_term must be 'max', 'first', or 'random'.")

            # get all the good leaves in this walker
            inds_good = np.where(inds[start_walker_ind])[0]
            groups = []
            groups_inds = []
            
            # set up this information to load the information into the group lists
            for leaf_i, leaf in enumerate(inds_good):
                groups.append([])
                groups_inds.append([])
                groups[leaf_i].append(coords[start_walker_ind, leaf].copy())
                groups_inds[leaf_i].append([start_walker_ind, leaf])
        else:
            # allows us to check groups based on groups we already have
            groups = input_groups
            groups_inds = input_groups_inds

        if len(groups) == 0:
            return [], [], []
        for w in range(coords.shape[0]):

            # we have already loaded this group
            if input_groups is None and w == start_walker_ind:
                continue

            # walker has no binaries 
            if not np.any(inds[w]):
                continue

            # coords in this walker
            coords_here = coords[w][inds[w]]
            inds_for_group_stuff = np.arange(len(inds[w]))[inds[w]]
            nleaves, ndim = coords_here.shape

            
            params_for_test = []
            for group in groups:
                group_params = np.asarray(group)

                if index_within_group == "first":
                    test_walker_ind = 0
                elif index_within_group == "random":
                    test_walker_ind = np.random.randint(0, group_params.shape[0])
                else:
                    raise ValueError("start_term must be 'max', 'first', or 'random'.")

                params_for_test.append(group_params[test_walker_ind])
            params_for_test = np.asarray(params_for_test)
            
            # transform coords
            if self.transform_fn is not None:
                params_for_test_in = self.transform_fn[name_here].both_transforms(params_for_test, return_transpose=False)
                coords_here_in = self.transform_fn[name_here].both_transforms(coords_here, return_transpose=False)
            
            else:
                params_for_test_in = params_for_test.copy()
                coords_here_in = coords_here.copy()

            inds_tmp_test = np.arange(len(params_for_test_in))
            inds_tmp_here = np.arange(len(coords_here_in))
            inds_tmp_test, inds_tmp_here = [tmp.ravel() for tmp in np.meshgrid(inds_tmp_test, inds_tmp_here)]

            params_for_test_in_full = params_for_test_in[inds_tmp_test]
            coords_here_in_full = coords_here_in[inds_tmp_here]
            # build the waveforms at the same time

            df = 1. / waveform_kwargs["T"]
            max_f = 1. / 2 * 1/waveform_kwargs["dt"]
            frqs = self.xp.arange(0.0, max_f, df)
            data_minus_template = self.xp.asarray([
                self.xp.ones_like(frqs, dtype=complex),
                self.xp.ones_like(frqs, dtype=complex)
            ])[None, :, :]
            psd = self.xp.asarray([
                self.xp.ones_like(frqs, dtype=np.float64),
                self.xp.ones_like(frqs, dtype=np.float64)
            ])

            waveform_kwargs_fill = waveform_kwargs.copy()
            waveform_kwargs_fill.pop("start_freq_ind")
 
            # TODO: could use real data and get observed snr for each if needed
            check = self.gb_wave_generator.swap_likelihood_difference(
                params_for_test_in_full,
                coords_here_in_full,
                data_minus_template,
                psd,
                start_freq_ind=0,
                data_index=None,
                noise_index=None,
                **waveform_kwargs_fill,
            )

            numerator = self.gb_wave_generator.add_remove
            norm_here = self.gb_wave_generator.add_add
            norm_for_test = self.gb_wave_generator.remove_remove

            normalized_autocorr = numerator / np.sqrt(norm_here * norm_for_test)
            normalized_against_test = numerator / norm_for_test

            normalized_autocorr = normalized_autocorr.reshape(coords_here_in.shape[0], params_for_test_in.shape[0]).real
            normalized_against_test = normalized_against_test.reshape(coords_here_in.shape[0], params_for_test_in.shape[0]).real
            
            # TODO: do based on Likelihood? make sure on same posterior
            # TODO: add check based on amplitude
            test1 = np.abs(1.0 - normalized_autocorr.real)  # (numerator / norm_for_test[None, :]).real)
            best = test1.argmin(axis=1)
            try:
                best = best.get()
            except AttributeError:
                pass
            best_mismatch = test1[(np.arange(test1.shape[0]), best)]
            check_normalized_against_test = np.abs(1.0 - normalized_against_test[(np.arange(test1.shape[0]), best)])


            f0_here = coords_here[:, 1]
            f0_test = params_for_test[best, 1]

            for leaf in range(nleaves):
                if best_mismatch[leaf] < mismatch_lim and check_normalized_against_test[leaf] < double_check_lim:
                    groups[best[leaf]].append(coords_here[leaf].copy())
                    groups_inds[best[leaf]].append([w, inds_for_group_stuff[leaf]])

                elif not fix_group_count:
                    # this only works for high snr limit
                    groups.append([coords_here[leaf]].copy())
                    groups_inds.append([[w, inds_for_group_stuff[leaf]]])

        group_lens = [len(group) for group in groups]

        return groups, groups_inds, group_lens


class GetLastGBState:
    def __init__(self, gb_wave_generator, transform_fn=None, waveform_kwargs={}):
        self.gb_wave_generator = gb_wave_generator
        self.xp = self.gb_wave_generator.xp
        self.transform_fn = transform_fn
        self.waveform_kwargs = waveform_kwargs

    def __call__(self, mgh, reader, df, supps_base_shape, fix_temp_initial_ind:int=None, fix_temp_inds:list=None, nleaves_max_in=None, waveform_kwargs={}):

        xp.cuda.runtime.setDevice(mgh.gpus[0])

        if fix_temp_initial_ind is not None or fix_temp_inds is not None:
            if fix_temp_initial_ind is None or fix_temp_inds is None:
                raise ValueError("If giving fix_temp_initial_ind or fix_temp_inds, must give both.")

        state = reader.get_last_sample()

        waveform_kwargs = {**self.waveform_kwargs, **waveform_kwargs}
        if "start_freq_ind" not in waveform_kwargs:
            raise ValueError("In get_last_gb_state, waveform_kwargs must include 'start_freq_ind'.")

        #check = reader.get_last_sample()
        ntemps, nwalkers, nleaves_max_old, ndim = state.branches["gb"].shape
        
        #out = get_groups_for_remixing(check, check_temp=0, input_groups=None, input_groups_inds=None, fix_group_count=False, name_here="gb")

        #lengths = []
        #for group in out[0]:
        #    lengths.append(len(group))
        #breakpoint()
        try:
            if fix_temp_initial_ind is not None: 
                for i in fix_temp_inds:
                    if i < fix_temp_initial_ind:
                        raise ValueError("If providing fix_temp_initial_ind and fix_temp_inds, all values in fix_temp_inds must be greater than fix_temp_initial_ind.")

                    state.log_like[i] = state.log_like[fix_temp_initial_ind]
                    state.log_prior[i] = state.log_prior[fix_temp_initial_ind]
                    state.branches_coords["gb"][i] = state.branches_coords["gb"][fix_temp_initial_ind]
                    state.branches_coords["gb"][i] = state.branches_coords["gb"][fix_temp_initial_ind]
                    state.branches_inds["gb"][i] = state.branches_inds["gb"][fix_temp_initial_ind]
                    state.branches_inds["gb"][i] = state.branches_inds["gb"][fix_temp_initial_ind]

            ntemps, nwalkers, nleaves_max_old, ndim = state.branches["gb"].shape
            if nleaves_max_in is None:
                nleaves_max = nleaves_max_old
            else:
                nleaves_max = nleaves_max_in
            if nleaves_max_old <= nleaves_max:
                coords_tmp = np.zeros((ntemps, nwalkers, nleaves_max, ndim))
                coords_tmp[:, :, :nleaves_max_old, :] = state.branches["gb"].coords

                inds_tmp = np.zeros((ntemps, nwalkers, nleaves_max), dtype=bool)
                inds_tmp[:, :, :nleaves_max_old] = state.branches["gb"].inds
                state.branches["gb"].coords = coords_tmp
                state.branches["gb"].inds = inds_tmp
                state.branches["gb"].nleaves_max = nleaves_max
                state.branches["gb"].shape = (ntemps, nwalkers, nleaves_max, ndim)
                
            else:
                raise ValueError("new nleaves_max is less than nleaves_max_old.")

            # add "gb" if there are any
            data_index_in = groups_from_inds({"gb": state.branches_inds["gb"]})["gb"]

            data_index = xp.asarray(mgh.get_mapped_indices(data_index_in)).astype(xp.int32)

            params_add_in = self.transform_fn["gb"].both_transforms(state.branches_coords["gb"][state.branches_inds["gb"]])
            
            # batch_size is ignored if waveform_kwargs["use_c_implementation"] is True
            #  -1 is to do -(-d + h) = d - h  
            mgh.multiply_data(-1.)
            self.gb_wave_generator.generate_global_template(params_add_in, data_index, mgh.data_list, data_length=mgh.data_length, data_splits=mgh.gpu_splits, batch_size=1000, **waveform_kwargs)
            mgh.multiply_data(-1.)
            

        except KeyError:
            # no "gb"
            pass

        data_index_in = groups_from_inds({"gb": state.branches_inds["gb"]})["gb"]
        data_index = xp.asarray(mgh.get_mapped_indices(data_index_in)).astype(xp.int32)

        params_add_in = self.transform_fn["gb"].both_transforms(state.branches_coords["gb"][state.branches_inds["gb"]])
        
        #  -1 is to do -(-d + h) = d - h  
        mgh.multiply_data(-1.)
        self.gb_wave_generator.generate_global_template(params_add_in, data_index, mgh.data_list, data_length=mgh.data_length, data_splits=mgh.gpu_splits, batch_size=1000, **waveform_kwargs)
        mgh.multiply_data(-1.)

        self.gb_wave_generator.d_d = np.asarray(mgh.get_inner_product(use_cpu=True))
        
        state.log_like = -1/2 * self.gb_wave_generator.d_d.real.reshape(ntemps, nwalkers)

        temp_inds = mgh.temp_indices.copy()
        walker_inds = mgh.walker_indices.copy()
        overall_inds = mgh.overall_indices.copy()
            
        supps = BranchSupplimental({ "temp_inds": temp_inds, "walker_inds": walker_inds, "overall_inds": overall_inds,}, obj_contained_shape=supps_base_shape, copy=True)
        state.supplimental = supps

        return state


class HeterodynedUpdate:
    def __init__(self, update_kwargs, set_d_d_zero=False):
        self.update_kwargs = update_kwargs
        self.set_d_d_zero = set_d_d_zero

    def __call__(self, it, sample_state, sampler, **kwargs):

        samples = sample_state.branches_coords["mbh"].reshape(-1, sampler.ndims[0])
        lp_max = sample_state.log_like.argmax()
        best = samples[lp_max]

        lp = sample_state.log_like.flatten()
        sorted = np.argsort(lp)
        inds_best = sorted[-1000:]
        inds_worst = sorted[:1000]

        best_full = sampler.log_like_fn.f.parameter_transforms["mbh"].both_transforms(
            best, copy=True
        )

        sampler.log_like_fn.f.template_model.init_heterodyne_info(
            best_full, **self.update_kwargs
        )

        if self.set_d_d_zero:
            sampler.log_like_fn.f.template_model.reference_d_d = 0.0

        # TODO: make this a general update function in Eryn (?)
        # samples[inds_worst] = samples[inds_best].copy()
        samples = samples.reshape(sampler.ntemps, sampler.nwalkers, 1, sampler.ndims[0])
        logp = sampler.compute_log_prior({"mbh": samples})
        logL, blobs = sampler.compute_log_like({"mbh": samples}, logp=logp)

        sample_state.branches["mbh"].coords = samples
        sample_state.log_like = logL
        sample_state.blobs = blobs

        # sampler.backend.save_step(sample_state, np.full_like(lp, True))
