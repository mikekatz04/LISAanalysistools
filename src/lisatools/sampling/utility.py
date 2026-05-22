"""Helper utilities for LISA sampling: GB grouping, state restoration, updates."""

import os
from multiprocessing.sharedctypes import Value

import numpy as np

try:
    import cupy as xp

except (ImportError, ModuleNotFoundError) as e:
    pass

from eryn.state import BranchSupplemental, State
from eryn.utils.transform import TransformContainer
from eryn.utils.utility import groups_from_inds

from ..utils.utility import asnumpy


class DetermineGBGroups:
    """Group equivalent GB sources across walkers using waveform mismatches.

    Given an ``eryn`` state holding GB parameters from many walkers, walks
    through the walkers and clusters their leaves into "groups" by computing
    pairwise waveform mismatches against representatives drawn from each
    group. Two leaves are merged into the same group if their mismatch (and
    a secondary ``normalized_against_test`` check) falls below the supplied
    thresholds.

    Args:
        gb_wave_generator: GB waveform generator with a
            ``swap_likelihood_difference`` method (typically from
            ``gbgpu``). Provides the array module used internally.
        transform_fn: Optional dictionary of ``TransformContainer`` objects
            keyed by branch name; applied to coordinates before evaluating
            waveforms.
        waveform_kwargs: Default waveform keyword arguments; merged with the
            per-call kwargs.
    """

    def __init__(self, gb_wave_generator, transform_fn=None, waveform_kwargs={}):
        self.gb_wave_generator = gb_wave_generator
        self.xp = self.gb_wave_generator.xp
        self.transform_fn = transform_fn
        self.waveform_kwargs = waveform_kwargs

    def __call__(
        self,
        last_sample,
        name_here,
        check_temp=0,
        input_groups=None,
        input_groups_inds=None,
        fix_group_count=False,
        mismatch_lim=0.2,
        double_check_lim=0.2,
        start_term="random",
        waveform_kwargs={},
        index_within_group="random",
    ):
        """Cluster GB leaves into groups across walkers.

        Args:
            last_sample: Either an ``eryn`` ``State`` or a dict-like sample
                with the same ``branches_coords`` / ``branches_inds`` layout.
            name_here: Branch name to operate on (e.g. ``"gb"``).
            check_temp: Temperature index to use when reading ``last_sample``.
            input_groups: Pre-existing list of groups to extend rather than
                seed from a single walker.
            input_groups_inds: Index list paired with ``input_groups``
                containing ``[walker, leaf]`` pairs for each group entry.
            fix_group_count: If ``True``, do not create new groups; orphan
                leaves are discarded.
            mismatch_lim: Maximum waveform mismatch for a leaf to join an
                existing group.
            double_check_lim: Secondary tolerance on the
                ``add_remove / remove_remove`` ratio used as a sanity check.
            start_term: Strategy for the seed walker when ``input_groups`` is
                ``None``: ``"max"`` (most leaves), ``"first"``, or
                ``"random"``.
            waveform_kwargs: Per-call waveform keyword arguments; merged with
                the constructor defaults.
            index_within_group: Strategy for picking a representative within
                each group: ``"first"`` or ``"random"``.

        Returns:
            Tuple ``(groups, groups_inds, group_lens)`` where each entry of
            ``groups`` is a list of GB coordinate arrays, the matching entry
            of ``groups_inds`` is a list of ``[walker, leaf]`` index pairs,
            and ``group_lens`` lists the size of each group.
        """
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
                params_for_test_in = self.transform_fn[name_here].both_transforms(
                    params_for_test, return_transpose=False
                )
                coords_here_in = self.transform_fn[name_here].both_transforms(
                    coords_here, return_transpose=False
                )

            else:
                params_for_test_in = params_for_test.copy()
                coords_here_in = coords_here.copy()

            inds_tmp_test = np.arange(len(params_for_test_in))
            inds_tmp_here = np.arange(len(coords_here_in))
            inds_tmp_test, inds_tmp_here = [
                tmp.ravel() for tmp in np.meshgrid(inds_tmp_test, inds_tmp_here)
            ]

            params_for_test_in_full = params_for_test_in[inds_tmp_test]
            coords_here_in_full = coords_here_in[inds_tmp_here]
            # build the waveforms at the same time

            df = 1.0 / waveform_kwargs["T"]
            max_f = 1.0 / 2 * 1 / waveform_kwargs["dt"]
            frqs = self.xp.arange(0.0, max_f, df)
            data_minus_template = self.xp.asarray(
                [
                    self.xp.ones_like(frqs, dtype=complex),
                    self.xp.ones_like(frqs, dtype=complex),
                ]
            )[None, :, :]
            psd = self.xp.asarray(
                [
                    self.xp.ones_like(frqs, dtype=np.float64),
                    self.xp.ones_like(frqs, dtype=np.float64),
                ]
            )

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

            normalized_autocorr = normalized_autocorr.reshape(
                coords_here_in.shape[0], params_for_test_in.shape[0]
            ).real
            normalized_against_test = normalized_against_test.reshape(
                coords_here_in.shape[0], params_for_test_in.shape[0]
            ).real

            # TODO: do based on Likelihood? make sure on same posterior
            # TODO: add check based on amplitude
            test1 = np.abs(
                1.0 - normalized_autocorr.real
            )  # (numerator / norm_for_test[None, :]).real)
            best = asnumpy(test1.argmin(axis=1))
            best_mismatch = test1[(np.arange(test1.shape[0]), best)]
            check_normalized_against_test = np.abs(
                1.0 - normalized_against_test[(np.arange(test1.shape[0]), best)]
            )

            f0_here = coords_here[:, 1]
            f0_test = params_for_test[best, 1]

            for leaf in range(nleaves):
                if (
                    best_mismatch[leaf] < mismatch_lim
                    and check_normalized_against_test[leaf] < double_check_lim
                ):
                    groups[best[leaf]].append(coords_here[leaf].copy())
                    groups_inds[best[leaf]].append([w, inds_for_group_stuff[leaf]])

                elif not fix_group_count:
                    # this only works for high snr limit
                    groups.append([coords_here[leaf]].copy())
                    groups_inds.append([[w, inds_for_group_stuff[leaf]]])

        group_lens = [len(group) for group in groups]

        return groups, groups_inds, group_lens


class GetLastGBState:
    """Reload the most recent GB ``eryn`` state and rebuild the GPU residual.

    Used when restarting a galactic-binary search: takes the latest sample
    from a backend reader, optionally copies one temperature into a list of
    other temperatures, resizes the leaf axis to match the requested
    ``nleaves_max``, and reconstructs the residual ``data - template``
    stored in the multi-GPU data holder ``mgh`` so that subsequent sampling
    is consistent with the saved state.

    Args:
        gb_wave_generator: GB waveform generator capable of populating
            global templates on the GPU.
        transform_fn: Mapping from branch names to ``TransformContainer``
            instances applied to coordinates before generating waveforms.
        waveform_kwargs: Default waveform keyword arguments.
    """

    def __init__(self, gb_wave_generator, transform_fn=None, waveform_kwargs={}):
        self.gb_wave_generator = gb_wave_generator
        self.xp = self.gb_wave_generator.xp
        self.transform_fn = transform_fn
        self.waveform_kwargs = waveform_kwargs

    def __call__(
        self,
        mgh,
        reader,
        df,
        supps_base_shape,
        fix_temp_initial_ind: int = None,
        fix_temp_inds: list = None,
        nleaves_max_in=None,
        waveform_kwargs={},
    ):
        """Restore the GB state and rebuild the residual on the GPU.

        Args:
            mgh: Multi-GPU data holder providing
                ``data_list``, ``data_length``, ``gpu_splits``, and the
                temperature / walker index arrays used for supplementals.
            reader: ``eryn`` backend reader exposing ``get_last_sample``.
            df: Frequency spacing used to scale the residual contributions.
            supps_base_shape: Shape passed to
                :class:`eryn.state.BranchSupplemental` for the rebuilt
                supplementals object.
            fix_temp_initial_ind: Optional source temperature index to copy
                into ``fix_temp_inds`` (must be supplied together).
            fix_temp_inds: Optional list of destination temperature indices.
            nleaves_max_in: New maximum-leaves dimension; if larger than the
                stored value, the coordinate / inds arrays are zero-padded.
            waveform_kwargs: Per-call waveform kwargs; merged with the
                constructor defaults. Must contain ``"start_freq_ind"``.

        Returns:
            The reconstructed ``State`` with updated ``log_like`` and
            ``supplimental`` fields.
        """

        xp.cuda.runtime.setDevice(mgh.gpus[0])

        if fix_temp_initial_ind is not None or fix_temp_inds is not None:
            if fix_temp_initial_ind is None or fix_temp_inds is None:
                raise ValueError("If giving fix_temp_initial_ind or fix_temp_inds, must give both.")

        state = reader.get_last_sample()

        waveform_kwargs = {**self.waveform_kwargs, **waveform_kwargs}
        if "start_freq_ind" not in waveform_kwargs:
            raise ValueError("In get_last_gb_state, waveform_kwargs must include 'start_freq_ind'.")

        # check = reader.get_last_sample()
        ntemps, nwalkers, nleaves_max_old, ndim = state.branches["gb"].shape

        # out = get_groups_for_remixing(check, check_temp=0, input_groups=None, input_groups_inds=None, fix_group_count=False, name_here="gb")

        # lengths = []
        # for group in out[0]:
        #    lengths.append(len(group))
        # breakpoint()
        try:
            if fix_temp_initial_ind is not None:
                for i in fix_temp_inds:
                    if i < fix_temp_initial_ind:
                        raise ValueError(
                            "If providing fix_temp_initial_ind and fix_temp_inds, all values in fix_temp_inds must be greater than fix_temp_initial_ind."
                        )

                    state.log_like[i] = state.log_like[fix_temp_initial_ind]
                    state.log_prior[i] = state.log_prior[fix_temp_initial_ind]
                    state.branches_coords["gb"][i] = state.branches_coords["gb"][
                        fix_temp_initial_ind
                    ]
                    state.branches_coords["gb"][i] = state.branches_coords["gb"][
                        fix_temp_initial_ind
                    ]
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

            params_add_in = self.transform_fn["gb"].both_transforms(
                state.branches_coords["gb"][state.branches_inds["gb"]]
            )

            # batch_size is ignored if waveform_kwargs["use_c_implementation"] is True
            #  -1 is to do -(-d + h) = d - h
            mgh.multiply_data(-1.0)
            self.gb_wave_generator.generate_global_template(
                params_add_in,
                data_index,
                mgh.data_list,
                data_length=mgh.data_length,
                data_splits=mgh.gpu_splits,
                batch_size=1000,
                **waveform_kwargs,
            )
            mgh.multiply_data(-1.0)

        except KeyError:
            # no "gb"
            pass

        data_index_in = groups_from_inds({"gb": state.branches_inds["gb"]})["gb"]
        data_index = xp.asarray(mgh.get_mapped_indices(data_index_in)).astype(xp.int32)

        params_add_in = self.transform_fn["gb"].both_transforms(
            state.branches_coords["gb"][state.branches_inds["gb"]]
        )

        #  -1 is to do -(-d + h) = d - h
        mgh.multiply_data(-1.0)
        self.gb_wave_generator.generate_global_template(
            params_add_in,
            data_index,
            mgh.data_list,
            data_length=mgh.data_length,
            data_splits=mgh.gpu_splits,
            batch_size=1000,
            **waveform_kwargs,
        )
        mgh.multiply_data(-1.0)

        self.gb_wave_generator.d_d = np.asarray(mgh.get_inner_product(use_cpu=True))

        state.log_like = -1 / 2 * self.gb_wave_generator.d_d.real.reshape(ntemps, nwalkers)

        temp_inds = mgh.temp_indices.copy()
        walker_inds = mgh.walker_indices.copy()
        overall_inds = mgh.overall_indices.copy()

        supps = BranchSupplemental(
            {
                "temp_inds": temp_inds,
                "walker_inds": walker_inds,
                "overall_inds": overall_inds,
            },
            obj_contained_shape=supps_base_shape,
            copy=True,
        )
        state.supplimental = supps

        return state


class HeterodynedUpdate:
    """Periodic update that re-centers the heterodyne reference for MBH likelihoods.

    Designed to be passed to ``eryn`` as an ``update_fn``. On each call it
    finds the highest-likelihood walker in the current state, calls
    ``init_heterodyne_info`` on the underlying MBH template model with that
    point as the new reference, optionally zeros the model's ``d_d`` term,
    and recomputes log-prior, log-likelihood, and blobs for the existing
    samples so they are consistent with the new heterodyne expansion.

    Args:
        update_kwargs: Keyword arguments forwarded to
            ``template_model.init_heterodyne_info``.
        set_d_d_zero: If ``True``, set ``template_model.reference_d_d = 0``
            after the update.
    """

    def __init__(self, update_kwargs, set_d_d_zero=False):
        self.update_kwargs = update_kwargs
        self.set_d_d_zero = set_d_d_zero

    def __call__(self, it, sample_state, sampler, **kwargs):
        """Re-center the heterodyne and refresh likelihoods on the current state."""

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

        sampler.log_like_fn.f.template_model.init_heterodyne_info(best_full, **self.update_kwargs)

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


def get_psd_transform_container(
    Soms_fill: float = None,
    Sa_fill: float = None,
    n_knots: int = 5,
    freq_min: float = None,
    freq_max: float = None,
) -> TransformContainer:
    """Prepare a :class:`eryn.utils.transform.TransformContainer` for PSD sampling.

    Args:
        Soms_fill: Optical metrology noise level used to fill PSD knots.
        Sa_fill: Test-mass acceleration noise level used to fill PSD knots.
        n_knots: Number of spline knots used to parameterize the PSD.
        freq_min: Minimum frequency (Hz) of the PSD spline.
        freq_max: Maximum frequency (Hz) of the PSD spline.

    Returns:
        Configured ``TransformContainer`` for PSD parameter sampling.
    """
    # TODO/DOCS: function body is currently empty; this docstring describes
    # the intended interface but no transforms are actually constructed yet.
