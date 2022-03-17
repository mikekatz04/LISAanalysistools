# -*- coding: utf-8 -*-

from copy import deepcopy
from multiprocessing.sharedctypes import Value
import numpy as np
import warnings

try:
    import cupy as xp

    gpu_available = True
except ModuleNotFoundError:
    import numpy as xp

    gpu_available = False

from eryn.moves import ReversibleJump
from eryn.prior import PriorContainer
from eryn.utils.utility import groups_from_inds
from .bruterejection import BruteRejection

__all__ = ["GBBruteRejectionRJ"]


class GBBruteRejectionRJ(BruteRejection, ReversibleJump):
    """Generate Revesible-Jump proposals for GBs with brute-force rejection

    Will use gpu if template generator uses GPU.

    Args:
        priors (object): :class:`PriorContainer` object that has ``logpdf``
            and ``rvs`` methods.

    """

    def __init__(
        self,
        gb,
        priors,
        num_brute,
        start_freq_ind,
        data_length,
        data,
        noise_factors,
        *args,
        waveform_kwargs={},
        parameter_transforms=None,
        search=False,
        search_samples=None,
        search_snrs=None,
        search_snr_lim=None,
        search_snr_accept_factor=1.0,
        take_max_ll=False,
        global_template_builder=None,
        point_generator_func=None,
        **kwargs
    ):

        # TODO: make priors optional like special generate function? 
        for key in priors:
            if not isinstance(priors[key], PriorContainer):
                raise ValueError("Priors need to be eryn.priors.PriorContainer object.")
        self.priors = priors
        self.gb = gb

        # use gpu from template generator
        self.use_gpu = gb.use_gpu
        if self.use_gpu:
            self.xp = xp
        else:
            self.xp = np

        self.num_brute = num_brute
        self.start_freq_ind = start_freq_ind
        self.data_length = data_length
        self.waveform_kwargs = waveform_kwargs
        self.parameter_transforms = parameter_transforms
        self.noise_factors = self.xp.asarray(noise_factors).copy()
        self.noise_factors_list = [
            self.xp.asarray(noise_factors_i) for noise_factors_i in noise_factors
        ]
        self.data = self.xp.asarray(data).copy()
        self.data_list = [self.xp.asarray(data_i) for data_i in data]
        self.search = search
        self.global_template_builder = global_template_builder
        self.point_generator_func = point_generator_func

        if search_snrs is not None:
            if search_snr_lim is None:
                search_snr_lim = 0.1

            assert len(search_samples) == len(search_snrs)

        self.search_samples = search_samples
        self.search_snrs = search_snrs
        self.search_snr_lim = search_snr_lim
        self.search_snr_accept_factor = search_snr_accept_factor
        if search_snr_lim is not None:
            self.update_with_new_snr_lim(search_snr_lim)
        self.take_max_ll = take_max_ll

        ReversibleJump.__init__(self, *args, **kwargs)
        BruteRejection.__init__(self, self.num_brute, take_max_ll=take_max_ll)

    def update_with_new_snr_lim(self, search_snr_lim):
        if self.search_snrs is not None:
            self.search_inds = np.arange(len(self.search_snrs))[
                self.search_snrs > search_snr_lim
            ]
        self.search_snr_lim = search_snr_lim

    def special_generate_func(self, coords, nwalkers, current_priors=None, random=None, size:int=1):

        if self.search_samples is not None:
            # TODO: make replace=True ? in PE
            inds_drawn = random.choice(
                self.search_inds, size=(nwalkers * size), replace=False,
            )
            generated_points = self.search_samples[inds_drawn].copy().reshape(nwalkers, size, -1)
            generate_factors = np.zeros((nwalkers, size))

        elif self.point_generator_func is not None:
            generated_points, generate_factors = self.point_generator_func(size=size * nwalkers).reshape(nwalkers, size, -1)
        else:
            if current_priors is None:
                raise ValueError("If generating from the prior, must provide current_priors kwargs.")

            generated_points = current_priors.rvs(size=nwalkers * size)
            generate_factors = current_priors.logpdf(generated_points).reshape(nwalkers, size)
            generated_points = generated_points.reshape(nwalkers, size, -1)

        return generated_points, generate_factors
        
    def special_like_func(self, generated_points, coords, inds, branch_supps=None):
        # group everything

        # GENERATED POINTS MUST BE PASSED IN by reference not copied 
        num_inds_change, nleaves_max, ndim = coords.shape
        num_inds_change_gen, num_brute, ndim_gen = generated_points.shape
        inds_here = inds.copy()
        assert num_inds_change_gen == num_inds_change and ndim == ndim_gen

        if hasattr(self, "inds_turn_off"):
            inds_here[self.inds_turn_off] = False

        # catch 
        templates = self.xp.zeros(
            (num_inds_change, 2, self.data_length), dtype=self.xp.complex128
        )  # 2 is channels
        
        # guard against having no True values in inds_here
        if np.any(inds_here):
            groups = groups_from_inds({"temp": inds_here[None, :, :]})["temp"]
            
            # branch_supps = None
            if branch_supps is not None:
                # TODO fix error with buffer
                branch_supps_in = {}
                for key, value in branch_supps.items():
                    branch_supps_in[key] = value[inds_here]

                self.global_template_builder.generate_global_template(None, groups, templates, branch_supps=branch_supps_in)

            else:
                raise NotImplementedError
                params = coords[inds_here[:2]][inds[inds_here[:2]]]

                if self.parameter_transforms is not None:
                    params_in = self.parameter_transforms.both_transforms(
                        params.copy(), return_transpose=False
                    )
                
                try:
                    self.gb.generate_global_template(
                        params_in,
                        groups,
                        templates,
                        # start_freq_ind=self.start_freq_ind,
                        **{
                            **self.waveform_kwargs,
                            **{
                                "start_freq_ind": self.start_freq_ind
                                - self.gb.shift_ind
                            },
                        },
                    )
                except ValueError:
                    breakpoint()
            
        # data should not be whitened
        in_vals = (templates - self.data[None, :, :]) * self.noise_factors[
            None, :, :
        ]
        self.d_h_d_h = d_h_d_h = 4 * np.sum((in_vals.conj() * in_vals), axis=(1, 2))
        ll_out = np.zeros((num_inds_change, num_brute)).flatten()
        # TODO: take out of loop later?

        phase_marginalize = self.search
        generated_points_here = generated_points.reshape(-1, ndim)
        batch_size = 5
        num_splits = int(np.ceil(float(templates.shape[0]) / float(batch_size)))

        back_d_d = self.gb.d_d.copy()
        for split in range(num_splits):
            # TODO: add waveform to branch_supps

            data = [
                (
                    (self.data_list[0][None, :] - templates[split * batch_size: (split + 1) * batch_size, 0].copy())
                    * self.noise_factors[0][None, :]
                ).copy(),
                (
                    (self.data_list[1][None, :] - templates[split * batch_size: (split + 1) * batch_size, 1].copy())
                    * self.noise_factors[1][None, :]
                ).copy(),
            ]

            current_batch_size = data[0].shape[0]

            self.gb.d_d = self.xp.repeat(d_h_d_h[split * batch_size: (split + 1) * batch_size], self.num_brute)

            data_index = self.xp.repeat(self.xp.arange(current_batch_size, dtype=self.xp.int32), self.num_brute)

            inds_slice = slice(split * batch_size * self.num_brute, (split + 1) * batch_size * self.num_brute)
            
            prior_generated_points = generated_points_here[inds_slice]

            if self.parameter_transforms is not None:
                    prior_generated_points_in = self.parameter_transforms.both_transforms(
                        prior_generated_points.copy(), return_transpose=True
                    )
            try:
                ll = self.gb.get_ll(
                    prior_generated_points_in,
                    data,
                    self.noise_factors_list,
                    data_index=data_index,
                    calc_d_d=False,
                    phase_marginalize=phase_marginalize,
                    **{
                        **self.waveform_kwargs,
                        **{
                            "start_freq_ind": self.start_freq_ind
                            - self.gb.shift_ind
                        },
                    },
                )
            except AssertionError:
                breakpoint()

            if self.use_gpu:
                try:
                    ll = ll.get()
                except AttributeError:
                    pass

            if self.search:
                phase_maximized_snr = (
                    self.xp.abs(self.gb.d_h) / self.xp.sqrt(self.gb.h_h)
                ).real.copy()
                opt_snr = self.xp.sqrt(self.gb.h_h)
                
                phase_change = self.xp.angle(self.xp.asarray(self.gb.non_marg_d_h) / self.xp.sqrt(self.gb.h_h.real))

                try:
                    phase_maximized_snr = phase_maximized_snr.get()
                    phase_change = phase_change.get()
                    opt_snr = opt_snr.get()

                except AttributeError:
                    pass

                if np.any(np.isnan(prior_generated_points)) or np.any(np.isnan(phase_change)):
                    breakpoint()

                # adjust for phase change from maximization

                generated_points_here[inds_slice, 3] = (generated_points_here[inds_slice, 3] - phase_change) % (2 * np.pi)

                ll[
                    (phase_maximized_snr
                    < self.search_snr_lim * 0.95)
                    | (opt_snr
                    < self.search_snr_lim * self.search_snr_accept_factor)
                ] = -1e300

            generated_points[:] = generated_points_here.reshape(generated_points.shape)
            ll_out[inds_slice] = ll.copy()     
        
        # return gb.d_d
        self.gb.d_d = back_d_d.copy()
        return ll_out.reshape(num_inds_change, num_brute)

    def get_proposal(self, all_coords, all_inds, all_inds_for_change, random, supp=None, branch_supps=None):
        """Make a proposal

        Args:
            all_coords (dict): Keys are ``branch_names``. Values are
                np.ndarray[ntemps, nwalkers, nleaves_max, ndim]. These are the curent
                coordinates for all the walkers.
            all_inds (dict): Keys are ``branch_names``. Values are
                np.ndarray[ntemps, nwalkers, nleaves_max]. These are the boolean
                arrays marking which leaves are currently used within each walker.
            all_inds_for_change (dict): Keys are ``branch_names``. Values are
                dictionaries. These dictionaries have keys ``"+1"`` and ``"-1"``,
                indicating waklkers that are adding or removing a leafm respectively.
                The values for these dicts are ``int`` np.ndarray[..., 3]. The "..." indicates
                the number of walkers in all temperatures that fall under either adding
                or removing a leaf. The second dimension, 3, is the indexes into
                the three-dimensional arrays within ``all_inds`` of the specific leaf
                that is being added or removed from those leaves currently considered.
            random (object): Current random state of the sampler.

        Returns:
            tuple: Tuple containing proposal information.
                First entry is the new coordinates as a dictionary with keys
                as ``branch_names`` and values as
                ``double `` np.ndarray[ntemps, nwalkers, nleaves_max, ndim] containing
                proposed coordinates. Second entry is the new ``inds`` array with
                boolean values flipped for added or removed sources. Third entry
                is the factors associated with the
                proposal necessary for detailed balance. This is effectively
                any term in the detailed balance fraction. +log of factors if
                in the numerator. -log of factors if in the denominator.

        """
        q = {}
        new_inds = {}
        factors = {}

        for i, (name, coords, inds, inds_for_change) in enumerate(
            zip(
                all_coords.keys(),
                all_coords.values(),
                all_inds.values(),
                all_inds_for_change.values(),
            )
        ):
        
            ntemps, nwalkers, nleaves_max, ndim = coords.shape
            new_inds[name] = inds.copy()
            q[name] = coords.copy()

            if i > 0:
                raise NotImplementedError

            if i == 0:
                factors = np.zeros((ntemps, nwalkers))

            # adjust inds

            # adjust deaths from True -> False
            inds_here = tuple(inds_for_change["-1"].T)
            new_inds[name][inds_here] = False

            # factor is +log q()
            current_priors = self.priors[name]
            factors[inds_here[:2]] += +1 * current_priors.logpdf(q[name][inds_here])

            # adjust births from False -> True
            inds_here = tuple(inds_for_change["+1"].T)
            new_inds[name][inds_here] = True

            # add coordinates for new leaves
            current_priors = self.priors[name]
            inds_here = tuple(inds_for_change["+1"].T)
            
            # it can pile up with low signal binaries at the maximum number (observation so far)
            if len(inds_here[0]) != 0:
                if self.search and self.search_samples is not None:
                    assert len(self.search_inds) >= self.num_brute

                num_inds_change = len(inds_here[0])
                generate_points_out, ll_out, factors_out = self.get_bf_proposal(
                    coords[inds_here],
                    len(inds_here[0]), 
                    random, 
                    kwargs_generate={"current_priors": current_priors}, 
                    args_like=(coords[inds_here[:2]], inds[inds_here[:2]]), 
                    kwargs_like={"branch_supps": branch_supps[name][inds_here[:2]]}
                )
                q[name][inds_here] = generate_points_out.copy()

                # TODO: make sure detailed balance this will move to detailed balance in brute rejection
                factors[inds_here[:2]] = factors_out

        return q, new_inds, factors
