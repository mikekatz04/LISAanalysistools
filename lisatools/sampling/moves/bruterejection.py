# -*- coding: utf-8 -*-

import numpy as np

try:
    import cupy as xp

    gpu_available = True
except ModuleNotFoundError:
    import numpy as xp

    gpu_available = False

from eryn.moves import ReversibleJump
from eryn.prior import PriorContainer
from eryn.utils.utility import groups_from_inds

__all__ = ["PriorGenerate"]


class BruteRejection(ReversibleJump):
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
        **kwargs
    ):

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

        if search_snrs is not None or search_snr_lim is not None:
            if search_snrs is None or search_snr_lim is None:
                raise ValueError(
                    "If entering search_snrs or search_snr_lim, must enter both."
                )
            if search_samples is None:
                raise ValueError(
                    "If providing search_snrs/search_snr_lim, need to provide search_samples."
                )

            assert len(search_samples) == len(search_snrs)

        self.search_samples = search_samples
        self.search_snrs = search_snrs
        self.search_snr_lim = search_snr_lim
        self.search_snr_accept_factor = search_snr_accept_factor
        self.update_with_new_snr_lim(search_snr_lim)
        self.take_max_ll = take_max_ll

        super(BruteRejection, self).__init__(*args, **kwargs)

    def update_with_new_snr_lim(self, search_snr_lim):
        self.search_inds = np.arange(len(self.search_snrs))[
            self.search_snrs > search_snr_lim
        ]
        self.search_snr_lim = search_snr_lim

        assert len(self.search_inds) > self.num_brute

    def get_proposal(self, all_coords, all_inds, all_inds_for_change, random):
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

            if i == 0:
                factors = np.zeros((ntemps, nwalkers))

            # adjust inds

            # adjust deaths from True -> False
            inds_here = tuple(inds_for_change["-1"].T)
            new_inds[name][inds_here] = False

            # factor is -log q()
            current_priors = self.priors[name]
            factors[inds_here[:2]] += -1 * current_priors.logpdf(q[name][inds_here])

            # adjust births from False -> True
            inds_here = tuple(inds_for_change["+1"].T)
            new_inds[name][inds_here] = True

            # add coordinates for new leaves
            current_priors = self.priors[name]
            inds_here = tuple(inds_for_change["+1"].T)

            # it can pile up with low signal binaries at the maximum number (observation so far)
            if len(inds_here[0]) != 0:

                num_inds_change = len(inds_here[0])

                # group everything
                groups = groups_from_inds({name: inds[inds_here[:2]][None, :, :]})[name]
                # TODO: adjust to cupy

                templates = self.xp.zeros(
                    (num_inds_change, 2, self.data_length), dtype=self.xp.complex128
                )  # 2 is channels

                params = coords[inds_here[:2]][inds[inds_here[:2]]]

                if self.parameter_transforms is not None:
                    params_in = self.parameter_transforms.both_transforms(
                        params.copy(), return_transpose=False
                    )

                # TODO fix error with buffer
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
                d_h_d_h = 4 * np.sum((in_vals.conj() * in_vals), axis=(1, 2))

                out_temp = np.zeros((num_inds_change, ndim))
                ll_out = np.zeros((num_inds_change,))
                log_prob_factors = np.zeros((num_inds_change,))
                # TODO: take out of loop later?
                for j in range(num_inds_change):
                    if self.search_samples is not None:
                        # TODO: make replace=True ? in PE
                        inds_drawn = random.choice(
                            self.search_inds, size=self.num_brute, replace=False,
                        )
                        prior_generated_points = self.search_samples[inds_drawn].copy()
                    else:
                        prior_generated_points = current_priors.rvs(size=self.num_brute)
                    data = [
                        (
                            (self.data_list[0] - templates[j, 0].copy())
                            * self.noise_factors[0]
                        ).copy(),
                        (
                            (self.data_list[1] - templates[j, 1].copy())
                            * self.noise_factors[1]
                        ).copy(),
                    ]

                    if self.parameter_transforms is not None:
                        prior_generated_points_in = self.parameter_transforms.both_transforms(
                            prior_generated_points.copy(), return_transpose=True
                        )

                    self.gb.d_d = d_h_d_h[j]

                    phase_marginalize = self.search

                    ll = self.gb.get_ll(
                        prior_generated_points_in,
                        data,
                        self.noise_factors_list,
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

                    if self.use_gpu:
                        try:
                            ll = ll.get()
                        except AttributeError:
                            pass

                    if self.search:
                        phase_maximized_snr = (
                            self.xp.abs(self.gb.d_h) / self.xp.sqrt(self.gb.h_h)
                        ).real.copy()

                        phase_change = self.xp.angle(self.gb.d_h / xp.sqrt(self.gb.h_h))

                        try:
                            phase_maximized_snr = phase_maximized_snr.get()
                            phase_change = phase_change.get()

                        except AttributeError:
                            pass

                        prior_generated_points[:, 3] -= phase_change
                        ll[
                            phase_maximized_snr
                            < self.search_snr_lim * self.search_snr_accept_factor
                        ] = -1e300

                    probs = np.exp(ll - ll.max()) / np.sum(np.exp(ll - ll.max()))
                    if self.take_max_ll:
                        # get max
                        ind_keep = np.argmax(ll)

                    else:
                        # draw based on likelihood
                        if np.any(np.isnan(probs)):
                            breakpoint()
                        ind_keep = np.random.choice(np.arange(self.num_brute), p=probs,)

                    ll_out[j] = ll[ind_keep]
                    out_temp[j] = prior_generated_points[ind_keep].copy()
                    log_prob_factors[j] = np.log(probs[ind_keep])

                q[name][inds_here] = out_temp

                # factor is +log q() for prior
                factors[inds_here[:2]] += +1 * current_priors.logpdf(q[name][inds_here])
                # add factor from likelihood draw
                factors[inds_here[:2]] += +1 * log_prob_factors
                self.last_run_ok = True
                print("GOOD", num_inds_change)
            else:
                self.last_run_ok = False
        return q, new_inds, factors
