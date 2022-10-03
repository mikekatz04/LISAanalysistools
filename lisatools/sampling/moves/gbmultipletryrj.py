# -*- coding: utf-8 -*-

from copy import deepcopy
from inspect import Attribute
from multiprocessing.sharedctypes import Value
import numpy as np
import warnings

try:
    import cupy as xp

    gpu_available = True
except ModuleNotFoundError:
    import numpy as xp

    gpu_available = False

from eryn.state import State, BranchSupplimental
from eryn.moves import ReversibleJump, MultipleTryMove
from eryn.prior import PriorContainer
from eryn.utils.utility import groups_from_inds

__all__ = ["GBMutlipleTryRJ"]


class GBMutlipleTryRJ(MultipleTryMove, ReversibleJump):
    """Generate Revesible-Jump proposals for GBs with multiple try

    Will use gpu if template generator uses GPU.

    Args:
        priors (object): :class:`PriorContainer` object that has ``logpdf``
            and ``rvs`` methods.

    """

    def __init__(
        self,
        gb,
        priors,
        num_try,
        start_freq_ind,
        data_length,
        data,
        psd,
        fd,
        *args,
        waveform_kwargs={},
        noise_kwargs={},
        parameter_transforms=None,
        search=False,
        search_samples=None,
        search_snrs=None,
        search_snr_lim=None,
        search_snr_accept_factor=1.0,
        take_max_ll=False,
        global_template_builder=None,
        point_generator_func=None,
        psd_func=None,
        provide_betas=False,
        alternate_priors=None,
        use_recent_search_samples=False,
        batch_size=5,
        **kwargs
    ):
    
        self.accepted_points = None
        self.time = 0
        self.is_rj = True
        # TODO: make priors optional like special generate function? 
        for key in priors:
            if not isinstance(priors[key], PriorContainer):
                raise ValueError("Priors need to be eryn.priors.PriorContainer object.")
        self.priors = priors
        self.gb = gb
        self.provide_betas = provide_betas
        self.batch_size = batch_size
        self.use_recent_search_samples = use_recent_search_samples

        # use gpu from template generator
        self.use_gpu = gb.use_gpu
        if self.use_gpu:
            self.xp = xp
        else:
            self.xp = np

        self.num_try = num_try
        self.start_freq_ind = start_freq_ind
        self.data_length = data_length
        self.waveform_kwargs = waveform_kwargs
        self.noise_kwargs = noise_kwargs
        self.parameter_transforms = parameter_transforms
        self.psd = psd
        self.psd_func = psd_func
        self.fd = fd
        self.df = (fd[1] - fd[0]).item()
        self.data = data
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
        MultipleTryMove.__init__(self, self.num_try, take_max_ll=take_max_ll)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = self.xp.asarray(data).copy()
        self.data_list = [self.xp.asarray(data_i) for data_i in data]
        return

    @property
    def psd(self):
        return self._psd

    @psd.setter
    def psd(self, psd):
        self._psd = self.xp.asarray(psd).copy()
        self.psd_list = [self.xp.asarray(psd_i) for psd_i in psd]
        return

    def update_with_new_snr_lim(self, search_snr_lim):
        if self.search_snrs is not None:
            self.search_inds = np.arange(len(self.search_snrs))[
                self.search_snrs > search_snr_lim
            ]
        self.search_snr_lim = search_snr_lim

    def special_generate_func(self, coords, nwalkers, current_priors=None, random=None, size:int=1, fill=None, fill_inds=None):
        if self.search_samples is not None:
            # TODO: make replace=True ? in PE
            inds_drawn = np.array([random.choice(
                self.search_inds, size=size, replace=False,
            ) for w in range(nwalkers)])
            generated_points = self.search_samples[inds_drawn].copy()  # .reshape(nwalkers, size, -1)
            # since this is in search only, we will pretend these are coming from the prior
            # so that they are accepted based on the Likelihood (and not the posterior)
            generate_factors = current_priors.logpdf(generated_points.reshape(nwalkers * size, -1)).reshape(nwalkers, size)

        elif self.point_generator_func is not None:
            generated_points, generate_factors = self.point_generator_func(size=size * nwalkers)

            generated_points = generated_points.reshape(nwalkers, size, -1)
            generate_factors = generate_factors.reshape(nwalkers, size)

        else:
            if current_priors is None:
                raise ValueError("If generating from the prior, must provide current_priors kwargs.")

            generated_points = current_priors.rvs(size=nwalkers * size).reshape(nwalkers, size, -1)

            # fill before getting logpdf
            if fill is not None or fill_inds is not None:
                if fill is None or fill_inds is None:
                    raise ValueError("If providing fill_inds or fill, must provide both.")
                generated_points[fill_inds] = fill.copy()

            generate_factors = current_priors.logpdf(generated_points.reshape(nwalkers * size, -1)).reshape(nwalkers, size)

        return generated_points, generate_factors

    def special_like_func(self, generated_points, coords, inds, supps=None, branch_supps=None, noise_params=None, inds_reverse=None):
        # group everything

        # GENERATED POINTS MUST BE PASSED IN by reference not copied 
        num_inds_change, nleaves_max, ndim = coords.shape
        num_inds_change_gen, num_try, ndim_gen = generated_points.shape
        inds_here = inds.copy()
        assert num_inds_change_gen == num_inds_change and ndim == ndim_gen

        if hasattr(self, "inds_turn_off"):
            inds_here[self.inds_turn_off] = False

        # catch 
        templates = self.xp.zeros(
            (num_inds_change, 2, self.data_length), dtype=self.xp.complex128
        )  # 2 is channels
        
        # guard against having no True values in inds_here
        if False:  # np.any(inds_here):
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
                            },
                        },
                    )
                except ValueError:
                    breakpoint()
            
        # data should not be whitened
        if noise_params is None:
            use_stock_psd = True
            psd = self.psd[None, :, :]   

        else:
            use_stock_psd = False
            if self.psd_func is None:
                raise ValueError("When providing noise_params, psd_func kwargs in __init__ function must be given.")

            if noise_params.ndim == 3:
                noise_params = noise_params[0]
            try:
                tmp = self.xp.asarray([self.psd_func(self.fd, *noise_params, **self.noise_kwargs) for _ in range(2)])
                psd = tmp.transpose((1,0,2))
            except ValueError:
                breakpoint()
        in_vals =  supps["data_minus_template"] # (templates - self.data[None, :, :])
        self.d_h_d_h = d_h_d_h = 4 * self.df * self.xp.sum((in_vals.conj() * in_vals) / psd, axis=(1, 2))

        ll_out = np.zeros((num_inds_change, num_try)).flatten()
        # TODO: take out of loop later?

        phase_marginalize = self.search
        generated_points_here = generated_points.reshape(-1, ndim)
        
        if self.batch_size > 0:
            num_splits = int(np.ceil(float(templates.shape[0]) / float(self.batch_size)))
            batch_size = self.batch_size
        else:
            num_splits = 1
            batch_size = num_inds_change
            
        back_d_d = self.gb.d_d.copy()
        for split in range(num_splits):
            # TODO: add waveform to branch_supps

            data = [
                (
                    (in_vals[split * batch_size: (split + 1) * batch_size, 0].copy())
                ).copy(),
                (
                    (in_vals[split * batch_size: (split + 1) * batch_size, 1].copy())
                ).copy(),
            ]

            current_batch_size = data[0].shape[0]

            self.gb.d_d = self.xp.repeat(d_h_d_h[split * batch_size: (split + 1) * batch_size], self.num_try)

            data_index = self.xp.repeat(self.xp.arange(current_batch_size, dtype=self.xp.int32), self.num_try)

            if use_stock_psd:
                psd_in = list(psd[0])
                noise_index = self.xp.zeros(self.num_try * current_batch_size, dtype=self.xp.int32)
            
            else:
                # moves channel to outside
                tmp = psd.transpose((1, 0, 2))
                psd_in = [tmp_i.copy() for tmp_i in tmp]
                noise_index = self.xp.repeat(self.xp.arange(current_batch_size, dtype=self.xp.int32), self.num_try)

            inds_slice = slice(split * batch_size * self.num_try, (split + 1) * batch_size * self.num_try)

            # TODO: Remove batch_size if GPU only ?
            prior_generated_points = generated_points_here[inds_slice]

            if self.parameter_transforms is not None:
                    prior_generated_points_in = self.parameter_transforms.both_transforms(
                        prior_generated_points.copy(), return_transpose=True
                    )

            try:
                ll = self.gb.get_ll(
                    prior_generated_points_in,
                    data,
                    psd_in,
                    data_index=data_index,
                    noise_index=noise_index,
                    phase_marginalize=phase_marginalize,
                    **{
                        **self.waveform_kwargs,
                        **{
                            "start_freq_ind": self.start_freq_ind
                        },
                    },
                )
            except AssertionError:
                breakpoint()

            """ll_check = self.gb.get_ll(
                prior_generated_points_in,
                data,
                psd_in,
                data_index=data_index,
                noise_index=noise_index,
                phase_marginalize=phase_marginalize,
                **{
                    **self.waveform_kwargs,
                    **{
                        "start_freq_ind": self.start_freq_ind,
                        "use_c_implementation": False,
                    },
                },
            )
            """
            if self.use_gpu:
                try:
                    ll = ll.get()
                except AttributeError:
                    pass

            opt_snr = self.xp.sqrt(self.gb.h_h)

            if self.search:
                phase_maximized_snr = (
                    self.xp.abs(self.gb.d_h) / self.xp.sqrt(self.gb.h_h)
                ).real.copy()
                
                
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

                snr_comp = phase_maximized_snr

            else:
                snr_comp = (
                    self.gb.d_h.real / self.xp.sqrt(self.gb.h_h)
                ).real.copy()
                try:
                    snr_comp = snr_comp.get()
                    opt_snr = opt_snr.get()

                except AttributeError:
                    pass

            ##print(opt_snr[snr_comp.argmax()].real, snr_comp.max(), ll[snr_comp.argmax()].real - -1/2 * self.gb.d_d[snr_comp.argmax()].real)
            if self.search and self.search_snr_lim is not None:
                ll[
                    (snr_comp
                    < self.search_snr_lim * 0.8)
                    | (opt_snr
                    < self.search_snr_lim * self.search_snr_accept_factor)
                ] = -1e300
                """if np.any(~((snr_comp
                    < self.search_snr_lim * 0.95)
                    | (opt_snr
                    < self.search_snr_lim * self.search_snr_accept_factor))):
                    breakpoint()"""

            generated_points[:] = generated_points_here.reshape(generated_points.shape)
            ll_out[inds_slice] = ll.copy()  
            #if inds_reverse is not None and len(inds_reverse) != 0 and split == num_splits - 1:
            #    breakpoint()  

        if inds_reverse is not None:
            try:
                tmp_d_h_d_h = d_h_d_h.get()
            except AttributeError:
                tmp_d_h_d_h = d_h_d_h

            self.special_aux_ll = (-1./2. * tmp_d_h_d_h[inds_reverse]).real + self.noise_ll[inds_reverse]

        #breakpoint()
        # return gb.d_d
        self.gb.d_d = back_d_d.copy()
        # add noise term
        return ll_out.reshape(num_inds_change, num_try) + self.noise_ll[:, None]

    def special_prior_func(self, generated_points, coords, inds, **kwargs):
        nwalkers, nleaves_max, ndim = coords.shape
    
        lp_new = self.priors["gb"].logpdf(generated_points.reshape(-1, 8)).reshape(nwalkers, self.num_try)
        lp_total = self.lp_old[:, None] + lp_new
        # add noise lp
        return lp_total

    def get_proposal(self, all_coords, all_inds, all_inds_for_change, random, supps=None, branch_supps=None):
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
            tmp_inds = inds.copy()
            q[name] = coords.copy()

            if name != "gb":
                continue

            if i == 0:
                factors = np.zeros((ntemps, nwalkers))

            # adjust inds

            # adjust deaths from True -> False
            inds_reverse = tuple(inds_for_change["-1"].T)

            inds_reverse_in = np.arange(len(inds_reverse[0]))
            inds_reverse_individual = inds_for_change["-1"][:, 2]

            new_inds[name][inds_reverse] = False

            # change for going into the proposal, need to pretend this one is not there
            # do not need to do this for sources where it is added
            tmp_inds[inds_reverse] = False


            # adjust births from False -> True
            inds_forward = tuple(inds_for_change["+1"].T)
            new_inds[name][inds_forward] = True

            num_forward = len(inds_forward[0])

            inds_reverse_in += num_forward

            # keep tmp_inds without added value here

            # add coordinates for new leaves
            current_priors = self.priors[name]
            inds_here = tuple(np.concatenate([inds_for_change["+1"], inds_for_change["-1"]], axis=0).T)
            
            # allows for adjustment during search
            reset_num_try = False
            # it can pile up with low signal binaries at the maximum number (observation so far)
            if len(inds_here[0]) != 0:
                if self.search and self.search_samples is not None:
                    if len(self.search_inds) >= self.num_try:
                        reset_num_try = True
                        old_num_try = self.num_try
                    self.num_try = len(self.search_inds)

                num_inds_change = len(inds_here[0])

                if self.provide_betas:  #  and not self.search:
                    betas = self.temperature_control.betas[inds_here[0]]
                else:
                    betas = None

                ll_here = self.current_state.log_prob.copy()[inds_here[:2]]
                lp_here = self.current_state.log_prior[inds_here[:2]]

                if "noise_params" in all_coords:
                    noise_params_here = all_coords["noise_params"][all_inds["noise_params"]].reshape(ntemps, nwalkers, -1)[inds_here[:2]]

                    if self.psd_func is None:
                        raise ValueError("If providing noise_params, need to provide psd_func to __init__ function.")
                    
                    psd = self.xp.asarray([self.psd_func(self.fd, *noise_params_here.T, **self.noise_kwargs) for _ in range(2)]).transpose((1, 0, 2))
                    noise_ll = -self.xp.sum(self.xp.log(psd), axis=(1, 2))
                    
                    try:
                        noise_ll = noise_ll.get()
                    except AttributeError:
                        pass
                
                    self.noise_ll = noise_ll.copy()
                    
                    # TODO: check this
                    #ll_here -= noise_ll

                    noise_lp = self.priors["noise_params"].logpdf(noise_params_here)
                    self.noise_lp = noise_lp.copy()
                    #lp_here -= noise_lp

                self.lp_old = lp_here

                rj_info = dict(
                    ll=ll_here,
                    lp=lp_here
                )
                
                self.inds_reverse = inds_reverse
                self.inds_forward = inds_forward
                new_supps = deepcopy(supps)

                # remove templates from new copy of residuals
                # that are proposed to be removed. 
                # This is so that the pretend forward computation is legit. 
                data_minus_template = new_supps.holder["data_minus_template"]

                if len(inds_reverse[0]) > 0:
                    parameters_remove = self.parameter_transforms.both_transforms(all_coords["gb"][inds_reverse])

                    templates_to_remove = self.xp.zeros((parameters_remove.shape[0],) + data_minus_template.shape[2:], dtype=complex)

                    self.gb.generate_global_template(
                        parameters_remove,
                        self.xp.arange(parameters_remove.shape[0], dtype=self.xp.int32), 
                        templates_to_remove, 
                        **self.waveform_kwargs
                    )

                    self.checkit3 =  (-1/2 * self.df * 4 * self.xp.sum(data_minus_template.conj() * data_minus_template / self.psd, axis=(2,3))) + self.xp.asarray(self.noise_ll).reshape(ntemps, nwalkers)
                    # this will update it in new_supps for when it is passed through MT
                    # add new binaries to data d - (h + add) = d - h - add
                    # remove removed binaries: d - (h - remove) = d - h + remove
                    data_minus_template[inds_reverse[0], inds_reverse[1]] += templates_to_remove

                    self.checkit4 =  (-1/2 * self.df * 4 * self.xp.sum(data_minus_template.conj() * data_minus_template / self.psd, axis=(2,3))) + self.xp.asarray(self.noise_ll).reshape(ntemps, nwalkers)

                coords_inds = (coords[inds_here[:2]], tmp_inds[inds_here[:2]])

                generate_points_out, ll_out, factors_out = self.get_bf_proposal(
                    coords[inds_here],
                    len(inds_here[0]), 
                    inds_reverse_in,
                    inds_reverse_individual,
                    random, 
                    args_prior=coords_inds,
                    kwargs_generate={"current_priors": current_priors}, 
                    args_like=coords_inds, 
                    kwargs_like={"supps": new_supps[inds_here[:2]], "branch_supps": branch_supps[name][inds_here[:2]], "noise_params": all_coords["noise_params"][inds_here[:2]].T,
                    "inds_reverse": inds_reverse_in},
                    betas=betas,
                    rj_info=rj_info,
                )

                q[name][inds_forward] = generate_points_out.copy()

                # TODO: make sure detailed balance this will move to detailed balance in multiple try
                factors[inds_here[:2]] = factors_out

                self.ll_for_transfer = np.zeros_like(self.current_state.log_prob)
                self.lp_for_transfer = np.zeros_like(self.current_state.log_prob)
                ll_tmp = self.ll_out.copy()
                lp_tmp = self.lp_out.copy()
                self.lp_for_transfer[inds_here[:2]] = self.lp_out
                self.ll_for_transfer[inds_here[:2]] = self.ll_out
                #if "noise_params" in all_coords:
                #    ll_tmp += noise_ll
                #    lp_tmp += noise_lp

                if np.any(self.ll_for_transfer > 1e7):
                    breakpoint()
            
            else:
                breakpoint()
        
        if reset_num_try:
            self.num_try = old_num_try

        return q, new_inds, factors

    def propose(self, model, state):
        """Use the move to generate a proposal and compute the acceptance

        Args:
            model (:class:`eryn.model.Model`): Carrier of sampler information.
            state (:class:`State`): Current state of the sampler.

        Returns:
            :class:`State`: State of sampler after proposal is complete.

        """
        import time
        # st = time.perf_counter()
        # TODO: keep this?
        # this exposes anywhere in the proposal class to this information
        self.current_state = state
        self.current_model = model

        # Run any move-specific setup.
        self.setup(state.branches)

        # ll_before = model.compute_log_prob_fn(state.branches_coords, inds=state.branches_inds, supps=state.supplimental, branch_supps=state.branches_supplimental)

        # if not np.allclose(ll_before[0], state.log_prob):
        #    breakpoint()

        ntemps, nwalkers, _, _ = state.branches[list(state.branches.keys())[0]].shape

        accepted = np.zeros((ntemps, nwalkers), dtype=bool)

        # TODO: do we want an probability that the model count will not change?
        inds_for_change = self.get_model_change_proposal(state, model)

        coords_propose_in = state.branches_coords
        inds_propose_in = state.branches_inds
        branches_supp_propose_in = state.branches_supplimental

        if len(list(coords_propose_in.keys())) == 0:
            raise ValueError("Right now, no models are getting a reversible jump proposal. Check min_k and max_k or do not use rj proposal.")
        #et = time.perf_counter()
        #print("start", et - st)
        # st = time.perf_counter()
        # propose new sources and coordinates
        q, new_inds, factors = self.get_proposal(
            coords_propose_in, inds_propose_in, inds_for_change, model.random, branch_supps=branches_supp_propose_in, supps=state.supplimental
        )
        #et = time.perf_counter()
        #print("proposal", et - st)
        # st = time.perf_counter()
        
        # TODO: check this
        edge_factors = np.zeros((ntemps, nwalkers))
        # get factors for edges
        for (name, branch), min_k, max_k in zip(
            state.branches.items(), self.min_k, self.max_k
        ):
            nleaves = branch.nleaves

            # do not work on sources with fixed source count
            if min_k == max_k:
                continue

            # fix proposal asymmetry at bottom of k range
            inds_min = np.where(nleaves == min_k)
            # numerator term so +ln
            edge_factors[inds_min] += np.log(1 / 2.0)

            # fix proposal asymmetry at top of k range
            inds_max = np.where(nleaves == max_k)
            # numerator term so -ln
            edge_factors[inds_max] += np.log(1 / 2.0)

            # fix proposal asymmetry at bottom of k range (kmin + 1)
            inds_min = np.where(nleaves == min_k + 1)
            # numerator term so +ln
            edge_factors[inds_min] -= np.log(1 / 2.0)

            # fix proposal asymmetry at top of k range (kmax - 1)
            inds_max = np.where(nleaves == max_k - 1)
            # numerator term so -ln
            edge_factors[inds_max] -= np.log(1 / 2.0)

        factors += edge_factors
        #et = time.perf_counter()
        #print("edge factors", et - st)
        # st = time.perf_counter()
        for name, branch in state.branches.items():
            if name not in q:
                q[name] = state.branches[name].coords[:].copy()
            if name not in new_inds:
                new_inds[name] = state.branches[name].inds[:].copy()

        #et = time.perf_counter()
        #print("coords copy", et - st)
        # st = time.perf_counter()
        # TODO: change to getting the change in prior
        logp = model.compute_log_prior_fn(q, inds=new_inds)
        #et = time.perf_counter()
        #print("prior", et - st)
        # st = time.perf_counter()
        logl = self.ll_for_transfer.reshape(ntemps, nwalkers)
        #loglcheck, new_blobs = model.compute_log_prob_fn(q, inds=new_inds, logp=logp, supps=new_supps, branch_supps=new_branch_supps)
        #if not np.all(np.abs(logl[logl != -1e300] - loglcheck[logl != -1e300]) < 1e-5):
        #    breakpoint()
        
        logP = self.compute_log_posterior(logl, logp)

        prev_logl = state.log_prob

        prev_logp = state.log_prior

        # TODO: check about prior = - inf
        # takes care of tempering
        prev_logP = self.compute_log_posterior(prev_logl, prev_logp)

        # TODO: fix this
        # this is where _metropolisk should come in
        lnpdiff = factors + logP - prev_logP

        accepted = lnpdiff > np.log(model.random.rand(ntemps, nwalkers))
        #et = time.perf_counter()
        #print("through accepted", et - st)
        # st = time.perf_counter()
        if np.any(accepted):
            wk_temp = self.waveform_kwargs.copy()
            if "start_freq_ind" not in wk_temp:
                wk_temp["start_freq_ind"] = self.start_freq_ind

            data_minus_template = state.supplimental.holder["data_minus_template"]
            inds_forward = tuple(inds_for_change["gb"]["+1"].T)
            info_forward = (inds_forward[0] * 1e6 + inds_forward[1]).astype(int)

            inds_reverse = tuple(inds_for_change["gb"]["-1"].T)
            info_reverse = (inds_reverse[0] * 1e6 + inds_reverse[1]).astype(int)
            
            temps_tmp = np.repeat(np.arange(ntemps)[:, None], nwalkers, axis=-1)[accepted]
            walkers_tmp = np.tile(np.arange(nwalkers), (ntemps, 1))[accepted]
            accepted_info = (temps_tmp * 1e6 + walkers_tmp).astype(int)

            forward_accepted = np.in1d(info_forward, accepted_info)
            reverse_accepted = np.in1d(info_reverse, accepted_info)

            tmp_bac = data_minus_template.copy()
            inds_forward_keep = tuple([tmp[forward_accepted] for tmp in list(inds_forward)])
            inds_reverse_keep = tuple([tmp[reverse_accepted] for tmp in list(inds_reverse)])
            params_forward_keep = q["gb"][inds_forward_keep]
            params_reverse_keep = q["gb"][inds_reverse_keep]

            params_for_update_tmp = np.concatenate([params_forward_keep, params_reverse_keep], axis=0)

            # need to order like temps and walkers
            params_for_update = self.parameter_transforms.both_transforms(params_for_update_tmp)
            num_add = params_forward_keep.shape[0]
            num_remove = params_reverse_keep.shape[0]

            templates_to_add = self.xp.zeros((params_for_update.shape[0],) + data_minus_template.shape[2:], dtype=complex)
            try:
                self.gb.generate_global_template(
                    params_for_update,
                    self.xp.arange(params_for_update.shape[0], dtype=self.xp.int32), 
                    templates_to_add, 
                    **self.waveform_kwargs
                )
            except ValueError:
                breakpoint()

            inds_out = (
                np.concatenate([inds_forward_keep[0], inds_reverse_keep[0]]),
                np.concatenate([inds_forward_keep[1], inds_reverse_keep[1]]),
                np.concatenate([inds_forward_keep[2], inds_reverse_keep[2]])
            )
            # reverse
            # add new binaries to data d - (h + add) = d - h - add
            # remove removed binaries: d - (h - remove) = d - h + remove
            templates_to_add[:num_add] *= -1.0

            self.checkit1 =  (-1/2 * self.df * 4 * self.xp.sum(data_minus_template.conj() * data_minus_template / self.psd, axis=(2,3)))  + self.xp.asarray(self.noise_ll).reshape(ntemps, nwalkers)

            # this will update in place which is also the supplimental
            data_minus_template[(inds_out[0], inds_out[1])] += templates_to_add

            # update log prob and log prior
            state.log_prob[(inds_out[0], inds_out[1])] = self.ll_for_transfer[(inds_out[0], inds_out[1])]
            state.log_prior[(inds_out[0], inds_out[1])] = logp[(inds_out[0], inds_out[1])]

            # top three temps
            keep_accepted_points = inds_forward_keep[0] <= 2
            if self.accepted_points is None:
                # TODO: adjust this (keeps everything in the first 3 temperatures.)
                self.accepted_points = params_forward_keep[keep_accepted_points].copy()
            else:
                self.accepted_points = np.append(self.accepted_points, params_forward_keep[keep_accepted_points], axis=0)

            if (self.time % 100) == 0:
                np.save("accepted_rj_points", self.accepted_points)

            state.branches["gb"].coords[inds_forward_keep] = params_forward_keep
            state.branches["gb"].inds[inds_forward_keep] = True
            state.branches["gb"].inds[inds_reverse_keep] = False

            
            self.checkit2 =  (-1/2 * self.df * 4 * self.xp.sum(data_minus_template.conj() * data_minus_template / self.psd, axis=(2,3)))  + self.xp.asarray(self.noise_ll).reshape(ntemps, nwalkers)

            if np.abs(state.log_prob - self.checkit2.get()).max() > 1e-3:
                breakpoint()
            
            #et = time.perf_counter()
            #print("adjusting data", et - st)
            # st = time.perf_counter()
        if self.temperature_control is not None and not self.prevent_swaps:
             state, accepted = self.temperature_control.temper_comps(state, accepted, adapt=False)
        if np.any(state.log_prob > 1e10):
            breakpoint()
        #et = time.perf_counter()
        #print("swapping", et - st)

        self.time += 1
        return state, accepted

