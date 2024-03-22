import warnings
from eryn.state import Branch, BranchSupplimental

import numpy as np

try:
    import cupy as cp

except (ModuleNotFoundError, ImportError):
    pass


class Likelihood(object):
    def __init__(
        self,
        template_model,
        num_channels,
        dt=None,
        df=None,
        f_arr=None,
        parameter_transforms=None,
        use_gpu=False,
        vectorized=False,
        separate_d_h=False,
        return_cupy=False,
        fill_data_noise=False,
        transpose_params=False,
        subset=None,
        adjust_psd=False,
    ):
        self.subset = subset
        self.adjust_psd = adjust_psd
        self.transpose_params = transpose_params
        self.template_model = template_model

        self.parameter_transforms = parameter_transforms

        self.fill_data_noise = fill_data_noise

        self.use_gpu = use_gpu
        self.vectorized = vectorized

        self.num_channels = num_channels

        self.separate_d_h = separate_d_h

        if dt is None and df is None and f_arr is None:
            raise ValueError("Must provide dt, df or f_arr.")

        self.dt, self.df, self.f_arr = dt, df, f_arr

        if df is not None or f_arr is not None:
            self.frequency_domain = True
        else:
            self.frequency_domain = False

        self.return_cupy = return_cupy

        self.noise_has_been_added = False
        self._specific_likelihood_setup()

    def _specific_likelihood_setup(self):
        if isinstance(self.template_model, list):
            raise ValueError("For single likelihood, template model cannot be a list.")
        if hasattr(self.template_model, "get_ll"):
            self.get_ll = self.template_model.get_ll
            self.like_here = False

        else:
            self.fill_data_noise = False
            self.like_here = True

    # TODO: add previously injected signal from a file for example
    # TODO: add SNR scaling, need to read out new distance
    def inject_signal(
        self,
        data_stream=None,
        params=None,
        waveform_kwargs={},
        noise_fn=None,
        noise_args=[],
        noise_kwargs={},
        add_noise=False,
    ):
        xp = cp if self.use_gpu else np

        if params is not None:
            if self.parameter_transforms is not None:
                key = list(self.parameter_transforms.keys())[0]
                params = self.parameter_transforms[key].both_transforms(params)

            injection_channels = xp.asarray(
                self.template_model(*params, **waveform_kwargs)
            )
            try:
                injection_channels = injection_channels.get()

            except AttributeError:
                pass

        elif data_stream is not None:
            if isinstance(data_stream, list) is False:
                raise ValueError("If data_stream is provided, it must be as a list.")
            try:
                injection_channels = xp.asarray(data_stream).get()
            except AttributeError:
                injection_channels = np.asarray(data_stream)

        else:
            raise ValueError(
                "Must provide data_stream or params kwargs to inject signal."
            )

        self.injection_length = len(injection_channels[0])

        for inj in injection_channels:
            if len(inj) != self.injection_length:
                raise ValueError("Length of all injection channels must match.")

        if len(injection_channels) != self.num_channels:
            raise ValueError(
                "Number of channels from template_model does not match number of channels declare by user."
            )

        if isinstance(noise_fn, list):
            if len(noise_fn) != 1 and len(noise_fn) != self.num_channels:
                raise ValueError(
                    "Number of noise functions does not match number of channels declared by user."
                )

            elif len(noise_fn) == 1:
                noise_fn = [noise_fn[0] for _ in range(self.num_channels)]

        else:
            noise_fn = [noise_fn for _ in range(self.num_channels)]

        if isinstance(noise_kwargs, list):
            if len(noise_kwargs) != 1 and len(noise_kwargs) != self.num_channels:
                raise ValueError(
                    "Number of noise kwargs does not match number of channels declared by user."
                )
            elif len(noise_kwargs) == 1:
                noise_kwargs = [noise_kwargs[0] for _ in range(self.num_channels)]

        else:
            noise_kwargs = [noise_kwargs for _ in range(self.num_channels)]

        if isinstance(noise_args, list):
            if len(noise_args) != 1 and len(noise_args) != self.num_channels:
                raise ValueError(
                    "Number of noise args does not match number of channels declared by user."
                )
            elif len(noise_args) == 1:
                noise_args = [noise_args[0] for _ in range(self.num_channels)]

        else:
            noise_args = [noise_args for _ in range(self.num_channels)]

        if self.frequency_domain:
            if self.df is not None and self.f_arr is None:
                df = self.df
                can_add_noise = True
                freqs = np.arange(self.injection_length) * df
                injection_channels = [inj for inj in injection_channels]

            else:
                freqs = self.f_arr
                can_add_noise = False
                self.df = (freqs[1] - freqs[0]).item()

        else:
            dt = self.dt
            freqs = np.fft.rfftfreq(self.injection_length, dt)
            can_add_noise = True
            self.df = df = 1.0 / (self.injection_length * dt)

        if self.frequency_domain is False:
            injection_channels = [np.fft.rfft(inj) * dt for inj in injection_channels]

        if not self.adjust_psd:
            psd = [
                noise_fn_temp(freqs, *noise_args_temp, **noise_kwargs_temp)
                for noise_fn_temp, noise_args_temp, noise_kwargs_temp in zip(
                    noise_fn, noise_args, noise_kwargs
                )
            ]

            if np.isnan(psd[0][0]):
                for i in range(len(psd)):
                    psd[i][0] = 1e100

            diff_freqs = np.zeros_like(freqs)
            diff_freqs[1:] = np.diff(freqs)
            diff_freqs[0] = diff_freqs[1]

            self.base_injections = injection_channels
            if add_noise and can_add_noise and self.noise_has_been_added is False:
                raise NotImplementedError
                norm = 0.5 * (1.0 / df) ** 0.5
                noise_to_add = [
                    psd_temp ** (1 / 2)
                    * (
                        np.random.normal(0, norm, len(freqs))
                        + 1j * np.random.normal(0, norm, len(freqs))
                    )
                    for psd_temp in psd
                ]

                self.noise_to_add = noise_to_add

                injection_channels = [
                    inj + noise for inj, noise in zip(injection_channels, noise_to_add)
                ]

                # TODO: need to check this
                self.noise_likelihood_factor = np.sum(
                    [
                        1.0 / 2.0 * (2 * np.pi) * np.sum(diff_freqs * psd_temp)
                        for psd_temp in psd
                    ]
                )
                self.noise_has_been_added = True

                self.noise_added_base_injections = injection_channels

            # noise weighting
            # injection_channels = [
            #    inj * (diff_freqs / psd_temp) ** (1 / 2)
            #    for inj, psd_temp in zip(injection_channels, psd)
            # ]

            # self.psd = xp.asarray(
            #    [(diff_freqs / psd_temp) ** (1 / 2) for psd_temp in psd]
            # )
            self.psd = xp.asarray(psd)

            # if self.like_here is False:
            #    self.psd = [xp.asarray(nf.copy()) for nf in self.psd]

        # if we need to evaluate the psd each time
        else:
            self.noise_fn, self.noise_args, self.noise_kwargs = (
                noise_fn,
                noise_args,
                noise_kwargs,
            )

        self.freqs = xp.asarray(freqs)

        if hasattr(self, "injection_channels") is False:
            self.injection_channels = xp.asarray(injection_channels)
        else:
            self.injection_channels += xp.asarray(injection_channels)

        if self.like_here is False:
            self.injection_channels = [inj.copy() for inj in self.injection_channels]

        self.data_length = len(self.injection_channels[0])

        self.start_freq_ind = int(self.freqs[0] / self.df)

    def get_ll(self, params, data, psd, *args, **kwargs):
        xp = cp if self.use_gpu else np

        if psd is None:
            psd = self.psd

        if data is None:
            data = self.injection_channels

        # TODO: make sure parameter transformations appear in posterior if possible
        num_likes = params.shape[0]
        if self.vectorized:
            template_channels = xp.asarray(
                self.template_model(*params, *args, **kwargs)
            )

        else:
            if isinstance(params[0], np.float64):
                params = [params]
            template_channels = [None for _ in range(len(params))]
            for i, params_i in enumerate(params):
                # np.save("params_last", params_i)
                template_channels[i] = self.template_model(*params_i, *args, **kwargs)

            template_channels = xp.asarray(template_channels)

            # template_channels = xp.asarray(
            #    [self.template_model(*params_i, *args, **kwargs) for params_i in params]
            # )

        if self.frequency_domain is False:
            template_channels = xp.fft.rfft(template_channels, axis=-1) * self.dt

        if psd.ndim == 2:
            psd = psd[xp.newaxis, :, :]

        h = template_channels
        if self.separate_d_h:
            raise NotImplementedError

        else:
            if data.ndim == 2:
                data = data[xp.newaxis, :, :]

            psd[xp.isnan(psd) | xp.isinf(psd)] = 1e20
            # combines all channels into 1D array per likelihood

            d_minus_h = data - h

            # TODO: add inds_slice to here from global
            # start_ind = 1 if np.isnan(psd[0, 0, 0]) else 0

            ll = -(
                1.0
                / 2.0
                * (
                    4.0
                    * self.df
                    * xp.sum(((d_minus_h.conj() * d_minus_h) / psd).real, axis=(1, 2))
                )
            )

            if self.adjust_psd:
                ll += xp.sum(xp.log(psd), axis=(1, 2))

            if self.noise_has_been_added:
                raise NotImplementedError
                # TODO
                ll += self.noise_likelihood_factor

            out = xp.atleast_1d(ll.squeeze())

        if self.use_gpu:
            if self.return_cupy:
                return out
            else:
                try:
                    return out.get()
                except AttributeError:
                    return out

        else:
            return out

    def evaluate_psd(
        self,
        noise_params,
        f_arr=None,
        noise_fn: list = None,
        noise_kwargs: list = None,
        noise_groups=None,
    ):
        xp = cp if self.use_gpu else np

        if noise_groups is None:
            if len(np.unique(noise_groups)) != len(noise_groups):
                raise ValueError(
                    "If providing noise_groups with adjustable leaf count, need to write custom evaluate_psd function."
                )
        if f_arr is None:
            f_arr = self.freqs

        assert isinstance(f_arr, xp.ndarray)

        if noise_fn is None:
            # must be a list
            noise_fn = self.noise_fn

        if noise_kwargs is None:
            # must be a list
            noise_kwargs = self.noise_kwargs

        psd = xp.asarray(
            [
                noise_fn_temp(f_arr, *noise_params, **noise_kwargs_temp)
                for noise_fn_temp, noise_kwargs_temp in zip(noise_fn, noise_kwargs)
            ]
        ).transpose((1, 0, 2))
        return psd

    def __call__(self, params, data=None, psd=None, *args, **kwargs):
        xp = cp if self.use_gpu else np
        if isinstance(params, list):
            if len(params) != 2:
                ValueError(
                    "If providing params for a single source Likelihood, must be an array if just parameters or a list of length 2 where the first entry in the parameter array and the second entry is the parameterization of the noise curve."
                )

            if not self.adjust_psd:
                raise ValueError(
                    "If providing a list with noise parameters, adjust_psd kwarg in __init__ method must be true."
                )
            # must be transpose for noise
            noise_params = params[1].T

            params = params[0]

            if psd is not None:
                raise ValueError(
                    "If providing noise parameters to likelihood, cannot also provide psd kwarg."
                )

        else:
            noise_params = None

        assert isinstance(params, np.ndarray)

        if self.parameter_transforms is not None:
            keys = list(self.parameter_transforms.keys())
            if len(keys) > 1:
                if len(keys) > 2:
                    raise ValueError(
                        "parameter_transforms should only contain transforms for the parameters and the noise parameters."
                    )
                if "noise_params" not in keys or "noise_params" != keys[0]:
                    raise ValueError(
                        "'noise_params' must be the model name given for noise information to maintain consistency. It must be provided in the second position in the parameter_transforms dictionary."
                    )

            params = self.parameter_transforms[keys[0]].both_transforms(params)

            if "noise_params" in keys:
                noise_params = self.parameter_transforms[
                    "noise_params"
                ].both_transforms(noise_params)

        # only has to do with params, not noise params
        if self.transpose_params:
            params = params.T
            subset_axis = 1
        else:
            subset_axis = 0

        num_likes = params.shape[subset_axis]

        inds_likes = np.arange(num_likes)
        if self.subset is not None:
            if not isinstance(self.subset, int):
                raise ValueError("Subset must be int.")
            subset_sep = np.arange(self.subset, num_likes, self.subset)
            inds_subset = np.split(inds_likes, subset_sep)
        else:
            inds_subset = [inds_likes]

        out_ll = []
        for inds in inds_subset:
            if subset_axis == 0:
                args_in = (params[inds],)
            else:
                args_in = (params[:, inds],)

            args_in += args

            if self.fill_data_noise or self.like_here or noise_params is not None:
                if data is None:
                    data = self.injection_channels

                if noise_params is not None:
                    # assumes that there is one set of noise parameters per regular parameters
                    psd = self.evaluate_psd(noise_params[:, inds])
                else:
                    psd = self.psd

                args_in += (data, psd)

            out_ll.append(self.get_ll(*args_in, **kwargs))
        return np.concatenate(out_ll, axis=0)

        # TODO add Subset
        """
        num_inds = len(inds_eval)
            ind_skip = np.arange(self.subset, num_inds, self.subset)
            inds_eval_temp = [inds for inds in np.split(inds_eval, ind_skip)]

            temp = [None for j in range(len(inds_eval_temp))]
            if self.get_d_h:
                d_h = [None for j in range(len(inds_eval_temp))]
                h_h = [None for j in range(len(inds_eval_temp))]

            for j, inds in enumerate(inds_eval_temp):
                if self.add_inds:
                    self.lnlike_kwargs["waveform_kwargs"]["inds"] = inds

                temp[j] = self.lnlike.get_ll(x_in[inds], **self.lnlike_kwargs)
                if self.get_d_h:
                    d_h[j] = self.lnlike.d_h
                    h_h[j] = self.lnlike.h_h
            temp = np.concatenate(temp)
            if self.get_d_h:
                h_h = np.concatenate(h_h)
                d_h = np.concatenate(d_h)

        loglike_vals[inds_eval] = temp
        loglike_vals[np.isnan(loglike_vals)] = np.inf

        if self.get_d_h:
            d_h_vals[inds_eval] = d_h
            h_h_vals[inds_eval] = h_h

        array_1 = (
            -loglike_vals if self.add_prior is False else -loglike_vals + prior_vals
        )
        list_of_arrays = [array_1, prior_vals]
        if self.get_d_h:
            list_of_arrays = list_of_arrays + [d_h_vals, h_h_vals]
        return np.asarray(list_of_arrays).T
    @property
    def d_h(self):
        if self.separate_d_h is False:
            raise ValueError("Cannot get dh term if self.separate_d_h if False.")

        if hasattr(self.template_model, "d_h"):
            return self.template_model.d_h.copy()

        else:
            raise ValueError("Template model does not have the d_h term available.")

    @property
    def h_h(self):
        if self.separate_d_h is False:
            raise ValueError("Cannot get dh term if self.separate_d_h if False.")

        if hasattr(self.template_model, "h_h"):
            return self.template_model.h_h.copy()

        else:
            raise ValueError("Template model does not have the d_h term available.")

        """


class GlobalLikelihood(Likelihood):
    def __init__(
        self,
        *args,
        fill_templates=False,
        **kwargs,
    ):
        super(GlobalLikelihood, self).__init__(*args, **kwargs)
        self.fill_templates = fill_templates

    def _specific_likelihood_setup(self):
        if not isinstance(self.template_model, list):
            self.template_model = [self.template_model]

        if not isinstance(self.parameter_transforms, list):
            self.parameter_transforms = [self.parameter_transforms]

        if not isinstance(self.vectorized, list):
            self.vectorized = [self.vectorized for _ in self.template_model]

        assert (
            len(self.template_model)
            == len(self.parameter_transforms)
            == len(self.vectorized)
        )

        self.like_here = True

    def get_ll(
        self,
        params,
        groups,
        data,
        psd,
        data_length=None,
        start_freq_ind=None,
        args_list=None,
        kwargs_list=None,
        supps=None,
        branch_supps=None,
    ):
        # get supps
        if not isinstance(params, list):
            params = [params]
        if not isinstance(groups, list):
            groups = [groups]

        assert len(groups) == len(params)

        if branch_supps is not None:
            if not isinstance(branch_supps, list):
                branch_supps = [branch_supps]

        else:
            branch_supps = [None for _ in params]

        assert len(groups) == len(branch_supps)

        if args_list is None:
            args_list = [[] for _ in params]

        if kwargs_list is None:
            kwargs_list = [{} for _ in params]

        # TODO: make sure parameter transformations appear in posterior if possible
        total_groups = np.max(np.concatenate(groups)) + 1

        if data_length is not None or start_freq_ind is not None:
            if data_length is None:
                data_length = self.data_length
            elif not isinstance(data_length, int):
                raise ValueError("data_length must be int.")

            if start_freq_ind is None:
                start_freq_ind = self.start_freq_ind
            elif not isinstance(start_freq_ind, int):
                raise ValueError("start_freq_ind must be int.")

            if (start_freq_ind - self.start_freq_ind) + data_length > self.data_length:
                raise ValueError("start_freq_ind + data_length > full data length.")

            if kwargs_list is None:
                kwargs_list = [{}]

            for kwargs in kwargs_list:
                if isinstance(kwargs, dict):
                    kwargs["start_freq_ind"] = start_freq_ind

        else:
            start_freq_ind = self.start_freq_ind
            data_length = self.data_length

        if data is None:
            data = self.injection_channels[xp.newaxis, :, inds_slice]

        if psd is None:
            psd = self.psd[xp.newaxis, :, inds_slice]

        if supps is None or "data_minus_template" not in supps:
            template_all = xp.zeros(
                (total_groups, self.num_channels, data_length),
                dtype=xp.complex128,
            )

            for i, (
                params_i,
                groups_i,
                args_i,
                kwargs_i,
                tm_i,
                vec_i,
                branch_supp_i,
            ) in enumerate(
                zip(
                    params,
                    groups,
                    args_list,
                    kwargs_list,
                    self.template_model,
                    self.vectorized,
                    branch_supps,
                )
            ):
                # TODO: make fill templates adjustable per model
                if not self.fill_templates:  # False
                    if vec_i:
                        template_channels = xp.asarray(
                            tm_i(params_i, *args_i, **kwargs_i)
                        )

                    else:
                        template_channels = xp.asarray(
                            [
                                tm_i(params_ij, *args_i, **kwargs_i)
                                for params_ij in params_i.T
                            ]
                        )

                    if self.frequency_domain is False:
                        # TODO: vectorize this
                        # 2: is removal of DC component + right summation approximation
                        template_channels = (
                            xp.fft.rfft(template_channels, axis=-1) * self.dt
                        )

                    # TODO: could put this in c?
                    for group_ij in np.unique(groups_i):
                        inds1 = np.where(groups_i == group_ij)
                        template_all[group_ij] += template_channels[inds1].sum(axis=0)

                else:  # model will fill templates
                    kwargs_i_in = kwargs_i.copy()
                    if branch_supp_i is not None:
                        kwargs_i_in["branch_supps"] = branch_supp_i
                    if supps is not None:
                        kwargs_i_in["supps"] = supps

                    if vec_i:
                        tm_i.generate_global_template(
                            params_i, groups_i, template_all, *args_i, **kwargs_i_in
                        )

                    else:
                        for params_ij, groups_ij in zip(params_i.T, groups_i):
                            tm_i.generate_global_template(
                                params_ij,
                                groups_ij,
                                template_all,
                                *args_i,
                                **kwargs_i_in,
                            )
            breakpoint()
            # accelerate ?
            d_minus_h = (data - template_all).reshape(
                total_groups, len(self.injection_channels), -1
            )

        else:
            d_minus_h = supps["data_minus_template"]

        start_here = start_freq_ind - self.start_freq_ind
        end_here = start_here + data_length
        inds_slice = slice(start_here, end_here)

        # avoid f = 0
        start_ind = 1 if np.isnan(psd[0, 0, inds_slice][0]) else 0

        self.signal_ll = -(
            1.0
            / 2.0
            * (
                4.0
                * self.df
                * xp.sum(
                    (
                        d_minus_h[:, :, start_ind:].conj() * d_minus_h[:, :, start_ind:]
                    ).real
                    / psd[:, :, start_ind:],
                    axis=(1, 2),
                )
            )
        )

        ll = self.signal_ll.copy()
        if self.adjust_psd:
            self.noise_ll = -xp.sum(xp.log(psd), axis=(1, 2))
            ll += self.noise_ll

        out = xp.atleast_1d(ll.squeeze())
        if xp.any(xp.isnan(out)):
            breakpoint()
        if self.use_gpu:
            if self.return_cupy:
                return out
            else:
                try:
                    return out.get()
                except AttributeError:
                    return out

        else:
            return out

    def __call__(self, params, groups, *args, data=None, psd=None, **kwargs):
        if isinstance(params, np.ndarray):
            params = [params]
        elif not isinstance(params, list):
            raise ValueError("params must be np.ndarray or list of np.ndarray.")

        if isinstance(groups, np.ndarray):
            groups = [groups]
        elif not isinstance(groups, list):
            raise ValueError("groups must be np.ndarray or list of np.ndarray.")

        # if np.any(np.abs(params[0][:, 4]) > 1.0) or np.any(np.abs(params[0][:, 7]) > 1.0):
        #    breakpoint()

        if self.parameter_transforms is not None:
            for i, (params_i, transform_i) in enumerate(
                zip(params, self.parameter_transforms)
            ):
                params[i] = transform_i.both_transforms(params_i.copy())

        else:
            params = [params_i.T for params_i in params]

        for par_i in params:
            if np.any(np.isnan(par_i)):
                breakpoint()

        if self.adjust_psd:
            assert len(params) > 1
            assert len(groups) > 1

            noise_params = params[-1]
            noise_groups = groups[-1]

            if "branch_supps" in kwargs:
                assert len(kwargs["branch_supps"]) == len(groups)
                noise_supps = kwargs["branch_supps"][-1]
                kwargs["branch_supps"] = kwargs["branch_supps"][:-1]

            params = params[:-1]
            groups = groups[:-1]

        else:
            noise_params = None

        args_in = [params] + [groups] + list(args)

        if noise_params is not None:
            psd = self.evaluate_psd(noise_params.T, noise_groups=noise_groups)
        else:
            if psd is None:
                psd = self.psd
            if self.like_here:
                if isinstance(psd, list):
                    psd = xp.asarray(psd)[None, :, :]  # .transpose(1, 0, 2)

        args_in += args

        if self.fill_data_noise or self.like_here or noise_params is not None:
            if data is None:
                data = self.injection_channels

            args_in += [data, psd]

        return self.get_ll(*args_in, **kwargs)

        # TODO add Subset
        """
        num_inds = len(inds_eval)
            ind_skip = np.arange(self.subset, num_inds, self.subset)
            inds_eval_temp = [inds for inds in np.split(inds_eval, ind_skip)]

            temp = [None for j in range(len(inds_eval_temp))]
            if self.get_d_h:
                d_h = [None for j in range(len(inds_eval_temp))]
                h_h = [None for j in range(len(inds_eval_temp))]

            for j, inds in enumerate(inds_eval_temp):
                if self.add_inds:
                    self.lnlike_kwargs["waveform_kwargs"]["inds"] = inds

                temp[j] = self.lnlike.get_ll(x_in[inds], **self.lnlike_kwargs)
                if self.get_d_h:
                    d_h[j] = self.lnlike.d_h
                    h_h[j] = self.lnlike.h_h
            temp = np.concatenate(temp)
            if self.get_d_h:
                h_h = np.concatenate(h_h)
                d_h = np.concatenate(d_h)

        loglike_vals[inds_eval] = temp
        loglike_vals[np.isnan(loglike_vals)] = np.inf

        if self.get_d_h:
            d_h_vals[inds_eval] = d_h
            h_h_vals[inds_eval] = h_h

        array_1 = (
            -loglike_vals if self.add_prior is False else -loglike_vals + prior_vals
        )
        list_of_arrays = [array_1, prior_vals]
        if self.get_d_h:
            list_of_arrays = list_of_arrays + [d_h_vals, h_h_vals]
        return np.asarray(list_of_arrays).T
    @property
    def d_h(self):
        if self.separate_d_h is False:
            raise ValueError("Cannot get dh term if self.separate_d_h if False.")

        if hasattr(self.template_model, "d_h"):
            return self.template_model.d_h.copy()

        else:
            raise ValueError("Template model does not have the d_h term available.")

    @property
    def h_h(self):
        if self.separate_d_h is False:
            raise ValueError("Cannot get dh term if self.separate_d_h if False.")

        if hasattr(self.template_model, "h_h"):
            return self.template_model.h_h.copy()

        else:
            raise ValueError("Template model does not have the d_h term available.")

        """
