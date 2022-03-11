import warnings
from eryn.state import Branch, BranchSupplimental

import numpy as np

try:
    import cupy as xp

except (ModuleNotFoundError, ImportError):
    import numpy as xp


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
    ):

        self.subset = subset
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
        if use_gpu:
            self.xp = xp

        else:
            self.xp = np

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
        noise_kwargs={},
        add_noise=False,
    ):

        if params is not None:
            if self.parameter_transforms is not None:
                key = list(self.parameter_transforms.keys())[0]
                params = self.parameter_transforms[key].both_transforms(params)

            injection_channels = self.xp.asarray(
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
                injection_channels = self.xp.asarray(data_stream).get()
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

        if self.frequency_domain:
            if self.df is not None:
                df = self.df
                can_add_noise = True
                freqs = np.arange(self.injection_length) * df
                injection_channels = [inj for inj in injection_channels]

            else:
                freqs = self.f_arr
                can_add_noise = False

        else:
            dt = self.dt
            freqs = np.fft.rfftfreq(self.injection_length, dt)
            can_add_noise = True
            df = 1.0 / (self.injection_length * dt)

        psd = [
            noise_fn_temp(freqs, **noise_kwargs_temp)
            for noise_fn_temp, noise_kwargs_temp in zip(noise_fn, noise_kwargs)
        ]

        if self.frequency_domain is False:
            injection_channels = [
                np.fft.rfft(inj) * dt for inj in injection_channels
            ]

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
        injection_channels = [
            inj * (diff_freqs / psd_temp) ** (1 / 2)
            for inj, psd_temp in zip(injection_channels, psd)
        ]

        self.noise_factor = self.xp.asarray(
            [(diff_freqs / psd_temp) ** (1 / 2) for psd_temp in psd]
        )

        if hasattr(self, "injection_channels") is False:
            self.injection_channels = self.xp.asarray(injection_channels)
            self.freqs = self.xp.asarray(freqs)

        else:
            self.injection_channels += self.xp.asarray(injection_channels)

        if self.like_here is False:
            self.injection_channels = [inj.copy() for inj in self.injection_channels]
            self.noise_factor = [nf.copy() for nf in self.noise_factor]

        self.data_length = len(self.injection_channels[0])
        self.psd = psd

    def get_ll(self, params, *args, **kwargs):

        # TODO: make sure parameter transformations appear in posterior if possible
        num_likes = params.shape[0]
        if self.vectorized:
            template_channels = self.xp.asarray(
                self.template_model(*params, *args, **kwargs)
            )

        else:
            template_channels = self.xp.asarray(
                [self.template_model(*params_i, *args, **kwargs) for params_i in params]
            )

        if self.frequency_domain is False:
            # TODO: vectorize this
            # 2: is removal of DC component + right summation approximation
            template_channels = (
                self.xp.fft.rfft(template_channels, axis=-1)[:, :, 2:] * self.dt
            )

        h = template_channels * self.noise_factor[self.xp.newaxis, :, :]
        if self.separate_d_h:
            raise NotImplementedError

        else:
            # combines all channels into 1D array per likelihood
            d_minus_h = (self.injection_channels[self.xp.newaxis, :, :] - h).reshape(
                num_likes, len(self.injection_channels), -1
            )

            # TODO: add inds_slice to here from global
            start_ind = 1 if np.isnan(self.noise_factor[0, 0]) else 0

            ll = -(
                1.0
                / 2.0
                * (4.0 * self.xp.sum((d_minus_h[:, :, start_ind:].conj() * d_minus_h[:, :, start_ind:]).real, axis=(1, 2)))
            )

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

    def __call__(self, params, *args, **kwargs):

        if not isinstance(params, np.ndarray):
            raise ValueError("params must be np.ndarray.")

        if self.parameter_transforms is not None:
            key = list(self.parameter_transforms.keys())[0]
            params = self.parameter_transforms[key].both_transforms(params)

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

            if self.fill_data_noise:
                args_in += (self.injection_channels, self.noise_factor)

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
        self, *args, fill_templates=False, **kwargs,
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
        total_groups = np.max(groups) + 1

        if data_length is not None or start_freq_ind is not None:
            if data_length is None:
                data_length = self.data_length
            elif not isinstance(data_length, int):
                raise ValueError("data_length must be int.")

            if start_freq_ind is None:
                start_freq_ind = 0
            elif not isinstance(start_freq_ind, int):
                raise ValueError("start_freq_ind must be int.")

            if start_freq_ind + data_length > self.data_length:
                raise ValueError("start_freq_ind + data_length > full data length.")

            if kwargs_list is None:
                kwargs_list = [{}]

            for kwargs in kwargs_list:
                kwargs["start_freq_ind"] = start_freq_ind

        else:
            start_freq_ind = 0
            data_length = self.data_length

        template_all = self.xp.zeros(
            (total_groups, self.num_channels, data_length), dtype=self.xp.complex128,
        )

        for i, (params_i, groups_i, args_i, kwargs_i, tm_i, vec_i, branch_supp_i) in enumerate(
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
                    template_channels = self.xp.asarray(
                        tm_i(params_i, *args_i, **kwargs_i)
                    )

                else:
                    template_channels = self.xp.asarray(
                        [
                            tm_i(params_ij, *args_i, **kwargs_i)
                            for params_ij in params_i.T
                        ]
                    )

                if self.frequency_domain is False:
                    # TODO: vectorize this
                    # 2: is removal of DC component + right summation approximation
                    template_channels = (
                        self.xp.fft.rfft(template_channels, axis=-1)[:, :, 2:] * self.dt
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
                            params_ij, groups_ij, template_all, *args_i, **kwargs_i_in
                        )

        inds_slice = slice(start_freq_ind, start_freq_ind + data_length)

        template_all *= self.noise_factor[self.xp.newaxis, :, inds_slice]

        # accelerate ?
        d_minus_h = (
            self.injection_channels[self.xp.newaxis, :, inds_slice] - template_all
        ).reshape(total_groups, len(self.injection_channels), -1)

        start_ind = 1 if np.isnan(self.noise_factor[0, inds_slice][0]) else 0

        ll = -(
            1.0 / 2.0 * (4.0 * self.xp.sum((d_minus_h[:, :, start_ind:].conj() * d_minus_h[:, :, start_ind:]).real, axis=(1, 2)))
        )

        if self.noise_has_been_added:
            ll -= self.noise_likelihood_factor

        out = self.xp.atleast_1d(ll.squeeze())

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

    def __call__(self, params, groups, *args, **kwargs):
        if isinstance(params, np.ndarray):
            params = [params]
        elif not isinstance(params, list):
            raise ValueError("params must be np.ndarray or list of np.ndarray.")

        if isinstance(groups, np.ndarray):
            groups = [groups]
        elif not isinstance(groups, list):
            raise ValueError("groups must be np.ndarray or list of np.ndarray.")

        if self.parameter_transforms is not None:
            for i, (params_i, transform_i) in enumerate(
                zip(params, self.parameter_transforms)
            ):
                params_temp = transform_i.fill_values(params_i.copy())
                params[i] = transform_i.transform_base_parameters(params_temp)

        else:
            params = [params_i.T for params_i in params]

        args_in = params + groups + list(args)
        if self.fill_data_noise:
            args_in += [self.injection_channels, self.noise_factor]

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
