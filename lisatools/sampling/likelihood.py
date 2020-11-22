import warnings

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
        frequency_domain=True,
        parameter_transforms={},
        use_gpu=False,
        vectorized=False,
        separate_d_h=False,
    ):
        self.template_model = template_model
        self.frequency_domain = frequency_domain
        self.parameter_transforms = parameter_transforms

        self.use_gpu = use_gpu
        self.vectorized = vectorized

        self.num_channels = num_channels

        self.separate_d_h = separate_d_h

        if use_gpu:
            self.xp = xp

        else:
            self.xp = np

        if hasattr(self.template_model, "get_ll"):
            self.like_here = False

        else:
            self.like_here = True

        self.noise_has_been_added = False

    # TODO: add previously injected signal from a file for example
    # TODO: add SNR scaling, need to read out new distance
    def inject_signal(
        self,
        x,
        data_stream=None,
        params=None,
        waveform_kwargs={},
        noise_fn=None,
        noise_kwargs={},
        add_noise=False,
    ):

        if params is not None:
            for ind, transform_fn in self.parameter_transforms.items():
                params[ind] = transform_fn(params[ind])

            injection_channels = np.asarray(
                self.template_model(*params, **waveform_kwargs)
            )

        elif data_stream is not None:
            if isinstance(data_stream, list) is False:
                raise ValueError("If data_stream is provided, it must be as a list.")
            injection_channels = np.asarray(data_stream)

        else:
            raise ValueError(
                "Must provide data_stream or params kwargs to inject signal."
            )

        self.injection_length = len(injection_channels[0])

        for inj in injection_channels:
            if len(inj) != self.injection_length:
                raise ShapeError("Length of all injection channels must match.")

        if len(injection_channels) != self.num_channels:
            raise ValueError(
                "Number of channels from template_model does not match number of channels declare by user."
            )

        if isinstance(noise_fn, list):
            if len(noise_fn) != 1 and len(noise_fn) != self.num_channels:
                raise ValueError(
                    "Number of noise functions does not match number of channels declare by user."
                )

            else:
                noise_fn = [noise_fn[0] for _ in range(self.num_channels)]

        else:
            noise_fn = [noise_fn for _ in range(self.num_channels)]

        if self.frequency_domain:
            if isinstance(x, float):
                df = x
                can_add_noise = True
                freqs = np.arange(self.injection_length)[1:] * x

            elif isinstance(x, np.ndarray):
                freqs = x
                can_add_noise = False

            else:
                raise ValueError(
                    "When in the frequency domain, x must be scalar or a np.ndarray."
                )

        else:
            if isinstance(x, float) is False:
                raise ValueError("When in the time domain, x must be equivalent to dt.")

            dt = x
            self.dt = dt
            freqs = np.fft.rfftfreq(self.injection_length, dt)[1:]
            can_add_noise = True
            df = 1.0 / (self.injection_length * dt)

        psd = [noise_fn_temp(freqs, **noise_kwargs) for noise_fn_temp in noise_fn]

        if self.frequency_domain is False:
            injection_channels = [
                np.fft.rfft(inj)[1:] * dt for inj in injection_channels
            ]

        diff_freqs = np.diff(freqs)

        self.base_injections = injection_channels
        if add_noise and can_add_noise and self.noise_has_been_added is False:
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
                    1.0 / 2.0 * (2 * np.pi) * np.sum(diff_freqs * psd_temp[1:])
                    for psd_temp in psd
                ]
            )
            self.noise_has_been_added = True

        # noise weighting
        injection_channels = [
            inj[1:] * (diff_freqs / psd_temp[1:]) ** (1 / 2)
            for inj, psd_temp in zip(injection_channels, psd)
        ]

        self.noise_factor = self.xp.asarray(
            [(diff_freqs / psd_temp[1:]) ** (1 / 2) for psd_temp in psd]
        )

        if hasattr(self, "injection_channels") is False:
            self.injection_channels = self.xp.asarray(injection_channels)
            self.freqs = self.xp.asarray(freqs)[1:]

        else:
            self.injection_channels = self.xp.asarray(injection_channels)

    def get_ll(self, params, waveform_kwargs={}):

        params = self.xp.atleast_2d(self.xp.asarray(params.copy()))

        # TODO: make sure parameter transformations appear in posterior if possible
        num_likes = params.shape[0]
        params = params.T

        for ind, transform_fn in self.parameter_transforms.items():
            params[ind, :] = transform_fn(params[ind, :])

        if self.like_here is False:
            args = (params,)
            if self.separate_d_h is False:
                args += (self.injection_channels, self.noise_factor)

            return self.template_model.get_ll(*args, **waveform_kwarg)

        if self.vectorized:
            template_channels = self.xp.asarray(
                self.template_model(*params, **waveform_kwargs)
            )

        else:
            template_channels = self.xp.asarray(
                [
                    self.template_model(*params_i, **waveform_kwargs)
                    for params_i in params.T
                ]
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
            d_minus_h = (self.injection_channels[self.xp.newaxis, :, :] - h).reshape(
                num_likes, -1
            )

            ll = (
                1.0
                / 2.0
                * (4.0 * self.xp.sum((d_minus_h.conj() * d_minus_h).real, axis=1))
            )

            if self.noise_has_been_added:
                ll -= self.noise_likelihood_factor

            return ll.squeeze()
