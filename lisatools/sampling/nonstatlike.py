import warnings

import numpy as np

try:
    import cupy as xp

except (ModuleNotFoundError, ImportError):
    import numpy as xp

import scipy


class NonStationaryContainer(object):
    def __init__(
        self,
        w_t_window,
        w_t_noise,
        dt,
        noise_fn,
        noise_kwargs,
        bands=1,
        use_gpu=False,
        verbose=-1,
    ):

        self.use_gpu = use_gpu
        if use_gpu:
            self.xp = xp
        else:
            self.xp = np

        self.seg_length = len(w_t_window)
        self.dt = dt

        self.bands = bands
        self.df = df = 1 / (dt * self.seg_length)

        fft_window = self.xp.fft.rfft(self.xp.asarray(w_t_window + w_t_noise)) * dt

        self.fft_length = fft_length = len(fft_window)

        self.mid = mid = int((2 * bands + 1) / 2) - 1  # second -1 for index
        corr_mats = self.xp.zeros((2 * bands - 1, fft_length), dtype=self.xp.complex128)
        Sn = self.xp.asarray(noise_fn(np.arange(fft_length) * df, **noise_kwargs))

        # TODO: make options?
        Sn[0] = Sn[1]
        for i in range(fft_length):
            w_i = fft_window[0 : i + 1][::-1]

            if i < fft_length - (bands - 1):
                n = bands
            else:
                n = fft_length - i
            inds = (
                self.xp.tile(self.xp.arange(i + 1), (n, 1))
                + self.xp.arange(i, i + n)[:, None]
                - i
            )[:, ::-1]

            bad = inds >= fft_length
            inds[bad] = 0

            w_j = fft_window[inds]

            w_j[bad] = 0.0
            temp = (
                1
                / (self.seg_length * dt) ** 2
                * self.xp.sum((Sn[: i + 1] * w_i.conj())[None, :] * w_j, axis=-1)
            )

            corr_mats[mid : mid + n, i] = temp

            if verbose > 0:
                if not (i % verbose):
                    print(i, fft_length)

        if bands > 1:
            corr_mats[:mid] = corr_mats[mid + 1 :][::-1].conj()

            for b in range(0, mid):
                corr_mats[b] = self.xp.roll(corr_mats[b], mid - b)

        self.corr_mats = self.xp.asarray(corr_mats)
        self.Sn = self.xp.asarray(Sn)
        self.fft_window = self.xp.asarray(fft_window)


class NonStationaryLikelihood(object):
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
    ):
        self.template_model = template_model

        self.parameter_transforms = parameter_transforms

        self.use_gpu = use_gpu
        self.vectorized = vectorized

        self.num_channels = num_channels

        self.separate_d_h = separate_d_h

        if dt is None:
            raise NotImplementedError("For now, this only takes time-domain signals.")

        self.frequency_domain = False

        self.dt, self.df, self.f_arr = dt, df, f_arr

        self.return_cupy = return_cupy
        if use_gpu:
            self.xp = xp

        else:
            self.xp = np

        if hasattr(self.template_model, "get_ll"):
            self.like_here = False
            raise NotImplementedError

        else:
            self.like_here = True

        self.noise_has_been_added = False

    # TODO: add SNR scaling, need to read out new distance
    def inject_signal(
        self,
        data_stream=None,
        params=None,
        waveform_kwargs={},
        noise_fn=None,
        noise_kwargs={},
        w_t_noise=None,
        w_t_window=None,
        add_noise=False,
        bands=1,
    ):

        self.bands = bands
        self.w_t_window = w_t_window

        if params is not None:
            if self.parameter_transforms is not None:
                params = self.parameter_transforms.transform_base_parameters(params)

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
                raise ShapeError("Length of all injection channels must match.")

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
            # stationary component
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
            raise NotImplementedError
            if self.df is not None:
                df = self.df
                can_add_noise = True
                freqs = np.arange(self.injection_length)[1:] * df
                injection_channels = [inj[1:] for inj in injection_channels]

            else:
                freqs = self.f_arr
                can_add_noise = False

        else:
            dt = self.dt

            # find gaps
            if w_t_window is not None:
                if isinstance(w_t_window, np.ndarray):
                    # assumes all channels turn off and on at same time
                    w_t_window = [w_t_window for _ in range(self.num_channels)]
                elif not isinstance(w_t_window, list):
                    raise ValueError("w_t_window must be array or list of arrays.")
            else:
                raise NotImplementedError
            if w_t_noise is not None:
                if isinstance(w_t_noise, np.ndarray):
                    # assumes all channels turn off and on at same time
                    w_t_noise = [w_t_noise for _ in range(self.num_channels)]
                elif not isinstance(w_t_noise, list):
                    raise ValueError("w_t_noise must be array or list of arrays.")
            else:
                raise NotImplementedError

            single_channel = w_t_window[0]
            inds_zero = np.where(single_channel == 0.0)[0]

            jumps_start = np.where(np.diff(inds_zero) > 1)[0]

            # fix
            start_of_observing_segments = [0]
            end_of_observing_segments = []
            sc_last = 1
            for i, sc in enumerate(single_channel):
                if i == 0:
                    continue
                if sc_last != 0 and sc == 0.0:
                    end_of_observing_segments.append(i - 1)
                if sc_last == 0.0 and sc != 0.0:
                    start_of_observing_segments.append(i)

                sc_last = sc

            end_of_observing_segments.append(len(single_channel) - 1)

            start_of_observing_segments = np.asarray(start_of_observing_segments)
            end_of_observing_segments = np.asarray(end_of_observing_segments)

            observing_segment_length = (
                end_of_observing_segments - start_of_observing_segments + 1
            )
            num_segments = len(start_of_observing_segments)

            fft_windows = [_ for _ in range(num_segments)]
            fft_data = [_ for _ in range(num_segments)]
            df_segments = [_ for _ in range(num_segments)]
            fft_freqs_length = [_ for _ in range(num_segments)]
            for j, (start, end, length) in enumerate(
                zip(
                    start_of_observing_segments,
                    end_of_observing_segments,
                    observing_segment_length,
                )
            ):

                fft_freqs = np.fft.rfftfreq(length, dt)
                df_segments[j] = fft_freqs[1]
                fft_freqs_length[j] = len(fft_freqs)

                fft_windows[j] = [_ for _ in range(self.num_channels)]
                fft_data[j] = [_ for _ in range(self.num_channels)]

                for chan in range(self.num_channels):

                    fft_windows[j] = (
                        np.fft.rfft(
                            w_t_window[start : end + 1] + w_t_noise[start : end + 1]
                        )
                        * dt
                    )
                    fft_data[j] = np.fft.rfft(data_stream[start : end + 1]) * dt

            mid = int((2 * bands + 1) / 2) - 1  # second -1 for index
            corr_mats = [
                np.zeros(
                    (self.num_channels, 2 * bands - 1, fft_length), dtype=np.complex128
                )
                for fft_length in fft_freqs_length
            ]
            for k, (fft_wind, fft_d, df, fft_length, seg_length) in enumerate(
                zip(
                    fft_windows,
                    fft_data,
                    df_segments,
                    fft_freqs_length,
                    observing_segment_length,
                )
            ):
                for chan in range(self.num_channels):
                    Sn = noise_fn(np.arange(fft_length) * df, **noise_kwargs)
                    Sn[0] = Sn[1]
                    for i in range(fft_length):
                        w_i = fft_windows[k][0 : i + 1][::-1]

                        if i < fft_length - (bands - 1):
                            n = bands
                        else:
                            n = fft_length - i
                        inds = (
                            np.tile(np.arange(i + 1), (n, 1))
                            + np.arange(i, i + n)[:, None]
                            - i
                        )[:, ::-1]

                        bad = inds >= fft_length
                        inds[bad] = 0

                        w_j = fft_windows[k][inds]

                        w_j[bad] = 0.0
                        temp = (
                            1
                            / (seg_length * dt) ** 2
                            * np.sum((Sn[: i + 1] * w_i.conj())[None, :] * w_j, axis=-1)
                        )

                        corr_mats[k][chan, mid : mid + n, i] = temp

                    if bands > 1:
                        corr_mats[k][chan, :mid] = corr_mats[k][chan, mid + 1 :][
                            ::-1
                        ].conj()

                        for b in range(0, mid):
                            corr_mats[k][chan, b] = np.roll(
                                corr_mats[k][chan, b], mid - b
                            )

                    print(k, chan)

        self.num_segments = num_segments
        self.start_of_observing_segments = start_of_observing_segments
        self.end_of_observing_segments = end_of_observing_segments
        self.fft_data = fft_data
        self.df_segments = df_segments
        self.observing_segment_length = observing_segment_length
        self.corr_mats = corr_mats

        # TODO: add other things about injection of multiple signals etc.

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

    def get_ll(self, params, waveform_kwargs={}):
        if self.parameter_transforms is not None:
            params = self.parameter_transforms.transform_base_parameters(params)

        # TODO: make sure parameter transformations appear in posterior if possible
        num_likes = params.shape[1]

        if self.like_here is False:
            args = (params,)
            if self.separate_d_h is False:
                args += (self.injection_channels, self.noise_factor)

            out = self.template_model.get_ll(*args, **waveform_kwargs)

        else:
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

            h = template_channels * self.w_t_window

            # TODO: check factor of 2 for not having two w(t) thing
            like_all = np.zeros(num_likes, dtype=np.complex128)
            for bin in range(num_likes):
                like_bin = 0.0 + 0.0 * 1j
                for seg, (start, end, df) in enumerate(
                    zip(
                        self.start_of_observing_segments,
                        self.end_of_observing_segments,
                        self.df_segments,
                    )
                ):
                    like_seg = 0.0 + 0.0 * 1j
                    for chan in range(self.num_channels):
                        like_chan = 0.0 + 0.0 * 1j
                        h_temp = h[bin, chan, start : end + 1]
                        hf_temp = self.xp.fft.rfft(h_temp) * self.dt

                        df_temp = self.fft_data[seg]
                        cov = self.corr_mats[seg]

                        d_minus_h = df_temp - hf_temp
                        x = scipy.linalg.solve_banded(
                            (self.bands - 1, self.bands - 1), cov, d_minus_h
                        )

                        like_chan = 4 * df * np.dot(d_minus_h.conj(), x)

                        like_seg += like_chan
                    like_bin += like_seg
                like_all[bin] = like_bin

        out = like_all
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
