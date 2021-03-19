#!/usr/bin/env python
# coding: utf-8

# In[1]:


import h5py
import numpy as np
import matplotlib.pyplot as plt

#%matplotlib inline

from ldc.waveform.lisabeta.fast_mbhb import FastMBHB

from bbhx.utils.waveformbuild import BBHWaveform
from lisatools.utils.transform import tLfromSSBframe, TransformContainer
from scipy import constants as ct
from lisatools.diagnostic import *
from lisatools.sampling.likelihood import Likelihood
from lisatools.sampling.samplers.ptemcee import PTEmceeSampler
from bbhx.utils.likelihood import Likelihood as MBHLikelihood
from bbhx.utils.likelihood import RelativeBinning
from lisatools.sensitivity import get_sensitivity
from bbhx.utils.constants import *
from lisatools.utils.transform import (
    LISA_to_SSB,
    SSB_to_LISA,
    mT_q,
    transfer_tref,
    mbh_sky_mode_transform,
)
from lisatools.utils.utility import uniform_dist
from lisatools.sampling.utility import ModifiedHDFBackend
from lisatools.utils.constants import *

try:
    import cupy as cp

    gpu_available = True

except (ModuleNotFoundError, ImportError):
    import numpy as np

    gpu_available = False


class SamplerGuide:
    def __init__(
        self,
        nwalkers_all,
        start_points=None,
        include_precession=False,
        priors=None,
        sampler_kwargs={},
        data=None,
        injection_params=None,
        test_inds=None,
        likelihood_kwargs={},
        injection_setup_kwargs={},
        waveform_kwargs={},
        parameter_transforms=None,
        periodic=None,
        use_gpu=False,
        verbose=False,
    ):

        self.use_gpu = use_gpu
        if self.use_gpu:
            self.xp = cp
        else:
            self.xp = np

        self.nwalkers_all = nwalkers_all
        self.include_precession = include_precession
        self.injection_params = injection_params
        self.data = data
        if data is None and injection_params is None:
            raise ValueError("Must provide either data or injection_params.")

        self.waveform_kwargs = waveform_kwargs
        if injection_setup_kwargs == {}:
            injection_setup_kwargs = self.default_injection_setup_kwargs

        self.injection_setup_kwargs = injection_setup_kwargs
        self.injection_setup_kwargs["data_stream"] = data
        self.injection_setup_kwargs["params"] = injection_params

        self.likelihood_kwargs = likelihood_kwargs

        self.sampler_kwargs = sampler_kwargs

        self.initial_setup(priors, start_points, test_inds)

        self.parameter_transforms = parameter_transforms

        self.periodic = periodic
        self.sampler_kwargs["periodic"] = self.periodic
        self.likelihood_kwargs["parameter_transforms"] = self.parameter_transforms
        self.sampler_kwargs["test_inds"] = self.test_inds
        self.sampler_kwargs["fill_values"] = self.fill_values

        if verbose == False:
            self.test_start_likelihood = False
        else:
            self.test_start_likelihood = True

    def initial_setup(self, priors, start_points, test_inds):

        self.priors = priors
        self.set_test_inds_info(test_inds)
        self.start_points = start_points

    @property
    def start_points(self):
        return self._start_points

    @start_points.setter
    def start_points(self, start_points):
        if "resume" in self.sampler_kwargs and self.sampler_kwargs["resume"]:
            reader = ModifiedHDFBackend(self.sampler_kwargs["fp"])
            self._start_points = reader.get_last_sample().coords.reshape(-1, self.ndim)

        elif isinstance(start_points, str) or start_points is None:
            if start_points == "prior" or start_points is None:
                self._start_points = np.asarray(
                    [prior.rvs(size=self.nwalkers_all) for prior in self.priors]
                ).T
            elif start_points == "fisher":
                if self.start_mean is None or self.cov is None:
                    raise ValueError(
                        "If generating points from fisher matrix, must store start_mean and cov attributes."
                    )
                self._start_points = np.random.multivariate_normal(
                    self.start_mean, self.cov
                )

        elif isinstance(start_points, np.ndarray):
            self._start_points = start_points

        else:
            raise ValueError(
                "start_point arguments incorrect. Must be 'prior', 'fisher', or np.ndarray of shape (nwalkers * ntempts, ndim) to start at specified points."
            )

    @property
    def priors(self):
        return self._priors

    @priors.setter
    def priors(self, priors):
        if priors is None:
            self._priors = self.default_priors

        elif isinstance(priors, dict):
            self._priors = self.default_priors
            for ind, distribution in priors.items():
                self._priors[ind] = distribution
        elif isinstance(priors, list):
            self._priors = priors
        else:
            raise ValueError(
                "If providing a prior, it must be dictionary to add specific priors into the defaults or list of all priors."
            )

    def set_test_inds_info(self, value):
        if value is None:
            self.test_inds = self.default_test_inds
            self.ndim = len(self.test_inds)
            self.fill_inds = self.default_fill_inds
            self.fill_values = self.default_fill_values
            return

        elif isinstance(value, np.ndarray):
            pass
        elif isinstance(value, list):
            value = np.asarray(value)
        else:
            raise ValueError(
                "test_inds, if provided, needs to be either an np.ndarray or list of indices."
            )

        if hasattr(self, injection) is False or self.injection is None:
            raise ValueError(
                "If providing test_inds, need to provide an injection to get the fill values."
            )

        self.test_inds = value
        self.ndim = len(self.test_inds)
        self.fill_values = np.delete(self.injection, self.test_inds)
        self.fill_inds = np.delete(self.default_ndim_fill, self.test_inds)
        return

    @property
    def lnprob(self):
        return self._lnprob

    @lnprob.setter
    def lnprob(self, template_generator):
        self._lnprob = Likelihood(
            template_generator, self.nchannels, **self.likelihood_kwargs
        )

    def perform_test_start_likelihood(self):
        # test starting points:
        test_start = np.zeros((self.start_points.shape[0], self.default_ndim_full))
        test_start[:, self.test_inds] = self.start_points
        test_start[:, self.fill_inds] = self.fill_values

        if "subset" in self.sampler_kwargs:
            pts_in = test_start[: self.sampler_kwargs["subset"]]

        else:
            pts_in = test_start

        check = self.lnprob.get_ll(pts_in, waveform_kwargs=self.waveform_kwargs)

        print(check)

    @property
    def parameter_transforms(self):
        return self._parameter_transforms

    @parameter_transforms.setter
    def parameter_transforms(self, transform_in):
        temp = self.default_parameter_transforms
        if transform_in is None:
            pass
        elif isinstance(transform_in, dict):
            for ind, fn in transform_in.items():
                temp[ind] = fn
        else:
            raise ValueError(
                "Transfor function should either be None or dict with index-fn pairs."
            )
        self._parameter_transforms = TransformContainer(temp)

    @property
    def periodic(self):
        return self._periodic

    @periodic.setter
    def periodic(self, periodic_in):
        self._periodic = self.default_periodic
        if periodic_in is None:
            pass

        # TODO: make every parameter periodic?
        elif isinstance(transform_in, dict):
            self._periodic = periodic_in
        else:
            raise ValueError(
                "Transfor function should either be None or dict with index-fn pairs."
            )

    def setup_sampler(self):

        if "ntemps" in self.sampler_kwargs:
            ntemps = self.sampler_kwargs["ntemps"]
        else:
            ntemps = 1

        self.nwalkers = int(self.nwalkers_all / ntemps)

        self.sampler = PTEmceeSampler(
            self.nwalkers,
            self.ndim,
            self.default_ndim_full,
            self.lnprob,
            self.priors,
            **self.sampler_kwargs,
        )

    def run_sampler(self, thin_by=1, iterations=10000, progress=True, **kwargs):
        self.sampler.sample(
            self.start_points,
            thin_by=thin_by,
            iterations=iterations,
            progress=True,
            **kwargs,
        )


class MBHGuide(SamplerGuide):
    @property
    def default_priors(self):
        default_priors = [
            uniform_dist(np.log(1e4), np.log(1e8)),
            uniform_dist(0.01, 0.999999999),
            uniform_dist(-0.99999999, +0.99999999),
            uniform_dist(-0.99999999, +0.99999999),
            uniform_dist(0.01, 1000.0),
            uniform_dist(0.0, 2 * np.pi),
            uniform_dist(-1.0, 1.0),
            uniform_dist(0.0, 2 * np.pi),
            uniform_dist(-1.0, 1.0),
            uniform_dist(0.0, np.pi),
            uniform_dist(0.0, self.Tobs),
        ]

        if hasattr(self, "include_precession") and self.include_precession:
            raise NotImplementedError

        return default_priors

    @property
    def default_ndim(self):
        if hasattr(self, "include_precession") and self.include_precession:
            return 15

        return 11

    @property
    def default_ndim_full(self):
        if hasattr(self, "include_precession") and self.include_precession:
            return 17

        return 13

    @property
    def default_test_inds(self):
        if hasattr(self, "include_precession") and self.include_precession:
            return np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15])

        return np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11])

    @property
    def default_fill_inds(self):
        if hasattr(self, "include_precession") and self.include_precession:
            return np.array([6, 16])

        return np.array([6, 12])

    @property
    def default_fill_values(self):
        if hasattr(self, "include_precession") and self.include_precession:
            return np.array([0.0, 0.0])

        return np.array([0.0, 0.0])

    @property
    def default_injection_setup_kwargs(self):
        return dict(
            waveform_kwargs={"freqs": self.f_arr, "length": 8192, "fill": True},
            noise_fn=get_sensitivity,
            noise_kwargs=[
                dict(sens_fn="noisepsd_AE", model="SciRDv1", includewd=None),
                dict(sens_fn="noisepsd_AE", model="SciRDv1", includewd=None),
                dict(sens_fn="noisepsd_T"),
            ],
            add_noise=False,
        )

    @property
    def default_parameter_transforms(self):
        parameter_transforms = {
            0: (lambda x: np.exp(x)),
            (0, 1): mT_q,
            4: (lambda x: x * PC_SI * 1e9),
            7: (lambda x: np.arccos(x)),
            9: (lambda x: np.arcsin(x)),
        }

        if self.sampler_frame == "LISA":
            parameter_transforms[(11, 12)] = transfer_tref
            parameter_transforms[(11, 8, 9, 10)] = LISA_to_SSB(0.0)

        return parameter_transforms

    @property
    def default_periodic(self):
        if hasattr(self, "include_precession") and self.include_precession:
            raise NotImplementedError

        return {5: 2 * np.pi, 7: 2 * np.pi, 9: np.pi}

    @property
    def default_relbin_kwargs(self):
        template_kwargs = dict(
            tBase=0.0,
            t_obs_start=1.0,
            t_obs_end=0.0,
            modes=[(2, 2)],
            direct=True,
            compress=True,
        )
        return dict(
            template_gen_kwargs=template_kwargs, noise_kwargs_AE={}, noise_kwargs_T={},
        )

    @property
    def default_relbin_args(self):
        return (256,)

    def __init__(
        self,
        *args,
        include_precession=False,
        amp_phase_kwargs={},
        response_kwargs={},
        sampler_frame="LISA",
        convert_from_ldc=False,
        use_gpu=False,
        f_arr=None,
        dt=10.0,
        Tobs=YRSID_SI,
        relbin=False,
        relbin_args=None,
        relbin_kwargs=None,
        relbin_template=None,
        multi_mode_start=True,
        **kwargs,
    ):

        self.sampler_frame = sampler_frame
        self.include_precession = include_precession
        self.nchannels = 3

        if f_arr is None:
            N = int(Tobs / dt)
            Tobs = dt * N
            df = 1 / Tobs
            f_arr = np.arange(0.0, 1 / (2 * dt) + df, df)[1:]  # remove dc
        else:
            df = f_arr[1]
            Tobs = 1 / df

        self.Tobs = Tobs
        self.f_arr = f_arr
        if "likelihood_kwargs" not in kwargs:
            kwargs["likelihood_kwargs"] = {}
        kwargs["likelihood_kwargs"]["f_arr"] = f_arr

        SamplerGuide.__init__(self, *args, use_gpu=use_gpu, **kwargs)

        if multi_mode_start:
            self.start_points = mbh_sky_mode_transform(
                self.start_points,
                cos_i=True,
                inplace=True,
                ind_map=dict(inc=6, lam=7, beta=8, psi=9),
                kind="both",
            )

        bbh = BBHWaveform(
            response_kwargs=response_kwargs,
            amp_phase_kwargs=amp_phase_kwargs,
            use_gpu=use_gpu,
        )

        # only temperary to help prepare data
        self.lnprob = bbh

        self.lnprob.inject_signal(**self.injection_setup_kwargs)

        noiseFactors = self.xp.asarray(self.lnprob.noise_factor)
        dataChannels = self.xp.asarray(self.lnprob.injection_channels)

        if relbin:
            if relbin_args is None:
                relbin_args = self.default_relbin_args
            if relbin_kwargs is None:
                relbin_kwargs = self.default_relbin_kwargs

            if relbin_template is None:
                if (
                    "resume" not in self.sampler_kwargs
                    or not self.sampler_kwargs["resume"]
                ):
                    raise ValueError(
                        "If using relative binning and not providin relbin_template parameters, must be resuming a run to get last sample."
                    )
                pass
                reader = ModifiedHDFBackend(self.sampler_kwargs["fp"])
                best_ind = reader.get_log_prob().argmax()
                relbin_template = np.zeros(self.default_ndim_full)
                relbin_template[self.test_inds] = reader.get_chain().reshape(
                    -1, self.ndim
                )[best_ind]
                relbin_template[self.fill_inds] = self.fill_values

            relbin_template = self.parameter_transforms.transform_base_parameters(
                relbin_template
            )

            # TODO: update this
            dataChannels /= noiseFactors
            mbh_like = RelativeBinning(
                bbh,
                self.xp.asarray(f_arr[1:]),
                dataChannels,
                relbin_template,
                *relbin_args,
                **relbin_kwargs,
                use_gpu=use_gpu,
            )

        else:

            mbh_like = MBHLikelihood(
                bbh,
                self.xp.asarray(f_arr[1:]),
                dataChannels,
                noiseFactors,
                use_gpu=use_gpu,
            )

        self.lnprob = mbh_like

        if self.test_start_likelihood:
            self.perform_test_start_likelihood()

        self.setup_sampler()


if __name__ == "__main__":

    from lisatools.utils.utility import AET
    from ldc.waveform.lisabeta import FastMBHB

    use_gpu = True

    xp = cp if use_gpu else np

    nwalkers = 80
    ntemps = 10

    nwalkers_all = nwalkers * ntemps

    amp_phase_kwargs = {"run_phenomd": True}

    params = {}
    with h5py.File("../GPU4GW/ldc/datasets/LDC1-1_MBHB_v2_FD.hdf5", "r") as f:
        grp = f["H5LISA"]["GWSources"]["MBHB-0"]
        for key in grp:
            params[key] = grp[key][()]
        print(list(f["H5LISA"]))

        data = f["H5LISA"]["PreProcess"]["TDIdata"][:]
        t, Xd, Yd, Zd = data[:4194304].T

    t -= t[0]

    dt = params["Cadence"]
    T = t[-1]
    fd, Xfd, Yfd, Zfd = (
        np.fft.rfftfreq(len(Xd), dt),
        np.fft.rfft(Xd) * dt,
        np.fft.rfft(Yd) * dt,
        np.fft.rfft(Zd) * dt,
    )

    Afd, Efd, Tfd = AET(Xfd, Yfd, Zfd)

    template_kwargs = dict(
        tBase=0.0,
        t_obs_start=1.0,
        t_obs_end=0.0,
        modes=[(2, 2)],
        direct=True,
        compress=True,
    )
    template_kwargs_full = dict(
        tBase=0.0,
        length=1024,
        freqs=xp.asarray(fd[1:]),
        t_obs_start=1.0,
        t_obs_end=0.0,
        modes=[(2, 2)],
        direct=False,
        compress=True,
    )

    fp_search = "/projects/b1095/mkatz/mbh/test_new_code.h5"
    fp_pe = "/projects/b1095/mkatz/mbh/test_new_code_pe.h5"

    reader = ModifiedHDFBackend(fp_search)
    log_prob = reader.get_log_prob().flatten()
    start_inds = np.argsort(log_prob)
    uni, inds = np.unique(log_prob[start_inds], return_index=True)
    start_inds = start_inds[inds[-int(ntemps * nwalkers) :]]

    ndim = 11
    ndim_full = 13
    test_inds = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11])

    start_points = reader.get_chain().reshape(-1, ndim)[start_inds]

    relbin_template = np.zeros(ndim_full)
    relbin_template[test_inds] = start_points[-1]

    sampler_kwargs = dict(
        lnlike_kwargs=dict(waveform_kwargs=template_kwargs),
        fp=fp_pe,
        resume=False,
        plot_iterations=1000,
        plot_source="mbh",
        ntemps=ntemps,
        get_d_h=True,
        # subset=int(nwalkers / 2),
        autocorr_multiplier=10000,
        burn=1000,
    )
    kwargs = dict(
        dt=dt,
        start_points=start_points,
        Tobs=len(t) * dt,
        f_arr=fd,
        sampler_kwargs=sampler_kwargs,
        likelihood_kwargs=dict(separate_d_h=True),
        data=[Afd, Efd, Tfd],
        waveform_kwargs=template_kwargs_full,
        verbose=True,
        multi_mode_start=True,
        relbin=True,
        relbin_template=relbin_template,
        use_gpu=True,
    )

    mbh_guide = MBHGuide(nwalkers * ntemps, **kwargs)
    mbh_guide.lnprob.template_model.d_d = 0.0
    mbh_guide.run_sampler(thin=1, iterations=50000)
    # breakpoint()

    """
    lisabeta_mbhb = FastMBHB(T=1 / fd[1], delta_t=dt, approx="IMRPhenomD", orbits=None)

    params_dc = params.copy()

    params_dc["Distance"] = 1e3 * params["Distance"]
    params_dc["PhaseAtCoalescence"] = params["PhaseAtCoalescence"]
    params_dc.pop("Approximant")
    params_dc.pop("AzimuthalAngleOfSpin1")
    params_dc.pop("AzimuthalAngleOfSpin2")

    try:
        params_dc.pop("hphcData")
    except KeyError:
        pass


    params_lb = lisabeta_mbhb.rename_as_lisabeta(params_dc)

    tSSB, lamSSB, betaSSB, psiSSB = (
        params_lb["Deltat"],
        params_lb["lambda"],
        params_lb["beta"],
        params_lb["psi"],
    )

    back_to_LISA = SSB_to_LISA(0.0)
    tL, lamL, betaL, psiL = back_to_LISA(tSSB, lamSSB, betaSSB, psiSSB)
    params_lb["Deltat"], params_lb["lambda"], params_lb["beta"], params_lb["psi"] = (
        tL,
        lamL,
        betaL,
        psiL,
    )
    """

    """
    nwalkers_all,
    *args,
    include_precession=False,
    amp_phase_kwargs={},
    response_kwargs={},
    sampler_frame="LISA",
    convert_from_ldc=False,
    use_gpu=False,
    f_arr=None,
    dt=10.0,
    Tobs=1.0,
    relbin=False,
    multi_mode_start=True,

    start_points=None,
    include_precession=False,
    priors=None,
    sampler_kwargs={},
    data=None,
    injection_params=None,
    test_inds=None,
    likehood_kwargs={},
    injection_setup_kwargs={},
    waveform_kwargs={},
    parameter_transforms=None,
    periodic=None,
    use_gpu=False,
    verbose=False,
    """
