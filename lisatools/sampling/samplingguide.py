#!/usr/bin/env python
# coding: utf-8

# In[1]:


import h5py
import numpy as np
import matplotlib.pyplot as plt

#%matplotlib inline

from bbhx.waveformbuild import BBHWaveformFD
from bbhx.utils.transform import tLfromSSBframe
from scipy import constants as ct
from lisatools.diagnostic import *
from lisatools.sampling.likelihood import Likelihood
from bbhx.likelihood import Likelihood as MBHLikelihood
from bbhx.likelihood import HeterodynedLikelihood
from lisatools.sensitivity import get_sensitivity
from bbhx.utils.constants import *
from bbhx.utils.transform import (
    LISA_to_SSB,
    SSB_to_LISA,
    mT_q,
    mbh_sky_mode_transform,
)
from eryn.prior import uniform_dist
from eryn.backends import HDFBackend
from lisatools.utils.constants import *
from gbgpu.utils.constants import YEAR
from eryn.prior import PriorContainer
from eryn.utils import TransformContainer, PeriodicContainer
from eryn.state import State
from eryn.ensemble import EnsembleSampler
from eryn.backends import HDFBackend
from eryn.moves.tempering import make_ladder

from gbgpu.gbgpu import GBGPU

try:
    import cupy as cp

    gpu_available = True

except (ModuleNotFoundError, ImportError):
    import numpy as np

    gpu_available = False


class SamplerGuide:
    def __init__(
        self,
        nwalkers,
        start_state=None,
        include_precession=False,
        priors=None,
        sampler_kwargs={},
        data=None,
        injection_params=None,
        fill_dict=None,
        likelihood_kwargs={},
        injection_setup_kwargs={},
        waveform_kwargs={},
        parameter_transforms=None,
        transform_container=None,
        periodic=None,
        use_gpu=False,
        mean_and_cov=None,
        start_factor=None,
        verbose=False,
    ):

        self.input_start_state = start_state

        self.use_gpu = use_gpu
        if self.use_gpu:
            self.xp = cp
        else:
            self.xp = np

        if mean_and_cov is not None:
            self.start_mean, self.cov = mean_and_cov
        else:
            self.start_mean, self.cov = None, None

        if start_factor is None:
            start_factor = 1.0

        self.start_factor = start_factor

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

        self.nwalkers = nwalkers
        if "tempering_kwargs" in self.sampler_kwargs:
            if "ntemps" in self.sampler_kwargs["tempering_kwargs"]:
                self.ntemps = self.sampler_kwargs["tempering_kwargs"]["ntemps"]
            elif "betas" in self.sampler_kwargs["tempering_kwargs"]:
                self.ntemps = len(self.sampler_kwargs["tempering_kwargs"]["betas"])
        else:
            self.ntemps = 1

        self.priors = priors

        self._set_parameter_transforms(
            transform_container, parameter_transforms, fill_dict
        )

        self.periodic = periodic
        self.sampler_kwargs["periodic"] = self.periodic
        self.likelihood_kwargs["parameter_transforms"] = self.parameter_transforms

        if verbose == False:
            self.test_start_likelihood = False
        else:
            self.test_start_likelihood = True

    @property
    def start_state(self):
        return self._start_state

    @start_state.setter
    def start_state(self, start_state):
        if "resume" in self.sampler_kwargs and self.sampler_kwargs["resume"]:
            reader = HDFBackend(self.sampler_kwargs["fp"])
            self._start_state = reader.get_last_sample()

        elif isinstance(start_state, State):
            self._start_state = start_state

        elif isinstance(start_state, str) or start_state is None:
            if start_state == "prior" or start_state is None:
                self._start_state = State(
                    {
                        key: self.priors[key].rvs(
                            size=(self.ntemps, self.nwalkers, self.nleaves_max[i])
                        )
                        for i, key in enumerate(self.priors)
                    }
                )

            elif start_state == "fisher":
                if self.start_mean is None or self.cov is None:
                    raise ValueError(
                        "If generating points from fisher matrix, must store start_mean and cov attributes."
                    )
                
                run_flag = True
                while run_flag:
                    coords_all = {}
                    for i, key in enumerate(self.branch_names):

                        coords_all[key] = np.zeros(
                            self.ntemps,
                            self.nwalkers,
                            self.nleaves_max[i],
                            self.default_ndims[i],
                        )
                        for t in range(self.ntemps):
                            for w in range(self.nwalkers):
                                for l in range(nleaves_max):
                                    if l >= len(self.start_mean):
                                        continue

                                    coords = np.zeros(
                                        (self.nleaves_max[i], self.default_ndims[i],)
                                    )
                                    inds_fix = np.arange(self.nleaves_max[i])

                                    num_max_iter = 1000
                                    iter_num = 0
                                    while len(inds_fix) > 0:
                                        if iter_num > num_max_iter:
                                            raise RuntimeError(
                                                "Covariance matrix walker generation is not generating walkers in the prior range."
                                            )
                                        coords[inds_fix] = np.random.multivariate_normal(
                                            self.start_mean[l],
                                            self.start_factor * self.cov[l],
                                            size=len(inds_fix),
                                        )

                                        inds_fix = np.where(
                                            np.isinf(self.priors.logpdf(self._start_state))
                                        )[0]

                                        iter_num += 1
                                    coords_all[key][t, w, l] = coords

                    temp_state = State(coords_all)
                    breakpoint()

                self._start_state = State(coords_all)

        elif isinstance(start_state, np.ndarray):
            start_state = {self.branch_names[0]: start_state}
            self._start_state = State(start_state)

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
            temp = {}
            for key, priors_sub in priors.items():
                temp[key] = self.default_priors[key].priors_in
                for ind, distribution in priors_sub.items():
                    temp[key][ind] = distribution

            self._priors = {name: PriorContainer(temp[name]) for name in temp}

        else:
            raise ValueError(
                "If providing a prior, it must be dictionary to add specific priors into the defaults."
            )

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
        if not self.global_fit:
            check = self.sampler.compute_log_prob(self.start_state.branches_coords)

            try:
                self.like_check_before_start = check.copy()
            except AttributeError:
                self.like_check_before_start = check[0].copy()
        else:
            raise NotImplementedError(
                "You can check this with the `compute_log_prob` in the sampler."
            )

        print("start like check:", check)

    @property
    def parameter_transforms(self):
        return self._parameter_transforms

    def _set_parameter_transforms(
        self, transform_container, parameter_transforms_in, fill_dict_in
    ):

        if transform_container is not None:
            if isinstance(transform_container, TransformContainer):
                raise ValueError(
                    "If directly adding TransformContainer, put it in the form of a dictionary with a key as the branch or model name."
                )
            self._parameter_transforms = transform_container
            return

        temp_transform = self.default_parameter_transforms
        if parameter_transforms_in is None:
            pass
        elif isinstance(parameter_transforms_in, dict):
            for key, parameter_transforms_sub in parameter_transforms_in.items():
                if isinstance(parameter_transforms_sub, TransformContainer):
                    raise ValueError(
                        "If adding parameter_transforms and/or fill_dict separately, cannot add actual TransformContainer."
                    )
                elif isinstance(parameter_transforms_sub, dict):
                    for ind, fn in parameter_transforms_sub.items():
                        temp_transform[key][ind] = fn
                else:
                    raise ValueError(
                        "If providing dict for parameter_transforms_in, must contain dictionaries."
                    )
        else:
            raise ValueError(
                "Transform function should either be None or dict with {model_name: {index: fn}}."
            )

        temp_fill_dict = self.default_fill_dict
        if fill_dict_in is None:
            pass
        elif isinstance(fill_dict_in, dict):
            for key, fill_dict_in_sub in transform_in.items():
                for ind, fn in fill_dict_in_sub.items():
                    temp_fill_dict[key][ind] = fn

        else:
            raise ValueError(
                "fill_dict_in should either be None or dict with index-fn pairs."
            )

        assert len(temp_fill_dict.keys()) == len(temp_transform.keys())

        self._parameter_transforms = {}
        for key in temp_transform:
            self._parameter_transforms[key] = TransformContainer(
                parameter_transforms=temp_transform[key], fill_dict=temp_fill_dict[key]
            )
        self.fill_dict = temp_fill_dict

    @property
    def periodic(self):
        return self._periodic

    @periodic.setter
    def periodic(self, periodic_in):
        temp_periodic = self.default_periodic.periodic
        if periodic_in is None:
            pass

        # TODO: make every parameter periodic?
        elif isinstance(periodic_in, dict):
            for key, periodic_temp_sub in temp_periodic.items():
                if isinstance(periodic_temp_sub, PeriodicContainer):
                    temp_periodic[key] = periodic_temp_sub
                elif (isinstance, dict):
                    for ind, fn in transform_in.items():
                        temp_periodic[key][ind] = fn
                else:
                    raise ValueError(
                        "If providing dict for periodic_in, must contain dictionaries or PeriodicContainer as values."
                    )

        else:
            raise ValueError(
                "Periodic function should either be None or dict with index-fn pairs."
            )
        self._periodic = PeriodicContainer(temp_periodic)

    def setup_sampler(self):
        self.sampler = EnsembleSampler(
            self.nwalkers,
            self.default_ndims,  # assumes ndim_max
            self.lnprob,
            self.priors,
            **self.sampler_kwargs,
        )

        self.run_sampler = self.sampler.run_mcmc


class MBHGuide(SamplerGuide):
    @property
    def default_priors(self):
        default_priors = {
            "mbh": PriorContainer(
                {
                    0: uniform_dist(np.log(1e5), np.log(1e8)),
                    1: uniform_dist(0.01, 0.999999999),
                    2: uniform_dist(-0.99999999, +0.99999999),
                    3: uniform_dist(-0.99999999, +0.99999999),
                    4: uniform_dist(0.01, 1000.0),
                    5: uniform_dist(0.0, 2 * np.pi),
                    6: uniform_dist(-1.0, 1.0),
                    7: uniform_dist(0.0, 2 * np.pi),
                    8: uniform_dist(-1.0, 1.0),
                    9: uniform_dist(0.0, np.pi),
                    10: uniform_dist(0.0, self.Tobs),
                }
            )
        }

        if hasattr(self, "include_precession") and self.include_precession:
            raise NotImplementedError

        return default_priors

    @property
    def default_ndims(self):
        if hasattr(self, "include_precession") and self.include_precession:
            return [15]

        return [11]

    @property
    def default_fill_dict(self):
        if hasattr(self, "include_precession") and self.include_precession:
            return {
                "mbh": {
                    "fill_inds": np.array([6]),
                    "fill_values": np.array([0.0]),
                    "ndim_full": 16,
                }
            }
        else:
            return {
                "mbh": {
                    "fill_inds": np.array([6]),
                    "fill_values": np.array([0.0]),
                    "ndim_full": 12,
                }
            }

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
            "mbh": {
                0: (lambda x: np.exp(x)),
                (0, 1): mT_q,
                4: (lambda x: x * PC_SI * 1e9),
                7: (lambda x: np.arccos(x)),
                9: (lambda x: np.arcsin(x)),
            }
        }

        if self.sampler_frame == "LISA":
            parameter_transforms["mbh"][(11, 8, 9, 10)] = LISA_to_SSB

        return parameter_transforms

    @property
    def default_periodic(self):
        if hasattr(self, "include_precession") and self.include_precession:
            raise NotImplementedError

        return PeriodicContainer({"mbh": {5: 2 * np.pi, 7: 2 * np.pi, 9: np.pi}})

    @property
    def default_relbin_kwargs(self):
        template_kwargs = dict(
            tBase=0.0,
            t_obs_start=1.0,
            t_obs_end=0.0,
            modes=None,
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
        global_fit=False,
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

        self.global_fit = global_fit
        if global_fit:
            raise NotImplementedError

        self.nleaves_max = [1]
        self.sampler_frame = sampler_frame
        self.include_precession = include_precession
        self.nchannels = 3

        if f_arr is None:
            N = int(Tobs / dt)
            Tobs = dt * N
            df = 1 / Tobs
            f_arr = np.arange(0.0, 1 / (2 * dt) + df, df)[1:]  # remove dc
        else:
            df = f_arr[1] - f_arr[0]
            Tobs = 1 / df

        self.Tobs = Tobs
        self.f_arr = f_arr
        if "likelihood_kwargs" not in kwargs:
            kwargs["likelihood_kwargs"] = {}
        kwargs["likelihood_kwargs"]["f_arr"] = f_arr

        kwargs["likelihood_kwargs"]["transpose_params"] = True

        SamplerGuide.__init__(self, *args, use_gpu=use_gpu, **kwargs)

        if multi_mode_start:
            self.start_state.branches_coords["mbh"] = mbh_sky_mode_transform(
                self.start_state.branches_coords["mbh"].reshape(
                    -1, self.default_ndims[0]
                ),
                cos_i=True,
                inplace=True,
                ind_map=dict(inc=6, lam=7, beta=8, psi=9),
                kind="both",
            ).reshape(self.ntemps, self.nwalkers, 1, -1)

        bbh = BBHWaveformFD(
            response_kwargs=response_kwargs,
            amp_phase_kwargs=amp_phase_kwargs,
            use_gpu=use_gpu,
        )

        # only temperary to help prepare data
        self.lnprob = bbh

        self.lnprob.inject_signal(**self.injection_setup_kwargs)

        noiseFactors = self.xp.asarray(self.lnprob.noise_factor)
        dataChannels = self.xp.asarray(self.lnprob.injection_channels)
        psd = self.xp.asarray(self.lnprob.psd)
        if relbin:
            if self.global_fit:
                raise ValueError("Cannot do relative binning for global fit.")

            if relbin_args is None:
                relbin_args = self.default_relbin_args
            if relbin_kwargs is None:
                relbin_kwargs = self.default_relbin_kwargs

            if relbin_template is None:
                raise ValueError(
                    "When using relative binning, relbin_template must be set to 'multi', 'single', or it must be an array of start points."
                )
            elif relbin_template == "single":
                if "resume" in self.sampler_kwargs and self.sampler_kwargs["resume"]:
                    reader = HDFBackend(self.sampler_kwargs["fp"])
                    best_ind = np.where(
                        reader.get_log_prob().max() == reader.get_log_prob()
                    )
                    # assumes mbhs are only or first in dataset
                    relbin_template = reader.get_chain()
                    relbin_template = relbin_template[list(relbin_template.keys())[0]][
                        best_ind
                    ][0]
                    relbin_template = self.parameter_transforms["mbh"].fill_values(
                        relbin_template
                    )

                else:
                    relbin_template = self.start_state.branches_coords["mbh"][0, 0, 0]
                    relbin_template = self.parameter_transforms["mbh"].fill_values(
                        relbin_template
                    )

            elif not isinstance(relbin_template, np.ndarray):
                raise ValueError(
                    "When using relative binning, relbin_template must be set to 'multi', 'single', or it must be an array of start points."
                )

            elif isinstance(relbin_template, np.ndarray):
                if len(relbin_template) > self.fill_dict["mbh"]["ndim_full"]:
                    raise ValueError(
                        "More parametes in relbin_template than ndim_full."
                    )
                elif len(relbin_template) < self.fill_dict["mbh"]["ndim_full"]:
                    if (
                        len(relbin_template) + len(self.fill_dict["mbh"]["fill_values"])
                    ) != self.fill_dict["mbh"]["ndim_full"]:
                        raise ValueError(
                            "relbin_template does not have correct dimensionality."
                        )

                    relbin_template = self.parameter_transforms["mbh"].fill_values(
                        relbin_template
                    )

            relbin_template = self.parameter_transforms[
                "mbh"
            ].transform_base_parameters(relbin_template)

            # TODO: update this
            dataChannels /= noiseFactors
            mbh_like = HeterodynedLikelihood(
                bbh,
                self.xp.asarray(f_arr[1:]),
                dataChannels,
                relbin_template,
                *relbin_args,
                # template_gen_args=relbin_template,
                **relbin_kwargs,
                use_gpu=use_gpu,
            )

        else:
            dataChannels /= noiseFactors  # remove noise factors
            mbh_like = MBHLikelihood(
                bbh,
                self.xp.asarray(f_arr[1:]),
                dataChannels,
                psd[:, 1:],
                use_gpu=use_gpu,
            )

        self.lnprob = mbh_like

        self.setup_sampler()

        if self.test_start_likelihood:
            self.perform_test_start_likelihood()


class GBGuide(SamplerGuide):
    @property
    def default_priors(self):

        if hasattr(self, "include_fddot") and self.include_fddot:
            ind_change = 1
        else:
            ind_change = 0

        default_priors = {
            0: uniform_dist(np.log(1e-24), np.log(1e-20)),
            1: uniform_dist(0.5, 20.0),
            2: uniform_dist(1e-20, 1e-13),
            3 + ind_change: uniform_dist(0.0, 2 * np.pi),
            4 + ind_change: uniform_dist(-1, 1),
            5 + ind_change: uniform_dist(0.0, np.pi),
            6 + ind_change: uniform_dist(0.0, 2 * np.pi),
            7 + ind_change: uniform_dist(-1, 1),
        }

        if hasattr(self, "include_fddot") and self.include_fddot:
            default_priors[3] = uniform_dist(1e-32, 1e-17)

        if hasattr(self, "include_third") and self.include_third:
            default_priors[8] = uniform_dist(1.0, 20000.0)
            default_priors[9] = uniform_dist(0.0, np.pi * 2)
            default_priors[10] = uniform_dist(0.0001, 0.99)
            default_priors[11] = uniform_dist(0.001, 50.0)
            default_priors[12] = uniform_dist(0.0, 1.0)

        return {"gb": PriorContainer(default_priors)}

    @property
    def default_ndim(self):
        if hasattr(self, "include_third") and self.include_third:
            return 13

        elif hasattr(self, "include_fddot") and self.include_fddot:
            return 9

        else:
            return 8

    @property
    def default_fill_dict(self):

        if hasattr(self, "include_fddot") and self.include_fddot:
            return {
                "gb": {
                    "fill_inds": np.array([]),
                    "fill_values": np.array([]),
                    "ndim_full": 9,
                }
            }
        if hasattr(self, "include_third") and self.include_third:
            return {
                "gb": {
                    "fill_inds": np.array([3]),
                    "fill_values": np.array([0.0]),
                    "ndim_full": 13,
                }
            }
        else:
            return {
                "gb": {
                    "fill_inds": np.array([3]),
                    "fill_values": np.array([0.0]),
                    "ndim_full": 8,
                }
            }

    @property
    def default_injection_setup_kwargs(self):
        return dict(
            waveform_kwargs={"dt": 15.0, "N": None, "T": self.Tobs},
            noise_fn=get_sensitivity,
            noise_kwargs=[
                dict(
                    sens_fn="noisepsd_AE", model="SciRDv1", includewd=self.Tobs / YEAR,
                ),
                dict(
                    sens_fn="noisepsd_AE", model="SciRDv1", includewd=self.Tobs / YEAR,
                ),
            ],
            add_noise=False,
        )

    @property
    def default_parameter_transforms(self):
        parameter_transforms = {
            "gb": {
                0: (lambda x: np.exp(x)),
                1: (lambda x: x * 1e-3),
                5: (lambda x: np.arccos(x)),
                8: (lambda x: np.arcsin(x)),
            }
        }

        if hasattr(self, "include_third") and self.include_third:
            # transform from fraction of period for T2 to regular T2
            parameter_transforms["gb"][(12, 13)] = lambda x, y: (x, y * x)

        if not hasattr(self, "include_fddot"):
            # determine GR fddot
            parameter_transforms["gb"][(1, 2, 3)] = lambda f0, fdot, fddot: (
                f0,
                fdot,
                11.0 / 3.0 * fdot * fdot / f0,
            )

        return parameter_transforms

    @property
    def default_periodic(self):
        if hasattr(self, "include_fddot") and self.include_fddot:
            ind_change = 1
        else:
            ind_change = 0

        periodic = {
            3 + ind_change: 2 * np.pi,
            5 + ind_change: np.pi,
            6 + ind_change: 2 * np.pi,
        }
        if hasattr(self, "include_third") and self.include_third:
            periodic[9] = 2 * np.pi
            periodic[12] = 1.0

        return PeriodicContainer({"gb": periodic})

    @property
    def default_ndims(self):
        if hasattr(self, "include_fddot") and self.include_fddot:
            return [9]

        elif hasattr(self, "include_third") and self.include_third:
            return [13]

        return [8]

    @property
    def default_plot_labels(self):
        labels = [
            r"$A$",
            r"$f_0$ (mHz)",
            r"$\dot{f}_0$",
            r"$\phi_0$",
            r"cos$\iota$",
            r"$\psi$",
            r"$\lambda$",
            r"sin$\beta$",
        ]
        if self.include_third:
            labels += [r"$A_2$", r"$\bar{\omega}$", r"$e_2$", r"$P_2$", r"$T_2$"]

        if self.include_fddot:
            labels.insert(3, r"$\ddot{f}_0$")

        return labels

    def adjust_start_state(self):
        if hasattr(self, "include_fddot") and self.include_fddot:
            ind_change = 1
        else:
            ind_change = 0

        self.start_state.branches_coords["gb"][
            :, :, :, 3 + ind_change
        ] = self.start_state.branches_coords["gb"][:, :, :, 3 + ind_change] % (
            2 * np.pi
        )
        self.start_state.branches_coords["gb"][
            :, :, :, 5 + ind_change
        ] = self.start_state.branches_coords["gb"][:, :, :, 5 + ind_change] % (np.pi)
        self.start_state.branches_coords["gb"][
            :, :, :, 6 + ind_change
        ] = self.start_state.branches_coords["gb"][:, :, :, 6 + ind_change] % (
            2 * np.pi
        )

        if self.include_third:
            self.start_state.branches_coords["gb"][:, :, :, -1] = np.abs(
                self.start_state.branches_coords["gb"][:, :, :, -1]
            )
            self.start_state.branches_coords["gb"][:, :, :, -4] = np.abs(
                self.start_state.branches_coords["gb"][:, :, :, -4]
            )
            self.start_state.branches_coords["gb"][
                :, :, :, 9
            ] = self.start_state.branches_coords["gb"][:, :, :, 9] % (2 * np.pi)
            self.start_state.branches_coords["gb"][:, :, :, 10] = np.abs(
                self.start_state.branches_coords["gb"][:, :, :, 10]
            )

    @property
    def default_inner_product_kwargs(self):
        return dict(PSD="noisepsd_AE", PSD_kwargs={"includewd": self.Tobs / YEAR})

    def __init__(
        self,
        *args,
        include_third=False,
        include_fddot=False,
        use_gpu=False,
        f_arr=None,
        dt=10.0,
        Tobs=YEAR,
        fix_snr=None,
        inner_product_kwargs={},
        global_fit=False,
        adjust_betas=False,
        **kwargs,
    ):

        self.global_fit = global_fit
        if global_fit:
            raise NotImplementedError

        self.include_third = include_third
        self.include_fddot = include_fddot

        if self.include_third and self.include_fddot:
            raise ValueError(
                "Running a template with fddot and third body currently unvailable."
            )

        self.nchannels = 2

        if f_arr is None:
            Npts = int(Tobs / dt)
            Tobs = dt * Npts
            df = 1 / Tobs
            f_arr = np.arange(0.0, 1 / (2 * dt) + df, df)  # remove dc
        else:
            df = f_arr[1]
            Tobs = 1 / df

        self.Tobs = Tobs
        self.f_arr = f_arr
        if "likelihood_kwargs" not in kwargs:
            kwargs["likelihood_kwargs"] = {}
        kwargs["likelihood_kwargs"]["f_arr"] = f_arr
        kwargs["likelihood_kwargs"]["use_gpu"] = use_gpu

        if "sampler_kwargs" not in kwargs:
            kwargs["sampler_kwargs"] = {}

        gb = GBGPU(use_gpu=use_gpu, shift_ind=1)

        if "kwargs" in kwargs["sampler_kwargs"]:
            template_kwargs = kwargs["sampler_kwargs"]["kwargs"]
        else:
            template_kwargs = self.default_injection_setup_kwargs["waveform_kwargs"]

        template_kwargs_in = template_kwargs.copy()
        if "phase_marginalize" in template_kwargs_in:
            del template_kwargs_in["phase_marginalize"]

        # only temperary to help prepare data
        if "injection_params" in kwargs and (
            "data" not in kwargs or kwargs["data"] is None
        ):
            injection_params = kwargs["injection_params"]

            if "transform_container" in kwargs:
                transform = kwargs["transform_container"]
            else:
                transform = None
            if transform is None:
                transform = {
                    "gb": TransformContainer(
                        parameter_transforms=self.default_parameter_transforms["gb"],
                        fill_dict=self.default_fill_dict,
                    )
                }

            make_params = transform["gb"].transform_base_parameters(
                injection_params, copy=True
            )

            if len(make_params) > 9:
                template_kwargs_in["oversample"] = 8

            print("template_kwargs_in check", template_kwargs_in)
            print("template_kwargs check", template_kwargs)
            A_inj, E_inj = gb.inject_signal(*make_params, **template_kwargs_in)

            temp_inner_product_kwargs = self.default_inner_product_kwargs

            if inner_product_kwargs != {}:
                for key, value in inner_product_kwargs.items():
                    temp_inner_product_kwargs[key] = value

            temp_inner_product_kwargs["df"] = df

            self.snr_check = snr_check = snr(
                [A_inj, E_inj], **temp_inner_product_kwargs
            )

            if fix_snr is not None:
                raise NotImplementedError
                factor = fix_snr / snr_check

                make_params[0] *= factor
                injection_params[0] = np.log(make_params[0])

                A_inj, E_inj = gb.inject_signal(*make_params, **template_kwargs_in,)
                snr_check2 = snr([A_inj, E_inj], **temp_inner_product_kwargs)

                # print(snr_check2)

            # kwargs["sampler_kwargs"]["injection"] = injection_params
            kwargs["data"] = [A_inj, E_inj]

        """
        if "start_state" in kwargs and kwargs["start_state"] == "fisher":
            warnings.warn("Not using Fisher currently.")
            #raise NotImplementedError("Need to correct for test_inds")
            mean = injection_params[self.default_test_inds]

            fish_kwargs = template_kwargs_in.copy()
            fish_kwargs["N"] = 1024
            fish_kwargs["inds"] = self.default_test_inds
            fish_kwargs["parameter_transforms"] = self.default_parameter_transforms
            fisher = gb.fisher(np.array([injection_params]).T, **fish_kwargs).squeeze()
            cov = np.linalg.pinv(fisher)
            kwargs["mean_and_cov"] = [mean, cov]
        """
        """
        betas = make_ladder(
            self.default_ndim,
            ntemps=kwargs["sampler_kwargs"]["tempering_kwargs"]["ntemps"],
            Tmax=kwargs["sampler_kwargs"]["tempering_kwargs"]["Tmax"],
        )

        ind = np.where(self.snr_check * np.sqrt(betas) >= 0.001)[0][-1]

        betas = make_ladder(
            self.default_ndim,
            ntemps=kwargs["sampler_kwargs"]["tempering_kwargs"]["ntemps"] - 1,
            Tmax=1 / betas[ind],
        )
        betas = np.concatenate([betas, np.array([0.0])])
        """

        if adjust_betas:
            ntemps = kwargs["sampler_kwargs"]["tempering_kwargs"]["ntemps"]
            ntemps_over_5 = int(ntemps / 5)

            ntemps_1 = 4 * ntemps_over_5
            ntemps_2 = ntemps - ntemps_1

            betas_1 = np.logspace(-4, 0, ntemps_1)[::-1]
            betas_2 = np.logspace(-10, -4, ntemps_2 - 1, endpoint=False)[::-1]

            betas = np.concatenate([betas_1, betas_2, np.array([0.0])])

            kwargs["sampler_kwargs"]["tempering_kwargs"].pop("ntemps")
            kwargs["sampler_kwargs"]["tempering_kwargs"]["betas"] = betas

        print(kwargs["sampler_kwargs"]["tempering_kwargs"])
        SamplerGuide.__init__(self, *args, use_gpu=use_gpu, **kwargs)

        self.lnprob = gb

        #self.start_state = self.input_start_state

        self.injection_setup_kwargs["params"] = None
        self.lnprob.inject_signal(**self.injection_setup_kwargs)
        self.setup_sampler()
        #self.adjust_start_state()

        #if self.test_start_likelihood:
        #    self.perform_test_start_likelihood()

        # check = self.lnprob.get_ll(
        #    np.array([injection_params]), waveform_kwargs=template_kwargs
        # )


class EMRIGuide(SamplerGuide):
    @property
    def default_priors(self):

        default_priors = {
            0: uniform_dist(np.log(1e5), np.log(1e7)),
            1: uniform_dist(1.0, 100.0),
            2: uniform_dist(0.001, 0.95),
            3: uniform_dist(7.5, 16.0),
            4: uniform_dist(0.01, 0.75),
            5: uniform_dist(-0.999, 0.999),
            6: uniform_dist(0.01, 100.0),
            7: uniform_dist(0.0, np.pi),
            8: uniform_dist(0.0, 2 * np.pi),
            9: uniform_dist(0.0, np.pi),
            10: uniform_dist(0.0, 2 * np.pi),
            11: uniform_dist(0.0, 2 * np.pi),
            12: uniform_dist(0.0, 2 * np.pi),
            13: uniform_dist(0.0, 2 * np.pi),
        }

        if hasattr(self, "intrinsic") and self.intrinsic:
            keep = [0, 1, 2, 3, 4, 5, 11, 12, 13]
            for keep_i in keep:
                try:
                    del default_priors[keep_i]
                except KeyError:
                    pass

        if hasattr(self, "schwarzschild") and self.schwarzschild:
            keep = [0, 1, 3, 4, 11, 12]
            for keep_i in keep:
                try:
                    del default_priors[keep_i]
                except KeyError:
                    pass

        default_priors = PriorContainer(default_priors)

        return default_priors

    @property
    def default_ndim(self):
        intrinsic = False
        schwarzschild = False
        if hasattr(self, "intrinsic") and self.intrinsic:
            intrinsic = self.intrinsic

        if hasattr(self, "schwarzschild") and self.schwarzschild:
            schwarzschild = self.schwarzschild

        if intrinsic and schwarzschild:
            return 6
        elif intrinsic:
            return 9
        elif schwarzschild:
            return 11
        else:
            return 14

    @property
    def default_ndim_full(self):
        return self.default_ndim

    @property
    def default_test_inds(self):
        intrinsic = False
        schwarzschild = False
        if hasattr(self, "intrinsic") and self.intrinsic:
            intrinsic = self.intrinsic

        if hasattr(self, "schwarzschild") and self.schwarzschild:
            schwarzschild = self.schwarzschild

        if intrinsic and schwarzschild:
            return np.array([0, 1, 3, 4, 5, 11, 12])
        elif intrinsic:
            return np.array([0, 1, 2, 3, 4, 5, 11, 12, 13])
        elif schwarzschild:
            return np.array([0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        else:
            return np.arange(self.default_ndim)

    @property
    def default_fill_inds(self):
        return np.delete(np.arange(self.default_ndim), self.default_test_inds)

    @property
    def default_fill_values(self):
        return np.array(
            [
                1e6,
                1e1,
                0.5,
                12.0,
                0.25,
                0.5,
                1.0,
                np.pi / 3.0,
                np.pi / 4.0,
                np.pi / 5.0,
                np.pi / 6.0,
                0.0,
                0.0,
                0.0,
            ]
        )[self.default_fill_inds]

    @property
    def default_injection_setup_kwargs(self):
        return dict(
            waveform_kwargs={"dt": self.dt, "T": self.Tobs},
            noise_fn=get_sensitivity,
            noise_kwargs=[
                dict(
                    sens_fn="lisasens", model="SciRDv1", includewd=self.Tobs / YRSID_SI,
                ),
                dict(
                    sens_fn="lisasens", model="SciRDv1", includewd=self.Tobs / YRSID_SI,
                ),
            ],
            add_noise=False,
        )

    @property
    def default_parameter_transforms(self):
        parameter_transforms = {
            0: (lambda x: np.exp(x)),
            7: (lambda x: np.arccos(x)),
            9: (lambda x: np.arccos(x)),
        }

        if hasattr("intrinsic") and self.intrinsic:
            for i in [7, 9]:
                del parameter_transforms[i]

        return parameter_transforms

    @property
    def default_periodic(self):

        intrinsic = False
        schwarzschild = False
        if hasattr(self, "intrinsic") and self.intrinsic:
            intrinsic = self.intrinsic

        if hasattr(self, "schwarzschild") and self.schwarzschild:
            schwarzschild = self.schwarzschild

        if intrinsic and schwarzschild:
            return {
                4: 2 * np.pi,
                5: 2 * np.pi,
            }
        elif intrinsic:
            return {
                6: 2 * np.pi,
                7: 2 * np.pi,
                8: 2 * np.pi,
            }

        elif schwarzschild:
            return {
                6: 2 * np.pi,
                8: 2 * np.pi,
                9: 2 * np.pi,
                10: 2 * np.pi,
            }

        else:
            return {
                8: 2 * np.pi,
                10: 2 * np.pi,
                11: 2 * np.pi,
                12: 2 * np.pi,
                13: 2 * np.pi,
            }
        return periodic

    @property
    def default_plot_labels(self):
        labels = [
            r"ln$M$",
            r"$\mu$",
            r"$a$",
            r"$p_0$",
            r"$e_0$",
            r"$Y_0$",
            r"$d_L$",
            r"sin$q_S$",
            r"$\phi_S$",
            r"sin$q_K$",
            r"$\phi_K$",
            r"$\Phi_{\varphi, 0}$",
            r"$\Phi_{\theta, 0}$",
            r"$\Phi_{r, 0}$",
        ]

        return [labels[i] for i in self.test_inds]

    @property
    def default_inner_product_kwargs(self):
        return dict(PSD="lisasens", PSD_kwargs={"includewd": self.Tobs / YRSID_SI})

    def __init__(
        self,
        *args,
        use_gpu=False,
        f_arr=None,
        dt=10.0,
        Tobs=YRSID_SI,
        fix_snr=None,
        inner_product_kwargs={},
        **kwargs,
    ):

        self.nchannels = 2

        if f_arr is None:
            Npts = int(Tobs / dt)
            Tobs = dt * Npts
            df = 1 / Tobs
            f_arr = np.arange(0.0, 1 / (2 * dt) + df, df)  # remove dc
        else:
            df = f_arr[1]
            Tobs = 1 / df

        self.Tobs = Tobs
        self.f_arr = f_arr
        if "likelihood_kwargs" not in kwargs:
            kwargs["likelihood_kwargs"] = {}
        kwargs["likelihood_kwargs"]["f_arr"] = f_arr
        kwargs["likelihood_kwargs"]["use_gpu"] = use_gpu

        if "sampler_kwargs" not in kwargs:
            kwargs["sampler_kwargs"] = {}

            # kwargs["sampler_kwargs"]["plot_kwargs"]["corner_kwargs"][
            # "labels"
        # ] = self.default_plot_labels

        gb = GBGPU(use_gpu=use_gpu, shift_ind=1)

        if "kwargs" in kwargs["sampler_kwargs"]:
            template_kwargs = kwargs["sampler_kwargs"]["kwargs"]
        else:
            template_kwargs = self.default_injection_setup_kwargs["waveform_kwargs"]

        # only temperary to help prepare data
        if "injection_params" in kwargs and (
            "data" not in kwargs or kwargs["data"] is None
        ):
            injection_params = kwargs["injection_params"]

            if (
                "kwargs" in kwargs["sampler_kwargs"]
                and "parameter_transforms" in kwargs["sampler_kwargs"]["kwargs"]
            ):
                transform = kwargs["sampler_kwargs"]["kwargs"]["parameter_transforms"]
            else:
                transform = TransformContainer(self.default_parameter_transforms)

            make_params = transform.transform_base_parameters(injection_params)

            A_inj, E_inj = gb.inject_signal(*make_params, **template_kwargs)

            if fix_snr is not None:
                temp_inner_product_kwargs = self.default_inner_product_kwargs

                if inner_product_kwargs != {}:
                    for key, value in inner_product_kwargs.items():
                        temp_inner_product_kwargs[key] = value

                temp_inner_product_kwargs["df"] = df

                snr_check = snr([A_inj, E_inj], **temp_inner_product_kwargs)

                factor = fix_snr / snr_check

                make_params[0] *= factor
                injection_params[0] = np.log(make_params[0])

                A_inj, E_inj = gb.inject_signal(*make_params, **template_kwargs,)
                snr_check2 = snr([A_inj, E_inj], **temp_inner_product_kwargs)

                # print(snr_check2)

            kwargs["sampler_kwargs"]["injection"] = injection_params[
                self.default_test_inds
            ]
            kwargs["data"] = [A_inj, E_inj]

        if "start_state" in kwargs and kwargs["start_state"] == "fisher":
            mean = injection_params[self.default_test_inds]

            fish_kwargs = template_kwargs.copy()
            fish_kwargs["N"] = 1024
            fish_kwargs["inds"] = self.default_test_inds
            fish_kwargs["parameter_transforms"] = self.default_parameter_transforms
            fisher = gb.fisher(np.array([injection_params]).T, **fish_kwargs).squeeze()
            cov = np.linalg.pinv(fisher)
            kwargs["mean_and_cov"] = [mean, cov]

        SamplerGuide.__init__(self, *args, use_gpu=use_gpu, **kwargs)

        self.lnprob = gb

        self.injection_setup_kwargs["params"] = None
        self.lnprob.inject_signal(**self.injection_setup_kwargs)

        if self.test_start_likelihood:
            self.perform_test_start_likelihood()

        self.setup_sampler()
        self.adjust_start_state()
        # check = self.lnprob.get_ll(
        #    np.array([injection_params]), waveform_kwargs=template_kwargs
        # )


if __name__ == "__main__":

    from lisatools.utils.utility import AET
    from gbgpu.utils.constants import *

    use_gpu = False
    waveform_kwargs = dict(modes=modes, N=N, dt=dt, T=Tobs)

    fp = "test_new_code.h5"  # "/projects/b1095/mkatz/gb/test_new_code.h5"

    injection_params = injection_params = np.array(
        [
            np.log(amp),
            f0 * 1e3,
            np.log(fdot),
            fddot,
            phi0,
            np.cos(iota),
            psi,
            lam,
            np.sin(beta_sky),
            # A2,
            # omegabar,
            # e2,
            # P2,
            # T2,
        ]
    )

    sampler_kwargs = dict(
        kwargs=dict(waveform_kwargs=waveform_kwargs),
        fp=fp,
        resume=False,
        plot_iterations=1000,
        plot_source="gb",
        ntemps=ntemps,
        # get_d_h=True,
        # subset=int(nwalkers / 2),
        autocorr_multiplier=10000,
        # burn=1000,
    )

    kwargs = dict(
        dt=dt,
        start_state="fisher",
        Tobs=Tobs,
        sampler_kwargs=sampler_kwargs,
        likelihood_kwargs=dict(separate_d_h=False),
        injection_params=injection_params,
        waveform_kwargs=waveform_kwargs,
        verbose=True,
        use_gpu=use_gpu,
        start_factor=10000.0,
    )

    gb_guide = GBGuide(nwalkers * ntemps, **kwargs)
    gb_guide.run_sampler(thin=10, iterations=50000)

    # can add extra temperatures of 1 to have multiple temps accessing the target distribution
    ntemps_target_extra = 0
    # define max temperature (generally should be inf if you want to sample prior)
    Tmax = np.inf

    # not all walkers can fit in memory. subset says how many to do at one time
    subset = 4

    # set periodic parameters
    periodic = {
        4: 2 * np.pi,
        5: 2 * np.pi,
    }  # the indexes correspond to the index within test_inds

    # define sampling quantities
    nwalkers = 32  # per temperature
    ntemps = 1

    ndim_full = 14  # full dimensionality of inputs to waveform model

    # which of the injection parameters you actually want to sample over
    test_inds = np.array([0, 1, 3, 4, 11, 13])

    # ndim for sampler
    ndim = len(test_inds)

    # need to get values to fill arrays for quantities that we are not sampling
    fill_inds = np.delete(np.arange(ndim_full), test_inds)

    for i, (M, mu, p0, e0, dist) in tqdm.tqdm(
        enumerate(zip(M_arr, mu_arr, p0_arr, e0_arr, dist_arr))
    ):

        for j, (wave_gen, wave_kwargs, key) in enumerate(
            zip(wave_gens, wave_kwargs_list, keys)
        ):

            print(M, mu, e0, key)

            e0_temp = e0

            if M < 3e6 or mu != 10.0 and e0 != 0.7:
                continue

                # if mu == 10.0 and e0 >= 0.5:
                #    continue

                # if mu == 3.0 and e0 == 0.7:
                #    continue

                # if p0 <= 15.8:
                #    continue

                if key in ["22", "1e-2"]:
                    continue

                wave_gen.return_list = True
                if e0 > 0.7:
                    e0 = 0.7

                # define priors, it really can only do uniform cube at the moment
                priors = [
                    uniform_dist(np.log(1e5), np.log(1e7)),
                    uniform_dist(1.0, 100.0),
                    uniform_dist(6.0 + 2 * e0, 16.0 + 2 * e0),
                    uniform_dist(0.01, 0.75),
                    uniform_dist(0.0, 2 * np.pi),
                    uniform_dist(0.0, 2 * np.pi),
                ]

                injection_params = np.array(
                    [
                        np.log(M),
                        mu,
                        a,
                        p0,
                        e0,
                        Y0,
                        dist,
                        qS,
                        phiS,
                        qK,
                        phiK,
                        Phi_phi0,
                        Phi_theta0,
                        Phi_r0,
                    ]
                )

                injection_params = injection_params.copy()

                if j == 0:
                    injection_params_trans = transform_fn.transform_base_parameters(
                        injection_params
                    )
                    data = wave_gen_base(*injection_params_trans, **wave_kwargs_base)

                    snr_check = snr(data, **inner_product_kwargs)

                    data = [snr_goal / snr_check * data_i for data_i in data]
                    dist = snr_check / snr_goal * dist

                    try:
                        dist = dist.item()
                    except AttributeError:
                        pass

                if dist < 0.1:
                    continue

                if key == "1e-5":
                    continue

                fp2 = "new_fix_p_2_output_2yr_snr_{:d}_no_noise_{}_{}_{}_{}.h5".format(
                    int(snr_goal), M, mu, e0, key
                )

                if fp2 in os.listdir(folder):
                    continue

                test = wave_gen(*injection_params_trans, **wave_kwargs)

                with h5py.File(folder + fp, "r") as f:
                    cov = f[key]["cov"][i]

                fill_values = injection_params[fill_inds]

                # to store in sampler file, get injection points we are sampling
                injection_test_points = injection_params[test_inds]

                # instantiate the likelihood class
                nchannels = 2
                like = Likelihood(
                    wave_gen,
                    nchannels,
                    dt=dt,
                    parameter_transforms=transform_fn,
                    use_gpu=gpu_available,
                )

                # inject
                like.inject_signal(
                    data_stream=data,
                    noise_fn=get_sensitivity,
                    noise_kwargs=dict(sens_fn="lisasens"),
                    add_noise=False,
                )

                factor = 1
                start_state = injection_params[
                    np.newaxis, test_inds
                ] + factor * np.random.multivariate_normal(
                    np.zeros(len(test_inds)), cov, size=nwalkers
                )

                start_state_check = np.zeros((start_state.shape[0], ndim_full))

                start_state_check[:, test_inds] = start_state
                start_state_check[:, fill_inds] = fill_values

                check = [
                    like.get_ll(start_state_check[:subset], waveform_kwargs=wave_kwargs)
                ]

                # setup sampler
                sampler = PTEmceeSampler(
                    nwalkers,
                    ndim,
                    ndim_full,
                    like,
                    priors,
                    subset=subset,
                    kwargs={"waveform_kwargs": wave_kwargs},
                    test_inds=test_inds,
                    fill_values=fill_values,
                    ntemps=ntemps,
                    autocorr_multiplier=50,
                    autocorr_iter_count=50,
                    injection=injection_test_points,
                    plot_iterations=50,
                    plot_source="emri",
                    periodic=periodic,
                    fp=folder + fp2,
                )

                print(fp2, "start")
                thin_by = 1
                max_iter = 20000

                sampler.sample(
                    start_state,
                    iterations=max_iter,
                    progress=False,
                    skip_initial_state_check=False,
                    thin_by=thin_by,
                )

                print(fp2, "done")
