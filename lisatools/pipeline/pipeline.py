from abc import ABC

import numpy as np
import h5py

from lisatools.utils.utility import AET
from lisatools.sampling.samplingguide import MBHGuide

from eryn.state import State

# from ldc.waveform.lisabeta import FastMBHB
from eryn.backends import HDFBackend
from eryn.moves import StretchMove
from lisatools.sampling.stopping import SNRStopping, SearchConvergeStopping
from lisatools.sampling.moves.skymodehop import SkyMove

try:
    import cupy as cp

    gpu_available = True

except (ModuleNotFoundError, ImportError):
    import numpy as np

    gpu_available = False

use_gpu = gpu_available

xp = cp if use_gpu else np


class InfoManager:
    def __init__(self, name=None, data=None, dt=None, T=None, fd=None, **kwargs):
        self.name = name

        self.dt, self.T, self.fd = dt, T, fd
        self.data = data
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self.nchannels = len(data)
        self.data_length = len(data[0])
        self._data = data

    def update_info(self, data, *args, **kwargs):
        self.data = data


class PipelineModule(ABC):
    def __init__(self, name=None):
        self.name = name

    @classmethod
    def initialize_module(self, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def update_module(self, info_manager, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def run_module(self, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def update_information(self, *args, **kwargs):
        raise NotImplementedError


class PipelineGuide:
    def __init__(self, info_manager, module_list):
        self.module_list = module_list
        self.info_manager = info_manager

    def run(self, progress=False, verbose=False, **update_kwargs):
        for i, module in enumerate(self.module_list):
            if verbose:
                print_str = "starting module {}".format(i)
                if module.name is not None:
                    print_str += ": {}".format(module.name)
                print(print_str)

            module.update_module(self.info_manager, **update_kwargs)
            module.run_module(progress=progress)

            if verbose:
                print_str = "finished module {}".format(i)
                if module.name is not None:
                    print_str += ": {}".format(module.name)
                print(print_str)

            # TODO: add output manager


class MBHBase(PipelineModule):
    def initialize_module(
        self,
        fp,
        nwalkers,
        ntemps,
        search=True,
        snr_stopping=None,
        n_iter_stop=None,
        use_gpu=False,
        run_phenomd=True,
        set_d_d_zero=False,
        injection=None,
        **kwargs,
    ):
        self.nwalkers = nwalkers
        self.ntemps = ntemps
        self.nwalkers_all = nwalkers * ntemps
        self.use_gpu = use_gpu
        self.set_d_d_zero = set_d_d_zero
        self.amp_phase_kwargs = {"run_phenomd": run_phenomd}

        self.fp = fp

        self.snr_stopping = snr_stopping
        self.n_iter_stop = n_iter_stop

        self.search = search
        self.injection = injection

    def update_information(self, info_manager, fp, *args, **kwargs):
        if self.search:
            info_manager.fp_search_init = fp
        else:
            info_manager.fp_pe = fp

    def update_module(self, info_manager, *args, **kwargs):
        self.info_manager = info_manager

        from bbhx.utils.constants import YRSID_SI

        self.template_kwargs = dict(
            length=1024,
            freqs=xp.asarray(info_manager.fd[1:]),
            t_obs_start=1.0,
            t_obs_end=3600.0 / YRSID_SI,
            modes=None,
            direct=False,
            compress=True,
        )

        if "template_kwargs" in kwargs:
            self.template_kwargs = {**self.template_kwargs, **kwargs["template_kwargs"]}

        if self.n_iter_stop is not None:
            stop = SearchConvergeStopping(n_iters=self.n_iter_stop)
            stopping_iter = 30
        elif self.snr_stopping is not None:
            stop = SNRStopping(snr_limit=self.snr_stopping, verbose=True)
            stopping_iter = 30
        else:
            stop = None
            stopping_iter = -1

        if self.search:
            burn = None
            resume = False

        else:
            burn = 1000
            resume = False

        plot_name = self.fp[:-3] + "_output"
        # TODO: ADD SUBSET
        self.sampler_kwargs = dict(
            tempering_kwargs={"ntemps": self.ntemps, "Tmax": np.inf},
            moves=[
                (StretchMove(), 0.85),
                (SkyMove(which="both"), 0.04),
                (SkyMove(which="lat"), 0.08),
                (SkyMove(which="long"), 0.03),
            ],
            args=None,
            kwargs=self.template_kwargs,
            backend=self.fp,
            vectorize=True,
            autocorr_multiplier=10000,  # TODO: adjust this to 50
            plot_iterations=15,
            plot_name=plot_name,
            stopping_fn=stop,
            stopping_iterations=stopping_iter,
            info={},
            branch_names=["mbh"],
            # subset=int(self.nwalkers / 2),
        )

        ndim = 11
        if self.search:
            start_state = None
        elif self.injection is None:
            reader = HDFBackend(self.info_manager.fp_search_init)
            log_prob = reader.get_log_prob().flatten()
            start_inds = np.argsort(log_prob)
            uni, inds = np.unique(log_prob[start_inds], return_index=True)
            start_inds = start_inds[inds[-int(self.ntemps * self.nwalkers) :]]

            print(log_prob[start_inds])
            start_points = reader.get_chain()["mbh"].reshape(-1, ndim)[start_inds]

            start_state = State(
                {"mbh": start_points.reshape(self.ntemps, self.nwalkers, 1, ndim)}
            )
            # start_state = reader.get_chain()['mbh'][-1, :2].reshape(-1, ndim)

            relbin_template = start_points[-1]

        else:
            if not isinstance(self.injection, np.ndarray):
                raise ValueError("injection must be np.ndarray")
            factor = 1e-8

            start_points = self.injection * (
                1 + factor * np.random.randn(self.ntemps, self.nwalkers, 1, 11)
            )
            start_state = State(
                {"mbh": start_points.reshape(self.ntemps, self.nwalkers, 1, ndim)}
            )
            relbin_template = self.injection.copy()

        if "priors" in kwargs:
            priors = kwargs["priors"]
        else:
            priors = None

        print(info_manager.T, info_manager.dt)
        self.guide_kwargs = dict(
            dt=info_manager.dt,
            start_state=start_state,
            Tobs=info_manager.T * info_manager.dt,
            f_arr=info_manager.fd,
            sampler_kwargs=self.sampler_kwargs,
            likelihood_kwargs=dict(separate_d_h=True, subset=int(self.nwalkers / 2.0)),
            data=info_manager.data,
            multi_mode_start=True,
            waveform_kwargs=self.template_kwargs,
            verbose=True,
            use_gpu=use_gpu,
            amp_phase_kwargs=self.amp_phase_kwargs,
            priors=priors,
            global_fit=False,
        )

        self.mbh_guide = MBHGuide(self.nwalkers, **self.guide_kwargs)

        # TODO: remove
        if self.set_d_d_zero:
            self.mbh_guide.lnprob.template_model.d_d = 0.0

        print(self.mbh_guide.lnprob.template_model.d_d)

    def run_module(self, *args, progress=False, **kwargs):
        print(progress, "progress")
        self.mbh_guide.run_sampler(
            self.mbh_guide.start_state, 10000, thin_by=5, progress=progress
        )
        self.update_information(self.info_manager, self.fp)
        # del self.mbh_guide


class MBHRelBinSearch(PipelineModule):
    def initialize_module(
        self,
        fp_search_rel_bin,
        nwalkers,
        ntemps,
        n_iter_stop=None,
        n_iter_update=None,
        use_gpu=False,
        set_d_d_zero=False,
        **kwargs,
    ):
        self.nwalkers = nwalkers
        self.ntemps = ntemps
        self.nwalkers_all = nwalkers * ntemps
        self.use_gpu = use_gpu
        self.set_d_d_zero = set_d_d_zero

        self.fp_search_rel_bin = fp_search_rel_bin

        self.amp_phase_kwargs = {"run_phenomd": True}

    def update_information(self, info_manager, fp_search, *args, **kwargs):
        info_manager.fp_search_rel_bin = fp_search_rel_bin

    def update_module(self, info_manager, *args, **kwargs):
        self.info_manager = info_manager

        self.template_kwargs = dict(
            t_obs_start=1.0, t_obs_end=0.0, modes=None, direct=True, compress=True,
        )

        # relbin_template = "multi"
        from lisatools.sampling.utility import RelBinUpdate

        update_kwargs = dict(
            template_gen_kwargs=self.template_kwargs.copy(),
            noise_kwargs_AE={},
            noise_kwargs_T={},
        )

        rel_bin_update = RelBinUpdate(update_kwargs, set_d_d_zero=self.set_d_d_zero)

        ll_stop = SearchConvergeStopping(n_iters=30, verbose=True)

        plot_name = self.fp_search_rel_bin[:-3] + "_output"

        exsnr_limit = 380.0
        self.fp_search_rel_bin = (
            self.fp_search_rel_bin[:-3] + f"_{int(exsnr_limit)}_limit.h5"
        )
        self.sampler_kwargs = dict(
            tempering_kwargs={"ntemps": self.ntemps, "Tmax": np.inf},
            moves=[
                (StretchMove(), 0.85),
                (SkyMove(which="both"), 0.04),
                (SkyMove(which="lat"), 0.08),
                (SkyMove(which="long"), 0.03),
            ],
            args=None,
            kwargs=self.template_kwargs,
            backend=self.fp_search_rel_bin,
            vectorize=True,
            autocorr_multiplier=10000,  # TODO: adjust this to 50
            plot_iterations=-1,
            plot_name=plot_name,
            stopping_fn=ll_stop,
            stopping_iterations=250,
            update_fn=rel_bin_update,
            update_iterations=1000,
            info={},
            branch_names=["mbh"],
            # subset=int(self.nwalkers / 2),
        )

        # TODO: change this
        ndim = 11

        reader = HDFBackend(self.info_manager.fp_search_init)
        log_prob = reader.get_log_prob().reshape(reader.get_log_prob().shape[0], -1)
        exsnr = reader.get_blobs().reshape(reader.get_blobs().shape[0], -1)

        ind_start = np.where(exsnr.max(axis=1) < exsnr_limit)[0][-1] + 1
        print("ind_start:", ind_start)

        exsnr = exsnr[:ind_start].flatten()
        log_prob = log_prob[:ind_start].flatten()
        start_inds = np.argsort(log_prob)
        uni, inds = np.unique(log_prob[start_inds], return_index=True)
        start_inds = start_inds[inds[-int(self.ntemps * self.nwalkers) :]]

        print(log_prob[start_inds])

        start_points = reader.get_chain()["mbh"].reshape(-1, ndim)[start_inds]

        start_state = State(
            {"mbh": start_points.reshape(self.ntemps, self.nwalkers, 1, ndim)}
        )
        # start_state = reader.get_chain()['mbh'][-1, :2].reshape(-1, ndim)

        relbin_template = start_points[-1]

        if "priors" in kwargs:
            priors = kwargs["priors"]
        else:
            priors = None

        self.guide_kwargs = dict(
            dt=info_manager.dt,
            start_state=start_state,
            Tobs=info_manager.T * info_manager.dt,
            f_arr=info_manager.fd,
            sampler_kwargs=self.sampler_kwargs,
            likelihood_kwargs=dict(separate_d_h=True),
            data=info_manager.data,
            waveform_kwargs=self.template_kwargs,
            multi_mode_start=True,
            verbose=False,
            relbin=True,
            relbin_template=relbin_template,
            use_gpu=use_gpu,
            amp_phase_kwargs=self.amp_phase_kwargs,
            priors=priors,
            global_fit=False,
        )

        self.mbh_guide = MBHGuide(self.nwalkers, **self.guide_kwargs)

        if self.set_d_d_zero:
            self.mbh_guide.lnprob.template_model.base_d_d = 0.0
        print(self.mbh_guide.lnprob.template_model.base_d_d)

    def run_module(self, *args, progress=False, **kwargs):
        print(progress, "progress")
        self.mbh_guide.run_sampler(
            self.mbh_guide.start_state, 10000, thin_by=5, progress=progress
        )
        self.update_information(self.info_manager, self.fp_search_rel_bin)


class MBHRelBinPE(PipelineModule):
    def initialize_module(
        self,
        fp_pe_rel_bin,
        nwalkers,
        ntemps,
        use_gpu=False,
        run_phenomd=True,
        set_d_d_zero=False,
        injection=None,
        **kwargs,
    ):
        self.nwalkers = nwalkers
        self.ntemps = ntemps
        self.nwalkers_all = nwalkers * ntemps
        self.use_gpu = use_gpu
        self.set_d_d_zero = set_d_d_zero
        self.injection = injection

        self.fp_pe_rel_bin = fp_pe_rel_bin

        self.amp_phase_kwargs = {"run_phenomd": run_phenomd}

    def update_information(self, info_manager, fp_pe, *args, **kwargs):
        info_manager.fp_pe = fp_pe

    def update_module(self, info_manager, *args, **kwargs):
        self.info_manager = info_manager

        ndim = 11

        if self.injection is None:
            reader = HDFBackend(self.info_manager.fp_search_rel_bin)
            log_prob = reader.get_log_prob().flatten()
            start_inds = np.argsort(log_prob)
            uni, inds = np.unique(log_prob[start_inds], return_index=True)
            start_inds = start_inds[inds[-int(self.ntemps * self.nwalkers) :]]

            print(log_prob[start_inds])

            start_points = reader.get_chain()["mbh"].reshape(-1, ndim)[start_inds]

            start_state = State(
                {"mbh": start_points.reshape(self.ntemps, self.nwalkers, 1, ndim)}
            )

            print(log_prob[start_inds])

            relbin_template = start_points[-1].copy()
        else:
            if not isinstance(self.injection, np.ndarray):
                raise ValueError("injection must be np.ndarray")
            factor = 1e-9

            start_points = self.injection * (
                1 + factor * np.random.randn(self.ntemps, self.nwalkers, 1, 11)
            )
            start_state = State(
                {"mbh": start_points.reshape(self.ntemps, self.nwalkers, 1, ndim)}
            )
            relbin_template = self.injection.copy()

        from bbhx.utils.constants import YRSID_SI

        self.template_kwargs = dict(
            t_obs_start=1.0,
            t_obs_end=3600.0 / YRSID_SI,
            modes=None,
            direct=True,
            compress=True,
        )

        # relbin_template = "multi"
        from lisatools.sampling.utility import RelBinUpdate

        update_kwargs = dict(
            template_gen_kwargs=self.template_kwargs.copy(),
            noise_kwargs_AE={},
            noise_kwargs_T={},
        )

        rel_bin_update = RelBinUpdate(update_kwargs, set_d_d_zero=self.set_d_d_zero)

        plot_name = self.fp_pe_rel_bin[:-3] + "_output"

        self.sampler_kwargs = dict(
            tempering_kwargs={"ntemps": self.ntemps, "Tmax": np.inf},
            moves=[
                (StretchMove(), 0.85),
                (SkyMove(which="both"), 0.04),
                (SkyMove(which="lat"), 0.08),
                (SkyMove(which="long"), 0.03),
            ],
            args=None,
            kwargs=self.template_kwargs,
            backend=self.fp_pe_rel_bin,
            vectorize=True,
            autocorr_multiplier=10000,  # TODO: adjust this to 50
            plot_iterations=500,
            plot_name=plot_name,
            stopping_iterations=-1,
            update_fn=rel_bin_update,
            update_iterations=1000,
            info={},
            branch_names=["mbh"],
        )

        self.guide_kwargs = dict(
            dt=info_manager.dt,
            start_state=start_state,
            Tobs=info_manager.T * info_manager.dt,
            f_arr=info_manager.fd,
            sampler_kwargs=self.sampler_kwargs,
            likelihood_kwargs=dict(separate_d_h=True),
            data=info_manager.data,
            waveform_kwargs=self.template_kwargs,
            multi_mode_start=True,
            verbose=True,
            relbin=True,
            relbin_template=relbin_template,
            relbin_args=(256,),
            relbin_kwargs=dict(template_gen_kwargs=self.template_kwargs),
            use_gpu=use_gpu,
            amp_phase_kwargs=self.amp_phase_kwargs,
            global_fit=False,
        )

        self.mbh_guide = MBHGuide(self.nwalkers, **self.guide_kwargs)

        if self.set_d_d_zero:
            self.mbh_guide.lnprob.template_model.base_d_d = 0.0

        print(self.mbh_guide.lnprob.template_model.base_d_d)

    def run_module(self, *args, progress=False, **kwargs):
        print(progress, "progress")
        self.mbh_guide.run_sampler(
            self.mbh_guide.start_state, 50000, burn=4000, thin_by=10, progress=progress
        )
        self.update_information(self.info_manager, self.fp_pe_rel_bin)
