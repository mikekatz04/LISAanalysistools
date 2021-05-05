from abc import ABC

import numpy as np
import h5py

from lisatools.utils.utility import AET
from lisatools.sampling.samplingguide import MBHGuide

# from ldc.waveform.lisabeta import FastMBHB
from lisatools.sampling.utility import ModifiedHDFBackend
from lisatools.sampling.stopping import SNRStopping, SearchConvergeStopping

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

    def run(self, progress=False, verbose=False):
        for i, module in enumerate(self.module_list):
            if verbose:
                print_str = "starting module {}".format(i)
                if module.name is not None:
                    print_str += ": {}".format(module.name)
                print(print_str)

            module.update_module(self.info_manager)
            module.run_module()

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
        **kwargs
    ):
        self.nwalkers = nwalkers
        self.ntemps = ntemps
        self.nwalkers_all = nwalkers * ntemps
        self.use_gpu = use_gpu

        self.amp_phase_kwargs = {"run_phenomd": True}

        self.fp = fp

        self.snr_stopping = snr_stopping
        self.n_iter_stop = n_iter_stop

        self.search = search

    def update_information(self, info_manager, fp, *args, **kwargs):
        if self.search:
            info_manager.fp_search_init = fp
        else:
            info_manager.fp_pe = fp

    def update_module(self, info_manager, *args, **kwargs):
        self.info_manager = info_manager

        self.template_kwargs = dict(
            tBase=0.0,
            length=1024,
            freqs=xp.asarray(info_manager.fd[1:]),
            t_obs_start=1.0,
            t_obs_end=0.0,
            modes=[(2, 2)],
            direct=False,
            compress=True,
        )

        if self.n_iter_stop is not None:
            stop = SearchConvergeStopping(n_iters=self.n_iter_stop)
            stopping_iter = 10
        elif self.snr_stopping is not None:
            stop = SNRStopping(snr_limit=self.snr_stopping)
            stopping_iter = 10
        else:
            stop = None
            stopping_iter = -1

        if self.search:
            burn = None
            resume = False

        else:
            burn = 1000
            resume = False

        print("\n\ncheck it\n\n")
        self.sampler_kwargs = dict(
            lnlike_kwargs=dict(waveform_kwargs=self.template_kwargs),
            fp=self.fp,
            resume=resume,
            plot_iterations=100,
            stopping_iter=stopping_iter,
            stopping_fn=stop,
            plot_source="mbh",
            ntemps=self.ntemps,
            get_d_h=True,
            subset=int(self.nwalkers / 2),
            autocorr_multiplier=10000,
            burn=burn,
        )

        if self.search:
            start_points = None
        else:
            reader = ModifiedHDFBackend(self.info_manager.fp_search_init)
            log_prob = reader.get_log_prob().flatten()
            start_inds = np.argsort(log_prob)
            uni, inds = np.unique(log_prob[start_inds], return_index=True)
            start_inds = start_inds[inds[-int(self.ntemps * self.nwalkers) :]]

            print(log_prob[start_inds])

            ndim = 11
            ndim_full = 13
            test_inds = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11])

            # start_points = reader.get_chain().reshape(-1, ndim)[start_inds]
            start_points = reader.get_chain()[-1, :2].reshape(-1, ndim)

            relbin_template = np.zeros(ndim_full)
            relbin_template[test_inds] = start_points[-1]

        self.guide_kwargs = dict(
            dt=info_manager.dt,
            start_points=start_points,
            Tobs=info_manager.T * info_manager.dt,
            f_arr=info_manager.fd,
            sampler_kwargs=self.sampler_kwargs,
            likelihood_kwargs=dict(separate_d_h=True),
            data=info_manager.data,
            waveform_kwargs=self.template_kwargs,
            verbose=True,
            use_gpu=use_gpu,
            amp_phase_kwargs=self.amp_phase_kwargs,
        )

        self.mbh_guide = MBHGuide(self.nwalkers * self.ntemps, **self.guide_kwargs)

        # TODO: remove
        # self.mbh_guide.lnprob.template_model.d_d = 2 * 7.5e4
        print(self.mbh_guide.lnprob.template_model.d_d)

    def run_module(self, *args, **kwargs):
        self.mbh_guide.run_sampler(thin=5, iterations=100000, progress=False)
        self.update_information(self.info_manager, self.fp)
        # del self.mbh_guide


class MBHRelBinSearch(PipelineModule):
    def initialize_module(
        self, fp_search_rel_bin, nwalkers, ntemps, use_gpu=False, **kwargs
    ):
        self.nwalkers = nwalkers
        self.ntemps = ntemps
        self.nwalkers_all = nwalkers * ntemps
        self.use_gpu = use_gpu

        self.fp_search_rel_bin = fp_search_rel_bin

        self.amp_phase_kwargs = {"run_phenomd": True}

    def update_information(self, info_manager, fp_search, *args, **kwargs):
        info_manager.fp_search = fp_search

    def update_module(self, info_manager, *args, **kwargs):
        self.info_manager = info_manager

        reader = ModifiedHDFBackend(self.info_manager.fp_search_init)
        log_prob = reader.get_log_prob().flatten()
        start_inds = np.argsort(log_prob)
        uni, inds = np.unique(log_prob[start_inds], return_index=True)
        start_inds = start_inds[inds[-int(self.ntemps * self.nwalkers) :]]

        print(log_prob[start_inds])

        ndim = 11
        ndim_full = 13
        test_inds = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11])

        start_points = reader.get_chain().reshape(-1, ndim)[start_inds]

        relbin_template = np.zeros(ndim_full)
        relbin_template[test_inds] = start_points[-1]

        self.template_kwargs = dict(
            tBase=0.0,
            t_obs_start=1.0,
            t_obs_end=0.0,
            modes=[(2, 2)],
            direct=True,
            compress=True,
        )

        # relbin_template = "multi"
        from lisatools.sampling.utility import rel_bin_update

        update_kwargs = dict(
            template_gen_kwargs=self.template_kwargs.copy(),
            noise_kwargs_AE={},
            noise_kwargs_T={},
        )

        ll_stop = SearchConvergeStopping(n_iters=30)
        print("\n\ncheck it  22222222\n\n")
        self.sampler_kwargs = dict(
            lnlike_kwargs=dict(waveform_kwargs=self.template_kwargs),
            fp=self.fp_search_rel_bin,
            resume=False,
            plot_iterations=-1,
            stopping_iter=20,
            stopping_fn=ll_stop,
            plot_source="mbh",
            ntemps=self.ntemps,
            get_d_h=True,
            autocorr_multiplier=10000,
            update_kwargs=update_kwargs,
            update_fn=rel_bin_update,
            update=1000,
        )

        self.guide_kwargs = dict(
            dt=info_manager.dt,
            start_points=start_points,
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
            use_gpu=use_gpu,
            amp_phase_kwargs=self.amp_phase_kwargs,
        )

        self.mbh_guide = MBHGuide(self.nwalkers * self.ntemps, **self.guide_kwargs)

        # TODO: remove
        # self.mbh_guide.lnprob.template_model.base_d_d = 2 * 7.5e4

    def run_module(self, *args, **kwargs):
        self.mbh_guide.run_sampler(thin=10, iterations=100000, progress=True)
        self.update_information(self.info_manager, self.fp_search_rel_bin)


class MBHRelBinPE(PipelineModule):
    def initialize_module(
        self, fp_pe_rel_bin, nwalkers, ntemps, use_gpu=False, **kwargs
    ):
        self.nwalkers = nwalkers
        self.ntemps = ntemps
        self.nwalkers_all = nwalkers * ntemps
        self.use_gpu = use_gpu

        self.fp_pe_rel_bin = fp_pe_rel_bin

        self.amp_phase_kwargs = {"run_phenomd": True}

    def update_information(self, info_manager, fp_pe, *args, **kwargs):
        info_manager.fp_pe = fp_pe

    def update_module(self, info_manager, *args, **kwargs):
        self.info_manager = info_manager

        reader = ModifiedHDFBackend(self.info_manager.fp_search)
        log_prob = reader.get_log_prob().flatten()
        start_inds = np.argsort(log_prob)
        uni, inds = np.unique(log_prob[start_inds], return_index=True)
        start_inds = start_inds[inds[-int(self.ntemps * self.nwalkers) :]]

        print(log_prob[start_inds])

        ndim = 11
        ndim_full = 13
        test_inds = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11])

        start_points = reader.get_chain().reshape(-1, ndim)[start_inds]

        relbin_template = np.zeros(ndim_full)
        relbin_template[test_inds] = start_points[-1]

        self.template_kwargs = dict(
            tBase=0.0,
            t_obs_start=1.0,
            t_obs_end=0.0,
            modes=[(2, 2)],
            direct=True,
            compress=True,
        )

        # relbin_template = "multi"
        from lisatools.sampling.utility import rel_bin_update

        update_kwargs = dict(
            template_gen_kwargs=self.template_kwargs.copy(),
            noise_kwargs_AE={},
            noise_kwargs_T={},
        )

        self.sampler_kwargs = dict(
            lnlike_kwargs=dict(waveform_kwargs=self.template_kwargs),
            fp=self.fp_pe_rel_bin,
            resume=False,
            plot_iterations=50,
            # stopping_iter=10,
            # stopping_fn=ll_stop,
            plot_source="mbh",
            ntemps=self.ntemps,
            get_d_h=True,
            autocorr_multiplier=10000,
            update_kwargs=update_kwargs,
            update_fn=rel_bin_update,
            update=1000,
            burn=1000,
        )

        self.guide_kwargs = dict(
            dt=info_manager.dt,
            start_points=start_points,
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
            use_gpu=use_gpu,
            amp_phase_kwargs=self.amp_phase_kwargs,
        )

        self.mbh_guide = MBHGuide(self.nwalkers * self.ntemps, **self.guide_kwargs)
        breakpoint()

    def run_module(self, *args, **kwargs):
        self.mbh_guide.run_sampler(thin=200, iterations=100000, progress=True)
        self.update_information(self.info_manager, self.fp_pe_rel_bin)
