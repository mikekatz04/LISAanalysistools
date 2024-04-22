import time
import numpy as np
import shutil
from copy import deepcopy
from eryn.state import State
from eryn.ensemble import EnsembleSampler
from eryn.prior import ProbDistContainer, uniform_dist
from lisatools.glitch import tdi_glitch_XYZ1
import corner
from lisatools.utils.utility import AET
from global_fit_input.global_fit_settings import get_global_fit_settings
import os
from bbhx.waveformbuild import BBHWaveformFD
from bbhx.waveforms.phenomhm import PhenomHMAmpPhase
from bbhx.response.fastfdresponse import LISATDIResponse
from bbhx.likelihood import Likelihood as MBHLikelihood
from bbhx.likelihood import HeterodynedLikelihood
from bbhx.utils.constants import *
from bbhx.utils.transform import *
from eryn.backends import HDFBackend
from lisatools.utils.utility import generate_noise_fd
import pickle 

from eryn.moves import StretchMove
from lisatools.sampling.likelihood import Likelihood
from lisatools.sampling.moves.skymodehop import SkyMove
from lisatools.sampling.stopping import SearchConvergeStopping

from lisatools.sensitivity import get_sensitivity
from lisatools.sampling.utility import HeterodynedUpdate

from eryn.utils import TransformContainer

np.random.seed(111222)

try:
    import cupy as cp

    # set GPU device
    gpu_available = True

except (ImportError, ModuleNotFoundError) as e:
    import numpy as xp

    gpu_available = False

# whether you are using
use_gpu = True

if use_gpu is False:
    xp = np

if use_gpu and not gpu_available:
    raise ValueError("Requesting gpu with no GPU available or cupy issue.")


def search_likelihood_wrap(x, wave_gen, initial_t_vals, end_t_vals, d_d_vals, t_ref_lims, transform_fn, like_args, mbh_kwargs):
    x_in = transform_fn.both_transforms(x)

    data_index = noise_index = (np.searchsorted(t_ref_lims, x[:, -1], side="right") - 1).astype(np.int32)
    # wave_gen.amp
    
    wave_gen.amp_phase_gen.initial_t_val = initial_t_vals[data_index][:, None]
    t_obs_start = initial_t_vals[data_index] / YRSID_SI
    t_obs_end = end_t_vals[data_index] / YRSID_SI
    wave_gen.d_d = cp.asarray(d_d_vals[data_index])
    fd, all_data, psd, df = like_args
    ll = wave_gen.get_direct_ll(fd, all_data, psd, df, *x_in.T, noise_index=noise_index, data_index=data_index, t_obs_start=t_obs_start, t_obs_end=t_obs_end, **mbh_kwargs).real.get()
    
    return ll


# function call
def run_mbh_search(gpu, settings, rank, time_split, total_time_splits, best_points, num_run):

    cp.cuda.runtime.setDevice(gpu)

    mbh_info = settings["mbh"]

    stop = SearchConvergeStopping(**mbh_info["search_info"]["stop_kwargs"])
    
    ntemps = mbh_info["search_info"]["ntemps"]
    nwalkers = mbh_info["search_info"]["nwalkers"]

    mbh_kwargs = mbh_info["search_info"]["mbh_kwargs"]

    transform_fn = mbh_info["transform"]
    priors_input = mbh_info["priors"]["mbh"]

    periodic = mbh_info["periodic"]
    
    try:
        del sampler
        del all_data
        del psd
        del wave_gen
        cp.get_default_memory_pool().free_all_blocks()
        
    except NameError:
        pass

    # wave generating class
    wave_gen = BBHWaveformFD(**mbh_info["initialize_kwargs"])

    t, X, Y, Z = (
        settings["general"]["t"].squeeze().copy(),
        settings["general"]["X"].squeeze().copy(),
        settings["general"]["Y"].squeeze().copy(),
        settings["general"]["Z"].squeeze().copy(),
    )
    dt = settings["general"]["dt"]

    t_lims = np.linspace(0.0, (t.shape[0] - 1) * dt, 2 * 365)

    # omits first day and last day
    t_ref_lims = t_lims[1:-1]

    num_t_ref_bins = len(t_ref_lims) - 1
    split_val = num_t_ref_bins // total_time_splits + 1

    split_inds = np.split(np.arange(num_t_ref_bins), np.arange(split_val, num_t_ref_bins, split_val))[time_split]

    start_here = split_inds[0] - 1 if split_inds[0] > 0 else 0
    end_here = split_inds[-1] + 1 if split_inds[-1] < num_t_ref_bins - 1 else num_t_ref_bins - 1
    num_here = end_here - start_here + 1

    t_ref_lims_here = t_ref_lims[start_here:end_here + 1]
    priors_in = deepcopy(priors_input.priors_in)
    priors_in[10] = uniform_dist(t_ref_lims_here[0], t_ref_lims_here[-1])
    priors = {"mbh": ProbDistContainer(priors_in)}

    A, E, T = AET(X, Y, Z)

    # A[:] = 0.0
    # E[:] = 0.0
    # T[:] = 0.0
    

    if len(best_points) > 0:
        # fucking dt
        Af, Ef, Tf = (
            np.fft.rfft(A) * dt,
            np.fft.rfft(E) * dt,
            np.fft.rfft(T) * dt,
        )

        check = []
        freqs_check = np.fft.rfftfreq(len(A), dt)
        
        fd_tmp = np.fft.rfftfreq(len(X), dt)
        fd_tmp[0] = fd_tmp[1]

        psd_tmp = get_sensitivity(fd_tmp, sens_fn="noisepsd_AE", model="sangria", includewd=1)

        # last_ll = -1/2 * 4 * settings["general"]["df"] * np.sum((Af.conj() * Af + Ef.conj() * Ef) / psd_tmp)

        for i in range(len(best_points)):

            psd_tmp = get_sensitivity(fd_tmp, sens_fn="noisepsd_AE", model="sangria", includewd=1)

            remove_point = best_points[i:i+1]

            remove_point_in = transform_fn.both_transforms(
                    remove_point, return_transpose=True
                )

            # get XYZ
            data_channels_AET = wave_gen(
                *remove_point_in,
                freqs=cp.asarray(fd_tmp),
                modes=[(2, 2)],
                direct=False,
                fill=True,
                squeeze=True,
                length=1024,
            )[0]

            # psd_tmp[np.abs(data_channels_AET[0]).argmax().get() - 100000 : np.abs(data_channels_AET[0]).argmax().get() + 100000] = 1e300

            # h_h = 4 * settings["general"]["df"] * np.sum((data_channels_AET[0].get().conj() * data_channels_AET[0].get() + data_channels_AET[1].get().conj() * data_channels_AET[1].get()) / psd_tmp)

            # d_h = 4 * settings["general"]["df"] * np.sum((Af.conj() * data_channels_AET[0].get() + Ef.conj() * data_channels_AET[1].get()) / psd_tmp)

            # Af += data_channels_AET[0].get()
            # Ef += data_channels_AET[1].get()
            # Tf += data_channels_AET[2].get()

            Af -= data_channels_AET[0].get()
            Ef -= data_channels_AET[1].get()
            Tf -= data_channels_AET[2].get()

            # ll = -1/2 * 4 * settings["general"]["df"] * np.sum((Af.conj() * Af + Ef.conj() * Ef) / psd_tmp)

            # for (j, cA, cE, d_d, tref) in check:
            #     d_h_tmp = 4 * settings["general"]["df"] * np.sum((cA.conj() * data_channels_AET[0].get() + cE.conj() * data_channels_AET[1].get()) / psd_tmp)
            #     if np.abs(d_h_tmp) / np.sqrt(d_d.real * h_h.real) > 0.2:
            #         print(i, remove_point[0, -1], j, tref, np.abs(d_h_tmp) / np.sqrt(d_d.real * h_h.real), d_h / h_h ** (1/2))

            # check_A = data_channels_AET[0].get()
            # check_E = data_channels_AET[1].get()
            # d_d = 4 * settings["general"]["df"] * np.sum((check_A.conj() * check_A + check_E.conj() * check_E) / psd_tmp)
            # check.append([i, check_A, check_E, d_d, remove_point[0, -1]])
            
            # # check.append([ll.real, ll.real - last_ll.real, remove_point[:, -1], h_h ** (1/2), np.abs(d_h / h_h ** (1/2))])
            # print(i, ll.real, ll.real - last_ll.real, remove_point[:, -1], h_h ** (1/2), np.abs(d_h / h_h ** (1/2)))
            # last_ll = ll

        A = np.fft.irfft(Af) / dt
        E = np.fft.irfft(Ef) / dt
        T = np.fft.irfft(Tf) / dt

        # import matplotlib.pyplot as plt
        # plt.close()
        # plt.plot(np.fft.irfft(np.fft.rfft(A) / get_sensitivity(fd_tmp, sens_fn="noisepsd_AE", model="sangria", includewd=1) ** (1/2))[4419700: 4420300])
        # plt.plot(np.fft.irfft(data_channels_AET.get()[0] / get_sensitivity(fd_tmp, sens_fn="noisepsd_AE", model="sangria", includewd=1) ** (1/2))[4419700: 4420300])
        # plt.savefig("check1.png")
        # plt.close()
        # plt.plot(np.fft.irfft(np.fft.rfft(E) / get_sensitivity(fd_tmp, sens_fn="noisepsd_AE", model="sangria", includewd=1) ** (1/2))[4419700: 4420300])
        # plt.plot(np.fft.irfft(data_channels_AET.get()[1] / get_sensitivity(fd_tmp, sens_fn="noisepsd_AE", model="sangria", includewd=1) ** (1/2))[4419700: 4420300])
        # plt.savefig("check2.png")
        # breakpoint()

        print("adjusted")

    data = []

    initial_t_vals = []
    end_t_vals = []
    for i, t_i in enumerate(range(start_here, end_here + 2)):
        start_t = t_lims[t_i]
        end_t = t_lims[t_i + 2]
        # start_t = 0.0
        keep_t = (t >= start_t) & (t < end_t)
        t_here = t[keep_t]
        A_here = A[keep_t]
        E_here = E[keep_t]
        T_here = T[keep_t]

        # plt.plot(X_here)
        # plt.savefig("check0.png")
        # plt.close()
        # fucking dt
        Af, Ef, Tf = (
            np.fft.rfft(A_here) * dt,
            np.fft.rfft(E_here) * dt,
            np.fft.rfft(T_here) * dt,
        )
        
        data.append(np.asarray([Af, Ef, Tf]))
        initial_t_vals.append(start_t)
        end_t_vals.append(end_t)
        if i == 0:
            length_check = len(Af)
            fd = np.fft.rfftfreq(len(A_here), dt)
            df = fd[1] - fd[0]
            Tobs = dt * len(t_here)
        else:
            assert length_check == len(Af)  
    
    fd = cp.asarray(fd)
    all_data_cpu = np.asarray(data).transpose(1, 0, 2)
    data_length = len(fd)
    initial_t_vals = np.asarray(initial_t_vals)
    end_t_vals = np.asarray(end_t_vals)
        
    # A_psd = get_sensitivity(fd.get(), sens_fn="noisepsd_AE", model="sangria", includewd=1.0)
    
    # E_psd = get_sensitivity(
    #     fd.get(), sens_fn="noisepsd_AE", model="sangria", includewd=1.0
    # )

    psd_reader = HDFBackend(settings['general']['file_information']['fp_psd_pe'])

    psd_best = psd_reader.get_chain()["psd"][psd_reader.get_log_like() == psd_reader.get_log_like().max()].squeeze()
    galfor_best = psd_reader.get_chain()["galfor"][psd_reader.get_log_like() == psd_reader.get_log_like().max()].squeeze()
    
    A_psd = get_sensitivity(fd.get(), sens_fn="noisepsd_AE", model=psd_best[:2], foreground_params=galfor_best)
    
    E_psd = get_sensitivity(
        fd.get(), sens_fn="noisepsd_AE", model=psd_best[2:], foreground_params=galfor_best
    )

    A_psd[0] = A_psd[1]
    E_psd[0] = E_psd[1]

    psd_cpu = np.asarray([np.tile(A_psd, (all_data_cpu.shape[1], 1)), np.tile(E_psd, (all_data_cpu.shape[1], 1)), np.tile(np.full_like(A_psd, 1e10), (all_data_cpu.shape[1], 1))])

    d_d_vals = np.zeros(psd_cpu.shape[1])  # 4 * df * np.sum(np.asarray([(all_data_cpu[i].conj() * all_data_cpu[i]) / psd_cpu[i] for i in range(all_data_cpu.shape[0])]), axis=(0, 2))

    all_data = cp.asarray(all_data_cpu.flatten())
    psd = cp.asarray(psd_cpu.flatten())

    like_args = (wave_gen, initial_t_vals, end_t_vals, d_d_vals, t_ref_lims_here, transform_fn, (fd, all_data, psd, df), mbh_kwargs)

    # check_points = best_points.copy()
    # ll_check = search_likelihood_wrap(check_points, *like_args)
    # breakpoint()

    fp = settings["general"]["file_information"]["fp_mbh_search_base"] + f"_{time_split}_run_{num_run}.h5"
    # old_fp = f"test_new_mbh_search_run_{run}.h5"
    if os.path.exists(fp):
        print(fp)
        start_params = HDFBackend(fp).get_last_sample().branches["mbh"].coords
    else:
        start_params = priors["mbh"].rvs(size=(nwalkers, ntemps,))

    current_start_points = start_params.copy()
    ndim = mbh_info["pe_info"]["ndim"]
    ll_vals = search_likelihood_wrap(current_start_points.reshape(-1, ndim), *like_args).reshape(ntemps, nwalkers)

    cp.get_default_memory_pool().free_all_blocks()

    ndims = {"mbh": ndim}

    # MCMC moves (move, percentage of draws)
    moves = mbh_info["pe_info"]["inner_moves"]
    
    # prepare sampler
    sampler = EnsembleSampler(
        nwalkers,
        ndims,  # assumes ndim_max
        search_likelihood_wrap,
        priors,
        tempering_kwargs={"ntemps": ntemps, "Tmax": np.inf},
        moves=moves,
        args=like_args,
        kwargs={},
        backend=fp,
        vectorize=True,
        periodic=periodic,  # TODO: add periodic to proposals
        stopping_fn=stop,
        stopping_iterations=1,
        branch_names=["mbh"],
    )

    sampler.proc = rank

    nsteps = 10000

    iteration = 0

    while iteration == 0 or np.any(ll_vals[0].max() - ll_vals[0] > 50.0):
        cp.get_default_memory_pool().free_all_blocks()
        print(f"iteration {iteration} start")
        if iteration > 0:
            mbh_injection_params = out.branches_coords["mbh"][
                0, np.argmax(out.log_like[0]), 0
            ]

            factor = 1e-5
            cov = np.ones(ndim) * 1e-3
            cov[0] = 1e-5
            cov[-1] = 1e-5

            start_like = np.zeros((nwalkers * ntemps))

            iter_check = 0
            max_iter = 1000
            while np.std(start_like) < 5.0:
                logp = np.full_like(start_like, -np.inf)
                tmp = np.zeros((ntemps * nwalkers, ndim))
                fix = np.ones((ntemps * nwalkers), dtype=bool)
                while np.any(fix):
                    tmp[fix] = (
                        mbh_injection_params[None, :]
                        * (
                            1.0
                            + factor * cov * np.random.randn(nwalkers * ntemps, ndim)
                        )
                    )[fix]

                    tmp[:, 5] = tmp[:, 5] % (2 * np.pi)
                    tmp[:, 7] = tmp[:, 7] % (2 * np.pi)
                    tmp[:, 9] = tmp[:, 9] % (1 * np.pi)
                    logp = priors["mbh"].logpdf(tmp)

                    fix = np.isinf(logp)
                    if np.all(fix):
                        breakpoint()

                start_like = search_likelihood_wrap(tmp, *like_args)

                iter_check += 1
                factor *= 1.5

                print(np.std(start_like))

                if iter_check > max_iter:
                    raise ValueError("Unable to find starting parameters.")

            current_start_points = tmp

        out = sampler.run_mcmc(
            {"mbh": current_start_points.reshape(ntemps, nwalkers, 1, ndim)}, nsteps, progress=False, thin_by=50, burn=0
        )

        ll_vals = out.log_like.copy()
        print(f"iteration {iteration} end", (ll_vals[0].max() - ll_vals[0]).max())
        iteration += 1

    output_points = out.branches["mbh"].coords[0, :, 0]

    out_ll = search_likelihood_wrap(output_points.reshape(-1, ndim), *like_args)

    phase_change = np.angle(wave_gen.non_marg_d_h.get())[:output_points.shape[0]]
    output_points[:, 5] = (output_points[:, 5] + 1 / 2 * phase_change) % (2 * np.pi)

    out_ll2 = search_likelihood_wrap(output_points.reshape(-1, ndim), *like_args)
    assert np.all(np.abs(wave_gen.non_marg_d_h.imag.get()) < 1e-5)

    mbh_best = output_points[out_ll.argmax()]

    opt_snr = (wave_gen.h_h ** (1/2)).max()
    det_snr = (wave_gen.d_h / (wave_gen.h_h ** (1/2))).max()
    print("Found:", mbh_best, opt_snr, det_snr) 

    if opt_snr < mbh_info["search_info"]["snr_lim"] or det_snr < mbh_info["search_info"]["snr_lim"]:
        finished_this_split = True
    else:
        finished_this_split = False

    try:
        del sampler
        del all_data
        del psd
        del wave_gen
        cp.get_default_memory_pool().free_all_blocks()
        
    except NameError:
        pass
                
    cp.get_default_memory_pool().free_all_blocks()

    return output_points, mbh_best, finished_this_split

class ParallelMBHSearchControl:
    def __init__(self, settings, comm, gpus, head_rank=0, max_num_per_gpu=2, verbose=False):
        self.comm = comm
        self.rank = comm.Get_rank()
        self.num_procs = comm.Get_size()
        self.head_rank = head_rank
        self.gpus = gpus
        self.settings = settings
        self.verbose = verbose

        self.time_splits = settings["mbh"]["search_info"]["time_splits"]
        self.output_points_file = settings["general"]["file_information"]["fp_mbh_search_base"] +"_output.pickle"

        self.total_procs_max = len(gpus) * max_num_per_gpu
        self.proc_inds = np.delete(np.arange(0, self.total_procs_max + 1), self.head_rank)

        if os.path.exists(self.output_points_file):
            with open(self.output_points_file, "rb") as fp:
                self.output_points_info = pickle.load(fp)

            # assert self.output_points_info["time_splits"] == self.time_splits

        else:
            self.output_points_info = {
                "time_splits": self.time_splits,
                "best_points": [],
                "output_points": [],
                "num_run_per_split": [0 for _ in range(self.time_splits)],
                "still_running": [True for _ in range(self.time_splits)]
            }

        # setup gpus for each process
        self.gpu_procs = np.repeat(np.asarray(gpus), repeats=max_num_per_gpu)
        assert len(self.gpu_procs) == len(self.proc_inds)
        # assert  self.num_procs >= self.total_procs_max + 1

    def send_initial_gpu_information(self):
        for proc_i, gpu_i in zip(self.proc_inds, self.gpu_procs):
            self.comm.send({"gpu": gpu_i}, dest=proc_i)

    def check_for_receiving_info(self, i, current_status, current_time_segment_status):
        check_remove = self.comm.irecv(source=self.proc_inds[i])

        if check_remove.get_status():
            check_dict = check_remove.wait()
            if "receive" in check_dict and check_dict["receive"]:
                new_source_dict = self.comm.recv(source=self.proc_inds[i])

                output_points = new_source_dict['output_points']
                mbh_best = new_source_dict["mbh_best"]
                time_split = new_source_dict["time_split"]
                finished_this_split = new_source_dict["finished_this_split"]

                # above snr 40
                if not finished_this_split:
                    self.output_points_info["output_points"].append(output_points)
                    self.output_points_info["best_points"].append(mbh_best)
                    
                self.output_points_info["num_run_per_split"][time_split] += 1
                self.output_points_info["still_running"][time_split] = (not finished_this_split)

                with open(self.output_points_file, "wb") as fp:
                    pickle.dump(self.output_points_info, fp, protocol=pickle.HIGHEST_PROTOCOL)

                current_status[i] = False
                current_time_segment_status[time_split] = False

        else: 
            check_remove.cancel()

    def launch_search_process(self, i, current_status, current_time_segment_status):
        time_split_ind = 0
        try:
            while current_time_segment_status[time_split_ind] or not self.output_points_info["still_running"][time_split_ind]:
                time_split_ind += 1
                if time_split_ind >= self.time_splits:
                    return
        except IndexError:
            breakpoint()

        if self.verbose:
            print("going in", self.proc_inds[i], time_split_ind, current_time_segment_status)
        
        self.comm.send({"best_points": np.asarray(self.output_points_info["best_points"]), "time_split": time_split_ind, "num_run": self.output_points_info["num_run_per_split"][time_split_ind], "total_splits": self.time_splits}, dest=self.proc_inds[i])
        current_status[i] = True
        current_time_segment_status[time_split_ind] = True

    def end_run(self):
        for proc_i in self.proc_inds:
            self.comm.send({"complete": True}, dest=proc_i)

    def run_single_search(self, gpu, best_points, time_split, total_time_splits, num_run):


        print(f"Starting: split {time_split} for run number {num_run}")
        new_output_points, new_best_point, finished_this_split = run_mbh_search(gpu, self.settings, self.rank, time_split, self.time_splits, best_points, num_run)
        print(f"End: split {time_split} for run number {num_run}")
        cp.get_default_memory_pool().free_all_blocks()
        return (new_output_points, new_best_point, finished_this_split)

    def search_process_control(self):
        gpu_check = self.comm.recv(source=self.head_rank)
        cp.cuda.runtime.setDevice(gpu_check['gpu'])
        run = True
        while run:
            print('waiting', self.rank)
            incoming_dict = self.comm.recv(source=self.head_rank)
            print('received', self.rank)
            if "complete" in incoming_dict and incoming_dict["complete"]:
                run = False
                continue

            best_points = incoming_dict["best_points"]
            time_split = incoming_dict["time_split"]
            total_time_splits = incoming_dict['total_splits']
            num_run = incoming_dict['num_run']

            new_output_points, new_best_point, finished_this_split = self.run_single_search(gpu_check['gpu'], best_points, time_split, total_time_splits, num_run)

            print(f"send before receive: split {time_split} for run number {num_run}")
            
            self.comm.send({"receive": True}, dest=self.head_rank)
            print(f"send after receive: split {time_split} for run number {num_run}")
            
            self.comm.send({"output_points": new_output_points, "mbh_best": new_best_point, "time_split": time_split, "finished_this_split": finished_this_split}, dest=self.head_rank)
            print(f"after update: split {time_split} for run number {num_run}")
        
    def run_parallel_mbh_search(self, testing_time_split=None):

        # testing
        if testing_time_split is not None:
            assert isinstance(testing_time_split, int) and testing_time_split < self.time_splits
            cp.cuda.runtime.setDevice(self.gpus[0])
            output_points, mbh_best = run_mbh_search(self.gpus[0], self.settings, self.head_rank, testing_time_split, self.time_splits, np.asarray(self.output_points_info["best_points"]), self.output_points_info["num_run_per_split"][testing_time_split])
            breakpoint()

        if self.rank == self.head_rank:
            self.send_initial_gpu_information()

            #controller
            current_status = [False for _ in range(self.total_procs_max)]
            current_time_segment_status = [False for _ in range(self.time_splits)]
            run = True
            while run:
                time.sleep(1)
                for i in range(len(current_status)):
                    if current_status[i]:
                        self.check_for_receiving_info(i, current_status, current_time_segment_status)
                        
                for i in range(len(current_status)):
                    if not current_status[i]:
                        self.launch_search_process(i, current_status, current_time_segment_status)

                if not np.any(self.output_points_info["still_running"]):
                    run = False

            self.end_run()

            self.prune_via_matching()

        # let others just pass by
        elif self.rank in self.proc_inds:
            self.search_process_control()
            
        print(self.comm.Get_rank(), "done")

    def prune_via_matching(self):
        if os.path.exists(self.output_points_file):
            with open(self.output_points_file, "rb") as fp:
                self.output_points_info = pickle.load(fp)
        else:
            raise ValueError("Trying to prune with no input file.")

        best_points = np.asarray(self.output_points_info["best_points"])

        best_points_in = self.settings["mbh"]["transform"].both_transforms(best_points, return_transpose=True)

        init_kwargs = self.settings["mbh"]["initialize_kwargs"].copy()
        init_kwargs["use_gpu"] = False
        wave_gen = BBHWaveformFD(**init_kwargs)
        
        data_channels_AET = wave_gen(
            *best_points_in,
            freqs=self.settings["general"]["fd"],
            direct=False,
            fill=True,
            squeeze=True,
            **self.settings["mbh"]["waveform_kwargs"]
        )
        
        data_channels_AET[:, 2] = 0.0

        psd_tmp = get_sensitivity(self.settings["general"]["fd"], sens_fn="noisepsd_AE", model="sangria", includewd=1.0)
        psd_tmp[0] = psd_tmp[1]
        psd = np.array([psd_tmp, psd_tmp, np.full_like(psd_tmp, 1e10)])

        A_noise = generate_noise_fd(self.settings["general"]["fd"], self.settings["general"]["df"], func=get_sensitivity, sens_fn="noisepsd_AE", model="sangria", includewd=1.0)
        E_noise = generate_noise_fd(self.settings["general"]["fd"], self.settings["general"]["df"], func=get_sensitivity, sens_fn="noisepsd_AE", model="sangria", includewd=1.0)
        A_noise[0] = A_noise[1]
        E_noise[0] = A_noise[1]
        noise = np.array([A_noise, E_noise, E_noise])
        
        df = self.settings["general"]["df"]
        prune_remove = []
        out_norm_with_noise = []
        # keep in order. The earlier one will be the louder one found
        for i in range(data_channels_AET.shape[0]):
            for j in range(i, data_channels_AET.shape[0]):
                if (j in prune_remove or i in prune_remove) and i != j:
                    continue

                a_a = 4 * df * np.sum((noise + data_channels_AET[i]).conj() * (noise + data_channels_AET[i]) / psd[None, :], axis=(1, 2))
                b_b = 4 * df * np.sum(data_channels_AET[j].conj() * data_channels_AET[j] / psd[None, :], axis=(1, 2))
                a_b = 4 * df * np.sum((noise + data_channels_AET[i]).conj() * data_channels_AET[j] / psd[None, :], axis=(1, 2))

                normalized_noise_weighted_corr_with_noise = -1/2 * (a_a + b_b - 2 * a_b) #  np.abs(a_b).real / np.sqrt(a_a * b_b).real
                if i == j:
                    out_norm_with_noise.append([normalized_noise_weighted_corr_with_noise.copy(), np.sqrt(b_b), a_b / np.sqrt(b_b)])
                
                c_c = 4 * df * np.sum(data_channels_AET[i].conj() * data_channels_AET[i] / psd[None, :], axis=(1, 2))
                d_d = 4 * df * np.sum(data_channels_AET[j].conj() * data_channels_AET[j] / psd[None, :], axis=(1, 2))
                c_d = 4 * df * np.sum(data_channels_AET[i].conj() * data_channels_AET[j] / psd[None, :], axis=(1, 2))

                normalized_noise_weighted_corr = np.abs(c_d).real / np.sqrt(c_c * d_d).real
            
                if normalized_noise_weighted_corr > 0.25 and i != j:
                    # print(i, j, normalized_noise_weighted_corr, best_points[i, -1], best_points[j, -1])
                
                    prune_remove.append(j)

        print(prune_remove, np.unique(prune_remove).astype(int))
        prune_keep = np.delete(np.arange(data_channels_AET.shape[0]), np.unique(prune_remove).astype(int))

        output_points_pruned = np.asarray([self.output_points_info["output_points"][i] for i in prune_keep])
        best_points_pruned = np.asarray([self.output_points_info["best_points"][i] for i in prune_keep])

        self.output_points_info["output_points_pruned"] = output_points_pruned
        self.output_points_info["best_points_pruned"] = best_points_pruned

        # backup file
        shutil.copy(self.output_points_file, self.output_points_file[:-7] + "_backup.pickle")
        with open(self.output_points_file, "wb") as fp:
            pickle.dump(self.output_points_info, fp, protocol=pickle.HIGHEST_PROTOCOL)




