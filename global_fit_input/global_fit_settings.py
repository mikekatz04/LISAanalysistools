import h5py
import numpy as np
import shutil

from gbgpu.utils.constants import *
from gbgpu.utils.utility import get_fdot

from bbhx.utils.transform import *

from lisatools.globalfit.generatefuncs import *
from lisatools.utils.utility import AET
from lisatools.sampling.prior import SNRPrior, AmplitudeFromSNR, AmplitudeFrequencySNRPrior, GBPriorWrap

from eryn.prior import uniform_dist
from eryn.utils import TransformContainer
from eryn.prior import ProbDistContainer

from eryn.moves import StretchMove
from lisatools.sampling.moves.skymodehop import SkyMove



# basic transform functions for pickling
def f_ms_to_s(x):
    return x * 1e-3


def mbh_dist_trans(x):
    return x * PC_SI * 1e9  # Gpc


def get_global_fit_settings(copy_settings_file=False):

    ###############################
    ###############################
    ###  Global Fit File Setup  ###
    ###############################
    ###############################

    file_information = {}
    file_store_dir = "global_fit_output/"
    file_information["file_store_dir"] = file_store_dir
    base_file_name = "sixth_run_through"
    file_information["base_file_name"] = base_file_name
    file_information["plot_base"] = file_store_dir + base_file_name + '/output_plots.png'

    file_information["fp_psd_search_initial"] = file_store_dir + base_file_name + "_initial_search_psd.h5"
    file_information["fp_psd_search"] = file_store_dir + base_file_name + "_search_psd.h5"
    file_information["fp_mbh_search_base"] = file_store_dir + base_file_name + "_search_mbh"


    file_information["fp_gb_pe"] = file_store_dir + base_file_name + "_parameter_estimation_gb.h5"
    file_information["fp_psd_pe"] = file_store_dir + base_file_name + "_parameter_estimation_psd.h5"
    file_information["fp_mbh_pe"] = file_store_dir + base_file_name + "_parameter_estimation_mbh.h5"

    file_information["fp_gb_gmm_info"] = file_store_dir + base_file_name + "_gmm_info.pickle"

    file_information["gb_main_chain_file"] = file_store_dir + base_file_name + "_gb_main_chain_file.h5"
    file_information["gb_all_chain_file"] = file_store_dir + base_file_name + "_gb_all_chain_file.h5"

    file_information["mbh_main_chain_file"] = file_store_dir + base_file_name + "_mbh_main_chain_file.h5"

    file_information["status_file"] = file_store_dir + base_file_name + "_status_file.txt"

    if copy_settings_file:
        shutil.copy(__file__, file_store_dir + base_file_name + "_" + __file__.split("/")[-1])
    
    ###############################
    ###############################
    ###  Global Fit data Setup  ###
    ###############################
    ###############################

    ldc_source_file = "LDC2_sangria_training_v2.h5"
    with h5py.File(ldc_source_file, "r") as f:
        tXYZ = f["obs"]["tdi"][:]

        # remove mbhb and igb
        for source in []:  # "igb"]:  # "vgb" "mbhb",
            change_arr = f["sky"][source]["tdi"][:]
            for change in ["X", "Y", "Z"]:
                tXYZ[change] -= change_arr[change]

        # tXYZ = f["sky"]["dgb"]["tdi"][:]
        # tXYZ["X"] += f["sky"]["dgb"]["tdi"][:]["X"]
        # tXYZ["Y"] += f["sky"]["dgb"]["tdi"][:]["Y"]
        # tXYZ["Y"] += f["sky"]["dgb"]["tdi"][:]["Z"]

    t, X, Y, Z = (
        tXYZ["t"].squeeze(),
        tXYZ["X"].squeeze(),
        tXYZ["Y"].squeeze(),
        tXYZ["Z"].squeeze(),
    )
    dt = t[1] - t[0]

    Nobs = len(t)
    if Nobs > int(YEAR / dt):
        Nobs = int(YEAR / dt)
        t = t[:Nobs]
        X = X[:Nobs]
        Y = Y[:Nobs]
        Z = Z[:Nobs]

    Tobs = Nobs * dt
    df = 1 / Tobs

    # f***ing dt
    Xf, Yf, Zf = (np.fft.rfft(X) * dt, np.fft.rfft(Y) * dt, np.fft.rfft(Z) * dt)
    Af, Ef, Tf = AET(Xf, Yf, Zf)

    start_freq_ind = 0
    end_freq_ind = int(0.034 / df)  # len(A_inj) - 1
    
    A_inj, E_inj = (
        Af[start_freq_ind:end_freq_ind],
        Ef[start_freq_ind:end_freq_ind],
    )

    data_length = len(A_inj)
    fd = np.arange(data_length) * df
    
    generate_current_state = GenerateCurrentState(A_inj, E_inj)

    gpus = [5, 6, 7, 7]

    all_general_info = dict(
        file_information=file_information,
        fd=fd,
        A_inj=A_inj,
        E_inj=E_inj,
        t=t, 
        X=X,
        Y=Y,
        Z=Z,
        data_length=data_length,
        start_freq_ind=start_freq_ind,
        end_freq_ind=end_freq_ind,
        df=df,
        Tobs=Tobs,
        dt=dt,
        source_file=ldc_source_file,
        generate_current_state=generate_current_state,
        random_seed=1024,
        begin_new_likelihood=False,
        plot_iter=4,
        backup_iter=10,
        gpus=gpus
    )


    ###############################
    ###############################
    ######    Rank/GPU setup  #####
    ###############################
    ###############################

    head_rank = 0

    gb_pe_rank = 1
    gb_pe_gpu = gpus[0]

    # should be one more rank than GPUs for refit
    gb_search_rank = [2, 3]
    gb_search_gpu = gpus[1]

    psd_rank = 4
    psd_gpu = gpus[3]

    mbh_rank = 5
    mbh_gpu = gpus[2]

    # run results rank will be next available rank if used
    # gmm_ranks will be all other ranks

    rank_info = dict(
        head_rank=head_rank,
        gb_pe_rank=gb_pe_rank,
        gb_search_rank=gb_search_rank,
        psd_rank=psd_rank,
        mbh_rank=mbh_rank,
    )

    gpu_assignments = dict(
        gb_pe_gpu=gb_pe_gpu,
        gb_search_gpu=gb_search_gpu,
        psd_gpu=psd_gpu,
        mbh_gpu=mbh_gpu,
    )

    ##################################
    ##################################
    ###  Galactic Binary Settings  ###
    ##################################
    ##################################

    # limits on parameters
    delta_safe = 1e-5
    A_lims = [7e-24, 1e-21]
    f0_lims = [0.05e-3, 2.5e-2]
    m_chirp_lims = [0.001, 1.0]
    # now with negative fdots
    fdot_max_val = get_fdot(f0_lims[-1], Mc=m_chirp_lims[-1])
    fdot_lims = [-fdot_max_val, fdot_max_val]
    phi0_lims = [0.0, 2 * np.pi]
    iota_lims = [0.0 + delta_safe, np.pi - delta_safe]
    psi_lims = [0.0, np.pi]
    lam_lims = [0.0, 2 * np.pi]
    beta_sky_lims = [-np.pi / 2.0 + delta_safe, np.pi / 2.0 - delta_safe]

    # band separation setup
    width_low = 256 + 10
    width_mid = 512 + 10
    width_high = 2048 + 10

    first_barrier = (0.001 / df).astype(int) * df
    second_barrier = (0.01 / df).astype(int) * df

    low_fs_propose = np.arange(f0_lims[0], first_barrier - width_mid * df, width_low * df)
    mid_fs_propose = np.arange(first_barrier, second_barrier - width_high * df, width_mid * df)
    high_fs_propose = np.append(
        np.arange(second_barrier, f0_lims[-1], width_high * df)[:-1], np.array([f0_lims[-1]])
    )

    band_edges = np.concatenate([low_fs_propose, mid_fs_propose, high_fs_propose])
    num_sub_bands = len(band_edges)

    # waveform settings
    oversample = 4
    # TODO: check beta versus theta
    gb_waveform_kwargs = dict(
        dt=dt, T=Tobs, use_c_implementation=True, oversample=oversample
    )

    pe_gb_waveform_kwargs = dict(
        dt=dt, T=Tobs, use_c_implementation=True, oversample=oversample
    )

    gb_initialize_kwargs = dict(use_gpu=True)

    L = 2.5e9
    amp_transform = AmplitudeFromSNR(L, Tobs, fd, model="sangria", sens_fn="lisasens")

    gb_transform_fn_in = {
        1: f_ms_to_s,
        5: np.arccos,
        8: np.arcsin,
        # (1, 2, 3): (lambda f0, fdot, fddot: (f0, fdot, 11 / 3.0 * fdot ** 2 / f0)),
        # (0, 1): amp_transform,  # do not need this when running with amplitude parameter
    }

    gb_fill_dict = {"fill_inds": np.array([3]), "ndim_full": 9, "fill_values": np.array([0.0])}

    gb_transform_fn = TransformContainer(
        parameter_transforms=gb_transform_fn_in, fill_dict=gb_fill_dict
    )

    gb_periodic = {3: 2 * np.pi, 5: np.pi, 6: 2 * np.pi}

    # prior setup
    rho_star = 10.0
    # snr_prior = SNRPrior(rho_star)

    frequency_prior = uniform_dist(*(np.asarray(f0_lims) * 1e3))

    priors_gb = {
        (0, 1): AmplitudeFrequencySNRPrior(rho_star, frequency_prior, L, Tobs, fd=fd),  # use sangria as a default
        2: uniform_dist(*fdot_lims),
        3: uniform_dist(*phi0_lims),
        4: uniform_dist(*np.cos(iota_lims)),
        5: uniform_dist(*psi_lims),
        6: uniform_dist(*lam_lims),
        7: uniform_dist(*np.sin(beta_sky_lims)),
    }

    priors_gb_fin = GBPriorWrap(8, ProbDistContainer(priors_gb))

    snrs_ladder = np.array([1., 1.5, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0, 15.0, 20.0, 35.0, 50.0, 75.0, 125.0, 250.0, 5e2])
    ntemps_pe = 24  # len(snrs_ladder)
    # betas =  1 / snrs_ladder ** 2  # make_ladder(ndim * 10, Tmax=5e6, ntemps=ntemps_pe)
    betas = 1 / 1.2 ** np.arange(ntemps_pe)
    betas[-1] = 0.0001

    stopping_kwargs = dict(
        n_iters=1000,
        diff=1.0,
        verbose=True
    )

    stop_search_kwargs = dict(
        convergence_iter=5,  # really * thin_by
        verbose=True
    )

    # mcmc info for main run
    gb_main_run_mcmc_info = dict(
        branch_names=["gb"],
        nleaves_max=15000,
        ndim=8,
        ntemps=len(betas),
        betas=betas,
        nwalkers=36,
        start_resample_iter=-1,  # -1 so that it starts right at the start of PE
        iter_count_per_resample=10,
        pe_waveform_kwargs=pe_gb_waveform_kwargs,
        group_proposal_kwargs=dict(
            n_iter_update=1,
            live_dangerously=True,
            a=1.75,
            num_repeat_proposals=30
        ),
        other_tempering_kwargs=dict(
            adaptation_time=2,
            permute=True
        ),
        use_prior_removal=False,
        rj_refit_fraction=0.2,
        rj_search_fraction=0.2,
        rj_prior_fraction=0.6,
        nsteps=10000,
        update_iterations=1,
        thin_by=3,
        progress=True,
        rho_star=rho_star,
        stop_kwargs=stopping_kwargs,
        stop_search_kwargs=dict(convergence_iter=5, verbose=True),  # really 5 * thin_by
        stopping_iterations=1,
        in_model_phase_maximize=False,
        rj_phase_maximize=False,
    )

    # mcmc info for search runs
    gb_search_run_mcmc_info = dict(
        ndim=8,
        ntemps=10,
        nwalkers=100,
        pe_waveform_kwargs=pe_gb_waveform_kwargs,
        m_chirp_lims=[0.001, 1.2],
        snr_lim=5.0,
        # stop_kwargs=dict(newly_added_limit=1, verbose=True),
        stopping_iterations=1,
    )

    # template generator
    get_gb_templates = GetGBTemplates(
        gb_initialize_kwargs,
        gb_waveform_kwargs
    )

    all_gb_info = dict(
        band_edges=band_edges,
        periodic=gb_periodic,
        priors=priors_gb_fin,
        transform=gb_transform_fn,
        waveform_kwargs=gb_waveform_kwargs,
        initialize_kwargs=gb_initialize_kwargs,
        pe_info=gb_main_run_mcmc_info,
        search_info=gb_search_run_mcmc_info,
        get_templates=get_gb_templates,
    )



    ##################################
    ##################################
    ###  PSD Settings  ###############
    ##################################
    ##################################


    priors_psd = {
        0: uniform_dist(6.0e-12, 20.0e-12),  # Soms_d
        1: uniform_dist(2.0e-15, 20.0e-15),  # Sa_a
        2: uniform_dist(6.0e-12, 20.0e-12),  # Soms_d
        3: uniform_dist(2.0e-15, 20.0e-15),  # Sa_a
    }

    psd_kwargs = dict(sens_fn="noisepsd_AE", use_gpu=True)
    psd_initialize_kwargs = {}

    get_psd = GetPSDModel(
        psd_kwargs
    )

    ### Galactic Foreground Settings #
 
    priors_galfor = {
        0: uniform_dist(1e-45, 2e-43),  # amp
        1: uniform_dist(1.0, 3.0),  # alpha
        2: uniform_dist(5e1, 1e7),  # Slope1
        3: uniform_dist(1e-4, 5e-2),  # knee
        4: uniform_dist(5e1, 8e3),  # Slope2
    }

    search_stopping_kwargs = dict(
        n_iters=5,
        diff=0.01,
        verbose=False
    )

    # mcmc info for main run
    psd_main_run_mcmc_info = dict(
        branch_names=["psd", "galfor"],
        ndims={"psd": 4, "galfor": 5},
        nleaves_max={"psd": 1, "galfor": 1},
        ntemps=10,
        nwalkers=50,
        progress=False,
        thin_by=100,
        update_iterations=20,
        stop_kwargs=search_stopping_kwargs,
        stopping_iterations=1
    )

    all_psd_info = dict(
        periodic=None,
        priors={"psd": ProbDistContainer(priors_psd), "galfor": ProbDistContainer(priors_galfor)},
        psd_kwargs=psd_kwargs,
        initalize_kwargs=psd_initialize_kwargs,
        get_psd=get_psd,
        pe_info=psd_main_run_mcmc_info,
        stopping_iterations=1,
    )


    ##################################
    ##################################
    ### MBHB Settings ################
    ##################################
    ##################################


    # for transforms
    fill_dict_mbh = {
        "ndim_full": 12,
        "fill_values": np.array([0.0]),
        "fill_inds": np.array([6]),
    }

    # priors
    priors_mbh = {
        0: uniform_dist(np.log(1e4), np.log(1e8)),
        1: uniform_dist(0.01, 0.999999999),
        2: uniform_dist(-0.99999999, +0.99999999),
        3: uniform_dist(-0.99999999, +0.99999999),
        4: uniform_dist(0.01, 1000.0),
        5: uniform_dist(0.0, 2 * np.pi),
        6: uniform_dist(-1.0 + 1e-6, 1.0 - 1e-6),
        7: uniform_dist(0.0, 2 * np.pi),
        8: uniform_dist(-1.0 + 1e-6, 1.0 - 1e-6),
        9: uniform_dist(0.0, np.pi),
        10: uniform_dist(0.0, Tobs + 3600.0),
    }

    # transforms from pe to waveform generation
    parameter_transforms_mbh = {
        0: np.exp,
        4: mbh_dist_trans,
        7: np.arccos,
        9: np.arcsin,
        (0, 1): mT_q,
        (11, 8, 9, 10): LISA_to_SSB,
    }

    transform_fn_mbh = TransformContainer(
        parameter_transforms=parameter_transforms_mbh,
        fill_dict=fill_dict_mbh,
    )

    # sampler treats periodic variables by wrapping them properly
    periodic_mbh = {
        "mbh": {5: 2 * np.pi, 7: np.pi, 8: np.pi}
    }

    # waveform kwargs
    initialize_kwargs_mbh = dict(
        amp_phase_kwargs=dict(run_phenomd=True),
        response_kwargs=dict(TDItag="AET"),
        use_gpu=True
    )

    # for MBH waveform class initialization
    waveform_kwargs_mbh = dict(
        modes=[(2,2)],
        length=1024,
    )

    get_mbh = GetMBHTemplates(
        initialize_kwargs_mbh,
        waveform_kwargs_mbh
    )

    inner_moves = [
        (SkyMove(which="both"), 0.02),
        (SkyMove(which="long"), 0.05),
        (SkyMove(which="lat"), 0.05),
        (StretchMove(), 0.88)
    ]

    mix_stopping_kwargs = dict(
        n_iters=5,
        diff=0.01,
        verbose=False
    )

    # mcmc info for main run
    mbh_main_run_mcmc_info = dict(
        branch_names=["mbh"],
        nleaves_max=15,
        ndim=11,
        ntemps=10,
        nwalkers=50,
        num_prop_repeats=200,
        inner_moves=inner_moves,
        progress=False,
        thin_by=1,
        stop_kwargs=mix_stopping_kwargs,
        stopping_iterations=1
    )

    mbh_search_kwargs = {
        "modes": [(2, 2)],
        "length": 1024,
        "shift_t_limits": True,
        "phase_marginalize": True
    }

    search_stopping_kwargs = dict(
        n_iters=50,
        diff=0.01,
        verbose=False
    )

    mbh_search_run_info = dict(
        ntemps=10,
        nwalkers=100,
        mbh_kwargs=mbh_search_kwargs,
        time_splits=8,
        max_num_per_gpu=2, 
        verbose=False,
        snr_lim=20.0, 
        stop_kwargs=search_stopping_kwargs,
        stopping_iterations=4
    )

    all_mbh_info = dict(
        periodic=periodic_mbh,
        priors={"mbh": ProbDistContainer(priors_mbh)},
        transform=transform_fn_mbh,
        waveform_kwargs=waveform_kwargs_mbh,
        initialize_kwargs=initialize_kwargs_mbh,
        pe_info=mbh_main_run_mcmc_info,
        search_info=mbh_search_run_info,
        get_templates=get_mbh,
        stop_kwargs=search_stopping_kwargs,
    )

    ##############
    ## READ OUT ##
    ##############

    return {
        "gb": all_gb_info,
        "mbh": all_mbh_info,
        "psd": all_psd_info,
        "general": all_general_info,
        "rank_info": rank_info,
        "gpu_assignments": gpu_assignments,
    }


if __name__ == "__main__":
    settings = get_global_fit_settings()
    breakpoint()