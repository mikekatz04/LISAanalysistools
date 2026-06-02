import os
import sys
import numpy as np
from matplotlib import pyplot as plt

from copy import deepcopy

# allow importing gf_dev helpers without installing them as a package
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "LISAanalysistools", "gf_dev"))
# from galfor_model_py import (
#     _fiducial_galfor_params,
#     S_gal,
#     galaxy_common_tdi_factor,
#     hot_path_foreground,
# )

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from eryn.backends.hdfbackend import HDFBackend
from eryn.utils.transform import TransformContainer

from lisatools.detector import L1Orbits
from lisatools.globalfit.preprocessing import L1ProcessingStep
from lisatools.utils.constants import YRSID_SI
from lisatools.utils.utility import tukey
from lisatools.analysiscontainer import AnalysisContainer, AnalysisContainerArray
from lisatools.sensitivity import XYZSensitivityBackend
from lisatools.domains import TDSettings, FDSettings, TDSignal, FDSignal
from lisatools.sources.utils import evolve_galactic_binary
from lisatools.globalfit.postprocessing import to_characteristic_strain

import jax
jax.config.update("jax_enable_x64", True)
import jaxgb

import cupy as cp

from gbgpu.gbgpu import GBGPU
from gbgpu.utils.utility import get_N

import logging

# import cantuccio

# style = cantuccio.visuals.get_paper_style()
# plt.style.use(style)

logger = logging.getLogger(__name__)
# setup logging to print to console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger.setLevel(logging.INFO)

START_FREQ, END_FREQ = 1e-4, 2.9e-2 #1.081e-2, 1.1e-2
dt = 5.0
Tobs = 0.9 * YRSID_SI
WINDOW_TAPER_DURATION = 1 / START_FREQ 
BACKEND = "cuda_12x"
datapath = "/data/asantini/globalfit/MOJITO_DATA/mojito_light_2p5s/"
iteratively_resolved_population_path = os.path.join("/data/asantini/globalfit/erebor", "galaxy_catalogs", "iteratively_resolved_gbs_075yrs_snr7.npy")

source_types = [
    'gb', 
    # 'vgb',
    'noise'
    ]

source_ids = dict()

downsample_kwargs = {
        "target_fs": 1 / dt,  # Hz — target sampling rate (None = no downsampling).
        "window": (
            "kaiser",
            31.0,
        ),  # Kaiser window beta parameter (higher = more aggressive anti-aliasing)
    }

highpass_kwargs = {
    "cutoff": 1e-5,  # Hz — highpass cutoff frequency
    "order": 2,  # Butterworth filter order
    "zero_phase": True,
}

lowpass_kwargs = {
    "cutoff": 1e-1,  # Hz — lowpass cutoff frequency
    "order": 2,  # Butterworth filter order
    "zero_phase": True,
}

trim_kwargs = {
    "duration": 0.02,  # seconds — duration to trim from each end
    "is_percent": True,  # If True, 'duration' is interpreted as a percentage of the total signal length
    "trimming_type": "from_each_end",  # "from_each_end" or "from_start"
}

Soms_d    = 15e-12
Sa_a      = 3e-15
Amp       = 10**(-45.184)
alpha     = 2.6
f_1       = 10**(-2.72)
f_2       = 10**(-3.876)
kn        = 10**(-2.176)

def get_noise_transform_fn():

    def ten_to_the_x(x):
        return 10.0 ** x

    psd_input_basis = [
        r"$S_{\rm oms}$",
        r"$S_{\rm tm}$",
    ]

    galfor_input_basis = [
        r'$\log_{10} A_{\rm gal}$',
        r'$\alpha_{\rm gal}$',
        r'$\log_{10} f_1$',
        r'$\log_{10} f_{\rm knee}$',
        r'$\log_{10} f_2$',
    ]


    psd_transform = TransformContainer(
        input_basis=psd_input_basis,
        output_basis=psd_input_basis,
        parameter_transforms={
            r"$S_{\rm oms}$": ten_to_the_x,
            r"$S_{\rm tm}$": ten_to_the_x,
        },
    )

    galfor_transform = TransformContainer(
        input_basis=galfor_input_basis,
        output_basis=galfor_input_basis,
        parameter_transforms={
            r'$\log_{10} A_{\rm gal}$': ten_to_the_x,
            r'$\log_{10} f_1$': ten_to_the_x,
            r'$\log_{10} f_{\rm knee}$': ten_to_the_x,
            r'$\log_{10} f_2$': ten_to_the_x,
        },
    )

    return psd_transform, galfor_transform

def get_band_structure(start_freq: float, end_freq: float, Tobs: float, df: float):
        
        oversample = 4
        extra_buffer = 5

        # TODO: assign to binned f or leave general? probably better to be general
        band_edges_in_reverse_order = [end_freq]
        current_N = get_N(1e-30, end_freq, Tobs, oversample=oversample).item()
        min_N = get_N(1e-30, start_freq, Tobs, oversample=oversample).item()
        band_N_vals_reverse_order = [current_N]

        current_freq = end_freq - df / 2
        last_freq = end_freq
        while current_freq > start_freq + min_N * df:
            current_freq = last_freq - (current_N * 2 + extra_buffer) * df
            band_edges_in_reverse_order.append(current_freq)
            current_N = get_N(1e-30, current_freq, Tobs, oversample=oversample).item()
            band_N_vals_reverse_order.append(current_N)
            last_freq = current_freq
        band_edges_in_reverse_order.append(
            last_freq - (current_N * 2 + extra_buffer) * df
        )

        band_edges = np.asarray(band_edges_in_reverse_order)[::-1]
        band_N_vals = np.asarray(band_N_vals_reverse_order)[::-1]
        
        # trim edges to avoid out of bound indexing
        band_edges = band_edges[2:-1]
        band_N_vals = band_N_vals[2:-1]

        num_sub_bands = len(band_edges) - 1
        
        logger.info(f"The number of subbands is {num_sub_bands}")
        logger.info(f"Min freq of subbands is {band_edges.min()}")
        logger.info(f"Max freq of subbands is {band_edges.max()}")

        return band_edges, band_N_vals

def get_periodogram(x_fd, dt, window=None):
      
    if window is not None:
        num_times_term = np.sum(window**2)
    else:
        num_times_term = len(x_fd) * 2 - 1

    norm = 2 / (dt * num_times_term)
    periodogram = np.abs(x_fd) ** 2 * norm

    return periodogram

def get_catalog_entries(loader: L1ProcessingStep, subset_inds: np.ndarray, data_t0: float, original_t0: float):
    """
    Get the catalog entries for the specified subset of sources.
    """

    catalogue_entry = loader.catalogue['GB'][0]

    amp = np.array(catalogue_entry["Amplitude"])[subset_inds]
    f_ref = np.array(catalogue_entry["GW22FrequencySourceFrame"])[subset_inds]
    fdot = np.array(catalogue_entry["GW22FrequencyDerivativeSourceFrame"])[subset_inds]
    phi_ref = np.array(catalogue_entry["TrueAnomaly"])[subset_inds]
    t_ref = np.unique(np.array(catalogue_entry["TimeReferenceSSBFrame"])).item()
    iota = np.array(catalogue_entry["InclinationAngle"])[subset_inds]
    ra = np.array(catalogue_entry["RightAscension"])[subset_inds]
    dec = np.array(catalogue_entry["Declination"])[subset_inds]
    psi = np.array(catalogue_entry["PolarisationAngle"])[subset_inds]

    t_init = data_t0 # recipe_steps has t_init = t_ref + 850.5 + trim_duration, with trim_duration = data_t0 - t_ref. why?

    f_ref_evolved, phi_ref_evolved, _ = evolve_galactic_binary(
        t_ref, 
        t_init,
        f_ref,
        phi_ref,
        fdot,
        phase_sign=-1
        )
    
    catalog_entries = {
        "amp": amp,
        "f0": f_ref_evolved,
        "fdot": fdot,
        "phi0": phi_ref_evolved,
        "iota": iota,
        "psi": psi,
        "ra": ra,
        "dec": dec,
    }
    
    return catalog_entries

def get_gbgpu_args(catalog_entries: dict, sign_phi0: int = 1) -> tuple:

    fddot = np.zeros_like(catalog_entries["fdot"])
    
    args = (
        catalog_entries["amp"],
        catalog_entries["f0"],
        catalog_entries["fdot"],
        fddot,
        sign_phi0 * catalog_entries["phi0"],
        catalog_entries["iota"],
        catalog_entries["psi"],
        catalog_entries["ra"],
        catalog_entries["dec"],
    )

    return args

def get_jaxgb_args(catalog_entries: dict, sign_phi0: int = 1):

    
    args = (
        catalog_entries["amp"],
        catalog_entries["f0"],
        catalog_entries["fdot"],
        catalog_entries["iota"],
        catalog_entries["psi"],
        catalog_entries["ra"],
        catalog_entries["dec"],
        catalog_entries["phi0"],
    )

    return args

def subtract_gb_signal_from_residual(acs: AnalysisContainerArray, 
                                     gb_wave_gen: GBGPU, 
                                     waveform_kwargs: dict, 
                                     catalog_entries: dict, 
                                     band_edges: np.ndarray, 
                                     band_N_vals: np.ndarray, 
                                     sign_phi0: int = 1
                                     ) -> np.ndarray:

    gbgpu_args = get_gbgpu_args(catalog_entries, sign_phi0=sign_phi0)

    coords_in = np.asarray(gbgpu_args).T #shape (num_sources,)

    band_inds = np.searchsorted(band_edges, coords_in[:, 1], side="right") - 1

    walker_vals = np.tile(np.arange(1), (gbgpu_args[0].shape[0], 1)).T[0] #shape (num_sources,)

    data_index = cp.asarray(walker_vals).astype(
            cp.int32
        )

    factors = -cp.ones_like(data_index, dtype=cp.float64)

    N_vals = band_N_vals[band_inds]

    logger.info("Removing GBs from residuals")
    #template_in = deepcopy(acs.linear_data_arr)

    template_out = deepcopy(acs.linear_data_arr)
    gb_wave_gen.generate_global_template(
        coords_in,
        data_index,
        template_out,
        data_length=acs.data_length,
        factors=factors,
        data_splits=acs.gpu_map,
        N=N_vals,
        **waveform_kwargs,
    )

    template_out = template_out[0].reshape(3, -1)

    return template_out.get() if hasattr(template_out, "get") else template_out

def plot_comparison(freqs: np.ndarray, residuals: np.ndarray, color: str, label: str, dt: float, fig = None, **kwargs) -> tuple[plt.Figure, np.ndarray, np.ndarray]:
    """
    Add plots to figure.

    """
    if fig is None:
        fig = plt.figure()
    
    periodogram = get_periodogram(residuals, dt)
    decimated_freqs, decimated_power, _ = average_periodogram_static(freqs, periodogram, f_min=START_FREQ, f_max=END_FREQ, f_segments=1e-5, n_segments=None)

    plt.loglog(freqs, periodogram, **kwargs, alpha=0.1, color=color)
    plt.loglog(decimated_freqs, decimated_power, **kwargs, alpha=1.0, label=label, color=color)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD [1/Hz]")
    plt.legend()

    return fig, decimated_freqs, decimated_power

def average_periodogram_static(freqs, power, f_min=None, f_max=None, f_segments=1e-5, n_segments=None, **kwargs):
        """
        Average the periodogram matrix over segments. Snippet credits: Nikolaos Karnesis.
        
        Parameters
        ----------
        freqs : array
            The frequencies of the periodogram matrix.
         power : array
            The periodogram matrix to average.
        f_segments : float or array_like
            The segment frequency or an array of bin edges.
            
        Returns
        -------
        freqs_h : array
            The frequencies where the averaged periodogram is computed.
        power_avg : array
            The averaged periodogram matrix.
        segment_sizes : array
            The sizes of the frequency segments.
        """
    
        df = (freqs[1] - freqs[0])
        if n_segments is None:
            if isinstance(f_segments, float):
                # Smoothing bandwidth
                bandwidth = int(f_segments / df)
                # Segment frequencies
                f_segments_arr = freqs[0::bandwidth]
                # Add the last frequency if it is not included
                if freqs[-1] not in f_segments_arr:
                    f_segments_arr = np.concatenate((f_segments_arr, np.atleast_1d(freqs[-1])))
            elif hasattr(f_segments, '__array__') or isinstance(f_segments, list):
                f_segments_arr = np.asarray(f_segments)
            else:
                raise TypeError("f0 should be a float or array_like")
            
            # Number of segments
            n_segments = len(f_segments_arr)
        # Indices of the segment bounds
        else:
            f_segments_arr = np.linspace(freqs[0], freqs[-1], n_segments)

        i_seg = np.round(f_segments_arr / df).astype(int)
        # Sizes of all intervals
        segment_sizes = i_seg[1:] - i_seg[:-1]
        # Middle frequencies
        freqs_h = (f_segments_arr[:-1] + f_segments_arr[1:]) / 2.0

        # Compute the averages over each segment
        # if method == 'mean':
        power_avg = np.array(
            [np.sum(power[i_seg[j]:i_seg[j+1]], axis=0) / segment_sizes[j]
            for j in range(n_segments-1)], dtype=power.dtype)

        # mask out the frequencies outside the specified range
        f_min = f_min or freqs_h.min()
        f_max = f_max or freqs_h.max()

        mask = (freqs_h >= f_min) & (freqs_h <= f_max)
        freqs_h = freqs_h[mask]
        power_avg = power_avg[mask]
        segment_sizes = segment_sizes[mask]
        
        return freqs_h, power_avg, segment_sizes

def main():

    backend_path = "/data/asantini/globalfit/erebor/mojito_runs/galfor_test_fd_parameter_estimation_main.h5"
    backend = HDFBackend(backend_path, read_only=True)

    psd_transform, galfor_transform = get_noise_transform_fn()

    chains = backend.get_chain(temp_index=0)

    noise_params = psd_transform.both_transforms(chains["psd"][:, 0])
    num_dims = noise_params.shape[-1]
    median_noise_params = np.median(noise_params.reshape(-1, num_dims), axis=0)

    galfor_params = galfor_transform.both_transforms(chains["galfor"][:, 0])
    num_dims = galfor_params.shape[-1]
    median_galfor_params = np.median(galfor_params.reshape(-1, num_dims), axis=0)

    loader = L1ProcessingStep(
        L1_folder=datapath,
        source_types=source_types,
        source_ids=source_ids,
        orbits_class=L1Orbits,
        orbits_kwargs=dict(force_backend=BACKEND, frame="icrs"),
        verbose=True,
        do_plots=False
    )

    preprocess_kwargs = dict(
        highpass_kwargs=highpass_kwargs,
        lowpass_kwargs=lowpass_kwargs,
        trim_kwargs=trim_kwargs,
        downsample_kwargs=downsample_kwargs,
        Tobs=Tobs,
    )

    times, _ = loader.process(**preprocess_kwargs)

    Nt = len(times)
    df = 1.0 / (Nt * dt)
    Nf = Nt // 2 + 1

    basis_kwargs = dict(N=Nf, df=df, min_freq=START_FREQ, max_freq=END_FREQ)
    domain_settings = FDSettings(**basis_kwargs, force_backend=BACKEND)

    window_alpha = WINDOW_TAPER_DURATION / (Nt * dt)
    window = tukey(Nt, alpha=window_alpha)

    input_data_residual_array, orbits = loader.pour(
            settings=domain_settings, window=window, return_orbits=True
        )

    input_data_residual_array.data_length = len(domain_settings.f_arr)
    
    input_data_residual_array._store_time_and_frequency_information(df=domain_settings.df, f_arr=domain_settings.f_arr)
    original_t0 = loader.original_t0
    data_t0 = loader.td_signal.settings.t0
    
    galactic_grid_kwargs = {'R_d': 2.5, 'z_d': 0.5, 't0': data_t0, 'N_lambda': 90, 'N_beta': 60}
    sensitivity_init_kwargs = {'tdi_generation': 2, 'mask_percentage': 0.02, 'galactic_grid_kwargs': galactic_grid_kwargs}

    sensitivity_backend = XYZSensitivityBackend(orbits=orbits, settings=domain_settings, force_backend=BACKEND, window_values=window, **sensitivity_init_kwargs)
    
    sensitivity_backend.set_sensitivity_matrix(*median_noise_params)
    
    logger.info("Initialized sensitivity backend with instrumental noise parameters")

    analysis_container = AnalysisContainer(
        input_data_residual_array,
        sensitivity_backend
        )
    
    acs = AnalysisContainerArray([analysis_container], gpus=[0])

    start_freq_ind = input_data_residual_array.start_freq_ind

    band_edges, band_N_vals = get_band_structure(domain_settings.f_arr.get()[0], domain_settings.f_arr.get()[-1], Tobs, df)

    logger.info(f"Start frequency index for input data residuals is {start_freq_ind}")    
    logger.info("Created analysis container with input data and sensitivity backend")

    iteratively_resolved_population = np.load(iteratively_resolved_population_path, allow_pickle=True)

    frequencies = iteratively_resolved_population["Frequency"]

    in_band = (frequencies > band_edges[0]) & (frequencies < band_edges[-1])
    logger.info(f"Keeping {np.sum(in_band)} out of {len(iteratively_resolved_population)} iteratively resolved GB sources within the band limits {START_FREQ} - {END_FREQ}")
    iteratively_resolved_population = iteratively_resolved_population[in_band]

    subset_inds = np.array([int(name.split('_')[1]) for name in iteratively_resolved_population["Name"]])

    catalog_entries = get_catalog_entries(loader, subset_inds, data_t0, original_t0)
    
    gb_wave_gen = GBGPU(orbits=orbits, t0=data_t0, force_backend=BACKEND)
    gb_wave_gen.gpus = [0]

    waveform_kwargs = {'dt': dt, 'T': Tobs, 'use_c_implementation': True, 'start_freq_ind': start_freq_ind, 'tdi_channel_setup': 'XYZ', 'tdi2': True, 'oversample': 4,'window': 'tukey', 'window_alpha': window_alpha}

    residual_plus_one = subtract_gb_signal_from_residual(acs, gb_wave_gen, waveform_kwargs, catalog_entries, band_edges, band_N_vals, sign_phi0=1)
    residual_minus_one = subtract_gb_signal_from_residual(acs, gb_wave_gen, waveform_kwargs, catalog_entries, band_edges, band_N_vals, sign_phi0=-1)

    # plt.style.use(style)
    # from cantuccio.visuals import CATEGORICAL_COLORLIST
    # set it as color cycle
    # plt.rcParams['axes.prop_cycle'] = plt.cycler(color=CATEGORICAL_COLORLIST)

    # color = CATEGORICAL_COLORLIST[0]
    color = "tab:blue"
    fig, decimated_freqs, decimated_power = plot_comparison(domain_settings.f_arr.get(), analysis_container.data_res_arr.data_res_arr.arr.get()[0], label="Input data", dt=dt, color=color)
    # color = CATEGORICAL_COLORLIST[1]
    color = "tab:orange"
    fig, average_freqs, average_power = plot_comparison(domain_settings.f_arr.get(), residual_minus_one[0], label="Residual", dt=dt, fig=fig, color=color)

    # -- add noise / MCMC foreground / fiducial foreground
    f_arr = domain_settings.f_arr.get()
    sens_mat_noise = sensitivity_backend.compute_sensitivity_matrix(domain_settings.f_arr, *median_noise_params).get()
    sens_mat_galfor = sensitivity_backend.compute_sensitivity_matrix(domain_settings.f_arr, *median_noise_params, *median_galfor_params).get()
    sens_mat_galfor_fiducial = sensitivity_backend.compute_sensitivity_matrix(domain_settings.f_arr, Soms_d, Sa_a, Amp, alpha, f_1, kn, f_2).get()

    plt.loglog(f_arr, sens_mat_noise[0, 0], label="Noise", color="white", linestyle="--")
    plt.loglog(f_arr, sens_mat_galfor[0, 0], label="Noise + Foreground (MCMC median)", color="k", linestyle=":")
    plt.loglog(f_arr, sens_mat_galfor_fiducial[0, 0], label="Noise + Foreground (Fiducial)", color="tab:green", linestyle="-.")

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc="lower center", bbox_to_anchor=(0.5, 1), ncols=2)

    plt.savefig("residual_comparison_full.png", dpi=300)

    from scipy.interpolate import CubicSpline
    noise_spline = CubicSpline(f_arr, sens_mat_galfor_fiducial[0, 0])

    ratio = average_power / noise_spline(average_freqs) 
    plt.figure(); plt.loglog(average_freqs, ratio); plt.xlabel('Frequency [Hz]'); plt.ylabel("residual / fiducial"); plt.savefig('ratio.png')

    breakpoint()

if __name__ == "__main__":
    main()



