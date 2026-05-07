import numpy as np
from scipy import signal
from lisatools.domains import *
import matplotlib.pyplot as plt
import numpy as np

# Ensure LaTeX is disabled
plt.rcParams['text.usetex'] = False

# dt = 5.0

# Nf = 1536
# Nt = 100
# N =  Nf * Nt # int(1e6 / nperseg) * nperseg
# t_arr = np.arange(N) * dt
# phi0 = 0.782340988
# f0 = 3.9e-3
# wave = 1e-24 * np.sin(2 * np.pi * f0 * t_arr + phi0)
# y = np.tile(wave, (3, 1))


# df = 1. / (N * dt)

from fastlisaresponse.tdionfly import GBTDIonTheFly

import numpy as np
import matplotlib.pyplot as plt

try:
    import cupy as cp
except (ImportError, ModuleNotFoundError) as e:
    pass

from fastlisaresponse.tdionfly import GBTDIonTheFly
from fastlisaresponse.tdiconfig import TDIConfig
from lisatools.detector import DefaultOrbits
from lisatools.utils.constants import *
from lisatools.utils.utility import AET

from lisatools.domains import WAVELET_DURATION, TDSignal, TDSettings, FDSignal, FDSettings, WDMSignal, WDMSettings, WDMLookupTable
from fastlisaresponse.gbcomps import GBWDMComputations
from scipy.signal.windows import tukey

force_backend = "cpu"
xp = np if force_backend == "cpu" else cp
orbits = DefaultOrbits(force_backend=force_backend)
orbits.configure(linear_interp_setup=True)
tdi_config = TDIConfig("1st generation", force_backend=force_backend)
dt = 2.5
Tobs = 2 * YRSID_SI

N_sparse = 256
t_tdi = xp.linspace(0.0, Tobs, N_sparse + 2)[1:-1]

from lisatools.sensitivity import XYZ2SensitivityMatrix
from lisatools.detector import sangria, scirdv1
from lisatools.datacontainer import DataResidualArray
from lisatools.analysiscontainer import AnalysisContainer

# sens_mat_wdm = XYZ2SensitivityMatrix(wdm_settings, model=scirdv1)
# breakpoint()
tukey_alpha = 0.00

store_path = "./wdm_lookup_new_test_5.h5"
if os.path.exists(store_path):
    wdm_lookup_table = WDMLookupTable.from_file(store_path, force_backend=force_backend)
    wdm_settings = WDMSettings(*wdm_lookup_table.args, **wdm_lookup_table.kwargs)
    Nt = wdm_settings.Nt
    Nf = wdm_settings.Nf
    N = wdm_settings.N
    Tobs = wdm_settings.Tobs

else:
    Nf = -1
    Nt = -1
    for tmp in np.linspace(.5, 0.75, 1000):
        wavelet_duration = int(tmp / 365 * YRSID_SI / dt) * dt
        Nt = int(Tobs / wavelet_duration)
        Tobs = Nt * wavelet_duration
        N = int(Tobs / dt)
        Nf = int(N / Nt)
        print(tmp, Nf, Nt)
        if (Nt % 2 == 0) and (Nf % 2 == 0):
            break
    
    wdm_settings = WDMSettings(Nf, Nt, dt, force_backend=force_backend)
    time_layers = wdm_settings.Nt
    td_window = xp.asarray(tukey(wdm_settings.Nf * time_layers, alpha=tukey_alpha))
    m_ref = int(3e-3 / wdm_settings.layer_df)
    norm_freq_single_layer, m_diffs, _ = WDMLookupTable.apply_eps_frequency(0.02, wdm_settings, m_ref=m_ref, num_layers_diff=3)
    fdot_vals = np.array([0.0])  # WDMLookupTable.apply_eps_fdot(0.01, wdm_settings, fdot_max_factor=1.0) 

    wdm_lookup_table = WDMLookupTable(wdm_settings, 3, norm_freq_single_layer=norm_freq_single_layer, m_diffs=m_diffs, fdot_vals=fdot_vals, m_ref=m_ref, time_layers=time_layers, batch_size_gen=5, td_window=td_window, store_path=store_path)

# f_arr = xp.linspace(wdm_lookup_table.f_vals.min(), wdm_lookup_table.f_vals.max(), 100)
#xp.random.uniform(wdm_settings.f_arr.min(), wdm_settings.f_arr.max(), 10)
# fdot_arr = xp.random.uniform(wdm_lookup_table.fdot_vals.min(), wdm_lookup_table.fdot_vals.max(), 10)

t_wdm = wdm_settings.t_arr

# amp0 = 1.0
# f0_check = wdm_lookup_table.f_vals[1300]
# fdot0_check = 0.0  # wdm_lookup_table.fdot_vals[2]
# phi0 = np.pi / 2.0
# for amp0, f0_check, fdot0_check, phi0, note in [
#     [1.0, wdm_lookup_table.f_ref, 0.0, 0.0, "exact fref"],
#     [1.0, wdm_lookup_table.f_ref, 0.0, np.pi/ 4., "exact fref"],
#     [1.0, wdm_lookup_table.f_vals[404], wdm_lookup_table.fdot_vals[102], 0.0, "on table node"],
#     [1.0, 4.1340193841e-3, 0.0, 0.0, ""],
#     [1.0, 4.1340193841e-3, 1e-15, 0.0, ""],
#     [1.0, 4.1340193841e-3, 1e-14, 0.0, ""],
#     [1.0, 4.1340193841e-3, 1e-13, 0.0, ""],
# ]:
#     t_ref = int(wdm_settings.Nt / 2) * wdm_settings.layer_dt
#     td_set = TDSettings(wdm_settings.N, dt, force_backend=force_backend)
#     t_check = xp.arange(wdm_settings.N) * dt
#     wave_check = amp0 * xp.sin(2 * xp.pi * (f0_check * (t_check - t_ref) + 1/2 * fdot0_check * (t_check - t_ref) ** 2) + phi0)

#     wave_check_wdm = TDSignal(wave_check[None, :], td_set).wdmtransform(wdm_settings, window=xp.asarray(tukey(wdm_settings.N, alpha=tukey_alpha)))
#     phi_t = 2 * xp.pi * (f0_check * (t_wdm - t_ref) + 1/2 * fdot0_check * (t_wdm - t_ref) ** 2) + phi0
#     freq_t = f0_check + fdot0_check * (t_wdm - t_ref)
#     fdot_t = xp.full_like(freq_t, fdot0_check)
#     amp_t = amp0 * xp.ones_like(t_wdm)

#     n_arr = xp.arange(wdm_settings.Nt)
#     wdm_coeffs, m_layers = wdm_lookup_table.get_wdm_coeffs(amp_t, phi_t, freq_t, fdot_t, n_arr, num_m_layers=1)
#     fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, sharey=True)
#     fig.set_size_inches(14, 10)
#     _check_fill_wave = xp.zeros_like(wave_check_wdm[0])
#     _check_fill_wave[m_layers.flatten(), xp.repeat(n_arr[:, None], m_layers.shape[-1], axis=-1).flatten()] = wdm_coeffs.flatten()
#     _check_fill_wdm = WDMSignal(_check_fill_wave[None, :], wdm_settings)
#     # min_val = np.min([wave_check_wdm[:].min().item(), _check_fill_wdm[:].min().item()]).item()
#     max_val = np.max([wave_check_wdm[:].max().item(), _check_fill_wdm[:].max().item()]).item()

#     cax1 = fig.add_axes([0.9, 0.55, 0.05, 0.25])
#     cax2 = fig.add_axes([0.9, 0.2, 0.05, 0.25])

#     ind_check = int(f0_check / wdm_settings.layer_df)
#     wave_check_wdm.heatmap(index=0, fig=fig, ax=ax1, vmin=-max_val, vmax=max_val, cax=cax1)
#     _check_fill_wdm.heatmap(index=0, fig=fig, ax=ax2, vmin=-max_val, vmax=max_val)
#     try:
#         _tmp_diff = (wave_check_wdm[:] - _check_fill_wdm[:]).get()  #  / wave_check_wdm[:].get()
#     except AttributeError:
#         _tmp_diff = (wave_check_wdm[:] - _check_fill_wdm[:])  # / wave_check_wdm[:]

#     difference = WDMSignal(np.log10(np.abs(_tmp_diff)), wdm_settings)  #  / wave_check_wdm[:]
#     difference.heatmap(index=0, fig=fig, ax=ax3, vmin=difference[:].min().item(), vmax=difference[:].max().item(), cax=cax2, cmap=cm.Blues)
#     ax1.set_ylim((m_layers.min() - 1) * wdm_settings.layer_df, (m_layers.max() + 2) * wdm_settings.layer_df)
#     ax1.set_ylabel("Frequency (Hz)")
#     ax2.set_ylabel("Frequency (Hz)")
#     ax3.set_ylabel("Frequency (Hz)")
#     ax3.set_xlabel("Time (s)")
#     cax1.set_ylabel("WDM Coeff")
#     cax2.set_ylabel("log10(abs(delta))")
#     # fig.savefig(f"wdm_check_3_alpha_{tukey_alpha}.png")
#     check1 = wave_check_wdm[0, wdm_settings.f_ind_array, 10:-10]
#     check2 = _check_fill_wdm[0, wdm_settings.f_ind_array, 10:-10]
#     keep = np.where(check2)
#     overlap = np.sum(check1 * check2) / np.sqrt(np.sum(check1 * check1) * np.sum(check2 * check2))

#     check3 = wave_check_wdm[0, wdm_settings.f_ind_array, int(Nt / 2) - 10:int(Nt / 2) + 10]
#     check4 = _check_fill_wdm[0, wdm_settings.f_ind_array, int(Nt / 2) - 10:int(Nt / 2) + 10]
#     keep2 = np.where(check4)
#     overlap_center = np.sum(check3 * check4) / np.sqrt(np.sum(check3 * check3) * np.sum(check4 * check4))

#     fig.suptitle(f"f0: {f0_check:.2e}, fdot: {fdot0_check:.2e}, phi0: {phi0:.2g}, note: {note}, mismatch: {1 - overlap:.2e}, center_mismatch {1 - overlap_center:.2e}")
#     # fig.savefig(f"f0_{f0_check:.2e}_fdot_{fdot0_check:.2e}_phi0_{phi0:.2g}_main.png")
#     # ax1.set_xlim((int(wdm_settings.Nt / 2) - 10) * wdm_settings.layer_dt, (int(wdm_settings.Nt / 2) + 10) * wdm_settings.layer_dt)
#     # fig.savefig(f"f0_{f0_check:.2e}_fdot_{fdot0_check:.2e}_phi0_{phi0:.2g}zoom_center.png")
#     plt.show()
#     plt.close()
#     breakpoint()
# # plt.close()
# # ax1.plot(wave_check_wdm[0, m_layers[0,0]], lw=3)
# # ax2.plot(wdm_coeffs.squeeze(), "--", lw=2)
# # plt.show()
# # breakpoint()
# plt.close()

num_bin = 1

data_t_arr = np.arange(N) * dt
keep = (data_t_arr > t_tdi[0]) & (data_t_arr < t_tdi [-1])
tdi_t_arr = data_t_arr[keep]

ind = int(3e-3 / wdm_settings.layer_df) + 3
num = 10

f_max = None  # 30e-3
f_min = None  # 0.1e-3

del wdm_settings
_wdm_settings = WDMSettings(Nf, Nt, dt, max_freq=f_max, min_freq=f_min, force_backend=force_backend)
t_min = None  # 20 * _wdm_settings.layer_dt
t_max = None  # (_wdm_settings.Nt - 20) * _wdm_settings.layer_dt
wdm_settings = wdm_set = WDMSettings(Nf, Nt, dt, max_freq=f_max, min_freq=f_min, max_time=t_max, min_time=t_min, force_backend=force_backend)

# tmp_dat = np.zeros(wdm_set.N)
# tmp_dat[0] = 1.0
# _frq = np.fft.rfftfreq(wdm_set.N, wdm_set.data_dt)
# tmp_dat_td = TDSignal(tmp_dat, TDSettings(wdm_set.N, wdm_set.data_dt))
# tmp_dat_fd = tmp_dat_td.fft(FDSettings(_frq.shape[0], _frq[1] - _frq[0]))
# check_1 = tmp_dat_td.fft().ifft()
# tmp_dat_check_td_1 = tmp_dat_fd.wdmtransform(wdm_set).wdm_to_fd(tmp_dat_fd.settings).ifft()
# tmp_dat_check_td = tmp_dat_td.wdmtransform(wdm_set).wdm_to_td()

# assert np.allclose((_tmp := tmp_dat_check_td[:, 0]), np.ones_like(_tmp))
# assert np.allclose((_tmp := tmp_dat_check_td[:, 1:]), np.zeros_like(_tmp))


t_ref = int(Nt / 2) * wdm_settings.layer_dt

gb_comps = GBWDMComputations(wdm_lookup_table, Tobs, t_ref, orbits=orbits, tdi_config=tdi_config, force_backend=force_backend)

template_fill = xp.zeros(3 * np.prod(wdm_settings.basis_shape), dtype=float)

amp = np.full(num_bin, 7.0e-23)
f0 = np.full(num_bin, 12.0e-3)  # (ind + i / num) * wdm_settings.layer_df)
fdot = np.full(num_bin, 1e-15)
fddot = np.full(num_bin, 0.0)
phi0 = np.full(num_bin, 2.09802430298)
inc = np.full(num_bin, 0.23984234)
psi = np.full(num_bin, 1.234019814)
lam = np.full(num_bin, 4.09808143)
beta = np.full(num_bin, 0.7090)
params = np.array([amp, f0, fdot, fddot, phi0, inc, psi, lam, beta]).T

from lisatools.sensitivity import XYZ2SensitivityMatrix, AET1SensitivityMatrix, XYZ1SensitivityMatrix
from lisatools.detector import sangria, scirdv1
from lisatools.datacontainer import DataResidualArray
from lisatools.analysiscontainer import AnalysisContainer, AnalysisContainerArray

freqs = np.fft.rfftfreq(wdm_settings.N, wdm_settings.data_dt)
fd_set = FDSettings(freqs.shape[0], freqs[1] - freqs[0])
sens_mat_fd = XYZ1SensitivityMatrix(fd_set, model=scirdv1)
sens_mat_wdm = XYZ1SensitivityMatrix(wdm_set, model=scirdv1)
wdm_dat = DataResidualArray(WDMSignal(np.zeros((3,) + wdm_set.basis_shape), wdm_set))
wdm_holder = AnalysisContainerArray([AnalysisContainer(wdm_dat, sens_mat_wdm)])

gb_comps.fill_global_wdm(template_fill, params, wdm_holder, data_index=None)
gb_comps.fill_global_wdm(wdm_holder.linear_data_arr[0], params, wdm_holder, data_index=None)
check_ll = gb_comps.get_ll_wdm(params, wdm_holder, data_index=None, noise_index=None)
check_opt_snr = gb_comps.h_h_out[0].item() ** (1/2)
holder_ip = wdm_holder[0].inner_product()  # source_only=True)
breakpoint()

for i in range(0, num)[:1]:
    tmp_diff = 500.0

    t_ref = int(Nt / 2) * wdm_settings.layer_dt
    gb_gen = GBTDIonTheFly(
        t_tdi, 
        Tobs,
        t_ref,
        1. / dt,
        num_bin,
        n_params=9,
        tdi_config=tdi_config,
        orbits=orbits,
        tdi_chan="XYZ",
        force_backend=force_backend,
    )
    
    t_td_wdm = wdm_settings.t_arr[1:-1]
    gb_gen_deriv = GBTDIonTheFly(
        t_td_wdm, 
        Tobs,
        t_ref,
        1. / dt,
        num_bin,
        n_params=9,
        tdi_config=tdi_config,
        orbits=orbits,
        tdi_chan="XYZ",
        force_backend=force_backend,
    )

    gb_gen_down_1 = GBTDIonTheFly(
        (t_td_wdm - tmp_diff),  # 1e-5 * t_td_wdm[-1]), 
        Tobs,
        t_ref,
        1. / dt,
        num_bin,
        n_params=9,
        tdi_config=tdi_config,
        orbits=orbits,
        tdi_chan="XYZ",
        force_backend=force_backend,
    )

    gb_gen_up_1 = GBTDIonTheFly(
        (t_td_wdm + tmp_diff),  #  + 1e-5 * t_td_wdm[-1]), 
        Tobs,
        t_ref,
        1. / dt,
        num_bin,
        n_params=9,
        tdi_config=tdi_config,
        orbits=orbits,
        tdi_chan="XYZ",
        force_backend=force_backend,
    )

    print("\n\n")
    output_deriv = gb_gen_deriv(amp, f0, fdot, fddot, phi0, inc, psi, lam, beta, return_spline=True)
    
    output = gb_gen(amp, f0, fdot, fddot, phi0, inc, psi, lam, beta, return_spline=True)
    output_down_1 = gb_gen_down_1(amp, f0, fdot, fddot, phi0, inc, psi, lam, beta, return_spline=True)
    output_up_1 = gb_gen_up_1(amp, f0, fdot, fddot, phi0, inc, psi, lam, beta, return_spline=True)
    
    # FD 
    # bin_i = int(f0[0] * Tobs) - int(output.t_arr.shape[-1] / 4)
    # carrier_frequency = bin_i / Tobs

    # heterodyned_phase = output.tdi_phase + output.phase_ref - 2 * np.pi * carrier_frequency * output.t_arr
    # heterodyned_signal = -output.tdi_amp * np.sin(heterodyned_phase)
    # gb_fd = np.fft.fftshift(np.fft.fft(heterodyned_signal, axis=-1))

    # breakpoint()
    tdi_output = np.zeros((num_bin, 3, len(data_t_arr))) 

    # t_tdi = 
    tdi_output[:, :, keep]= output.eval_tdi(tdi_t_arr)
    from scipy.signal.windows import tukey

    td = TDSignal(tdi_output[0, :3], settings=TDSettings(tdi_output.shape[-1], dt, force_backend=force_backend))

    # y[:, 0] = 1.0
    # y[:, 1:] = 0.0
    # td = TDSignal(y, TDSettings(N, dt, force_backend=force_backend))
    # fd = FDSignal(np.fft.rfft(y), FDSettings(df))
    # stft = STFTSignal(signal.stft(y, fs=(1 / dt), nperseg=nperseg), STFTSettings(big_dt, big_df))
    # new_fd = td.transform(FDSettings(df), window=tukey(y.shape[-1], alpha=0.05))
    # new_td = fd.transform(TDSettings(dt))

    Tobs = N * dt

    # wdm_set.frequency_layer_mask = ((wdm_set.f_arr >= 5e-5) &(wdm_set.f_arr <= 40e-3))

    _fd_from_td = td.fft()

    _fd_set = _fd_from_td.settings
    fd_set = FDSettings(_fd_set.N, _fd_set.df, min_freq=f_min, max_freq=f_max, force_backend=force_backend)
    fd_from_td = FDSignal(_fd_from_td[:], fd_set)
    
    # fd_set.frequency_layer_mask = ((fd_set.f_arr >= 5e-5) &(fd_set.f_arr <= 40e-3))
    # fd_from_td.frequency_layer_mask = ((fd_from_td.f_arr >= 5e-5) &(fd_from_td.f_arr <= 40e-3))
    # wdm_from_fd = fd_from_td.transform(wdm_set)
    wdm_from_td = td.transform(wdm_set)
    # new_td = TDSignal(np.fft.irfft(td.fft()[:] * np.exp(-1j * np.pi / 2.) / dt, axis=-1), td.settings)
    # wdm_from_td_new = new_td.transform(wdm_set)
    wdm_comples_val = wdm_from_td[1, ind, int(Nt / 2):int(Nt / 2) + 5] + 1j * wdm_from_td[0, ind, int(Nt / 2):int(Nt / 2) + 5]
    print(f0[0], ind * wdm_settings.layer_df, f0[0] - ind * wdm_settings.layer_df, i)
    # print(wdm_from_td[0, ind+ i, int(Nt /2)] ** 2 + wdm_from_td_new[0, ind+ i, int(Nt /2)] ** 2)
    print("\n")

    #simulate kernel
    ref_phase_down = output_down_1.phase_ref  # 2 * np.pi * int(f0[0] / wdm_settings.layer_df) * wdm_settings.layer_df * (output_deriv.t_arr - t_ref) 
    ref_phase_mid = output_deriv.phase_ref  # 2 * np.pi * int(f0[0] / wdm_settings.layer_df) * wdm_settings.layer_df * (output_deriv.t_arr - t_ref) 
    ref_phase_up = output_up_1.phase_ref  # 2 * np.pi * int(f0[0] / wdm_settings.layer_df) * wdm_settings.layer_df * (output_deriv.t_arr - t_ref) 
    tdi_phase_down = -np.angle(output_down_1.X * np.exp(1j * ref_phase_down))
    tdi_phase_mid = -np.angle(output_deriv.X * np.exp(1j * ref_phase_mid))
    tdi_phase_up = -np.angle(output_up_1.X * np.exp(1j * ref_phase_up))

    deriv_delta_t = output_up_1.t_arr - output_down_1.t_arr
    # f_deriv_tdi = (tdi_phase_up[0] - tdi_phase_down[0]) / (2 * deriv_delta_t) / (2 * np.pi)
    
    layer_base_freq = int(f0[0] / wdm_settings.layer_df) * wdm_settings.layer_df
    ref_phase_layer_down = 2 * np.pi * layer_base_freq * (output_down_1.t_arr - t_ref) 
    ref_phase_layer_mid = 2 * np.pi * layer_base_freq * (output_deriv.t_arr - t_ref) 
    ref_phase_layer_up = 2 * np.pi * layer_base_freq * (output_up_1.t_arr - t_ref) 
    residual_phase_down = ref_phase_down #  - ref_phase_layer_down
    residual_phase_mid = ref_phase_mid #  - ref_phase_layer_mid
    residual_phase_up = ref_phase_up #  - ref_phase_layer_up

    # if we assume constant over window
    # we can also 
    # residual_frequency = np.diff(residual_phase) / np.diff(output_deriv.t_arr) / ( 2* np.pi)
    # tdi_frequency = np.diff(tdi_phase_mid) / np.diff(output_deriv.t_arr) / ( 2* np.pi)
    residual_frequency = (residual_phase_up - residual_phase_down) / (deriv_delta_t) / (2. * np.pi)
    tdi_frequency = (tdi_phase_up - tdi_phase_down) / (deriv_delta_t) / (2. * np.pi)
    
    f_deriv = residual_frequency + tdi_frequency #  + layer_base_freq

    residual_fdot = 4 * (residual_phase_up - 2 * residual_phase_mid + residual_phase_down) / (deriv_delta_t ** 2) / (2 * np.pi)
    # 4? from a 1 / 2**2 maybe?
    tdi_fdot = 4 * (tdi_phase_up - 2 * tdi_phase_mid + tdi_phase_down) / (deriv_delta_t ** 2) / (2 * np.pi)
    
    # f_deriv = residual_frequency + tdi_frequency#  + layer_base_freq
    
    # residual_fdot = np.diff(residual_frequency) / np.diff(output_deriv.t_arr[0, :-1])
    # tdi_fdot = np.diff(tdi_frequency) / np.diff(output_deriv.t_arr[0, :-1])
    fdot_deriv = residual_fdot + tdi_fdot

    _f_deriv_tdi = output_deriv.tdi_phase_spl(output_deriv.tdi_phase_spl.x, derivative=1)[0, 0] / (2 * np.pi)
    _f_deriv_ref = output_deriv.phase_ref_spl(output_deriv.phase_ref_spl.x, derivative=1)[0] / (2 * np.pi)
    _f_deriv = _f_deriv_ref + _f_deriv_tdi

    fdot_deriv_tdi = output_deriv.tdi_phase_spl(output_deriv.tdi_phase_spl.x, derivative=2)[0, 0] / (2 * np.pi)
    fdot_deriv_ref = output_deriv.phase_ref_spl(output_deriv.phase_ref_spl.x, derivative=2)[0]  / (2 * np.pi)
    _fdot_deriv = fdot_deriv_ref + fdot_deriv_tdi

    # pi/2 PHASE SHIFT !!!!!!!!!!!!!!!!!!!!!!!!!
    phi_t = (tdi_phase_mid + ref_phase_mid)[0] + np.pi / 2. #  (np.angle(output_deriv.X).squeeze())# [:-2] # % (2 * np.pi)
    freq_t = f_deriv.copy().squeeze() # np.full_like(phi_t, f0[0])  # 
    fdot_t = np.full_like(freq_t, 0.0)  # sfdot[0])  # 
    amp_t = np.abs(output_deriv.X).squeeze()# [:-2]

    n_arr = xp.arange(wdm_settings.Nt)[wdm_settings.active_slice_t][1:-1]
    n_min = wdm_settings.ind_min_t
    m_min = wdm_settings.ind_min_f

    # breakpoint()
    wdm_coeffs, m_layers = wdm_lookup_table.get_wdm_coeffs(amp_t, phi_t, freq_t, fdot_t, n_arr, num_m_layers=2)

    gb_fill_wave = xp.zeros((wdm_set.Nf, wdm_set.Nt))
    keep_m = (m_layers >= 0) & (m_layers < wdm_settings.Nf)
    gb_fill_wave[m_layers[keep_m], xp.repeat(n_arr[:, None], m_layers.shape[-1], axis=-1)[keep_m]] = wdm_coeffs[keep_m]
    # gb_fill_wave[:] = xp.roll(gb_fill_wave, 2, axis=-1)

    gb_fill_wave = WDMSignal(np.asarray([gb_fill_wave, gb_fill_wave]), wdm_settings)
    # gb_fill_wave = WDMSignal(template_fill.reshape((3,) + wdm_settings.basis_shape), wdm_settings)
    
    # fdot_deriv = (phase_up - 2 * phase_mid + phase_up) / (deriv_delta_t * deriv_delta_t) / (2 * np.pi)
    plt.close()
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, sharey=True)
    fig.set_size_inches(14, 10)
     # min_val = np.min([wave_check_wdm[:].min().item(), _check_fill_wdm[:].min().item()]).item()
    max_val = np.max([wdm_from_td[:].max().item(), wdm_from_td[:].max().item()]).item()

    cax1 = fig.add_axes([0.9, 0.55, 0.05, 0.25])
    cax2 = fig.add_axes([0.9, 0.2, 0.05, 0.25])

    # ind_check = int(fdot[0] / wdm_settings.layer_df)
    wdm_from_td.heatmap(index=0, fig=fig, ax=ax1, vmin=-max_val, vmax=max_val, cax=cax1)
    gb_fill_wave.heatmap(index=0, fig=fig, ax=ax2, vmin=-max_val, vmax=max_val) # , mag=True
    try:
        _tmp_diff = (np.abs(wdm_from_td[0]) - np.abs(gb_fill_wave[:])).get()  #  / wave_check_wdm[:].get()
    except AttributeError:
        _tmp_diff = (np.abs(wdm_from_td[0]) - np.abs(gb_fill_wave[:]))  # / wave_check_wdm[:]

    difference = WDMSignal(np.log10(np.abs(_tmp_diff)), wdm_settings)  #  / wave_check_wdm[:]
    difference.heatmap(index=0, fig=fig, ax=ax3, vmin=difference[:].min().item(), vmax=difference[:].max().item(), cax=cax2, cmap=cm.Blues)

    m_layer_min = m_layers[m_layers != -1].min()
    m_layer_max = m_layers[m_layers != -1].max()
    ax1.set_ylim(m_layer_min * wdm_settings.layer_df, (m_layer_max + 1) * wdm_settings.layer_df)
    ax1.set_ylabel("Frequency (Hz)")
    ax2.set_ylabel("Frequency (Hz)")
    ax3.set_ylabel("Frequency (Hz)")
    ax3.set_xlabel("Time (s)")
    cax1.set_ylabel("WDM Coeff")
    cax2.set_ylabel("log10(abs(delta))")
    # fig.savefig(f"wdm_check_3_alpha_{tukey_alpha}.png")
    
    for cut in [1, 20, 100, 200, 500, 1000]:
        check3 = wdm_from_td[0, :, cut:-cut]
        check4 = gb_fill_wave[0, :, cut:-cut]
        keep2 = np.where(check4)
        overlap_center = np.sum(check3 * check4) / np.sqrt(np.sum(check3 * check3) * np.sum(check4 * check4))
        print(f"overlap center. Cut ends amount: {cut}. Mismatch: {1.- overlap_center}.")
    # fig.suptitle(f"f0: {f0_check:.2e}, fdot: {fdot0_check:.2e}, phi0: {phi0:.2g}, note: {note}, mismatch: {1 - overlap:.2e}, center_mismatch {1 - overlap_center:.2e}")
    # fig.savefig(f"f0_{f0_check:.2e}_fdot_{fdot0_check:.2e}_phi0_{phi0:.2g}_main.png")
    # ax1.set_xlim((int(wdm_settings.Nt / 2) - 10) * wdm_settings.layer_dt, (int(wdm_settings.Nt / 2) + 10) * wdm_settings.layer_dt)
    # fig.savefig(f"f0_{f0_check:.2e}_fdot_{fdot0_check:.2e}_phi0_{phi0:.2g}zoom_center.png")
    # plt.show()
    plt.close()
    # breakpoint()

    # breakpoint()
    # exit()
# breakpoint()
# assert np.allclose(wdm_from_fd[:] / np.abs(wdm_from_fd[:]).max(), wdm_from_td[:] / np.abs(wdm_from_td[:]).max())

# olitas_check = np.genfromtxt("olitas_wdm_impulse.dat")

# wdm_from_td = td.transform(wdm_set)


# fd_from_wdm = wdm_from_fd.transform(fd_set)

# td_from_td = wdm_from_fd.transform(fd_set)

from copy import deepcopy

min_freq = 0.1e-3 # (int(f0[0] / wdm_set.layer_df) - 5) * wdm_set.layer_df
max_freq = 30e-3 # (int(f0[0] / wdm_set.layer_df) + 5) * wdm_set.layer_df
t_min =  None  # 20 * _wdm_settings.layer_dt
t_max = None  # (_wdm_settings.Nt - 20) * _wdm_settings.layer_dt


wdm_set_here = WDMSettings(wdm_set.Nf, wdm_set.Nt, wdm_set.data_dt, min_freq=min_freq, max_freq=max_freq, min_time=t_min, max_time = t_max)
fd_set_here = FDSettings(fd_set.N, fd_set.df, min_freq=min_freq, max_freq=max_freq)
wdm_aet = DataResidualArray(WDMSignal(np.asarray(AET(*td.wdmtransform(wdm_set_here))), wdm_set_here))
fd_aet = DataResidualArray(FDSignal(np.asarray(AET(*td.fft(fd_set_here))), fd_set_here))
wdm_xyz = DataResidualArray(WDMSignal(td.wdmtransform(wdm_set_here), wdm_set_here))
fd_xyz = DataResidualArray(FDSignal(td.fft(fd_set_here), fd_set_here))

wdm_dat_here = DataResidualArray(WDMSignal(wdm_dat[:], wdm_set_here))

sens_mat_wdm_here_xyz = XYZ1SensitivityMatrix(wdm_set_here, model=scirdv1)
sens_mat_fd_here_xyz = XYZ1SensitivityMatrix(fd_set_here, model=scirdv1)

sens_mat_wdm_here_aet = AET1SensitivityMatrix(wdm_set_here, model=scirdv1)
sens_mat_fd_here_aet = AET1SensitivityMatrix(fd_set_here, model=scirdv1)

analysis_wdm_aet = AnalysisContainer(wdm_aet, sens_mat_wdm_here_aet)
analysis_fd_aet = AnalysisContainer(fd_aet, sens_mat_fd_here_aet)
ip_wdm_aet = analysis_wdm_aet.inner_product()
ip_fd_aet = analysis_fd_aet.inner_product()

analysis_wdm_xyz = AnalysisContainer(wdm_xyz, sens_mat_wdm_here_xyz)
analysis_fd_xyz = AnalysisContainer(fd_xyz, sens_mat_fd_here_xyz)
ip_wdm_xyz = analysis_wdm_xyz.inner_product()
ip_fd_xyz = analysis_fd_xyz.inner_product()

template_ip_wdm_xyz = analysis_wdm_xyz.template_snr(wdm_dat_here)[0] ** 2

gb_comps.d_d = analysis_wdm_xyz.inner_product()
check_ll_2 = gb_comps.get_ll_wdm(params, wdm_holder, data_index=None, noise_index=None)

template_ll_wdm_xyz = analysis_wdm_xyz.likelihood(wdm_dat_here)
breakpoint()
