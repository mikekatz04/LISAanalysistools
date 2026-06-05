#!/usr/bin/env python
"""Streamlined runner for gb_lookup_table_test_script.py.

Mirrors the original script but with larger dt and a half-year observation
so it fits in 8GB RAM. Skips interactive breakpoints, plt.show, and MCMC.
"""

import sys
import traceback

import matplotlib
matplotlib.use("Agg")

import numpy as np
from scipy import signal

print("[step] importing fastlisaresponse / lisatools", flush=True)
from lisatools.detector import ESAOrbits
from lisatools.utils.constants import YRSID_SI
from lisatools.response.tdiconfig import TDIConfig
from lisatools.response.tdionfly import GBTDIonTheFly
from gbgpu.gbcomps import GBWDMComputations

from lisatools.datacontainer import DataResidualArray
from lisatools.analysiscontainer import AnalysisContainer, AnalysisContainerArray
from lisatools.sensitivity import XYZ2SensitivityMatrix, AET1SensitivityMatrix
from lisatools.domains import (
    TDSettings, TDSignal, FDSettings, WDMSettings, WDMSignal, WDMLookupTable
)

backend = "cpu"
xp = np

print("[step] building settings", flush=True)
orbits = ESAOrbits(force_backend=backend)

# Same structure as the original script, scaled to fit 8GB RAM.
# Original was dt=2.5, Nf=17292, Nt=1460, Tobs ≈ 2 yr.
# Use dt=10.0 + half-year Tobs.
dt = 10.0
_Tobs = 0.5 * YRSID_SI
Nt = 64               # WDM requires Nt even — small for quick iteration
Nf = 1024             # WDM requires Nf even — small for quick iteration
wavelet_duration = Nf * dt
Tobs = Nt * wavelet_duration
Nobs = Nf * Nt
print(f"[step] dt={dt}, Nf={Nf}, Nt={Nt}, Nobs={Nobs}, Tobs/yr={Tobs/YRSID_SI:.3f}", flush=True)

tdi_config = TDIConfig("1st generation")
t_start = int(0.25 * YRSID_SI / dt) * dt    # ~3-month offset
t_arr = np.arange(Nobs) * dt + t_start
data_inj = np.zeros((3, t_arr.shape[-1]))
t_ref = t_start
N_inj = 16384

gb_tdi_kwargs = dict(
    tdi_config=tdi_config,
    orbits=orbits,
    tdi_chan="XYZ",
    force_backend=backend,
)
t_tdi_inj = xp.linspace(t_arr[0], t_arr[-1], N_inj)
gb_gen_inj = GBTDIonTheFly(t_tdi_inj, Tobs, t_ref, 1.0 / dt, 1, **gb_tdi_kwargs)

num_bin = 1
amp = np.full(num_bin, 8.0e-23)
f0 = np.full(num_bin, 3.0e-3)
fdot = np.full(num_bin, 1e-16)
fddot = np.full(num_bin, 0.0)
phi0 = np.full(num_bin, 2.09802430298)
inc = np.full(num_bin, 0.23984234)
psi = np.full(num_bin, 1.234019814)
lam = np.full(num_bin, 4.09808143)
beta = np.full(num_bin, 1.1)
params = np.array([amp, f0, fdot, fddot, phi0, inc, psi, lam, beta]).T

N = data_inj.shape[-1]
td_set = TDSettings(N, dt, force_backend=backend)
freqs = np.fft.rfftfreq(N, dt)
df = freqs[1] - freqs[0]
N_fd = len(freqs)
window = xp.asarray(signal.windows.tukey(N, alpha=0.05))

# Active-frequency band: 10 layers around f0.
layer_df = 1.0 / wavelet_duration
print(f"[step] layer_df={layer_df:.6e}", flush=True)
m_ref_central = int(round(f0[0] / layer_df))
min_freq = (m_ref_central - 5) * layer_df
max_freq = (m_ref_central + 5) * layer_df

fd_set = FDSettings(N_fd, df, min_freq=min_freq, max_freq=max_freq, force_backend=backend)
min_time = 10 * wavelet_duration
max_time = (Nt - 10) * wavelet_duration

wdm_set = WDMSettings(Nf, Nt, dt, min_freq=min_freq, max_freq=max_freq,
                     min_time=min_time, max_time=max_time)
print(
    f"[step] wdm_set: Nf={wdm_set.Nf}, Nt={wdm_set.Nt}, "
    f"Nf_active={wdm_set.Nf_active}, Nt_active={wdm_set.Nt_active}, "
    f"ind_min_f={wdm_set.ind_min_f}, ind_max_f={wdm_set.ind_max_f}, "
    f"ind_min_t={wdm_set.ind_min_t}, ind_max_t={wdm_set.ind_max_t}",
    flush=True,
)

print("[step] generating injection", flush=True)
inj_tmp = gb_gen_inj(amp, f0, fdot, fddot, phi0, inc, psi, lam, beta,
                    convert_to_ra_dec=False, return_spline=True)
data_inj[:] = inj_tmp.eval_tdi(t_arr)
print(f"[step] injection generated, |max|={np.abs(data_inj).max():.3e}", flush=True)

output_set = wdm_set
print("[step] transforming injection to WDM", flush=True)
data_inj_all = TDSignal(data_inj, settings=td_set).transform(output_set, window=window)
injection = DataResidualArray(data_inj_all)
print(f"[step] WDM injection shape={injection.data_res_arr.arr.shape}", flush=True)
sens_mat = XYZ2SensitivityMatrix(injection.data_res_arr.settings, model="scirdv1")

import os
store_path = "wdm_lookup_smoke_test.h5"
if os.path.exists(store_path):
    os.remove(store_path)
print(f"[step] building lookup table (store_path={store_path})", flush=True)
time_layers = wdm_set.Nt
td_window = xp.asarray(signal.windows.tukey(wdm_set.Nf * time_layers, alpha=0.05))
m_ref = int(f0[0] / wdm_set.layer_df)
print(f"[step] m_ref={m_ref}", flush=True)
NUM_LAYERS_DIFF = int(os.environ.get("NUM_LAYERS_DIFF", 5))
EPS_FREQ = float(os.environ.get("EPS_FREQ", 0.0025))
print(f"[step] NUM_LAYERS_DIFF={NUM_LAYERS_DIFF}, EPS_FREQ={EPS_FREQ}", flush=True)
norm_freq_single_layer, m_diffs, _ = WDMLookupTable.apply_eps_frequency(
    EPS_FREQ, wdm_set, m_ref=m_ref, num_layers_diff=NUM_LAYERS_DIFF
)
print(f"[step] m_diffs={m_diffs.tolist()} (len={len(m_diffs)}), "
      f"norm_freq_single_layer len={len(norm_freq_single_layer)}", flush=True)
fdot_vals = np.array([0.0])
nchannel = 3
wdm_lookup_table = WDMLookupTable(
    wdm_set, nchannel,
    norm_freq_single_layer=norm_freq_single_layer,
    m_diffs=m_diffs,
    fdot_vals=fdot_vals,
    m_ref=m_ref,
    batch_size_gen=20,
    td_window=td_window,
    store_path=store_path,
)
print(f"[step] lookup table built: table_cos.shape={wdm_lookup_table.table_cos.shape}, "
      f"Nt={wdm_lookup_table.settings.Nt}, fdot_steps={wdm_lookup_table.fdot_steps}, "
      f"f_steps={wdm_lookup_table.f_steps}", flush=True)

print("[step] building GBWDMComputations", flush=True)
N_sparse = 2048
t_tdi_sparse = xp.linspace(t_arr[0], t_arr[-1], N_sparse)
gb_comps = GBWDMComputations(wdm_lookup_table, Tobs, t_ref, orbits=orbits,
                            tdi_config=tdi_config, force_backend=backend)

# Build a minimal in-process clone of GBLookupWaveWrap from the test script
# (cannot import the original because it triggers __main__ guard logic).
import time as _time

class GBLookupWaveWrap:
    def __init__(self, t_arr, t_tdi_sparse, Tobs, t_ref, dt, num_bin, gb_tdi_kwargs,
                 td_set, output_set, td_window, lookup, wdm_set):
        self.t_arr, self.t_tdi_sparse = t_arr, t_tdi_sparse
        self.Tobs, self.t_ref, self.dt, self.num_bin = Tobs, t_ref, dt, num_bin
        self.gb_tdi_kwargs = gb_tdi_kwargs
        self.td_set, self.output_set = td_set, output_set
        self.td_window = td_window
        self.lookup = lookup
        self.wdm_set = wdm_set
        self.gb_gen = GBTDIonTheFly(
            self.t_tdi_sparse, self.Tobs, self.t_ref, self.dt, 1,
            **self.gb_tdi_kwargs,
        )

    def __call__(self, *params):
        params = np.asarray([params])
        assert params.shape[-1] == 9
        wave_tmp = self.gb_gen(*params.T, convert_to_ra_dec=False, return_spline=True)

        t_arr = self.output_set.t_arr + self.t_ref
        f_deriv_tdi = wave_tmp.tdi_phase_spl(np.tile(t_arr, (1, 3, 1)), derivative=1)[0] / (2 * np.pi)
        f_deriv_ref = wave_tmp.phase_ref_spl(t_arr[None, :], derivative=1)[0] / (2 * np.pi)
        f_deriv = f_deriv_ref + f_deriv_tdi

        tdi_amp = wave_tmp.tdi_amp_spl(np.tile(t_arr, (1, 3, 1)))[0]
        tdi_phase = wave_tmp.tdi_phase_spl(np.tile(t_arr, (1, 3, 1)))[0]
        ref_phase = wave_tmp.phase_ref_spl(t_arr[None, :])[0]

        phi_t = ((tdi_phase + ref_phase) + np.pi / 2.0).flatten().copy()
        freq_t = f_deriv.flatten().copy()
        fdot_t = np.full_like(freq_t, 0.0)
        amp_t = tdi_amp.flatten().copy()

        n_arr = np.tile(xp.arange(self.output_set.Nt)[self.output_set.active_slice_t], (3, 1))
        n_arr_in = n_arr.flatten().copy()
        n_min = self.output_set.ind_min_t
        m_min = self.output_set.ind_min_f

        _wdm_coeffs, _m_layers = self.lookup.get_wdm_coeffs(
            amp_t, phi_t, freq_t, fdot_t, n_arr_in,
            num_m_layers=int(os.environ.get("NUM_M_LAYERS", 2)),
        )
        wdm_coeffs = _wdm_coeffs.reshape(3, -1, _wdm_coeffs.shape[-1])
        m_layers = _m_layers.reshape(3, -1, _wdm_coeffs.shape[-1])
        n_layers = np.repeat(n_arr[:, :, None], m_layers.shape[-1], axis=-1)
        gb_fill_wave = xp.zeros((3, self.wdm_set.Nf_active, self.wdm_set.Nt_active))

        keep_m = (m_layers >= self.wdm_set.ind_min_f) & (m_layers <= self.wdm_set.ind_max_f)
        keep_n = (m_layers >= self.wdm_set.ind_min_f) & (m_layers <= self.wdm_set.ind_max_f)
        keep = keep_m & keep_n
        channel_ind = np.repeat(np.arange(3)[:, None],
                                 m_layers.shape[-1] * m_layers.shape[-2],
                                 axis=-1).reshape(m_layers.shape)
        gb_fill_wave[channel_ind[keep], m_layers[keep] - m_min, n_layers[keep] - n_min] = wdm_coeffs[keep]
        return WDMSignal(gb_fill_wave, self.output_set)


gb_gen_wrap = GBLookupWaveWrap(
    t_arr, t_tdi_sparse, Tobs, t_ref, dt, params.shape[0], gb_tdi_kwargs,
    td_set, output_set, window, wdm_lookup_table, wdm_set,
)
analysis = AnalysisContainer(injection, sens_mat, signal_gen=gb_gen_wrap)
wdm_holder = AnalysisContainerArray([analysis])

print("[step] calling gb_comps.fill_global_wdm  <-- C/CUDA path with n_layer", flush=True)
template_fill = xp.zeros(3 * int(np.prod(wdm_set.basis_shape_active)), dtype=float)
t0 = _time.perf_counter()
try:
    # Match the Python wrap and the injection, which both use convert_to_ra_dec=False
    gb_comps.fill_global_wdm(template_fill, params, wdm_holder, data_index=None, convert_to_ra_dec=False)
    print(f"[step] fill_global_wdm OK in {_time.perf_counter()-t0:.2f}s", flush=True)
except Exception as exc:
    print(f"[FAIL] fill_global_wdm threw: {type(exc).__name__}: {exc}", flush=True)
    traceback.print_exc()
    sys.exit(1)

template_fill_wdm = WDMSignal(template_fill.reshape((3,) + wdm_set.basis_shape_active), wdm_set)
print(f"[step] template_fill_wdm shape={template_fill_wdm.arr.shape}, "
      f"|max|={np.abs(template_fill_wdm.arr).max():.3e}, "
      f"sum={template_fill_wdm.arr.sum():.3e}", flush=True)

print("[step] running python-side lookup (gb_gen_wrap)", flush=True)
# Debug instrumentation: tap into the lookup to print f/phi/amp at n=28, channel 0
_orig_get_wdm_coeffs = wdm_lookup_table.get_wdm_coeffs
def _tapped(amp_arr, phi_arr, f_arr, fdot_arr, n_arr, num_m_layers=1):
    # find the index for channel 0 n=28: reshape says (3, Nt_active), n_active=28-ind_min_t=18
    Nt_active_local = wdm_set.Nt_active
    target_idx = 0 * Nt_active_local + (28 - wdm_set.ind_min_t)
    if target_idx < len(amp_arr):
        print(f"[PY-DEBUG] chan=0 n=28  amp={amp_arr[target_idx]:.6e}  phi={phi_arr[target_idx]:.12e}  f={f_arr[target_idx]:.12e}  fdot={fdot_arr[target_idx]:.12e}  n={n_arr[target_idx]}", flush=True)
        ms_at_target = int(f_arr[target_idx] / wdm_lookup_table.layer_df)
        print(f"[PY-DEBUG] layer_m_base={ms_at_target}", flush=True)
    return _orig_get_wdm_coeffs(amp_arr, phi_arr, f_arr, fdot_arr, n_arr, num_m_layers=num_m_layers)
wdm_lookup_table.get_wdm_coeffs = _tapped

t0 = _time.perf_counter()
try:
    py_wdm_lookup = gb_gen_wrap(*params[0])
    print(f"[step] python lookup OK in {_time.perf_counter()-t0:.2f}s, "
          f"|max|={np.abs(py_wdm_lookup.arr).max():.3e}, "
          f"sum={py_wdm_lookup.arr.sum():.3e}", flush=True)
except Exception as exc:
    print(f"[FAIL] gb_gen_wrap threw: {type(exc).__name__}: {exc}", flush=True)
    traceback.print_exc()
    sys.exit(1)

print("[step] computing inner products / likelihood", flush=True)
try:
    check_ip_d_d = analysis.inner_product()
    check_ll_2 = analysis.template_likelihood(template_fill_wdm)
    check_ip_2 = analysis.template_inner_product(template_fill_wdm)
    check_ll_3 = analysis.template_likelihood(py_wdm_lookup)
    tmp_val1 = analysis.template_inner_product(py_wdm_lookup)
    tmp_val2 = analysis.calculate_signal_inner_product(*params[0])
except Exception as exc:
    print(f"[FAIL] inner-product step threw: {type(exc).__name__}: {exc}", flush=True)
    traceback.print_exc()
    sys.exit(1)


print(f"[result] template_likelihood (C lookup)      = {check_ll_2}", flush=True)
print(f"[result] template_likelihood (py lookup)      = {check_ll_3}", flush=True)
print(f"[result] base inner_product                  = {check_ip_d_d}", flush=True)
print(f"[result] template_inner_product (C lookup)   = {check_ip_2}", flush=True)
print(f"[result] template_inner_product (py lookup)  = {tmp_val1}", flush=True)
print(f"[result] calculate_signal_inner_product       = {tmp_val2}", flush=True)

# ====== XYZ → AET conversion (in WDM space) + per-channel inner products ======
print()
print("[step] convert XYZ → AET in WDM space and compute per-channel inner products", flush=True)

def xyz_to_aet(arr):
    X, Y, Z = arr[0], arr[1], arr[2]
    A = (Z - X) / np.sqrt(2.0)
    E = (X - 2.0 * Y + Z) / np.sqrt(6.0)
    T = (X + Y + Z) / np.sqrt(3.0)
    return np.stack([A, E, T], axis=0)

inj_xyz = injection.data_res_arr.arr                 # (3, Nf_a, Nt_a)
c_tpl_xyz = template_fill_wdm.arr                    # C lookup template
py_tpl_xyz = py_wdm_lookup.arr                       # Python lookup template

inj_aet  = xyz_to_aet(inj_xyz)
c_tpl_aet  = xyz_to_aet(c_tpl_xyz)
py_tpl_aet = xyz_to_aet(py_tpl_xyz)

# Use AET sensitivity (diagonal, since AET is orthogonal)
inj_aet_sig = WDMSignal(inj_aet, wdm_set)
c_tpl_sig = WDMSignal(c_tpl_aet, wdm_set)
py_tpl_sig = WDMSignal(py_tpl_aet, wdm_set)
injection_aet_drr = DataResidualArray(inj_aet_sig)
sens_aet = AET1SensitivityMatrix(injection_aet_drr.data_res_arr.settings)
print(f"[step] sens_aet.sens_mat shape={sens_aet.sens_mat.shape}", flush=True)

invC = sens_aet.invC          # (3, Nf_a, Nt_a)
prefactor = 4.0 * sens_aet.differential_component
print()
print(f"==== per-channel AET inner products  (4 * df_pix * dt_pix factor included) ====")
print(f"{'chan':6s} {'<d|d>':>12s} {'<d|h_C>':>12s} {'<d|h_py>':>12s} {'r_C':>9s} {'r_py':>9s}")
for chan, name in enumerate("AET"):
    d_c = inj_aet[chan]
    hC_c = c_tpl_aet[chan]
    hPy_c = py_tpl_aet[chan]
    ic_c = invC[chan]
    dd = (d_c * d_c * ic_c).sum() * prefactor
    dhC = (d_c * hC_c * ic_c).sum() * prefactor
    dhPy = (d_c * hPy_c * ic_c).sum() * prefactor
    r_C  = dhC / dd if dd != 0 else float('nan')
    r_py = dhPy / dd if dd != 0 else float('nan')
    print(f"  {name:4s} {dd:+12.4e} {dhC:+12.4e} {dhPy:+12.4e} {r_C:+9.4f} {r_py:+9.4f}")
print()
print(f"AET totals:  <d|d>={(inj_aet*inj_aet*invC).sum()*prefactor:+.4e}  "
      f"<d|h_C>={(inj_aet*c_tpl_aet*invC).sum()*prefactor:+.4e}  "
      f"<d|h_py>={(inj_aet*py_tpl_aet*invC).sum()*prefactor:+.4e}")

# === Direct per-channel XYZ check: is X swapped with Z in the lookup template? ===
print()
print("==== per-channel XYZ template vs injection (loud-pixel ratios) ====")
print(f"      {'tpl/inj (median)':>20s}  {'sign-corr':>10s}  {'cross-test: tpl[X]/inj[Z]':>28s}  {'tpl[Z]/inj[X]':>16s}")
for chan, name in enumerate("XYZ"):
    d = inj_xyz[chan]; h = py_tpl_xyz[chan]
    if np.abs(d).max() == 0: continue
    mask = np.abs(d) > 0.1 * np.abs(d).max()
    med = np.median(h[mask] / d[mask])
    sg = (d[mask]*h[mask]).sum() / np.abs(d[mask]*h[mask]).sum()
    # cross-check vs swapped indices (only meaningful for X,Z)
    extra = ""
    if chan == 0:
        cross1 = np.median(py_tpl_xyz[0][mask] / inj_xyz[2][mask]) if np.abs(inj_xyz[2][mask]).max()>0 else float('nan')
        cross2 = np.median(py_tpl_xyz[2][mask] / inj_xyz[0][mask]) if np.abs(inj_xyz[0][mask]).max()>0 else float('nan')
        extra = f"{cross1:+.4f}              {cross2:+.4f}"
    print(f"  {name}    {med:+12.4f}      {sg:+10.4f}    {extra}")

print("[step] pixel-by-pixel C vs Python template diff", flush=True)

import matplotlib
matplotlib.use('qtagg')
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = False
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, sharey=True)

template_fill_wdm.heatmap(fig=fig, ax=ax2, index=0, add_cax=True)
injection.data_res_arr.heatmap(fig=fig, ax=ax1, index=0)
py_wdm_lookup.heatmap(fig=fig, ax=ax3, index=0)
plt.show()
plt.close()
breakpoint()

c_arr = template_fill_wdm.arr
py_arr = py_wdm_lookup.arr
print(f"  C   shape={c_arr.shape}  |max|={np.abs(c_arr).max():.3e}  sum={c_arr.sum():.3e}", flush=True)
print(f"  Py  shape={py_arr.shape}  |max|={np.abs(py_arr).max():.3e}  sum={py_arr.sum():.3e}", flush=True)
diff = c_arr - py_arr
print(f"  C - Py: |max|={np.abs(diff).max():.3e}  fro_norm={(diff**2).sum()**0.5:.3e}", flush=True)

# Top-5 loudest pixels of python template, compare values
for chan in range(3):
    flat = np.abs(py_arr[chan]).ravel()
    top_inds = np.argsort(flat)[-5:][::-1]
    print(f"  channel {chan}:", flush=True)
    for fi in top_inds:
        m_idx, n_idx = np.unravel_index(fi, py_arr[chan].shape)
        m_full = m_idx + wdm_set.ind_min_f
        n_full = n_idx + wdm_set.ind_min_t
        c_val = c_arr[chan, m_idx, n_idx]
        py_val = py_arr[chan, m_idx, n_idx]
        ratio = c_val / py_val if py_val != 0 else float('nan')
        print(f"    m_idx={m_idx:3d}(m={m_full}) n_idx={n_idx:3d}(n={n_full}) C={c_val:+.3e} Py={py_val:+.3e} C/Py={ratio:+.4f}", flush=True)

print("[step] direct lookup test: identical (f, phi, amp, n, fdot) into both paths", flush=True)
# Pick representative values near the loud pixel: m=61 (=m_ref), n=28, f at center+small offset
f_test   = 3.0e-3 + 0.1 * wdm_set.layer_df     # a sub-bin offset in the central layer
fdot_test = 0.0
amp_test = 1.0
phi_test = 0.7
n_test   = 28

# Path A: Python get_wdm_coeffs (the linear-spline-with-spacing approach)
amp_arr = np.array([amp_test])
phi_arr = np.array([phi_test])
f_arr   = np.array([f_test])
fdot_arr = np.array([fdot_test])
n_arr_pt = np.array([n_test])
wdm_lookup_table.get_wdm_coeffs = _orig_get_wdm_coeffs   # untap
wdm_coeffs_py, m_map_py = wdm_lookup_table.get_wdm_coeffs(
    amp_arr, phi_arr, f_arr, fdot_arr, n_arr_pt, num_m_layers=2
)
print(f"[DIRECT-PY] wdm_coeffs={wdm_coeffs_py[0]}", flush=True)
print(f"[DIRECT-PY] m_map     ={m_map_py[0]}", flush=True)

# Path B: Naive python that reads the raw table[n, 0, :] and 1D-interps along f
table_sin_raw = wdm_lookup_table.table_sin   # (Nt, fdot_steps, f_steps)
table_cos_raw = wdm_lookup_table.table_cos
f_vals_norm = wdm_lookup_table.f_vals_norm.get() if hasattr(wdm_lookup_table.f_vals_norm, 'get') else wdm_lookup_table.f_vals_norm
print(f"[DIRECT-NAIVE] f_vals_norm.min={f_vals_norm.min():.6e}, max={f_vals_norm.max():.6e}, n={f_vals_norm.size}", flush=True)

m_center = int(f_test / wdm_lookup_table.layer_df)
for diff in range(-2, 3):
    m_use = m_center + diff
    f_norm = f_test - m_use * wdm_lookup_table.layer_df
    # naive: pull out n-slice of the table and use np.interp
    sin_slice = table_sin_raw[n_test, 0, :]
    cos_slice = table_cos_raw[n_test, 0, :]
    _s_naive = np.interp(f_norm, f_vals_norm, sin_slice)
    _c_naive = np.interp(f_norm, f_vals_norm, cos_slice)
    # apply parity
    is_mn_even = ((m_use + n_test) % 2) == 0
    is_m_even = (m_use % 2) == 0
    if (not is_mn_even) and is_m_even:
        sc, cc = _s_naive, -_c_naive
    elif (not is_mn_even) and (not is_m_even):
        sc, cc = _s_naive, _c_naive
    elif is_mn_even and is_m_even:
        sc, cc = _c_naive, _s_naive
    else:
        sc, cc = -_c_naive, _s_naive
    coeff_naive = amp_test * (sc * np.sin(phi_test) + cc * np.cos(phi_test))
    coeff_py    = wdm_coeffs_py[0, diff + 2]
    print(f"[CMP] diff={diff:+d}  m_use={m_use}  f_norm={f_norm:+.4e}  naive={coeff_naive:+.6e}  py_iface={coeff_py:+.6e}  rel_diff={(coeff_naive-coeff_py)/max(abs(coeff_naive),1e-30):+.3e}", flush=True)

# Path C: the C kernel doesn't have a "single query" entry point we can call from Python,
# but we can drive the C side by hand-constructing a single-pixel injection equivalent.
# Instead, we replicate the C interp logic directly here (bilinear in (fdot, f), n-indexed).
def c_style_lookup(f, fdot, m, n, table):
    # offset by layer_n
    z_slice = table[n]  # shape (fdot_steps, f_steps)
    df_interp = float(f_vals_norm[1] - f_vals_norm[0])
    min_f_scaled = float(f_vals_norm.min())
    f_scaled = f - m * wdm_lookup_table.layer_df
    f_index = int((f_scaled - min_f_scaled) / df_interp)
    if f_index < 0 or f_index >= len(f_vals_norm) - 1:
        return 0.0
    if z_slice.shape[0] > 1:
        raise NotImplementedError("only no-fdot path")
    # 1D linear
    x1 = df_interp * f_index + min_f_scaled
    x2 = df_interp * (f_index + 1) + min_f_scaled
    z1 = z_slice[0, f_index]
    z2 = z_slice[0, f_index + 1]
    return z1 + (f_scaled - x1) * (z2 - z1) / (x2 - x1)

print("[DIRECT-CSTYLE]", flush=True)
for diff in range(-2, 3):
    m_use = m_center + diff
    _s = c_style_lookup(f_test, fdot_test, m_use, n_test, table_sin_raw)
    _c = c_style_lookup(f_test, fdot_test, m_use, n_test, table_cos_raw)
    is_mn_even = ((m_use + n_test) % 2) == 0
    is_m_even = (m_use % 2) == 0
    if (not is_mn_even) and is_m_even:
        sc, cc = _s, -_c
    elif (not is_mn_even) and (not is_m_even):
        sc, cc = _s, _c
    elif is_mn_even and is_m_even:
        sc, cc = _c, _s
    else:
        sc, cc = -_c, _s
    # C-style w_mn computation: cos_coeff * |M|*cos(arg(M)+pi/2) + sin_coeff * |M|*sin(arg(M)+pi/2)
    # equivalently (with arg(M) = -phi_test+pi/2 so that cos/sin matches python convention):
    # We use the python formula here for direct comparison
    coeff = amp_test * (sc * np.sin(phi_test) + cc * np.cos(phi_test))
    coeff_py = wdm_coeffs_py[0, diff + 2]
    print(f"[CMP-C] diff={diff:+d}  m_use={m_use}  f_norm={f_test-m_use*wdm_lookup_table.layer_df:+.4e}  c_style={coeff:+.6e}  py_iface={coeff_py:+.6e}", flush=True)

# Path D: use the *real* C wrapper exposed in lisa-on-gpu (binding_tof) — drives the
# C code path through pybind11 directly. This is the one-stop check.
print("[DIRECT-CWRAP] calling C-binding get_w_mn_arr", flush=True)
cpp_lookup = gb_comps.cpp_wdm_lookup_table  # WaveletLookupTableWrap*
N_cmp = 5
out_arr  = np.zeros(N_cmp, dtype=float)
amp_in   = np.full(N_cmp, amp_test, dtype=float)
phi_in   = np.full(N_cmp, phi_test, dtype=float)
f_in     = np.full(N_cmp, f_test,  dtype=float)
fdot_in  = np.full(N_cmp, fdot_test, dtype=float)
m_in     = np.array([m_center + d for d in range(-2, 3)], dtype=np.int32)
n_in     = np.full(N_cmp, n_test, dtype=np.int32)
cpp_lookup.get_w_mn_arr(out_arr, amp_in, phi_in, f_in, fdot_in, m_in, n_in, N_cmp)
for i, d in enumerate(range(-2, 3)):
    coeff_py_i = wdm_coeffs_py[0, d + 2]
    print(f"[CMP-CWRAP] diff={d:+d}  m_use={m_center+d}  c_wrap={out_arr[i]:+.6e}  py_iface={coeff_py_i:+.6e}  ratio={(out_arr[i]/coeff_py_i if coeff_py_i!=0 else float('nan')):+.6f}", flush=True)

# Cross-feed: take Python wrap's actual (f, phi, amp) at n=28 chan=0 and pass them
# into the C-binding wrapper. If C-wrap output here matches the Python lookup's pixel
# value, then the formula is consistent end-to-end and the C *kernel*'s mismatch is
# purely from its own f/phi/amp computation (numerical-derivative spline divergence).
print("[step] cross-feed: py (f, phi, amp) into C wrapper", flush=True)
# Run the wrap once and capture its internal values
wave_tmp2 = gb_gen_wrap.gb_gen(*params.T, convert_to_ra_dec=False, return_spline=True)
t_arr_wrap = wdm_set.t_arr + t_ref
f_dtdi = wave_tmp2.tdi_phase_spl(np.tile(t_arr_wrap, (1, 3, 1)), derivative=1)[0] / (2 * np.pi)
f_dref = wave_tmp2.phase_ref_spl(t_arr_wrap[None, :], derivative=1)[0] / (2 * np.pi)
f_full = f_dref + f_dtdi
tdi_amp_py  = wave_tmp2.tdi_amp_spl(np.tile(t_arr_wrap, (1, 3, 1)))[0]
tdi_phase_py = wave_tmp2.tdi_phase_spl(np.tile(t_arr_wrap, (1, 3, 1)))[0]
ref_phase_py = wave_tmp2.phase_ref_spl(t_arr_wrap[None, :])[0]
phi_py_full = (tdi_phase_py + ref_phase_py) + np.pi / 2.0

# pick chan=0 n=28; n_active = 28 - ind_min_t
n_active = 28 - wdm_set.ind_min_t
chan = 0
amp_used  = tdi_amp_py[chan, n_active]
phi_used  = phi_py_full[chan, n_active]
f_used    = f_full[chan, n_active]
fdot_used = 0.0
print(f"[XFEED] using py (chan=0, n=28): amp={amp_used:.6e} phi={phi_used:.6e} f={f_used:.6e}", flush=True)

m_center = int(f_used / wdm_set.layer_df)
m_arr_xf  = np.array([m_center + d for d in range(-2, 3)], dtype=np.int32)
n_arr_xf  = np.full(5, 28, dtype=np.int32)
amp_xf    = np.full(5, amp_used, dtype=float)
phi_xf    = np.full(5, phi_used, dtype=float)
f_xf      = np.full(5, f_used, dtype=float)
fdot_xf   = np.full(5, fdot_used, dtype=float)
out_xf    = np.zeros(5, dtype=float)
cpp_lookup.get_w_mn_arr(out_xf, amp_xf, phi_xf, f_xf, fdot_xf, m_arr_xf, n_arr_xf, 5)
# Compare to the python lookup's actual pixel values at chan=0, n=28
py_actual = py_wdm_lookup.arr[0, m_center - wdm_set.ind_min_f, 28 - wdm_set.ind_min_t]
print(f"[XFEED] py_lookup pixel value at (chan=0, m={m_center}, n=28) = {py_actual:+.6e}", flush=True)
for i, d in enumerate(range(-2, 3)):
    print(f"[XFEED-CMP] diff={d:+d}  m={m_center+d}  c_wrap_with_py_inputs={out_xf[i]:+.6e}", flush=True)

print("[step] DONE", flush=True)
