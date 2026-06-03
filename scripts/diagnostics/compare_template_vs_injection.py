#!/usr/bin/env python
"""Compare the lookup-table template (Python wrap & C kernel — they now agree)
against the WDM-transformed injection. Plot/inspect the residual to expose any
every-other-m parity pattern."""

import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal

import fastlisaresponse
from lisatools.detector import ESAOrbits
from lisatools.utils.constants import YRSID_SI
from fastlisaresponse.tdiconfig import TDIConfig
from fastlisaresponse.tdionfly import GBTDIonTheFly
from fastlisaresponse.gbcomps import GBWDMComputations

from lisatools.datacontainer import DataResidualArray
from lisatools.analysiscontainer import AnalysisContainer, AnalysisContainerArray
from lisatools.sensitivity import XYZ2SensitivityMatrix
from lisatools.domains import (TDSettings, TDSignal, FDSettings, WDMSettings, WDMSignal, WDMLookupTable)

backend = "cpu"
xp = np
orbits = ESAOrbits(force_backend=backend)
dt = 10.0
Nt = 64; Nf = 1024
wavelet_duration = Nf * dt
Tobs = Nt * wavelet_duration
Nobs = Nf * Nt
tdi_config = TDIConfig("2nd generation")
t_start = int(0.25 * YRSID_SI / dt) * dt
t_ref = t_start
t_arr = np.arange(Nobs) * dt + t_start
data_inj = np.zeros((3, t_arr.shape[-1]))

gb_tdi_kwargs = dict(tdi_config=tdi_config, orbits=orbits, tdi_chan="XYZ", force_backend=backend)

amp = np.full(1, 8.0e-23); f0 = np.full(1, 3.0e-3); fdot = np.full(1, 1e-16); fddot = np.full(1, 0.0)
phi0 = np.full(1, 2.09802430298); inc = np.full(1, 0.23984234); psi = np.full(1, 1.234019814)
lam = np.full(1, 4.09808143); beta = np.full(1, 1.1)
params = np.array([amp, f0, fdot, fddot, phi0, inc, psi, lam, beta]).T

N = data_inj.shape[-1]
td_set = TDSettings(N, dt, force_backend=backend)
window = xp.asarray(signal.windows.tukey(N, alpha=0.05))

layer_df_naive = 1.0 / wavelet_duration
m_ref_central = int(round(f0[0] / layer_df_naive))
min_freq = (m_ref_central - 5) * layer_df_naive
max_freq = (m_ref_central + 5) * layer_df_naive
min_time = 10 * wavelet_duration
max_time = (Nt - 10) * wavelet_duration

wdm_set = WDMSettings(Nf, Nt, dt, min_freq=min_freq, max_freq=max_freq,
                     min_time=min_time, max_time=max_time)
print(f"wdm_set.layer_df={wdm_set.layer_df:.6e}, ind_min_f={wdm_set.ind_min_f}, ind_max_f={wdm_set.ind_max_f}")
print(f"          layer_dt={wdm_set.layer_dt}, ind_min_t={wdm_set.ind_min_t}, ind_max_t={wdm_set.ind_max_t}")

# === Injection ===
N_inj = 16384
t_tdi_inj = xp.linspace(t_arr[0], t_arr[-1], N_inj)
gb_gen_inj = GBTDIonTheFly(t_tdi_inj, Tobs, t_ref, 1.0 / dt, 1, **gb_tdi_kwargs)
inj_tmp = gb_gen_inj(*params.T, convert_to_ra_dec=False, return_spline=True)
data_inj[:] = inj_tmp.eval_tdi(t_arr)
data_inj_wdm = TDSignal(data_inj, settings=td_set).transform(wdm_set, window=window)
inj_arr = data_inj_wdm.arr   # (3, Nf_active, Nt_active)
print(f"injection WDM shape={inj_arr.shape}, |max|={np.abs(inj_arr).max():.3e}")

injection = DataResidualArray(data_inj_wdm)
sens_mat = XYZ2SensitivityMatrix(injection.data_res_arr.settings, model="scirdv1")

# === Lookup table + Python wrap → template ===
store_path = "wdm_lookup_smoke_test.h5"
if os.path.exists(store_path):
    wdm_lookup_table = WDMLookupTable.from_file(store_path)
else:
    raise RuntimeError("Need to run test_runner.py first to build the table file.")

# Python wrap (mirrors GBLookupWaveWrap from test_runner)
N_sparse = 2048
t_tdi_sparse = xp.linspace(t_arr[0], t_arr[-1], N_sparse)
gb_gen_py = GBTDIonTheFly(t_tdi_sparse, Tobs, t_ref, 1.0 / dt, 1, **gb_tdi_kwargs)
wave_tmp = gb_gen_py(*params.T, convert_to_ra_dec=False, return_spline=True)
t_active = wdm_set.t_arr + t_ref
f_deriv_tdi = wave_tmp.tdi_phase_spl(np.tile(t_active, (1, 3, 1)), derivative=1)[0] / (2 * np.pi)
f_deriv_ref = wave_tmp.phase_ref_spl(t_active[None, :], derivative=1)[0] / (2 * np.pi)
f_deriv = f_deriv_ref + f_deriv_tdi
tdi_amp = wave_tmp.tdi_amp_spl(np.tile(t_active, (1, 3, 1)))[0]
tdi_phase = wave_tmp.tdi_phase_spl(np.tile(t_active, (1, 3, 1)))[0]
ref_phase = wave_tmp.phase_ref_spl(t_active[None, :])[0]
phi_t = ((tdi_phase + ref_phase) + np.pi / 2.0).flatten().copy()
freq_t = f_deriv.flatten().copy()
fdot_t = np.full_like(freq_t, 0.0)
amp_t = tdi_amp.flatten().copy()

n_arr = np.tile(xp.arange(wdm_set.Nt)[wdm_set.active_slice_t], (3, 1))
n_arr_in = n_arr.flatten().copy()
n_min = wdm_set.ind_min_t
m_min = wdm_set.ind_min_f

_wdm_coeffs, _m_layers = wdm_lookup_table.get_wdm_coeffs(amp_t, phi_t, freq_t, fdot_t, n_arr_in, num_m_layers=2)
wdm_coeffs = _wdm_coeffs.reshape(3, -1, _wdm_coeffs.shape[-1])
m_layers = _m_layers.reshape(3, -1, _wdm_coeffs.shape[-1])
n_layers = np.repeat(n_arr[:, :, None], m_layers.shape[-1], axis=-1)
gb_fill_wave = xp.zeros((3, wdm_set.Nf_active, wdm_set.Nt_active))
keep_m = (m_layers >= wdm_set.ind_min_f) & (m_layers <= wdm_set.ind_max_f)
keep_n = (m_layers >= wdm_set.ind_min_f) & (m_layers <= wdm_set.ind_max_f)
keep = keep_m & keep_n
channel_ind = np.repeat(np.arange(3)[:, None], m_layers.shape[-1] * m_layers.shape[-2], axis=-1).reshape(m_layers.shape)
gb_fill_wave[channel_ind[keep], m_layers[keep] - m_min, n_layers[keep] - n_min] = wdm_coeffs[keep]
tpl_arr = gb_fill_wave   # (3, Nf_active, Nt_active)
print(f"template (python lookup) shape={tpl_arr.shape}, |max|={np.abs(tpl_arr).max():.3e}")

# === Diagnostics ===
# Per-m diagnostic of how injection vs template compare
print()
print("==== |inj|.sum() and |tpl|.sum() per m (channel 0) ====")
print("m_full    |inj|.sum()   |tpl|.sum()    sgn-corr   ratio   m%2")
chan = 0
for m_idx in range(wdm_set.Nf_active):
    m_full = m_idx + wdm_set.ind_min_f
    inj_col = inj_arr[chan, m_idx, :]
    tpl_col = tpl_arr[chan, m_idx, :]
    inj_abs_sum = np.abs(inj_col).sum()
    tpl_abs_sum = np.abs(tpl_col).sum()
    sgn_corr = (inj_col * tpl_col).sum() / max(np.abs(inj_col*tpl_col).sum(), 1e-300)
    ratio = tpl_abs_sum / max(inj_abs_sum, 1e-300)
    print(f"  {m_full:3d}     {inj_abs_sum:.3e}   {tpl_abs_sum:.3e}    {sgn_corr:+.4f}   {ratio:.4f}   {m_full%2}")

# Same for ratio computed pixel-by-pixel
print()
print("==== inj vs tpl at the loud pixels of each loud m (channel 0) ====")
for m_full_query in [60, 61, 62]:
    mi = m_full_query - wdm_set.ind_min_f
    print(f"  --- m={m_full_query} (m%2={m_full_query%2}) ---")
    print("  n_full   inj          tpl          tpl/inj    (m+n)%2")
    # sort by |inj| at this row
    abs_inj_row = np.abs(inj_arr[chan, mi, :])
    top_n = np.argsort(abs_inj_row)[-10:][::-1]
    for ni in top_n:
        n_full = ni + wdm_set.ind_min_t
        iv = inj_arr[chan, mi, ni]
        tv = tpl_arr[chan, mi, ni]
        r = tv / iv if iv != 0 else float('nan')
        print(f"    {n_full:3d}    {iv:+.3e}   {tv:+.3e}   {r:+.4f}     {(m_full_query+n_full)%2}")

# === Pixel-by-pixel decomposition of <d|h> with the actual sensitivity weighting ===
# `sens_mat.invC` has shape (3, 3, Nf_active, Nt_active) — inverse-covariance per pixel.
# Inner product is sum over (i, j) of d_i * h_j * invC[i,j] * differential
invC = sens_mat.invC
diff = sens_mat.differential_component
# contrib per (i, j, m, n)
contrib_ij = (inj_arr[:, None, :, :] * tpl_arr[None, :, :, :]) * invC * diff
contrib_dd_ij = (inj_arr[:, None, :, :] * inj_arr[None, :, :, :]) * invC * diff
# sum the channel matrix to per-pixel contributions
contrib   = contrib_ij.sum(axis=(0, 1))      # (Nf_active, Nt_active)
contrib_dd = contrib_dd_ij.sum(axis=(0, 1))
print()
print("==== <d|h> pixel-by-pixel decomposition (full XYZ inverse covariance) ====")
print(f"sum contrib_dh = {contrib.sum():+.6e}")
print(f"sum contrib_dd = {contrib_dd.sum():+.6e}")
# Per-(i,j) totals to find which channel pair flips sign
for i in range(3):
    for j in range(3):
        sij_dh = contrib_ij[i, j].sum()
        sij_dd = contrib_dd_ij[i, j].sum()
        print(f"  (i={i},j={j}): dh={sij_dh:+.4e}  dd={sij_dd:+.4e}  ratio={sij_dh/sij_dd if sij_dd!=0 else float('nan'):+.4f}")
print()
print("==== top 10 NEGATIVE-contribution pixels to <d|h> (summed over i,j) ====")
flat_c = contrib.ravel()
idx_sort = np.argsort(flat_c)
print("m_full n_full   contribution_dh   contribution_dd   ratio")
for k in idx_sort[:10]:
    mi, ni = np.unravel_index(k, contrib.shape)
    m_full = mi + wdm_set.ind_min_f
    n_full = ni + wdm_set.ind_min_t
    print(f"  {m_full:3d}    {n_full:3d}    {contrib[mi,ni]:+.3e}      {contrib_dd[mi,ni]:+.3e}      {contrib[mi,ni]/max(abs(contrib_dd[mi,ni]),1e-300):+.4f}")
print()
print("==== top 10 POSITIVE-contribution pixels to <d|h> ====")
for k in idx_sort[-10:][::-1]:
    mi, ni = np.unravel_index(k, contrib.shape)
    m_full = mi + wdm_set.ind_min_f
    n_full = ni + wdm_set.ind_min_t
    print(f"  {m_full:3d}    {n_full:3d}    {contrib[mi,ni]:+.3e}      {contrib_dd[mi,ni]:+.3e}      {contrib[mi,ni]/max(abs(contrib_dd[mi,ni]),1e-300):+.4f}")

# Heatmaps
fig, axes = plt.subplots(3, 3, figsize=(15, 9), sharex=True, sharey=True)
for chan in range(3):
    extent = [wdm_set.ind_min_t, wdm_set.ind_max_t+1, wdm_set.ind_min_f, wdm_set.ind_max_f+1]
    vmax = max(np.abs(inj_arr[chan]).max(), np.abs(tpl_arr[chan]).max())
    axes[chan, 0].imshow(inj_arr[chan], aspect="auto", origin="lower", extent=extent, vmin=-vmax, vmax=vmax, cmap="RdBu_r")
    axes[chan, 0].set_title(f"inj chan={chan}")
    axes[chan, 1].imshow(tpl_arr[chan], aspect="auto", origin="lower", extent=extent, vmin=-vmax, vmax=vmax, cmap="RdBu_r")
    axes[chan, 1].set_title(f"tpl chan={chan}")
    axes[chan, 2].imshow(tpl_arr[chan] - inj_arr[chan], aspect="auto", origin="lower", extent=extent, vmin=-vmax, vmax=vmax, cmap="RdBu_r")
    axes[chan, 2].set_title(f"tpl - inj chan={chan}")
    axes[chan, 0].set_ylabel("m")
plt.tight_layout()
plt.savefig("/tmp/template_vs_injection.png", dpi=120)
print("\nsaved /tmp/template_vs_injection.png")
