#!/usr/bin/env python
"""Sig-het vs chunked-het AT THE OFFENDER COORDINATES on the REAL mojito galaxy.

The synthetic in-vitro ladder (gb_sighet_invitro_probe.py) exonerated every
axis reachable with one clean source on CPU: grid, epoch, caches, n_pad,
foreground level, XYZ CSD, the narrow-slab path, a two-source batch, and the
chunked exact side itself (bracket == 1.0000). The production dissect
meanwhile shows hh_sig/hh_chunk ~ 4-6 and eps0 ~ 160-200 for band-10 sources
(1.95-2.08 mHz). This probe carries every remaining non-GPU ingredient at
once, then lets env knobs remove them one at a time:

  * the REAL mojito GB galaxy stream (full confusion) as the slab content --
    tdis/X2,Y2,Z2 of the local L1 file, whose t0 = 97729939.827664 IS the
    production engine's _g["t0"];
  * MANY references batched into one setup_in_model (production blocks run
    thousands; the crowding gradient is 14%->38% broken);
  * catalogue parameters in the offender band, and PERTURBED ("hot walker")
    variants of them;
  * optionally a TIME-MODULATED invC (the cyclostationary foreground --
    strong at low f, absent above ~5 mHz: the measured curve's own shape,
    and the one invC property that does NOT cancel between the two engines'
    different time aggregations).

Scoring: identical holder, identical params -- sig-het (setup_in_model +
get_ll at the anchor) vs chunked task-b (the trusted exact side). Reported
per source: eps0 = |ll_sig - ll_chunk| and hh_sig/hh_chunk, i.e. exactly the
production dissect quantities. Reproduction target for BAND=10:
med eps0 ~ 160-200, med hh ratio ~ 2-6.

Run (CPU, pinned):
  OMP_NUM_THREADS=1 python scripts/gb_chunked_het/gb_sighet_realdata_probe.py
Env: BAND=10 (GB band index; band k spans [5.5556e-4 + k*layer_df, +1]),
     NREF=16, MODE=catalogue|perturbed, TMOD=0|1, BATCH=1 (0 = one ref per
     setup call), SYNTH=1 replaces the real data slab with the refs' own
     templates (the clean-slab control).
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import h5py

from lisatools import detector as lisa_models
from lisatools.detector import DefaultOrbits
from lisatools.sensitivity import get_sensitivity, A2TDISens
from lisatools.stochastic import HyperbolicTangentGalacticForeground as HTGF
from lisatools.domains import TDSettings, TDSignal, WDMSettings

from gbgpu.gbcomps import GBWDMComputations
from gbgpu.gbsignalhetcomputations import GBSignalHetComputations
from gbgpu.gb_likelihood import make_band_likelihood_engine

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gb_sighet_invitro_probe import Holder, PSD_P, GAL_P, GB_MOJITO_T_REF

# local layout; on the cluster: MOJITO_DIR=/shared/home/mlkatz1/mojito_cache
MOJ = os.environ.get(
    "MOJITO_DIR",
    os.path.expanduser("~/.mojito_cache/brickmarket/mojito_light_v1_0_0"))
YR = 31558149.763545603


def load_real_wdm(backend="cpu"):
    """First 3 months of the real GB galaxy stream, on the production grid."""
    Nf, Nt, dt = 1440, 2160, 2.5
    N = Nf * Nt
    fp = sorted(glob.glob(f"{MOJ}/data/GB/L1/*.h5"))[0]
    with h5py.File(fp, "r") as f:
        t0 = float(f["tdis"]["sampling"].attrs["t0"])
        td = np.stack([np.asarray(f["tdis"][k][:N], dtype=float)
                       for k in ("X2", "Y2", "Z2")])
    wavelet_duration = Nf * dt
    wdm = WDMSettings(Nf, Nt, dt, t0=t0,
                      min_freq=1e-4, max_freq=2.5e-2,
                      min_time=20 * wavelet_duration,
                      max_time=(Nt - 20) * wavelet_duration,
                      force_backend=backend)
    td_set = TDSettings(N, dt, force_backend=backend)
    sig = TDSignal(td, settings=td_set).transform(wdm, window=None)
    arr = np.asarray(sig.data_arr if hasattr(sig, "data_arr") else sig[:])
    return wdm, arr, t0


def main():
    backend = os.environ.get("PROBE_BACKEND", "cpu")
    band = int(os.environ.get("BAND", "10"))
    nref = int(os.environ.get("NREF", "16"))
    mode = os.environ.get("MODE", "catalogue")
    tmod = os.environ.get("TMOD", "0") == "1"
    batch = os.environ.get("BATCH", "1") == "1"
    synth = os.environ.get("SYNTH", "0") == "1"

    wdm, data_act, t0 = load_real_wdm(backend)
    layer_df = wdm.layer_df
    F = wdm.ind_max_f - wdm.ind_min_f + 1
    T = wdm.ind_max_t - wdm.ind_min_t + 1
    if data_act.shape != (3, F, T):
        # transform returns the ACTIVE grid signal; reconcile defensively
        data_act = np.asarray(data_act).reshape(3, F, T)
    band_lo = 5.555556e-4 + band * layer_df
    band_hi = band_lo + layer_df
    print(f"[real] data active {data_act.shape}, t0={t0:.6f}, band {band} = "
          f"[{band_lo*1e3:.4f}, {band_hi*1e3:.4f}] mHz, mode={mode}, "
          f"tmod={tmod}, batch={batch}, synth={synth}", flush=True)

    # ---- catalogue refs in the band --------------------------------------
    from lisatools.globalfit.recipe import gb_catalogue_to_sampling_basis
    with h5py.File(f"{MOJ}/catalogues/wdwd_cat_mojito_lite_processed.hdf5",
                   "r") as f:
        b = f["Binaries"]
        f0c = np.asarray(b["GW22FrequencySSBFrame"][:], float)
        sel = np.where((f0c >= band_lo) & (f0c <= band_hi))[0]
        amp = np.asarray(b["Amplitude"][sel])
        order = sel[np.argsort(amp)[::-1][:nref]]
        order = np.sort(order)
        entry = {k: np.asarray(b[k][order]) for k in (
            "Amplitude", "GW22FrequencySSBFrame",
            "GW22FrequencyDerivativeSourceFrame", "InclinationAngle",
            "PolarisationAngle", "RightAscension", "Declination",
            "TrueAnomaly", "TimeReferenceSSBFrame", "ChirpMassSSBFrame",
            "LuminosityDistance", "Eccentricity")}
    rows = np.atleast_2d(gb_catalogue_to_sampling_basis(entry))
    params = np.column_stack([
        np.exp(rows[:, 0]), rows[:, 1] * 1e-3, rows[:, 2],
        np.zeros(len(rows)), rows[:, 3], np.arccos(rows[:, 4]),
        rows[:, 5], rows[:, 6], np.arcsin(np.clip(rows[:, 7], -1, 1))])
    n = len(params)
    print(f"[real] {n} catalogue refs, amp {params[:,0].min():.2e}.."
          f"{params[:,0].max():.2e}", flush=True)
    if mode == "perturbed":
        # "hot walker" coordinates: same band, scrambled everything else
        rng = np.random.default_rng(1)
        params[:, 0] *= rng.uniform(0.3, 3.0, n)
        params[:, 1] = rng.uniform(band_lo, band_hi, n)
        params[:, 4] = rng.uniform(0, 2 * np.pi, n)
        params[:, 5] = np.arccos(rng.uniform(-1, 1, n))
        params[:, 6] = rng.uniform(0, np.pi, n)

    # ---- chunked comp (production knobs) + invC --------------------------
    orbits = DefaultOrbits(force_backend=backend, frame="icrs")
    chunked = GBWDMComputations(
        wdm, t_ref=GB_MOJITO_T_REF, Nt_sub=256, n_pad=8, N_sparse=256,
        N_cp_sig=48, N_cp_orbit=32, orbits=orbits,
        tdi_config="2nd generation", force_backend=backend,
        d_d=0.0, tdi_type="XYZ")
    xp = chunked.xp

    model = lisa_models.LISAModel(PSD_P[0] ** 2, PSD_P[1] ** 2,
                                  lisa_models.DefaultOrbits(), "probe")
    kw_fg = dict(model=model, stochastic_params=tuple(GAL_P),
                 stochastic_function=HTGF)
    kw_in = dict(model=model)

    W = 5
    m_carrier = np.floor(params[:, 1] / layer_df).astype(int) - wdm.ind_min_f
    slab_lo = np.clip(m_carrier - W // 2, 0, F - W).astype(np.int32)

    data = np.zeros((n, 3, W, T))
    invc = np.zeros((n, 3, 3, W, T))
    t_rel = np.arange(T) * wdm.layer_dt
    for i in range(n):
        lo = slab_lo[i]
        data[i] = data_act[:, lo: lo + W, :]
        m_abs = np.arange(lo, lo + W) + wdm.ind_min_f
        f_rows = np.maximum(m_abs * layer_df, 1e-5)
        Sfg = np.asarray(get_sensitivity(f_rows, sens_fn=A2TDISens, **kw_fg),
                         float)
        for c in range(3):
            invc[i, c, c] = (1.0 / Sfg)[:, None]
        if tmod:
            # cyclostationary foreground: modulate the FOREGROUND SHARE of
            # the noise over the year (2 cycles/yr, galactic-plane sweep)
            Sin = np.asarray(get_sensitivity(f_rows, sens_fn=A2TDISens,
                                             **kw_in), float)
            share = np.clip(1.0 - Sin / Sfg, 0.0, 0.95)          # (W,)
            mod = 1.0 + 0.6 * share[:, None] * np.cos(
                4 * np.pi * (t0 + t_rel[None, :]) / YR + 1.0)
            for c in range(3):
                invc[i, c, c] = invc[i, c, c] * mod
    if synth:
        data[:] = 0.0
        for i in range(n):
            full = xp.zeros((3, wdm.Nf, wdm.Nt), dtype=xp.float64)
            chunked.fill_global_wdm(params[i].reshape(1, 9), full,
                                    convert_to_ra_dec=False, factors=None)
            act = np.asarray(full[:, wdm.ind_min_f: wdm.ind_max_f + 1,
                                  wdm.ind_min_t: wdm.ind_max_t + 1])
            data[i] = act[:, slab_lo[i]: slab_lo[i] + W, :]

    holder = Holder(data, invc, xp, slab_min_f=slab_lo + wdm.ind_min_f,
                    band_slab_Nf=W)

    sig = GBSignalHetComputations.for_band_engine(
        chunked, nt_layer=60, n_sparse_fd=1024, m_active_half_width=2,
        max_r=0.0, n_cp_build=32, v3_n_nodes=64, v4_knots=128, v4_band=16,
        v5=1)
    eng = make_band_likelihood_engine(
        wdm, gb_wdm_comp=sig, nchannels=3, tdi_channel_setup="XYZ")

    def _np(a):
        return np.asarray(a.get() if hasattr(a, "get") else a)

    idx_all = np.arange(n)
    ll_s = np.zeros(n); hh_s = np.zeros(n)
    groups = [idx_all] if batch else [np.array([i]) for i in idx_all]
    for gidx in groups:
        eng.setup_in_model(holder, params[gidx], gidx)
        out = eng.get_ll(holder, params[gidx], data_index=gidx,
                         noise_index=gidx, N_vals=np.full(gidx.size, 1024),
                         waveform_kwargs={})
        ll_s[gidx] = _np(out).ravel()[:gidx.size]
        hh_s[gidx] = _np(eng.h_h_out).real.ravel()[:gidx.size]
        eng.clear_in_model()

    out = eng.get_ll(holder, params, data_index=idx_all, noise_index=idx_all,
                     N_vals=np.full(n, 1024), waveform_kwargs={})
    ll_c = _np(out).ravel()[:n]
    hh_c = _np(eng.h_h_out).real.ravel()[:n]

    eps0 = np.abs(ll_s - ll_c)
    ratio = hh_s / np.maximum(hh_c, 1e-300)
    print(f"\n  {'f0 mHz':>9} {'snr_c':>7} {'ll_chunk':>10} {'ll_sig':>10} "
          f"{'eps0':>9} {'hh_s/hh_c':>10}")
    for i in np.argsort(eps0)[::-1]:
        print(f"  {params[i,1]*1e3:9.4f} {np.sqrt(max(hh_c[i],0)):7.1f} "
              f"{ll_c[i]:10.2f} {ll_s[i]:10.2f} {eps0[i]:9.3g} "
              f"{ratio[i]:10.4f}")
    print(f"\n  eps0 med={np.median(eps0):.3g} max={eps0.max():.3g} | "
          f"hh ratio med={np.median(ratio):.4f} max={ratio.max():.4f}")
    print("  production dissect target for band 10: eps0 med ~ 160-200, "
          "hh ratio med ~ 2-6. Reproduce -> bisect with SYNTH=1 (clean "
          "slab), BATCH=0 (solo), MODE=catalogue, TMOD=0.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
