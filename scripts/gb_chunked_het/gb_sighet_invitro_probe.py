#!/usr/bin/env python
"""In-vitro probe of the sig-het reference h_h against the direct pixel sum.

WHY (2026-08-19). The production dissect measured a smooth, population-wide
multiplicative bias in the sig-het reference template power:

    hh_sighet / hh_exact ~ 6.4 @ 1 mHz -> 1.0 @ 7-10 mHz -> 0.984 @ 10-22 mHz

null-independent, crowding-enhanced-but-present-isolated, flat in
layer-position, and UNMOVED by every engine knob (nt_layer, n_r nodes, knots,
v4/v5, m_half, n_sparse_fd -- the in-run sweep, LAT eea2d0b4). Meanwhile the
validated harness (``gb_chunked_prior_draws.py``) shows chunked-het matching
the dense WDM transform to fp precision, and past sig-het A/Bs against it
passed. So the bug lives in what the FIT does differently from the TESTS.

THIS probe reproduces the discrepancy in vitro -- one source, no sampler, no
cluster -- by walking the validated test construction toward the production
configuration ONE AXIS AT A TIME:

    config 0  TEST      dt=10 grid, epoch 0.5 yr, ESAOrbits, N_cp caches OFF,
                        n_pad=32, instrument-only invC   (must give ratio ~1,
                        or the probe itself is wrong)
    config 1  +GRID     production WDM grid (Nf=1440, Nt=2160, dt=2.5, band
                        [1e-4, 2.5e-2] Hz, time crop 20 layers each side)
    config 2  +EPOCH    production epoch (t_start=9.772994e7 s, t_ref =
                        GB_MOJITO_T_REF = 97729089.327664)
    config 3  +CACHES   N_cp_sig=48, N_cp_orbit=32 (tests run 0/0 = direct)
    config 4  +NPAD     n_pad=8 (production) vs 32 (test recommendation)
    config 5  +FGINVC   invC includes the fitted galactic foreground

For each config x f0 in F0_LIST:
  1. build the chunked comp (the validated engine) on that config;
  2. template h = chunked ``fill_global_wdm`` cropped to the active grid --
     fp-validated against the dense WDM transform;
  3. holder: data slab = h itself (so d_h == h_h is a built-in consistency
     check), invC = the config's (1, 3, 3, F, T) XYZ-layout array;
  4. sig-het engine via ``for_band_engine`` (production knobs) ->
     ``setup_in_model`` -> ``get_ll`` at the ANCHOR -> ``h_h_out``;
  5. h_h_direct = einsum over the SAME arrays (the WDM inner product the
     chunked path is validated against; shared normalization cancels).

Output: one table, configs x f0, of hh_sighet / hh_direct. The first config
that reproduces the production curve IS the answer.

Run (CPU, pinned -- laptop policy):
    OMP_NUM_THREADS=1 python scripts/gb_chunked_het/gb_sighet_invitro_probe.py
Env: F0_LIST="1.0,2.25,4.5,10,16" (mHz), CONFIGS="0,1,2,3,4,5",
     PROBE_BACKEND=cpu.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

from lisatools.detector import ESAOrbits, DefaultOrbits
from lisatools.utils.constants import YRSID_SI
from lisatools import detector as lisa_models
from lisatools.sensitivity import (get_sensitivity, A2TDISens, X2TDISens,
                                   XY2TDISens)
from lisatools.stochastic import HyperbolicTangentGalacticForeground as HTGF
from lisatools.domains import WDMSettings

from gbgpu.gbcomps import GBWDMComputations
from gbgpu.gbsignalhetcomputations import GBSignalHetComputations
from gbgpu.gb_likelihood import make_band_likelihood_engine

GB_MOJITO_T_REF = 97729089.327664
# v4's own fitted noise (store cold medians) -- for the foreground axis.
PSD_P = [1.52274518e-11, 2.74762992e-15]
GAL_P = (3.74443460e-44, 5.29868210e-02, 9.26921788e-01, 2.75873152e-03,
         5.70634746e+03)


class Holder:
    """Duck-typed full-band buffer: exactly what setup_in_model reads."""

    def __init__(self, data, invc, xp=np):
        # data (1, 3, F, T) float; invc (1, 3, 3, F, T) float
        self.linear_data_arr = [xp.ascontiguousarray(
            xp.asarray(data, dtype=xp.float64)).ravel()]
        self.linear_psd_arr = [xp.ascontiguousarray(
            xp.asarray(invc, dtype=xp.float64)).ravel()]

    def __len__(self):
        return 1


def build_config(tag, *, prod_grid, prod_epoch, prod_caches, prod_npad,
                 fg_invc, xyz_csd=False, backend="cpu"):
    if prod_grid:
        Nf, Nt, dt = 1440, 2160, 2.5
        min_freq, max_freq = 1e-4, 2.5e-2
        wavelet_duration = Nf * dt
        min_time, max_time = 20 * wavelet_duration, (Nt - 20) * wavelet_duration
        nt_layer = 60
    else:
        Nf, Nt, dt = 1460, 2560, 10.0
        min_freq, max_freq = 0.0, 35.0e-3
        wavelet_duration = Nf * dt
        min_time, max_time = 20 * wavelet_duration, (Nt - 20) * wavelet_duration
        nt_layer = 64
    Tobs = Nf * Nt * dt

    if prod_epoch:
        t_start = 9.772994e7
        t_ref = GB_MOJITO_T_REF
        orbits = DefaultOrbits(force_backend=backend, frame="icrs")
    else:
        t_start = int(0.5 * YRSID_SI / dt) * dt
        t_ref = t_start
        orbits = ESAOrbits(force_backend=backend)

    wdm_set = WDMSettings(Nf, Nt, dt, t0=t_start,
                          min_freq=min_freq, max_freq=max_freq,
                          min_time=min_time, max_time=max_time,
                          force_backend=backend)

    chunked = GBWDMComputations(
        wdm_set, t_ref=t_ref,
        Nt_sub=256,
        n_pad=(8 if prod_npad else 32),
        N_sparse=256,
        N_cp_sig=(48 if prod_caches else 0),
        N_cp_orbit=(32 if prod_caches else 0),
        orbits=orbits, tdi_config="2nd generation",
        force_backend=backend, d_d=0.0, tdi_type="XYZ",
    )

    # invC on the ACTIVE grid, XYZ (3,3,F,T) layout, off-diagonals zero.
    F = wdm_set.ind_max_f - wdm_set.ind_min_f + 1
    T = wdm_set.ind_max_t - wdm_set.ind_min_t + 1
    m_abs = np.arange(wdm_set.ind_min_f, wdm_set.ind_max_f + 1)
    f_rows = np.maximum(m_abs * wdm_set.layer_df, 1e-5)
    model = lisa_models.LISAModel(PSD_P[0] ** 2, PSD_P[1] ** 2,
                                  lisa_models.DefaultOrbits(), "probe")
    kw = dict(model=model)
    if fg_invc:
        kw.update(stochastic_params=tuple(GAL_P), stochastic_function=HTGF)
    invc = np.zeros((1, 3, 3, F, T))
    if xyz_csd:
        # THE PRODUCTION NOISE STRUCTURE: full XYZ cross-channel covariance.
        # At low f the XYZ channels are strongly correlated (the X+Y+Z null
        # combination), C(f) is near-singular, and invC's OFF-DIAGONALS blow
        # up toward the band floor -- the same smooth shape as the measured
        # h_h inflation. A fold that mishandles the c1 != c2 terms errs in
        # proportion to |invC_offdiag / invC_diag|, i.e. exactly this curve.
        Cxx = np.asarray(get_sensitivity(f_rows, sens_fn=X2TDISens, **kw),
                         float)
        Cxy = np.asarray(get_sensitivity(f_rows, sens_fn=XY2TDISens, **kw),
                         float)
        C = np.empty((F, 3, 3))
        for a in range(3):
            for b in range(3):
                C[:, a, b] = Cxx if a == b else Cxy
        invC_rows = np.linalg.inv(C)                      # (F, 3, 3)
        odf = np.abs(invC_rows[:, 0, 1]) / np.abs(invC_rows[:, 0, 0])
        print(f"      [{tag}] |invC_xy/invC_xx| at row 0/mid/last: "
              f"{odf[0]:.3f} / {odf[F//2]:.3f} / {odf[-1]:.3f}")
        for a in range(3):
            for b in range(3):
                invc[0, a, b] = invC_rows[:, a, b][:, None]
    else:
        Sn = np.asarray(get_sensitivity(f_rows, sens_fn=A2TDISens, **kw),
                        float)
        for c in range(3):
            invc[0, c, c] = (1.0 / Sn)[:, None]

    return dict(tag=tag, wdm=wdm_set, chunked=chunked, invc=invc,
                nt_layer=nt_layer, F=F, T=T)


def probe_one(cfg, f0_hz, backend="cpu"):
    wdm, chunked = cfg["wdm"], cfg["chunked"]
    # GPU axis: production ran the compiled CUDA v5 kernel; the CPU probe
    # exercises the CPU build. PROBE_BACKEND=cuda12x/cuda13x runs the SAME
    # ladder through the compiled GPU kernels -- the axis a laptop cannot
    # reach, and (after every CPU config came back clean) a prime suspect.
    xp = chunked.xp
    params = np.array([1e-22, f0_hz, 1e-16, 0.0, 1.0, 0.7, 0.5, 1.2, 0.3])

    # 1. the validated template on the full grid, cropped active
    full = xp.zeros((3, wdm.Nf, wdm.Nt), dtype=xp.float64)
    chunked.fill_global_wdm(params.reshape(1, 9), full,
                            convert_to_ra_dec=False, factors=None)
    h = full[:, wdm.ind_min_f: wdm.ind_max_f + 1,
             wdm.ind_min_t: wdm.ind_max_t + 1]
    assert h.shape == (3, cfg["F"], cfg["T"]), (h.shape, cfg["F"], cfg["T"])

    invc = xp.asarray(cfg["invc"])
    holder = Holder(h[None], invc, xp)

    # 2. the untainted reference: direct pixel sum with the SAME invC
    hh_direct = float(xp.einsum("cft,dft,cdft->", h, h, invc[0]))

    # 3. sig-het at the anchor through the production entry points
    sig = GBSignalHetComputations.for_band_engine(
        chunked, nt_layer=cfg["nt_layer"], n_sparse_fd=1024,
        m_active_half_width=2, max_r=0.0, n_cp_build=32,
        v3_n_nodes=64, v4_knots=128, v4_band=16, v5=1)
    eng = make_band_likelihood_engine(
        wdm, gb_wdm_comp=sig, nchannels=3, tdi_channel_setup="XYZ")
    eng.setup_in_model(holder, params.reshape(1, 9), np.array([0]))
    eng.get_ll(holder, params.reshape(1, 9),
               data_index=np.array([0]), noise_index=np.array([0]),
               N_vals=np.array([1024]), waveform_kwargs={})
    hh_sig = float(np.asarray(eng.h_h_out).real.ravel()[0])
    dh_sig = float(np.asarray(eng.d_h_out).real.ravel()[0])
    eng.clear_in_model()
    return hh_sig, hh_direct, dh_sig


def main():
    backend = os.environ.get("PROBE_BACKEND", "cpu")
    f0_list = [float(x) * 1e-3 for x in
               os.environ.get("F0_LIST", "1.0,2.25,4.5,10,16").split(",")]
    which = [int(x) for x in
             os.environ.get("CONFIGS", "0,1,2,3,4,5").split(",")]

    ladder = [
        ("0 TEST", dict(prod_grid=False, prod_epoch=False, prod_caches=False,
                        prod_npad=False, fg_invc=False)),
        ("1 +GRID", dict(prod_grid=True, prod_epoch=False, prod_caches=False,
                         prod_npad=False, fg_invc=False)),
        ("2 +EPOCH", dict(prod_grid=True, prod_epoch=True, prod_caches=False,
                          prod_npad=False, fg_invc=False)),
        ("3 +CACHES", dict(prod_grid=True, prod_epoch=True, prod_caches=True,
                           prod_npad=False, fg_invc=False)),
        ("4 +NPAD", dict(prod_grid=True, prod_epoch=True, prod_caches=True,
                         prod_npad=True, fg_invc=False)),
        ("5 +FGINVC", dict(prod_grid=True, prod_epoch=True, prod_caches=True,
                           prod_npad=True, fg_invc=True)),
        ("6 +XYZCSD", dict(prod_grid=True, prod_epoch=True, prod_caches=True,
                           prod_npad=True, fg_invc=True, xyz_csd=True)),
    ]

    print(f"{'config':>10} " + "".join(f"{f*1e3:>9.2f}m" for f in f0_list)
          + "   (hh_sighet / hh_direct; d_h/h_h consistency in parens)")
    for i, (tag, kw) in enumerate(ladder):
        if i not in which:
            continue
        t0 = time.perf_counter()
        cfg = build_config(tag, backend=backend, **kw)
        row, chk = [], []
        for f0 in f0_list:
            hh_s, hh_d, dh_s = probe_one(cfg, f0, backend=backend)
            row.append(hh_s / hh_d if hh_d else np.nan)
            chk.append(dh_s / hh_s if hh_s else np.nan)
        print(f"{tag:>10} " + "".join(f"{r:>10.4f}" for r in row)
              + "   (" + " ".join(f"{c:.3f}" for c in chk) + ")"
              + f"  [{time.perf_counter()-t0:.0f}s]", flush=True)
    print("\nread: config 0 must be ~1.0000 everywhere (probe validity); the "
          "first config whose row bends into the production curve "
          "(~6 @ 1 mHz -> ~0.98 @ 16 mHz) names the axis. d_h/h_h should "
          "be ~1.000 (slab IS the template).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
