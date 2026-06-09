#!/usr/bin/env python
"""Compare logL(x_inj + Df0) from v2 polyphase-heterodyne vs the
lisatools dense ``calculate_signal_likelihood`` reference.

For each of the first ``N_DRAWS_PLOT`` prior draws (SEED matches the
mm sweep), the injection is the lisatools dense real-WDM at x_inj.
Then sweep DF0_FRAC over a range and compute:

  logL_lisa(x_cand) = analysis.calculate_signal_likelihood(*x_cand, source_only=True)
  logL_v2  (x_cand) = <d|tpl_v2(x_cand)> - 0.5 <tpl_v2|tpl_v2> - 0.5 <d|d>

with ``tpl_v2`` built by v2 polyphase + carrier de-rotation, using
c0 = lisatools dense complex-WDM at x_inj as the heterodyne reference.

Plot per draw:
  col 0: logL vs Df0/layer_df  (two curves, lisa + v2)
  col 1: logL_v2 - logL_lisa  (absolute residual)
  col 2: |residual| / max(|logL_lisa|, 1)  (relative)
  col 3: numeric panel with key params + worst residual

Run::
    python gb_signal_het_wdm_v2_logL_compare.py
Env vars:
    DF0_FRACS     comma-list of Df0/layer_df values
                  (default "-0.1,-0.05,-0.01,-0.005,-0.001,0,0.001,0.005,0.01,0.05,0.1")
    N_DRAWS_PLOT  number of prior draws (default 5)
    NT_LAYER      polyphase iFFT length (default 64)
    SEED          prior RNG seed (default 54321)
    OUT_PNG       output png path
"""

from __future__ import annotations

import os
import sys

import matplotlib
if not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal.windows import tukey as _tukey

from lisatools.analysiscontainer import AnalysisContainer
from lisatools.datacontainer import DataResidualArray
from lisatools.detector import ESAOrbits
from lisatools.domains import TDSettings, TDSignal, WDMSettings, WDMSignal
from lisatools.sensitivity import XYZ2SensitivityMatrix
from lisatools.utils.constants import YRSID_SI

from lisatools.response.tdiconfig import TDIConfig
from lisatools.response.tdionfly import GBTDIonTheFly

from gb_signal_het_wdm_v2 import GBSparseComplexWDMGen
from gb_signal_het_wdm_v2_mm_sweep import (
    build_gb_prior, build_v2_real_template,
)


def main():
    backend = "cpu"
    dt = 10.0
    Nf, Nt = 1460, 2560
    Nobs = Nf * Nt
    EC = 20

    df0_env = os.environ.get(
        "DF0_FRACS",
        "-0.1,-0.05,-0.02,-0.01,-0.005,-0.001,0,0.001,0.005,0.01,0.02,0.05,0.1",
    )
    DF0_FRACS = np.array([float(s) for s in df0_env.split(",")])
    N_DRAWS_PLOT = int(os.environ.get("N_DRAWS_PLOT", "5"))
    Nt_layer = int(os.environ.get("NT_LAYER", "64"))
    SEED = int(os.environ.get("SEED", "54321"))
    out_png = os.environ.get("OUT_PNG", "v2_logL_compare.png")

    t_start = int(0.5 * YRSID_SI / dt) * dt
    t_arr = np.arange(Nobs) * dt + t_start
    Tobs = Nt * Nf * dt

    orbits = ESAOrbits(force_backend=backend)
    tdi_config = TDIConfig("2nd generation", force_backend=backend)
    t_tdi = np.linspace(t_arr[0], t_arr[-1], 16384)
    gb_gen = GBTDIonTheFly(
        t_tdi, Tobs, t_start, 1.0 / dt, 1,
        tdi_config=tdi_config, orbits=orbits, tdi_chan="XYZ",
        force_backend=backend,
    )

    def real_td_cb(p):
        amp, f0, fdot, fddot, phi0, inc, psi, lam, beta = p
        spline = gb_gen(
            np.array([amp]), np.array([f0]), np.array([fdot]),
            np.array([fddot]), np.array([phi0]), np.array([inc]),
            np.array([psi]), np.array([lam]), np.array([beta]),
            convert_to_ra_dec=False, return_spline=True,
        )
        return np.asarray(spline.eval_tdi(t_arr))[0]

    td_set = TDSettings(Nobs, dt, force_backend=backend)
    window = _tukey(Nobs, alpha=0.05).astype(float)

    wdm_set_real = WDMSettings(
        Nf, Nt, dt, t0=t_start,
        min_freq=1e-4, max_freq=35e-3,
        min_time=EC * Nf * dt, max_time=(Nt - EC) * Nf * dt,
        is_complex=False, force_backend=backend,
    )
    wdm_set_complex = WDMSettings(
        Nf, Nt, dt, t0=t_start,
        min_freq=1e-4, max_freq=35e-3,
        min_time=EC * Nf * dt, max_time=(Nt - EC) * Nf * dt,
        is_complex=True, force_backend=backend,
    )
    layer_df = wdm_set_real.layer_df
    ind_min_t = int(wdm_set_real.ind_min_t)
    Nt_active = int(wdm_set_real.Nt_active)
    Nf_active = int(wdm_set_real.ind_max_f - wdm_set_real.ind_min_f + 1)
    ind_min_f = int(wdm_set_real.ind_min_f)
    print(f"[grid] layer_df={layer_df:.3e}Hz  Nt_active={Nt_active}  "
          f"ind_min_t={ind_min_t}  ind_min_f={ind_min_f}", flush=True)
    print(f"[scan] DF0_FRACS={DF0_FRACS}", flush=True)

    sparse_gen = GBSparseComplexWDMGen(
        real_td_callable=real_td_cb, wdm_set_complex=wdm_set_complex,
        data_dt=dt, ind_min_t=ind_min_t, Nt_active=Nt_active,
        Nt_layer=Nt_layer, m_active_half_width=2,
    )
    n_sparse = sparse_gen.n_sparse_local
    stride = sparse_gen.stride
    print(f"[v2] Nt_layer={Nt_layer}  stride={stride}  "
          f"N_sparse_t={sparse_gen.N_sparse_t}", flush=True)

    # Prior matches the sweep.
    np.random.seed(SEED)
    f0_lo = (ind_min_f + 7) * layer_df
    f0_hi = 0.025
    prior, tc = build_gb_prior(
        A_lims=(1e-23, 1e-20), f0_lims_hz=(f0_lo, f0_hi),
        fdot_lims=(-1e-15, 1e-15), beta_lims=None,
    )

    SNR_MIN, SNR_MAX = 5.0, 1100.0
    MAX_REJECT = 500

    fig, axs = plt.subplots(N_DRAWS_PLOT, 4, figsize=(22, 4.4 * N_DRAWS_PLOT))
    if N_DRAWS_PLOT == 1:
        axs = axs[None, :]

    sens_mat = None
    drawi = 0
    while drawi < N_DRAWS_PLOT:
        chosen = None
        for _ in range(MAX_REJECT):
            x_samp = prior.rvs(size=1)
            params_inj = tc.both_transforms(x_samp.copy())[0]
            td_inj = real_td_cb(params_inj)
            wdm_inj_real_sig = TDSignal(td_inj, settings=td_set).transform(
                wdm_set_real, window=window,
            )
            inj_data_arr = DataResidualArray(wdm_inj_real_sig)
            if sens_mat is None:
                sens_mat = XYZ2SensitivityMatrix(
                    inj_data_arr.data_res_arr.settings, model="scirdv1",
                )
            analysis = AnalysisContainer(inj_data_arr, sens_mat)
            snr_i = float(analysis.snr())
            if SNR_MIN <= snr_i <= SNR_MAX:
                chosen = (params_inj, td_inj, wdm_inj_real_sig, snr_i, analysis)
                break
        if chosen is None:
            print(f"[warn] draw {drawi}: exhausted reject; keeping last", flush=True)
            chosen = (params_inj, td_inj, wdm_inj_real_sig, snr_i, analysis)
        params_inj, td_inj, wdm_inj_real_sig, snr_i, analysis = chosen

        # Tell AnalysisContainer how to build templates for direct logL:
        # callable: params -> WDMSignal (real WDM, on the parent grid).
        def _lisa_signal_gen(*params9):
            td = real_td_cb(np.asarray(params9, dtype=float).reshape(9))
            return TDSignal(td, settings=td_set).transform(
                wdm_set_real, window=window,
            )
        analysis.signal_gen = _lisa_signal_gen

        # Reference c0 at x_inj (lisatools complex WDM)
        wdm_ref_complex = np.asarray(
            TDSignal(td_inj, settings=td_set).transform(
                wdm_set_complex, window=window,
            ).arr
        ).copy()
        c0_sparse = wdm_ref_complex[:, :, n_sparse].copy()

        # d_d for residual normalization etc.
        d_d = float(np.real(analysis.inner_product()))

        logL_lisa = np.zeros(len(DF0_FRACS))
        logL_v2 = np.zeros(len(DF0_FRACS))
        for i, dff in enumerate(DF0_FRACS):
            params_cand = params_inj.copy()
            params_cand[1] = params_inj[1] + dff * layer_df
            # lisatools direct
            logL_lisa[i] = float(
                analysis.calculate_signal_likelihood(
                    *params_cand, source_only=True,
                )
            )
            # v2 heterodyne
            td_cand = real_td_cb(params_cand)
            fd_rfft_cand = np.fft.rfft(td_cand * window, axis=-1)
            tpl_real = build_v2_real_template(
                sparse_gen, fd_rfft_cand, params_cand, params_inj,
                Nf_active, Nt_active, wdm_ref_complex, c0_sparse,
                dt, Nf, ind_min_t,
            )
            tpl_sig = WDMSignal(tpl_real, wdm_set_real)
            d_h = float(analysis.template_inner_product(tpl_sig))
            h_h = float(analysis.template_snr(tpl_sig)[0]) ** 2
            logL_v2[i] = d_h - 0.5 * h_h - 0.5 * d_d

        resid = logL_v2 - logL_lisa
        denom = np.maximum(np.abs(logL_lisa), 1.0)
        rel_resid = np.abs(resid) / denom

        # col 0: logL vs Df0
        ax = axs[drawi, 0]
        ax.plot(DF0_FRACS, logL_lisa, "C0o-", label="lisatools direct", markersize=4)
        ax.plot(DF0_FRACS, logL_v2, "C3x--", label="v2 heterodyne", markersize=6)
        ax.axvline(0, color="k", linewidth=0.5, alpha=0.4)
        ax.set_xlabel("Df0 / layer_df")
        ax.set_ylabel("logL  (source_only)")
        ax.set_title(
            f"draw {drawi}: logL  vs Df0\n"
            f"f0={params_inj[1]*1e3:.4f}mHz snr={snr_i:.1f} "
            f"m_floor={int(params_inj[1]/layer_df)}",
            fontsize=10,
        )
        ax.grid(alpha=0.3); ax.legend(loc="best", fontsize=9)

        # col 1: absolute residual
        ax = axs[drawi, 1]
        ax.semilogy(DF0_FRACS, np.maximum(np.abs(resid), 1e-300),
                    "C2o-", markersize=4)
        ax.axvline(0, color="k", linewidth=0.5, alpha=0.4)
        ax.set_xlabel("Df0 / layer_df")
        ax.set_ylabel("|logL_v2 - logL_lisa|")
        ax.set_title("absolute residual")
        ax.grid(alpha=0.3, which="both")

        # col 2: relative residual
        ax = axs[drawi, 2]
        ax.semilogy(DF0_FRACS, np.maximum(rel_resid, 1e-20),
                    "C1o-", markersize=4)
        ax.axvline(0, color="k", linewidth=0.5, alpha=0.4)
        ax.set_xlabel("Df0 / layer_df")
        ax.set_ylabel("|resid| / max(|logL_lisa|, 1)")
        ax.set_title("relative residual")
        ax.grid(alpha=0.3, which="both")

        # col 3: numeric panel
        ax = axs[drawi, 3]
        ax.axis("off")
        worst_idx = int(np.argmax(np.abs(resid)))
        text = (
            f"DRAW {drawi}\n\n"
            f"params_inj:\n"
            f"  amp     = {params_inj[0]:.2e}\n"
            f"  f0      = {params_inj[1]*1e3:.5f} mHz\n"
            f"            (m_floor={int(params_inj[1]/layer_df)}, "
            f"frac={params_inj[1]/layer_df - int(params_inj[1]/layer_df):.3f})\n"
            f"  fdot    = {params_inj[2]:.2e}\n"
            f"  beta    = {params_inj[8]:+.3f}\n"
            f"  snr     = {snr_i:.1f}\n\n"
            f"<d|d>     = {d_d:+.3e}\n\n"
            f"worst residual:\n"
            f"  Df0/lay = {DF0_FRACS[worst_idx]:+.4g}\n"
            f"  abs     = {abs(resid[worst_idx]):.3e}\n"
            f"  rel     = {rel_resid[worst_idx]:.3e}\n"
            f"  logL_li = {logL_lisa[worst_idx]:+.3e}\n"
            f"  logL_v2 = {logL_v2[worst_idx]:+.3e}\n"
        )
        ax.text(
            0.02, 0.98, text, transform=ax.transAxes,
            va="top", ha="left", family="monospace", fontsize=9,
        )

        print(
            f"[draw {drawi}] f0={params_inj[1]*1e3:.4f}mHz snr={snr_i:.1f} "
            f"max|resid|={np.abs(resid).max():.3e} "
            f"max_rel={rel_resid.max():.3e}",
            flush=True,
        )
        drawi += 1

    fig.suptitle(
        f"logL(x_inj + Df0) comparison: v2 polyphase-heterodyne vs lisatools "
        f"direct  (Nt_layer={Nt_layer}, SEED={SEED})",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[plot] {out_png}", flush=True)
    print("DONE.")


if __name__ == "__main__":
    sys.exit(main())
