#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, sys
import numpy as np
import matplotlib
# Use Agg by default so the verification run is non-interactive. If the user
# wants the heatmap window, they can override with MPLBACKEND=Qt5Agg etc.
if not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal

try:
    import cupy as cp

except (ImportError, ModuleNotFoundError) as e:
    pass

from lisatools.detector import ESAOrbits, EqualArmlengthOrbits
from lisaconstants import ASTRONOMICAL_YEAR
from lisatools.utils.constants import YRSID_SI
from fastlisaresponse import ResponseWrapper
from fastlisaresponse.tdiconfig import TDIConfig
from fastlisaresponse.response import icrs_to_ecliptic
from fastlisaresponse.tdionfly import GBTDIonTheFly
from fastlisaresponse.gbcomps import GBWDMComputations

from lisatools.datacontainer import DataResidualArray
from lisatools.analysiscontainer import AnalysisContainer, AnalysisContainerArray
from lisatools.sensitivity import XYZ2SensitivityMatrix
from lisatools.domains import TDSettings, TDSignal, FDSettings, FDSignal, WDMSettings, WDMSignal, WDMLookupTable

from eryn.utils import TransformContainer
from eryn.prior import ProbDistContainer, uniform_dist, log_uniform

from eryn.moves import StretchMove
from eryn.ensemble import EnsembleSampler
from eryn.utils import PeriodicContainer

from eryn.state import State
from eryn.backends import HDFBackend
# credit Michael Katz and Alessandro Santini (with internal code contrubtions in docs)


    # HIGHLY RECOMMEND RUNNING THESE THINGS IN A SCRIPT IN THE TERMINAL, OTHERWISE BE CAREFUL TO RUN CELLS IN ORDER AS MUCH AS POSSIBLE

if __name__ == "__main__":
    backend = "cpu"

    xp = np if backend == "cpu" else cp

    from lisatools.domains import WDMLookupTable

    wdm_lookup_table = WDMLookupTable.from_file("wdm_lookup_parity_fix.h5")
    wdm_set = wdm_lookup_table.settings

    Nf = wdm_set.Nf
    Nt = wdm_set.Nt
    wavelet_duration = Nf * wdm_set.data_dt
    Tobs = Nt * wavelet_duration
    Nobs = Nf * Nt

    # Conventions required to match this lookup table:
    #   * The table was built from waveforms `cos/sin(2 pi f (t - t_ref))`
    #     time-windowed with a Tukey window (alpha=0.05). Apply the same
    #     window to the injection so its WDM matches what the lookup encodes.
    #   * Phase is referenced at `t_ref = n_ref * layer_dt`.
    #   * `get_wdm_coeffs` returns `WDM[A * sin(2 pi f (t - t_n) + phi_arr[n])]`
    #     once the parity/(-1)^m_diff dance is applied (verified numerically:
    #     a sin injection matches the lookup output to ~1e-12; cos does not).
    #     So inject `sin(...)` if the user's phi0 should be interpreted as
    #     the phase at t_ref of a sin source.
    #   * `n_arr` passed to `get_wdm_coeffs` MUST be the FULL-grid n index,
    #     not the sliced one. The build interpolator was constructed over
    #     `arange(Nt)`, so passing sliced indices reads the wrong rows.

    m_ref = int(wdm_lookup_table.m_ref)
    n_ref = int(wdm_lookup_table.n_ref)
    t_ref = n_ref * wdm_set.layer_dt

    # Deterministic "on-grid" cases (f_frac = 0 or exact 0.5) followed by
    # random off-grid cases that exercise interpolation between layer centres.
    on_grid_sets = [
        # [m_ref * wdm_set.layer_df, 0.0],
        [(m_ref + 0.5195234524234) * wdm_set.layer_df, 4.082342323423],
        [(m_ref + 0.0195234524234) * wdm_set.layer_df, 3.92823423],
        [(m_ref + 0.5) * wdm_set.layer_df, np.pi/4.],
        [(m_ref + 0.5) * wdm_set.layer_df, np.pi/2.],
        [(m_ref + 0.5) * wdm_set.layer_df, np.pi],
        [(m_ref + 1) * wdm_set.layer_df, 0.0],
        [m_ref * wdm_set.layer_df, np.pi / 2.],
        [m_ref * wdm_set.layer_df, np.pi],
        [m_ref * wdm_set.layer_df, np.pi / 4.],
        [m_ref * wdm_set.layer_df, 0.82395472938472397],
        # half-bin cases (worst off-grid)
        [(m_ref + 0.5) * wdm_set.layer_df, 0.0],
        [(m_ref + 0.5) * wdm_set.layer_df, np.pi / 3.],
        # quarter-bin
        [(m_ref + 0.25) * wdm_set.layer_df, 0.0],
        [(m_ref + 0.75) * wdm_set.layer_df, np.pi / 5.],
    ]

    # Randomized off-grid sweep across the WDM band. Stay away from the very
    # edges of [min_freq, max_freq] so the lookup's 3-layer window never
    # falls off the table.
    rng = np.random.default_rng(20260518)
    n_rand = 30
    f_lo = wdm_set.min_freq + 3 * wdm_set.layer_df
    f_hi = wdm_set.max_freq - 3 * wdm_set.layer_df
    rand_f = rng.uniform(f_lo, f_hi, n_rand)
    rand_phi = rng.uniform(0.0, 2.0 * np.pi, n_rand)
    rand_sets = [[float(f), float(p)] for f, p in zip(rand_f, rand_phi)]

    sets = on_grid_sets + rand_sets

    # Track which (f0, phi0) gives the worst off-grid per-layer error so we
    # can save a focused heatmap for it.
    worst = {"err_pm1": -1.0, "i": -1, "set": None,
             "errs": (None, None, None), "f_frac": None}
    summary = []

    t_arr = np.arange(Nobs) * wdm_set.data_dt
    td_window = np.ones(Nobs)  # NO Tukey -- expect mismatch vs the tukey-built table

    # FULL-grid n indices for the lookup query (not the sliced wdm_set.t_arr).
    n_arr_full = np.arange(Nt)
    t_n_full = n_arr_full * wdm_set.layer_dt
    amp_t = np.ones(Nt)
    fdot_t = np.zeros(Nt)

    for i, (f0, phi0) in enumerate(sets):
        # Injection: sin source phase-referenced at t_ref so that `phi0` is
        # literally the phase at t = t_ref. cos(2 pi f0 t + phi0) does NOT
        # match the lookup; sin(...) does.
        phase_t_arr = 2 * np.pi * f0 * (t_arr - t_ref) + phi0
        wave = np.sin(phase_t_arr)

        # Match the table's build window so the WDM is on the same footing.
        wave_wdm = TDSignal(
            np.asarray([wave, wave]),
            TDSettings(Nobs, wdm_set.data_dt),
        ).wdmtransform(settings=wdm_set, window=td_window)

        wave_cos = np.cos(phase_t_arr)

        # Match the table's build window so the WDM is on the same footing.
        wave_cos_wdm = TDSignal(
            np.asarray([wave_cos, wave_cos]),
            TDSettings(Nobs, wdm_set.data_dt),
        ).wdmtransform(settings=wdm_set, window=td_window)

        # Lookup query at every full WDM bin. phi_arr[n] is the source's
        # phase at the absolute time t_n of bin n.
        phase_at_t_n = 2 * np.pi * f0 * (t_n_full - t_ref) + phi0
        freq_t = np.full(Nt, f0)
        wmn_look, m_look = wdm_lookup_table.get_wdm_coeffs(
            amp_t, phase_at_t_n, freq_t, fdot_t, n_arr_full,
        )
        wmn_look = np.asarray(wmn_look)
        m_look = np.asarray(m_look)

        # Scatter the lookup coefficients into a full (Nf, Nt) array; the
        # WDMSignal constructor will slice it down to the active region.
        new_wave_full = np.zeros((Nf, Nt))
        for j in range(wmn_look.shape[1]):
            mm = m_look[:, j]
            valid = (mm >= 0) & (mm < Nf)
            new_wave_full[mm[valid], n_arr_full[valid]] = wmn_look[valid, j]
        new_wave_wdm = WDMSignal(np.asarray([new_wave_full, new_wave_full]), wdm_set)

        # Per-layer numerical check: m_floor-1, m_floor, m_floor+1 (the three
        # layers the lookup actually emits). Edge layers carry interpolation
        # error when f_frac != 0, so they're the interesting probe.
        m_floor = int(f0 / wdm_set.layer_df)
        f_frac = f0 / wdm_set.layer_df - m_floor

        def layer_err(m_full):
            m_s = m_full - wdm_set.ind_min_f
            if m_s < 0 or m_s >= wave_wdm.arr.shape[1]:
                return (float("nan"), float("nan"), float("nan"))
            ref = wave_wdm.arr[0, m_s]
            lt = new_wave_wdm.arr[0, m_s]
            d_norm = float(np.linalg.norm(ref))
            n_diff = float(np.linalg.norm(ref - lt))
            rel = n_diff / d_norm if d_norm > 0 else float("nan")
            max_abs = float(np.max(np.abs(ref - lt)))
            return (d_norm, rel, max_abs)

        e_m1 = layer_err(m_floor - 1)
        e_0 = layer_err(m_floor)
        e_p1 = layer_err(m_floor + 1)

        print(
            f"[set {i:02d}] f0={f0*1e3:8.4f} mHz  phi0={phi0:6.4f}  "
            f"m_floor={m_floor:4d}  f_frac={f_frac:+.3f}\n"
            f"        m-1:  ||inj||={e_m1[0]:.3e}  rel={e_m1[1]:.3e}  max|d|={e_m1[2]:.3e}\n"
            f"        m  :  ||inj||={e_0[0]:.3e}  rel={e_0[1]:.3e}  max|d|={e_0[2]:.3e}\n"
            f"        m+1:  ||inj||={e_p1[0]:.3e}  rel={e_p1[1]:.3e}  max|d|={e_p1[2]:.3e}",
            flush=True,
        )

        summary.append({
            "i": i, "f0_mHz": f0 * 1e3, "phi0": phi0,
            "m_floor": m_floor, "f_frac": f_frac,
            "rel_m_minus_1": e_m1[1], "rel_m": e_0[1], "rel_m_plus_1": e_p1[1],
            "max_m_minus_1": e_m1[2], "max_m": e_0[2], "max_m_plus_1": e_p1[2],
        })

        # Worst-of-the-edges score = max of rel_err at m-1 and m+1 where the
        # layer actually carries energy (skip layers whose ||inj|| is tiny so
        # division noise doesn't dominate).
        cand = []
        for (norm_, rel_, _max) in (e_m1, e_p1):
            if norm_ > 1e-3:  # non-trivial energy
                cand.append(rel_)
        if cand:
            err_pm1 = max(cand)
            if err_pm1 > worst["err_pm1"]:
                worst.update({"err_pm1": err_pm1, "i": i,
                              "set": (f0, phi0), "errs": (e_m1, e_0, e_p1),
                              "f_frac": f_frac})

        # Only render heatmaps for the original 10 deterministic sets, to
        # keep the run cheap (full-grid pcolormesh on 1460x2560 is slow).
        if i < len(on_grid_sets):
            plt.rcParams['text.usetex'] = False
            vmax = float(np.abs(wave_wdm.arr[0]).max())
            vmax = max(vmax, float(np.abs(new_wave_wdm.arr[0]).max()))
            if vmax == 0.0:
                vmax = 1.0
            diff_arr = wave_wdm.arr - new_wave_wdm.arr
            diff_wdm = WDMSignal(diff_arr.copy(), wdm_set)
            diff_vmax = float(np.abs(diff_arr).max())
            diff_vmax_2 = float(np.abs(diff_arr[:, :, 50:-50]).max())
            if diff_vmax == 0.0:
                diff_vmax = 1.0
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(11, 8))

            wave_wdm.heatmap(fig=fig, ax=ax1, index=0, add_cax=True, vmin=-vmax, vmax=vmax)
            new_wave_wdm.heatmap(fig=fig, ax=ax2, index=0, add_cax=True, vmin=-vmax, vmax=vmax)
            # wave_cos_wdm.heatmap(fig=fig, ax=ax3, index=0, add_cax=True, vmin=-vmax, vmax=vmax)
            diff_wdm.heatmap(fig=fig, ax=ax3, index=0, add_cax=True,
                             vmin=-diff_vmax, vmax=diff_vmax)
            ax1.set_title(
                f"injection  sin(2 pi f0 (t - t_ref) + phi0)   "
                f"f0={f0*1e3:.4f} mHz  phi0={phi0:.4f}  f_frac={f_frac:+.3f}"
            )
            ax2.set_title("py lookup template")
            ax3.set_title(f"difference (inj - lookup),  max|diff1,2|=({diff_vmax:.3e}, {diff_vmax_2: .3e})", fontsize=12)
            ax1.set_ylim(f0 - 10 * wdm_set.layer_df, f0 + 10 * wdm_set.layer_df)
            plt.tight_layout()
            out_path = f"gb_lookup_test_heatmap_set{i:02d}.png"
            plt.savefig(out_path, dpi=120)
            plt.show()
            
            plt.close(fig)
            print(f"  saved {out_path}  max|diff|={diff_vmax:.3e}", flush=True)

        # Stash the WDMSignals for the worst case (we may render a plot after
        # the loop). Avoid keeping every iteration's arrays in memory.
        if worst["i"] == i:
            worst["wave_wdm_arr"] = wave_wdm.arr.copy()
            worst["new_wave_wdm_arr"] = new_wave_wdm.arr.copy()

    # --------- aggregate summary ---------
    print("\n=== off-grid summary (random sets only) ===")
    rand_summary = [s for s in summary if s["i"] >= len(on_grid_sets)]
    if rand_summary:
        for col in ("rel_m_minus_1", "rel_m", "rel_m_plus_1"):
            vals = np.array([s[col] for s in rand_summary], dtype=float)
            vals_finite = vals[np.isfinite(vals)]
            if vals_finite.size:
                print(f"  {col:>14s}  median={np.median(vals_finite):.3e}  "
                      f"p90={np.percentile(vals_finite, 90):.3e}  "
                      f"max={vals_finite.max():.3e}")

    # Render a heatmap for the worst off-grid case so any edge-layer bleed
    # is visible.
    if worst["i"] >= 0 and "wave_wdm_arr" in worst:
        f0_w, phi0_w = worst["set"]
        arr_inj = worst["wave_wdm_arr"]
        arr_lt = worst["new_wave_wdm_arr"]
        diff_arr_w = arr_inj - arr_lt
        vmax_w = max(float(np.abs(arr_inj).max()), float(np.abs(arr_lt).max()))
        if vmax_w == 0.0:
            vmax_w = 1.0
        diff_vmax_w = float(np.abs(diff_arr_w).max())
        if diff_vmax_w == 0.0:
            diff_vmax_w = 1.0
        wdm_inj_w = WDMSignal(arr_inj.copy(), wdm_set)
        wdm_lt_w = WDMSignal(arr_lt.copy(), wdm_set)
        wdm_diff_w = WDMSignal(diff_arr_w.copy(), wdm_set)
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(11, 8))
        wdm_inj_w.heatmap(fig=fig, ax=ax1, index=0, add_cax=True, vmin=-vmax_w, vmax=vmax_w)
        wdm_lt_w.heatmap(fig=fig, ax=ax2, index=0, add_cax=True, vmin=-vmax_w, vmax=vmax_w)
        wdm_diff_w.heatmap(fig=fig, ax=ax3, index=0, add_cax=True,
                           vmin=-diff_vmax_w, vmax=diff_vmax_w)
        ax1.set_title(
            f"WORST OFF-GRID  f0={f0_w*1e3:.4f} mHz  phi0={phi0_w:.4f}  "
            f"f_frac={worst['f_frac']:+.3f}  rel(m±1)={worst['err_pm1']:.3e}"
        )
        ax2.set_title("py lookup template")
        ax3.set_title(f"difference (inj - lookup),  max|diff|={diff_vmax_w:.3e}")
        ax1.set_ylim(f0_w - 10 * wdm_set.layer_df, f0_w + 10 * wdm_set.layer_df)
        plt.tight_layout()
        out_path = "gb_lookup_test_heatmap_worst_offgrid.png"
        plt.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"\n  saved {out_path}", flush=True)

