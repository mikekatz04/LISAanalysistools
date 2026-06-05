"""
SOBBH prior-draws validation of the chunked-heterodyne pipeline.

Mirror of ``gb_chunked_prior_draws.py`` for SOBBHs. Same lisatools
ground-truth pattern: build the injection via ``SOBBHTDIonTheFly`` ->
TD -> WDM full transform; build the template via ``SOBBHWDMHeterodyne``
(chunked heterodyne). Compute mismatch (1 - O) on three bands:

  full       -- the full WDM-active band
  5-layer    -- ``[f_low - 3*layer_df, f_low + 2*layer_df]``
  2-layer    -- ``[m_floor*layer_df, (m_floor+2)*layer_df - layer_df/2]``

Plots:
  - log-hist of the three 1-O values
  - f_low / f_frac vs mm5 / mm2 scatter (4 panels), colored by log10(SNR)

The SOBBH prior is uniform over (m1, m2, s1, s2, distance, f_low, phi_c,
inc, psi, lam, beta), with mass and f_low ranges chosen so the signal
stays comfortably in the LISA band over the obs window (no late-inspiral
merger). No eryn dependency; sampling is rejection-based on SNR.

Env knobs (mirror the GB version):
    N_DRAWS, SNR_MIN, SNR_MAX, NF, NT, NT_SUB, N_SPARSE, N_PAD,
    F_LOW_LO_HZ, F_LOW_HI_HZ, M1_LO, M1_HI, M2_LO, M2_HI,
    N_CP_SIG, N_CP_ORBIT, OUTPUT_PREFIX
"""
import os
import time

import numpy as np
import matplotlib
if os.environ.get("DISPLAY", "") == "":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lisatools.detector import ESAOrbits
from lisatools.utils.constants import YRSID_SI
from lisatools.response.tdiconfig import TDIConfig
from lisatools.response.tdionfly import SOBBHTDIonTheFly

from lisatools.datacontainer import DataResidualArray
from lisatools.analysiscontainer import AnalysisContainer
from lisatools.sensitivity import XYZ2SensitivityMatrix
from lisatools.domains import (
    TDSettings, FDSettings, WDMSettings, TDSignal, WDMSignal,
)

from gb_wdm_het import SOBBHWDMHeterodyne


def _sample_sobbh_params(rng, f_low_lo, f_low_hi, m1_lo, m1_hi, m2_lo, m2_hi):
    """Draw one SOBBH parameter vector from a uniform prior.

    Returns ``params`` of shape (11,) in SOBBHTDIonTheFly order:
        [m1, m2, s1, s2, distance, f_low, phi_c, inc, psi, lam, beta]
    """
    m1 = rng.uniform(m1_lo, m1_hi)
    m2 = rng.uniform(m2_lo, m1)                     # m2 <= m1
    s1 = rng.uniform(-0.5, 0.5)
    s2 = rng.uniform(-0.5, 0.5)
    distance = rng.uniform(500.0, 5000.0)           # Mpc
    f_low = rng.uniform(f_low_lo, f_low_hi)
    phi_c = rng.uniform(0.0, 2.0 * np.pi)
    inc = np.arccos(rng.uniform(-1.0, 1.0))
    psi = rng.uniform(0.0, np.pi)
    lam = rng.uniform(0.0, 2.0 * np.pi)
    beta = np.arcsin(rng.uniform(-1.0, 1.0))
    return np.array([m1, m2, s1, s2, distance, f_low, phi_c,
                      inc, psi, lam, beta], dtype=float)


def main():
    backend = os.environ.get("CHUNKED_BACKEND", "cpu")

    # --- config ----------------------------------------------------------
    N_DRAWS = int(os.environ.get("N_DRAWS", 20))
    SNR_MIN = float(os.environ.get("SNR_MIN", 5.0))
    SNR_MAX = float(os.environ.get("SNR_MAX", 1100.0))
    N_INJ = int(os.environ.get("N_INJ", 16384))
    MAX_REJECT = int(os.environ.get("MAX_REJECT", 500))
    SEED = int(os.environ.get("SEED", 12345))
    OUTPUT_PREFIX = os.environ.get("OUTPUT_PREFIX", "sobbh_chunked_test")
    PROGRESS_EVERY = int(os.environ.get("PROGRESS_EVERY", 1))

    Nt_sub = int(os.environ.get("NT_SUB",  256))
    N_sparse = int(os.environ.get("N_SPARSE", 256))
    n_pad = int(os.environ.get("N_PAD", Nt_sub // 8))

    rng = np.random.default_rng(SEED)

    # --- detector / WDM grid ---------------------------------------------
    orbits = ESAOrbits(force_backend=backend)
    dt = 10.0
    Nf = int(os.environ.get("NF", 1460))
    Nt = int(os.environ.get("NT", 256 * 10 * (1460 // Nf if Nf <= 1460 else 1)))
    print(f"[run] Nf={Nf} Nt={Nt} dt={dt}  (Tobs={Nf*Nt*dt:.3e}s, "
          f"Nt_sub={Nt_sub}, N_sparse={N_sparse}, n_pad={n_pad})", flush=True)
    wavelet_duration = Nf * dt
    Tobs = Nt * wavelet_duration

    tdi_config = TDIConfig("2nd generation")
    t_start = int(0.5 * YRSID_SI / dt) * dt
    t_arr = np.arange(Nf * Nt) * dt + t_start
    t_ref = t_start

    sobbh_tdi_kwargs = dict(
        tdi_config=tdi_config, orbits=orbits,
        tdi_chan="XYZ", force_backend=backend,
    )
    t_tdi_inj = np.linspace(t_arr[0], t_arr[-1], N_INJ)
    sobbh_gen_inj = SOBBHTDIonTheFly(
        t_tdi_inj, Tobs, t_ref, 1.0 / dt, 1, **sobbh_tdi_kwargs,
    )

    N = t_arr.shape[-1]
    td_set = TDSettings(N, dt, force_backend=backend)
    freqs = np.fft.rfftfreq(N, dt)
    df = freqs[1] - freqs[0]
    N_fd = len(freqs)

    min_freq = float(os.environ.get("MIN_FREQ_HZ", 0.0))
    max_freq = float(os.environ.get("MAX_FREQ_HZ", 35.0e-3))
    _ = FDSettings(N_fd, df, min_freq=min_freq, max_freq=max_freq,
                   force_backend=backend)
    min_time = 20 * wavelet_duration
    max_time = (Nt - 20) * wavelet_duration
    wdm_set = WDMSettings(
        Nf, Nt, dt,
        min_freq=min_freq, max_freq=max_freq,
        min_time=min_time, max_time=max_time,
        force_backend=backend,
    )

    chunked = SOBBHWDMHeterodyne(
        Nf=Nf, Nt=Nt, dt=dt, T_full=Tobs, t_ref_full=t_ref,
        Nt_sub=Nt_sub, n_pad=n_pad, N_sparse=N_sparse,
        backend=backend, tdi_gen="2nd generation",
        orbits=orbits, t_obs_start=float(t_start),
        N_cp_sig=int(os.environ.get("N_CP_SIG", 0)),
        N_cp_orbit=int(os.environ.get("N_CP_ORBIT", 0)),
    )
    print(f"[run] chunked: n_chunks={len(chunked.geometry['starts'])}, "
          f"T_chunk={chunked.T_chunk:.3e}s, alpha={chunked.tukey_alpha}, "
          f"use_tukey={chunked.use_tukey}", flush=True)

    layer_df = wdm_set.layer_df
    buffer_layers = 7
    f_low_lo_default = (wdm_set.ind_min_f + buffer_layers) * layer_df
    f_low_hi_default = (wdm_set.ind_max_f - buffer_layers) * layer_df
    f_low_lo_hz = float(os.environ.get("F_LOW_LO_HZ", f_low_lo_default))
    f_low_hi_hz = float(os.environ.get("F_LOW_HI_HZ", f_low_hi_default))
    m1_lo = float(os.environ.get("M1_LO", 20.0))
    m1_hi = float(os.environ.get("M1_HI", 50.0))
    m2_lo = float(os.environ.get("M2_LO", 10.0))
    m2_hi = float(os.environ.get("M2_HI", 40.0))

    print(f"[run] N_DRAWS={N_DRAWS} SNR window=[{SNR_MIN}, {SNR_MAX}]", flush=True)
    print(f"[run] f_low range = [{f_low_lo_hz*1e3:.4f}, {f_low_hi_hz*1e3:.4f}] mHz "
          f"(layer_df = {layer_df:.3e} Hz)", flush=True)
    print(f"[run] mass ranges m1=[{m1_lo}, {m1_hi}]  m2=[{m2_lo}, min(m2_hi, m1)]", flush=True)

    sens_mat = None
    snr_list, log_like_list, mismatch_list = [], [], []
    mismatch_5_layers_list, mismatch_2_layers_list = [], []
    params_list = []
    attempt_total = 0
    t_loop_start = time.perf_counter()

    for i in range(N_DRAWS):
        chosen = None
        for _ in range(MAX_REJECT):
            attempt_total += 1
            params_i = _sample_sobbh_params(
                rng, f_low_lo_hz, f_low_hi_hz, m1_lo, m1_hi, m2_lo, m2_hi,
            )
            try:
                inj_spline = sobbh_gen_inj(
                    *params_i.reshape(11, 1),
                    convert_to_ra_dec=False, return_spline=True,
                )
                td_inj = np.asarray(inj_spline.eval_tdi(t_arr))[0]
                if not np.all(np.isfinite(td_inj)):
                    continue
                wdm_inj_sig = TDSignal(td_inj, settings=td_set).transform(
                    wdm_set, window=None,
                )
                injection = DataResidualArray(wdm_inj_sig)
                if sens_mat is None:
                    sens_mat = XYZ2SensitivityMatrix(
                        injection.data_res_arr.settings, model="scirdv1",
                    )
                analysis = AnalysisContainer(injection, sens_mat)
                d_d = float(np.real(analysis.inner_product()))
                snr = float(analysis.snr())
            except Exception as e:
                continue
            if SNR_MIN <= snr <= SNR_MAX:
                chosen = (params_i, wdm_inj_sig, analysis, d_d, snr)
                break

        if chosen is None:
            print(f"[warn] draw {i}: exhausted {MAX_REJECT} attempts", flush=True)
            continue
        params_i, wdm_inj_sig, analysis, d_d, snr = chosen

        template_full = np.zeros((3, Nf, Nt), dtype=float)
        chunked.fill_global(template_full, [tuple(params_i.tolist())], factors=None)
        tpl_active = template_full[:, wdm_set.ind_min_f: wdm_set.ind_max_f + 1, :]
        if wdm_set.Nt_active != wdm_set.Nt:
            tpl_active = tpl_active[:, :, wdm_set.active_slice_t]
        tpl_wdm = WDMSignal(tpl_active, wdm_set)

        log_like = analysis.template_likelihood(tpl_wdm)
        mismatch = analysis.template_inner_product(tpl_wdm, normalize=True)
        snr_list.append(snr)
        log_like_list.append(log_like)
        mismatch_list.append(mismatch)
        params_list.append(params_i)

        f_low = float(params_i[5])
        m_floor = int(f_low / layer_df)
        new_wdm_set = WDMSettings(
            wdm_set.Nf, wdm_set.Nt, wdm_set.data_dt,
            min_time=wdm_set.min_time, max_time=wdm_set.max_time,
            min_freq=f_low - 3 * layer_df,
            max_freq=f_low + 2 * layer_df,
            force_backend=backend,
        )
        wdm_inj_arr = np.asarray(wdm_inj_sig.arr)
        inj_band = WDMSignal(
            wdm_inj_arr[:,
                new_wdm_set.ind_min_f - wdm_set.ind_min_f
                : new_wdm_set.ind_max_f - wdm_set.ind_min_f + 1],
            new_wdm_set,
        )
        tpl_band = WDMSignal(
            tpl_active[:,
                new_wdm_set.ind_min_f - wdm_set.ind_min_f
                : new_wdm_set.ind_max_f - wdm_set.ind_min_f + 1],
            new_wdm_set,
        )
        analysis_5 = AnalysisContainer(
            DataResidualArray(inj_band),
            XYZ2SensitivityMatrix(new_wdm_set, model="scirdv1"),
        )
        mm5 = 1.0 - analysis_5.template_inner_product(
            DataResidualArray(tpl_band), normalize=True,
        )
        mismatch_5_layers_list.append(mm5)

        new_wdm_set_2 = WDMSettings(
            wdm_set.Nf, wdm_set.Nt, wdm_set.data_dt,
            min_time=wdm_set.min_time, max_time=wdm_set.max_time,
            min_freq=m_floor * layer_df,
            max_freq=(m_floor + 2) * layer_df - 0.5 * layer_df,
            force_backend=backend,
        )
        inj_band_2 = WDMSignal(
            wdm_inj_arr[:,
                new_wdm_set_2.ind_min_f - wdm_set.ind_min_f
                : new_wdm_set_2.ind_max_f - wdm_set.ind_min_f + 1],
            new_wdm_set_2,
        )
        tpl_band_2 = WDMSignal(
            tpl_active[:,
                new_wdm_set_2.ind_min_f - wdm_set.ind_min_f
                : new_wdm_set_2.ind_max_f - wdm_set.ind_min_f + 1],
            new_wdm_set_2,
        )
        analysis_2 = AnalysisContainer(
            DataResidualArray(inj_band_2),
            XYZ2SensitivityMatrix(new_wdm_set_2, model="scirdv1"),
        )
        mm2 = 1.0 - analysis_2.template_inner_product(
            DataResidualArray(tpl_band_2), normalize=True,
        )
        mismatch_2_layers_list.append(mm2)

        if (i + 1) % PROGRESS_EVERY == 0 or i + 1 == N_DRAWS:
            f_frac = (f_low - m_floor * layer_df) / layer_df
            elapsed = time.perf_counter() - t_loop_start
            print(
                f"  [{i + 1:4d}/{N_DRAWS}] snr={snr:7.2f}  log_like={log_like:.3e} "
                f"1-O={mismatch:.3e}  m={m_floor:4d} f_frac={f_frac:.3f}\n"
                f"     mm5={mm5:.3e}   mm2={mm2:.3e}   "
                f"({attempt_total} att, {(i + 1)/elapsed:.2f} draw/s, {elapsed:.1f}s)",
                flush=True,
            )

    # --- save + plots ----------------------------------------------------
    out_npz = f"{OUTPUT_PREFIX}_{int(time.time())}.npz"
    np.savez(
        out_npz,
        params=np.asarray(params_list),
        snr=np.asarray(snr_list),
        log_like=np.asarray(log_like_list),
        mismatch=np.asarray(mismatch_list),
        mismatch_5=np.asarray(mismatch_5_layers_list),
        mismatch_2=np.asarray(mismatch_2_layers_list),
        Nf=Nf, Nt=Nt, dt=dt, Nt_sub=Nt_sub, N_sparse=N_sparse, n_pad=n_pad,
    )
    print(f"[save] wrote {out_npz}", flush=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, arr, title in zip(
        axes,
        [mismatch_list, mismatch_5_layers_list, mismatch_2_layers_list],
        ["full-band", "5-layer", "2-layer"],
    ):
        arr = np.asarray(arr, dtype=float)
        arr = arr[np.isfinite(arr) & (arr > 0)]
        if len(arr) == 0:
            ax.set_title(title + " (no positive values)")
            continue
        ax.hist(np.log10(arr), bins=40, color="indianred", alpha=0.8)
        ax.set_xlabel("log10(1 - O)")
        ax.set_ylabel("count")
        ax.set_title(f"{title} (N={len(arr)}, median={np.median(arr):.2e})")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_PREFIX}_hist.png", dpi=120)
    plt.close(fig)
    print(f"[plot] wrote {OUTPUT_PREFIX}_hist.png", flush=True)

    if len(params_list) > 0:
        params_arr = np.asarray(params_list, dtype=float)
        f_low_arr = params_arr[:, 5]
        m_floor_arr = np.floor(f_low_arr / layer_df).astype(int)
        f_frac_arr = f_low_arr / layer_df - m_floor_arr
        mm5_arr = np.asarray(mismatch_5_layers_list, dtype=float)
        mm2_arr = np.asarray(mismatch_2_layers_list, dtype=float)
        snr_arr = np.asarray(snr_list, dtype=float)
        fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey="row")
        for ax, x, xlabel in [
            (axes[0, 0], f_low_arr * 1e3, "f_low (mHz)"),
            (axes[0, 1], f_low_arr * 1e3, "f_low (mHz)"),
            (axes[1, 0], f_frac_arr,      "f_frac (within layer)"),
            (axes[1, 1], f_frac_arr,      "f_frac (within layer)"),
        ]:
            ax.set_xlabel(xlabel)
        for ax, y, ylabel in [
            (axes[0, 0], mm5_arr, "1 - O (mm5)"),
            (axes[0, 1], mm2_arr, "1 - O (mm2)"),
            (axes[1, 0], mm5_arr, "1 - O (mm5)"),
            (axes[1, 1], mm2_arr, "1 - O (mm2)"),
        ]:
            ax.set_yscale("log")
            ax.set_ylabel(ylabel)
            ax.axhline(1e-9, color="red", lw=0.8, ls="--",
                        label="mm = 1e-9 (science threshold)")
        for ax, x, y in [
            (axes[0, 0], f_low_arr * 1e3, mm5_arr),
            (axes[0, 1], f_low_arr * 1e3, mm2_arr),
            (axes[1, 0], f_frac_arr,      mm5_arr),
            (axes[1, 1], f_frac_arr,      mm2_arr),
        ]:
            sc = ax.scatter(x, y, c=np.log10(np.maximum(snr_arr, 1e-3)),
                              cmap="viridis", s=18, alpha=0.85)
            ax.grid(True, alpha=0.3)
        cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), label="log10(SNR)",
                              shrink=0.85, pad=0.02)
        cbar.ax.tick_params(labelsize=9)
        fig.suptitle(f"{OUTPUT_PREFIX}  N={len(params_list)} draws  "
                     f"Nf={Nf} Nt_sub={Nt_sub} N_sparse={N_sparse}",
                     fontsize=11)
        fig.savefig(f"{OUTPUT_PREFIX}_f0_ffrac_mm.png", dpi=120)
        plt.close(fig)
        print(f"[plot] wrote {OUTPUT_PREFIX}_f0_ffrac_mm.png", flush=True)


if __name__ == "__main__":
    main()
