#!/usr/bin/env python
"""Prior-draw mismatch test of the WDM GB CHUNKED-HETERODYNE template.

Drop-in counterpart of ``gb_lookup_prior_draws.py`` that swaps the
WDM-lookup-table template step for the new chunked-heterodyne pipeline
in :mod:`gb_wdm_het` (validated to floating-point precision against the
lisatools WDM transform in ``check_shortened_wdm.py:test_K`` and
``test_L``).

For each draw:

  1. Draw an 8-D GB parameter from the same prior as
     ``gb_lookup_prior_draws.py``; transform to the 9-param physical
     vector.
  2. Build an accurate dense-TD waveform via ``GBTDIonTheFly`` and
     transform it to the WDM domain (injection).
  3. Build the WDM template via
     :meth:`fastlisaresponse.gbcomps.GBWDMComputations.fill_global_wdm`
     (chunked-heterodyne + Tukey-auto). NO lookup table.
  4. Compute ``log_like``, full-band, 5-layer, and 2-layer
     mismatches via ``AnalysisContainer.template_likelihood`` /
     ``template_inner_product`` -- identical to the lookup script so
     numbers are directly comparable.
  5. Save NPZ + diagnostic plots with the same column names as the
     lookup script.

Env-var knobs match the lookup script unless explicitly noted.

Run:
    python gb_chunked_prior_draws.py
"""
from __future__ import annotations

import os
import time

import numpy as np
import matplotlib
if not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import cupy as cp
except (ImportError, ModuleNotFoundError):
    cp = None

from lisatools.detector import ESAOrbits
from lisatools.utils.constants import YRSID_SI
from fastlisaresponse.tdiconfig import TDIConfig
from fastlisaresponse.tdionfly import GBTDIonTheFly

from lisatools.datacontainer import DataResidualArray
from lisatools.analysiscontainer import AnalysisContainer
from lisatools.sensitivity import XYZ2SensitivityMatrix
from lisatools.domains import (
    TDSettings, TDSignal, FDSettings, WDMSettings, WDMSignal,
)

from gb_lookup_prior_draws import build_gb_prior
from fastlisaresponse.gbcomps import GBWDMComputations


class _FullGridWDMHolder:
    """Minimal duck-type for :meth:`GBWDMComputations.get_ll_wdm`'s
    ``wdm_holder`` argument: exposes ``linear_data_arr[0]`` /
    ``linear_psd_arr[0]`` / ``__len__`` over full-grid ``(3, Nf, Nt)``
    data + diagonal invC buffers.
    """
    def __init__(self, data_full, invC_diag_full):
        self.linear_data_arr = [np.ascontiguousarray(data_full).ravel()]
        self.linear_psd_arr  = [np.ascontiguousarray(invC_diag_full).ravel()]

    def __len__(self):
        return 1


def _make_wdm_signal_slice(full_wdm_arr, parent_set, min_freq, max_freq,
                           min_time, max_time, backend):
    """Slice ``full_wdm_arr`` onto a band/time-cropped ``WDMSettings``."""
    band_set = WDMSettings(
        parent_set.Nf, parent_set.Nt, parent_set.data_dt,
        min_freq=min_freq, max_freq=max_freq,
        min_time=min_time, max_time=max_time,
        force_backend=backend,
    )
    arr_band = full_wdm_arr[
        :,
        band_set.ind_min_f - parent_set.ind_min_f
        : band_set.ind_max_f - parent_set.ind_min_f + 1,
    ]
    return WDMSignal(arr_band, band_set), band_set


def main():
    backend = os.environ.get("CHUNKED_BACKEND", "cpu")
    xp = np if backend == "cpu" else cp

    # --- config (matches gb_lookup_prior_draws.py defaults) ----------------
    N_DRAWS = int(os.environ.get("N_DRAWS", 50))
    SNR_MIN = float(os.environ.get("SNR_MIN", 5.0))
    SNR_MAX = float(os.environ.get("SNR_MAX", 1100.0))
    N_INJ = int(os.environ.get("N_INJ", 16384))
    MAX_REJECT = int(os.environ.get("MAX_REJECT", 500))
    SEED = int(os.environ.get("SEED", 12345))
    OUTPUT_PREFIX = os.environ.get("OUTPUT_PREFIX", "gb_prior_chunked_test")
    PROGRESS_EVERY = int(os.environ.get("PROGRESS_EVERY", 1))
    SAVE_HEATMAPS_N = int(os.environ.get("SAVE_HEATMAPS_N", 0))
    # Number of draws for which to compare lisatools.calculate_signal_likelihood
    # (source_only=True) vs chunked-het cpp.get_ll across a small ladder of
    # parameter perturbations (slope-1 scatter test). 0 = disabled.
    LIKELIHOOD_COMPARE_N = int(os.environ.get("LIKELIHOOD_COMPARE_N", 0))

    # --- chunked-heterodyne knobs ------------------------------------------
    Nt_sub = int(os.environ.get("NT_SUB",  256))
    N_sparse = int(os.environ.get("N_SPARSE", 256))
    # n_pad: per-chunk edge discard (in WDM pixels). Recommended fraction is
    # 1/8 of Nt_sub. Phase 1's joint sweep showed mm5/mm2 < 1e-9 at this
    # default with Tukey-auto.
    n_pad = int(os.environ.get("N_PAD", Nt_sub // 8))

    np.random.seed(SEED)

    # --- detector / WDM grid (same as the lookup script) ------------------
    orbits = ESAOrbits(force_backend=backend)
    dt = 10.0
    Nf = int(os.environ.get("NF", 1460))
    Nt = int(os.environ.get("NT", 256 * 10 * (1460 // Nf if Nf <= 1460 else 1)))
    print(f"[run] Nf={Nf} Nt={Nt} dt={dt}  (Tobs={Nf*Nt*dt:.3e}s, "
          f"Nt_sub={Nt_sub}, N_sparse={N_sparse}, n_pad={n_pad})", flush=True)
    wavelet_duration = Nf * dt
    Tobs = Nt * wavelet_duration
    Nobs = Nf * Nt

    tdi_config = TDIConfig("2nd generation")
    t_start = int(0.5 * YRSID_SI / dt) * dt              # 6 months in
    t_arr = np.arange(Nobs) * dt + t_start
    t_ref = t_start

    gb_tdi_kwargs = dict(
        tdi_config=tdi_config, orbits=orbits,
        tdi_chan="XYZ", force_backend=backend,
    )
    t_tdi_inj = xp.linspace(t_arr[0], t_arr[-1], N_INJ)
    gb_gen_inj = GBTDIonTheFly(
        t_tdi_inj, Tobs, t_ref, 1.0 / dt, 1, **gb_tdi_kwargs,
    )

    N = t_arr.shape[-1]
    td_set = TDSettings(N, dt, force_backend=backend)
    freqs = np.fft.rfftfreq(N, dt)
    df = freqs[1] - freqs[0]
    N_fd = len(freqs)

    # WDM active band. We need ind_min_f = 0 to support priors down to
    # f0 ~ 1 mHz (mm5 band runs from m_floor-3 -> needs layer 0 at m_floor=2).
    min_freq = float(os.environ.get("MIN_FREQ_HZ", 0.0))
    max_freq = float(os.environ.get("MAX_FREQ_HZ", 35.0e-3))
    _ = FDSettings(N_fd, df, min_freq=min_freq, max_freq=max_freq,
                   force_backend=backend)
    min_time = 20 * wavelet_duration
    max_time = (Nt - 20) * wavelet_duration

    wdm_set = WDMSettings(
        Nf, Nt, dt, t0=t_start,
        min_freq=min_freq, max_freq=max_freq,
        min_time=min_time, max_time=max_time,
        force_backend=backend,
    )

    # --- chunked-heterodyne computations (template + likelihood) ----------
    # ``GBWDMComputations`` is the FastLISAResponseParallelModule whose
    # ``fill_global_wdm`` / ``get_ll_wdm`` route through
    # ``GBComputationGroupWrap.gb_wdm_het_*`` on the chosen backend
    # (C++ on cpu/cuda, pure-JAX on the ``jax`` backend).
    chunked = GBWDMComputations(
        wdm_set, t_ref=t_ref,
        Nt_sub=Nt_sub, n_pad=n_pad, N_sparse=N_sparse,
        # Cache knobs (default 0 = direct; override via env to engage cache).
        N_cp_sig   = int(os.environ.get("N_CP_SIG",   0)),
        N_cp_orbit = int(os.environ.get("N_CP_ORBIT", 0)),
        orbits=orbits,                                  # MUST match injection
        tdi_config="2nd generation",
        force_backend=backend,
        d_d=0.0,                                        # source-only return
        tdi_type="XYZ",
    )
    print(f"[run] chunked: n_chunks={chunked.n_chunks}, "
          f"T_chunk={chunked.T_chunk:.3e}s, "
          f"alpha={chunked.resolved_tukey_alpha} "
          f"(use_tukey={chunked.use_tukey})", flush=True)

    # --- prior (same as the lookup script) --------------------------------
    layer_df = wdm_set.layer_df
    buffer_layers = 7                                    # ~5 main + 2 margin
    f0_lo_default = (wdm_set.ind_min_f + buffer_layers) * layer_df
    f0_hi_default = (wdm_set.ind_max_f - buffer_layers) * layer_df
    f0_lo_hz = float(os.environ.get("F0_LO_HZ", f0_lo_default))
    f0_hi_hz = float(os.environ.get("F0_HI_HZ", f0_hi_default))
    fdot_max = float(os.environ.get("FDOT_MAX", 1e-15))
    A_lims = (float(os.environ.get("A_LO", 1e-23)),
              float(os.environ.get("A_HI", 1e-20)))

    beta_env = os.environ.get("BETA_LIMS", "")
    if beta_env:
        beta_lims = tuple(float(s) for s in beta_env.split(","))
    else:
        beta_lims = None

    prior, tc = build_gb_prior(
        A_lims=A_lims, f0_lims_hz=(f0_lo_hz, f0_hi_hz),
        fdot_lims=(-fdot_max, fdot_max), beta_lims=beta_lims,
    )

    print(f"[run] N_DRAWS={N_DRAWS} SNR window=[{SNR_MIN}, {SNR_MAX}]", flush=True)
    print(f"[run] f0 range = [{f0_lo_hz*1e3:.4f}, {f0_hi_hz*1e3:.4f}] mHz "
          f"(layer_df = {layer_df:.3e} Hz)", flush=True)

    # --- signal_gen wrapper: params (9,) -> WDMSignal --------------------
    # Closes over gb_gen_inj + td_set + wdm_set so the analysis container
    # can drop it in as `signal_gen`. Returns a WDMSignal directly so
    # AnalysisContainer can wrap it in a DataResidualArray.
    def _chunked_signal_gen(*params, **kwargs):
        params_arr = np.asarray(params, dtype=float).reshape(9, 1)
        inj_spline = gb_gen_inj(
            *params_arr, convert_to_ra_dec=False, return_spline=True,
        )
        td = np.asarray(inj_spline.eval_tdi(t_arr))[0]
        return TDSignal(td, settings=td_set).transform(wdm_set, window=None)

    # --- per-draw loop ----------------------------------------------------
    sens_mat = None
    snr_list, log_like_list, mismatch_list = [], [], []
    log_like_5_layers_list, mismatch_5_layers_list = [], []
    log_like_2_layers_list, mismatch_2_layers_list = [], []
    params_list = []
    # logL slope-1 scatter (lisatools vs chunked-het cpp.get_ll)
    ll_lisatools_list = []
    ll_cpp_list = []
    attempt_total = 0
    t_loop_start = time.perf_counter()

    for i in range(N_DRAWS):
        chosen = None
        for _ in range(MAX_REJECT):
            attempt_total += 1
            x_sampled = prior.rvs(size=1)
            params_i = tc.both_transforms(x_sampled.copy())[0]

            inj_spline = gb_gen_inj(
                *params_i.reshape(9, 1),
                convert_to_ra_dec=False, return_spline=True,
            )
            td_inj = np.asarray(inj_spline.eval_tdi(t_arr))[0]
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
            if SNR_MIN <= snr <= SNR_MAX:
                chosen = (params_i, wdm_inj_sig, analysis, d_d, snr)
                break

        if chosen is None:
            print(f"[warn] draw {i}: exhausted {MAX_REJECT} attempts; "
                  f"keeping last (snr={snr:.2f})", flush=True)
            chosen = (params_i, wdm_inj_sig, analysis, d_d, snr)
        params_i, wdm_inj_sig, analysis, d_d, snr = chosen

        # --- CHUNKED-HETERODYNE TEMPLATE (Phase 4 swap-in) ---------------
        # injection is on the full WDM grid (3, Nf_active, Nt_active).
        # The chunked generator builds the template on the same FULL grid
        # (3, Nf, Nt). For the full-band mismatch we have to mask the
        # template down to the active band to match the injection.
        template_full = np.zeros((3, Nf, Nt), dtype=float)
        chunked.fill_global_wdm(
            np.asarray(params_i, dtype=float).reshape(1, 9),
            template_full,
            convert_to_ra_dec=False,
            factors=None,
        )
        # crop to the active band that matches the injection's WDMSettings
        tpl_active = template_full[
            :, wdm_set.ind_min_f: wdm_set.ind_max_f + 1, :
        ]
        # Carry over the active-time slicing from wdm_set if any
        if wdm_set.Nt_active != wdm_set.Nt:
            tpl_active = tpl_active[:, :, wdm_set.active_slice_t]
        tpl_wdm = WDMSignal(tpl_active, wdm_set)

        # --- full-band log_like + mismatch (same as lookup script) -------
        log_like = analysis.template_likelihood(tpl_wdm)
        mismatch = analysis.template_inner_product(tpl_wdm, normalize=True)
        snr_list.append(snr)
        log_like_list.append(log_like)
        mismatch_list.append(mismatch)
        params_list.append(params_i)

        # --- 5-layer mismatch -------------------------------------------
        f0 = float(params_i[1])
        m_floor = int(f0 / layer_df)
        new_wdm_set = WDMSettings(
            wdm_set.Nf, wdm_set.Nt, wdm_set.data_dt,
            min_time=wdm_set.min_time, max_time=wdm_set.max_time,
            min_freq=f0 - 3 * wdm_set.layer_df,
            max_freq=f0 + 2 * wdm_set.layer_df,
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
        log_like_5 = analysis_5.template_likelihood(DataResidualArray(tpl_band))
        mm5 = 1.0 - analysis_5.template_inner_product(
            DataResidualArray(tpl_band), normalize=True,
        )
        log_like_5_layers_list.append(log_like_5)
        mismatch_5_layers_list.append(mm5)

        # --- 2-layer mismatch (m_floor, m_floor+1) -----------------------
        new_wdm_set_2 = WDMSettings(
            wdm_set.Nf, wdm_set.Nt, wdm_set.data_dt,
            min_time=wdm_set.min_time, max_time=wdm_set.max_time,
            min_freq=(m_floor - 0.5) * wdm_set.layer_df,
            max_freq=(m_floor + 1 + 0.5) * wdm_set.layer_df,
            force_backend=backend,
        )
        inj_2 = WDMSignal(
            wdm_inj_arr[:,
                new_wdm_set_2.ind_min_f - wdm_set.ind_min_f
                : new_wdm_set_2.ind_max_f - wdm_set.ind_min_f + 1],
            new_wdm_set_2,
        )
        tpl_2 = WDMSignal(
            tpl_active[:,
                new_wdm_set_2.ind_min_f - wdm_set.ind_min_f
                : new_wdm_set_2.ind_max_f - wdm_set.ind_min_f + 1],
            new_wdm_set_2,
        )
        analysis_2 = AnalysisContainer(
            DataResidualArray(inj_2),
            XYZ2SensitivityMatrix(new_wdm_set_2, model="scirdv1"),
        )
        log_like_2 = analysis_2.template_likelihood(DataResidualArray(tpl_2))
        mm2 = 1.0 - analysis_2.template_inner_product(
            DataResidualArray(tpl_2), normalize=True,
        )
        log_like_2_layers_list.append(log_like_2)
        mismatch_2_layers_list.append(mm2)

        # --- heatmap PNG for the first SAVE_HEATMAPS_N draws ------------
        if i < SAVE_HEATMAPS_N:
            inj_arr_5 = np.asarray(inj_band.arr)[0]      # (5_layers, Nt_active)
            tpl_arr_5 = np.asarray(tpl_band.arr)[0]
            diff_5    = tpl_arr_5 - inj_arr_5
            vmax = max(np.abs(inj_arr_5).max(),
                       np.abs(tpl_arr_5).max(), 1e-30)
            vmin = -vmax
            layers = np.arange(m_floor - 3, m_floor + 3)[:inj_arr_5.shape[0]]
            t_days = np.arange(inj_arr_5.shape[1]) * (dt * Nf) / 86400.0
            fig, axes = plt.subplots(1, 3, figsize=(15, 3.8), sharey=True)
            for ax, arr, ttl in zip(
                axes, (inj_arr_5, tpl_arr_5, diff_5),
                ("injection (WDM)",
                 "chunked-het template",
                 "template - injection"),
            ):
                im = ax.pcolormesh(t_days, layers, arr,
                                    shading="auto", cmap="RdBu_r",
                                    vmin=vmin, vmax=vmax)
                ax.set_xlabel("t [days]")
                ax.set_title(ttl)
            axes[0].set_ylabel(f"m-layer  (m_floor={m_floor})")
            fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02,
                          label="WDM coeff (channel 0)")
            f_frac_hm = (f0 - m_floor * layer_df) / layer_df
            fig.suptitle(
                f"draw {i+1}/{SAVE_HEATMAPS_N}  "
                f"SNR={snr:.1f}  logL={log_like:+.3e}  "
                f"mm5={mm5:.2e}  mm2={mm2:.2e}\n"
                f"f0={f0*1e3:.4f} mHz  m_floor={m_floor}  f_frac={f_frac_hm:.2f}  "
                f"beta={float(params_i[8]):+.3f}rad  "
                f"lam={float(params_i[7]):.3f}rad  "
                f"psi={float(params_i[6]):.3f}rad  "
                f"inc={float(params_i[5]):.3f}rad",
                fontsize=9,
            )
            fig.tight_layout(rect=(0, 0, 1, 0.92))
            hm_png = f"{OUTPUT_PREFIX}_heatmap_draw{i+1:02d}.png"
            fig.savefig(hm_png, dpi=110)
            plt.close(fig)
            print(f"     wrote {hm_png}", flush=True)

        # --- lisatools.calculate_signal_likelihood vs cpp.get_ll ---------
        # For the first LIKELIHOOD_COMPARE_N draws, evaluate logL at the
        # injection plus a small ladder of parameter perturbations and
        # collect (lisatools, cpp) pairs for a slope-1 scatter plot.
        #
        # The injection + sens_mat live on the WDM active band (Nf_active,
        # Nt_active). The chunked-het kernel needs them on the FULL grid
        # (Nf, Nt) -- we zero-fill outside the active band. The inner
        # product is unchanged because both d and h are zero off-band.
        if i < LIKELIHOOD_COMPARE_N:
            # Attach signal_gen for lisatools path.
            analysis.signal_gen = _chunked_signal_gen
            # The chunked-het kernel currently iterates m in [0, Nf) on
            # the full WDM grid, so we zero-fill the active-band injection
            # and inverse-PSD onto (3, Nf, Nt). FUTURE OPT (per design
            # note): teach the kernel to iterate only [ind_min_f,
            # ind_max_f] and accept active-band arrays directly --
            # cuts ~50% of the per-pixel work on this Nf=1460 config.
            nch = 3
            inj_active = np.asarray(wdm_inj_sig.arr)
            psd_active = np.asarray(sens_mat.sens_mat)
            # sens_mat may be cross-channel (3, 3, Nf_a, Nt_a) -- take the
            # diagonal (per-channel PSD) for the chunked-het kernel which
            # only consumes a diagonal invC (3, Nf, Nt). FUTURE OPT: pipe
            # the cross-channel PSD through the kernel.
            if psd_active.ndim == 4:
                psd_diag = np.stack([psd_active[c, c] for c in range(nch)], axis=0)
            else:
                psd_diag = psd_active
            # Sanitize PSD: lisatools' sens model divides by f, so f=0
            # produces inf/NaN. Treat inf/NaN/<=0 as "infinite PSD"
            # (invC=0), which mirrors lisatools' template_likelihood
            # behavior of skipping those pixels.
            with np.errstate(divide="ignore", invalid="ignore"):
                invC_active = 1.0 / np.where(
                    np.isfinite(psd_diag) & (psd_diag > 0),
                    psd_diag, np.inf,
                )
            invC_active = np.where(np.isfinite(invC_active),
                                    invC_active, 0.0)

            data_d_full = np.zeros((nch, Nf, Nt), dtype=float)
            invC_full   = np.zeros_like(data_d_full)
            ilo = wdm_set.ind_min_f
            ihi = wdm_set.ind_max_f + 1
            if wdm_set.Nt_active == wdm_set.Nt:
                data_d_full[:, ilo:ihi, :] = inj_active
                invC_full  [:, ilo:ihi, :] = invC_active
            else:
                tslice = wdm_set.active_slice_t
                data_d_full[:, ilo:ihi, tslice] = inj_active
                invC_full  [:, ilo:ihi, tslice] = invC_active
            # Wrap as the duck-type ``get_ll_wdm`` expects.
            holder_full = _FullGridWDMHolder(data_d_full, invC_full)

            # lisatools' inner_product is 4 * differential_component *
            # sum(sig1.conj * sig2 / PSD); the chunked-het kernel just
            # returns the bare sum. Calibrate the scale empirically from
            # the known <d|d> value (analysis.inner_product()):
            #   d_d_lt = 4 * dc * sum(d*d*invC)  =>  scale = d_d_lt / sum_dd
            sum_dd_chunked = float(np.sum(data_d_full * data_d_full * invC_full))
            ll_scale = d_d / max(sum_dd_chunked, 1e-300)

            #          [A,     f0,    fdot,   fddot,  phi0, inc,  psi,  lam,  beta]
            pert_unit = np.array([
                params_i[0] * 1e-3,   # A: 1e-3 fractional
                params_i[1] * 1e-7,   # f0: 1e-7 fractional (~ sub-bin)
                1e-19,                 # fdot
                1e-23,                 # fddot
                1e-2, 1e-2, 1e-2, 1e-2, 1e-2,  # angles: ~1e-2 rad
            ])
            rng_pert = np.random.default_rng(SEED + i)

            # Pick a random direction (unit vector in pert_unit space).
            direction = rng_pert.standard_normal(9)
            direction = direction / np.linalg.norm(direction)

            # Quick calibration via cpp.get_ll: pick a small alpha,
            # measure resulting dlogL, then solve for alphas that hit
            # target dlogL ~ 10/100/1000/10000 (quadratic scaling).
            alpha_cal = max(1.0 / max(snr, 1.0), 1e-3)
            p_cal = params_i + alpha_cal * pert_unit * direction
            chunked.get_ll_wdm(
                np.asarray(p_cal, dtype=float).reshape(1, 9),
                holder_full,
                convert_to_ra_dec=False,
            )
            dh_cal = chunked.d_h_out
            hh_cal = chunked.h_h_out
            ll_cal = (ll_scale * float(dh_cal[0] - 0.5 * hh_cal[0])
                       - 0.5 * d_d)
            dll_cal = abs(ll_cal)  # vs injection ll=0
            if dll_cal <= 0 or not np.isfinite(dll_cal):
                print(f"     [warn] calibration failed (dll_cal={dll_cal}), "
                      f"skipping logL-compare for this draw", flush=True)
            else:
                target_dlogLs = [10.0, 100.0, 1000.0, 10000.0]
                alphas = [alpha_cal * float(np.sqrt(t / dll_cal))
                           for t in target_dlogLs]
                eval_alphas = [0.0] + alphas        # injection + 4 targets

                ll_lt_local, ll_cpp_local = [], []
                for alpha in eval_alphas:
                    if alpha == 0.0:
                        p = params_i
                    else:
                        p = params_i + alpha * pert_unit * direction
                    try:
                        ll_lt = float(np.real(
                            analysis.calculate_signal_likelihood(
                                *p, source_only=True,
                            )
                        ))
                    except Exception as e:
                        print(f"     [warn] lisatools logL failed at "
                              f"alpha={alpha:.3e}: {e}", flush=True)
                        continue
                    chunked.get_ll_wdm(
                        np.asarray(p, dtype=float).reshape(1, 9),
                        holder_full,
                        convert_to_ra_dec=False,
                    )
                    dh_cpp = chunked.d_h_out
                    hh_cpp = chunked.h_h_out
                    dh_cpp_lt = ll_scale * float(dh_cpp[0])
                    hh_cpp_lt = ll_scale * float(hh_cpp[0])
                    ll_cpp = -0.5 * (d_d + hh_cpp_lt - 2.0 * dh_cpp_lt)
                    ll_lt_local.append(ll_lt)
                    ll_cpp_local.append(ll_cpp)
                ll_lisatools_list.extend(ll_lt_local)
                ll_cpp_list.extend(ll_cpp_local)
                tgt_str = "[inj] " + " ".join(
                    f"~{int(t)}" for t in target_dlogLs
                )
                print(f"     [logL-compare] targets={tgt_str}  "
                      f"lt={[f'{v:+.3e}' for v in ll_lt_local]}  "
                      f"cpp={[f'{v:+.3e}' for v in ll_cpp_local]}",
                      flush=True)

        if (i + 1) % PROGRESS_EVERY == 0 or i == 0:
            elapsed = time.perf_counter() - t_loop_start
            rate = (i + 1) / max(elapsed, 1e-9)
            f_frac = (f0 - m_floor * layer_df) / layer_df
            print(
                f"  [{i+1:4d}/{N_DRAWS}] snr={snr:7.2f}  logL={log_like:+.3e} "
                f"1-O={mismatch:.3e}  m={m_floor:4d} f_frac={f_frac:.3f}\n"
                f"     mm5={mm5:.3e}   mm2={mm2:.3e}   "
                f"({attempt_total} att, {rate:.2f} draw/s, {elapsed:.1f}s)",
                flush=True,
            )

    # --- save NPZ + summary plots -----------------------------------------
    out_npz = f"{OUTPUT_PREFIX}_{int(time.time())}.npz"
    np.savez(
        out_npz,
        params=np.asarray(params_list),
        snr=np.asarray(snr_list),
        log_like=np.asarray(log_like_list),
        mismatch=np.asarray(mismatch_list),
        log_like_5=np.asarray(log_like_5_layers_list),
        mismatch_5=np.asarray(mismatch_5_layers_list),
        log_like_2=np.asarray(log_like_2_layers_list),
        mismatch_2=np.asarray(mismatch_2_layers_list),
        ll_lisatools=np.asarray(ll_lisatools_list),
        ll_cpp=np.asarray(ll_cpp_list),
        Nf=Nf, Nt=Nt, dt=dt, Nt_sub=Nt_sub, N_sparse=N_sparse, n_pad=n_pad,
    )
    print(f"[save] wrote {out_npz}", flush=True)

    # log-histograms of the three mismatches
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
        ax.hist(np.log10(arr), bins=40, color="steelblue", alpha=0.8)
        ax.set_xlabel("log10(1 - O)")
        ax.set_ylabel("count")
        ax.set_title(f"{title} (N={len(arr)}, median={np.median(arr):.2e})")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_PREFIX}_hist.png", dpi=120)
    plt.close(fig)
    print(f"[plot] wrote {OUTPUT_PREFIX}_hist.png", flush=True)

    # f0 / f_frac vs mm scatter (2x2: rows = (f0, f_frac), cols = (mm5, mm2)).
    if len(params_list) > 0:
        params_arr = np.asarray(params_list, dtype=float)
        f0_arr = params_arr[:, 1]
        m_floor_arr = np.floor(f0_arr / layer_df).astype(int)
        f_frac_arr = f0_arr / layer_df - m_floor_arr
        mm5_arr = np.asarray(mismatch_5_layers_list, dtype=float)
        mm2_arr = np.asarray(mismatch_2_layers_list, dtype=float)
        snr_arr = np.asarray(snr_list, dtype=float)
        fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey="row")
        for ax, x, xlabel in [
            (axes[0, 0], f0_arr * 1e3, "f0 (mHz)"),
            (axes[0, 1], f0_arr * 1e3, "f0 (mHz)"),
            (axes[1, 0], f_frac_arr,   "f_frac (within layer)"),
            (axes[1, 1], f_frac_arr,   "f_frac (within layer)"),
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
        # Plot points colored by log10(SNR).
        for ax, x, y in [
            (axes[0, 0], f0_arr * 1e3, mm5_arr),
            (axes[0, 1], f0_arr * 1e3, mm2_arr),
            (axes[1, 0], f_frac_arr,   mm5_arr),
            (axes[1, 1], f_frac_arr,   mm2_arr),
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

    # --- log-likelihood slope-1 scatter (lisatools vs chunked-het C++) ----
    if len(ll_lisatools_list) >= 2:
        ll_lt = np.asarray(ll_lisatools_list)
        ll_cpp = np.asarray(ll_cpp_list)
        fig, ax = plt.subplots(1, 1, figsize=(6.5, 6))
        ax.scatter(ll_lt, ll_cpp, s=14, alpha=0.6, c="steelblue",
                    edgecolors="none")
        # diagonal
        lo = float(min(ll_lt.min(), ll_cpp.min()))
        hi = float(max(ll_lt.max(), ll_cpp.max()))
        margin = 0.05 * (hi - lo) if hi > lo else 1.0
        diag = np.array([lo - margin, hi + margin])
        ax.plot(diag, diag, "k--", lw=1, alpha=0.6, label="slope-1")
        # residual stats
        resid = ll_cpp - ll_lt
        denom = np.maximum(np.abs(ll_lt), np.abs(ll_cpp)) + 1e-300
        rel = resid / denom
        rmse = float(np.sqrt(np.mean(resid**2)))
        med_abs = float(np.median(np.abs(resid)))
        ax.set_xlabel("log-likelihood -- lisatools calculate_signal_likelihood "
                       "(source_only=True)")
        ax.set_ylabel("log-likelihood -- chunked-het cpp.get_ll  "
                       "(<d|h> - 0.5<h|h>)")
        ax.set_title(f"{OUTPUT_PREFIX}   N={len(ll_lt)}   "
                      f"RMSE={rmse:.2e}  med|resid|={med_abs:.2e}  "
                      f"max|rel|={float(np.max(np.abs(rel))):.2e}")
        ax.legend(loc="upper left")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{OUTPUT_PREFIX}_ll_scatter.png", dpi=120)
        plt.close(fig)
        print(f"[plot] wrote {OUTPUT_PREFIX}_ll_scatter.png  "
              f"(N={len(ll_lt)}, RMSE={rmse:.2e})", flush=True)


if __name__ == "__main__":
    main()
