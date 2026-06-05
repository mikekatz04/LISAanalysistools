#!/usr/bin/env python
"""Prior-draw mismatch/likelihood test of the WDM GB lookup table.

For each draw from an erebor-style GB ``ProbDistContainer`` prior:

  1. draw 8 sampled params, transform via ``TransformContainer`` to the
     9-param physical vector  [A, f0, fdot, fddot=0, phi0, inc, psi, lam, beta],
  2. build an *accurate* TD waveform with ``GBTDIonTheFly`` (high ``N_INJ``),
  3. transform that TD waveform to the WDM domain — this is the injection,
  4. build the WDM template at the same params via
     ``GBWDMComputations.fill_global_wdm`` (the same C/CUDA path the global
     fit uses),
  5. record the WDM log-likelihood (full active band) and the noise-weighted
     mismatch ``1 - <d|h>/sqrt(<d|d><h|h>)`` restricted to the frequency
     layers where the lookup template has power (the GB-only band).

All inner products are routed through ``lisatools.diagnostic.inner_product``
(directly, or through ``AnalysisContainer.{inner_product,template_inner_product}``).
The GB-layer-restricted mismatch is obtained by zeroing the WDM injection
outside the GB layers and reusing ``template_inner_product(..., normalize=True)``;
the lookup template is already zero outside those layers, so contributions
from non-GB layers drop out of every <d|h>, <d|d>, <h|h> term.

Sources whose WDM SNR falls outside ``[SNR_MIN, SNR_MAX]`` are redrawn.

Most knobs are env-var driven so it can be re-run without editing the file::

    N_DRAWS=200  python gb_lookup_prior_draws.py
    N_DRAWS=1000 SNR_MIN=5 SNR_MAX=1100 python gb_lookup_prior_draws.py
"""

import os
import time

import numpy as np
import matplotlib
if not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal

try:
    import cupy as cp
except (ImportError, ModuleNotFoundError):
    pass

from lisatools.detector import ESAOrbits
from lisatools.utils.constants import YRSID_SI
from fastlisaresponse.tdiconfig import TDIConfig
from fastlisaresponse.tdionfly import GBTDIonTheFly
from fastlisaresponse.gbcomps import GBWDMComputations

from lisatools.datacontainer import DataResidualArray
from lisatools.analysiscontainer import AnalysisContainer, AnalysisContainerArray
from lisatools.sensitivity import XYZ2SensitivityMatrix
from lisatools.domains import (
    TDSettings, TDSignal, FDSettings, WDMSettings, WDMSignal, WDMLookupTable,
)

from eryn.prior import ProbDistContainer, uniform_dist
from eryn.utils import TransformContainer


# ---------------------------------------------------------------------------
# Prior (erebor-style ProbDistContainer + TransformContainer)
# ---------------------------------------------------------------------------
def build_gb_prior(*, A_lims, f0_lims_hz, fdot_lims, beta_lims=None):
    """Build (ProbDistContainer, TransformContainer) matching erebor's GB setup.

    Only the (A, f0, fdot) ranges are adjustable from the outside — the
    angular priors are fixed to their full physical ranges (uniform on
    sphere / circle as appropriate).

    Sampled basis (8 params, in this order):
        ["A", "f0", "fdot", "phi0", "cos_iota", "psi", "lam", "sin_beta"]

    Output basis (9 params, in this order — matches the GBTDIonTheFly call
    signature  ``(amp, f0, fdot, fddot, phi0, inc, psi, lam, beta)``):
        ["A", "f0", "fdot", "fddot", "phi0", "cos_iota", "psi", "lam", "sin_beta"]
    with transforms:
        A: exp,  f0: mHz->Hz,  cos_iota: arccos,  sin_beta: arcsin,  fddot: 0.
    """
    sampled_basis = ["A", "f0", "fdot", "phi0", "cos_iota", "psi", "lam", "sin_beta"]
    full_basis = ["A", "f0", "fdot", "fddot", "phi0", "cos_iota", "psi", "lam", "sin_beta"]

    parameter_transforms = {
        "A": np.exp,
        "f0": (lambda x: x * 1e-3),
        "cos_iota": np.arccos,
        "sin_beta": np.arcsin,
    }
    tc = TransformContainer(
        input_basis=sampled_basis,
        output_basis=full_basis,
        parameter_transforms=parameter_transforms,
        fill_dict={"fddot": 0.0},
    )

    # Fixed full-range angular priors (matches erebor with delta_safe = 1e-5).
    delta_safe = 1e-5
    iota_lims = (0.0 + delta_safe, np.pi - delta_safe)
    if beta_lims is None:
        beta_lims = (-0.05, 0.05)  # (-np.pi / 2.0 + delta_safe, np.pi / 2.0 - delta_safe)

    priors_gb = {
        "A":        uniform_dist(*(np.log(np.asarray(A_lims)))),
        "f0":       uniform_dist(f0_lims_hz[0] * 1e3, f0_lims_hz[1] * 1e3),
        "fdot":     uniform_dist(fdot_lims[0], fdot_lims[1]),
        "phi0":     uniform_dist(0.0, 2.0 * np.pi),
        "cos_iota": uniform_dist(*np.cos(iota_lims)),
        "psi":      uniform_dist(0.0, np.pi),
        "lam":      uniform_dist(0.0, 2.0 * np.pi),
        "sin_beta": uniform_dist(*np.sin(beta_lims)),
    }
    prior = ProbDistContainer(priors_gb)
    return prior, tc


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    backend = os.environ.get("LOOKUP_BACKEND", "cpu")
    xp = np if backend == "cpu" else cp

    # ---- configuration ----------------------------------------------------
    N_DRAWS = int(os.environ.get("N_DRAWS", 1000))
    SNR_MIN = float(os.environ.get("SNR_MIN", 5.0))
    SNR_MAX = float(os.environ.get("SNR_MAX", 1100.0))
    N_INJ = int(os.environ.get("N_INJ", 16384))
    NUM_LAYERS_DIFF = int(os.environ.get("NUM_LAYERS_DIFF", 5))
    EPS_FREQ = float(os.environ.get("EPS_FREQ", 0.001))
    MAX_REJECT = int(os.environ.get("MAX_REJECT", 500))
    SEED = int(os.environ.get("SEED", 12345))
    OUTPUT_PREFIX = os.environ.get("OUTPUT_PREFIX", "gb_prior_lookup_test")
    PROGRESS_EVERY = int(os.environ.get("PROGRESS_EVERY", 1))

    # eryn's uniform_dist uses np.random under the hood
    np.random.seed(SEED)

    # ---- detector / domain setup ----------------------------------------
    orbits = ESAOrbits(force_backend=backend)
    dt = 10.0
    Nf = int(os.environ.get("NF", 1460))
    Nt = int(os.environ.get("NT", 256 * 10 * (1460 // Nf if Nf <= 1460 else 1)))
    print(f"[run] Nf={Nf} Nt={Nt} dt={dt} (wavelet_duration={Nf*dt:.1f}s, Tobs={Nf*Nt*dt:.3e}s)", flush=True)
    wavelet_duration = Nf * dt
    Tobs = Nt * wavelet_duration
    Nobs = Nf * Nt

    tdi_config = TDIConfig("2nd generation")

    t_start = int(0.5 * YRSID_SI / dt) * dt  # 6 months in
    t_arr = np.arange(Nobs) * dt + t_start
    t_ref = t_start

    gb_tdi_kwargs = dict(
        tdi_config=tdi_config,
        orbits=orbits,
        tdi_chan="XYZ",
        force_backend=backend,
    )
    t_tdi_inj = xp.linspace(t_arr[0], t_arr[-1], N_INJ)
    gb_gen_inj = GBTDIonTheFly(
        t_tdi_inj, Tobs, t_ref, 1.0 / dt, 1, **gb_tdi_kwargs
    )

    N = t_arr.shape[-1]
    td_set = TDSettings(N, dt, force_backend=backend)
    freqs = np.fft.rfftfreq(N, dt)
    df = freqs[1] - freqs[0]
    N_fd = len(freqs)
    # Optional Tukey window. Set TUKEY_ALPHA env var to apply (default no window).
    # MUST match the alpha the lookup table was built with for apples-to-apples.
    _tukey_alpha_env = os.environ.get("TUKEY_ALPHA", "").strip()
    if _tukey_alpha_env and float(_tukey_alpha_env) > 0:
        from scipy.signal import windows as _sp_windows
        _tukey_alpha = float(_tukey_alpha_env)
        Nobs = Nf * Nt
        window = _sp_windows.tukey(Nobs, alpha=_tukey_alpha, sym=False)
        print(f"[run] TUKEY_ALPHA={_tukey_alpha}  (tapers "
              f"~{int(Nobs * _tukey_alpha / 2 / Nf):.0f} WDM pixels at each end)",
              flush=True)
    else:
        window = None

    min_freq = 0.0001          # 0.1 mHz
    max_freq = 35.0e-3         # 35  mHz
    _ = FDSettings(N_fd, df, min_freq=min_freq, max_freq=max_freq,
                   force_backend=backend)

    min_time = 20 * wavelet_duration
    max_time = (Nt - 20) * wavelet_duration

    wdm_set = WDMSettings(
        Nf, Nt, dt,
        min_freq=min_freq, max_freq=max_freq,
        min_time=min_time, max_time=max_time,
    )

    # ---- lookup table ---------------------------------------------------
    store_path = os.environ.get("LOOKUP_PATH", "wdm_lookup_parity_fix.h5")
    if os.path.exists(store_path):
        print(f"[lookup] loading existing table from {store_path}", flush=True)
        wdm_lookup_table = WDMLookupTable.from_file(store_path, force_backend=backend)
        _wdm_settings = WDMSettings(*wdm_lookup_table.args, **wdm_lookup_table.kwargs)
        if not _wdm_settings.eq_without_inds(wdm_set):
            raise ValueError(
                "WDM Settings are not equivalent to lookup table. Either adjust "
                "to lookup-table settings or regenerate the table."
            )
    else:
        print(f"[lookup] building new table at {store_path}", flush=True)
        # No tukey window — unwindowed transforms.
        td_window = None
        m_ref = int(3e-3 / wdm_set.layer_df)
        norm_freq_single_layer, m_diffs, _ = WDMLookupTable.apply_eps_frequency(
            EPS_FREQ, wdm_set, m_ref=m_ref, num_layers_diff=NUM_LAYERS_DIFF
        )
        fdot_vals = np.array([0.0])
        _time_layers_env = os.environ.get("TIME_LAYERS", "")
        _time_layers = int(_time_layers_env) if _time_layers_env else None
        wdm_lookup_table = WDMLookupTable(
            wdm_set, 3,
            norm_freq_single_layer=norm_freq_single_layer,
            m_diffs=m_diffs,
            fdot_vals=fdot_vals,
            m_ref=m_ref,
            batch_size_gen=int(os.environ.get("BATCH_SIZE_GEN", 5)),
            td_window=td_window,
            store_path=store_path,
            verbose=bool(int(os.environ.get("BUILD_VERBOSE", "0"))),
            time_layers=_time_layers,
        )

    gb_comps = GBWDMComputations(
        wdm_lookup_table, Tobs, t_ref, orbits=orbits,
        tdi_config=tdi_config, force_backend=backend,
    )

    # ---- prior ----------------------------------------------------------
    # Confine f0 to the WDM active band, with a small buffer so the lookup's
    # ±NUM_LAYERS_DIFF stencil never hits a band edge.
    layer_df = wdm_set.layer_df
    buffer_layers = NUM_LAYERS_DIFF + 2
    f0_lo_default = (wdm_set.ind_min_f + buffer_layers) * layer_df
    f0_hi_default = (wdm_set.ind_max_f - buffer_layers) * layer_df
    f0_lo_hz = float(os.environ.get("F0_LO_HZ", f0_lo_default))
    f0_hi_hz = float(os.environ.get("F0_HI_HZ", f0_hi_default))

    # The current lookup table carries only fdot=0, so default to a tight
    # ±FDOT_MAX; widen it to probe how mismatch degrades with fdot.
    fdot_max = float(os.environ.get("FDOT_MAX", 1e-15))
    A_lims = (float(os.environ.get("A_LO", 1e-23)),
              float(os.environ.get("A_HI", 1e-20)))

    beta_env = os.environ.get("BETA_LIMS", "")
    if beta_env:
        beta_lims = tuple(float(s) for s in beta_env.split(","))
    else:
        beta_lims = None

    prior, tc = build_gb_prior(
        A_lims=A_lims,
        f0_lims_hz=(f0_lo_hz, f0_hi_hz),
        fdot_lims=(-fdot_max, fdot_max),
        beta_lims=beta_lims,
    )
    if beta_lims is not None:
        print(f"[run] beta_lims = {beta_lims}", flush=True)

    print(f"[run] N_DRAWS={N_DRAWS}  SNR window=[{SNR_MIN}, {SNR_MAX}]", flush=True)
    print(f"[run] f0 range = [{f0_lo_hz*1e3:.4f}, {f0_hi_hz*1e3:.4f}] mHz "
          f"(layer_df = {layer_df:.3e} Hz)", flush=True)
    print(f"[run] log A   in [{np.log(A_lims[0]):.2f}, {np.log(A_lims[1]):.2f}]"
          f"   fdot in ±{fdot_max:.1e}", flush=True)

    # ---- per-draw loop --------------------------------------------------
    n_pix_active = int(np.prod(wdm_set.basis_shape_active))
    sens_mat = None

    snr_list = []
    log_like_list = []
    mismatch_list = []
    log_like_5_layers_list = []
    mismatch_5_layers_list = []
    log_like_2_layers_list = []
    mismatch_2_layers_list = []
    dd_list, dh_list, hh_list = [], [], []
    n_gb_layers_list = []
    params_list = []
    attempt_total = 0
    t_loop_start = time.perf_counter()

    for i in range(N_DRAWS):
        chosen = None
        try: 
            del analysis, injection
        except (NameError, UnboundLocalError) as e:
            pass

        for _ in range(MAX_REJECT):
            attempt_total += 1

            # eryn prior + transform → 9 physical params
            x_sampled = prior.rvs(size=1)                       # (1, 8)
            params_i = tc.both_transforms(x_sampled.copy())[0]  # (9,)

            # accurate TD waveform → WDM injection
            inj_spline = gb_gen_inj(
                *params_i.reshape(9, 1),
                convert_to_ra_dec=False,
                return_spline=True,
            )
            td_inj = np.asarray(inj_spline.eval_tdi(t_arr))[0]
            wdm_inj_sig = TDSignal(td_inj, settings=td_set).transform(
                wdm_set, window=window
            )
            injection = DataResidualArray(wdm_inj_sig)

            if sens_mat is None:
                sens_mat = XYZ2SensitivityMatrix(
                    injection.data_res_arr.settings, model="scirdv1"
                )

            analysis = AnalysisContainer(injection, sens_mat)
            # all inner products come from lisatools.diagnostic.inner_product
            d_d = float(np.real(analysis.inner_product()))
            snr = float(analysis.snr())
            if SNR_MIN <= snr <= SNR_MAX:
                chosen = (params_i, wdm_inj_sig, analysis, d_d, snr)
                break

        if chosen is None:
            print(f"[warn] draw {i}: exhausted {MAX_REJECT} attempts; keeping last "
                  f"(snr={snr:.2f})", flush=True)
            chosen = (params_i, wdm_inj_sig, analysis, d_d, snr)

        params_i, wdm_inj_sig, analysis, d_d, snr = chosen
        wdm_inj_arr = np.asarray(
            wdm_inj_sig.arr if hasattr(wdm_inj_sig, "arr") else wdm_inj_sig
        )

        # ---- WDM lookup template via lisa-on-gpu C/CUDA path ------------
        template_fill = xp.zeros(3 * n_pix_active, dtype=float)
        wdm_holder = AnalysisContainerArray([analysis])
        gb_comps.fill_global_wdm(
            template_fill,
            params_i.reshape(1, 9),
            wdm_holder,
            data_index=None,
            convert_to_ra_dec=False,
        )
        tpl_arr = np.asarray(
            template_fill.reshape((3,) + wdm_set.basis_shape_active)
        )
        tpl_wdm = WDMSignal(tpl_arr, wdm_set)

        # ---- full-band inner products → log-likelihood ------------------
        # (all via lisatools: AnalysisContainer wraps diagnostic.inner_product)
        log_like = analysis.template_likelihood(tpl_wdm)
        mismatch = analysis.template_inner_product(tpl_wdm, normalize=True)
        
        snr_list.append(snr)
        log_like_list.append(log_like)
        mismatch_list.append(mismatch)
        params_list.append(params_i)

        new_wdm_set = WDMSettings(
            wdm_set.Nf, wdm_set.Nt, wdm_set.data_dt, min_time=wdm_set.min_time, max_time=wdm_set.max_time,
            min_freq = params_i[1] - 3 * wdm_set.layer_df,
            max_freq = params_i[1] + 2 * wdm_set.layer_df,
        )

        inj_here = DataResidualArray(WDMSignal(injection[:, new_wdm_set.ind_min_f - wdm_set.ind_min_f: new_wdm_set.ind_max_f - wdm_set.ind_min_f + 1], new_wdm_set))
        template_here = DataResidualArray(WDMSignal(tpl_wdm[:, new_wdm_set.ind_min_f - wdm_set.ind_min_f: new_wdm_set.ind_max_f - wdm_set.ind_min_f + 1], new_wdm_set))
        analysis_here = AnalysisContainer(inj_here, XYZ2SensitivityMatrix(new_wdm_set, model="scirdv1"))
        
        log_like_5_layers = analysis_here.template_likelihood(template_here)
        mismatch_5_layers = 1.0 - analysis_here.template_inner_product(template_here, normalize=True)

        log_like_5_layers_list.append(log_like_5_layers)
        mismatch_5_layers_list.append(mismatch_5_layers)

        # ---- mismatch over the source's 2 DOMINANT layers.
        # For any f0 in (m_floor*df, (m_floor+1)*df), the two layer centres
        # bracketing f0 are m_floor and m_floor+1 — they're always the
        # two closest centres regardless of f_frac. The pair only switches
        # when m_floor itself ticks over (i.e. f_frac wraps through 0).
        #
        # Previous formulation used [f0, f0+df] which, after WDMSettings'
        # ceil(min_freq/df) / int(max_freq/df) snapping, ended up covering
        # ONE layer (m_floor+1) for any f_frac > 0 — so ~17% of sources
        # got mm2 ~ 1.0 not because of a real lookup error but because
        # the band missed the wavelet main lobe entirely.
        _mfloor = int(params_i[1] / wdm_set.layer_df)
        _ffrac = params_i[1] / wdm_set.layer_df - _mfloor
        _m_lo, _m_hi = _mfloor, _mfloor + 1
        # WDMSettings snapping: ind_min_f = ceil(min_freq/df),
        # ind_max_f = int(max_freq/df). Use half-layer offsets on BOTH
        # edges so we land exactly on [_m_lo, _m_hi] regardless of fp
        # round-off (placing min_freq at m_lo*df exactly can ceil up to
        # m_lo+1 if the fp multiply overshoots by 1 ULP).
        new_wdm_set_2 = WDMSettings(
            wdm_set.Nf, wdm_set.Nt, wdm_set.data_dt,
            min_time=wdm_set.min_time, max_time=wdm_set.max_time,
            min_freq=(_m_lo - 0.5) * wdm_set.layer_df,
            max_freq=(_m_hi + 0.5) * wdm_set.layer_df,
        )
        assert new_wdm_set_2.ind_min_f == _m_lo and new_wdm_set_2.ind_max_f == _m_hi, (
            f"2-layer band indexing wrong: got [{new_wdm_set_2.ind_min_f}, "
            f"{new_wdm_set_2.ind_max_f}], wanted [{_m_lo}, {_m_hi}] for "
            f"f_frac={_ffrac:.4f}"
        )
        inj_2 = DataResidualArray(WDMSignal(
            injection[:,
                      new_wdm_set_2.ind_min_f - wdm_set.ind_min_f:
                      new_wdm_set_2.ind_max_f - wdm_set.ind_min_f + 1],
            new_wdm_set_2,
        ))
        tpl_2 = DataResidualArray(WDMSignal(
            tpl_wdm[:,
                    new_wdm_set_2.ind_min_f - wdm_set.ind_min_f:
                    new_wdm_set_2.ind_max_f - wdm_set.ind_min_f + 1],
            new_wdm_set_2,
        ))
        analysis_2 = AnalysisContainer(inj_2, XYZ2SensitivityMatrix(new_wdm_set_2, model="scirdv1"))
        log_like_2_layers = analysis_2.template_likelihood(tpl_2)
        mismatch_2_layers = 1.0 - analysis_2.template_inner_product(tpl_2, normalize=True)
        log_like_2_layers_list.append(log_like_2_layers)
        mismatch_2_layers_list.append(mismatch_2_layers)

        if (i + 1) % PROGRESS_EVERY == 0 or i == 0:
            elapsed = time.perf_counter() - t_loop_start
            rate = (i + 1) / max(elapsed, 1e-9)
            print(
                f"  [{i+1:4d}/{N_DRAWS}] snr={snr:7.2f}  logL={log_like:+.3e} params={params_i}, f_frac ={(params_i[1] - int(params_i[1] / wdm_set.layer_df) * wdm_set.layer_df) / wdm_set.layer_df} "
                f"1-O={mismatch:.3e}. main m_layer = ({int(params_i[1] / wdm_set.layer_df)}).  "
                f"\nJust over 5 layers: logL5={log_like_5_layers:+.3e}, mismatch5={mismatch_5_layers:+.3e}"
                f"\nJust over 2 layers: logL2={log_like_2_layers:+.3e}, mismatch2={mismatch_2_layers:+.3e}\n"
                f"{attempt_total} attempts, {rate:.2f} draw/s, {elapsed:.1f}s)",
                flush=True,
            )

        # plt.rcParams['text.usetex'] = False
        # fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=True)
        # injection.data_res_arr.heatmap(fig=fig, ax=ax1, index=0)
        # tpl_wdm.heatmap(fig=fig, ax=ax2, index=0, add_cax=True)
        # ax1.set_title("injection")
        # ax2.set_title("C lookup template")
        # plt.tight_layout()
        # plt.ylim(params_i[1] - 2* wdm_set.layer_df, params_i[1] + 2 * wdm_set.layer_df)
        # plt.savefig("gb_lookup_test_heatmap.png", dpi=120)
        # plt.show()
        # plt.close()
        # plt.xlim
        # breakpoint()
        

    snrs = np.asarray(snr_list)
    log_likes = np.asarray(log_like_list)
    mismatches = np.asarray(mismatch_list)
    mismatches_5 = np.asarray(mismatch_5_layers_list)
    log_likes_5 = np.asarray(log_like_5_layers_list)
    mismatches_2 = np.asarray(mismatch_2_layers_list)
    log_likes_2 = np.asarray(log_like_2_layers_list)
    params_arr = np.asarray(params_list)

    # ---- save raw data --------------------------------------------------
    npz_path = OUTPUT_PREFIX + "_results.npz"
    np.savez(
        npz_path,
        snr=snrs,
        log_like=log_likes,
        mismatch=mismatches,
        mismatch_5=mismatches_5,
        log_like_5=log_likes_5,
        mismatch_2=mismatches_2,
        log_like_2=log_likes_2,
        params=params_arr,
        snr_min=SNR_MIN, snr_max=SNR_MAX,
        f0_lo_hz=f0_lo_hz, f0_hi_hz=f0_hi_hz,
        fdot_max=fdot_max,
        A_lims=np.asarray(A_lims),
        n_draws=N_DRAWS,
        total_attempts=attempt_total,
    )
    print(f"[saved] {npz_path}", flush=True)

    # ---- best vs worst sources (focused on the sign-flip diagnostic) ----
    param_names = ("A", "f0", "fdot", "fddot", "phi0", "inc", "psi", "lam", "beta")
    m5 = np.real(mismatches_5).astype(float)
    fin = np.isfinite(m5)
    if fin.any():
        order = np.argsort(m5[fin])
        idx_sorted = np.where(fin)[0][order]
        n_show = min(8, idx_sorted.size)
        layer_df_local = wdm_set.layer_df
        f0s = params_arr[:, 1]
        m_continuous = f0s / layer_df_local
        f0_frac = m_continuous - np.floor(m_continuous)  # in [0,1)

        m2 = np.real(mismatches_2).astype(float)

        def _row(k):
            return (f"  {k:4d}  snr={snrs[k]:8.2f}  mm5={m5[k]:+11.3e}  "
                    f"mm2={m2[k]:+11.3e}  "
                    f"f0={f0s[k]*1e3:.6f}mHz  m_cont={m_continuous[k]:.4f}  "
                    f"f0_frac={f0_frac[k]:.4f}  "
                    f"phi0={params_arr[k,4]:+.4f}  inc={params_arr[k,5]:.4f}  "
                    f"psi={params_arr[k,6]:.4f}  lam={params_arr[k,7]:.4f}  "
                    f"beta={params_arr[k,8]:+.4f}")

        print("\n[best sources by 5-layer mismatch]")
        for k in idx_sorted[:n_show]:
            print(_row(k))
        print("\n[worst sources by 5-layer mismatch]")
        for k in idx_sorted[-n_show:][::-1]:
            print(_row(k))

        # Pattern hunting: bin by f0_frac and see whether bad-mismatch sources
        # cluster at half-layer boundaries (~0.5).
        flipped = m5 > 0.5  # mismatches near +1 (or above) signal a sign flip
        bad = (m5 > 1e-3) & np.isfinite(m5)
        good = m5 < 1e-3
        print(f"\n[sign-flip stats] {int(flipped.sum())}/{N_DRAWS} sources have mm_5 > 0.5; "
              f"{int(bad.sum())}/{N_DRAWS} have mm_5 > 1e-3; "
              f"{int(good.sum())}/{N_DRAWS} have mm_5 < 1e-3")
        if flipped.any():
            print(f"  flipped f0_frac: median={np.median(f0_frac[flipped]):.3f}, "
                  f"mean={np.mean(f0_frac[flipped]):.3f}, "
                  f"std={np.std(f0_frac[flipped]):.3f}")
        if bad.any():
            print(f"  bad     f0_frac: median={np.median(f0_frac[bad]):.3f}, "
                  f"mean={np.mean(f0_frac[bad]):.3f}, "
                  f"std={np.std(f0_frac[bad]):.3f}, "
                  f"min={np.min(f0_frac[bad]):.3f}, max={np.max(f0_frac[bad]):.3f}")
        if good.any():
            print(f"  good    f0_frac: median={np.median(f0_frac[good]):.3f}, "
                  f"mean={np.mean(f0_frac[good]):.3f}, "
                  f"std={np.std(f0_frac[good]):.3f}")

        # ---- f_frac binning: count bad fraction per bin ----
        n_bins = 10
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        idx_bin = np.clip(np.digitize(f0_frac, edges) - 1, 0, n_bins - 1)
        print("\n[bad-fraction vs f0_frac] (bin -> n_total, n_bad, n_flipped, frac_bad)")
        for b in range(n_bins):
            mask = (idx_bin == b)
            n_tot = int(mask.sum())
            n_bad = int((mask & bad).sum())
            n_flip = int((mask & flipped).sum())
            frac = (n_bad / n_tot) if n_tot else 0.0
            print(f"  [{edges[b]:.2f},{edges[b+1]:.2f})  n={n_tot:4d}  bad={n_bad:4d}  "
                  f"flip={n_flip:4d}  frac_bad={frac:.3f}")

        np.savez(
            OUTPUT_PREFIX + "_signdiag.npz",
            snr=snrs, mismatch=np.real(mismatches), mismatch_5=m5,
            f0_frac=f0_frac, params=params_arr,
        )

        # ---- scatter: mismatch_5 vs f0_frac ----
        figf, axf = plt.subplots(1, 2, figsize=(11, 4))
        axf[0].scatter(f0_frac, np.clip(m5, 1e-12, None), s=10, alpha=0.6, color="steelblue")
        axf[0].set_yscale("log")
        axf[0].set_xlabel(r"$f_0 / \Delta f_{\rm layer}$ mod 1")
        axf[0].set_ylabel(r"$1 - \mathcal{O}$  (5-layer window)")
        axf[0].set_title("mm_5 vs f0_frac")
        axf[0].axvline(0.5, color="k", ls=":", alpha=0.5)
        axf[0].grid(True, ls=":", alpha=0.4)
        # Bar chart of bad fraction per bin
        centers = 0.5 * (edges[:-1] + edges[1:])
        bad_per_bin = np.array(
            [(((idx_bin == b) & bad).sum()) / max((idx_bin == b).sum(), 1)
             for b in range(n_bins)]
        )
        axf[1].bar(centers, bad_per_bin, width=(edges[1] - edges[0]) * 0.9, color="indianred", alpha=0.8)
        axf[1].set_xlabel(r"$f_0 / \Delta f_{\rm layer}$ mod 1")
        axf[1].set_ylabel("fraction with mm_5 > 1e-3")
        axf[1].set_title("bad-fraction vs f0_frac")
        axf[1].axvline(0.5, color="k", ls=":", alpha=0.5)
        plt.tight_layout()
        ffrac_path = OUTPUT_PREFIX + "_ffrac.png"
        plt.savefig(ffrac_path, dpi=130)
        plt.close(figf)
        print(f"[saved] {ffrac_path}", flush=True)

    # ---- summary --------------------------------------------------------
    fin_m = np.isfinite(mismatches) & (mismatches > 0)
    fin_l = np.isfinite(log_likes)
    if fin_m.any():
        print(f"[stats] mismatch  median={np.median(mismatches[fin_m]):.3e}  "
              f"min={np.min(mismatches[fin_m]):.3e}  max={np.max(mismatches[fin_m]):.3e}")
    if fin_l.any():
        print(f"[stats] log-like  median={np.median(log_likes[fin_l]):+.3e}  "
              f"min={np.min(log_likes[fin_l]):+.3e}  max={np.max(log_likes[fin_l]):+.3e}")

    # ---- plots ----------------------------------------------------------
    plt.rcParams["text.usetex"] = False

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    if fin_m.any():
        ax[0].hist(np.log10(mismatches[fin_m]), bins=40, color="steelblue", alpha=0.8)
        ax[0].set_xlabel(r"$\log_{10}(1 - \mathcal{O})$  (GB layers only)")
        ax[0].set_ylabel("count")
        ax[0].set_title(f"mismatch  (N={int(fin_m.sum())})")
    if fin_l.any():
        ax[1].hist(log_likes[fin_l], bins=40, color="darkorange", alpha=0.8)
        ax[1].set_xlabel(r"$\log L = -\frac{1}{2}\langle d-h | d-h\rangle$  (full WDM)")
        ax[1].set_ylabel("count")
        ax[1].set_title(f"log-likelihood  (N={int(fin_l.sum())})")
    plt.tight_layout()
    hist_path = OUTPUT_PREFIX + "_hist.png"
    plt.savefig(hist_path, dpi=130)
    plt.close(fig)
    print(f"[saved] {hist_path}", flush=True)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    if fin_m.any():
        ax[0].scatter(snrs[fin_m], mismatches[fin_m], s=10, alpha=0.6,
                      color="steelblue")
        ax[0].set_xscale("log")
        ax[0].set_yscale("log")
        ax[0].set_xlabel("SNR")
        ax[0].set_ylabel(r"$1 - \mathcal{O}$  (GB layers only)")
        ax[0].set_title("mismatch vs SNR")
        ax[0].grid(True, which="both", ls=":", alpha=0.4)
    if fin_l.any():
        ax[1].scatter(snrs[fin_l], log_likes[fin_l], s=10, alpha=0.6,
                      color="darkorange")
        ax[1].set_xscale("log")
        ax[1].set_xlabel("SNR")
        ax[1].set_ylabel(r"$\log L$")
        ax[1].set_title("log-likelihood vs SNR")
        ax[1].grid(True, which="both", ls=":", alpha=0.4)
    plt.tight_layout()
    scat_path = OUTPUT_PREFIX + "_scatter.png"
    plt.savefig(scat_path, dpi=130)
    plt.close(fig)
    print(f"[saved] {scat_path}", flush=True)

    print(
        f"[done] {N_DRAWS} draws, {attempt_total} total attempts, "
        f"acceptance ~ {N_DRAWS / max(attempt_total, 1):.2%}",
        flush=True,
    )


if __name__ == "__main__":
    main()
