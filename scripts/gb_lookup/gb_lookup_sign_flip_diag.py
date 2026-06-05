#!/usr/bin/env python
"""Diagnostic: find GB draws where the C WDM template differs in sign from
the Python lookup-table template.

Setup is reused from ``gb_lookup_prior_draws.py`` (ProbDistContainer +
TransformContainer prior, WDM lookup table, GBTDIonTheFly response). For
each draw we generate the WDM template through two paths:

  * Python  : ``GBLookupWaveWrap`` from ``gb_lookup_table_test_script.py``
              (analytic spline derivative for f, ``WDMLookupTable.get_wdm_coeffs``).
  * C/CUDA  : ``GBWDMComputations.fill_global_wdm`` (central-difference f,
              ``WaveletLookupTable::get_w_mn_lookup``).

We then compute the normalized cross-overlap

    rho = <py | c> / sqrt(<py|py> <c|c>)         (Frobenius, unweighted)

over the GB layers (the union of layers where either template is non-zero):

  * rho ≈ +1     → Python and C agree (good)
  * rho ≈ -1     → entire template flipped sign     **bug**
  * |rho| ≈ 0    → templates uncorrelated           (different bug)
  * something else → partial cancellation           (mixed bug)

Sign-flipped draws are dumped (params + a side-by-side ``.png`` of the two
templates) so we can look for what triggers the flip in the C kernel.

Knobs (env vars):
  N_DRAWS, SNR_MIN, SNR_MAX, FLIP_TOL (default 0.5; sources with rho < -FLIP_TOL
  are classified as sign-flipped), MAX_DUMPS (default 8, cap on per-source
  diagnostic plots), LOOKUP_PATH.
"""
import os
import sys
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

# Reuse the Python lookup-table wrap straight from the existing test script.
# Importing it as a module means we get the same convention (pi/2 shift,
# spline-derivative frequencies, fdot=0 override) that the prior comparison
# code uses.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gb_lookup_table_test_script as gltt   # noqa: E402  (module exposes GBLookupWaveWrap)


def build_gb_prior(*, A_lims, f0_lims_hz, fdot_lims):
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
    delta = 1e-5
    iota_lims = (delta, np.pi - delta)
    beta_lims = (-np.pi / 2.0 + delta, np.pi / 2.0 - delta)
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
    return ProbDistContainer(priors_gb), tc


def _frobenius_overlap(a, b):
    """Unweighted Frobenius cosine between two real arrays (any shape)."""
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    na = float(np.sqrt((a * a).sum()))
    nb = float(np.sqrt((b * b).sum()))
    if na == 0.0 or nb == 0.0:
        return float("nan"), na, nb, float((a * b).sum())
    return float((a * b).sum() / (na * nb)), na, nb, float((a * b).sum())


def _dump_template_png(path, py_arr, c_arr, params, rho):
    """Side-by-side heatmaps of channel-0 Python vs C templates."""
    py0 = py_arr[0]
    c0 = c_arr[0]
    diff = py0 - c0
    sum_ = py0 + c0
    vmax = max(np.abs(py0).max(), np.abs(c0).max(), 1e-30)
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    im0 = ax[0, 0].imshow(py0, aspect="auto", origin="lower",
                          vmin=-vmax, vmax=vmax, cmap="RdBu_r")
    ax[0, 0].set_title("Python lookup (chan 0)")
    plt.colorbar(im0, ax=ax[0, 0])
    im1 = ax[0, 1].imshow(c0, aspect="auto", origin="lower",
                          vmin=-vmax, vmax=vmax, cmap="RdBu_r")
    ax[0, 1].set_title("C lookup (chan 0)")
    plt.colorbar(im1, ax=ax[0, 1])
    im2 = ax[1, 0].imshow(diff, aspect="auto", origin="lower",
                          vmin=-vmax, vmax=vmax, cmap="RdBu_r")
    ax[1, 0].set_title("py - C")
    plt.colorbar(im2, ax=ax[1, 0])
    im3 = ax[1, 1].imshow(sum_, aspect="auto", origin="lower",
                          vmin=-vmax, vmax=vmax, cmap="RdBu_r")
    ax[1, 1].set_title("py + C")
    plt.colorbar(im3, ax=ax[1, 1])
    fig.suptitle(
        f"rho_py_c = {rho:+.4f}   |   A={params[0]:.3e}  f0={params[1]*1e3:.4f} mHz  "
        f"fdot={params[2]:.2e}\nphi0={params[4]:.3f}  inc={params[5]:.3f}  "
        f"psi={params[6]:.3f}  lam={params[7]:.3f}  beta={params[8]:.3f}",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close(fig)


def main():
    backend = os.environ.get("LOOKUP_BACKEND", "cpu")
    xp = np if backend == "cpu" else cp

    N_DRAWS = int(os.environ.get("N_DRAWS", 200))
    SNR_MIN = float(os.environ.get("SNR_MIN", 5.0))
    SNR_MAX = float(os.environ.get("SNR_MAX", 1100.0))
    N_INJ = int(os.environ.get("N_INJ", 16384))
    N_SPARSE = int(os.environ.get("N_SPARSE", 4096))
    NUM_LAYERS_DIFF = int(os.environ.get("NUM_LAYERS_DIFF", 5))
    EPS_FREQ = float(os.environ.get("EPS_FREQ", 0.001))
    MAX_REJECT = int(os.environ.get("MAX_REJECT", 500))
    FLIP_TOL = float(os.environ.get("FLIP_TOL", 0.5))
    DISAGREE_TOL = float(os.environ.get("DISAGREE_TOL", 0.95))
    MAX_DUMPS = int(os.environ.get("MAX_DUMPS", 8))
    SEED = int(os.environ.get("SEED", 12345))
    OUTPUT_PREFIX = os.environ.get("OUTPUT_PREFIX", "gb_sign_flip_diag")

    np.random.seed(SEED)

    # ---- detector / domain setup (same as gb_lookup_prior_draws.py) -----
    orbits = ESAOrbits(force_backend=backend)
    dt = 10.0
    Nt = 256 * 10
    Nf = 1460
    wavelet_duration = Nf * dt
    Tobs = Nt * wavelet_duration
    Nobs = Nf * Nt

    tdi_config = TDIConfig("2nd generation")
    t_start = int(0.5 * YRSID_SI / dt) * dt
    t_arr = np.arange(Nobs) * dt + t_start
    t_ref = t_start

    gb_tdi_kwargs = dict(
        tdi_config=tdi_config,
        orbits=orbits,
        tdi_chan="XYZ",
        force_backend=backend,
    )
    t_tdi_inj = xp.linspace(t_arr[0], t_arr[-1], N_INJ)
    gb_gen_inj = GBTDIonTheFly(t_tdi_inj, Tobs, t_ref, 1.0 / dt, 1, **gb_tdi_kwargs)

    N = t_arr.shape[-1]
    td_set = TDSettings(N, dt, force_backend=backend)
    freqs = np.fft.rfftfreq(N, dt)
    df = freqs[1] - freqs[0]
    N_fd = len(freqs)
    window = xp.asarray(signal.windows.tukey(N, alpha=0.05))

    min_freq = 0.0001
    max_freq = 35.0e-3
    _ = FDSettings(N_fd, df, min_freq=min_freq, max_freq=max_freq, force_backend=backend)
    min_time = 20 * wavelet_duration
    max_time = (Nt - 20) * wavelet_duration

    wdm_set = WDMSettings(
        Nf, Nt, dt,
        min_freq=min_freq, max_freq=max_freq,
        min_time=min_time, max_time=max_time,
    )

    # ---- lookup table --------------------------------------------------
    store_path = os.environ.get("LOOKUP_PATH", "wdm_lookup_new_all_time_layers_1.h5")
    if os.path.exists(store_path):
        print(f"[lookup] loading from {store_path}", flush=True)
        wdm_lookup_table = WDMLookupTable.from_file(store_path, force_backend=backend)
        _wdm_settings = WDMSettings(*wdm_lookup_table.args, **wdm_lookup_table.kwargs)
        if not _wdm_settings.eq_without_inds(wdm_set):
            raise ValueError("lookup-table WDM settings != current wdm_set")
    else:
        print(f"[lookup] building new table at {store_path}", flush=True)
        td_window = xp.asarray(signal.windows.tukey(wdm_set.Nf * wdm_set.Nt, alpha=0.05))
        m_ref = int(3e-3 / wdm_set.layer_df)
        norm_freq_single_layer, m_diffs, _ = WDMLookupTable.apply_eps_frequency(
            EPS_FREQ, wdm_set, m_ref=m_ref, num_layers_diff=NUM_LAYERS_DIFF
        )
        wdm_lookup_table = WDMLookupTable(
            wdm_set, 3,
            norm_freq_single_layer=norm_freq_single_layer,
            m_diffs=m_diffs,
            fdot_vals=np.array([0.0]),
            m_ref=m_ref,
            batch_size_gen=5,
            td_window=td_window,
            store_path=store_path,
        )

    # Expose to the existing GBLookupWaveWrap (which reads `wdm_lookup_table`
    # and `xp` and `wdm_set` from its module globals — see gb_lookup_table_test_script.py).
    gltt.wdm_lookup_table = wdm_lookup_table
    gltt.xp = xp
    gltt.wdm_set = wdm_set

    t_tdi_sparse = xp.linspace(t_arr[0], t_arr[-1], N_SPARSE)
    gb_comps = GBWDMComputations(
        wdm_lookup_table, Tobs, t_ref, orbits=orbits,
        tdi_config=tdi_config, force_backend=backend,
    )
    gb_gen_wrap = gltt.GBLookupWaveWrap(
        t_arr, t_tdi_sparse, Tobs, t_ref, dt, 1, gb_tdi_kwargs,
        td_set, wdm_set, window,
    )

    # ---- prior ---------------------------------------------------------
    layer_df = wdm_set.layer_df
    buf = NUM_LAYERS_DIFF + 2
    f0_lo_hz = float(os.environ.get("F0_LO_HZ", (wdm_set.ind_min_f + buf) * layer_df))
    f0_hi_hz = float(os.environ.get("F0_HI_HZ", (wdm_set.ind_max_f - buf) * layer_df))
    fdot_max = float(os.environ.get("FDOT_MAX", 1e-15))
    A_lims = (float(os.environ.get("A_LO", 1e-23)),
              float(os.environ.get("A_HI", 1e-20)))
    prior, tc = build_gb_prior(
        A_lims=A_lims,
        f0_lims_hz=(f0_lo_hz, f0_hi_hz),
        fdot_lims=(-fdot_max, fdot_max),
    )

    print(f"[run] N_DRAWS={N_DRAWS}  SNR window=[{SNR_MIN}, {SNR_MAX}]"
          f"  FLIP_TOL={FLIP_TOL}", flush=True)

    n_pix_active = int(np.prod(wdm_set.basis_shape_active))
    sens_mat = None

    rho_list, na_list, nc_list, snr_list, dh_list = [], [], [], [], []
    n_signflip = 0
    n_disagree = 0
    n_dumped = 0
    flipped_params = []
    dumps_dir = OUTPUT_PREFIX + "_dumps"
    os.makedirs(dumps_dir, exist_ok=True)

    t0 = time.perf_counter()
    attempts = 0
    for i in range(N_DRAWS):
        # SNR rejection loop (lisatools snr())
        chosen = None
        for _ in range(MAX_REJECT):
            attempts += 1
            x_sampled = prior.rvs(size=1)
            params_i = tc.both_transforms(x_sampled.copy())[0]

            inj_spline = gb_gen_inj(
                *params_i.reshape(9, 1),
                convert_to_ra_dec=False, return_spline=True,
            )
            td_inj = np.asarray(inj_spline.eval_tdi(t_arr))
            wdm_inj_sig = TDSignal(td_inj, settings=td_set).transform(
                wdm_set, window=window
            )
            injection = DataResidualArray(wdm_inj_sig)
            if sens_mat is None:
                sens_mat = XYZ2SensitivityMatrix(
                    injection.data_res_arr.settings, model="scirdv1"
                )
            analysis = AnalysisContainer(injection, sens_mat)
            snr = float(analysis.snr())
            if SNR_MIN <= snr <= SNR_MAX:
                chosen = (params_i, analysis, snr)
                break

        if chosen is None:
            print(f"[warn] draw {i}: exhausted rejection (last snr={snr:.2f})", flush=True)
            chosen = (params_i, analysis, snr)
        params_i, analysis, snr = chosen

        # ---- C/CUDA template ---------------------------------------------
        template_fill = xp.zeros(3 * n_pix_active, dtype=float)
        wdm_holder = AnalysisContainerArray([analysis])
        gb_comps.fill_global_wdm(
            template_fill,
            params_i.reshape(1, 9),
            wdm_holder,
            data_index=None,
            convert_to_ra_dec=False,
        )
        tpl_c = np.asarray(
            template_fill.reshape((3,) + wdm_set.basis_shape_active)
        )
        c_wdm = WDMSignal(tpl_c, wdm_set)

        # ---- Python lookup template --------------------------------------
        py_wdm = gb_gen_wrap(*params_i)
        tpl_py = np.asarray(py_wdm.arr)

        # ---- shape check ------------------------------------------------
        assert tpl_py.shape == tpl_c.shape, (tpl_py.shape, tpl_c.shape)

        # ---- unweighted Frobenius cosine (sign-flip detector) -----------
        rho, na, nc, dot = _frobenius_overlap(tpl_py, tpl_c)

        # ---- lisatools noise-weighted overlap <py|C>/sqrt(<py|py><C|C>) ---
        # Built from three template_inner_product calls so that any
        # disagreement that survives the noise weighting (and not just the
        # Frobenius cosine) shows up too.
        ana_py = AnalysisContainer(DataResidualArray(py_wdm), sens_mat)
        ana_c = AnalysisContainer(DataResidualArray(c_wdm), sens_mat)
        ip_pyc = float(np.real(ana_py.template_inner_product(c_wdm)))
        ip_pypy = float(np.real(ana_py.inner_product()))
        ip_cc = float(np.real(ana_c.inner_product()))
        if ip_pypy > 0 and ip_cc > 0:
            wt_overlap = ip_pyc / np.sqrt(ip_pypy * ip_cc)
        else:
            wt_overlap = float("nan")

        rho_list.append(rho)
        na_list.append(na)
        nc_list.append(nc)
        snr_list.append(snr)
        dh_list.append(wt_overlap)

        flipped = (not np.isnan(rho)) and (rho < -FLIP_TOL)
        disagree = (not np.isnan(rho)) and (abs(rho) < DISAGREE_TOL) and not flipped

        if flipped:
            n_signflip += 1
            flipped_params.append((i, params_i.copy(), rho, wt_overlap))
            if n_dumped < MAX_DUMPS:
                _dump_template_png(
                    os.path.join(dumps_dir, f"flip_{i:04d}.png"),
                    tpl_py, tpl_c, params_i, rho,
                )
                n_dumped += 1
        elif disagree:
            n_disagree += 1

        elapsed = time.perf_counter() - t0
        marker = "FLIP" if flipped else ("xxx" if disagree else "ok ")
        print(
            f"  [{i+1:4d}/{N_DRAWS}] {marker}  rho_unwt={rho:+.4f}  "
            f"rho_wt={wt_overlap:+.4f}  snr={snr:7.2f}  "
            f"|py|={na:.3e}  |C|={nc:.3e}  ({elapsed:.1f}s)",
            flush=True,
        )

    snrs = np.asarray(snr_list)
    rhos = np.asarray(rho_list)
    wovs = np.asarray(dh_list)

    print(f"\n[summary] {n_signflip}/{N_DRAWS} sign-flipped (rho < -{FLIP_TOL})", flush=True)
    print(f"[summary] {n_disagree}/{N_DRAWS} partial mismatches (|rho| < {DISAGREE_TOL})", flush=True)
    print(f"[summary] {N_DRAWS - n_signflip - n_disagree}/{N_DRAWS} ok\n", flush=True)

    npz_path = OUTPUT_PREFIX + ".npz"
    np.savez(
        npz_path,
        rho_unweighted=rhos,
        rho_weighted=wovs,
        snr=snrs,
        norms_py=np.asarray(na_list),
        norms_c=np.asarray(nc_list),
        flipped_indices=np.asarray([f[0] for f in flipped_params], dtype=int),
        flipped_params=np.asarray(
            [f[1] for f in flipped_params] if flipped_params
            else np.empty((0, 9))
        ),
        n_draws=N_DRAWS,
        n_signflip=n_signflip,
        n_disagree=n_disagree,
        flip_tol=FLIP_TOL,
    )
    print(f"[saved] {npz_path}", flush=True)

    # ---- summary plots ------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(rhos[np.isfinite(rhos)], bins=60, color="steelblue")
    ax[0].axvline(+1.0, color="green", ls=":")
    ax[0].axvline(-1.0, color="red", ls=":")
    ax[0].axvline(-FLIP_TOL, color="orange", ls="--", label=f"flip thr -{FLIP_TOL}")
    ax[0].set_xlabel("Frobenius cosine  rho(py, C)")
    ax[0].set_ylabel("count")
    ax[0].set_title(f"{n_signflip}/{N_DRAWS} flipped")
    ax[0].legend()
    fin = np.isfinite(rhos) & np.isfinite(snrs)
    ax[1].scatter(snrs[fin], rhos[fin], s=10, alpha=0.6, color="steelblue")
    ax[1].axhline(+1.0, color="green", ls=":")
    ax[1].axhline(-1.0, color="red", ls=":")
    ax[1].axhline(-FLIP_TOL, color="orange", ls="--")
    ax[1].set_xscale("log")
    ax[1].set_xlabel("SNR")
    ax[1].set_ylabel("rho(py, C)")
    ax[1].grid(True, which="both", ls=":", alpha=0.4)
    ax[1].set_title("agreement vs SNR")
    plt.tight_layout()
    plt.savefig(OUTPUT_PREFIX + "_overview.png", dpi=130)
    plt.close(fig)
    print(f"[saved] {OUTPUT_PREFIX}_overview.png", flush=True)

    if flipped_params:
        print("\n[flipped params (index, A, f0_mHz, fdot, phi0, inc, psi, lam, beta, rho)]:")
        for idx, p, rho_v, wt in flipped_params:
            print(
                f"  i={idx:4d}  A={p[0]:.3e}  f0={p[1]*1e3:.6f} mHz  fdot={p[2]:+.2e}"
                f"  phi0={p[4]:+.4f}  inc={p[5]:.4f}  psi={p[6]:.4f}"
                f"  lam={p[7]:.4f}  beta={p[8]:+.4f}  rho={rho_v:+.4f}  wt_rho={wt:+.4f}"
            )

    print(f"\n[done] {time.perf_counter() - t0:.1f}s, "
          f"{N_DRAWS} kept / {attempts} attempts", flush=True)


if __name__ == "__main__":
    main()
