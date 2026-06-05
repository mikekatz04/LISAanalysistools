#!/usr/bin/env python
"""Per-source diagnostic: rebuild the WDM injection and the C lookup template
for the K best and K worst sources from a pattern-run ``_results.npz``, and
dump side-by-side heatmaps + per-layer / per-time error breakdowns so the
spatial pattern of disagreement is visible.

Usage:  python gb_lookup_source_diag.py [npz_prefix] [K]
        (defaults: 'gb_prior_pattern', 4)

Outputs (under ``<prefix>_srcdiag/``):
  src_<rank>_<role>.png   per source: 4-panel chan-0 (inj / tpl / diff / sum)
                          + per-layer L1 residual + per-time L1 residual
  src_<rank>_<role>.txt   per-source numeric summary
"""

import os
import sys
import numpy as np
import matplotlib
if not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal as sps

from lisatools.detector import ESAOrbits
from lisatools.utils.constants import YRSID_SI
from lisatools.response.tdiconfig import TDIConfig
from lisatools.response.tdionfly import GBTDIonTheFly
from gbgpu.gbcomps import GBWDMComputations
from lisatools.datacontainer import DataResidualArray
from lisatools.analysiscontainer import AnalysisContainer, AnalysisContainerArray
from lisatools.sensitivity import XYZ2SensitivityMatrix
from lisatools.domains import (
    TDSettings, TDSignal, WDMSettings, WDMSignal, WDMLookupTable,
)


def _heatmap(ax, arr, vlim, title):
    im = ax.imshow(arr, aspect="auto", origin="lower", vmin=-vlim, vmax=vlim,
                   cmap="RdBu_r")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.045)


def diag_one(out_path_png, out_path_txt, params_i, *,
             gb_gen_inj, t_arr, td_set, wdm_set, window, gb_comps,
             sens_mat, label):
    inj_spline = gb_gen_inj(
        *params_i.reshape(9, 1),
        convert_to_ra_dec=False, return_spline=True,
    )
    td_inj = np.asarray(inj_spline.eval_tdi(t_arr))[0]
    wdm_inj_sig = TDSignal(td_inj, settings=td_set).transform(wdm_set, window=window)
    injection = DataResidualArray(wdm_inj_sig)
    analysis = AnalysisContainer(injection, sens_mat)

    n_pix = int(np.prod(wdm_set.basis_shape_active))
    tpl_buf = np.zeros(3 * n_pix, dtype=float)
    gb_comps.fill_global_wdm(
        tpl_buf, params_i.reshape(1, 9),
        AnalysisContainerArray([analysis]),
        data_index=None, convert_to_ra_dec=False,
    )
    tpl_arr = np.asarray(tpl_buf.reshape((3,) + wdm_set.basis_shape_active))

    inj_arr = np.asarray(injection[:])  # (3, Nf_active, Nt_active)

    # Crop to the GB-relevant layers (template's footprint widened by 2)
    layer_power = (tpl_arr ** 2).sum(axis=(0, 2))
    nz = np.where(layer_power > 0)[0]
    if nz.size:
        m_lo = max(0, int(nz.min()) - 2)
        m_hi = min(layer_power.size - 1, int(nz.max()) + 2)
    else:
        m_lo, m_hi = 0, layer_power.size - 1
    m_slice = slice(m_lo, m_hi + 1)
    inj_crop = inj_arr[:, m_slice, :]
    tpl_crop = tpl_arr[:, m_slice, :]
    diff_crop = inj_crop - tpl_crop

    # Layer / time residual L2 envelopes (channel-0)
    per_layer = np.sqrt((diff_crop[0] ** 2).sum(axis=-1))
    per_time = np.sqrt((diff_crop[0] ** 2).sum(axis=0))
    layer_inds = np.arange(m_lo, m_hi + 1) + wdm_set.ind_min_f
    t_inds = np.arange(diff_crop.shape[-1]) + wdm_set.ind_min_t

    # Plot
    vmax = max(np.abs(inj_crop[0]).max(), np.abs(tpl_crop[0]).max(), 1e-30)
    fig = plt.figure(figsize=(13, 7))
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 0.9])
    ax_inj = fig.add_subplot(gs[0, 0])
    ax_tpl = fig.add_subplot(gs[0, 1])
    ax_dif = fig.add_subplot(gs[0, 2])
    ax_sum = fig.add_subplot(gs[0, 3])
    _heatmap(ax_inj, inj_crop[0], vmax, "injection (chan 0)")
    _heatmap(ax_tpl, tpl_crop[0], vmax, "C lookup template")
    _heatmap(ax_dif, diff_crop[0], vmax, "inj - tpl")
    _heatmap(ax_sum, inj_crop[0] + tpl_crop[0], vmax, "inj + tpl")

    ax_layer = fig.add_subplot(gs[1, :2])
    ax_layer.plot(layer_inds, per_layer, "-o", color="steelblue", markersize=3)
    ax_layer.set_xlabel("absolute layer m")
    ax_layer.set_ylabel("L2 residual (chan 0) over time")
    ax_layer.grid(True, ls=":", alpha=0.4)
    ax_layer.set_title("per-m residual envelope")

    ax_time = fig.add_subplot(gs[1, 2:])
    ax_time.plot(t_inds, per_time, "-", color="darkorange")
    ax_time.set_xlabel("absolute time bin n")
    ax_time.set_ylabel("L2 residual (chan 0) over m")
    ax_time.grid(True, ls=":", alpha=0.4)
    ax_time.set_title("per-n residual envelope")

    ax_txt = fig.add_subplot(gs[2, :])
    ax_txt.axis("off")
    pn = ("A", "f0[Hz]", "fdot", "fddot", "phi0", "inc", "psi", "lam", "beta")
    layer_df = wdm_set.layer_df
    m_central = params_i[1] / layer_df
    txt = (
        f"{label}  ||  A={params_i[0]:.3e}  f0={params_i[1]*1e3:.6f} mHz  "
        f"m_central={m_central:.4f}  f0_frac={m_central - np.floor(m_central):.4f}\n"
        f"fdot={params_i[2]:+.3e}  phi0={params_i[4]:+.4f}  inc={params_i[5]:.4f}  "
        f"cos(inc)={np.cos(params_i[5]):+.4f}  psi={params_i[6]:.4f}  "
        f"lam={params_i[7]:.4f}  beta={params_i[8]:+.4f}\n"
        f"SNR(d)={float(analysis.snr()):.2f}   "
        f"|d|={np.sqrt((inj_arr**2).sum()):.3e}   "
        f"|h|={np.sqrt((tpl_arr**2).sum()):.3e}   "
        f"|d-h|/|d|={np.sqrt((diff_crop**2).sum())/np.sqrt((inj_crop**2).sum()):.3e}"
    )
    ax_txt.text(0.0, 1.0, txt, fontsize=10, family="monospace", va="top")
    plt.tight_layout()
    plt.savefig(out_path_png, dpi=120)
    plt.close(fig)

    with open(out_path_txt, "w") as f:
        f.write(txt + "\n")
        f.write(f"per-layer residual L2 (layer index, value):\n")
        for li, v in zip(layer_inds.tolist(), per_layer.tolist()):
            f.write(f"  m={li}  L2={v:.6e}\n")
        f.write(f"per-time residual L2 max over n_inds: "
                f"argmax n={int(t_inds[np.argmax(per_time)])}, "
                f"max L2={float(per_time.max()):.3e}\n")


def main():
    prefix = sys.argv[1] if len(sys.argv) > 1 else "gb_prior_pattern"
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 4

    npz_path = prefix + "_results.npz"
    if not os.path.exists(npz_path):
        sys.exit(f"missing {npz_path}")
    d = np.load(npz_path, allow_pickle=True)
    mm5 = np.real(np.asarray(d["mismatch_5"], dtype=float))
    params_all = np.asarray(d["params"], dtype=float)
    fin = np.isfinite(mm5)
    order = np.argsort(mm5[fin])
    idx_all = np.where(fin)[0]
    best_idx = idx_all[order[:K]].tolist()
    worst_idx = idx_all[order[-K:][::-1]].tolist()

    out_dir = prefix + "_srcdiag"
    os.makedirs(out_dir, exist_ok=True)

    # ---- replicate the runtime setup --------------------------------
    backend = os.environ.get("LOOKUP_BACKEND", "cpu")
    xp = np

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
        tdi_config=tdi_config, orbits=orbits,
        tdi_chan="XYZ", force_backend=backend,
    )
    N_INJ = int(os.environ.get("N_INJ", 16384))
    t_tdi_inj = xp.linspace(t_arr[0], t_arr[-1], N_INJ)
    gb_gen_inj = GBTDIonTheFly(t_tdi_inj, Tobs, t_ref, 1.0 / dt, 1, **gb_tdi_kwargs)

    td_set = TDSettings(t_arr.shape[-1], dt, force_backend=backend)
    window = xp.asarray(sps.windows.tukey(t_arr.shape[-1], alpha=0.05))

    min_freq, max_freq = 0.0001, 35.0e-3
    min_time = 20 * wavelet_duration
    max_time = (Nt - 20) * wavelet_duration
    wdm_set = WDMSettings(Nf, Nt, dt,
                          min_freq=min_freq, max_freq=max_freq,
                          min_time=min_time, max_time=max_time)

    store_path = os.environ.get("LOOKUP_PATH", "wdm_lookup_new_all_time_layers_1.h5")
    if not os.path.exists(store_path):
        sys.exit(f"missing lookup table {store_path}")
    wdm_lookup_table = WDMLookupTable.from_file(store_path, force_backend=backend)

    gb_comps = GBWDMComputations(
        wdm_lookup_table, Tobs, t_ref,
        orbits=orbits, tdi_config=tdi_config, force_backend=backend,
    )

    # build sens_mat once
    dummy_params = params_all[best_idx[0]]
    inj_spline = gb_gen_inj(*dummy_params.reshape(9, 1),
                            convert_to_ra_dec=False, return_spline=True)
    td_dummy = np.asarray(inj_spline.eval_tdi(t_arr))[0]
    wdm_dummy = TDSignal(td_dummy, settings=td_set).transform(wdm_set, window=window)
    sens_mat = XYZ2SensitivityMatrix(
        DataResidualArray(wdm_dummy).data_res_arr.settings, model="scirdv1"
    )

    print(f"[diag] best={best_idx}  worst={worst_idx}")
    for rank, (idx, role) in enumerate(
        [(i, "best") for i in best_idx] + [(i, "worst") for i in worst_idx]
    ):
        label = f"src #{idx}  ({role}, mm_5={mm5[idx]:.3e})"
        out_png = os.path.join(out_dir, f"src_{role}_{rank:02d}_idx{idx:04d}.png")
        out_txt = os.path.join(out_dir, f"src_{role}_{rank:02d}_idx{idx:04d}.txt")
        print(f"  → {label}")
        diag_one(out_png, out_txt, params_all[idx],
                 gb_gen_inj=gb_gen_inj, t_arr=t_arr, td_set=td_set,
                 wdm_set=wdm_set, window=window, gb_comps=gb_comps,
                 sens_mat=sens_mat, label=label)
    print(f"\n[done] wrote {2 * K} diagnostic figures to {out_dir}/")


if __name__ == "__main__":
    main()
